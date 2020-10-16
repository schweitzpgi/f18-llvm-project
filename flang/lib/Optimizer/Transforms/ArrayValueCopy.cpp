//===-- ArrayValueCopy.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "flang-array-value-copy"

using namespace fir;

namespace {

/// Array copy analysis.
/// Perform an interference analysis between array values.
///
/// Lowering will generate a sequence of the following form.
/// ```mlir
///   %a_1 = fir.array_load %array_1(%shape) : ...
///   ...
///   %a_n = fir.array_load %array_n(%shape) : ...
///     ...
///     %v_i = fir.array_fetch %a_i, ...
///     %a_j1 = fir.array_update %a_j, ...
///     ...
///   fir.array_store %a_jn to %array_j : ...
/// ```
///
/// The analysis is to determine if there are any conflicts. A conflict is when
/// one the following cases occurs.
/// 1. There is an `array_update` to an array value, a_j, such that a_j was
/// loaded from the same array memory reference (array_j) but with a different
/// shape as the other array values a_i, where i != j. [Possible overlapping
/// arrays.]
/// 2. There is either an array_fetch or array_update of a_j with a different
/// set of index values. [Possible loop-carried dependence.]
/// If none of the array values overlap in storage and the accesses are not
/// loop-carried, then the arrays are conflict-free and no copies are required.
class ArrayCopyAnalysis {
public:
  using ConflictSetT = llvm::SmallPtrSet<mlir::Operation *, 16>;

  ArrayCopyAnalysis(mlir::Operation *op) : operation{op}, domInfo{op} {
    construct(op->getRegions());
  }

  mlir::Operation *getOperation() const { return operation; }

  /// Return true iff the `array_store` has potential conflicts.
  bool hasPotentialConflict(ArrayStoreOp st) {
    return conflicts.contains(st.getOperation());
  }

private:
  void arrayAccesses(llvm::SmallVectorImpl<mlir::Operation *> &accesses,
                     llvm::ArrayRef<mlir::Operation *> reach, mlir::Value addr,
                     mlir::Operation *store);
  void construct(mlir::MutableArrayRef<mlir::Region> regions);

private:
  mlir::Operation *operation;
  mlir::DominanceInfo domInfo;
  ConflictSetT conflicts;
};
} // namespace

// Recursively trace operands to find all array operations relating to the
// values merged.
static void populateSets(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                         llvm::SmallPtrSetImpl<mlir::Value> &visited,
                         mlir::Value val) {
  if (!val || visited.contains(val))
    return;
  visited.insert(val);

  if (auto *op = val.getDefiningOp()) {
    // `val` is defined by an Op, process the defining Op.
    // If `val` is defined by a region containing Op, we want to drill down and
    // through that Op's region(s).
    auto popFn = [&](auto rop) {
      auto resNum = val.cast<mlir::OpResult>().getResultNumber();
      llvm::SmallVector<mlir::Value, 2> results;
      rop.resultToSourceOps(results, resNum);
      for (auto u : results)
        populateSets(reach, visited, u);
    };
    if (auto rop = mlir::dyn_cast<DoLoopOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<IterWhileOp>(op)) {
      popFn(rop);
      return;
    }
    if (auto rop = mlir::dyn_cast<fir::IfOp>(op)) {
      popFn(rop);
      return;
    }

    // Otherwise, Op does not contain a region so just chase its operands.
    if (mlir::isa<ArrayLoadOp, ArrayUpdateOp, ArrayFetchOp>(op))
      reach.emplace_back(op);
    for (auto u : op->getOperands())
      populateSets(reach, visited, u);
    return;
  }

  // Process a block argument.
  auto ba = val.cast<mlir::BlockArgument>();
  auto *parent = ba.getOwner()->getParentOp();
  // If inside an Op holding a region, the block argument corresponds to an
  // argument passed to the containing Op.
  auto popFn = [&](auto rop) {
    populateSets(reach, visited, rop.blockArgToSourceOp(ba.getArgNumber()));
  };
  if (auto rop = mlir::dyn_cast<DoLoopOp>(parent)) {
    popFn(rop);
    return;
  }
  if (auto rop = mlir::dyn_cast<IterWhileOp>(parent)) {
    popFn(rop);
    return;
  }
  // Otherwise, a block argument is provided via the pred blocks.
  for (auto pred : ba.getOwner()->getPredecessors()) {
    auto u = pred->getTerminator()->getOperand(ba.getArgNumber());
    populateSets(reach, visited, u);
  }
}

/// Return all ops that produce the array value that is stored into the
/// `array_store`, st.
static void reachingValues(llvm::SmallVectorImpl<mlir::Operation *> &reach,
                           mlir::Value seq) {
  reach.clear();
  llvm::SmallPtrSet<mlir::Value, 16> visited;
  populateSets(reach, visited, seq);
}

void ArrayCopyAnalysis::arrayAccesses(
    llvm::SmallVectorImpl<mlir::Operation *> &accesses,
    llvm::ArrayRef<mlir::Operation *> reach, mlir::Value addr,
    mlir::Operation *store) {
  accesses.clear();
  llvm::SmallPtrSet<mlir::Value, 16> live;  // live values from addr
  llvm::SmallVector<mlir::Value, 16> queue; // uses of addr
  auto appendToQueue = [&](mlir::Value val) {
    for (auto &use : val.getUses())
      queue.push_back(use.get());
  };

  // Build the set of uses of all loads of `addr`.
  for (auto *ro : reach)
    if (auto load = mlir::dyn_cast<ArrayLoadOp>(ro))
      if (load.memref() == addr) {
        live.insert(load.getResult());
        appendToQueue(load.getResult());
      }

  // Process the worklist until done.
  while (!queue.empty()) {
    auto u = queue.pop_back_val();
    auto *owner = u.getDefiningOp();
    if (!owner || !domInfo.properlyDominates(owner, store))
      continue;
    auto structuredLoop = [&](auto ro) {
      if (auto blockArg = ro.iterArgToBlockArg(u)) {
        live.insert(blockArg);
        auto arg = blockArg.getArgNumber();
        auto output = ro.getResult(ro.finalValue() ? arg : arg - 1);
        live.insert(output);
        appendToQueue(output);
        appendToQueue(blockArg);
      }
    };
    if (auto ro = mlir::dyn_cast<DoLoopOp>(owner)) {
      structuredLoop(ro);
    } else if (auto ro = mlir::dyn_cast<IterWhileOp>(owner)) {
      structuredLoop(ro);
    } else if (auto rs = mlir::dyn_cast<ResultOp>(owner)) {
      auto *parent = rs.getParentRegion()->getParentOp();
      if (auto ifOp = mlir::dyn_cast<fir::IfOp>(parent))
        for (auto i : llvm::enumerate(rs.getOperands()))
          if (live.contains(i.value())) {
            auto res = ifOp.getResult(i.index());
            live.insert(res);
            appendToQueue(res);
          }
    } else if (auto ao = mlir::dyn_cast<ArrayFetchOp>(owner)) {
      if (live.contains(ao.sequence()))
        accesses.push_back(owner);
    } else if (auto ao = mlir::dyn_cast<ArrayUpdateOp>(owner)) {
      if (live.contains(ao.sequence())) {
        accesses.push_back(owner);
        live.insert(ao.getResult());
        appendToQueue(ao.getResult());
      }
    }
  }
}

static bool conflictOnLoad(llvm::ArrayRef<mlir::Operation *> reach,
                           ArrayStoreOp st) {
  auto addr = st.memref();
  mlir::Value load;
  for (auto *op : reach)
    if (auto ld = mlir::dyn_cast<ArrayLoadOp>(op))
      if (ld.memref() == addr) {
        if (load)
          return true;
        load = ld;
      }
  return false;
}

static bool conflictOnMerge(llvm::ArrayRef<mlir::Operation *> accesses) {
  llvm::SmallVector<mlir::Value, 8> indices;
  for (auto *op : accesses) {
    llvm::SmallVector<mlir::Value, 8> compareVector;
    if (auto u = mlir::dyn_cast<ArrayUpdateOp>(op)) {
      if (indices.empty()) {
        indices = u.indices();
        continue;
      }
      compareVector = u.indices();
    } else if (auto f = mlir::dyn_cast<ArrayFetchOp>(op)) {
      if (indices.empty()) {
        indices = f.indices();
        continue;
      }
      compareVector = f.indices();
    } else {
      mlir::emitError(op->getLoc(), "unexpected operation in analysis");
    }
    if (compareVector.size() != indices.size() ||
        llvm::any_of(llvm::zip(compareVector, indices), [&](auto pair) {
          return std::get<0>(pair) != std::get<1>(pair);
        }))
      return true;
  }
  return false;
}

// Are either of types of conflicts present?
inline bool conflictDetected(llvm::ArrayRef<mlir::Operation *> reach,
                             llvm::ArrayRef<mlir::Operation *> accesses,
                             ArrayStoreOp st) {
  return conflictOnLoad(reach, st) || conflictOnMerge(accesses);
}

void ArrayCopyAnalysis::construct(mlir::MutableArrayRef<mlir::Region> regions) {
  for (auto &region : regions)
    for (auto &block : region.getBlocks())
      for (auto &op : block.getOperations()) {
        if (op.getNumRegions())
          construct(op.getRegions());
        if (auto st = mlir::dyn_cast<fir::ArrayStoreOp>(op)) {
          llvm::SmallVector<Operation *, 16> values;
          reachingValues(values, st.sequence());
          llvm::SmallVector<Operation *, 16> accesses;
          arrayAccesses(accesses, values, st.memref(), &op);
          if (conflictDetected(values, accesses, st)) {
            conflicts.insert(&op);
            for (auto *acc : accesses)
              conflicts.insert(acc);
          }
        }
      }
}

namespace {
class ArrayLoadConversion : public mlir::OpRewritePattern<ArrayLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ArrayLoadOp upd,
                  mlir::PatternRewriter &rewriter) const override {
    return mlir::success();
  }
};

class ArrayStoreConversion : public mlir::OpRewritePattern<ArrayStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ArrayStoreOp upd,
                  mlir::PatternRewriter &rewriter) const override {
    return mlir::success();
  }
};

class ArrayFetchConversion : public mlir::OpRewritePattern<ArrayFetchOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ArrayFetchOp upd,
                  mlir::PatternRewriter &rewriter) const override {
    return mlir::success();
  }
};

class ArrayUpdateConversion : public mlir::OpRewritePattern<ArrayUpdateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(ArrayUpdateOp upd,
                  mlir::PatternRewriter &rewriter) const override {
    return mlir::success();
  }
};

class ArrayValueCopyConverter
    : public ArrayValueCopyBase<ArrayValueCopyConverter> {
public:
  void runOnFunction() override {
    auto func = getFunction();
    LLVM_DEBUG(llvm::dbgs() << "array-value-copy pass: " << func.getName());
    auto *context = &getContext();
    getAnalysisManager().getAnalysis<ArrayCopyAnalysis>();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<ArrayLoadConversion>(context);
    patterns.insert<ArrayStoreConversion>(context);
    patterns.insert<ArrayFetchConversion>(context);
    patterns.insert<ArrayUpdateConversion>(context);
    mlir::ConversionTarget target(*context);
    target
        .addIllegalOp<ArrayLoadOp, ArrayStoreOp, ArrayFetchOp, ArrayUpdateOp>();
    target.addLegalDialect<FIROpsDialect, mlir::scf::SCFDialect,
                           mlir::StandardOpsDialect>();
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in array-value-copy pass");
      signalPassFailure();
    }
  };
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createArrayValueCopyPass() {
  return std::make_unique<ArrayValueCopyConverter>();
}

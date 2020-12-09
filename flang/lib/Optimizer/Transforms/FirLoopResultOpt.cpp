//===- FirLoopResultOpt.cpp - Optimization pass for fir loops    ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "flang-fir-result-opt"

namespace {

class LoopResultRemoval : public mlir::OpRewritePattern<fir::DoLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LoopResultRemoval(mlir::MLIRContext *c) : OpRewritePattern(c) {}

  mlir::LogicalResult
  matchAndRewrite(fir::DoLoopOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "inspecting: " << loop << '\n');
    if (loop.getNumResults() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "loop failed: too many results\n");
      return mlir::failure();
    }
    for (auto r : loop.getResults())
      if (valueUseful(r)) {
        LLVM_DEBUG(llvm::dbgs() << "loop failed: result used\n");
        return mlir::failure();
      }
    auto loc = loop.getLoc();
    auto newLoop = rewriter.create<fir::DoLoopOp>(
        loc, loop.lowerBound(), loop.upperBound(), loop.step());
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(newLoop.getBody());
    mlir::BlockAndValueMapping valueMap;
    valueMap.map(loop.getInductionVar(), newLoop.getInductionVar());
    for (auto i = loop.getBody()->begin(), e = std::prev(loop.getBody()->end());
         i != e; ++i)
      rewriter.insert(i->clone(valueMap));
    rewriter.restoreInsertionPoint(insPt);
    LLVM_DEBUG(llvm::dbgs() << "replacing with: " << newLoop << '\n');
    rewriter.replaceOpWithNewOp<fir::UndefOp>(loop,
                                              loop.getResult(0).getType());
    return mlir::success();
  }

private:
  bool valueUseful(mlir::Value v) const {
    for (auto &use : v.getUses()) {
      if (auto convert = dyn_cast<fir::ConvertOp>(use.getOwner()))
        return valueUseful(convert.getResult());
      if (auto store = dyn_cast<fir::StoreOp>(use.getOwner())) {
        bool anyLoad = false;
        for (auto &su : store.memref().getUses())
          if (auto load = dyn_cast<fir::LoadOp>(su.getOwner()))
            anyLoad = true;
        return anyLoad;
      }
      return true;
    }
    return false;
  }
};

class FirLoopResultOptPass
    : public fir::FirLoopResultOptBase<FirLoopResultOptPass> {
public:
  void runOnFunction() override {
    auto *context = &getContext();
    auto function = getFunction();
    LLVM_DEBUG(llvm::dbgs() << "function: <<<<\n" << function << "\n>>>>\n");
    mlir::OwningRewritePatternList patterns;
    patterns.insert<LoopResultRemoval>(context);
    mlir::ConversionTarget target = *context;
    target.addLegalDialect<fir::FIROpsDialect, mlir::StandardOpsDialect>();
    target.addDynamicallyLegalOp<fir::DoLoopOp>(
        [&](fir::DoLoopOp op) { return op.getNumResults() == 0; });
    if (mlir::failed(mlir::applyPartialConversion(function, target,
                                                  std::move(patterns)))) {
      mlir::emitWarning(mlir::UnknownLoc::get(context),
                        "fir loop result optimization failed\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "new-func: <<<<\n" << function << "\n>>>>\n");
    }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> fir::createFirLoopResultOptPass() {
  return std::make_unique<FirLoopResultOptPass>();
}

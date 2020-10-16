//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "../runtime/clock.h"
#include "../runtime/stop.h"
#include "RTBuilder.h"
#include "StatementContext.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Support/BoxValue.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-runtime"

using namespace Fortran::runtime;

// TODO: We don't have runtime library support for various features. When they
// are encountered, we emit an error message and exit immediately.
static void noRuntimeSupport(mlir::Location loc, llvm::StringRef stmt) {
  mlir::emitError(loc, "There is no runtime support for ")
      << stmt << " statement.\n";
  std::exit(1);
}

/// Runtime calls that do not return to the caller indicate this condition by
/// terminating the current basic block with an unreachable op.
static void genUnreachable(Fortran::lower::FirOpBuilder &builder,
                           mlir::Location loc) {
  builder.create<fir::UnreachableOp>(loc);
  auto *newBlock = builder.getBlock()->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToStart(newBlock);
}

//===----------------------------------------------------------------------===//
// Misc. Fortran statements that lower to runtime calls
//===----------------------------------------------------------------------===//

void Fortran::lower::genStopStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::StopStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  Fortran::lower::StatementContext stmtCtx;
  llvm::SmallVector<mlir::Value, 8> operands;
  mlir::FuncOp callee;
  mlir::FunctionType calleeType;
  // First operand is stop code (zero if absent)
  if (const auto &code =
          std::get<std::optional<Fortran::parser::StopCode>>(stmt.t)) {
    auto expr =
        converter.genExprValue(*Fortran::semantics::GetExpr(*code), stmtCtx);
    LLVM_DEBUG(llvm::dbgs() << "stop expression: "; expr.dump();
               llvm::dbgs() << '\n');
    expr.match(
        [&](const fir::CharBoxValue &x) {
          callee = getRuntimeFunc<mkRTKey(StopStatementText)>(loc, builder);
          calleeType = callee.getType();
          // Creates a pair of operands for the CHARACTER and its LEN.
          operands.push_back(
              builder.createConvert(loc, calleeType.getInput(0), x.getAddr()));
          operands.push_back(
              builder.createConvert(loc, calleeType.getInput(1), x.getLen()));
        },
        [&](fir::UnboxedValue x) {
          callee = getRuntimeFunc<mkRTKey(StopStatement)>(loc, builder);
          calleeType = callee.getType();
          auto cast = builder.createConvert(loc, calleeType.getInput(0), x);
          operands.push_back(cast);
        },
        [&](auto) {
          mlir::emitError(loc, "unhandled expression in STOP");
          std::exit(1);
        });
  } else {
    callee = getRuntimeFunc<mkRTKey(StopStatement)>(loc, builder);
    calleeType = callee.getType();
    operands.push_back(
        builder.createIntegerConstant(loc, calleeType.getInput(0), 0));
  }

  // Second operand indicates ERROR STOP
  bool isError = std::get<Fortran::parser::StopStmt::Kind>(stmt.t) ==
                 Fortran::parser::StopStmt::Kind::ErrorStop;
  operands.push_back(builder.createIntegerConstant(
      loc, calleeType.getInput(operands.size()), isError));

  // Third operand indicates QUIET (default to false).
  if (const auto &quiet =
          std::get<std::optional<Fortran::parser::ScalarLogicalExpr>>(stmt.t)) {
    auto expr = Fortran::semantics::GetExpr(*quiet);
    assert(expr && "failed getting typed expression");
    auto q = fir::getBase(converter.genExprValue(*expr, stmtCtx));
    operands.push_back(
        builder.createConvert(loc, calleeType.getInput(operands.size()), q));
  } else {
    operands.push_back(builder.createIntegerConstant(
        loc, calleeType.getInput(operands.size()), 0));
  }

  builder.create<fir::CallOp>(loc, callee, operands);
  genUnreachable(builder, loc);
}

void Fortran::lower::genFailImageStatement(
    Fortran::lower::AbstractConverter &converter) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto callee = getRuntimeFunc<mkRTKey(FailImageStatement)>(loc, builder);
  builder.create<fir::CallOp>(loc, callee, llvm::None);
  genUnreachable(builder, loc);
}

void Fortran::lower::genEventPostStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventPostStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "EVENT POST");
}

void Fortran::lower::genEventWaitStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EventWaitStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "EVENT WAIT");
}

void Fortran::lower::genLockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::LockStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "LOCK");
}

void Fortran::lower::genUnlockStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::UnlockStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "UNLOCK");
}

void Fortran::lower::genSyncAllStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncAllStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC ALL");
}

void Fortran::lower::genSyncImagesStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncImagesStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC IMAGES");
}

void Fortran::lower::genSyncMemoryStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncMemoryStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC MEMORY");
}

void Fortran::lower::genSyncTeamStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::SyncTeamStmt &) {
  // FIXME: There is no runtime call to make for this yet.
  noRuntimeSupport(converter.getCurrentLocation(), "SYNC TEAM");
}

void Fortran::lower::genPauseStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PauseStmt &) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto callee = getRuntimeFunc<mkRTKey(PauseStatement)>(loc, builder);
  builder.create<fir::CallOp>(loc, callee, llvm::None);
}

void Fortran::lower::genDateAndTime(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc,
                                    llvm::Optional<fir::CharBoxValue> date,
                                    llvm::Optional<fir::CharBoxValue> time,
                                    llvm::Optional<fir::CharBoxValue> zone) {
  auto callee = getRuntimeFunc<mkRTKey(DateAndTime)>(loc, builder);
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value zero;
  auto splitArg = [&](llvm::Optional<fir::CharBoxValue> arg,
                      mlir::Value &buffer, mlir::Value &len) {
    if (arg) {
      buffer = arg->getBuffer();
      len = arg->getLen();
    } else {
      if (!zero)
        zero = builder.createIntegerConstant(loc, idxTy, 0);
      buffer = zero;
      len = zero;
    }
  };
  mlir::Value dateBuffer;
  mlir::Value dateLen;
  splitArg(date, dateBuffer, dateLen);
  mlir::Value timeBuffer;
  mlir::Value timeLen;
  splitArg(time, timeBuffer, timeLen);
  mlir::Value zoneBuffer;
  mlir::Value zoneLen;
  splitArg(zone, zoneBuffer, zoneLen);

  llvm::SmallVector<mlir::Value, 8> args{dateBuffer, timeBuffer, zoneBuffer,
                                         dateLen,    timeLen,    zoneLen};
  llvm::SmallVector<mlir::Value, 8> operands;
  for (auto [fst, snd] : llvm::zip(args, callee.getType().getInputs()))
    operands.emplace_back(builder.createConvert(loc, snd, fst));
  builder.create<fir::CallOp>(loc, callee, operands);
}

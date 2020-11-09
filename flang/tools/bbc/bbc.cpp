//===- bbc.cpp - Burnside Bridge Compiler -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a tool for translating Fortran sources to the FIR dialect of MLIR.
///
//===----------------------------------------------------------------------===//

#include "flang/Common/Fortran-features.h"
#include "flang/Common/default-kinds.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Support/Verifier.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/OptPasses.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/unparse-with-symbols.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// Some basic command-line options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::Required,
                                                llvm::cl::desc("<input file>"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Specify the output filename"),
                   llvm::cl::value_desc("filename"));

static llvm::cl::list<std::string>
    includeDirs("I", llvm::cl::desc("include search paths"));

static llvm::cl::opt<std::string>
    intrinsicModuleDir("intrinsic-module-directory",
                       llvm::cl::desc("intrinsic module directory"));

static llvm::cl::opt<std::string>
    moduleDir("module", llvm::cl::desc("module output directory (default .)"),
              llvm::cl::init("."));

static llvm::cl::opt<std::string>
    moduleSuffix("module-suffix", llvm::cl::desc("module file suffix override"),
                 llvm::cl::init(".mod"));

static llvm::cl::opt<bool>
    emitLLVM("emit-llvm",
             llvm::cl::desc("Add passes to lower to and emit LLVM IR"),
             llvm::cl::init(false));

static llvm::cl::opt<bool>
    emitFIR("emit-fir",
            llvm::cl::desc("Dump the FIR created by lowering and exit"),
            llvm::cl::init(false));

static llvm::cl::opt<bool> fixedForm("Mfixed",
                                     llvm::cl::desc("used fixed form"),
                                     llvm::cl::init(false));

static llvm::cl::opt<bool> freeForm("Mfree", llvm::cl::desc("used free form"),
                                    llvm::cl::init(false));

static llvm::cl::opt<bool> warnStdViolation("Mstandard",
                                            llvm::cl::desc("emit warnings"),
                                            llvm::cl::init(false));

static llvm::cl::opt<bool> warnIsError("Werror",
                                       llvm::cl::desc("warnings are errors"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool> dumpSymbols("dump-symbols",
                                       llvm::cl::desc("dump the symbol table"),
                                       llvm::cl::init(false));

static llvm::cl::opt<bool> pftDumpTest(
    "pft-test",
    llvm::cl::desc("parse the input, create a PFT, dump it, and exit"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableOpenMP("fopenmp",
                                        llvm::cl::desc("enable openmp"),
                                        llvm::cl::init(false));

static llvm::cl::opt<bool> enableOpenACC("fopenacc",
                                         llvm::cl::desc("enable openacc"),
                                         llvm::cl::init(false));

static llvm::cl::opt<bool> dumpModuleOnFailure("dump-module-on-failure",
                                               llvm::cl::init(false));

static llvm::cl::opt<std::string>
    targetTriple("target", llvm::cl::desc("specify a target triple"));

//===----------------------------------------------------------------------===//

using ProgramName = std::string;

// Print the module without the "module { ... }" wrapper.
static void printModule(mlir::ModuleOp mlirModule, llvm::raw_ostream &out) {
  for (auto &op : mlirModule.getBody()->without_terminator())
    out << op << '\n';
  out << '\n';
}

// Translate front-end KINDs for use in the IR and code gen.
static std::vector<fir::KindTy>
fromDefaultKinds(const Fortran::common::IntrinsicTypeDefaultKinds &defKinds) {
  return {static_cast<fir::KindTy>(defKinds.GetDefaultKind(
              Fortran::common::TypeCategory::Character)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Complex)),
          static_cast<fir::KindTy>(defKinds.doublePrecisionKind()),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Integer)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Logical)),
          static_cast<fir::KindTy>(
              defKinds.GetDefaultKind(Fortran::common::TypeCategory::Real))};
}

//===----------------------------------------------------------------------===//
// Translate Fortran input to FIR, a dialect of MLIR.
//===----------------------------------------------------------------------===//

static mlir::LogicalResult convertFortranSourceToMLIR(
    std::string path, Fortran::parser::Options options,
    const ProgramName &programPrefix,
    Fortran::semantics::SemanticsContext &semanticsContext,
    const mlir::PassPipelineCLParser &passPipeline) {
  if (!(fixedForm || freeForm)) {
    auto dot = path.rfind(".");
    if (dot != std::string::npos) {
      std::string suffix = path.substr(dot + 1);
      options.isFixedForm = suffix == "f" || suffix == "F" || suffix == "ff";
    }
  }

  // enable parsing of OpenMP
  if (enableOpenMP) {
    options.features.Enable(Fortran::common::LanguageFeature::OpenMP);
    options.predefinitions.emplace_back("_OPENMP", "201511");
  }

  // enable parsing of OpenACC
  if (enableOpenACC) {
    options.features.Enable(Fortran::common::LanguageFeature::OpenACC);
    options.predefinitions.emplace_back("_OPENACC", "201911");
  }

  // prep for prescan and parse
  options.searchDirectories = includeDirs;
  Fortran::parser::Parsing parsing{semanticsContext.allCookedSources()};
  parsing.Prescan(path, options);
  if (!parsing.messages().empty() &&
      (warnIsError || parsing.messages().AnyFatalError())) {
    llvm::errs() << programPrefix << "could not scan " << path << '\n';
    parsing.messages().Emit(llvm::errs(), parsing.allCooked());
    return mlir::failure();
  }

  // parse the input Fortran
  parsing.Parse(llvm::outs());
  parsing.messages().Emit(llvm::errs(), parsing.allCooked());
  if (!parsing.consumedWholeFile()) {
    parsing.EmitMessage(llvm::errs(), parsing.finalRestingPlace(),
                        "parser FAIL (final position)");
    return mlir::failure();
  }
  if ((!parsing.messages().empty() &&
       (warnIsError || parsing.messages().AnyFatalError())) ||
      !parsing.parseTree().has_value()) {
    llvm::errs() << programPrefix << "could not parse " << path << '\n';
    return mlir::failure();
  }

  // run semantics
  auto &parseTree = *parsing.parseTree();
  Fortran::semantics::Semantics semantics(semanticsContext, parseTree,
                                          parsing.cooked().AsCharBlock());
  semantics.Perform();
  semantics.EmitMessages(llvm::errs());
  if (semantics.AnyFatalError()) {
    llvm::errs() << programPrefix << "semantic errors in " << path << '\n';
    return mlir::failure();
  }
  if (dumpSymbols)
    semantics.DumpSymbols(llvm::outs());

  if (pftDumpTest) {
    if (auto ast = Fortran::lower::createPFT(parseTree, semanticsContext)) {
      Fortran::lower::dumpPFT(llvm::outs(), *ast);
      return mlir::success();
    }
    llvm::errs() << "Pre FIR Tree is NULL.\n";
    return mlir::failure();
  }

  // translate to FIR dialect of MLIR
  llvm::Triple triple(fir::determineTargetTriple(targetTriple));
  fir::NameUniquer nameUniquer;
  mlir::MLIRContext ctx;
  fir::registerAndLoadDialects(ctx);
  auto &defKinds = semanticsContext.defaultKinds();
  fir::KindMapping kindMap(
      &ctx, llvm::ArrayRef<fir::KindTy>{fromDefaultKinds(defKinds)});
  auto burnside = Fortran::lower::LoweringBridge::create(
      ctx, defKinds, semanticsContext.intrinsics(), parsing.allCooked(), triple,
      nameUniquer, kindMap);
  burnside.lower(parseTree, semanticsContext);
  mlir::ModuleOp mlirModule = burnside.getModule();
  std::error_code ec;
  std::string outputName = outputFilename;
  if (!outputName.size())
    outputName = llvm::sys::path::stem(inputFilename).str().append(".mlir");
  llvm::raw_fd_ostream out(outputName, ec);
  if (ec)
    return mlir::emitError(mlir::UnknownLoc::get(&ctx),
                           "could not open output file ")
           << outputName;

  // Otherwise run the default passes.
  mlir::PassManager pm(mlirModule.getContext());
  mlir::applyPassManagerCLOptions(pm);
  if (passPipeline.hasAnyOccurrences()) {
    // run the command-line specified pipeline
    passPipeline.addToPipeline(pm);
  } else if (emitFIR) {
    // --emit-fir: Build the IR, verify it, and dump the IR if the IR passes
    // verification. Use --dump-module-on-failure to dump invalid IR.
    pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());
    if (mlir::failed(pm.run(mlirModule))) {
      llvm::errs() << "FATAL: verification of lowering to FIR failed";
      if (dumpModuleOnFailure)
        mlirModule.dump();
      return mlir::failure();
    }
    printModule(mlirModule, out);
    return mlir::success();
  } else {
    // run the default canned pipeline
    pm.addPass(std::make_unique<Fortran::lower::VerifierPass>());

    // simplify the IR
    pm.addPass(fir::createArrayValueCopyPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(fir::createCSEPass());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCSEPass());

    // convert control flow to CFG form
    pm.addPass(fir::createFirToCfgPass());
    pm.addPass(fir::createControlFlowLoweringPass());
    pm.addPass(mlir::createLowerToCFGPass());

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(fir::createCSEPass());
  }

  if (emitLLVM) {
    // Continue to lower from MLIR down to LLVM IR. Emit LLVM and MLIR.
    pm.addPass(fir::createFirCodeGenRewritePass());
    pm.addPass(fir::createFirTargetRewritePass());
    pm.addPass(fir::createFIRToLLVMPass(nameUniquer));
    
    std::error_code ec;
    llvm::ToolOutputFile outFile(outputName + ".ll", ec,
                                 llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "can't open output file " + outputName + ".ll";
      return mlir::failure();
    }
    pm.addPass(fir::createLLVMDialectToLLVMPass(outFile.os()));
    if (mlir::succeeded(pm.run(mlirModule))) {
      outFile.keep();
      printModule(mlirModule, out);
      return mlir::success();
    }
  } else if (mlir::succeeded(pm.run(mlirModule))) {
    // Emit MLIR and do not lower to LLVM IR.
    printModule(mlirModule, out);
    return mlir::success();
  }
  // Something went wrong. Try to dump the MLIR module.
  llvm::errs() << "oops, pass manager reported failure\n";
  if (dumpModuleOnFailure)
    mlirModule.dump();
  return mlir::failure();
}

int main(int argc, char **argv) {
  fir::registerFIRPasses();
  fir::registerOptPasses();
  [[maybe_unused]] llvm::InitLLVM y(argc, argv);

  llvm::InitializeAllTargets();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  llvm::cl::ParseCommandLineOptions(argc, argv, "Burnside Bridge Compiler\n");

  ProgramName programPrefix;
  programPrefix = argv[0] + ": "s;

  if (includeDirs.size() == 0)
    includeDirs.push_back(".");

  if (!intrinsicModuleDir.empty())
    includeDirs.insert(includeDirs.begin(), intrinsicModuleDir);

  Fortran::parser::Options options;
  options.predefinitions.emplace_back("__F18", "1");
  options.predefinitions.emplace_back("__F18_MAJOR__", "1");
  options.predefinitions.emplace_back("__F18_MINOR__", "1");
  options.predefinitions.emplace_back("__F18_PATCHLEVEL__", "1");
#if __x86_64__
  options.predefinitions.emplace_back("__x86_64__", "1");
#endif

  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds;
  Fortran::parser::AllSources allSources;
  Fortran::parser::AllCookedSources allCookedSources(allSources);
  Fortran::semantics::SemanticsContext semanticsContext{
      defaultKinds, options.features, allCookedSources};
  semanticsContext.set_moduleDirectory(moduleDir)
      .set_moduleFileSuffix(moduleSuffix)
      .set_searchDirectories(includeDirs)
      .set_warnOnNonstandardUsage(warnStdViolation)
      .set_warningsAreErrors(warnIsError);

  return mlir::failed(convertFortranSourceToMLIR(
      inputFilename, options, programPrefix, semanticsContext, passPipe));
}

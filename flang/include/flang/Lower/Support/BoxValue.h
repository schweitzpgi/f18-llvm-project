//===-- Lower/Support/BoxValue.h -- internal box values ---------*- C++ -*-===//
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

#ifndef LOWER_SUPPORT_BOXVALUE_H
#define LOWER_SUPPORT_BOXVALUE_H

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/Matcher.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace fir {
class CharBoxValue;
class ArrayBoxValue;
class CharArrayBoxValue;
class BoxValue;
class ProcBoxValue;

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const CharBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ArrayBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const CharArrayBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const BoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ProcBoxValue &);

//===----------------------------------------------------------------------===//
//
// Boxed values
//
// Define a set of containers used internally by the lowering bridge to keep
// track of extended values associated with a Fortran subexpression. These
// associations are maintained during the construction of FIR.
//
//===----------------------------------------------------------------------===//

/// Most expressions of intrinsic type can be passed unboxed. Their properties
/// are known statically.
using UnboxedValue = mlir::Value;

/// Abstract base class.
class AbstractBox {
public:
  AbstractBox() = delete;
  AbstractBox(mlir::Value addr) : addr{addr} {}

  /// FIXME: this comment is not true anymore since genLoad
  /// is loading constant length characters. What is the impact  /// ?
  /// An abstract box always contains a memory reference to a value.
  mlir::Value getAddr() const { return addr; }

protected:
  mlir::Value addr;
};

/// Expressions of CHARACTER type have an associated, possibly dynamic LEN
/// value.
class CharBoxValue : public AbstractBox {
public:
  CharBoxValue(mlir::Value addr, mlir::Value len)
      : AbstractBox{addr}, len{len} {
    if (addr && addr.getType().template isa<fir::BoxCharType>())
      llvm::report_fatal_error("BoxChar should not be in CharBoxValue");
  }

  CharBoxValue clone(mlir::Value newBase) const { return {newBase, len}; }

  /// Convenience alias to get the memory reference to the buffer.
  mlir::Value getBuffer() const { return getAddr(); }

  mlir::Value getLen() const { return len; }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const CharBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

protected:
  mlir::Value len;
};

/// Abstract base class.
/// Expressions of type array have at minimum a shape. These expressions may
/// have lbound attributes (dynamic values) that affect the interpretation of
/// indexing expressions.
class AbstractArrayBox {
public:
  AbstractArrayBox() = default;
  AbstractArrayBox(llvm::ArrayRef<mlir::Value> extents,
                   llvm::ArrayRef<mlir::Value> lbounds)
      : extents{extents.begin(), extents.end()}, lbounds{lbounds.begin(),
                                                         lbounds.end()} {}

  // Every array has extents that describe its shape.
  const llvm::SmallVectorImpl<mlir::Value> &getExtents() const {
    return extents;
  }

  // An array expression may have user-defined lower bound values.
  // If this vector is empty, the default in all dimensions in `1`.
  const llvm::SmallVectorImpl<mlir::Value> &getLBounds() const {
    return lbounds;
  }

  bool lboundsAllOne() const { return lbounds.empty(); }

protected:
  llvm::SmallVector<mlir::Value, 4> extents;
  llvm::SmallVector<mlir::Value, 4> lbounds;
};

/// Expressions with rank > 0 have extents. They may also have lbounds that are
/// not 1.
class ArrayBoxValue : public AbstractBox, public AbstractArrayBox {
public:
  ArrayBoxValue(mlir::Value addr, llvm::ArrayRef<mlir::Value> extents,
                llvm::ArrayRef<mlir::Value> lbounds = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds} {}

  ArrayBoxValue clone(mlir::Value newBase) const {
    return {newBase, extents, lbounds};
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ArrayBoxValue &);
  LLVM_DUMP_METHOD void dump() const { operator<<(llvm::errs(), *this); }
};

/// Expressions of type CHARACTER and with rank > 0.
class CharArrayBoxValue : public CharBoxValue, public AbstractArrayBox {
public:
  CharArrayBoxValue(mlir::Value addr, mlir::Value len,
                    llvm::ArrayRef<mlir::Value> extents,
                    llvm::ArrayRef<mlir::Value> lbounds = {})
      : CharBoxValue{addr, len}, AbstractArrayBox{extents, lbounds} {}

  CharArrayBoxValue clone(mlir::Value newBase) const {
    return {newBase, len, extents, lbounds};
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const CharArrayBoxValue &);
  LLVM_DUMP_METHOD void dump() const { operator<<(llvm::errs(), *this); }
};

/// Expressions that are procedure POINTERs may need a set of references to
/// variables in the host scope.
class ProcBoxValue : public AbstractBox {
public:
  ProcBoxValue(mlir::Value addr, mlir::Value context)
      : AbstractBox{addr}, hostContext{context} {}

  ProcBoxValue clone(mlir::Value newBase) const {
    return {newBase, hostContext};
  }

  mlir::Value getHostContext() const { return hostContext; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ProcBoxValue &);
  LLVM_DUMP_METHOD void dump() const { operator<<(llvm::errs(), *this); }

protected:
  mlir::Value hostContext;
};

/// In the generalized form, a boxed value can have a dynamic size, be an array
/// with dynamic extents and lbounds, and take dynamic type parameters.
class BoxValue : public AbstractBox, public AbstractArrayBox {
public:
  BoxValue(mlir::Value addr) : AbstractBox{addr}, AbstractArrayBox{} {}
  BoxValue(mlir::Value addr, mlir::Value len)
      : AbstractBox{addr}, AbstractArrayBox{}, len{len} {}
  BoxValue(mlir::Value addr, llvm::ArrayRef<mlir::Value> extents,
           llvm::ArrayRef<mlir::Value> lbounds = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds} {}
  BoxValue(mlir::Value addr, mlir::Value len,
           llvm::ArrayRef<mlir::Value> params,
           llvm::ArrayRef<mlir::Value> extents,
           llvm::ArrayRef<mlir::Value> lbounds = {})
      : AbstractBox{addr}, AbstractArrayBox{extents, lbounds}, len{len},
        params{params.begin(), params.end()} {}

  BoxValue clone(mlir::Value newBase) const {
    return {newBase, len, params, extents, lbounds};
  }

  mlir::Value getLen() const { return len; }
  const llvm::SmallVectorImpl<mlir::Value> &getLenTypeParams() const {
    return params;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const BoxValue &);
  LLVM_DUMP_METHOD void dump() const { operator<<(llvm::errs(), *this); }

protected:
  mlir::Value len;
  llvm::SmallVector<mlir::Value, 2> params;
};

/// Used for triple notation (array slices)
using RangeBoxValue = std::tuple<mlir::Value, mlir::Value, mlir::Value>;

class ExtendedValue;

mlir::Value getBase(const ExtendedValue &exv);
mlir::Value getLen(const ExtendedValue &exv);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ExtendedValue &);
ExtendedValue substBase(const ExtendedValue &exv, mlir::Value base);
bool isArray(const ExtendedValue &exv);

/// An extended value is a box of values pertaining to a discrete entity. It is
/// used in lowering to track all the runtime values related to an entity. For
/// example, an entity may have an address in memory that contains its value(s)
/// as well as various attribute values that describe the shape and starting
/// indices if it is an array entity.
class ExtendedValue : public details::matcher<ExtendedValue> {
public:
  using VT = std::variant<UnboxedValue, CharBoxValue, ArrayBoxValue,
                          CharArrayBoxValue, BoxValue, ProcBoxValue>;

  ExtendedValue() : box{UnboxedValue{}} {}
  ExtendedValue(const ExtendedValue &) = default;
  ExtendedValue(ExtendedValue &&) = default;
  template <typename A, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<A>, ExtendedValue>>>
  constexpr ExtendedValue(A &&a) : box{std::forward<A>(a)} {
    if (auto b = getUnboxed()) {
      if (*b) {
        auto type = b->getType();
        if (type.template isa<fir::BoxCharType>())
          llvm::report_fatal_error("BoxChar should be unboxed");
        if (auto refType = type.template dyn_cast<fir::ReferenceType>())
          type = refType.getEleTy();
        if (auto seqType = type.template dyn_cast<fir::SequenceType>())
          type = seqType.getEleTy();
        if (type.template isa<fir::CharacterType>())
          llvm::report_fatal_error(
              "character buffer should be in CharBoxValue");
      }
    }
  }

  template <typename A>
  constexpr const A *getBoxOf() const {
    return std::get_if<A>(&box);
  }

  constexpr const CharBoxValue *getCharBox() const {
    return getBoxOf<CharBoxValue>();
  }

  constexpr const UnboxedValue *getUnboxed() const {
    return getBoxOf<UnboxedValue>();
  }

  /// LLVM style debugging of extended values
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this << '\n'; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ExtendedValue &);

  const VT &matchee() const { return box; }

private:
  VT box;
};
} // namespace fir

#endif // LOWER_SUPPORT_BOXVALUE_H

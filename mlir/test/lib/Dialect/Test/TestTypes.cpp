//===- TestTypes.cpp - MLIR Test Dialect Types ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the TestDialect for testing various
// features of MLIR.
//
//===----------------------------------------------------------------------===//

#include "TestTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {

// Custom parser for SignednessSemantics
static ParseResult Parse(DialectAsmParser &parser,
                         TestIntegerType::SignednessSemantics &result) {
  StringRef signStr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&signStr))
    return mlir::failure();
  if (signStr.compare_lower("u") || signStr.compare_lower("unsigned"))
    result = TestIntegerType::SignednessSemantics::Unsigned;
  else if (signStr.compare_lower("s") || signStr.compare_lower("signed"))
    result = TestIntegerType::SignednessSemantics::Signed;
  else if (signStr.compare_lower("n") || signStr.compare_lower("none"))
    result = TestIntegerType::SignednessSemantics::Signless;
  else {
    parser.emitError(loc, "expected signed, unsigned, or none");
    return mlir::failure();
  }
  return mlir::success();
}

// Custom printer for SignednessSemantics
static void Print(DialectAsmPrinter &printer,
                  const TestIntegerType::SignednessSemantics &ss) {
  switch (ss) {
  case TestIntegerType::SignednessSemantics::Unsigned:
    printer << "unsigned";
    break;
  case TestIntegerType::SignednessSemantics::Signed:
    printer << "signed";
    break;
  case TestIntegerType::SignednessSemantics::Signless:
    printer << "none";
    break;
  }
}

Type CompoundAType::parse(::mlir::MLIRContext *ctxt,
                          ::mlir::DialectAsmParser &parser) {
  int widthOfSomething;
  Type oneType;
  SmallVector<int, 4> arrayOfInts;
  if (parser.parseLess())
    return Type();
  if (parser.parseInteger(widthOfSomething))
    return Type();
  if (parser.parseComma())
    return Type();
  if (parser.parseType(oneType))
    return Type();
  if (parser.parseComma())
    return Type();

  if (parser.parseLSquare())
    return Type();
  int i;
  while (!*parser.parseOptionalInteger(i)) {
    arrayOfInts.push_back(i);
    if (parser.parseOptionalComma())
      break;
  }
  if (parser.parseRSquare())
    return Type();
  if (parser.parseGreater())
    return Type();

  return get(ctxt, widthOfSomething, oneType, arrayOfInts);
}
void CompoundAType::print(::mlir::DialectAsmPrinter &printer) const {
  printer << "cmpnd_a<" << getWidthOfSomething() << ", " << getOneType()
          << ", [";
  auto intArray = getArrayOfInts();
  for (size_t idx = 0; idx < intArray.size(); idx++) {
    printer << intArray[idx];
    if (idx < intArray.size() - 1)
      printer << ", ";
  }
  printer << "]>";
}

bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}

llvm::hash_code hash_value(const FieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type);
}

// Example type validity checker
LogicalResult TestIntegerType::verifyConstructionInvariants(
    mlir::Location loc, mlir::TestIntegerType::SignednessSemantics ss,
    unsigned int width) {

  if (width > 8)
    return mlir::failure();
  return mlir::success();
}

} // end namespace mlir

#define GET_TYPEDEF_CLASSES
#include "TestTypeDefs.cpp.inc"

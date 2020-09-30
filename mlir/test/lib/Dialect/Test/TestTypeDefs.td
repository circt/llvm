//===-- TestTypeDefs.td - Test dialect type definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TYPEDEFS
#define TEST_TYPEDEFS

// To get the test dialect def
include "TestOps.td"

class Test_Type<string name> : TypeDef<Test_Dialect, name> { }

def SimpleTypeA : Test_Type<"SimpleA"> {
    let mnemonic = "smpla";

    let printer = [{ $_printer << "smpla"; }];
    let parser = [{ return get($_ctxt); }];
}

// A more complex parameterized type
def CompoundTypeA : Test_Type<"CompoundA"> {
    // Override the default mnemonic
    let mnemonic = "cmpnd_a";

    // What types do we contain?
    let parameters = (
        ins
        "int":$widthOfSomething,
        "::mlir::Type":$oneType,
        ArrayRefParameter<"int", "An example of an array of ints">: $arrayOfInts
    );

    let extraClassDeclaration = [{
        struct SomeCppStruct {};
    }];
}

def IntegerType : Test_Type<"TestInteger"> {
  let mnemonic = "int";
  let genVerifyInvariantsDecl = 1;
  let parameters = (
      ins
      "::mlir::TestIntegerType::SignednessSemantics":$signedness,
      "unsigned":$width
  );

  let printer = [{
    $_printer << "int<";
    Print($_printer, getImpl()->signedness);
    $_printer << ", " << getImpl()->width << ">";
  }];

  let parser = [{
    if (parser.parseLess()) return Type();
    SignednessSemantics signedness;
    if (Parse($_parser, signedness)) return mlir::Type();
    if ($_parser.parseComma()) return Type();
    int width;
    if ($_parser.parseInteger(width)) return Type();
    if ($_parser.parseGreater()) return Type();
    return get(ctxt, signedness, width);
  }];

  let extraClassDeclaration = [{
  /// Signedness semantics.
  enum SignednessSemantics {
    Signless, /// No signedness semantics
    Signed,   /// Signed integer
    Unsigned, /// Unsigned integer
  };

  /// This extra function is necessary since it doesn't include signedness
  static IntegerType getChecked(unsigned width, Location location);

  /// Return true if this is a signless integer type.
  bool isSignless() const { return getSignedness() == Signless; }
  /// Return true if this is a signed integer type.
  bool isSigned() const { return getSignedness() == Signed; }
  /// Return true if this is an unsigned integer type.
  bool isUnsigned() const { return getSignedness() == Unsigned; }
    }];
}

class FieldInfo_Type<string name> : Test_Type<name> {
  let parameters = (
      ins
      ArrayRefOfSelfAllocationParameter<"::mlir::FieldInfo", "Models struct fields">: $fields
  );

  let printer = [{
      $_printer << "struct" << "<";
      for (size_t i=0; i<getImpl()->fields.size(); i++) {
      const auto& field = getImpl()->fields[i];
      $_printer << "{" << field.name << "," << field.type << "}";
      if (i < getImpl()->fields.size() - 1)
          $_printer << ",";
      }
      $_printer << ">";
  }];

  let parser = [{
      llvm::SmallVector<FieldInfo, 4> parameters;
      if ($_parser.parseLess()) return Type();
      while (mlir::succeeded($_parser.parseOptionalLBrace())) {
          StringRef name;
          if ($_parser.parseKeyword(&name)) return Type();
          if ($_parser.parseComma()) return Type();
          Type type;
          if ($_parser.parseType(type)) return Type();
          if ($_parser.parseRBrace()) return Type();
          parameters.push_back(FieldInfo {name, type});
          if ($_parser.parseOptionalComma()) break;
      }
      if ($_parser.parseGreater()) return Type();
      return get($_ctxt, parameters);
  }];
}

def StructType : FieldInfo_Type<"Struct"> {
    let mnemonic = "struct";
}


#endif // TEST_TYPEDEFS

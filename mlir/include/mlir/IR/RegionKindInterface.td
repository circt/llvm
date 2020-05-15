//===- RegionKindInterface.td - Region kind interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces to query the properties of regions
// in an operation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_REGIONKINDINTERFACE
#define MLIR_IR_REGIONKINDINTERFACE

include "mlir/IR/OpBase.td"

// OpInterface to query the properties of regions in an operation
def RegionKindInterface : OpInterface<"RegionKindInterface"> {
  let description = [{
    Interface for operations to describe the properties of their regions.
  }];

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{
      Return the types of regions in an operation. By default,
		operations of registered dialects contain regions with CFG
		semantics. This implies that they must satisfy the SSA
		dominance property.  However, other kinds of regions are often
		used which may not require SSA dominance.
      }],
      /*retTy=*/"RegionKind",
      /*methodName=*/"getRegionKind",
      /*args=*/(ins "unsigned":$index)
    >,
    StaticInterfaceMethod<
      /*desc=*/"Return true if the kind of the given region requires the "
		         "SSA-Dominance property",
      /*retTy=*/"bool",
      /*methodName=*/"hasSSADominance",
      /*args=*/(ins "unsigned":$index),
      /*methodBody=*/[{
        return getRegionKind(index) == RegionKind::SSACFG;
      }]
    >,
  ];
}

#endif // MLIR_IR_REGIONKINDINTERFACE

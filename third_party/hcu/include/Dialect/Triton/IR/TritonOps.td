#ifndef THIRDPARTY_HCU_INCLUDE_DIALECT_TRITON_IR_TRITONOPS
#define THIRDPARTY_HCU_INCLUDE_DIALECT_TRITON_IR_TRITONOPS

include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"

//
// MatrixLoad Ops
//

def TT_MatrixLoadOp : TT_Op<"matrix_load", [AttrSizedOperandSegments]> {
    let summary = "Load a matrix from a scalar pointer";

    let arguments = (ins
      Arg<TT_Ptr, "Global memory scalar base pointer to load from", [MemRead<GlobalMemory>]>:$base,
      Variadic<I32>:$shape,
      Variadic<I64>:$strides,
      DenseI32ArrayAttr:$tensorShape,
      Variadic<I32>:$indices,
      DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
      DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
      DefaultValuedAttr<TT_EvictionPolicyAttr, "::mlir::triton::EvictionPolicy::NORMAL">:$evict,
      DefaultValuedAttr<BoolAttr, "false">:$isVolatile
    );

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
      $base `,` `shape` `=` `[` $shape `]` `,` `strides` `=` `[` $strides `]` `,`
      `tensorShape` `=` $tensorShape `,` `indices` `=` `[` $indices `]`
      oilist(
        `boundaryCheck` `=` $boundaryCheck |
        `cacheModifier` `=` $cache |
        `evictionPolicy` `=` $evict
      )
      attr-dict `:` type($base) `->` type($result)
    }];
}

#endif
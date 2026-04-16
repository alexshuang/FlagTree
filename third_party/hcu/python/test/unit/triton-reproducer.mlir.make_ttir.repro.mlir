module {
  tt.func public @triton_() attributes {noinline = false} {
    tt.return
  ^bb1:  // no predecessors
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },triton-rewrite-tensor-pointer,triton-rewrite-tensor-descriptor-to-pointer,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},triton-combine,triton-reorder-broadcast,cse,triton-licm,symbol-dce,triton-loop-unroll))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}

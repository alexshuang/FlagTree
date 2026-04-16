module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx928", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @triton_() attributes {noinline = false} {
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(tritonhcugpu-update-async-wait-count{arch-generation-name=gfx928},optimize-hcu-lds-usage{lds-limit=0 target-arch=gfx928},convert-scf-to-cf,gluon-inline,convert-index-to-llvm{index-bitwidth=0},allocate-hcugpu-shared-memory,convert-triton-hcugpu-to-llvm{arch=gfx928 ftz=true},convert-hcu-distributed-to-llvm{arch=gfx928 ftz=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,convert-cf-to-llvm{index-bitwidth=0},convert-arith-to-llvm{index-bitwidth=0},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,symbol-dce,enable-line-info,convert-builtin-func-to-llvm{ftz=true},convert-lib-device-to-llvm{ftz=true}))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}

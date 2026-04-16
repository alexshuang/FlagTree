module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx928", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @triton_() attributes {noinline = false} {
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(tritongpu-coalesce,tritongpu-F32DotTC{emu-tf32=false},tritongpu-remove-layout-conversions,tritongpu-optimize-thread-locality,tritonhcugpu-accelerate-matmul{arch-generation-name=gfx928 kPack=1 matrix-instruction-size=0 mmac-layout-force=-1},tritongpu-remove-layout-conversions,tritonhcugpu-optimize-dot-operands{arch-generation-name=gfx928},tritonhcugpu-mls-encoding-insertion,tritongpu-remove-layout-conversions,tt.func(tritonhcugpu-hoist-layout-conversions),tritongpu-fuse-nested-loops,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},triton-licm,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},tritonhcugpu-mls-stream-pipeline{async_copy_single_buffer=1 global_prefetch=0 num_stages=2},tritonhcugpu-schedule-loops{num_stages=2},tritonhcugpu-pipeline{use_async_copy=false use_pingpong=false},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},tritongpu-remove-layout-conversions,tritonhcugpu-mls-lowering,tritongpu-reduce-data-duplication,tritonhcugpu-reorder-instructions,tt.func(tritonhcugpu-canonicalize-pointers{enable-large-tensor-ptr-canon=false}),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},tritonhcugpu-convert-buffer-ops{allow-buffer-atomics=true analyze-small-tensor-ofst=false arch-generation-name=gfx928},tritonhcugpu-fold-true-cmpi,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,symbol-dce,tritonhcugpu-update-async-wait-count{arch-generation-name=gfx928}))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}

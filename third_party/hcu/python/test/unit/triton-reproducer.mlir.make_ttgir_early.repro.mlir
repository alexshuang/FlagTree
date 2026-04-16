module {
  tt.func public @triton_() attributes {noinline = false} {
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(convert-triton-to-tritongpu{enable-source-remat=false num-ctas=1 num-warps=4 target=hip:gfx928 threads-per-warp=64}))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}

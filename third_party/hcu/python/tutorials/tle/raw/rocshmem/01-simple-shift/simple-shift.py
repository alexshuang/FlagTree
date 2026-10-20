import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.hcu.utils import (
    enable_rocshmem_device_bc,
    finalize_rocshmem_by_torch_pg,
    init_rocshmem_by_torch_pg,
    init_torch_distributed,
    load_common_host,
    load_host,
    tensor_from_pointer,
)

# The HIP backend must consume get_rocshmem_extern_libs() when compiling a
# dialect with library="rocshmem".
enable_rocshmem_device_bc(True)


@dialect(
    name="hip",
    library="rocshmem",
    file=(Path(__file__).parent / "simple-shift-device.hip"),
    extern_func_name="simple_shift",
    deferred=True,
)
def simple_shift(*args, **kwargs):
    ...


print(f"{type(simple_shift)}, simple_shift.__class__.__mro__:")
print(type(simple_shift), simple_shift.__class__.__mro__)


@triton.jit
def simple_shift_kernel(destination_ptr):
    tle_raw.call(simple_shift, [destination_ptr], output_indices=[0])


def _configure_host_api(host_lib):
    host_lib.simple_shift_before_launch.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    host_lib.simple_shift_before_launch.restype = ctypes.c_int
    host_lib.simple_shift_after_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    host_lib.simple_shift_after_launch.restype = ctypes.c_int


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)

    group = init_torch_distributed()
    host_source = Path(__file__).with_name("simple-shift-host.hip")
    host_lib = load_host(source=host_source)
    _configure_host_api(host_lib)
    common = load_common_host()

    initialized = False
    try:
        init_rocshmem_by_torch_pg(common, group)
        initialized = True

        mype = ctypes.c_int()
        npes = ctypes.c_int()
        mype_in_node = ctypes.c_int()
        npes_in_node = ctypes.c_int()
        stream_ptr = ctypes.c_void_p()
        destination_ptr = ctypes.c_void_p()
        host_data_ptr = ctypes.c_void_p()

        result = host_lib.simple_shift_before_launch(
            ctypes.byref(mype),
            ctypes.byref(npes),
            ctypes.byref(mype_in_node),
            ctypes.byref(npes_in_node),
            ctypes.byref(stream_ptr),
            ctypes.byref(destination_ptr),
            ctypes.byref(host_data_ptr),
        )
        if result != 0:
            raise RuntimeError(f"simple_shift_before_launch failed: {result}")

        destination = tensor_from_pointer(
            destination_ptr,
            (1, ),
            torch.int32,
            device,
        )
        stream = torch.cuda.ExternalStream(stream_ptr.value, device=device)

        with torch.cuda.stream(stream):
            simple_shift_kernel[(1, )](destination, num_warps=1)

        result = host_lib.simple_shift_after_launch(
            stream_ptr,
            destination_ptr,
            host_data_ptr,
            mype.value,
            npes.value,
        )
        if result != 0:
            raise RuntimeError(f"simple_shift_after_launch failed: {result}")
    finally:
        if initialized:
            finalize_rocshmem_by_torch_pg(common, group)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

# Copyright 2026-     Hygon Contributors
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ctypes
import datetime
import functools
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import torch
from triton.experimental.tle.raw.cache_key import compute_tle_raw_host_cache_key
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver
from triton.backends.hcu.compiler import HIPBackend

DUSHMEM_CUMODULE_INIT_HOOK = "dushmem_cumodule_init"

_cumodule_init = None

_dushmem_device_bc_enabled: bool = False


@functools.lru_cache()
def get_dushmem_home() -> Path:
    path = Path(os.environ.get("DUSHMEM_HOME", "/opt/dtk/dushmem"))
    if path:
        if path.is_dir():
            return path

    raise RuntimeError("Cannot resolve libdushmem* under /opt/dtk/dushmem/lib. Set DUSHMEM_HOME "
                       "to the dushmem installation prefix.")


def try_get_dushmem_home() -> Path | None:
    try:
        return get_dushmem_home()
    except RuntimeError:
        return None


def _library_dirs(home: Path) -> tuple[Path, ...]:
    return tuple(path for path in (home / "lib", home / "lib64") if path.is_dir())


def resolve_dushmem_host_library(dushmem_home: Path | None = None) -> Path:
    home = Path(dushmem_home) if dushmem_home is not None else get_dushmem_home()
    for lib_dir in _library_dirs(home):
        for name in ("libdushmem_host.so", "libdushmem.so"):
            path = lib_dir / name
            if path.is_file():
                return path

    raise RuntimeError(f"Cannot find libdushmem_host.so under {home}/lib{{,64}}")


def enable_dushmem_device_bc(enabled: bool = True) -> None:
    global _dushmem_device_bc_enabled
    _dushmem_device_bc_enabled = bool(enabled)


def is_dushmem_device_bc_enabled() -> bool:
    return _dushmem_device_bc_enabled


def _normalize_arch(arch: str | int | None) -> str | None:
    if arch is None:
        return None
    value = str(arch).lower().split(":", 1)[0]
    match = re.search(r"gfx[0-9a-f]+", value)
    return match.group(0) if match else value


def resolve_dushmem_device_bitcode(arch: str) -> Path:
    """Unbundle the device archive for ``arch`` and link the pieces Triton needs.

    ``init_device`` defines ``dushmemi_device_state_d``. ``transfer_device``
    defines the non-inline ``dushmemi_transfer_rma_p`` / ``quiet`` bodies that
    ``dushmem_int_p`` calls when ``DUSHMEM_ENABLE_ALL_DEVICE_INLINING`` is off.
    """
    archive = get_dushmem_home() / "libdushmem_device.a"
    if not archive.is_file():
        raise RuntimeError(f"Cannot find {archive}")
    stamp = archive.stat().st_mtime_ns
    cached = Path(tempfile.gettempdir()) / f"dushmem-device-slim2-{arch}-{stamp}.bc"
    if cached.is_file():
        return cached
    clang_dir = HIPBackend.path_to_rocm_llvm() / "bin"
    bundler = clang_dir / "clang-offload-bundler"
    llvm_ar = clang_dir / "llvm-ar"
    llvm_link = clang_dir / "llvm-link"
    members = ("init_device.cu.o", "transfer_device.cu.o")
    # simple_shift only needs the int put and the quiet helpers the header emits.
    # The rest of transfer_device is every collective; feeding that to dcc
    # produces an assembler error and a multi-minute compile.
    needed = (
        "_Z23dushmemi_transfer_rma_pIiEvPvT_i",
        "_Z23dushmemi_transfer_quietIL13threadgroup_t0EEvb",
        "_Z23dushmemi_transfer_quietIL13threadgroup_t1EEvb",
        "_Z23dushmemi_transfer_quietIL13threadgroup_t2EEvb",
    )
    with tempfile.TemporaryDirectory() as tmp:
        extract = subprocess.run([str(llvm_ar), "x", str(archive), *members], cwd=tmp, capture_output=True, text=True)
        if extract.returncode != 0:
            raise RuntimeError(f"llvm-ar failed to extract {members} from {archive}:\n{extract.stderr}")
        bitcodes: list[str] = []
        for member_name in members:
            member = Path(tmp) / member_name
            extracted = Path(tmp) / f"{member_name}.bc"
            unbundle = subprocess.run(
                [
                    str(bundler),
                    "-type=o",
                    f"-targets=hip-amdgcn-amd-amdhsa--{arch}",
                    f"-input={member}",
                    f"-output={extracted}",
                    "-unbundle",
                ],
                capture_output=True,
                text=True,
            )
            if unbundle.returncode != 0:
                raise RuntimeError(f"clang-offload-bundler failed for {member_name} ({arch}):\n{unbundle.stderr}")
            bitcodes.append(str(extracted))
        init_bc, transfer_bc = bitcodes
        slim = Path(tmp) / "transfer-slim.bc"
        slim_cmd = [str(clang_dir / "llvm-extract"), "--recursive", "-o", str(slim)]
        for func in needed:
            slim_cmd.append(f"--func={func}")
        # llvm-extract leaves string constants used by the put body as
        # declarations. Pull the one this gfx936 build references.
        slim_cmd.append("--glob=.str.3")
        slim_cmd.append(transfer_bc)
        extracted_funcs = subprocess.run(slim_cmd, capture_output=True, text=True)
        if extracted_funcs.returncode != 0:
            raise RuntimeError(f"llvm-extract failed:\n{extracted_funcs.stderr}")
        linked = Path(tmp) / "dushmem-device.bc"
        link = subprocess.run([str(llvm_link), init_bc, str(slim), "-o", str(linked)], capture_output=True, text=True)
        if link.returncode != 0:
            raise RuntimeError(f"llvm-link failed for DUSHMEM device bitcode:\n{link.stderr}")
        cached.write_bytes(linked.read_bytes())
    return cached


# def resolve_dushmem_device_bitcode(
#     dushmem_home: Path | None = None,
#     arch: str | int | None = None,
# ) -> Path | None:
#     home = Path(dushmem_home) if dushmem_home is not None else try_get_dushmem_home()
#     if home is None:
#         return None
#     normalized_arch = _normalize_arch(arch)
#     names = []
#     if normalized_arch:
#         names.append(f"libdushmem_device_{normalized_arch}.bc")
#     names.append("libdushmem_device.bc")
#     for lib_dir in _library_dirs(home):
#         for name in names:
#             path = lib_dir / name
#             if path.is_file():
#                 return path
#     return None


def get_dushmem_extern_libs(arch: str | int | None = None) -> dict[str, str]:
    if not is_dushmem_device_bc_enabled():
        return {}
    bitcode = resolve_dushmem_device_bitcode(arch=arch)
    if bitcode is None:
        raise RuntimeError(f"dushmem device bitcode for arch={arch!r} was requested but not found under "
                           f"{get_dushmem_home()}/lib{{,64}}")
    return {"libdushmem_device": str(bitcode)}


@functools.lru_cache()
def get_hipcc() -> tuple[str, str]:
    rocm_home = HIPBackend.path_to_rocm()
    aicc = str(rocm_home / "bin" / "aicc")
    if aicc.is_file():
        output = subprocess.check_output([aicc, "--version"], stderr=subprocess.STDOUT, text=True)
        match = re.search(r"(?:HIP version|clang version)\s*:?\s*([0-9]+(?:\.[0-9]+)+)", output)
        return aicc, match.group(1) if match else "unknown"
    raise RuntimeError(f"Cannot find {aicc}")


def _offload_arch() -> str:
    target = driver.active.get_current_target()
    if target.backend != "hip":
        raise RuntimeError("TLE-Raw HCU requires a HIP-compatible target, "
                           f"got backend={target.backend!r}")

    arch = str(target.arch).split(":", 1)[0]
    if re.fullmatch(r"gfx[0-9a-f]+", arch, re.IGNORECASE) is None:
        raise RuntimeError(f"invalid HCU architecture {arch!r}; "
                           "expected gfx928/gfx936 or another gfx* name")

    return arch


def _split_env_flags(name: str) -> list[str]:
    return shlex.split(os.environ.get(name, ""))


def _get_cumodule_init():
    global _cumodule_init
    if _cumodule_init is not None:
        return _cumodule_init
    library = ctypes.CDLL(str(resolve_dushmem_host_library()))
    fn = library.dushmemx_cumodule_init
    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    _cumodule_init = fn
    return fn


def initialize_dushmem_cumodule(kernel) -> None:
    kernel._init_handles()
    result = _get_cumodule_init()(ctypes.c_void_p(kernel.module))
    if result != 0:
        raise RuntimeError(f"dushmemx_cumodule_init failed: {result}")


def _compile_hip_host_to_cache(
    source,
    dushmem_home=None,
    arch: str = None,
    force: bool = False,
) -> tuple[Path, str, bool]:
    source_path = Path(source).expanduser().resolve()
    arch = _offload_arch()
    home = Path(dushmem_home).expanduser().resolve() if dushmem_home else get_dushmem_home()
    dushmem_library = resolve_dushmem_host_library(home)
    hipcc, hip_version = get_hipcc()

    fingerprint = (f"{arch}-hip{hip_version}-dushmem-"
                   f"{dushmem_library.stat().st_size}-{dushmem_library.stat().st_mtime_ns}")
    host_cache_key = compute_tle_raw_host_cache_key(source_path, fingerprint)
    output_name = source_path.with_suffix(".so").name
    cache = get_cache_manager(host_cache_key)
    cached = None if force else cache.get_file(output_name)
    if cached is not None:
        return Path(cached), host_cache_key, True

    # rocm_home = Path(HIPBackend.path_to_rocm())
    # include_flags = [f"-I{home / 'include'}"]
    # if (rocm_home / "include").is_dir() and rocm_home != home:
    #     include_flags.append(f"-I{rocm_home / 'include'}")

    # include_flags.append("-I/opt/mpi/include")

    with tempfile.TemporaryDirectory(prefix="tle-dushmem-host-") as temporary_dir:
        temporary_dir = Path(temporary_dir)
        shared_path = temporary_dir / output_name
        command = [
            hipcc,
            "-shared",
            "-fPIC",
            "-fgpu-rdc",
            f"--offload-arch={arch}",
            "-O3",
            "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
            "-mcode-object-version=4",
            "-I/opt/dtk/include",
            "-I/opt/dtk/include/dushmem",
            "-L/opt/dtk/lib/dushmem",
            str(source),
            "-o",
            str(shared_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"hipcc failed during dushmem host-wrapper\n"
                               f"command: {shlex.join(command)}\n"
                               f"stdout:\n{result.stdout}\n"
                               f"stderr:\n{result.stderr}")
        cached_path = cache.put(shared_path.read_bytes(), output_name, binary=True)
    return Path(cached_path), host_cache_key, False


_COMMON_SIGNATURES = {
    "dushmem_unique_id_size": ([], ctypes.c_size_t),
    "dushmem_get_unique_id_bytes": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
    "dushmem_init_from_torch_distributed": (
        [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
        ctypes.c_int,
    ),
    "dushmem_finalize_from_torch_distributed": ([], ctypes.c_int),
    "tle_hip_stream_write_value64": (
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64],
        ctypes.c_int,
    ),
    "tle_hip_memcpy_async": (
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
        ctypes.c_int,
    ),
    "tle_hip_error_string": ([ctypes.c_int], ctypes.c_char_p),
}


class HipHostLibrary:

    def __init__(self, library_path):
        self.path = Path(library_path).expanduser().resolve()
        self.library = ctypes.CDLL(str(self.path))
        self._configure_signatures()

    def _configure_signatures(self):
        for name, (argtypes, restype) in _COMMON_SIGNATURES.items():
            try:
                fn = getattr(self.library, name)
            except AttributeError:
                continue
            fn.argtypes = argtypes
            fn.restype = restype

    def __getattr__(self, name):
        return getattr(self.library, name)


def get_common_host_source() -> Path:
    return Path(__file__).resolve().parent / "common-host.hip"


def load_host(source, dushmem_home=None, arch=None, force: bool = False) -> HipHostLibrary:
    path, _, _ = _compile_hip_host_to_cache(source, dushmem_home, arch, force=force)
    return HipHostLibrary(path)


def load_common_host(source=None, dushmem_home=None, arch=None, force: bool = False) -> HipHostLibrary:
    return load_host(source or get_common_host_source(), dushmem_home, arch, force)


def init_torch_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", local_rank),
        timeout=datetime.timedelta(seconds=1800),
    )
    group = torch.distributed.new_group(ranks=list(range(world_size)), backend="nccl")
    torch.distributed.barrier(group=group)
    return group


def init_dushmem_by_torch_pg(common, group):
    rank = group.rank()
    world_size = group.size()
    uid_size = int(common.dushmem_unique_id_size())
    if uid_size <= 0:
        raise RuntimeError(f"Invalid dushmem unique-id size: {uid_size}")

    if rank == 0:
        temporary_buffer = ctypes.create_string_buffer(uid_size)
        result = common.dushmem_get_unique_id_bytes(temporary_buffer, uid_size)
        if result != 0:
            raise RuntimeError(f"dushmem_get_uniqueid failed: {result}")
        uid = bytes(temporary_buffer.raw)
    else:
        uid = bytes(uid_size)

    objects = [uid]
    torch.distributed.broadcast_object_list(
        objects,
        src=torch.distributed.get_global_rank(group, 0),
        group=group,
    )
    uid_buffer = ctypes.create_string_buffer(objects[0], uid_size)
    result = common.dushmem_init_from_torch_distributed(
        rank,
        world_size,
        int(os.environ["LOCAL_RANK"]),
        uid_buffer,
        uid_size,
    )
    if result != 0:
        raise RuntimeError(f"dushmem initialization failed: {result}")
    torch.distributed.barrier(group=group)


def finalize_dushmem_by_torch_pg(common, group=None):
    if group is not None:
        torch.distributed.barrier(group=group)
    result = common.dushmem_finalize_from_torch_distributed()
    if result != 0:
        raise RuntimeError(f"dushmem finalization failed: {result}")


def tensor_from_pointer(
    pointer: int | ctypes.c_void_p,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a non-owning Torch tensor view over a HIP device allocation."""
    address = pointer.value if isinstance(pointer, ctypes.c_void_p) else pointer
    if address is not None and not isinstance(address, int):
        raise TypeError(f"pointer must be int or ctypes.c_void_p, got {type(pointer).__name__}")
    if not address:
        raise ValueError("pointer cannot be null; HIP memory must be allocated")
    elements = 1
    for extent in shape:
        elements *= extent
    storage = torch._C._construct_storage_from_data_pointer(
        address,
        device,
        elements * dtype.itemsize,
    )
    return torch.empty(0, dtype=dtype, device=device).set_(storage).view(shape)


@functools.lru_cache()
def _runtime_host() -> HipHostLibrary:
    return load_common_host()


def _void_pointer(value) -> ctypes.c_void_p:
    if isinstance(value, ctypes.c_void_p):
        return value
    return ctypes.c_void_p(int(value))


def _hip_check(common, result: int, operation: str) -> None:
    if result == 0:
        return
    message = common.tle_hip_error_string(result)
    detail = message.decode("utf-8", errors="replace") if message else "unknown HIP error"
    raise RuntimeError(f"{operation} failed: {detail} ({result})")


def set_signal_hip_ptr(signal_ptr, signal, stream):
    common = _runtime_host()
    result = common.tle_hip_stream_write_value64(
        _void_pointer(stream.cuda_stream),
        _void_pointer(signal_ptr),
        ctypes.c_uint64(signal),
    )
    _hip_check(common, result, "hipStreamWriteValue64")


def copy_on_stream(dst_ptr, src_ptr, nbytes, stream):
    common = _runtime_host()
    result = common.tle_hip_memcpy_async(
        _void_pointer(dst_ptr),
        _void_pointer(src_ptr),
        nbytes,
        _void_pointer(stream.cuda_stream),
    )
    _hip_check(common, result, "hipMemcpyAsync")


def print_perf(
    name: str,
    value: float,
    group,
    rank: int,
    world_size: int,
    unit: str = "ms",
):
    for index in range(world_size):
        torch.distributed.barrier(group=group)
        if rank == index:
            print(f"{name} #{rank}: {value:.3f} {unit}", flush=True)


def print_perf_mean(
    name: str,
    value: float,
    group,
    rank: int,
    world_size: int,
    unit: str = "ms",
):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    value_tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    torch.distributed.all_reduce(
        value_tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )
    mean_value = (value_tensor / world_size).item()
    if rank == 0:
        print(f"{name} mean: {mean_value:.3f} {unit}", flush=True)


# Keep the old function name for callers shared with the CUDA implementation.
set_signal_cuda_ptr = set_signal_hip_ptr

CudaHostLibrary = HipHostLibrary

# Source-compatible aliases for existing NVSHMEM tutorial Python code.
get_nvshmem_home = get_dushmem_home
try_get_nvshmem_home = try_get_dushmem_home
resolve_nvshmem_host_library = resolve_dushmem_host_library
enable_nvshmem_device_bc = enable_dushmem_device_bc
is_nvshmem_device_bc_enabled = is_dushmem_device_bc_enabled
resolve_nvshmem_device_bitcode = resolve_dushmem_device_bitcode
get_nvshmem_extern_libs = get_dushmem_extern_libs
init_nvshmem_by_torch_pg = init_dushmem_by_torch_pg
finalize_nvshmem_by_torch_pg = finalize_dushmem_by_torch_pg

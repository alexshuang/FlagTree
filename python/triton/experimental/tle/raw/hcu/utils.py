# Copyright 2026      Hygon Contributors
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
import warnings
from pathlib import Path

import torch
from triton.experimental.tle.raw.cache_key import compute_tle_raw_host_cache_key
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver
from triton.backends.hcu.compiler import HIPBackend


@functools.lru_cache()
def get_rocshmem_home() -> Path:
    candidates = [
        os.environ.get("ROCSHMEM_HOME"),
        os.environ.get("ROCSHMEM_INSTALL_DIR"),
    ]
    try:
        from triton import knobs
        _knobs = getattr(knobs, "hcu", None)
        if _knobs:
            value = getattr(_knobs, "rocshmem_home", None)
            if value:
                candidates.insert(0, value)
    except (AttributeError, ImportError):
        pass

    candidates.extend(["/opt/rocshmem"])
    for candidate in candidates:
        if not candidate:
            continue
        home = Path(candidate).expanduser().resolve()
        if any((home / include).is_file() for include in (
                "include/rocshmem/rocshmem.hpp",
                "include/rocshmem.hpp",
        )):
            return home
    raise RuntimeError("Cannot resolve ROCSHMEM_HOME. Set ROCSHMEM_HOME or "
                       "ROCSHMEM_INSTALL_DIR to the rocSHMEM installation prefix.")


def try_get_rocshmem_home() -> Path | None:
    try:
        return get_rocshmem_home()
    except RuntimeError:
        return None


def _library_dirs(home: Path) -> tuple[Path, ...]:
    return tuple(path for path in (home / "lib", home / "lib64") if path.is_dir())


def resolve_rocshmem_library(rocshmem_home: Path | None = None) -> Path:
    home = Path(rocshmem_home) if rocshmem_home is not None else get_rocshmem_home()
    for lib_dir in _library_dirs(home):
        for name in ("librocshmem_host.so", "librocshmem.so"):
            path = lib_dir / name
            if path.is_file():
                return path

    for lib_dir in _library_dirs(home):
        path = lib_dir / "librocshmem.a"
        if path.is_file():
            warnings.warn(
                f"Only the static {path} was found. Each host wrapper .so will embed its own "
                "rocSHMEM runtime, so rocshmem_init() in one wrapper is invisible to the others.",
                RuntimeWarning,
                stacklevel=2,
            )
            return path

    raise RuntimeError(f"Cannot find librocshmem_host.so, librocshmem.so or librocshmem.a under {home}/lib{{,64}}")


_rocshmem_device_bc_enabled: bool = False


def enable_rocshmem_device_bc(enabled: bool = True) -> None:
    global _rocshmem_device_bc_enabled
    _rocshmem_device_bc_enabled = bool(enabled)


def is_rocshmem_device_bc_enabled() -> bool:
    return _rocshmem_device_bc_enabled


def _normalize_arch(arch: str | int | None) -> str | None:
    if arch is None:
        return None
    value = str(arch).lower().split(":", 1)[0]
    match = re.search(r"gfx[0-9a-f]+", value)
    return match.group(0) if match else value


def resolve_rocshmem_device_bitcode(
    rocshmem_home: Path | None = None,
    arch: str | int | None = None,
) -> Path | None:
    home = Path(rocshmem_home) if rocshmem_home is not None else try_get_rocshmem_home()
    if home is None:
        return None
    normalized_arch = _normalize_arch(arch)
    names = []
    if normalized_arch:
        names.append(f"librocshmem_device_{normalized_arch}.bc")
    names.append("librocshmem_device.bc")
    for lib_dir in _library_dirs(home):
        for name in names:
            path = lib_dir / name
            if path.is_file():
                return path
    return None


def get_rocshmem_extern_libs(arch: str | int | None = None) -> dict[str, str]:
    if not is_rocshmem_device_bc_enabled():
        return {}
    bitcode = resolve_rocshmem_device_bitcode(arch=arch)
    if bitcode is None:
        raise RuntimeError(f"rocSHMEM device bitcode for arch={arch!r} was requested but not found under "
                           f"{get_rocshmem_home()}/lib{{,64}}")
    return {"librocshmem_device": str(bitcode)}


@functools.lru_cache()
def get_hipcc() -> tuple[str, str]:
    # executable = "hipcc" + (sysconfig.get_config_var("EXE") or "")
    executable = "aicc"
    rocm_home = Path(os.environ.get("ROCM_PATH", HIPBackend.path_to_rocm()))
    candidates = (str(rocm_home / "bin" / executable), )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_file():
            continue
        output = subprocess.check_output([str(path), "--version"], stderr=subprocess.STDOUT, text=True)
        match = re.search(r"(?:HIP version|clang version)\s*:?\s*([0-9]+(?:\.[0-9]+)+)", output)
        return str(path), match.group(1) if match else "unknown"
    raise RuntimeError("Cannot find hipcc. Set ROCM_PATH, or install it under $ROCM_PATH/bin.")


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


def _compile_hip_host_to_cache(
    source,
    rocshmem_home=None,
    arch: str = None,
    force: bool = False,
) -> tuple[Path, str, bool]:
    source_path = Path(source).expanduser().resolve()
    # arch = _normalize_arch(arch) if arch else _get_amdgpu_arch()
    arch = _offload_arch()
    home = Path(rocshmem_home).expanduser().resolve() if rocshmem_home else get_rocshmem_home()
    rocshmem_library = resolve_rocshmem_library(home)
    hipcc, hip_version = get_hipcc()

    fingerprint = (f"{arch}-hip{hip_version}-rocshmem-"
                   f"{rocshmem_library.stat().st_size}-{rocshmem_library.stat().st_mtime_ns}")
    host_cache_key = compute_tle_raw_host_cache_key(source_path, fingerprint)
    output_name = source_path.with_suffix(".so").name
    cache = get_cache_manager(host_cache_key)
    cached = None if force else cache.get_file(output_name)
    if cached is not None:
        return Path(cached), host_cache_key, True

    # rocm_home = Path(os.environ.get("ROCM_PATH", os.environ.get("HIP_PATH", "/opt/rocm")))
    rocm_home = Path(HIPBackend.path_to_rocm())
    include_flags = [f"-I{home / 'include'}"]
    if (rocm_home / "include").is_dir() and rocm_home != home:
        include_flags.append(f"-I{rocm_home / 'include'}")

    include_flags.append("-I/opt/mpi/include")

    with tempfile.TemporaryDirectory(prefix="tle-rocshmem-host-") as temporary_dir:
        temporary_dir = Path(temporary_dir)
        object_path = temporary_dir / f"{source_path.stem}.o"
        shared_path = temporary_dir / output_name
        compile_command = [
            hipcc,
            "-c",
            "-x",
            "hip",
            "-O2",
            "-fPIC",
            "-fgpu-rdc",
            f"--offload-arch={arch}",
            *include_flags,
            *_split_env_flags("ROCSHMEM_EXTRA_CXXFLAGS"),
            "-o",
            str(object_path),
            str(source_path),
        ]
        link_command = [
            hipcc,
            "-shared",
            "-fgpu-rdc",
            "--hip-link",
            f"--offload-arch={arch}",
            str(object_path),
            str(rocshmem_library),
            f"-L{rocm_home / 'lib'}",
            f"-L{rocm_home / 'lib64'}",
            f"-L{home / 'lib'}",
            "-lrocshmem_device",
            "-lamdhip64",
            "-lhsa-runtime64",
            *_split_env_flags("ROCSHMEM_EXTRA_LDFLAGS"),
            f"-Wl,-rpath,{rocshmem_library.parent}",
            "-o",
            str(shared_path),
        ]
        for phase, command in (("compile", compile_command), ("link", link_command)):
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"hipcc failed during rocSHMEM host-wrapper {phase}\n"
                                   f"command: {shlex.join(command)}\n"
                                   f"stdout:\n{result.stdout}\n"
                                   f"stderr:\n{result.stderr}")
        cached_path = cache.put(shared_path.read_bytes(), output_name, binary=True)
    return Path(cached_path), host_cache_key, False


_COMMON_SIGNATURES = {
    "rocshmem_unique_id_size": ([], ctypes.c_size_t),
    "rocshmem_get_unique_id_bytes": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
    "rocshmem_init_from_torch_distributed": (
        [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
        ctypes.c_int,
    ),
    "rocshmem_finalize_from_torch_distributed": ([], ctypes.c_int),
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


# Compatibility alias for callers that only use the wrapper protocol.
CudaHostLibrary = HipHostLibrary


def get_common_host_source() -> Path:
    return Path(__file__).resolve().parent / "common-host.hip"


def load_host(source, rocshmem_home=None, arch=None, force: bool = False) -> HipHostLibrary:
    path, _, _ = _compile_hip_host_to_cache(source, rocshmem_home, arch, force=force)
    return HipHostLibrary(path)


def load_common_host(source=None, rocshmem_home=None, arch=None, force: bool = False) -> HipHostLibrary:
    return load_host(source or get_common_host_source(), rocshmem_home, arch, force)


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


def init_rocshmem_by_torch_pg(common, group):
    rank = group.rank()
    world_size = group.size()
    uid_size = int(common.rocshmem_unique_id_size())
    if uid_size <= 0:
        raise RuntimeError(f"Invalid rocSHMEM unique-id size: {uid_size}")

    if rank == 0:
        temporary_buffer = ctypes.create_string_buffer(uid_size)
        result = common.rocshmem_get_unique_id_bytes(temporary_buffer, uid_size)
        if result != 0:
            raise RuntimeError(f"rocshmem_get_uniqueid failed: {result}")
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
    result = common.rocshmem_init_from_torch_distributed(
        rank,
        world_size,
        int(os.environ["LOCAL_RANK"]),
        uid_buffer,
        uid_size,
    )
    if result != 0:
        raise RuntimeError(f"rocSHMEM initialization failed: {result}")
    torch.distributed.barrier(group=group)
    print(f"## rank {rank} run rocshmem_init_from_torch_distributed() done.")


def finalize_rocshmem_by_torch_pg(common, group=None):
    if group is not None:
        torch.distributed.barrier(group=group)
    result = common.rocshmem_finalize_from_torch_distributed()
    if result != 0:
        raise RuntimeError(f"rocSHMEM finalization failed: {result}")


# Source-compatible aliases for existing NVSHMEM tutorial Python code.
get_nvshmem_home = get_rocshmem_home
try_get_nvshmem_home = try_get_rocshmem_home
resolve_nvshmem_host_library = resolve_rocshmem_library
enable_nvshmem_device_bc = enable_rocshmem_device_bc
is_nvshmem_device_bc_enabled = is_rocshmem_device_bc_enabled
resolve_nvshmem_device_bitcode = resolve_rocshmem_device_bitcode
get_nvshmem_extern_libs = get_rocshmem_extern_libs
init_nvshmem_by_torch_pg = init_rocshmem_by_torch_pg
finalize_nvshmem_by_torch_pg = finalize_rocshmem_by_torch_pg


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


# Keep the old function name for callers shared with the CUDA implementation.
set_signal_cuda_ptr = set_signal_hip_ptr


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

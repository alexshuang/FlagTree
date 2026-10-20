from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import re
import sys
import json
import shlex
import subprocess

from triton._C.libtriton import hcu, llvm
from triton._C.libtriton.tle.llvm import parse_llvm_ir
from triton.runtime.driver import driver
from triton.backends.hcu.compiler import HIPBackend
from triton.experimental.tle.raw.runtime import RawJITFunction
from triton.experimental.tle.raw.source_store import register_source


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


def _sanitize_clang_ir(text: str) -> tuple[str, list[str]]:
    """Handle only known textual LLVM compatibility differences."""

    applied: list[str] = []

    # Keep this list intentionally small. In particular, never rewrite target
    # triple, data layout, calling convention or pointer address spaces.
    for attribute in (
            " nocreateundeforpoison",
            " contract",
    ):
        text, count = re.subn(
            re.escape(attribute),
            "",
            text,
        )
        if count:
            applied.append(f"dropped {count}x {attribute.strip()!r}")

    # Do not import private per-function subtarget feature set.
    patterns = (
        (
            r'\s+"target-cpu"="[^"]*"',
            "target-cpu",
        ),
        (
            r'\s+"target-features"="[^"]*"',
            "target-features",
        ),
    )

    for pattern, name in patterns:
        text, count = re.subn(
            pattern,
            "",
            text,
        )
        if count:
            applied.append(f"dropped {count}x {name}")

    return text, applied


def _extract_target_triple(text: str) -> str | None:
    match = re.search(
        r'^target triple\s*=\s*"([^"]+)"',
        text,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def _rocshmem_include_flags() -> list[str]:
    # Import lazily: utils.py also imports HIPBackend and host-side dependencies.
    from triton.experimental.tle.raw.hcu.utils import get_rocshmem_home
    return [
        f"-I{get_rocshmem_home() / 'include'}",
        "-I/opt/mpi/include",
        "-I/opt/ucx/include",
    ]


def _library_include_flags(library: str | None) -> list[str]:
    if library in (None, ""):
        return []
    if library == "rocshmem":
        return _rocshmem_include_flags()
    raise RuntimeError("TLE-Raw HCU supports only library=None or library='rocshmem'; "
                       f"got {library!r}")


_AST_SCAN_MACROS: Final[list[str]] = [
    "-D__device__=__attribute__((device))",
    "-D__host__=__attribute__((host))",
    "-D__global__=__attribute__((global))",
    "-D__shared__=__attribute__((shared))",
    "-D__constant__=__attribute__((constant))",
    "-D__forceinline__=inline __attribute__((always_inline))",
]

_STDIN_NAMES: Final[frozenset[str]] = frozenset({"<stdin>", "-"})


def _collect_device_functions(root: dict) -> list[str]:
    """Return names of __device__ functions with a body, defined in stdin."""
    # Clang's JSON dumper only emits "file" when it changes, so track it
    # in document order to know which file each node belongs to.
    state: dict[str, Any] = {"file": None}
    found: list[str] = []

    def track(loc: Any) -> None:
        if not isinstance(loc, dict):
            return
        if "spellingLoc" in loc or "expansionLoc" in loc:
            track(loc.get("spellingLoc"))
            track(loc.get("expansionLoc"))
        elif "file" in loc:
            state["file"] = loc["file"]

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        track(node.get("loc"))
        loc_file = state["file"]
        rng = node.get("range") or {}
        track(rng.get("begin"))
        track(rng.get("end"))

        inner = node.get("inner") or []
        if (node.get("kind") == "FunctionDecl" and not node.get("isImplicit") and node.get("name")
                and loc_file in _STDIN_NAMES):
            kinds = {c.get("kind") for c in inner if isinstance(c, dict)}
            if "CUDADeviceAttr" in kinds and "CompoundStmt" in kinds:
                found.append(node["name"])

        for child in inner:
            visit(child)

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10000))
    try:
        visit(root)
    finally:
        sys.setrecursionlimit(old_limit)

    return list(dict.fromkeys(found))


def _infer_device_function_name(source: str, source_file: str) -> str:
    clang = HIPBackend.path_to_rocm_clang()
    arch = _offload_arch()

    command = [
        str(clang),
        "-x",
        "hip",
        "--offload-device-only",
        f"--offload-arch={arch}",
        "-nogpulib",
        "-nogpuinc",
        "-fsyntax-only",
        "-Wno-everything",
        "-ferror-limit=0",
        *_AST_SCAN_MACROS,
        "-Xclang",
        "-ast-dump=json",
        "-",
    ]

    result = subprocess.run(
        command,
        input=source.encode("utf-8"),
        capture_output=True,
        check=False,
    )

    stdout = result.stdout.decode("utf-8", errors="replace")
    start = stdout.find("{")
    if start < 0:
        raise RuntimeError("TLE-Raw HCU AST scan produced no output\n"
                           f"source: {source_file}\n"
                           f"command: {shlex.join(command)}\n"
                           f"stderr:\n{result.stderr.decode(errors='replace')}")

    try:
        root, _ = json.JSONDecoder().raw_decode(stdout[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError("TLE-Raw HCU failed to parse Clang AST JSON\n"
                           f"source: {source_file}") from exc

    names = _collect_device_functions(root)

    if len(names) == 1:
        return names[0]

    if not names:
        raise RuntimeError("TLE-Raw HCU could not find a __device__ "
                           f"function with a body in {source_file}. "
                           "Specify extern_func_name= explicitly.")

    raise RuntimeError("TLE-Raw HCU found multiple __device__ "
                       f"functions in {source_file}: {names}. "
                       "Specify extern_func_name= explicitly.")


def _find_lowered_function_symbol(
    llvm_ir: str,
    source_name: str,
) -> str:
    definitions = re.findall(
        r"""
        ^\s*define\b
        [^{@\n]*
        @(?:
            "([^"]+)"
            |
            ([^\s(]+)
        )
        \s*\(
        """,
        llvm_ir,
        re.MULTILINE | re.VERBOSE,
    )

    names = [quoted or plain for quoted, plain in definitions]

    if source_name in names:
        return source_name

    itanium_prefix = (f"_Z{len(source_name)}{source_name}")

    candidates = [name for name in names if name.startswith(itanium_prefix)]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError("TLE-Raw HCU cannot uniquely resolve "
                       f"{source_name!r} to an LLVM symbol; "
                       f"definitions={names}, "
                       f"candidates={candidates}")


def _resolve_llvm_tool(name: str) -> Path:
    llvm_root = HIPBackend.path_to_rocm_llvm()
    tool = llvm_root / "bin" / name

    if not tool.is_file():
        raise RuntimeError(f"cannot find {name}: {tool}")

    return tool


def _extract_function_closure(
    llvm_ir: str,
    symbol: str,
) -> str:
    llvm_extract = _resolve_llvm_tool("llvm-extract")

    command = [
        str(llvm_extract),
        f"--func={symbol}",
        "--recursive",
        "-S",
        "-",
        "-o",
        "-",
    ]

    result = subprocess.run(
        command,
        input=llvm_ir.encode("utf-8"),
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError("TLE-Raw HCU llvm-extract failed\n"
                           f"symbol: {symbol}\n"
                           f"command: {shlex.join(command)}\n"
                           f"stderr:\n"
                           f"{result.stderr.decode(errors='replace')}")

    extracted = result.stdout.decode(
        "utf-8",
        errors="strict",
    )

    if not re.search(
            rf"^\s*define\b[^\n@]*"
            rf"@{re.escape(symbol)}\s*\(",
            extracted,
            re.MULTILINE,
    ):
        raise RuntimeError("llvm-extract output does not contain "
                           f"the requested function {symbol!r}")

    return extracted


def _link_rocshmem_device_bitcode(llvm_ir: str, arch: str, root_symbol: str) -> str:
    # Reuse the resolver in hcu/utils.py; do not duplicate installation policy.
    from triton.experimental.tle.raw.hcu.utils import (
        get_rocshmem_home,
        resolve_rocshmem_device_bitcode,
    )

    bitcode = resolve_rocshmem_device_bitcode(arch=arch)
    if bitcode is None:
        raise RuntimeError(f"rocSHMEM device bitcode for arch={arch!r} was not found under "
                           f"{get_rocshmem_home()}/lib{{,64}}")

    llvm_link = _resolve_llvm_tool("llvm-link")
    command = [
        str(llvm_link),
        "-S",
        "-",
        str(bitcode),
        "-o",
        "-",
    ]
    result = subprocess.run(
        command,
        input=llvm_ir.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("TLE-Raw HCU failed to link rocSHMEM device bitcode\n"
                           f"architecture: {arch}\n"
                           f"bitcode: {bitcode}\n"
                           f"command: {shlex.join(command)}\n"
                           f"stderr:\n{result.stderr.decode(errors='replace')}")

    linked = result.stdout.decode("utf-8", errors="strict")
    # Keep the user function, its transitive rocSHMEM callees, and referenced
    # device globals while avoiding import of unrelated rocSHMEM entry points.
    return _extract_function_closure(linked, root_symbol)


def _clang_flags(library: str | None = None) -> list[str]:
    return [
        "-fgpu-rdc",
        "-nogpulib",
        "-Wno-unused-command-line-argument",
        "-include",
        "hip/hip_runtime.h",
        f"-I/{HIPBackend.path_to_rocm()}/cuda/cuda/include",
        *_library_include_flags(library),
    ]


def _lower_read_register_exec(text):
    _MD_DEF_RE = re.compile(r'^(![\w.]+)\s*=\s*!\{!"([^"]*)"\}\s*$', re.M)
    _READ_REG_RE = re.compile(r'(?m)^(?P<lhs>\s*%[^\s=]+\s*=\s*)'
                              r'(?:(?:tail|musttail|notail)\s+)?call\s+i64\s+'
                              r'@llvm\.read_register\.i64\(metadata\s+(?P<md>![\w.]+)\)[^\n]*$')
    _READ_REG_DECL_RE = re.compile(r'(?m)^declare\s+i64\s+@llvm\.read_register\.i64\(metadata\)[^\n]*\n')
    _BALLOT_DECL = "declare i64 @llvm.amdgcn.ballot.i64(i1)"
    _METADATA_ARG_RE = re.compile(r'(?m)^(?!declare\b)[^;\n]*\bcall\b[^\n]*\bmetadata\s+![\w.]+')

    rewrites = []
    md_names = {k: v for k, v in _MD_DEF_RE.findall(text)}
    count = 0

    def repl(m):
        nonlocal count
        reg = md_names.get(m.group("md"))
        if reg != "exec":
            raise RuntimeError(f"unsupported llvm.read_register({reg!r}); only 'exec' can be lowered")
        count += 1
        return f'{m.group("lhs")}call i64 @llvm.amdgcn.ballot.i64(i1 true)'

    text = _READ_REG_RE.sub(repl, text)
    if count:
        rewrites.append(f"read_register(exec)->amdgcn.ballot x{count}")
        text = _READ_REG_DECL_RE.sub("", text)
        if "@llvm.amdgcn.ballot.i64(" in text and \
           "declare i64 @llvm.amdgcn.ballot.i64" not in text:
            text = text.rstrip("\n") + "\n" + _BALLOT_DECL + "\n"

    leftover = _METADATA_ARG_RE.findall(text)
    if leftover:
        raise RuntimeError("LLVM IR still has call with metadata operand that MLIR ModuleImport "
                           "cannot import:\n" + "\n".join(l.strip() for l in leftover[:10]))
    return text, rewrites


class HCUJITFunction(RawJITFunction):
    """Compile CUDA/HIP-compatible device source into an eager raw region."""

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(fn, **kwargs)

        self.code: Final[str] = file.read_text()
        self.source_file: Final[str] = str(file)

        # The frontend decorator may say name=\"cuda\", but downstream IR must
        # describe the toolchain that actually compiled the source.
        self.region_dialect: Final[str] = "hcu"
        self.lowered_region_dialect: Final[str] = "llvm"
        self.arg_dialect: Final[str] = "llvm"

        self.opt_level: Final[str] = kwargs.get("opt_level", "-O3")

        # if self.library:
        #     raise RuntimeError(f"TLE-Raw HCU does not support "
        #                        f"library={self.library!r}")
        _library_include_flags(self.library)

        if not self.extern_func_name:
            self.extern_func_name = (_infer_device_function_name(
                self.code,
                self.source_file,
                self.library,
            ))

    def make_llvm(self, mlir_context) -> str:
        clang = HIPBackend.path_to_rocm_clang()
        arch = _offload_arch()

        command = [
            str(clang),
            "-x",
            "hip",
            "--offload-device-only",
            f"--offload-arch={arch}",
            self.opt_level,
            "-emit-llvm",
            "-S",
            "-",
            "-o",
            "-",
            *_clang_flags(self.library),
        ]

        result = subprocess.run(
            command,
            input=self.code.encode(),
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError("TLE-Raw HCU source compilation failed\n"
                               f"source: {self.source_file}\n"
                               f"architecture: {arch}\n"
                               f"command: {shlex.join(command)}\n"
                               f"stderr:\n"
                               f"{result.stderr.decode(errors='replace')}")

        text = result.stdout.decode(errors="strict")
        text, rewrites = _sanitize_clang_ir(text)

        emitted_triple = _extract_target_triple(text)
        expected_triple = str(hcu.TARGET_TRIPLE)

        if (emitted_triple is not None and emitted_triple != expected_triple):
            raise RuntimeError("HCU Clang emitted an incompatible LLVM target triple: "
                               f"{emitted_triple!r}; expected {expected_triple!r}. "
                               "Use the same DTK compiler family as compiler.py.")

        lowered_func_name = (_find_lowered_function_symbol(
            text,
            self.extern_func_name,
        ))

        text = _extract_function_closure(
            text,
            lowered_func_name,
        )

        with open("/tmp/before_link_rocshmem.mlir", "w") as fp:
            fp.write(text)

        # # Re-run normalization on the final closure
        # text, rewrites = _sanitize_clang_ir(text)

        if self.library == "rocshmem":
            text = _link_rocshmem_device_bitcode(
                text,
                arch,
                lowered_func_name,
            )

        with open("/tmp/after_link_rocshmem.mlir", "w") as fp:
            fp.write(text)

        # Normalize the final closure too, because linked device bitcode can
        # carry the same attributes as Clang's source module.
        text, final_rewrites = _sanitize_clang_ir(text)
        text, rr = _lower_read_register_exec(text)
        final_rewrites.extend(rr)
        rewrites.extend(final_rewrites)

        with open("/tmp/after_sanitize.mlir", "w") as fp:
            fp.write(text)

        # C++ findExternalLLVMFunc() 必须使用 LLVM symbol，
        # 不能继续使用 source-level 的 kernel function name。
        self.extern_func_name = lowered_func_name

        # Validate that FlagTree's embedded LLVM and MLIR LLVM importer can
        # consume the textual IR. create_region_by_llvm() will import it again
        # while constructing tle.dsl_region.
        llvm_context = llvm.context()
        try:
            mlir_module = parse_llvm_ir(text, llvm_context, mlir_context)
        except Exception as exc:
            applied = (", ".join(rewrites) if rewrites else "none")
            raise RuntimeError("Failed to import HCU clang LLVM IR into "
                               "MLIR LLVM dialect\n"
                               f"source: {self.source_file}\n"
                               f"rewrites: {applied}\n"
                               f"LLVM IR:\n{mlir_module}") from exc

        return str(mlir_module)

    def register_pending_source(self, *, hint: str = "") -> str:
        if not self.extern_func_name:
            raise RuntimeError("Deferred TLE-Raw HCU requires "
                               "extern_func_name= to identify the device "
                               "function in the HIP/CUDA source")

        return register_source(
            region_dialect=self.region_dialect,
            extern_func_name=self.extern_func_name,
            source=self.code,
            hint=hint,
            extra={
                "source_file": self.source_file,
                "opt_level": self.opt_level,
                "library": self.library,
            },
        )

    def create_region_deferred(
        self,
        builder,
        source_id: str,
        handles,
        alias_indices,
        hint: str = "",
    ):
        return builder.create_tle_raw_region_deferred(
            source_id,
            self.region_dialect,
            self.arg_dialect,
            handles,
            alias_indices,
            hint,
        )


def compile_deferred_pending_source(entry: dict, *, context) -> str:
    source_text = entry["source"]

    class _HCUSourceFile:

        def read_text(self):
            return source_text

    hcu_fn = HCUJITFunction(
        fn=None,
        file=_HCUSourceFile(),
        extern_func_name=entry.get("extern_func_name"),
        deferred=True,
        opt_level=entry.get("opt_level"),
        library=entry.get("library"),
    )
    return hcu_fn.make_llvm(context)

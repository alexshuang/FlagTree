from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from triton._C.libtriton import hcu
from triton.experimental.tle.raw.hcu.runtime import take_link_sources
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
)
from triton.backends.hcu.compiler import HIPBackend


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compile pending sources to LLVM IR, dispatching by region_dialect.

    HCU/HIP and MLIR compile helpers live in their own runtime modules; this function
    only routes to them. Both paths use the same make_llvm() as eager mode. Device libraries stay
    unresolved; the final HCU HSACO stage owns rocSHMEM linking.
    """
    from triton._C.libtriton import tle

    context = mod.context
    tle.load_dialects(context)

    compiled: dict[str, dict[str, Any]] = {}
    for source_id, entry in pending.items():
        payload = dict(entry)
        region_dialect = payload.get("region_dialect")
        if region_dialect in ["hcu", "hip"]:
            from triton.experimental.tle.raw.hcu.runtime import compile_deferred_pending_source
            payload["llvm_ir"] = compile_deferred_pending_source(payload, context=context)
        elif region_dialect == "mlir":
            from triton.experimental.tle.raw.mlir.runtime import compile_deferred_pending_source
            payload["llvm_ir"] = compile_deferred_pending_source(payload, context=context)
        else:
            raise RuntimeError(f"deferred raw materialize does not support region_dialect={region_dialect!r}")
        compiled[source_id] = payload
    return compiled


def deferred_raw_materialize(mod: Any, pm: Any) -> tuple[str, ...]:
    pending = list_pending_sources()
    if not pending:
        return
    libraries = tuple(
        sorted({
            entry["library"]
            for entry in pending.values()
            if entry.get("region_dialect") in {"hcu", "hip"} and entry.get("library")
        }))
    try:
        compiled = _compile_pending_raw_sources(mod, pending)
        hcu.passes.tle_raw.deferred_raw_materialize(compiled, pm)
    except Exception:
        clear_pending_sources()
        raise
    return libraries


def finish_deferred_raw_materialize() -> None:
    if not list_pending_sources():
        return
    clear_pending_sources()


def consume_link_sources() -> list[dict[str, str]]:
    sources = take_link_sources()
    print(f"## consume_link_sources: {len(sources)}, {sources}")
    return sources


def link_raw_bitcode(llir: str, sources: list[dict[str, str]], arch: str) -> str:
    """Compile each HIP source with DTK clang and llvm-link it into ``llir``."""

    clang = HIPBackend.path_to_rocm_clang()
    llvm_link = str(Path(clang).with_name("llvm-link"))
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel = Path(tmpdir) / "kernel.ll"
        kernel.write_text(llir)
        bitcodes: list[str] = []
        for index, source in enumerate(sources):
            # hip = Path(tmpdir) / f"raw_{index}.hip"
            # bc = Path(tmpdir) / f"raw_{index}.bc"
            # hip.write_text(source["source"])
            # command = _clang_command(
            #     clang,
            #     source.get("arch") or arch,
            #     str(hip),
            #     str(bc),
            #     bitcode=True,
            #     dushmem=source.get("library") == "dushmem",
            # )
            # build = subprocess.run(command, capture_output=True, text=True)
            # if build.returncode != 0:
            #     raise RuntimeError(f"HCU clang failed to emit bitcode for {source.get('file')}:\n"
            #                        f"{build.stderr}")
            # bitcodes.append(str(bc))
            if source.get("library") == "dushmem":
                from triton.experimental.tle.raw.hcu.dushmem import resolve_dushmem_device_bitcode
                bitcodes.append(str(resolve_dushmem_device_bitcode(source.get("arch") or arch)))
        # hipcc links these device libraries before codegen. The copies under
        # Triton's backend lib do not define __ockl_get_local_id.
        device_lib = Path("/opt/dtk/amdgcn/bitcode")
        isa = "".join(ch for ch in (sources[0].get("arch") or arch) if ch.isdigit())
        for name in (
                "ockl.bc",
                "ocml.bc",
                "oclc_wavefrontsize64_on.bc",
                f"oclc_isa_version_{isa}.bc",
                "oclc_abi_version_500.bc",
        ):
            path = device_lib / name
            if path.is_file():
                bitcodes.append(str(path))
        linked = Path(tmpdir) / "linked.ll"
        command = [llvm_link, str(kernel), *bitcodes, "-S", "-o", str(linked)]
        print(f"## link_raw_bitcode: {command}")
        build = subprocess.run(command, capture_output=True, text=True)
        if build.returncode != 0:
            raise RuntimeError(f"llvm-link failed:\ncommand: {' '.join(command)}\n{build.stderr}")
        return linked.read_text()

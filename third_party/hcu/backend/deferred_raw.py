from __future__ import annotations

from typing import Any

from triton._C.libtriton import hcu
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
)


def _compile_pending_raw_sources(
    mod: Any,
    pending: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    compiled: dict[str, dict[str, Any]] = {}

    for source_id, entry in pending.items():
        dialect = entry.get("region_dialect")

        if dialect in {"hcu", "hip"}:
            from triton.experimental.tle.raw.hcu.runtime import (
                compile_deferred_pending_source, )

            llvm_ir = compile_deferred_pending_source(
                entry,
                context=mod.context,
            )
        elif dialect == "mlir":
            from triton.experimental.tle.raw.mlir.runtime import (
                compile_deferred_pending_source, )

            llvm_ir = compile_deferred_pending_source(
                entry,
                context=mod.context,
            )
        else:
            raise RuntimeError("Unsupported deferred TLE-Raw dialect "
                               f"for HCU backend: {dialect!r}; "
                               f"source_id={source_id}")

        compiled[source_id] = {
            "llvm_ir": llvm_ir,
            "extern_func_name": (entry.get("extern_func_name") or ""),
        }

    return compiled


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compile pending sources to LLVM IR, dispatching by region_dialect.

    HCU/HIP and MLIR compile helpers live in their own runtime modules; this function
    only routes to them. Both paths temporarily call the same make_llvm() used by eager
    mode.
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


def deferred_raw_materialize(mod: Any, pm: Any) -> None:
    pending = list_pending_sources()
    if not pending:
        return
    compiled = _compile_pending_raw_sources(mod, pending)
    hcu.passes.tle_raw.deferred_raw_materialize(compiled, pm)


def finish_deferred_raw_materialize() -> None:
    clear_pending_sources()

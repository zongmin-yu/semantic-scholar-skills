from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any

STANDALONE_MODULE = "semantic_scholar_skills.standalone"
OUTPUT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LaunchResult:
    exit_code: int
    payload: dict[str, Any]


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[1] / "_vendor"


def _clear_semantic_scholar_modules() -> None:
    for name in list(sys.modules):
        if name == "semantic_scholar_skills" or name.startswith("semantic_scholar_skills."):
            sys.modules.pop(name, None)


def _import_installed_runtime():
    vendor_root_str = str(_vendor_root())
    if vendor_root_str in sys.path:
        sys.path.remove(vendor_root_str)
        _clear_semantic_scholar_modules()
    return importlib.import_module(STANDALONE_MODULE)


def _import_vendored_runtime():
    vendor_root = _vendor_root()
    if not vendor_root.is_dir():
        raise ModuleNotFoundError(f"Vendored runtime not found at {vendor_root}")
    vendor_root_str = str(vendor_root)
    _clear_semantic_scholar_modules()
    if vendor_root_str in sys.path:
        sys.path.remove(vendor_root_str)
    sys.path.insert(0, vendor_root_str)
    return importlib.import_module(STANDALONE_MODULE)


def _load_runtime():
    try:
        return "installed", _import_installed_runtime()
    except (ImportError, ModuleNotFoundError) as installed_exc:
        try:
            return "vendored", _import_vendored_runtime()
        except (ImportError, ModuleNotFoundError) as vendored_exc:
            raise RuntimeError(
                "Unable to import Semantic Scholar standalone runtime via installed package or vendored bundle. "
                f"Installed import failed with: {installed_exc!r}. Vendored import failed with: {vendored_exc!r}."
            ) from vendored_exc


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _runtime_payload(mode: str, module: Any | None) -> dict[str, Any]:
    return {
        "mode": mode,
        "module": getattr(module, "__name__", STANDALONE_MODULE) if module is not None else STANDALONE_MODULE,
        "path": str(getattr(module, "__file__", "")) or None,
    }


def _error_payload(
    workflow: str,
    *,
    runtime_mode: str,
    runtime_module: Any | None,
    arguments: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    for attr in (
        "details",
        "field",
        "status_code",
        "endpoint",
        "method",
        "retry_after",
        "authenticated",
        "timeout_seconds",
        "resource_type",
        "resource_id",
    ):
        if hasattr(exc, attr):
            value = getattr(exc, attr)
            if value is not None:
                error[attr] = _serialize(value)

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "workflow": workflow,
        "status": "error",
        "runtime": _runtime_payload(runtime_mode, runtime_module),
        "arguments": _serialize(arguments),
        "error": error,
    }


def launch(workflow: str, /, **kwargs: Any) -> LaunchResult:
    try:
        runtime_mode, runtime_module = _load_runtime()
    except Exception as exc:
        return LaunchResult(
            exit_code=1,
            payload=_error_payload(
                workflow,
                runtime_mode="unavailable",
                runtime_module=None,
                arguments=kwargs,
                exc=exc,
            ),
        )

    try:
        result = asyncio.run(runtime_module.run_workflow(workflow, **kwargs))
    except Exception as exc:
        return LaunchResult(
            exit_code=1,
            payload=_error_payload(
                workflow,
                runtime_mode=runtime_mode,
                runtime_module=runtime_module,
                arguments=kwargs,
                exc=exc,
            ),
        )

    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "workflow": workflow,
        "status": "ok",
        "runtime": _runtime_payload(runtime_mode, runtime_module),
        "arguments": _serialize(kwargs),
        "result": _serialize(result),
    }
    return LaunchResult(exit_code=0, payload=payload)


def dumps_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

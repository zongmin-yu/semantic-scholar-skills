from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

import semantic_scholar_skills.mcp.server as server


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    server.stop_event = None
    server.http_server = None
    server.http_server_task = None
    server.mcp_task = None
    server._shutdown_started = False
    yield
    server.stop_event = None
    server.http_server = None
    server.http_server_task = None
    server.mcp_task = None
    server._shutdown_started = False


def test_handle_exception_is_sync_callback() -> None:
    assert not inspect.iscoroutinefunction(server.handle_exception)


@pytest.mark.asyncio
async def test_shutdown_is_idempotent_and_only_cancels_managed_tasks(monkeypatch) -> None:
    calls = {"cleanup_client": 0, "mcp_cleanup": 0}

    async def fake_cleanup_client() -> None:
        calls["cleanup_client"] += 1

    class DummyMCP:
        async def cleanup(self) -> None:
            calls["mcp_cleanup"] += 1

    async def never_finishes() -> None:
        await asyncio.Event().wait()

    unrelated_task = asyncio.create_task(never_finishes(), name="unrelated-task")
    server.mcp_task = asyncio.create_task(never_finishes(), name="managed-mcp-task")
    server.http_server_task = asyncio.create_task(never_finishes(), name="managed-http-task")
    server.http_server = SimpleNamespace(should_exit=False)
    server.stop_event = asyncio.Event()

    monkeypatch.setattr(server, "cleanup_client", fake_cleanup_client)
    monkeypatch.setattr(server, "mcp", DummyMCP())

    await server.shutdown()
    await server.shutdown()

    assert server.stop_event.is_set()
    assert server.http_server.should_exit is True
    assert server.mcp_task.cancelled()
    assert server.http_server_task.cancelled()
    assert not unrelated_task.cancelled()
    assert calls == {"cleanup_client": 1, "mcp_cleanup": 1}

    unrelated_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await unrelated_task


@pytest.mark.asyncio
async def test_run_server_returns_when_mcp_task_exits(monkeypatch) -> None:
    async def fake_initialize_client() -> None:
        return None

    async def fake_cleanup_client() -> None:
        return None

    class DummyMCP:
        async def run_async(self) -> None:
            return None

        async def cleanup(self) -> None:
            return None

    monkeypatch.setattr(server, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(server, "cleanup_client", fake_cleanup_client)
    monkeypatch.setattr(server, "mcp", DummyMCP())
    monkeypatch.setenv("SEMANTIC_SCHOLAR_ENABLE_HTTP_BRIDGE", "0")

    await asyncio.wait_for(server.run_server(), timeout=0.5)

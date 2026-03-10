"""
Main server module for the Semantic Scholar API Server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Any

import uvicorn

from . import cleanup_client, initialize_client, mcp
from . import tools_authors, tools_papers, tools_recommendations

logger = logging.getLogger(__name__)

stop_event: asyncio.Event | None = None
http_server: uvicorn.Server | None = None
http_server_task: asyncio.Task[Any] | None = None
mcp_task: asyncio.Task[Any] | None = None
_shutdown_started = False


def handle_exception(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Global exception handler for the event loop."""
    msg = context.get("exception", context.get("message", "unknown event loop exception"))
    logger.error("Caught exception: %s", msg)
    if loop.is_closed():
        return
    loop.create_task(shutdown())


async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _cleanup_mcp_runtime() -> None:
    cleanup_fn = getattr(mcp, "cleanup", None)
    if cleanup_fn:
        if asyncio.iscoroutinefunction(cleanup_fn):
            await cleanup_fn()
        else:
            cleanup_fn()
        return

    for name in ("shutdown", "stop", "close"):
        fn = getattr(mcp, name, None)
        if fn is None:
            continue
        if asyncio.iscoroutinefunction(fn):
            await fn()
        else:
            fn()
        return


async def shutdown() -> None:
    """Gracefully shut down only the tasks this module owns."""
    global _shutdown_started, stop_event, http_server, http_server_task, mcp_task

    if _shutdown_started:
        return
    _shutdown_started = True
    logger.info("Initiating graceful shutdown...")

    if stop_event is not None and not stop_event.is_set():
        stop_event.set()

    try:
        if http_server is not None:
            http_server.should_exit = True
        await _cancel_task(http_server_task)
        await _cancel_task(mcp_task)
        await cleanup_client()
        await _cleanup_mcp_runtime()
    except Exception as exc:
        logger.error("Error during shutdown: %s", exc)
    finally:
        logger.info("Shutdown complete")


def init_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Initialize signal handlers for graceful shutdown."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: loop.create_task(shutdown()))
    logger.info("Signal handlers initialized")


async def run_server() -> None:
    """Run the server with proper async context management."""
    global stop_event, http_server, http_server_task, mcp_task, _shutdown_started

    _shutdown_started = False
    stop_event = asyncio.Event()
    http_server = None
    http_server_task = None
    mcp_task = None

    stop_waiter: asyncio.Task[bool] | None = None
    try:
        await initialize_client()

        logger.info("Starting Semantic Scholar Server")
        mcp_task = asyncio.create_task(mcp.run_async(), name="semantic-scholar-mcp")

        enable_bridge = os.getenv("SEMANTIC_SCHOLAR_ENABLE_HTTP_BRIDGE", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if enable_bridge:
            bridge_host = os.getenv("SEMANTIC_SCHOLAR_HTTP_BRIDGE_HOST", "0.0.0.0").strip()
            bridge_port = int(os.getenv("SEMANTIC_SCHOLAR_HTTP_BRIDGE_PORT", "8000"))
            from .bridge import app as bridge_app

            config = uvicorn.Config(
                app=bridge_app,
                host=bridge_host,
                port=bridge_port,
                log_level="info",
                log_config=None,
                ws="none",
            )
            http_server = uvicorn.Server(config=config)
            http_server_task = asyncio.create_task(
                http_server.serve(),
                name="semantic-scholar-http-bridge",
            )
            logger.info("HTTP bridge enabled on %s:%s", bridge_host, bridge_port)
        else:
            logger.info("HTTP bridge disabled (SEMANTIC_SCHOLAR_ENABLE_HTTP_BRIDGE=0)")

        stop_waiter = asyncio.create_task(stop_event.wait(), name="semantic-scholar-stop-waiter")
        wait_set = {stop_waiter, mcp_task}
        if http_server_task is not None:
            wait_set.add(http_server_task)

        done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if task is stop_waiter:
                continue
            task.cancel()

        if stop_waiter not in done:
            for task in done:
                if task is stop_waiter:
                    continue
                await task
            await shutdown()
    except Exception as exc:
        logger.error("Server error: %s", exc)
        raise
    finally:
        if stop_waiter is not None and not stop_waiter.done():
            stop_waiter.cancel()
            try:
                await stop_waiter
            except asyncio.CancelledError:
                pass
        await shutdown()


def main() -> None:
    """Main entry point for the server."""
    loop: asyncio.AbstractEventLoop | None = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(handle_exception)
        init_signal_handlers(loop)
        loop.run_until_complete(run_server())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
    finally:
        if loop is not None:
            try:
                loop.run_until_complete(asyncio.sleep(0))
                loop.close()
            except Exception as exc:
                logger.error("Error during final cleanup: %s", exc)
        logger.info("Server stopped")


if __name__ == "__main__":
    main()

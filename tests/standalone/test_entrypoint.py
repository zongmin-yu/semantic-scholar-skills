from __future__ import annotations

import pytest

from semantic_scholar_skills.standalone import entrypoint
from semantic_scholar_skills.standalone.transport_stdlib import StdlibTransport


class FakeS2Client:
    def __init__(self, transport) -> None:
        self.transport = transport


@pytest.mark.asyncio
async def test_create_client_uses_stdlib_transport_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        entrypoint,
        "_load_runtime",
        lambda: (FakeS2Client, object(), object(), object()),
    )

    client = entrypoint.create_client()

    assert isinstance(client, FakeS2Client)
    assert isinstance(client.transport, StdlibTransport)


@pytest.mark.asyncio
async def test_run_expand_references_injects_created_client(monkeypatch) -> None:
    recorded = {}

    async def fake_expand(client, seeds, **kwargs):
        recorded["client"] = client
        recorded["seeds"] = list(seeds)
        recorded["kwargs"] = kwargs
        return "expanded"

    monkeypatch.setattr(
        entrypoint,
        "_load_runtime",
        lambda: (FakeS2Client, fake_expand, object(), object()),
    )

    result = await entrypoint.run_expand_references(["seed-a"], negative_seeds=["seed-b"], per_bucket_limit=2)

    assert result == "expanded"
    assert isinstance(recorded["client"], FakeS2Client)
    assert recorded["seeds"] == ["seed-a"]
    assert recorded["kwargs"]["negative_seeds"] == ["seed-b"]
    assert recorded["kwargs"]["per_bucket_limit"] == 2


@pytest.mark.asyncio
async def test_run_workflow_dispatches_to_named_runner(monkeypatch) -> None:
    async def fake_expand(**kwargs):
        return ("expand", kwargs)

    async def fake_trace(**kwargs):
        return ("trace", kwargs)

    async def fake_triage(**kwargs):
        return ("triage", kwargs)

    monkeypatch.setattr(entrypoint, "run_expand_references", fake_expand)
    monkeypatch.setattr(entrypoint, "run_trace_citations", fake_trace)
    monkeypatch.setattr(entrypoint, "run_paper_triage", fake_triage)

    assert await entrypoint.run_workflow("expand-references", seeds=["p1"]) == ("expand", {"seeds": ["p1"]})
    assert await entrypoint.run_workflow("trace-citations", focal_query="p1") == ("trace", {"focal_query": "p1"})
    assert await entrypoint.run_workflow("paper-triage", query="bert") == ("triage", {"query": "bert"})


def test_entrypoint_prefers_absolute_imports_and_falls_back_to_relative_imports(monkeypatch) -> None:
    sentinel = (FakeS2Client, "expand", "trace", "triage")
    calls: list[str] = []

    def fake_absolute():
        calls.append("absolute")
        raise ModuleNotFoundError("absolute failed")

    def fake_relative():
        calls.append("relative")
        return sentinel

    monkeypatch.setattr(entrypoint, "_import_absolute_runtime", fake_absolute)
    monkeypatch.setattr(entrypoint, "_import_relative_runtime", fake_relative)

    assert entrypoint._load_runtime() == sentinel
    assert calls == ["absolute", "relative"]

"""Durable checkpointer must persist conversation state across graph instances."""

from app.core.config import settings
from app.services.checkpointer import build_chat_checkpointer


def test_sqlite_checkpointer_built(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chat_checkpointer_backend", "sqlite")
    monkeypatch.setattr(settings, "chat_checkpointer_path", str(tmp_path / "cp.sqlite"))
    saver = build_chat_checkpointer()
    assert saver.__class__.__name__ == "SqliteSaver"


def test_memory_backend_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "chat_checkpointer_backend", "memory")
    saver = build_chat_checkpointer()
    assert saver.__class__.__name__ in {"MemorySaver", "InMemorySaver"}


def test_state_survives_new_checkpointer_instance(tmp_path, monkeypatch):
    """A second SqliteSaver on the same file sees checkpoints written by the first."""
    path = str(tmp_path / "persist.sqlite")
    monkeypatch.setattr(settings, "chat_checkpointer_backend", "sqlite")
    monkeypatch.setattr(settings, "chat_checkpointer_path", path)

    from langgraph.checkpoint.base import empty_checkpoint

    saver1 = build_chat_checkpointer()
    config = {"configurable": {"thread_id": "session-x", "checkpoint_ns": ""}}
    checkpoint = empty_checkpoint()
    saver1.put(config, checkpoint, {"source": "input", "step": 0, "writes": {}}, {})

    saver2 = build_chat_checkpointer()
    restored = saver2.get_tuple(config)
    assert restored is not None
    assert restored.checkpoint["id"] == checkpoint["id"]


def test_postgres_falls_back_to_sqlite_when_no_dsn(tmp_path, monkeypatch):
    """postgres backend with no DSN must not crash; falls back to durable sqlite."""
    monkeypatch.setattr(settings, "chat_checkpointer_backend", "postgres")
    monkeypatch.setattr(settings, "chat_checkpointer_pg_dsn", "")
    monkeypatch.setattr(settings, "database_url", "")
    monkeypatch.setattr(settings, "chat_checkpointer_path", str(tmp_path / "fb.sqlite"))
    saver = build_chat_checkpointer()
    assert saver.__class__.__name__ == "SqliteSaver"


def test_normalize_pg_dsn_strips_driver():
    from app.services.checkpointer import _normalize_pg_dsn
    assert _normalize_pg_dsn("postgresql+psycopg://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"
    assert _normalize_pg_dsn("") == ""

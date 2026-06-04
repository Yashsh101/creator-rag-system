"""Durable LangGraph checkpointer factory for the video chat graph.

Replaces in-process ``MemorySaver`` so conversation state survives process
restarts, redeploys, and multi-worker deployments.

Backends (``CHAT_CHECKPOINTER_BACKEND``):
- ``sqlite``   : zero-infra default, durable on a mounted volume (single box).
- ``postgres`` : horizontal multi-worker scale (shared state across replicas).
- ``memory``   : ephemeral, opt-in only (tests / throwaway demos).

Every backend degrades gracefully: a misconfigured durable backend falls back
to in-memory rather than crashing the chat path during a live demo.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_chat_checkpointer() -> Any:
    """Builds the configured durable checkpointer with graceful fallback."""
    backend = (settings.chat_checkpointer_backend or "sqlite").lower()

    if backend == "memory":
        logger.warning("chat_checkpointer_memory", extra={"event": "chat_checkpointer_memory"})
        return MemorySaver()

    if backend == "postgres":
        saver = _build_postgres()
        if saver is not None:
            return saver
        logger.warning(
            "chat_checkpointer_postgres_unavailable_fallback_sqlite",
            extra={"event": "chat_checkpointer_postgres_unavailable_fallback_sqlite"},
        )

    return _build_sqlite()


def _build_sqlite() -> Any:
    """SQLite saver on a mounted path; ``check_same_thread=False`` for the threadpool."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = settings.chat_checkpointer_path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        logger.info("chat_checkpointer_sqlite", extra={"event": "chat_checkpointer_sqlite", "path": path})
        return saver
    except Exception:
        logger.warning(
            "chat_checkpointer_sqlite_failed_fallback_memory",
            exc_info=True,
            extra={"event": "chat_checkpointer_sqlite_failed_fallback_memory"},
        )
        return MemorySaver()


def _build_postgres() -> Any | None:
    """Postgres saver for multi-worker deploys; returns None if unavailable.

    Uses a long-lived connection pool so chat workers share conversation state.
    Requires ``langgraph-checkpoint-postgres`` and ``CHAT_CHECKPOINTER_PG_DSN``
    (falls back to ``DATABASE_URL`` stripped of the SQLAlchemy driver suffix).
    """
    dsn = settings.chat_checkpointer_pg_dsn or _normalize_pg_dsn(settings.database_url)
    if not dsn or not dsn.startswith("postgres"):
        return None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(conninfo=dsn, max_size=settings.chat_checkpointer_pg_pool_size, open=True, kwargs={"autocommit": True})
        saver = PostgresSaver(pool)
        saver.setup()
        logger.info("chat_checkpointer_postgres", extra={"event": "chat_checkpointer_postgres"})
        return saver
    except Exception:
        logger.warning(
            "chat_checkpointer_postgres_init_failed",
            exc_info=True,
            extra={"event": "chat_checkpointer_postgres_init_failed"},
        )
        return None


def _normalize_pg_dsn(database_url: str) -> str:
    """Strips the SQLAlchemy ``+psycopg`` driver suffix for the raw psycopg DSN."""
    if not database_url:
        return ""
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)

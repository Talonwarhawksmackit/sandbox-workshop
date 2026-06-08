"""
core/library.py — Snippet Library Backend
==========================================
Pydantic-mapped SnippetLibrary class backed by a local SQLite database
(``sandbox_factory.db``).

Schema
------
  materials(id INTEGER PK, content TEXT, language TEXT,
            classification TEXT, domain TEXT, metadata TEXT)

Supported classification tags : UI · Logic · Config · Story
Supported domain targets      : Education · Utility · Gaming
                                 (Minecraft / Roblox / SixSeven hooks)
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH: Path = Path(__file__).resolve().parents[1] / "sandbox_factory.db"

VALID_CLASSIFICATIONS = {"UI", "Logic", "Config", "Story"}
VALID_DOMAINS = {"Education", "Utility", "Gaming"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Snippet(BaseModel):
    """A single code material stored in the Snippet Library."""

    id: Optional[int] = None
    content: str
    language: str = "Python"
    classification: str = "Logic"
    domain: str = "Education"
    metadata: str = ""

    def model_post_init(self, __context) -> None:  # noqa: D401
        """Warn (but do not reject) if tags fall outside the known set."""
        if self.classification not in VALID_CLASSIFICATIONS:
            logger.warning(
                "Classification '%s' is outside the standard set %s",
                self.classification,
                VALID_CLASSIFICATIONS,
            )
        if self.domain not in VALID_DOMAINS:
            logger.warning(
                "Domain '%s' is outside the standard set %s",
                self.domain,
                VALID_DOMAINS,
            )


class SnippetLibrary(BaseModel):
    """Pydantic-mapped wrapper around the local SQLite Snippet Library.

    Provides instance methods for connection management, insertion,
    counting, and randomised retrieval filtered by language and domain.

    Parameters
    ----------
    db_path : Path
        Filesystem path to the SQLite database file.
        Defaults to ``sandbox_factory.db`` in the project root.
    """

    db_path: Path = Field(default=DEFAULT_DB_PATH)

    # Internal connection handle — excluded from serialisation
    _conn: Optional[sqlite3.Connection] = None

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> "SnippetLibrary":
        """Open (or reuse) a connection and ensure the schema exists.

        Returns *self* so callers can chain: ``lib = SnippetLibrary().connect()``
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS materials (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    content         TEXT    NOT NULL,
                    language        TEXT    NOT NULL,
                    classification  TEXT    NOT NULL,
                    domain          TEXT    NOT NULL,
                    metadata        TEXT
                );
                """
            )
            self._conn.commit()
            logger.info("Connected to Snippet Library at %s", self.db_path)
        return self

    def close(self) -> None:
        """Close the database connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Snippet Library connection closed.")

    def _ensure_connected(self) -> sqlite3.Connection:
        """Return the active connection, auto-connecting if needed."""
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def insert(self, snippet: Snippet) -> int:
        """Insert a Snippet and return the new row id."""
        conn = self._ensure_connected()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO materials (content, language, classification, domain, metadata)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                snippet.content,
                snippet.language,
                snippet.classification,
                snippet.domain,
                snippet.metadata,
            ),
        )
        conn.commit()
        row_id: int = cur.lastrowid  # type: ignore[assignment]
        logger.info("Inserted snippet id=%d  [%s / %s]", row_id, snippet.classification, snippet.domain)
        return row_id

    def count(self, language: Optional[str] = None, domain: Optional[str] = None) -> int:
        """Return the number of materials, optionally filtered."""
        conn = self._ensure_connected()
        query = "SELECT COUNT(*) FROM materials"
        params: list = []
        clauses: list[str] = []
        if language:
            clauses.append("language = ?")
            params.append(language)
        if domain:
            clauses.append("domain = ?")
            params.append(domain)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        cur = conn.cursor()
        cur.execute(query, params)
        total: int = cur.fetchone()[0]
        return total

    def get_random_cluster(
        self,
        language: str = "Python",
        domain: str = "Education",
        limit: int = 5,
    ) -> List[Snippet]:
        """Pull a small, randomised cluster of language+domain-compatible materials.

        Parameters
        ----------
        language : str
            Programming language filter (e.g. ``"Python"``).
        domain : str
            Target domain filter (e.g. ``"Education"``).
        limit : int
            Maximum snippets to return.

        Returns
        -------
        list[Snippet]
            Randomised list of matching snippets (may be empty).
        """
        conn = self._ensure_connected()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, content, language, classification, domain, metadata
            FROM materials
            WHERE language = ? AND domain = ?
            ORDER BY RANDOM()
            LIMIT ?;
            """,
            (language, domain, limit),
        )
        rows = cur.fetchall()
        snippets = [
            Snippet(
                id=r[0],
                content=r[1],
                language=r[2],
                classification=r[3],
                domain=r[4],
                metadata=r[5] or "",
            )
            for r in rows
        ]
        logger.info(
            "Retrieved %d snippet(s) for language=%s, domain=%s",
            len(snippets),
            language,
            domain,
        )
        return snippets

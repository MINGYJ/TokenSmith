"""
src/sql/db.py

Builds and queries the SQLite metadata database from the intermediate
chunks_meta.json produced by index_builder.py.

Schema
------
documents  – one row per unique source file
chunks     – one row per sql_eligible chunk, keyed by FAISS chunk_id

Usage (called automatically by build_index in index_builder.py)
-------
    from src.sql.db import build_sql_db
    build_sql_db(meta_json_path, db_path)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional


# ─────────────────────────── schema ───────────────────────────────────

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    UNIQUE NOT NULL,
    indexed_at  TEXT    NOT NULL
);
"""

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        INTEGER PRIMARY KEY,
    doc_id          INTEGER NOT NULL REFERENCES documents(doc_id),
    chapter         INTEGER NOT NULL DEFAULT 0,
    section         TEXT    NOT NULL DEFAULT '',
    section_path    TEXT    NOT NULL DEFAULT '',
    page_numbers    TEXT    NOT NULL DEFAULT '[]',
    char_len        INTEGER NOT NULL DEFAULT 0,
    word_len        INTEGER NOT NULL DEFAULT 0,
    content_preview TEXT    NOT NULL DEFAULT ''
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_chapter      ON chunks(chapter);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id       ON chunks(doc_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_section      ON chunks(section);",
]


# ─────────────────────────── eligibility logic ────────────────────────

_GENERIC_HEADINGS = {"introduction", "full document"}

#check if this chunk is eligible for SQL search
def compute_sql_eligible(meta: Dict[str, Any]) -> tuple[bool, str]:
    """
    Decide whether a chunk has enough structural metadata for SQL indexing.

    A chunk only needs a real chapter number and at least one page number.
    Section heading does NOT disqualify a chunk — if the section is generic
    (e.g. 'Introduction') the chunk is still reachable via chapter + page
    queries, which is perfectly useful.

    Returns (eligible: bool, reason: str).
    """
    chapter = meta.get("chapter", 0)
    page_numbers = meta.get("page_numbers", [])
    section = meta.get("section", "").strip().lower()

    if chapter == 0:
        return False, "missing_chapter"
    if not page_numbers:
        return False, "no_pages"
    #check if we have specific section info
    if section in _GENERIC_HEADINGS or section == "":
        return True, "chapter_only"   # searchable by chapter/page, not section
    return True, "has_chapter_section_pages"


#Build SQL DB

def build_sql_db(meta_json_path: Path, db_path: Path) -> int:
    """
    Read the intermediate chunks_meta.json produced by build_index() and
    populate the SQLite database.

    Returns the number of SQL-eligible chunks inserted.

    Parameters
    ----------
    meta_json_path : Path
        Path to the  *_chunks_meta.json  file written by index_builder.py.
    db_path : Path
        Destination SQLite file (created/replaced on each full index run).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Load intermediate JSON
    with open(meta_json_path, "r", encoding="utf-8") as f:
        all_meta: List[Dict[str, Any]] = json.load(f)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        # Drop and recreate tables so a re-index always produces a clean DB,
        # with no stale rows from a previous run with more chunks.
        conn.execute("DROP TABLE IF EXISTS chunks;")
        conn.execute("DROP TABLE IF EXISTS documents;")
        conn.execute(_CREATE_DOCUMENTS)
        conn.execute(_CREATE_CHUNKS)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.commit()

        now = datetime.now(timezone.utc).isoformat()
        inserted = 0

        # Cache filename → doc_id to avoid repeated inserts
        doc_id_cache: Dict[str, int] = {}

        for meta in all_meta:
            eligible, reason = compute_sql_eligible(meta)
            if not eligible:
                continue  # FAISS-only chunk; skip SQL

            filename = meta.get("filename", "unknown")

            # Upsert document row
            if filename not in doc_id_cache:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO documents (filename, indexed_at) VALUES (?, ?)",
                    (filename, now),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT doc_id FROM documents WHERE filename = ?", (filename,)
                ).fetchone()
                doc_id_cache[filename] = row[0]

            doc_id = doc_id_cache[filename]

            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, chapter, section, section_path,
                     page_numbers, char_len, word_len, content_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta["chunk_id"],
                    doc_id,
                    meta.get("chapter", 0),
                    meta.get("section", ""),
                    meta.get("section_path", ""),
                    json.dumps(meta.get("page_numbers", [])),
                    meta.get("char_len", 0),
                    meta.get("word_len", 0),
                    meta.get("text_preview", ""),
                ),
            )
            inserted += 1

        conn.commit()
        print(
            f"  SQL DB: inserted {inserted} / {len(all_meta)} chunks "
            f"({len(all_meta) - inserted} FAISS-only) → {db_path}"
        )
        return inserted

    finally:
        conn.close()


#SQL Query

def get_section_names_for_chunks(db_path: Path, chunk_ids: List[int]) -> List[tuple]:
    """
    Return distinct (section, count) pairs for the given chunk_ids, sorted by count desc.
    Used to show the user which sections were matched at the SQL stage.
    """
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT section, COUNT(*) as cnt FROM chunks "
            f"WHERE chunk_id IN ({placeholders}) "
            f"GROUP BY section ORDER BY cnt DESC",
            chunk_ids,
        ).fetchall()
    return [(r[0] or "(no section)", r[1]) for r in rows]


def query_chunks_by_chapter(db_path: Path, chapter: int, limit: int = 50) -> List[int]:
    """Return FAISS chunk_ids for all chunks in the given chapter.

    Parameters
    ----------
    limit : cap on returned rows (default 50).  Pass 0 for no limit.
    """
    sql = "SELECT chunk_id FROM chunks WHERE chapter = ? ORDER BY chunk_id"
    params: list = [chapter]
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows]


def query_chunks_by_section_numeric(db_path: Path, section_num: str, chapter: int = 0) -> List[int]:
    """
    Precise numeric-section lookup for dotted references like '5.1'.

    Matches sections whose heading begins with exactly 'Section <num>' —
    including subsections (5.1.1, 5.1.2 …) — but excludes false positives
    such as '25.1' or '5.10' that a plain substring LIKE would incorrectly hit.

    Falls back to the full chapter (capped at 50) if nothing matches.
    """
    prefix = f"Section {section_num}"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT chunk_id FROM chunks
            WHERE (
                section = ?
                OR section LIKE ?
                OR section LIKE ?
            )
            AND (? = 0 OR chapter = ?)
            """,
            (prefix, f"{prefix} %", f"{prefix}.%", chapter, chapter),
        ).fetchall()
        chunk_ids = [r[0] for r in rows]

        if not chunk_ids and chapter != 0:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE chapter = ? ORDER BY chunk_id LIMIT 50",
                (chapter,),
            ).fetchall()
            chunk_ids = [r[0] for r in rows]

    return chunk_ids


def query_chunks_by_section(db_path: Path, section_keyword: str, chapter: int = 0) -> List[int]:
    """
    Return FAISS chunk_ids whose section heading contains the keyword.
    If no section match is found AND a chapter number is provided, falls back
    to returning all chunks from that chapter (chapter-level search).
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT chunk_id FROM chunks WHERE section LIKE ?",
            (f"%{section_keyword}%",),
        ).fetchall()
        chunk_ids = [r[0] for r in rows]

        # Fallback: if no section match, widen to the whole chapter
        if not chunk_ids and chapter != 0:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE chapter = ?", (chapter,)
            ).fetchall()
            chunk_ids = [r[0] for r in rows]

    return chunk_ids


def query_chunks_by_page(db_path: Path, page: int) -> List[int]:
    """Return FAISS chunk_ids that touch a given page number."""
    page_str = str(page)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT chunk_id, page_numbers FROM chunks"
        ).fetchall()
    results = []
    for chunk_id, page_json in rows:
        try:
            pages = json.loads(page_json)
            if page in pages or page_str in pages:
                results.append(chunk_id)
        except (json.JSONDecodeError, TypeError):
            continue
    return results


def query_chunks_by_document(db_path: Path, filename_keyword: str) -> List[int]:
    """Return FAISS chunk_ids from documents whose filename contains the keyword."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.chunk_id FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE d.filename LIKE ?
            """,
            (f"%{filename_keyword}%",),
        ).fetchall()
    return [r[0] for r in rows]


def get_db_stats(db_path: Path) -> Dict[str, Any]:
    """Return a summary dict of the database contents."""
    with sqlite3.connect(db_path) as conn:
        n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        chapters = conn.execute(
            "SELECT DISTINCT chapter FROM chunks ORDER BY chapter"
        ).fetchall()
    return {
        "documents": n_docs,
        "sql_eligible_chunks": n_chunks,
        "chapters": [r[0] for r in chapters],
    }

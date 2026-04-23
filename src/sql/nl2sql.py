"""
src/sql/nl2sql.py

Tier 1 structural query classifier and SQL-retrieval dispatcher.

Detects structural patterns in a query (chapter, page, section) using
regex and dispatches to the appropriate db helpers.  No LLM required —
zero latency.  Returns [] when the query has no structural signals,
signalling the caller to rely fully on FAISS/BM25.

Priority order:  page  >  named-section  >  numeric-section  >  chapter
Multiple signals are combined (e.g. chapter + section).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from src.sql.db import (
    query_chunks_by_chapter,
    query_chunks_by_section,
    query_chunks_by_page,
    query_chunks_by_document,
)

#match page name
_PAGE_RE = re.compile(
    r'\b(?:page|slide|p\.?)\s*(\d+)\b', re.IGNORECASE
)

#match chapter name
_CHAPTER_RE = re.compile(
    r'\bchapter\s+(\d+)\b', re.IGNORECASE
)

#match section name
_SECTION_NAMED_RE = re.compile(
    r'\bsection\s+(?:on|about|covering|for|regarding|called|named)?\s*'
    r'"?([A-Za-z][A-Za-z0-9 _\-]{2,50}?)"?'
    r'(?=\s*(?:$|\?|\.|,|\band\b|\bin\b|\bof\b|\bfrom\b))',
    re.IGNORECASE,
)

#numeric dotted reference
_SECTION_NUM_RE = re.compile(
    r'\bsection\s+([\d]+(?:\.[\d]+)+)\b', re.IGNORECASE
)


#Public methods for reference

def get_sql_chunk_ids(query: str, db_path: Path) -> List[int]:
    """
    Direct classifier: detect structural patterns and return matching chunk_ids.

    Returns an empty list when the query does not contain structural signals.
    The caller (SQLRetriever) treats an empty result as 'not a structural query'
    and returns {} so that FAISS/BM25 rankings are used unchanged.

    Parameters
    ----------
    query    : The raw user query string.
    db_path  : Path to the SQLite metadata database.
    """
    if not db_path.exists():
        return []
    print(f"SQL Hybrid enabled: analyzing query for structural signals: '{query}'")
    page_m          = _PAGE_RE.search(query)
    chapter_m       = _CHAPTER_RE.search(query)
    section_named_m = _SECTION_NAMED_RE.search(query)
    section_num_m   = _SECTION_NUM_RE.search(query)

    chapter_num = int(chapter_m.group(1)) if chapter_m else 0

    # 1. Page reference — most specific, return immediately
    if page_m:
        return query_chunks_by_page(db_path, int(page_m.group(1)))

    # 2. Named section — "section on deadlocks" (optionally with chapter)
    if section_named_m:
        keyword = section_named_m.group(1).strip()
        return query_chunks_by_section(db_path, keyword, chapter=chapter_num)

    # 3. Numeric section — "section 13.2" (searches section_path column via LIKE)
    if section_num_m:
        keyword = section_num_m.group(1)
        return query_chunks_by_section(db_path, keyword, chapter=chapter_num)

    # 4. Chapter only — broadest structural match
    if chapter_m:
        return query_chunks_by_chapter(db_path, chapter_num)

    # No structural signal detected — not a SQL query
    return []

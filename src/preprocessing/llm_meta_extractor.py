"""
src/preprocessing/llm_meta_extractor.py

LLM-based fallback metadata extractor for sql_eligible=False chunks.

Only called during indexing when enable_llm_meta_extraction=True and
compute_sql_eligible() has returned False — meaning the regex pipeline
could not recover a chapter number from the Markdown heading structure.

The extractor loads the generation LLM with grammar-constrained decoding
(GBNF) so it is forced to produce exactly this JSON shape:

    {"chapter": <int>, "section": "<str>", "page_numbers": [<int>, ...]}

Validation after extraction:
    - chapter must be in [1, 50];  0 → extraction failed → returns None
    - page_numbers: each value must be in [1, 5000]; others silently dropped
    - section: any string is accepted; empty string is fine

Returns None when the LLM produces output that cannot be parsed or
validated.  Callers must treat None as "enrichment failed, keep original
metadata".
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GBNF grammar — forces the Llama model to emit exactly the JSON we need.
# LlamaGrammar interprets this before sampling, so the output is always
# structurally valid; json.loads() will not raise on a well-formed response.
# ---------------------------------------------------------------------------
_GRAMMAR = r"""
root      ::= "{" ws "\"chapter\"" ws ":" ws integer ws "," ws
              "\"section\"" ws ":" ws string ws "," ws
              "\"page_numbers\"" ws ":" ws int-array ws "}"
integer   ::= [0-9]+
string    ::= "\"" char* "\""
char      ::= [^"\\] | "\\" ["\\/bfnrt]
int-array ::= "[" ws "]" | "[" ws integer (ws "," ws integer)* ws "]"
ws        ::= [ \t\n\r]*
"""

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = """\
You are a metadata extractor for a textbook. Read the chunk below and \
return one JSON object with three fields:
- "chapter": the chapter number as an integer (use 0 if you cannot determine it)
- "section": the section heading as a string (empty string if unknown)
- "page_numbers": a JSON array of page numbers that appear in the chunk \
(empty array if none are visible)

Hint — current section path from heading structure (may be incomplete):
{section_path}

Chunk text (first 600 characters):
{chunk_preview}

JSON:"""


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LLMMetaExtractor:
    """
    Lazy-loading wrapper around llama_cpp.Llama for single-chunk metadata
    extraction.  The LLM is loaded on the first call to extract() so that
    import time is not affected when the feature is disabled.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._llm = None
        self._grammar = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from llama_cpp import Llama, LlamaGrammar  # deferred import
        logger.info("LLMMetaExtractor: loading model %s", self._model_path)
        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=1024,       # prompt + JSON reply fits easily
            n_gpu_layers=-1,  # use GPU layers if available
            verbose=False,
        )
        self._grammar = LlamaGrammar.from_string(_GRAMMAR)
        logger.info("LLMMetaExtractor: model loaded")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        chunk_text: str,
        section_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Run grammar-constrained generation and return a validated metadata
        dict, or None if extraction fails.

        Parameters
        ----------
        chunk_text:
            The cleaned chunk content (page markers already stripped).
        section_path:
            The section path string built by index_builder.py; provided as
            context so the model can verify or refine its answer.

        Returns
        -------
        dict with keys "chapter" (int), "section" (str),
        "page_numbers" (list[int]) — or None on failure.
        """
        if self._llm is None:
            self._load()

        prompt = _PROMPT_TEMPLATE.format(
            section_path=section_path or "(none)",
            chunk_preview=chunk_text[:600],
        )

        try:
            response = self._llm(
                prompt,
                max_tokens=128,
                temperature=0.0,
                grammar=self._grammar,
            )
            raw = response["choices"][0]["text"].strip()
            data = json.loads(raw)
        except Exception as exc:
            logger.debug("LLMMetaExtractor: generation/parse error — %s", exc)
            return None

        return _validate(data)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and coerce the parsed JSON.  Returns None if chapter is 0
    (meaning the model could not determine it), which tells the caller to
    leave the chunk as sql_eligible=False.
    """
    try:
        chapter = int(data.get("chapter", 0))
        section = str(data.get("section", "")).strip()
        page_numbers = [int(p) for p in data.get("page_numbers", [])]
    except (TypeError, ValueError):
        return None

    # chapter=0 means "unknown" — treat as extraction failure
    if chapter == 0:
        return None

    # Sanity bounds: chapters beyond 50 or pages beyond 5000 are noise
    if not (1 <= chapter <= 50):
        return None

    page_numbers = [p for p in page_numbers if 1 <= p <= 5000]

    return {
        "chapter": chapter,
        "section": section,
        "page_numbers": page_numbers,
    }

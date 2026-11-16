"""HTTP adaptation of the exact platform context, generation and validation path."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import hashlib
import logging
from pathlib import Path
import re
import time

import httpx

from contender_context import ContextEngine
from solution import Model, answer_one
from psb_backend import Answer, BackendError, Question, env_number, source_record

LOG = logging.getLogger(__name__)


def _label(item: dict) -> str:
    value = re.sub(r"\s+", " ", item.get("source", "")).strip()
    if item.get("metadata", {}).get("packet_type") == "complete_document":
        value = value.split(" [полный текст;", 1)[0]
    return value


class ContenderService:
    """Keep HTTP source links around the same pipeline used by solution.py."""

    def __init__(self, directory: Path, client: httpx.AsyncClient):
        self.timeout = env_number("PSB_REQUEST_TIMEOUT", 300.0, 0.1, 600.0)
        self.client = client
        try:
            Model(client, time.monotonic() + self.timeout)
            self.engine = ContextEngine(directory)
        except (ValueError, OSError, ImportError):
            raise BackendError("unavailable") from None
        self._slots = asyncio.Semaphore(env_number("MAX_WORKERS", 2, 1, 8))
        self.documents = {
            doc_id: {"doc_id": doc_id, "doc_title": doc.metadata["doc_title"],
                     "file_path": doc.metadata["file_path"], "text": doc.text,
                     "offset_encoding": "unicode_codepoints"}
            for doc_id, doc in self.engine.documents.items()
        }
        self._fragments: dict[str, dict] = {}
        self._spans: dict[tuple, str] = {}
        for chunk in self.engine.retriever._index.chunks:
            record = {**source_record(chunk.chunk_id, chunk.source_reference, chunk.metadata), "text": chunk.text}
            self._fragments[chunk.chunk_id] = record
            self._spans[(chunk.doc_id, chunk.metadata.get("char_start"), chunk.metadata.get("char_end"))] = chunk.chunk_id
        # Several paragraphs of one law article share an identical human reference.
        # The batch text retains that reference, so HTTP exposes its entire original span.
        self._groups: dict[tuple[str, str], str] = {}
        groups = defaultdict(list)
        for chunk in self.engine.retriever._index.chunks:
            groups[(chunk.doc_id, _label({"source": chunk.source_reference}))].append(chunk)
        for (doc_id, reference), chunks in groups.items():
            if len(chunks) < 2:
                continue
            start = min(chunk.metadata["char_start"] for chunk in chunks)
            end = max(chunk.metadata["char_end"] for chunk in chunks)
            chunk_id = "provision_" + hashlib.sha256((doc_id + "\0" + reference).encode()).hexdigest()[:24]
            metadata = {**chunks[0].metadata, "char_start": start, "char_end": end}
            self._fragments[chunk_id] = {**source_record(chunk_id, reference, metadata),
                                         "text": self.engine.documents[doc_id].text[start:end]}
            self._groups[(doc_id, reference)] = chunk_id
        # Full document packets have stable IDs so saved dialog links survive restart.
        for doc_id, doc in self.engine.documents.items():
            chunk_id = "packet_" + hashlib.sha256(doc_id.encode()).hexdigest()[:24]
            metadata = {**doc.metadata, "char_start": 0, "char_end": len(doc.text)}
            self._fragments[chunk_id] = {**source_record(chunk_id, doc.metadata["doc_title"], metadata), "text": doc.text}
            self._spans[(doc_id, "packet")] = chunk_id

    def fragment(self, chunk_id: str) -> dict | None:
        return self._fragments.get(chunk_id)

    def _source(self, item: dict) -> dict:
        metadata = item["metadata"]
        key = (metadata["doc_id"], "packet") if metadata.get("packet_type") == "complete_document" else (
            metadata["doc_id"], metadata.get("char_start"), metadata.get("char_end"))
        chunk_id = (self._groups.get((metadata["doc_id"], _label(item)))
                    if metadata.get("packet_type") != "complete_document" else None) or self._spans.get(key)
        if chunk_id is None:
            raise BackendError("invalid_response")
        fragment = self._fragments[chunk_id]
        # Applicability is evaluated per question, while the original text is stable.
        values = {**fragment, **{k: v for k, v in metadata.items()
                                if k in fragment and k not in {"char_start", "char_end"}}}
        values.pop("text", None)
        values["source_reference"] = _label(item)
        return values

    async def ask(self, question: Question) -> Answer:
        if not question.text.strip():
            return Answer("clarification", "Уточните, пожалуйста, ваш вопрос.", ())
        deadline = time.monotonic() + self.timeout
        async def execute():
            async with self._slots:
                raw = "\n".join(
                    ("Пользователь: " if turn.role == "user" else "Ассистент: ") + turn.content
                    for turn in question.history
                )
                raw = raw + "\nПользователь: " + question.text if raw else question.text
                prepared = self.engine.prepare(raw)
                stats = Counter()
                text = await answer_one(Model(self.client, deadline), prepared, stats)
                sources = []
                # The batch renderer emits exact human-readable references in parentheses.
                # Recover the same references for the clickable HTTP source contract.
                by_label = defaultdict(list)
                for item in prepared["evidence"]:
                    by_label[_label(item)].append(item)
                for label, candidates in sorted(by_label.items(), key=lambda pair: len(pair[0]), reverse=True):
                    pattern = re.compile(r"(?<=[(;])\s*" + re.escape(label) + r"(?=\s*[;)])")
                    if label and pattern.search(text):
                        numbers = []
                        seen = set()
                        for item in sorted(candidates, key=lambda x: x.get("metadata", {}).get("packet_type") == "complete_document"):
                            doc_id = item["metadata"]["doc_id"]
                            if doc_id in seen:
                                continue
                            seen.add(doc_id)
                            source = self._source(item)
                            sources.append(source)
                            numbers.append("[" + str(len(sources)) + "]")
                        text = pattern.sub("; ".join(numbers), text)
                text = re.sub(r"\((\[\d+\](?:;\s*\[\d+\])*)\)", r"\1", text)
                kind = "answer" if sources else "insufficient"
                if any(token in text.lower() for token in ("уточните", "требуется уточн", "недостаточно")):
                    kind = "partial" if sources else "clarification"
                LOG.info("contender sources=%s stats=%s", len(sources), dict(stats))
                return Answer(kind, text, tuple(sources))
        try:
            return await asyncio.wait_for(execute(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise BackendError("timeout") from None

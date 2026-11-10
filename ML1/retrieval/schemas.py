from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ParsedDocument:
    metadata: dict[str, Any]
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    text_with_context: str
    source_reference: str
    normalized_text: str
    metadata: dict[str, Any]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    text_with_context: str
    source_reference: str
    metadata: dict[str, Any]
    score: float
    score_details: dict[str, Any]
    citation_payload: dict[str, Any]
    neighboring_context: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

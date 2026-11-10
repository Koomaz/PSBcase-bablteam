from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from .bm25_index import BM25_B, BM25_K1, BM25Index
from .chunker import chunk_document
from .config import CANDIDATE_DEPTH, LOCAL_VECTOR_MODEL, MAX_CHARS, OVERLAP, RRF_K, SCHEMA_VERSION, Settings
from .document_parser import load_documents
from .embeddings import ModelAPI, ModelAPIError
from .schemas import Chunk
from .vector_index import LocalVectorIndex, build_dense


@dataclass
class IndexData:
    chunks: list[Chunk]
    documents: list[dict]
    bm25: BM25Index
    local: LocalVectorIndex
    dense: list[list[float]] | None
    manifest: dict


def build_indexes(source: str | Path, settings: Settings, *, dense: bool = True) -> IndexData:
    documents, issues = load_documents(source)
    chunks = []
    for document in documents:
        parts, found = chunk_document(document)
        chunks.extend(parts)
        issues.extend(found)
    if not chunks:
        raise ValueError("Knowledge base contains no searchable chunks")
    matrix, limitations = None, []
    if dense:
        try:
            matrix = build_dense(chunks, ModelAPI(settings))
        except ModelAPIError as exc:
            limitations.append(f"dense_index_unavailable:{exc}")
    else:
        limitations.append("dense_index_disabled")
    bm25, local = BM25Index.build(chunks), LocalVectorIndex.build(chunks)
    records = [doc.metadata for doc in documents]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_count": len(records), "chunk_count": len(chunks),
        "source_files": [{"file_path": doc["file_path"], "sha256": doc["sha256"]} for doc in records],
        "embedding_model": settings.embedding_model if matrix else None,
        "embedding_dimensions": len(matrix[0]) if matrix else None,
        "chunking": {"max_chars": MAX_CHARS, "overlap_chars": OVERLAP, "strategy": "legal_structure"},
        "bm25": {"k1": BM25_K1, "b": BM25_B},
        "vector_search": {"metric": "cosine", "local_model": LOCAL_VECTOR_MODEL, "local_dimensions": len(local.idf)},
        "hybrid_fusion": {"rrf_k": RRF_K, "candidate_depth": CANDIDATE_DEPTH},
        "reranker_used_at_build": False,
        "issues": issues, "known_limitations": limitations + ["law_snapshot_only", "txt_md_only"],
    }
    return IndexData(chunks, records, bm25, local, matrix, manifest)


def save_indexes(data: IndexData, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".build.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Index publication is already locked: {lock}") from None
    try:
        with TemporaryDirectory(prefix=".index-", dir=directory) as tmp:
            stage = Path(tmp)
            payloads = {
                "chunks.jsonl": "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in data.chunks),
                "documents.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data.documents),
                "bm25_index.json": json.dumps(data.bm25.dump(), ensure_ascii=False, separators=(",", ":")),
                "local_vector_index.json": json.dumps(data.local.dump(), ensure_ascii=False, separators=(",", ":")),
            }
            if data.dense is not None:
                payloads["embeddings.json"] = json.dumps(data.dense, separators=(",", ":"), allow_nan=False)
            checksums = {}
            for name, body in payloads.items():
                raw = body.encode("utf-8")
                (stage / name).write_bytes(raw)
                checksums[name] = hashlib.sha256(raw).hexdigest()
            manifest = {**data.manifest, "files": checksums}
            (stage / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
            for name in (*payloads, "index_manifest.json"):
                os.replace(stage / name, directory / name)
            if data.dense is None:
                (directory / "embeddings.json").unlink(missing_ok=True)
    finally:
        lock.rmdir()


def load_indexes(directory: str | Path) -> IndexData:
    directory = Path(directory)
    manifest_file = directory / "index_manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Index missing at {manifest_file}. Run python -m scripts.build_index.")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported schema version")

        def read_payload(name: str, jsonl: bool = False):
            raw = (directory / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
                raise ValueError(f"checksum mismatch: {name}")
            if jsonl:
                return [json.loads(line) for line in raw.splitlines()]
            return json.loads(raw)

        chunks = [Chunk(**row) for row in read_payload("chunks.jsonl", True)]
        docs = read_payload("documents.jsonl", True)
        bm25 = BM25Index(**read_payload("bm25_index.json"))
        local = LocalVectorIndex(**read_payload("local_vector_index.json"))
        dense = read_payload("embeddings.json") if manifest["embedding_model"] else None
        if len(chunks) != manifest["chunk_count"] or len(docs) != manifest["document_count"]:
            raise ValueError("invalid row counts")
        if len(bm25.lengths) != len(chunks) or (dense is not None and (
            len(dense) != len(chunks) or any(len(row) != manifest["embedding_dimensions"] for row in dense)
        )):
            raise ValueError("inconsistent index row counts or embedding dimensions")
        return IndexData(chunks, docs, bm25, local, dense, manifest)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Index incompatible or incomplete at {directory}. Rebuild with "
                         f"python -m scripts.build_index ({exc}).") from None

from __future__ import annotations

import copy
import os
import re
from collections import defaultdict
from pathlib import Path

from .artifacts import IndexData, build_indexes, load_indexes
from .bm25_index import lexical_terms
from .chunker import DIRECTIVE
from .config import CANDIDATE_DEPTH, Settings
from .diagnostics import retrieval_warnings
from .embeddings import ModelAPI, ModelAPIError
from .hybrid_search import (
    analyze_query, document_match, exact_reference, ordered, reciprocal_fusion, temporal_status,
)
from .metadata import canonical_number, normalize
from .reranker import rerank
from .schemas import RetrievedChunk
from .vector_index import dense_search


CITATION_FIELDS = (
    "doc_id", "doc_title", "doc_type", "file_name", "file_path", "doc_number", "doc_date",
    "edition_date", "effective_date", "validity_period", "snapshot_date", "version_status",
    "structure_path", "section", "article", "part", "clause", "subclause", "appendix", "amends_clause",
    "provision_effective_date", "amendment_application_date", "amendment_application_relation",
    "page_start", "page_end", "char_start", "char_end", "source_url",
    "parent_doc_id", "temporal_status",
)
FILTER_FIELDS = frozenset(CITATION_FIELDS) - {"temporal_status", "structure_path", "validity_period"}
FILTER_FIELDS |= {"law_code", "counterparties", "edition_group", "parent_doc_number"}
LEGAL_SCOPE = re.compile(
    r"\b(?:договор|закон|норматив|регламент|трудов|работник|налог|ндс|санкци|право|персональн|"
    r"соглашен|судеб|аренд|поставк|неустойк|архив|претензи|взыскан|обязательств)\w*\b", re.I
)


class Retriever:
    """Load or build once, then share a read-only index across independent requests."""

    def __init__(self, indexes: IndexData, settings: Settings):
        self._index = indexes
        self._api = ModelAPI(settings)
        self._settings = settings
        self._aliases = {
            normalize(name) for doc in indexes.documents for name in doc["counterparties"]
            if normalize(name) != "меридиан"
        }
        self._known_numbers = {
            canonical_number(doc["doc_number"]) for doc in indexes.documents if doc["doc_number"]
        }
        self._core_terms = [set(lexical_terms(chunk.text)) for chunk in indexes.chunks]

    @classmethod
    def from_artifacts(cls, directory: str | Path = "artifacts", *, settings: Settings | None = None) -> Retriever:
        return cls(load_indexes(directory), settings or Settings.from_env())

    @classmethod
    def from_source(cls, source: str | Path, *, settings: Settings | None = None, dense: bool = True) -> Retriever:
        settings = settings or Settings.from_env()
        return cls(build_indexes(source, settings, dense=dense), settings)

    @classmethod
    def from_data_dir(cls, directory: str | Path | None = None, *,
                      settings: Settings | None = None, dense: bool = True) -> Retriever:
        directory = Path(directory or os.getenv("DATA_DIR", "data"))
        for name in ("knowledge-base.zip", "knowledge_base.zip", "kb.zip"):
            source = directory / name
            if source.is_file():
                return cls.from_source(source, settings=settings, dense=dense)
        for name in ("knowledge-base", "knowledge_base"):
            source = directory / name
            if source.is_dir():
                return cls.from_source(source, settings=settings, dense=dense)
        raise FileNotFoundError(
            f"Knowledge base not found in {directory}. Expected knowledge-base.zip, "
            "knowledge_base.zip, kb.zip, knowledge-base/ or knowledge_base/."
        )

    @property
    def manifest(self) -> dict:
        return copy.deepcopy(self._index.manifest)

    def search(self, query: str, top_k: int = 5, history: list | None = None,
               filters: dict | None = None) -> list[RetrievedChunk]:
        return self.search_with_diagnostics(query, top_k, history, filters)[0]

    def search_with_diagnostics(self, query: str, top_k: int = 5, history: list | None = None,
                                filters: dict | None = None) -> tuple[list[RetrievedChunk], dict]:
        return self._search(query, top_k, history, filters, apply_reranker=True, debug=False)

    def retrieve_candidates(self, query: str, top_k: int = 30,
                            filters: dict | None = None) -> list[RetrievedChunk]:
        return self._search(query, top_k, None, filters, apply_reranker=False, debug=False)[0]

    def explain_retrieval(self, query: str, top_k: int = 5, history: list | None = None,
                          filters: dict | None = None) -> dict:
        final, diagnostics = self._search(query, top_k, history, filters, apply_reranker=True, debug=True)
        return {**diagnostics, "final_chunks": [chunk.to_dict() for chunk in final]}

    def _eligible(self, filters: dict | None) -> set[int]:
        if not filters:
            return set(range(len(self._index.chunks)))
        invalid = filters.keys() - FILTER_FIELDS
        if invalid:
            raise ValueError(f"Unsupported retrieval filters: {', '.join(sorted(invalid))}")

        def match(key, wanted, value) -> bool:
            expected = wanted if isinstance(wanted, (tuple, list, set)) else [wanted]
            actual = value if isinstance(value, list) else [value]
            if key in {"doc_number", "parent_doc_number"}:
                convert = canonical_number
            else:
                convert = normalize
            return any(
                convert(a) == convert(b) if isinstance(a, str) and isinstance(b, str) else a == b
                for a in expected for b in actual
            )

        return {i for i, chunk in enumerate(self._index.chunks)
                if all(match(key, value, chunk.metadata.get(key)) for key, value in filters.items())}

    def _search(self, query, top_k, history, filters, *, apply_reranker, debug):
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        index = self._index
        q = analyze_query(query, history, self._aliases)
        bm25, local, dense, fusion, reranked = {}, {}, {}, {}, {}
        warnings = []
        mode = "local_tfidf"
        eligible = self._eligible(filters)
        final = []
        directive = bool(DIRECTIVE.search(q.latest))
        has_legal_scope = bool(q.refs or q.numbers or q.aliases or q.codes or LEGAL_SCOPE.search(q.latest))
        if directive:
            warnings.append("query_prompt_directive_suspected")
        if not q.latest.strip():
            warnings.append("empty_query")
        elif top_k > 0 and eligible and (not directive or has_legal_scope):
            if q.numbers - self._known_numbers:
                warnings.append("document_identifier_not_found")
            bm25 = index.bm25.search(q.terms, eligible)
            local = index.local.search(q.latest, eligible)
            if index.dense is None:
                warnings.append("dense_index_unavailable")
            elif index.manifest["embedding_model"] != self._settings.embedding_model:
                warnings.append("embedding_model_mismatch")
            else:
                try:
                    dense = dense_search(index.dense, self._api.embed([q.contextual])[0], eligible)
                    mode = "bge_dense"
                except ModelAPIError as exc:
                    warnings.append(f"embedding_api_unavailable:{exc}")
            depth = max(CANDIDATE_DEPTH, top_k)
            best_bm25 = ordered(bm25)[:depth]
            best_vector = ordered(dense or local)[:depth]
            lexical_weight = 0.57 if dense else 0.72
            fusion = reciprocal_fusion([(best_bm25, lexical_weight), (best_vector, 1 - lexical_weight)])
            anchored = bool(q.numbers or q.aliases or q.codes)
            matching = {i for i in eligible if document_match(index.chunks[i].metadata, q)}
            addenda = {i for i in eligible if index.chunks[i].metadata["doc_type"] == "addendum"}
            exact = {
                i for i in (matching if anchored and not q.regulations else eligible)
                if exact_reference(index.chunks[i].metadata, q)
                and (anchored or not q.regulations or "article" in q.refs
                     or index.chunks[i].metadata["doc_type"] == "regulation")
            }
            if q.refs and not exact:
                warnings.append("exact_reference_not_found")
            dates, adjusted, coverage = {}, {}, {}
            has_specific_scope = bool(matching and not q.regulations)
            for i in fusion.keys() | exact:
                meta = index.chunks[i].metadata
                kind = meta["doc_type"]
                if has_specific_scope and i not in matching:
                    continue
                if anchored and not matching and not q.regulations:
                    continue
                if q.refs and i not in exact:
                    continue
                overlap = self._core_terms[i] & q.intent_terms
                coverage[i] = sum(
                    min(index.bm25.idf.get(term, 0), 5) * q.terms[term]
                    for term in overlap if term[0].isalpha() or term.startswith("~")
                )
                evidence = (
                    coverage[i] >= 2.7
                    or local.get(i, 0) >= (0.17 if i in matching else 0.25)
                    or dense.get(i, 0) >= 0.60
                )
                if i not in exact and not evidence:
                    continue
                status = dates[i] = temporal_status(meta, q)
                score = fusion.get(i, 0) * (0.65 + min(coverage[i], 22) / 32)
                if meta["section"] and "реквизиты" in normalize(meta["section"]) and "реквизиты" not in q.terms:
                    score *= 0.35
                if i in matching:
                    score *= 1.48
                if q.regulations and kind == "regulation":
                    score *= 1.36
                if q.regulations and kind == "law" and not q.codes and "article" not in q.refs:
                    score *= 0.52
                if q.addenda and i in addenda:
                    score *= 1.65
                if i in exact:
                    score += 1.8
                if status == "applicable" and meta["edition_group"]:
                    score *= 1.22
                if not q.comparative:
                    if status == "superseded":
                        score *= 0.20
                    elif status == "outside_period":
                        score *= 0.19
                    elif status == "historical_version_missing":
                        score *= 0.72
                adjusted[i] = score
            ranking = ordered(adjusted)
            if not q.comparative:
                applicable_groups = {
                    index.chunks[i].metadata["edition_group"] for i in ranking
                    if dates[i] in {"applicable", "undated"} and index.chunks[i].metadata["edition_group"]
                }
                ranking = [
                    i for i in ranking if not (
                        dates[i] in {"outside_period", "superseded"}
                        and index.chunks[i].metadata["edition_group"] in applicable_groups
                    )
                ]
            if apply_reranker and ranking and self._settings.reranker_top_n:
                if self._settings.reranker_model and self._settings.configured:
                    try:
                        head = ranking[:self._settings.reranker_top_n]
                        results = rerank(self._api, q.contextual, [index.chunks[i].text_with_context for i in head])
                        reranked = {head[i]: score for i, score in results.items()}
                        # Blend ranks within the head while keeping exact and temporal priorities.
                        bonus = reciprocal_fusion([(ordered(reranked), 0.34)])
                        for i in head:
                            adjusted[i] *= 1 + bonus[i]
                        ranking = sorted(head, key=lambda i: (-adjusted[i], i)) + ranking[len(head):]
                    except ModelAPIError as exc:
                        warnings.append(f"reranker_unavailable:{exc}")
                else:
                    warnings.append("reranker_not_configured")

            selected = []
            seen = defaultdict(int)
            best_score = adjusted[ranking[0]] if ranking else 0.0
            for i in ranking:
                if len(selected) == top_k or adjusted[i] < best_score * 0.23:
                    break
                row = index.chunks[i]
                meta = row.metadata
                key = (row.doc_id, meta["article"], meta["clause"], meta["part"], meta["subclause"], meta["section"])
                # Deduplicate overlapping windows while retaining independent paragraphs in the same article.
                if any(
                    row.doc_id == index.chunks[j].doc_id and meta["char_start"] < index.chunks[j].metadata["char_end"]
                    and index.chunks[j].metadata["char_start"] < meta["char_end"] for j in selected
                ):
                    continue
                if seen[key] >= max(3, top_k // 2) and i not in exact:
                    continue
                selected.append(i)
                seen[key] += 1
                final.append(self._result(i, q, adjusted[i], {
                    "bm25_score": bm25.get(i), "vector_score": (dense or local).get(i),
                    "local_vector_score": local.get(i), "fusion_score": fusion.get(i, 0.0),
                    "rerank_score": reranked.get(i), "exact_match": i in exact,
                    "document_match": i in matching, "term_evidence": coverage[i],
                }))

        if top_k:
            warnings.extend(retrieval_warnings(q, final, index.manifest["issues"]))
        if q.period and "exact_reference_not_found" in warnings:
            warnings.append("check_requested_edition_and_reference")
        diagnostics = {
            "query": query, "normalized_query": normalize(q.latest),
            "contextual_query": q.contextual, "event_period": list(q.period) if q.period else None,
            "mode": mode, "warnings": warnings,
            "confidence": "low" if not final or any(
                name in warnings for name in ("historical_or_date_coverage_unverified",
                                              "applicable_edition_unconfirmed", "document_identifier_not_found")
            ) else "strong_signal" if final[0].score_details["exact_match"] or
                final[0].score_details["term_evidence"] >= 7 else "review",
            "sources": [row.citation_payload for row in final],
        }
        if debug:
            for name, scores in (("bm25", bm25), ("vector", dense or local), ("fusion", fusion), ("rerank", reranked)):
                diagnostics[name] = [
                    {"chunk_id": index.chunks[i].chunk_id,
                     "source_reference": index.chunks[i].source_reference, "score": scores[i]}
                    for i in ordered(scores)[:max(top_k, CANDIDATE_DEPTH)]
                ]
            diagnostics["parse_issues"] = copy.deepcopy(index.manifest["issues"])
        return final, diagnostics

    def _result(self, i: int, q, score: float, scores: dict) -> RetrievedChunk:
        chunks = self._index.chunks
        chunk = chunks[i]
        meta = copy.deepcopy(chunk.metadata)
        meta["temporal_status"] = temporal_status(meta, q)
        neighbors = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(chunks):
                other = chunks[j]
                if other.doc_id == chunk.doc_id and all(other.metadata[key] == meta[key] for key in (
                    "article", "section", "appendix"
                )):
                    neighbors.append({
                        "chunk_id": other.chunk_id, "text": other.text,
                        "source_reference": other.source_reference,
                        "char_start": other.metadata["char_start"], "char_end": other.metadata["char_end"],
                        "temporal_status": temporal_status(other.metadata, q),
                    })
        citation = {key: copy.deepcopy(meta[key]) for key in CITATION_FIELDS}
        citation.update(chunk_id=chunk.chunk_id, source_reference=chunk.source_reference)
        return RetrievedChunk(
            chunk.chunk_id, chunk.text, chunk.text_with_context, chunk.source_reference,
            meta, score, scores, citation, neighbors,
        )

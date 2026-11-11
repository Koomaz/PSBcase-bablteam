from __future__ import annotations

import copy
import os
import re
from collections import defaultdict
from pathlib import Path

from .artifacts import IndexData, build_indexes, load_indexes
from .bm25_index import lexical_terms, stem, words
from .chunker import DIRECTIVE
from .config import CANDIDATE_DEPTH, Settings
from .diagnostics import retrieval_warnings
from .embeddings import ModelAPI, ModelAPIError
from .hybrid_search import (
    amendment_status, analyze_query, document_match, exact_reference, ordered, reciprocal_fusion, temporal_status,
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
    "parent_doc_id", "parent_doc_number", "law_code", "temporal_status", "document_temporal_status", "amendment_status",
)
FILTER_FIELDS = frozenset(CITATION_FIELDS) - {
    "temporal_status", "document_temporal_status", "amendment_status", "structure_path", "validity_period",
}
FILTER_FIELDS |= {"law_code", "counterparties", "edition_group", "parent_doc_number"}
LEGAL_SCOPE = re.compile(
    r"\b(?:договор|закон|норматив|регламент|трудов|работник|налог|ндс|санкци|право|персональн|"
    r"соглашен|судеб|аренд|поставк|неустойк|архив|претензи|взыскан|обязательств)\w*\b", re.I
)
CLAUSE_LINK = re.compile(r"(?<!\w)(?:п\.|пункт\w*)\s*(\d+(?:\.\d+)+)(?![\d.])", re.I)
DEFINITION = re.compile(r"\bдатой\s+([а-яё]+)\s+считается\b", re.I)


def resolve_kb_source(directory: str | Path) -> Path:
    directory = Path(directory)
    for name in ("knowledge-base.zip", "knowledge_base.zip", "kb.zip"):
        source = directory / name
        if source.is_file():
            return source
    for name in ("knowledge-base", "knowledge_base"):
        source = directory / name
        if source.is_dir():
            return source
    raise FileNotFoundError(
        f"Knowledge base not found in {directory}. Expected knowledge-base.zip, "
        "knowledge_base.zip, kb.zip, knowledge-base/ or knowledge_base/."
    )


class Retriever:
    """Load or build once, then share a read-only index across independent requests."""

    def __init__(self, indexes: IndexData, settings: Settings):
        self._index = indexes
        self._api = ModelAPI(settings)
        self._settings = settings
        self._alias_numbers = defaultdict(set)
        for doc in indexes.documents:
            for name in doc["counterparties"]:
                alias = normalize(name)
                if alias != "меридиан":
                    numbers = (doc["doc_number"], doc["parent_doc_number"])
                    self._alias_numbers[alias].update(canonical_number(number) for number in numbers if number)
        self._known_numbers = {
            canonical_number(doc["doc_number"]) for doc in indexes.documents if doc["doc_number"]
        }
        self._core_terms = [set(lexical_terms(chunk.text)) for chunk in indexes.chunks]
        self._amendments = defaultdict(list)
        self._provisions = defaultdict(list)
        self._documents = {meta["doc_id"]: meta for meta in indexes.documents}
        self._rows_by_doc = defaultdict(list)
        self._sections = defaultdict(list)
        self._definitions = defaultdict(list)
        for i, chunk in enumerate(indexes.chunks):
            meta = chunk.metadata
            self._rows_by_doc[chunk.doc_id].append(i)
            if meta["clause"] and meta["section"]:
                self._sections[chunk.doc_id, meta["section"]].append(i)
            for term in DEFINITION.findall(chunk.text):
                self._definitions[chunk.doc_id, normalize(term)].append(i)
            if meta["parent_doc_id"] and meta["amends_clause"]:
                self._amendments[meta["parent_doc_id"], meta["amends_clause"]].append(i)
            if meta["doc_type"] in {"contract", "regulation"} and meta["clause"]:
                self._provisions[chunk.doc_id, meta["clause"]].append(i)
        for rows in self._rows_by_doc.values():
            rows.sort(key=lambda i: indexes.chunks[i].metadata["char_start"])

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
        return cls.from_source(resolve_kb_source(directory), settings=settings, dense=dense)

    @property
    def manifest(self) -> dict:
        return copy.deepcopy(self._index.manifest)

    def document_registry(self) -> list[dict]:
        """Return source metadata and available locators without chunk text."""
        registry = []
        for doc_id, metadata in self._documents.items():
            rows = self._rows_by_doc[doc_id]
            clauses = dict.fromkeys(self._index.chunks[i].metadata["clause"] for i in rows)
            registry.append({
                **copy.deepcopy(metadata), "clauses": [clause for clause in clauses if clause],
                "chunk_count": len(rows),
            })
        return registry

    def document_chunks(self, doc_id: str) -> list[RetrievedChunk]:
        """Return the indexed source in original order, with no query or network call."""
        return [self._result(i, None, 0.0, {}) for i in self._rows_by_doc.get(doc_id, ())]

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
        q = analyze_query(query, history, self._alias_numbers)
        bm25, local, dense, fusion, reranked = {}, {}, {}, {}, {}
        warnings = []
        mode = "local_tfidf"
        eligible = self._eligible(filters)
        scope = self._resolve_scope(q, eligible)
        final = []
        directive = bool(DIRECTIVE.search(q.latest))
        has_legal_scope = bool(q.refs or q.numbers or q.aliases or q.codes or LEGAL_SCOPE.search(q.latest))
        unresolved_party = scope["conflicting_identifiers"] or (
            scope["ambiguous"] and not (q.numbers or q.aliases or q.codes or q.regulations)
        )
        if directive:
            warnings.append("query_prompt_directive_suspected")
        if scope["ambiguous"]:
            warnings.append("ambiguous_document_scope")
        if not q.latest.strip():
            warnings.append("empty_query")
        elif top_k > 0 and eligible and not unresolved_party and (not directive or has_legal_scope):
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
            if scope["doc_id"] and q.numbers and not q.codes:
                doc_id = scope["doc_id"]
                matching = {i for i in matching if index.chunks[i].doc_id == doc_id
                            or index.chunks[i].metadata["parent_doc_id"] == doc_id}
            addenda = {i for i in eligible if index.chunks[i].metadata["doc_type"] == "addendum"}
            exact = {
                i for i in (matching if anchored and not q.regulations else eligible)
                if exact_reference(index.chunks[i].metadata, q)
                and (anchored or not q.regulations or "article" in q.refs
                     or index.chunks[i].metadata["doc_type"] == "regulation")
            }
            if q.refs and not exact:
                warnings.append("exact_reference_not_found")
            if q.refs and not anchored and not q.regulations and len({
                index.chunks[i].doc_id for i in exact
            }) > 1:
                warnings.append("ambiguous_document_scope")
            dates, adjusted, coverage = {}, {}, {}
            has_specific_scope = bool(matching and not q.regulations)
            for i in fusion.keys() | exact:
                meta = index.chunks[i].metadata
                kind = meta["doc_type"]
                if has_specific_scope and i not in matching:
                    continue
                if anchored and not matching and not q.regulations:
                    continue
                if matching and i not in matching and kind in {"contract", "addendum"}:
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
                if self._core_terms[i] & q.operation_terms:
                    score *= 1.52
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
            self._pair_amendments(q, adjusted, coverage, dates, eligible)
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
                        ranking = self._rerank_head(head, ranking[len(head):], reranked, adjusted, dates, exact, q)
                    except ModelAPIError as exc:
                        warnings.append(f"reranker_unavailable:{exc}")
                else:
                    warnings.append("reranker_not_configured")

            selected = []
            seen = defaultdict(int)
            best_score = adjusted[ranking[0]] if ranking else 0.0
            ranking, supplemental = self._expand_ranked(ranking, q, eligible, adjusted, coverage, top_k)
            for i in ranking:
                if len(selected) == top_k:
                    break
                if adjusted[i] < best_score * 0.23 and i not in supplemental and not any(
                    i in self._related(j) for j in selected
                ):
                    continue
                row = index.chunks[i]
                meta = row.metadata
                key = (row.doc_id, meta["article"], meta["clause"], meta["part"], meta["subclause"], meta["section"])
                # Retain split windows that add new source text, even if they overlap at the boundary.
                if any(
                    row.doc_id == index.chunks[j].doc_id
                    and index.chunks[j].metadata["char_start"] <= meta["char_start"]
                    and meta["char_end"] <= index.chunks[j].metadata["char_end"] for j in selected
                ):
                    continue
                if seen[key] >= max(3, top_k // 2) and i not in exact:
                    continue
                selected.append(i)
                seen[key] += 1
                result = self._result(i, q, adjusted[i], {
                    "bm25_score": bm25.get(i), "vector_score": (dense or local).get(i),
                    "local_vector_score": local.get(i), "fusion_score": fusion.get(i, 0.0),
                    "rerank_score": reranked.get(i), "exact_match": i in exact,
                    "document_match": i in matching, "term_evidence": coverage[i],
                })
                if i in supplemental:
                    result.score_details["selection_reason"] = supplemental[i]
                if supplemental.get(i) == "comparison_edition":
                    for metadata in (result.metadata, result.citation_payload):
                        metadata["evidence_role"] = "comparison_only"
                final.append(result)

        if top_k:
            warnings.extend(retrieval_warnings(q, final, index.manifest["issues"]))
        if q.period and "exact_reference_not_found" in warnings:
            warnings.append("check_requested_edition_and_reference")
        diagnostics = {
            "query": query, "normalized_query": normalize(q.latest),
            "contextual_query": q.contextual, "event_period": list(q.period) if q.period else None,
            "user_context": q.user_context,
            "specification_period": list(q.specification_period) if q.specification_period else None,
            "resolved_scope": scope,
            "mode": mode, "reranker_mode": self._settings.reranker_mode if reranked else None, "warnings": warnings,
            "confidence": "low" if not final or any(
                name in warnings for name in ("historical_or_date_coverage_unverified",
                                              "applicable_edition_unconfirmed", "document_identifier_not_found",
                                              "ambiguous_document_scope")
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

    def _related(self, i: int) -> list[int]:
        meta = self._index.chunks[i].metadata
        if meta["parent_doc_id"] and meta["amends_clause"]:
            return self._provisions.get((meta["parent_doc_id"], meta["amends_clause"]), [])
        return self._amendments.get((meta["doc_id"], meta["clause"]), [])

    def _resolve_scope(self, q, eligible: set[int]) -> dict:
        doc_ids = {self._index.chunks[i].doc_id for i in eligible}
        available = [meta for doc_id, meta in self._documents.items() if doc_id in doc_ids]
        candidates = [meta for meta in available if document_match(meta, q)]
        ambiguous = False
        conflicting_identifiers = False
        if q.numbers and not q.codes:
            candidates = [meta for meta in candidates if meta["doc_number"]
                          and canonical_number(meta["doc_number"]) in q.numbers]
            if q.aliases and candidates:
                compatible = [meta for meta in candidates if q.aliases.intersection(
                    normalize(name) for name in meta["counterparties"]
                )]
                conflicting_identifiers = ambiguous = not compatible
                candidates = compatible or candidates
        elif q.aliases:
            primary = [meta for meta in candidates if meta["doc_type"] == "contract"]
            candidates = primary or candidates
        elif not candidates and "article" not in q.refs and re.search(
            r"\b(?:регламент|согласован|одобрен)\w*\b", normalize(q.contextual)
        ):
            candidates = [meta for meta in available if meta["doc_type"] == "regulation"]
            ambiguous = len({meta["edition_group"] for meta in candidates}) > 1
        elif not (q.numbers or q.codes):
            query_roots = {stem(word) for word in words(q.contextual)}
            partial = {alias for alias in self._alias_numbers
                       if stem(words(alias)[0]) in query_roots}
            if len(partial) > 1:
                candidates = [meta for meta in available if meta["doc_type"] == "contract"
                              and partial.intersection(normalize(name) for name in meta["counterparties"])]
                ambiguous = len(candidates) > 1
        if articles := q.refs.get("article"):
            candidates = [meta for meta in candidates if meta.get("article") in articles]
        applicable = [meta for meta in candidates if temporal_status(meta, q) in {"applicable", "undated", "snapshot"}]
        if len(applicable) > 1:
            ambiguous = True
        primary = applicable[0] if len(applicable) == 1 and not ambiguous else None
        comparison = [
            meta["doc_id"] for meta in candidates if primary and primary["edition_group"]
            and meta["edition_group"] == primary["edition_group"] and meta["doc_id"] != primary["doc_id"]
        ]
        return {
            "document_numbers": sorted(q.numbers), "counterparties": sorted(q.aliases), "law_codes": sorted(q.codes),
            "doc_id": primary["doc_id"] if primary else None, "doc_number": primary["doc_number"] if primary else None,
            "event_date": q.period[0] if q.period and q.period[0] == q.period[1] else None,
            "candidate_doc_ids": [meta["doc_id"] for meta in candidates],
            "applicable_doc_ids": [meta["doc_id"] for meta in applicable],
            "comparison_doc_ids": comparison, "ambiguous": ambiguous,
            "conflicting_identifiers": conflicting_identifiers,
        }

    def _expand_ranked(self, ranking, q, eligible, scores, coverage, top_k):
        ranking = self._prioritize_related(ranking, top_k)
        if top_k < 3 or not ranking:
            return ranking, {}
        chunks = self._index.chunks
        first = ranking[0]
        anchor = chunks[first]
        metadata = anchor.metadata
        additions = {}

        if metadata["doc_type"] in {"contract", "regulation"}:
            for clause in CLAUSE_LINK.findall(anchor.text):
                for i in self._provisions.get((anchor.doc_id, clause), ()):
                    additions.setdefault(i, "cross_reference")
            for term in re.findall(r"\b(?:с|от|до)\s+дат[ыа]\s+([а-яё]+)", anchor.text, re.I):
                for i in self._definitions.get((anchor.doc_id, normalize(term)), ()):
                    additions.setdefault(i, "term_definition")
        if q.period and not q.edition_period and not q.comparative and metadata["edition_group"] and metadata["clause"]:
            if temporal_status(metadata, q) == "applicable" and metadata["version_status"] != "current":
                for doc in self._documents.values():
                    if doc["edition_group"] == metadata["edition_group"] and doc["version_status"] == "current":
                        for i in self._provisions.get((doc["doc_id"], metadata["clause"]), ()):
                            additions.setdefault(i, "comparison_edition")
        if not q.refs and len(ranking) > 1 and metadata["doc_type"] == "contract":
            second = chunks[ranking[1]]
            if second.doc_id == anchor.doc_id and second.metadata["section"] == metadata["section"]:
                for i in self._sections.get((anchor.doc_id, metadata["section"]), ()):
                    if i not in ranking[:2]:
                        additions.setdefault(i, "same_section")
                        break
        head = ranking[:2]
        supplemental = {i: reason for i, reason in additions.items() if i in eligible and i not in head}
        supplemental = dict(list(supplemental.items())[:2])
        for i in supplemental:
            scores.setdefault(i, scores[first] * 0.5)
            coverage.setdefault(i, 0.0)
        return head + list(supplemental) + [i for i in ranking[2:] if i not in supplemental], supplemental

    def _amendment_status(self, i: int, q) -> str | None:
        meta = self._index.chunks[i].metadata
        if meta["doc_type"] == "addendum":
            return amendment_status(meta, q)
        states = {
            amendment_status(self._index.chunks[j].metadata, q) for j in self._related(i)
        }
        if states == {"amendment_applicable"}:
            return "superseded_by_amendment"
        return None

    def _pair_amendments(self, q, scores: dict, coverage: dict, dates: dict, eligible: set[int]) -> None:
        for i in ordered(scores):
            for j in self._related(i):
                if j in eligible and j not in scores:
                    scores[j] = scores[i] * 0.86
                    coverage[j] = 0.0
                    dates[j] = temporal_status(self._index.chunks[j].metadata, q)
        for i in scores:
            status = self._amendment_status(i, q)
            if status in {"amendment_not_yet_applicable", "superseded_by_amendment"}:
                scores[i] *= 0.58
            if status == "amendment_applicable":
                scores[i] *= 1.30
            if any(self._amendment_status(j, q) == "amendment_not_yet_applicable" for j in self._related(i)):
                scores[i] *= 1.25
        if not q.comparative:
            for i in scores:
                if self._amendment_status(i, q) not in {"amendment_not_yet_applicable", "superseded_by_amendment"}:
                    continue
                partner_score = max((scores[j] for j in self._related(i) if j in scores), default=None)
                if partner_score is not None:
                    scores[i] = min(scores[i], partner_score * 0.85)

    def _prioritize_related(self, ranking: list[int], top_k: int) -> list[int]:
        if top_k < 2 or not ranking:
            return ranking
        first, *tail = ranking
        related = set(self._related(first))
        return [first, *(i for i in tail if i in related), *(i for i in tail if i not in related)]

    def _rerank_head(self, head, tail, reranked, adjusted, dates, exact, q) -> list[int]:
        mode = self._settings.reranker_mode
        semantic = ordered(reranked)
        if mode == "legacy":
            bonus = reciprocal_fusion([(semantic, 0.34)])
            for i in head:
                adjusted[i] *= 1 + bonus[i]
            return ordered({i: adjusted[i] for i in head}) + tail

        rank_score = {
            i: 1 - rank / max(1, len(head) - 1) for rank, i in enumerate(semantic)
        }
        weight = 1.0 if mode == "direct" else 0.60
        scale = adjusted[head[0]]
        blended = {i: weight * rank_score[i] + (1 - weight) * adjusted[i] / scale for i in head}

        def priority(i):
            outside = dates[i] in {"superseded", "outside_period", "historical_version_missing"}
            amendment = self._amendment_status(i, q)
            background = amendment in {"amendment_not_yet_applicable", "superseded_by_amendment"}
            return (not q.comparative and (outside or background), i not in exact, -blended[i], i)

        ordered_head = sorted(head, key=priority)
        for i in ordered_head:
            adjusted[i] = scale * (0.70 + 0.30 * blended[i])
        return ordered_head + tail

    def _source(self, i, q) -> tuple[dict, dict]:
        chunk = self._index.chunks[i]
        meta = copy.deepcopy(chunk.metadata)
        meta["document_temporal_status"] = temporal_status(meta, q) if q is not None else None
        meta["amendment_status"] = self._amendment_status(i, q) if q is not None else None
        meta["temporal_status"] = meta["document_temporal_status"]
        if meta["amendment_status"] and meta["temporal_status"] not in {"outside_period", "superseded"}:
            meta["temporal_status"] = meta["amendment_status"]
        citation = {key: copy.deepcopy(meta.get(key)) for key in CITATION_FIELDS}
        citation.update(chunk_id=chunk.chunk_id, source_reference=chunk.source_reference)
        return meta, citation

    def _result(self, i: int, q, score: float, scores: dict) -> RetrievedChunk:
        chunks = self._index.chunks
        chunk = chunks[i]
        meta, citation = self._source(i, q)
        neighbors = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(chunks):
                other = chunks[j]
                if other.doc_id == chunk.doc_id and all(other.metadata[key] == meta[key] for key in (
                    "article", "section", "appendix"
                )):
                    neighbor_meta, neighbor_citation = self._source(j, q)
                    neighbors.append({
                        "chunk_id": other.chunk_id, "text": other.text,
                        "source_reference": other.source_reference,
                        "char_start": other.metadata["char_start"], "char_end": other.metadata["char_end"],
                        "temporal_status": neighbor_meta["temporal_status"],
                        "metadata": neighbor_meta, "citation_payload": neighbor_citation,
                    })
        return RetrievedChunk(
            chunk.chunk_id, chunk.text, chunk.text_with_context, chunk.source_reference,
            meta, score, scores, citation, neighbors,
        )

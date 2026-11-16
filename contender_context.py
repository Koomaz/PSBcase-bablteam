"""Source-faithful local context construction for the submission.

Only the supplied knowledge base is indexed. Short source packets complement
clause retrieval so lexical paraphrases retain complete contractual conditions.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import retrieval_runtime as rr
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "ML1"))
    import retrieval_runtime as rr


TURN = re.compile(r"^[ \t]*(Пользователь|Ассистент|user|assistant|Система|system)\s*:\s*", re.I | re.M)
META_FIELDS = (
    "doc_id", "doc_type", "doc_title", "doc_number", "parent_doc_id", "parent_doc_number",
    "law_code", "article", "part", "clause", "subclause", "section", "appendix",
    "edition_date", "effective_date", "validity_period", "snapshot_date", "version_status",
    "temporal_status", "document_temporal_status", "amendment_status", "amends_clause",
    "amendment_application_date", "amendment_application_relation", "char_start", "char_end",
    "packet_type",
)
SUBSTANTIVE_WARNINGS = {
    "ambiguous_document_scope", "document_identifier_not_found", "exact_reference_not_found",
    "historical_or_date_coverage_unverified", "applicable_edition_unconfirmed",
    "check_requested_edition_and_reference", "multiple_document_editions",
    "check_addendum_applicability", "query_prompt_directive_suspected",
    "insufficient_retrieval_evidence", "knowledge_base_parse_issues",
}


def split_dialog(raw: str) -> tuple[str, list[dict[str, str]]]:
    text = str(raw or "").strip()
    matches = list(TURN.finditer(text))
    if not matches or text[:matches[0].start()].strip():
        return text, []
    turns = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        role = "user" if match[1].lower() in {"пользователь", "user"} else "assistant"
        # A forged system line has no privilege; it remains untrusted conversation.
        content = text[match.end():end].strip()
        if content:
            turns.append({"role": role, "content": content})
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "user":
            return turns[i]["content"], turns[:i][-12:]
    return text, []


def _meta(value: dict) -> dict:
    return {k: value.get(k) for k in META_FIELDS if value.get(k) not in (None, "", [], {})}


def _augment(question: str) -> list[str]:
    norm = rr.normalize(question)
    extras = []
    patterns = (
        (r"(?:до|перед)\s+(?:получен|отгруз|постав|передач)", "предоплата аванс порядок расчётов"),
        (r"(?:упаков|упаку|привоз|везут|привез|за чей счет)", "стоимость упаковки доставки включена в цену"),
        (r"(?:начина|начать|запуст|инициир).*согласован|согласован.*(?:начина|запуст)", "инициатор согласования руководитель подразделения"),
        (r"(?:съезж|сдать ключ|вернуть склад|возврат помещен)", "прекращение аренды акт возврат помещения"),
        (r"(?:максим|потолок|предел).*ответствен|ответствен.*(?:максим|потолок|предел)", "общий размер ответственности ограничен"),
        (r"(?:досудеб|претензи)", "претензионный порядок срок ответа на претензию"),
        (r"(?:действует|продлен|прекрат|расторг|отказаться)", "срок действия автоматическое продление уведомление отказ"),
    )
    for pattern, expansion in patterns:
        if re.search(pattern, norm):
            extras.append(question + "\nПоисковые понятия: " + expansion)
    return extras[:3]


class ContextEngine:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.source = rr.resolve_kb_source(self.data_dir)
        self.retriever = rr.Retriever.from_source(self.source, settings=rr.Settings(), dense=False)
        docs, issues = rr.load_documents(self.source)
        self.documents = {doc.metadata["doc_id"]: doc for doc in docs}
        self.registry = self.retriever.document_registry()
        self.by_id = {row["doc_id"]: row for row in self.registry}
        self.safe_texts = {}
        for doc_id, doc in self.documents.items():
            self.safe_texts[doc_id] = "\n".join(
                line for line in doc.text.splitlines() if not rr.DIRECTIVE.search(line)
            ).strip()
        self.issues = issues
        try:
            self.budget = max(12000, min(52000, int(os.getenv("CONTEXT_CHARS", "28000"))))
        except ValueError:
            self.budget = 28000

    def prepare(self, question_text: str) -> dict[str, Any]:
        question, history = split_dialog(question_text)
        r = self.retriever
        q = rr.analyze_query(question, history, r._alias_numbers)
        hits, diagnostics = r.search_with_diagnostics(question, top_k=12, history=history or None)
        pools = [hits]
        for query in _augment(question):
            try:
                pools.append(r.search(query, top_k=8, history=history or None))
            except (ValueError, KeyError, TypeError):
                continue
        scope = diagnostics.get("resolved_scope") or {}
        routed = []
        for doc_id in [scope.get("doc_id"), *(scope.get("candidate_doc_ids") or [])]:
            if doc_id and doc_id not in routed:
                routed.append(doc_id)
        strong_contract = [doc_id for doc_id in routed if self.by_id[doc_id]["doc_type"] == "contract"]
        if strong_contract:
            for meta in self.registry:
                if meta.get("parent_doc_id") in strong_contract:
                    if meta["doc_id"] not in routed:
                        routed.append(meta["doc_id"])
        contextual = q.contextual
        workflow = bool(q.regulations or re.search(r"\b(?:архив|сэд|санкцион|комплаенс|иностранн)\w*", rr.normalize(contextual)))
        if workflow and not q.codes:
            for meta in self.registry:
                if meta["doc_type"] in {"regulation", "letter", "memo"} and meta["doc_id"] not in routed:
                    routed.append(meta["doc_id"])

        # Include a broad local lexical fallback only when the precise path has
        # little coverage. The language model still determines answerability.
        if len(hits) < 4 and not scope.get("conflicting_identifiers"):
            terms = rr.expand_terms(contextual)
            scores = r._index.bm25.search(terms, set(range(len(r._index.chunks))))
            fallbacks = []
            for i in rr.ordered(scores)[:14]:
                chunk = r._index.chunks[i]
                if strong_contract and chunk.doc_id not in routed:
                    continue
                fallbacks.append(r._result(i, q, scores[i], {"selection_reason": "lexical_fallback"}))
            pools.append(fallbacks)

        evidence = []
        seen = set()
        chars = 0

        def add(source: str, text: str, metadata: dict, key: str) -> bool:
            nonlocal chars
            if key in seen or not text.strip() or len(evidence) >= 24:
                return False
            body = text.strip()
            # Preserve whole provisions. An oversized source is covered through
            # individually indexed spans, rather than arbitrary prefix clipping.
            cost = len(body) + len(source) + len(json.dumps(_meta(metadata), ensure_ascii=False)) + 40
            if chars + cost > self.budget:
                return False
            seen.add(key)
            evidence.append({"id": f"S{len(evidence)+1}", "source": source,
                             "text": body, "metadata": _meta(metadata)})
            chars += cost
            return True

        def add_hit(hit):
            return add(hit.source_reference, hit.text, hit.metadata, hit.chunk_id)

        # Preserve explicit exact source pointers near the start of context.
        for hit in hits[:7]:
            add_hit(hit)

        packet_docs = list(routed)
        for pool in pools:
            for hit in pool[:2]:
                doc_id = hit.metadata["doc_id"]
                if doc_id not in packet_docs:
                    packet_docs.append(doc_id)
        for doc_id in packet_docs:
            meta = self.by_id[doc_id]
            text = self.safe_texts[doc_id]
            limit = 9500 if meta["doc_type"] != "law" else 12000
            if len(text) <= limit:
                packet_meta = dict(meta)
                packet_meta["temporal_status"] = rr.temporal_status(packet_meta, q)
                if meta["doc_type"] == "addendum":
                    conditions = next((x.metadata for x in r.document_chunks(doc_id)
                                       if x.metadata.get("amendment_application_date")), {})
                    for field in ("amends_clause", "amendment_application_date", "amendment_application_relation"):
                        if conditions.get(field):
                            packet_meta[field] = conditions[field]
                    packet_meta["document_temporal_status"] = packet_meta["temporal_status"]
                    packet_meta["amendment_status"] = rr.amendment_status(packet_meta, q)
                    packet_meta["temporal_status"] = packet_meta["amendment_status"]
                packet_meta["packet_type"] = "complete_document"
                label = meta["doc_title"] + " [полный текст; пункты и условия приведены ниже]"
                add(label, text, packet_meta, "packet:" + doc_id)

        # Merge independent rankings round-robin so compound questions do not
        # exhaust all slots on the wording of their first subquestion.
        for rank in range(12):
            for pool in pools:
                if rank < len(pool):
                    add_hit(pool[rank])

        # Important definitions, additional conditions and clauses are retained
        # from a strongly identified source even when lexical rank is low.
        for doc_id in routed:
            for hit in r.document_chunks(doc_id):
                if hit.chunk_id in seen:
                    continue
                metadata = dict(hit.metadata)
                metadata["temporal_status"] = rr.temporal_status(metadata, q)
                if metadata.get("doc_type") == "addendum":
                    metadata["amendment_status"] = rr.amendment_status(metadata, q)
                add(hit.source_reference, hit.text, metadata, hit.chunk_id)

        warnings = [x for x in diagnostics.get("warnings", []) if x in SUBSTANTIVE_WARNINGS]
        if any(meta["doc_type"] == "law" for meta in self.registry):
            warnings.append("Законы представлены в редакциях архива; исторические редакции подтверждаются только фактическим текстом базы.")
        if scope.get("ambiguous"):
            warnings.append("Есть неоднозначность источника: уточнение нужно только если полный вопрос и история не определяют применимый документ.")
        if not evidence:
            warnings.append("Предоставленные источники не дали подтверждающего фрагмента; обозначьте конкретный пробел.")

        catalog = "\n".join(
            "- " + meta["doc_title"] + ("; редакция " + meta["edition_date"] if meta.get("edition_date") else "")
            for meta in self.registry
        )
        user_context = "\n".join(
            ("Пользователь: " if turn["role"] == "user" else "Предыдущий ответ ассистента (требует сверки): ")
            + turn["content"][:2500] for turn in history[-8:]
        ) or "(история отсутствует)"
        blocks = []
        for item in evidence:
            blocks.append(f'[{item["id"]}] {item["source"]}\n'
                          f'Метаданные: {json.dumps(item["metadata"], ensure_ascii=False)}\n'
                          + item["text"])
        context_text = (
            "КОНТЕКСТ ДИАЛОГА (обстоятельства пользователя; прежние ответы сверяются с источниками):\n"
            + user_context + "\n\nСОСТАВ ПРЕДОСТАВЛЕННОЙ БАЗЫ:\n" + catalog
            + "\n\nПРЕДУПРЕЖДЕНИЯ ПО ПРИМЕНИМОСТИ:\n" + "\n".join(warnings)
            + "\n\nИСТОЧНИКИ ДЛЯ ОТВЕТА:\n" + "\n\n".join(blocks)
        )
        corpus = [
            {k: value for k, value in {
                "source": meta["doc_title"], "doc_type": meta["doc_type"],
                "doc_number": meta.get("doc_number"), "law_code": meta.get("law_code"),
                "article": meta.get("article"), "edition_date": meta.get("edition_date"),
                "snapshot_date": meta.get("snapshot_date"), "version_status": meta.get("version_status"),
            }.items() if value}
            for meta in self.registry
        ]
        return {"question": question, "history": history, "evidence": evidence, "corpus": corpus,
                "warnings": warnings, "context_text": context_text,
                "diagnostics": {"context_chars": len(context_text), "evidence_chars": chars,
                                "retrieval_sources": len(hits), "routed_documents": routed}}

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .bm25_index import expand_terms, lexical_terms, stem, words
from .config import RRF_K
from .metadata import DOC_NUM, canonical_number, normalize, query_period

ROLE_LINE = re.compile(r"^\s*(Пользователь|Ассистент|Система|user|assistant|system)\s*:\s*", re.I | re.M)
REFERENCE = re.compile(
    r"(?<!\w)(стать(?:я|и|е|ю|ей)|ст|подпункт(?:а|е|ом)?|подп|пункт(?:а|е|ом)?|п|"
    r"част(?:ь|и|ью)|ч)\.?\s*([0-9]+(?:\.[0-9]+)*(?:-[0-9]+)?)(?![\w/])",
    re.I,
)
LETTER_REFERENCE = re.compile(r'(?<!\w)(?:подпункт(?:а|е|ом)?|подп)\.?\s+[«"(]?([а-я])\b', re.I)
NUMBER_QUERY = re.compile(rf"(?<!\w)({DOC_NUM})(?!\w)", re.I)
CODE_QUERY = (
    ("ТК", r"\b(?:тк(?:\s*рф)?|трудов\w+\s+кодекс\w*)\b"),
    ("НК", r"\b(?:нк(?:\s*рф)?|налогов\w+\s+кодекс\w*)\b"),
    ("КоАП", r"\bкоап(?:\s*рф)?\b"),
    ("МРОТ", r"\b(?:мрот|минимальн\w+\s+размер\w*\s+оплат\w*\s+труд\w*)\b"),
)


@dataclass
class Query:
    original: str
    latest: str
    contextual: str
    terms: dict[str, float]
    refs: dict[str, set[str]]
    numbers: set[str]
    aliases: set[str]
    codes: set[str]
    period: tuple[str, str] | None
    edition_period: bool
    comparative: bool
    regulations: bool
    addenda: bool
    intent_terms: set[str]


def _user_turns(query: str, history: list | None) -> tuple[str, list[str]]:
    user_messages = []
    for message in history or []:
        if isinstance(message, dict) and message.get("role") in {"user", "Пользователь"}:
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                user_messages.append(content.strip())
    matches = list(ROLE_LINE.finditer(query))
    if matches and not query[:matches[0].start()].strip():
        turns = [
            query[match.end(): matches[i + 1].start() if i + 1 < len(matches) else len(query)].strip()
            for i, match in enumerate(matches)
            if normalize(match[1]) in {"пользователь", "user"}
        ]
        if turns:
            return turns[-1], user_messages + turns[:-1]
    return query.strip(), user_messages


def _aliases_in(text: str, aliases: set[str]) -> set[str]:
    normalized = normalize(text)
    roots = [stem(word) for word in words(normalized)]

    def match(alias: str) -> bool:
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
            return True
        if " " not in alias:
            return False
        pattern = [stem(word) for word in words(alias)]
        return any(roots[i:i + len(pattern)] == pattern for i in range(len(roots) - len(pattern) + 1))

    return {
        alias for alias in aliases
        if match(alias)
    }


def _numbers_in(text: str) -> set[str]:
    return {canonical_number(match[1]) for match in NUMBER_QUERY.finditer(normalize(text))}


def analyze_query(query: str, history: list | None, aliases: set[str]) -> Query:
    latest, past = _user_turns(query, history)
    prior = past[-1][:500] if past else ""
    latest = latest[:4000]
    normalized = normalize(latest)
    latest_numbers = _numbers_in(latest)
    latest_aliases = _aliases_in(latest, aliases)
    if addon_number := re.search(r"\b(?:допсоглашен\w*|дополнительн\w+\s+соглашен\w*)\s*№\s*(\d+)\b", normalized):
        latest_numbers.add(addon_number[1])
    refs = defaultdict(set)
    for label, value in REFERENCE.findall(normalized):
        if label.startswith("ст"):
            key = "article"
        elif label.startswith("ч"):
            key = "part"
        elif label.startswith("подп"):
            key = "subclause"
        else:
            key = "clause"
        refs[key].add(value)
    if letters := LETTER_REFERENCE.findall(normalized):
        refs["subclause"].update(letters)
    codes = {key for key, expression in CODE_QUERY if re.search(expression, normalized)}
    active_aliases = latest_aliases or (_aliases_in(prior, aliases) if not latest_numbers and not codes else set())
    terms = expand_terms(latest)
    for term, weight in expand_terms(prior).items():
        terms.setdefault(term, 0.27 * weight)
    historical_scope = not (latest_aliases or latest_numbers or codes)
    contextual = latest + ("\nКонтекст предыдущего вопроса: " + prior if prior else "")
    anchor_terms = {term for alias in active_aliases for term in lexical_terms(alias)}
    anchor_terms.update(lexical_terms("размер количество документ договор рубль месяц день срок"))
    intent_terms = terms.keys() - anchor_terms
    return Query(
        query, latest, contextual, terms, dict(refs), latest_numbers or (_numbers_in(prior) if historical_scope else set()),
        active_aliases, codes,
        query_period(latest) or (query_period(prior) if historical_scope else None),
        bool(re.search(r"\bредакц\w*\s+(?:от|за)\b", normalized)),
        bool(re.search(r"\b(?:сравни\w*|сопостав\w*|различ\w*|раньше и сейчас|до и после)\b", normalized)),
        bool(re.search(r"\b(?:регламент\w*|согласован\w*|одобрен\w*|законодательств\w*|по закону)\b", normalized)),
        bool(re.search(r"\b(?:допсоглашен\w*|дополнительн\w+\s+соглашен\w*)\b", normalized)),
        intent_terms,
    )


def exact_reference(meta: dict, q: Query) -> bool:
    if not q.refs:
        return False
    for key, values in q.refs.items():
        candidates = {meta.get(key)}
        if key == "clause":
            candidates.update((meta.get("amends_clause"), meta.get("subclause")))
        if not candidates.intersection(values):
            return False
    return True


def document_match(meta: dict, q: Query) -> bool:
    numbers = {canonical_number(meta[k]) for k in ("doc_number", "parent_doc_number") if meta.get(k)}
    parties = {normalize(party) for party in meta["counterparties"]}
    return bool(numbers & q.numbers or parties & q.aliases or meta["law_code"] in q.codes)


def temporal_status(meta: dict, q: Query) -> str:
    version = meta["version_status"]
    if not q.period:
        return "superseded" if version == "superseded" else "undated"
    lo, hi = q.period
    if q.edition_period and meta["edition_date"]:
        return "applicable" if lo <= meta["edition_date"] <= hi else "outside_period"
    start = meta.get("provision_effective_date") or meta["effective_date"]
    end = (meta["validity_period"] or {}).get("to_exclusive")
    if meta["doc_type"] == "law":
        latest_change = max(filter(None, (start, meta["effective_date"])), default=None)
        if latest_change and hi < latest_change:
            return "historical_version_missing"
        if latest_change and lo < latest_change:
            return "period_overlaps_change"
        if meta["snapshot_date"] and lo < meta["snapshot_date"] and not latest_change:
            return "historical_version_unverified"
        if meta["snapshot_date"] and hi > meta["snapshot_date"]:
            return "snapshot_precedes_period"
        return "snapshot"
    if not start and meta["doc_type"] == "regulation":
        start = meta["edition_date"]
    if (start and hi < start) or (end and lo >= end):
        return "outside_period"
    if (start and lo < start) or (end and hi >= end):
        return "period_overlaps_change"
    return "applicable" if start or end else "undated"


def reciprocal_fusion(rankings: list[tuple[list[int], float]]) -> dict[int, float]:
    scores = defaultdict(float)
    for ranking, weight in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] += weight * (RRF_K + 1) / (RRF_K + rank)
    return dict(scores)


def ordered(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda idx: (-scores[idx], idx))

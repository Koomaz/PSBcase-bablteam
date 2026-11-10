from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .bm25_index import expand_terms, intent_vocabulary, lexical_terms, stem, words
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
CORRECT_SCOPE = re.compile(r"(?:^|[\s,])не\s+[^.!?\n]+?,\s*а\s+", re.I)
SPECIFICATION = re.compile(r"\bспецификац\w*\b", re.I)
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
    user_context: str
    terms: dict[str, float]
    refs: dict[str, set[str]]
    numbers: set[str]
    aliases: set[str]
    codes: set[str]
    period: tuple[str, str] | None
    specification_period: tuple[str, str] | None
    edition_period: bool
    comparative: bool
    regulations: bool
    addenda: bool
    intent_terms: set[str]
    operation_terms: set[str]


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
            query = turns[-1]
            user_messages.extend(turns[:-1])
    latest = query.strip()
    # Some callers append the current turn before passing the conversation.
    while user_messages and normalize(user_messages[-1]) == normalize(latest):
        user_messages.pop()
    return latest, user_messages[-5:]


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


def _scope(text: str, aliases: set[str]) -> tuple[set[str], set[str], set[str]]:
    normalized = normalize(text)
    if correction := CORRECT_SCOPE.search(normalized):
        corrected = _scope(normalized[correction.end():], aliases)
        if any(corrected):
            return corrected
    numbers = _numbers_in(normalized)
    if addon_number := re.search(r"\b(?:допсоглашен\w*|дополнительн\w+\s+соглашен\w*)\s*№\s*(\d+)\b", normalized):
        numbers.add(addon_number[1])
    codes = {key for key, expression in CODE_QUERY if re.search(expression, normalized)}
    return numbers, _aliases_in(normalized, aliases), codes


def _specification_period(text: str) -> tuple[str, str] | None:
    periods = set()
    for sentence in re.split(r"[,;!?]|\.(?=\s|$)|\n", normalize(text)):
        if not (match := SPECIFICATION.search(sentence)):
            continue
        before, after = sentence[:match.start()], sentence[match.end():]
        # A dated shipment mentioned alongside a specification is a different event.
        if not re.search(r"\b(?:постав|достав|отгруз)\w*", after):
            if period := query_period(after.strip()):
                periods.add(period)
                continue
        if re.search(r"\b(?:подпис|соглас|оформ)\w*", before):
            if period := query_period(before):
                periods.add(period)
    return next(iter(periods)) if len(periods) == 1 else None


def _compatible_scopes(scope, prior, alias_numbers: dict[str, set[str]]) -> bool:
    if any(left and right and not left & right for left, right in zip(scope, prior)):
        return False
    if any(left & right for left, right in zip(scope, prior)):
        return True
    return any(
        number in alias_numbers.get(alias, ())
        for numbers, aliases in ((scope[0], prior[1]), (prior[0], scope[1]))
        for number in numbers for alias in aliases
    )


def analyze_query(query: str, history: list | None, alias_numbers: dict[str, set[str]]) -> Query:
    aliases = set(alias_numbers)
    latest, past = _user_turns(query, history)
    latest = latest[:4000]
    normalized = normalize(latest)
    scope = _scope(latest, aliases)
    context_turns = []
    for turn in reversed(past):
        turn = turn[:500]
        previous_scope = _scope(turn, aliases)
        if any(previous_scope):
            if any(scope) and not _compatible_scopes(scope, previous_scope, alias_numbers):
                break
            if not any(scope):
                scope = previous_scope
        context_turns.append(turn)
    numbers, active_aliases, codes = scope
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
    terms = expand_terms(latest)
    for distance, turn in enumerate(context_turns, 1):
        for term, weight in expand_terms(turn).items():
            terms.setdefault(term, 0.27 * weight / distance)
    date_turn = next((turn for turn in [latest, *context_turns] if query_period(turn)), latest)
    period = query_period(date_turn)
    specification_period = _specification_period(date_turn)
    if correction := CORRECT_SCOPE.search(date_turn):
        if corrected_period := query_period(date_turn[correction.end():]):
            period = corrected_period
            if specification_period or SPECIFICATION.search(date_turn[:correction.end()]):
                specification_period = corrected_period
    user_context = "\n".join(reversed(context_turns))
    contextual = latest + ("\nКонтекст предыдущих вопросов: " + user_context if user_context else "")
    anchor_terms = {term for alias in active_aliases for term in lexical_terms(alias)}
    anchor_terms.update(lexical_terms("размер количество документ договор рубль месяц день срок"))
    intent_terms = terms.keys() - anchor_terms
    return Query(
        query, latest, contextual, user_context, terms, dict(refs), numbers, active_aliases, codes,
        period, specification_period,
        bool(re.search(r"\bредакц\w*\s+(?:от|за)\b", normalized)),
        bool(re.search(r"\b(?:сравни\w*|сопостав\w*|различ\w*|раньше и сейчас|до и после)\b", normalized)),
        bool(re.search(r"\b(?:регламент\w*|согласован\w*|одобрен\w*|законодательств\w*|по закону)\b", normalized)),
        bool(re.search(r"\b(?:допсоглашен\w*|дополнительн\w+\s+соглашен\w*)\b", normalized)),
        intent_terms, {feature for terms in intent_vocabulary(latest)
                       for feature in lexical_terms(terms.split()[0])},
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


def amendment_status(meta: dict, q: Query) -> str | None:
    threshold = meta.get("amendment_application_date")
    if not threshold:
        return None
    if not q.specification_period:
        return "amendment_trigger_unverified"
    lo, hi = q.specification_period
    strict = meta.get("amendment_application_relation") == "после"
    if hi < threshold or (strict and hi == threshold):
        return "amendment_not_yet_applicable"
    if lo < threshold or (strict and lo == threshold):
        return "amendment_period_overlaps_change"
    return "amendment_applicable"


def reciprocal_fusion(rankings: list[tuple[list[int], float]]) -> dict[int, float]:
    scores = defaultdict(float)
    for ranking, weight in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] += weight * (RRF_K + 1) / (RRF_K + rank)
    return dict(scores)


def ordered(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda idx: (-scores[idx], idx))

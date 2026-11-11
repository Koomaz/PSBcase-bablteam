"""Превращение служебных ссылок [E1] в человекочитаемые для LLM-судьи.

Модель ставит [En] — их проверяет валидатор. Перед записью в CSV код заменяет
метки на ссылку из метаданных доказательства: «(п. 3.1 договора № А-08/2024)».
Маппинг детерминированный, поэтому перепутать пункт невозможно.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

CITE_RUN = re.compile(r"(?:\[\s*[EЕ]\s*\d{1,2}\s*\])+")
CITE_ID = re.compile(r"(\d{1,2})")

LAW_CODES = re.compile(r"\b(ТК|НК|КоАП|ГК|ГПК|АПК|УК)\s*РФ\b")
KNOWN_LAW_CODES = {"ТК": "ТК РФ", "НК": "НК РФ", "КоАП": "КоАП РФ"}

CODE_ALIASES = (
    ("трудовой кодекс", "ТК РФ"),
    ("налоговый кодекс", "НК РФ"),
    ("кодекс об административных", "КоАП РФ"),
    ("гражданский кодекс", "ГК РФ"),
    ("197-фз", "ТК РФ"),
)


def _date_ru(value: str | None) -> str:
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value or ""))
    return f"{match.group(3)}.{match.group(2)}.{match.group(1)}" if match else ""


CASES = {
    "contract": ("договор", "договора"),
    "addendum": ("доп. соглашение", "доп. соглашения"),
    "regulation": ("регламент", "регламента"),
    "memo": ("служебная записка", "служебной записки"),
    "letter": ("письмо", "письма"),
}


def _doc_label(meta: dict[str, Any], genitive: bool = False) -> str:
    """Падеж важен: «п. 3.1 договора № А-08/2024», но «(договор № А-08/2024)»."""
    doc_type = (meta.get("doc_type") or "").lower()
    number = (meta.get("doc_number") or "").strip()
    title = (meta.get("doc_title") or "").strip()
    index = 1 if genitive else 0

    if doc_type in CASES:
        base = CASES[doc_type][index]
        if doc_type == "regulation":
            edition = _date_ru(meta.get("edition_date"))
            return f"{base} в ред. от {edition}" if edition else base
        label = f"{base} № {number}" if number else base
        if doc_type == "addendum" and meta.get("parent_doc_number"):
            label += f" к договору № {meta['parent_doc_number']}"
        return label
    if doc_type == "law":
        if label := KNOWN_LAW_CODES.get(meta.get("law_code")):
            return label
        haystack = f"{number} {title}".lower()
        for needle, alias in CODE_ALIASES:
            if needle in haystack:
                return alias
        code = LAW_CODES.search(f"{number} {title}")
        if code:
            return f"{code.group(1)} РФ"
        if number and "ФЗ" in number.upper():
            return f"Закона № {number}" if genitive else f"Закон № {number}"
        return number or "закон"
    if number:
        return f"№ {number}"
    return re.sub(r"\s+", " ", title)[:60] or "документ"


def _locator(meta: dict[str, Any], doc_type: str) -> str:
    article = (meta.get("article") or "").strip()
    part = (meta.get("part") or "").strip()
    clause = (meta.get("clause") or "").strip()
    subclause = (meta.get("subclause") or "").strip()
    appendix = (meta.get("appendix") or "").strip()

    if doc_type == "law":
        return " ".join(f"{label} {value}" for label, value in (
            ("подп.", subclause), ("п.", clause), ("ч.", part), ("ст.", article),
        ) if value)

    if clause:
        if subclause.startswith(clause + "."):
            return f"п. {subclause}"
        return f"подп. {subclause} п. {clause}" if subclause else f"п. {clause}"
    if article:
        return f"ст. {article}"
    if appendix:
        return appendix
    return ""


def human_reference(item: dict[str, Any]) -> str:
    """«(п. 3.1 договора № А-08/2024)» без скобок — их ставит render_citations."""
    meta = {**(item.get("meta") or {}), **(item.get("citation_payload") or {})}
    doc_type = (meta.get("doc_type") or "").lower()
    source = (item.get("source") or "").strip()
    if not doc_type and source:
        return source
    locator = _locator(meta, doc_type)
    label = _doc_label(meta, genitive=bool(locator))

    if locator:
        return f"{locator} {label}"

    if doc_type == "addendum" and meta.get("amends_clause"):
        return f"{label}, изменение п. {meta['amends_clause']} основного договора"
    tail = source.rsplit(", ", 1)[-1] if ", " in source else ""
    if tail and re.match(r"^(?:п\.|ст\.|ч\.|прил\.)", tail, re.I):
        return f"{tail} {_doc_label(meta, genitive=True)}"
    return label


def render_citations(
    text: str,
    items: Sequence[dict[str, Any]],
    style: str = "ref",
) -> str:
    """style="ref" — только человекочитаемая ссылка, "both" — ссылка + [En], "id" — как есть."""
    if not text or style == "id":
        return text

    references = {item.get("eid"): human_reference(item) for item in items}

    def replace(match: re.Match[str]) -> str:
        ids = CITE_ID.findall(match.group(0))
        labels: list[str] = []
        for number in ids:
            label = references.get(f"E{number}")
            if label and label not in labels:
                labels.append(label)
        if not labels:
            return ""
        rendered = f"({' ; '.join(labels).replace(' ; ', '; ')})"
        if style == "both":
            rendered += "".join(f"[E{number}]" for number in ids)
        return rendered

    result = CITE_RUN.sub(replace, text)
    # Модель иногда упоминает служебный ID без скобок («приведённая в E1»).
    result = re.sub(r"\s*\b[EЕ]\d{1,2}\b", "", result)
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"\(\s+", "(", result)
    result = re.sub(r"\s+\)", ")", result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    return result.strip()

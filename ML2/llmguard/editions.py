"""Подмешивание действующей редакции к датированному вопросу.

ML1 при вопросе «в августе 2024 года» правильно отдаёт историческую редакцию,
но не всегда — действующую для сравнения. Судья и эталон ценят явное разделение
редакций, поэтому ML2 делает один дополнительный поиск по названию документа
и ставит найденную действующую редакцию рядом с исторической.

Дополнительный поиск выполняется только для датированных вопросов и только если
в выдаче уже есть документ с признаками редакции (edition_date / validity_period /
version_status). Иначе поведение не меняется.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

EDITION_PAREN = re.compile(r"\s*\((?:ред\.?|редакция|версия)[^)]*\)\s*$", re.I)
EDITION_TAIL = re.compile(r"\s*,?\s*(?:ред\.?|редакция|версия)\s+от\s+[\d.]+\s*$", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_RE = re.compile(
    r"\b(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)",
    re.I,
)


def base_doc_title(title: str) -> str:
    text = re.sub(r"\s+", " ", (title or "").strip())
    text = EDITION_PAREN.sub("", text)
    text = EDITION_TAIL.sub("", text)
    return text.strip(" ,;.")


def question_is_dated(question: str) -> bool:
    return bool(YEAR_RE.search(question or "") or MONTH_RE.search(question or ""))


def _meta(chunk: Any) -> dict[str, Any]:
    meta = getattr(chunk, "metadata", None) or {}
    return meta if isinstance(meta, dict) else {}


def _title(chunk: Any) -> str:
    meta = _meta(chunk)
    return base_doc_title(str(meta.get("doc_title") or getattr(chunk, "source_reference", "") or ""))


def _edition_marker(chunk: Any) -> str:
    meta = _meta(chunk)
    return str(
        meta.get("edition_date")
        or meta.get("doc_date")
        or (meta.get("validity_period") or {}).get("from")
        or ""
    )


def locator_key(meta: dict[str, Any]) -> tuple[str, ...]:
    """Ключ положения: редакции сопоставляются по той же статье/пункту, а не по всему документу."""
    return tuple(str(meta.get(key) or "").strip() for key in ("article", "part", "clause", "subclause"))


def _same_locator(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_key, right_key = locator_key(left), locator_key(right)
    # Сопоставляем по самому конкретному непустому уровню.
    for index in range(len(left_key) - 1, -1, -1):
        if left_key[index]:
            return bool(right_key[index]) and left_key[index] == right_key[index]
    return False


def augment_editions(
    retriever: Any,
    question: str,
    chunks: Sequence[Any],
    *,
    top_k: int = 8,
    max_extra: int = 1,
) -> tuple[list[Any], str]:
    """Возвращает (chunks, note). При отключённом/неудачном поиске — исходный список."""
    original = list(chunks or [])
    if not original or not question_is_dated(question):
        return original, ""
    if not hasattr(retriever, "search"):
        return original, ""

    anchors = [
        chunk for chunk in original
        if _meta(chunk).get("edition_date")
        or _meta(chunk).get("validity_period")
        or _meta(chunk).get("version_status")
    ]
    if not anchors:
        return original, ""

    anchor = anchors[0]
    base = _title(anchor)
    anchor_meta = _meta(anchor)
    if len(base) < 8 or not any(locator_key(anchor_meta)):
        return original, ""

    locator = (
        f"п. {anchor_meta.get('clause')}" if anchor_meta.get("clause")
        else f"ст. {anchor_meta.get('article')}" if anchor_meta.get("article")
        else ""
    )
    query = f"{base} действующая редакция {locator}".strip()
    try:
        extra = retriever.search(query, top_k=top_k) or []
    except Exception:
        return original, ""

    seen = {getattr(chunk, "chunk_id", "") for chunk in original}
    editions = {_edition_marker(chunk) for chunk in original if _title(chunk) == base}
    picks: list[Any] = []
    for chunk in extra:
        cid = getattr(chunk, "chunk_id", "")
        if not cid or cid in seen:
            continue
        if _title(chunk) != base:
            continue
        if not _same_locator(anchor_meta, _meta(chunk)):
            continue
        marker = _edition_marker(chunk)
        if marker and marker in editions:
            continue
        picks.append(chunk)
        seen.add(cid)
        if len(picks) >= max_extra:
            break

    if not picks:
        return original, ""

    # Действующая редакция идёт после первых двух фрагментов: build_evidence берёт
    # первые max_items, а соседи собираются только с первых двух — так редакция
    # не тянет за собой случайные соседние пункты.
    at = min(2, len(original))
    augmented = original[:at] + picks + original[at:]
    editions_text = ", ".join(
        sorted({_edition_marker(chunk) for chunk in picks if _edition_marker(chunk)})
    )
    note = f"edition_comparison_added:{editions_text or 'current'}"
    return augmented, note

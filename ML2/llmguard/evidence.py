"""Преобразование выдачи ML1 в список доказательств [E1]..[En] и эвристики по посылке вопроса."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

MAX_ITEM_CHARS = 1100
MAX_TOTAL_CHARS = 7000

USER_PREFIX = re.compile(r"^\s*(?:пользователь|user|клиент)\s*[:\-]\s*", re.I)
ASSISTANT_PREFIX = re.compile(r"^\s*(?:ассистент|assistant|бот)\s*[:\-]\s*", re.I)

_NUM_RE = re.compile(r"\d[\d\s\u00a0.,]*\d|\d")
_QUOTED_RE = re.compile(r"[«\"„]([^»\"“]{3,60})[»\"“]")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def parse_question(raw: str) -> tuple[str, list[dict[str, str]]]:
    """Разбирает строку вопроса на (текущий вопрос, история диалога).

    ML1 умеет принимать историю вида [{"role": "user", "content": ...}].
    Формат строки в датасете: "Пользователь: ...\\nАссистент: ...\\nПользователь: ...".
    """
    text = (raw or "").strip()
    if not text:
        return "", []

    history: list[dict[str, str]] = []
    current: list[str] = []
    saw_dialog = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if USER_PREFIX.match(stripped):
            saw_dialog = True
            if current:
                history.append({"role": "assistant", "content": " ".join(current)})
                current = []
            body = USER_PREFIX.sub("", stripped).strip()
            history.append({"role": "user", "content": body})
        elif ASSISTANT_PREFIX.match(stripped):
            saw_dialog = True
            if current:
                history.append({"role": "user", "content": " ".join(current)})
                current = []
            history.append({"role": "assistant", "content": ASSISTANT_PREFIX.sub("", stripped).strip()})
        elif history and history[-1]["role"] == "assistant":
            history[-1]["content"] += " " + stripped
        else:
            current.append(stripped)

    if not saw_dialog:
        return text, []

    if current:
        history.append({"role": "user", "content": " ".join(current)})

    last_user = ""
    for turn in reversed(history):
        if turn["role"] == "user" and turn["content"].strip():
            last_user = turn["content"].strip()
            break

    return last_user, history


_REF_LOCATOR = re.compile(r",\s*(п\.|ст\.|ч\.)\s*([^,]+)\s*$")


def _locator_from_reference(reference: str) -> dict[str, str]:
    """Достаёт пункт из «…, п. 3.3», чтобы сосед цитировался своим номером."""
    match = _REF_LOCATOR.search(reference or "")
    if not match:
        return {}
    value = match.group(2).strip()
    return {"clause": value} if match.group(1) == "п." else {"article": value}


def _meta_summary(meta: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "doc_type", "doc_number", "doc_date", "edition_date", "effective_date",
        "validity_period", "version_status", "temporal_status", "article",
        "section", "clause", "part", "subclause", "appendix", "amends_clause",
        "counterparties", "law_code",
    )
    out = {k: meta.get(k) for k in keys if meta.get(k) not in (None, "", [], {})}
    return out


def build_evidence(
    chunks: Sequence[Any],
    max_items: int = 6,
    max_total_chars: int = MAX_TOTAL_CHARS,
    max_neighbors: int = 3,
) -> list[dict[str, Any]]:
    """Строит нумерованный список доказательств, убирая дубли и пустые чанки.

    К найденным фрагментам добавляются соседние положения из `neighboring_context`:
    ML1 отдаёт их бесплатно, а нужный пункт часто стоит вплотную к найденному
    (например, п. 3.3 между 3.2 и 3.4 по вопросу об арендной плате).
    """
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    total = 0

    for chunk in chunks or []:
        if len(items) >= max_items:
            break

        chunk_id = getattr(chunk, "chunk_id", "") or ""
        text = (getattr(chunk, "text", "") or "").strip()
        if not text or chunk_id in seen_ids:
            continue

        fingerprint = re.sub(r"\s+", " ", text.lower())[:220]
        if fingerprint in seen_text:
            continue

        if total + len(text) > max_total_chars and items:
            break

        seen_ids.add(chunk_id)
        seen_text.add(fingerprint)
        total += len(text)

        meta = getattr(chunk, "metadata", None) or {}
        items.append(
            {
                "eid": f"E{len(items) + 1}",
                "chunk_id": chunk_id,
                "source": getattr(chunk, "source_reference", "") or "",
                "text": text[:MAX_ITEM_CHARS],
                "meta": _meta_summary(meta),
            }
        )

    if max_neighbors > 0 and items:
        taken = 0
        for chunk in list(chunks)[:2]:
            if taken >= max_neighbors:
                break
            for neighbor in getattr(chunk, "neighboring_context", None) or []:
                if taken >= max_neighbors or len(items) >= max_items + max_neighbors:
                    break
                chunk_id = neighbor.get("chunk_id") or ""
                text = (neighbor.get("text") or "").strip()
                if not text or chunk_id in seen_ids:
                    continue
                fingerprint = re.sub(r"\s+", " ", text.lower())[:220]
                if fingerprint in seen_text:
                    continue
                seen_ids.add(chunk_id)
                seen_text.add(fingerprint)
                taken += 1
                meta = dict(_meta_summary(getattr(chunk, "metadata", None) or {}))
                meta.update(_locator_from_reference(neighbor.get("source_reference") or ""))
                meta["neighbor"] = True
                items.append(
                    {
                        "eid": f"E{len(items) + 1}",
                        "chunk_id": chunk_id,
                        "source": neighbor.get("source_reference") or "",
                        "text": text[:MAX_ITEM_CHARS],
                        "meta": meta,
                    }
                )

    return items


def _words(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]{3,}", text.lower())


def _name_present(name: str, blob_words: list[str]) -> bool:
    """«Северным Складом» ≈ «Северный Склад», но «Альфа-Трейдинг» ≠ «Альфа-Трейд».

    Сверяем по основе (первые 5 символов) и требуем близкую длину, иначе
    разные юрлица с общим корнем склеиваются.
    """
    for word in _words(name):
        stem = word[:5]
        if not any(candidate[:5] == stem and abs(len(candidate) - len(word)) <= 2
                   for candidate in blob_words):
            return False
    return True


def _numbers_in(text: str) -> set[str]:
    found: set[str] = set()
    for match in _NUM_RE.finditer(text):
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if digits:
            found.add(digits.lstrip("0") or "0")
    return found


def premise_warnings(question: str, items: Sequence[dict[str, Any]]) -> list[str]:
    """Мягкие сигналы: числа/имена/годы из вопроса, которых нет в доказательствах.

    Это подсказка модели проверить посылку, а не автоматический вердикт.
    """
    if not question or not items:
        return []

    # метаданные (контрагенты, номер, даты) — тоже часть доказательства:
    # вопрос может называть сторону, которая есть только в шапке документа
    def _flat(item: dict[str, Any]) -> str:
        meta = item.get("meta") or {}
        parts = [str(item.get("source", "")), str(item.get("text", ""))]
        for value in meta.values():
            if isinstance(value, (list, tuple)):
                parts.extend(str(v) for v in value)
            elif value is not None:
                parts.append(str(value))
        return " ".join(parts)

    blob = " ".join(_flat(item) for item in items)
    blob_numbers = _numbers_in(blob)
    blob_words = _words(blob)

    warnings: list[str] = []

    for match in _NUM_RE.finditer(question):
        digits = re.sub(r"\D", "", match.group(0)).lstrip("0") or "0"
        if len(digits) < 3:
            continue
        if digits in blob_numbers:
            continue
        # 12 млн -> 12000000, 1,5 -> 15 (не трогаем, только крупные суммы)
        if digits.startswith("1") and digits[1:].strip("0") == "" and len(digits) > 3:
            continue
        warnings.append(f"число «{match.group(0).strip()}» из вопроса не найдено в доказательствах дословно")

    for match in _QUOTED_RE.finditer(question):
        name = match.group(1).strip()
        core = re.sub(r"^(?:ооо|оао|зао|пао|ип)\s+", "", name.lower()).strip()
        if len(core) < 4:
            continue
        if _name_present(core, blob_words) or _name_present(name, blob_words):
            continue
        warnings.append(f"название «{name}» из вопроса не встречается в доказательствах")

    q_years = set(m.group(0) for m in _YEAR_RE.finditer(question))
    b_years = set(m.group(0) for m in _YEAR_RE.finditer(blob))
    for year in sorted(q_years - b_years):
        warnings.append(f"год {year} из вопроса не встречается в доказательствах")

    seen: set[str] = set()
    unique: list[str] = []
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            unique.append(warning)
    return unique[:6]


def summarize_diagnostics(diagnostics: dict[str, Any] | None) -> str:
    """Короткая сводка сигналов ML1 для промпта."""
    diagnostics = diagnostics or {}
    parts: list[str] = []
    confidence = diagnostics.get("confidence")
    if confidence:
        parts.append(f"confidence={confidence}")
    warnings = [w for w in (diagnostics.get("warnings") or []) if isinstance(w, str)]
    if warnings:
        parts.append("warnings=" + ", ".join(warnings[:10]))
    return "; ".join(parts)


def iter_citation_ids(items: Iterable[dict[str, Any]]) -> list[str]:
    return [item["eid"] for item in items]

"""Пост-валидация ответа: цитаты, объём, markdown, незаземлённые числа, ложный отказ."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

CITE_RE = re.compile(r"\[\s*(?:E|Е)\s*(\d{1,2})\s*\]")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)*")
NUM_RE = re.compile(r"\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:%|₽|руб\.?|рублей|дн\.?|дней|месяц\w*|год\w*|лет)?")

REFUSAL_MARKERS = (
    "ответа нет", "не найдено", "отсутствует", "не содержится", "не раскрыт",
    "нет сведений", "не указано", "не предусмотрено", "сведений нет",
    "в документах нет", "не относится", "вне рамок",
)

FORBIDDEN_PATTERNS = (
    (re.compile(r"\bя\s+(?:не\s+)?(?:юрист|ии|языкова\w* модель|искусственн\w* интеллект)\b", re.I),
     "ответ содержит отказ по признаку «я не юрист / языковая модель»"),
    (re.compile(r"\bкак\s+(?:правило|обычно)\b", re.I), "ответ содержит обобщение вместо факта («как правило»/«обычно»)"),
    (re.compile(r"\bнаверн(?:ое|яка)\b|\bвозможно\b", re.I), "ответ содержит догадку вместо факта"),
    (re.compile(r"^\s*(?:конечно|отличный вопрос|хороший вопрос)", re.I), "ответ начинается с разговорной вводной"),
)

MARKDOWN_RE = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*+]\s|^\s*\d+[.)]\s|\|)", re.M)

# Формулировки, которыми ответ фиксирует расхождение с посылкой вопроса.
DISCREPANCY_MARKERS = (
    "не подтвержда", "не соответству", "не закрепл", "не встреча", "не совпада",
    "не указан", "не назван", "отличается", "неверн", "ошибоч", "в действительности",
    "фактически", "на самом деле", "а не ",
)

REFUSAL_TEMPLATE_HINT = (
    "укажи, каких именно документов или положений не хватает"
)


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    cleaned: str = ""
    words: int = 0
    citations: list[str] = field(default_factory=list)
    is_refusal: bool = False

    @property
    def repairable(self) -> bool:
        return bool(self.cleaned)


def _numbers(text: str) -> set[str]:
    out: set[str] = set()
    for match in NUM_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) >= 3:
            out.add(digits.lstrip("0") or "0")
    return out


def question_parts(question: str) -> int:
    """Грубая оценка числа частей вопроса: «какая плата и до какого числа» — две."""
    text = (question or "").strip()
    if not text:
        return 1
    parts = 1
    parts += len(re.findall(r";", text))
    parts += len(re.findall(r",?\s+и\s+(?=до|как|кто|что|когда|какой|какую|какие|сколько|где|зачем|почему|можно|нужно|обязан)", text, re.I))
    parts += len(re.findall(r"\?", text)) - 1 if text.count("?") > 1 else 0
    return max(1, min(parts, 4))


def clean_text(answer: str) -> str:
    text = (answer or "").replace("\r", " ")
    text = re.sub(r"</?[A-Za-z][^>]{0,40}>", " ", text)          # случайные XML-теги
    text = re.sub(r"^\s*(?:Ответ|ОТВЕТ)\s*:\s*", "", text)
    text = MARKDOWN_RE.sub(" ", text)
    text = text.replace("**", " ").replace("`", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip().strip('"').strip()


def validate(
    answer: str,
    items: Sequence[dict[str, Any]],
    max_words: int = 180,
    min_words: int = 5,
    check_numbers: bool = True,
    question: str = "",
    premise_conflicts: Sequence[str] = (),
) -> ValidationResult:
    cleaned = clean_text(answer)
    issues: list[str] = []
    valid_ids = {item["eid"] for item in items}
    valid_numbers = _numbers(" ".join(item.get("text", "") for item in items))
    # Число из вопроса можно повторить, чтобы опровергнуть ложную посылку.
    if question:
        valid_numbers |= _numbers(question)

    raw_ids = [f"E{n}" for n in CITE_RE.findall(answer or "")]
    unknown = sorted({eid for eid in raw_ids if eid not in valid_ids})
    citations = [eid for eid in raw_ids if eid in valid_ids]
    if unknown:
        issues.append(
            "ссылки " + ", ".join(unknown) + " отсутствуют в списке доказательств; "
            "цитировать можно только " + (", ".join(sorted(valid_ids)) or "ничего")
        )
        for eid in unknown:
            cleaned = re.sub(r"\[\s*[EЕ]\s*%s\s*\]" % eid[1:], "", cleaned)

    # Отказ — это когда ПЕРВОЕ предложение говорит об отсутствии данных.
    # «...в документах нет» в конце разбора ложной посылки отказом не является.
    lowered = cleaned.lower()
    first_sentence = re.split(r"(?<=[.!?])\s", lowered, maxsplit=1)[0]
    marker_hit = any(marker in lowered for marker in REFUSAL_MARKERS)
    is_refusal = any(marker in first_sentence for marker in REFUSAL_MARKERS) or (
        marker_hit and not citations
    )

    words = len(WORD_RE.findall(cleaned))
    if not cleaned:
        issues.append("пустой ответ")
    else:
        without_cites = re.sub(r"(?:\s*\[[EЕ]\d+\])+\s*$", "", cleaned).strip()
        if without_cites and without_cites[-1] not in ".!?)]»":
            issues.append(
                "ответ обрывается на середине: нет завершающей точки "
                "(вероятно, упёрлись в лимит токенов)"
            )
        if words < min_words and not citations:
            issues.append(f"ответ слишком короткий ({words} слов)")
        elif not is_refusal and words < 18 and question:
            parts = question_parts(question)
            if parts > 1:
                issues.append(
                    f"вопрос содержит {parts} части, а ответ — {words} слов: похоже, "
                    "часть вопроса осталась без ответа"
                )
            else:
                issues.append(
                    f"ответ слишком короткий для целевого объёма ({words} слов): "
                    "нужен развёрнутый ответ с основанием и ссылкой"
                )
        elif words > max_words:
            issues.append(f"ответ длиннее лимита: {words} слов при пределе {max_words}")

    if items and is_refusal and not citations:
        issues.append(
            "похоже на необоснованный отказ: в доказательствах есть релевантные фрагменты, "
            "но ответ не содержит ни одной ссылки"
        )
    if items and not is_refusal and not citations:
        issues.append("в ответе нет ни одной ссылки [En] на доказательства")
    if not items and not is_refusal:
        issues.append("доказательств нет, а ответ не сообщает об отсутствии данных")

    if is_refusal and not re.search(r"отсутству|не\s+найден|нет\s+сведен|не\s+раскрыт|нужен|требуется", cleaned, re.I):
        issues.append("отказ не объясняет, чего именно не хватает: " + REFUSAL_TEMPLATE_HINT)

    if check_numbers and items:
        for number in sorted(_numbers(cleaned) - valid_numbers):
            issues.append(f"число {number} отсутствует в доказательствах (возможная выдумка)")

    for pattern, message in FORBIDDEN_PATTERNS:
        if pattern.search(cleaned):
            issues.append(message)

    if re.search(r"\[E\d+\]", " ".join(i.get("text", "") for i in items)):
        pass  # шум в самих документах не считаем ошибкой ответа

    return ValidationResult(
        ok=not issues,
        issues=issues,
        cleaned=cleaned,
        words=words,
        citations=citations,
        is_refusal=is_refusal,
    )


def premise_ack_prefix(conflicts: Sequence[str]) -> str:
    """Детерминированная преамбула для ложной посылки: модель её не всегда пишет сама."""
    values = ", ".join(f"«{value}»" for value in conflicts if value)
    if not values:
        return ""
    return (
        f"Посылка в вопросе не подтверждается: {values} в предоставленных документах не встречается. "
    )


def ensure_premise_ack(text: str, conflicts: Sequence[str], max_words: int = 180) -> str:
    if not conflicts or not text:
        return text
    if any(marker in text.lower() for marker in DISCREPANCY_MARKERS):
        return text
    prefix = premise_ack_prefix(conflicts)
    if not prefix:
        return text
    if len(WORD_RE.findall(prefix + text)) > max_words:
        return text
    return (prefix + text).replace("  ", " ").strip()


def fallback_refusal(items: Sequence[dict[str, Any]]) -> str:
    """Гарантированный детерминированный ответ, если модель не справилась за два прохода."""
    if not items:
        return (
            "В предоставленных документах ответа нет: отсутствуют релевантные положения. "
            "Для ответа нужен документ, регулирующий этот вопрос."
        )
    source = items[0].get("source") or "документ"
    excerpt = re.sub(r"\s+", " ", items[0].get("text", "")).strip()
    excerpt = excerpt[:300].rsplit(" ", 1)[0] if len(excerpt) > 300 else excerpt
    return (
        f"По этому вопросу в документах есть только следующее: {excerpt} [E1]. "
        f"Остальные условия в предоставленных материалах не раскрыты — уточните вопрос "
        f"или запросите недостающий документ ({source})."
    )

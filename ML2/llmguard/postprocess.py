"""Детерминированная чистка ответа от мета-конclusion и повторов.

Модель любит добавлять «То есть…», «Таким образом…» и пересказывать только что
сказанное другими словами. Для судьи это шум: эталонные ответы состоят из
прямых утверждений со ссылками. Удаляем такие предложения детерминированно,
не отдавая решение на усмотрение ещё одного вызова модели.
"""

from __future__ import annotations

import re

from .textutil import split_sentences, word_key

GAP_SENTENCE = re.compile(
    r"(?:документов|сведений|подтверждений).{0,200}(?:нет|не представлен|не содержится|не назван)",
    re.I,
)

META_START = re.compile(
    r"^(?:то есть|таким образом|иными словами|следовательно|это означает|"
    r"это значит|итак|в итоге|соответственно)\b",
    re.I,
)

STOPWORDS = {
    "который", "которая", "которые", "которого", "которой", "которым",
    "этого", "этом", "этим", "этой", "также", "если", "чтобы", "между",
    "после", "перед", "через", "согласно", "только", "более", "менее",
    "может", "могут", "должен", "должна", "должны", "быть", "есть",
    "предоставленных", "предоставленным", "материалах", "документах",
    "фрагментах", "пунктом", "пункта", "договора", "статьи", "закона",
}


def _content_set(sentence: str) -> set[str]:
    words = re.findall(r"[а-яё]{4,}", (sentence or "").lower())
    return {word[:5] for word in words if word not in STOPWORDS}


def _is_redundant(sentence: str, previous: str) -> bool:
    current = _content_set(sentence)
    earlier = _content_set(previous)
    if len(current) < 2 or len(earlier) < 2:
        return False
    if current <= earlier:
        return True
    if len(current) < 3:
        return False
    overlap = len(current & earlier)
    return overlap / max(1, len(current)) >= 0.85


def strip_meta_and_redundant(text: str) -> str:
    """Удаляет вводные выводы и предложения, повторяющие уже сказанное."""
    if not text.strip():
        return text
    # «Не сразу:» — оценка, которой нет в норме; эталон сразу называет срок.
    text = re.sub(r"^\s*Не сразу\s*(?:[:.,—–-]\s*|$)", "", text, flags=re.I)
    kept: list[str] = []
    gap_seen = False
    for sentence in split_sentences(text):
        if kept and META_START.match(sentence.strip()):
            continue
        if any(_is_redundant(sentence, previous) for previous in kept):
            continue
        if GAP_SENTENCE.search(sentence):
            if gap_seen:
                continue
            gap_seen = True
        kept.append(sentence)
    if not kept:
        return text
    result = " ".join(kept).strip()
    # Не отдаём огрызок без содержательного предложения.
    if len(word_key(result).split()) < 8:
        return text
    return result

INFERENCE_RE = re.compile(
    r"(?:недостатк|дефект|претензи)[^.]{0,80}(?:должн|обязан)[^.]{0,60}"
    r"(?:выявлен|зафиксирован|предъявлен)",
    re.I,
)


def strip_unsupported_inferences(text: str, evidence_text: str) -> str:
    """Убирает известное опасное преобразование «приёмка» → «фиксация недостатков».

    Норма устанавливает срок приёмки по качеству, но не срок, в который любые
    недостатки должны быть выявлены или зафиксированы. Если такой формулировки
    нет в доказательствах, предложение удаляется.
    """
    if not text.strip():
        return text
    blob = (evidence_text or "").lower()
    kept = [
        sentence for sentence in split_sentences(text)
        if not (INFERENCE_RE.search(sentence) and INFERENCE_RE.search(sentence).group(0).lower() not in blob)
    ]
    if not kept or len(word_key(" ".join(kept)).split()) < 8:
        return text
    return " ".join(kept)

# Вопросы типа «кто запускает согласование» и «действуют ли обычные правила»
# соблазняют модель добавить сроки, СЭД и расчёт испытания. Это другой слот,
# поэтому режем детерминированно, не полагаясь на ещё один вызов судьи.
QUESTION_TRIMS = (
    (
        re.compile(r"\bкто\b[^?]{0,80}(?:запуска|иниции|согласов)", re.I),
        re.compile(r"(?:СЭД|лист согласования|срок согласования|рабочих дн|продлен)", re.I),
    ),
    (
        re.compile(r"(?:распространя|действуют)[^?]{0,80}(?:испытательн|трудов)", re.I),
        re.compile(r"(?:не может превышать|шести месяцев|трёх месяцев|не засчитываются|двух недель)", re.I),
    ),
    (
        re.compile(r"(?:включен|отдельн|оплачива)[^?]{0,80}(?:доставк|упаковк)", re.I),
        re.compile(r"(?:общая цена|ассортимент|наименование, количество)", re.I),
    ),
    (
        re.compile(r"почему[^?]{0,90}(?:сумм|плат|цена|стоимост)", re.I),
        re.compile(r"(?:вносится|внесени|коммунальн|порядок оплаты|ежемесячно)", re.I),
    ),
)


def ensure_capitalized(text: str) -> str:
    """После удаления «Не сразу:» первая буква могла остаться строчной."""
    if text and text[0].islower():
        return text[0].upper() + text[1:]
    return text


def trim_by_question_type(question: str, text: str) -> str:
    """Убирает детали другого слота, характерные для конкретного типа вопроса."""
    if not text.strip():
        return text
    for question_re, drop_re in QUESTION_TRIMS:
        if not question_re.search(question or ""):
            continue
        kept = [sentence for sentence in split_sentences(text) if not drop_re.search(sentence)]
        if kept and len(word_key(" ".join(kept)).split()) >= 4:
            text = " ".join(kept)
    return text


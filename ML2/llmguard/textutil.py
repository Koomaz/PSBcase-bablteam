"""Общие утилиты работы с предложениями ответа."""

from __future__ import annotations

import re

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«(])")
ABBREVIATIONS = {
    "г", "д", "п", "ст", "ч", "руб", "т", "е", "им", "ред", "доп", "стр",
    "рис", "табл", "млн", "млрд", "тыс", "коп", "подп", "абз", "прил",
}


def split_sentences(text: str) -> list[str]:
    """Делит ответ на предложения, не разрывая «г. Химки», «п. 3.1», «т. е.»."""
    parts = [part.strip() for part in SENTENCE_SPLIT.split(text or "") if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged:
            previous = merged[-1]
            tail = re.search(r"([А-ЯЁA-Zа-яёa-z]{1,5})\.$", previous)
            if tail and tail.group(1).lower() in ABBREVIATIONS:
                merged[-1] = previous + " " + part
                continue
        merged.append(part)
    return merged


def word_key(text: str) -> str:
    return " ".join(re.findall(r"[а-яёa-z0-9]+", (text or "").lower()))

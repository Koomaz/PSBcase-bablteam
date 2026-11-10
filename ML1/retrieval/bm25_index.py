from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .metadata import normalize

BM25_K1, BM25_B = 1.35, 0.70
TOKEN_RE = re.compile(r"\d+(?:[.,/]\d+)+|[a-zа-я]+[a-zа-я0-9]*|\d+|№")
STOP = set("""
а без бы был была были было быть в во вот все вы где да для до его ее если есть же за и из или им
их к как какие какая какой какую когда кто ли ли либо мне может мы на над нам нас наш наша не него
нее нет ни но ну о об об этом она они он от по под при про раз с со так такой там те то того тоже
только том тот ту ты у уже что чтобы эта это этот я нужно надо можно должны должен должен ли какой
сколько каких какое были будет могут мог могла который которых с чем всем дальше просто почему
значит сразу вообще действительно подскажите хочу можем сутки новые новый после перед сам
""".split())
# Compact suffix normalization for Russian inflection. Retain the untouched form as a separate feature.
SUFFIX = re.compile(
    r"(иями|ями|ами|остью|остями|ого|ему|ому|ыми|ими|иях|ах|ях|ых|их|ым|им|ий|ый|ая|яя|ое|ее|ые|ие|"
    r"ой|ей|ую|юю|ом|ем|ов|ев|ам|ям|ою|ею|ия|ию|ии|ие|ые|ать|ять|ить|ется|утся|"
    r"а|я|о|е|у|ю|ы|и|ь)$"
)
# Search vocabulary only. Terms are grounded by actual indexed text at retrieval time.
SYNONYMS = (
    ("аванс", "авансом", "предоплата"),
    ("оплата", "оплатить", "оплачивает", "оплачивать", "оплачивается"),
    ("дефект", "дефекты", "недостаток", "недостатки", "брак"),
    ("испытательный", "испытательном", "испытание", "испытания"),
    ("проверить качество", "проверка качества", "приемка"),
    ("за сколько дней", "срок"),
    ("внести", "вносить", "вносится", "оплата"),
    ("пропускать", "пропуск", "доступ"),
    ("грузовик", "автомобиль", "транспорт"),
    ("съезжать", "возврат", "возвращает"),
    ("юротдел", "юридический отдел"),
    ("запустить согласование", "инициатор согласования"),
    ("обработка данных", "персональные данные"),
    ("отозвал", "отозвать", "отозвано", "отзыв"),
    ("гарантия", "гарантийный", "гарантийный срок"),
    ("расторжение", "прекращение", "съезжать"),
    ("передача", "передает", "передается", "передавать"),
    ("акт", "акту", "акта", "актом"),
    ("приложить", "приложение", "приложением", "прикрепить"),
    ("сезонный", "сезонные", "сезонных", "сезона"),
    ("ночью", "ночи", "круглосуточно", "круглосуточный"),
    ("сдать", "возвращает", "возврат", "возвратить", "вернуть"),
    ("после поставки", "дата поставки", "датой поставки"),
    ("компьютер", "компьютеры", "компьютерная техника"),
)


def stem(word: str) -> str:
    if len(word) > 4 and re.fullmatch(r"[а-я]+", word):
        candidate = SUFFIX.sub("", word)
        return candidate if len(candidate) >= 4 else word
    return word


def words(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def lexical_terms(text: str) -> list[str]:
    result = []
    for word in words(text):
        if (word not in STOP and len(word) > 1) or word in {"№", "п", "ч"} or word.isdigit():
            result.append(word)
            root = stem(word)
            if len(word) >= 4 and word.isalpha():
                result.append("~" + root)
    return result


def expand_terms(text: str) -> dict[str, float]:
    normalized = normalize(text)
    weighted = {term: (0.70 if term.startswith("~") else 1.0) for term in lexical_terms(text)}
    query_roots = {stem(word) for word in words(text)}
    for group in SYNONYMS:
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized) if " " in term
               else stem(term) in query_roots for term in group):
            for term in group:
                for feature in lexical_terms(term):
                    weighted.setdefault(feature, 0.52)
    return weighted


class BM25Index:
    def __init__(self, postings: dict, lengths: list[int]):
        self.postings = postings
        self.lengths = lengths
        self.average = sum(lengths) / len(lengths) if lengths else 1.0
        self.idf = {
            token: math.log1p((len(lengths) - len(rows) + 0.5) / (len(rows) + 0.5))
            for token, rows in postings.items()
        }

    @classmethod
    def build(cls, chunks: list) -> BM25Index:
        postings, lengths = defaultdict(list), []
        for i, chunk in enumerate(chunks):
            counts = Counter(lexical_terms(chunk.normalized_text))
            lengths.append(sum(counts.values()))
            for token, count in counts.items():
                postings[token].append([i, count])
        return cls(dict(postings), lengths)

    def dump(self) -> dict:
        return {"postings": self.postings, "lengths": self.lengths}

    def search(self, terms: dict[str, float], eligible: set[int]) -> dict[int, float]:
        scores = defaultdict(float)
        for term, weight in terms.items():
            if term not in self.postings:
                continue
            multiplier = weight * self.idf[term] * (BM25_K1 + 1)
            for idx, freq in self.postings[term]:
                if idx in eligible:
                    norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[idx] / (self.average or 1))
                    scores[idx] += multiplier * freq / (freq + norm)
        return dict(scores)

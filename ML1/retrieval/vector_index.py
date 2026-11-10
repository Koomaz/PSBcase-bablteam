from __future__ import annotations

import math
from collections import Counter, defaultdict
from operator import mul

from .bm25_index import STOP, words
from .embeddings import ModelAPI, ModelAPIError


def _features(text: str) -> Counter:
    return Counter(
        word[i:i + width]
        for token in words(text) if len(token) >= 4 and token not in STOP
        for word in [f"^{token}$"]
        for width in (3, 4)
        for i in range(len(word) - width + 1)
    )


def _vector(counts: Counter, idf: dict[str, float]) -> dict[str, float]:
    weights = {gram: (1 + math.log(count)) * idf[gram] for gram, count in counts.items() if gram in idf}
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    return {gram: weight / norm for gram, weight in weights.items()} if norm else {}


class LocalVectorIndex:
    def __init__(self, idf: dict, postings: dict):
        self.idf, self.postings = idf, postings

    @classmethod
    def build(cls, chunks: list) -> LocalVectorIndex:
        counts = [_features(chunk.text) for chunk in chunks]
        df = Counter(gram for count in counts for gram in count)
        idf = {gram: math.log((1 + len(counts)) / (1 + freq)) + 1 for gram, freq in df.items() if freq >= 2}
        postings = defaultdict(list)
        for idx, count in enumerate(counts):
            for gram, weight in _vector(count, idf).items():
                postings[gram].append([idx, round(weight, 8)])
        return cls(idf, dict(postings))

    def dump(self) -> dict:
        return {"idf": self.idf, "postings": self.postings}

    def search(self, query: str, eligible: set[int]) -> dict[int, float]:
        scores = defaultdict(float)
        for gram, weight in _vector(_features(query), self.idf).items():
            for idx, value in self.postings[gram]:
                if idx in eligible:
                    scores[idx] += weight * value
        return {idx: score for idx, score in scores.items() if score >= 0.075}


def build_dense(chunks: list, api: ModelAPI, batch_size: int = 32) -> list[list[float]]:
    if not api.settings.configured:
        raise ModelAPIError("proxy_not_configured")
    vectors = []
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        vectors.extend(api.embed([chunk.text_with_context for chunk in batch], building=True))
        if len(vectors[-1]) != len(vectors[0]):
            raise ModelAPIError("inconsistent_embedding_dimension")
    return vectors


def dense_search(vectors: list[list[float]], query: list[float], eligible: set[int]) -> dict[int, float]:
    if not vectors or len(vectors[0]) != len(query):
        raise ModelAPIError("incompatible_embedding_dimension")
    if hasattr(math, "sumprod"):
        return {i: math.sumprod(vectors[i], query) for i in eligible}
    return {i: sum(map(mul, vectors[i], query)) for i in eligible}

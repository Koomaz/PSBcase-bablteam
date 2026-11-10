from __future__ import annotations

import math

from .embeddings import ModelAPI, ModelAPIError


def rerank(api: ModelAPI, query: str, documents: list[str]) -> dict[int, float]:
    response = api.post(api.settings.reranker_path, {
        "model": api.settings.reranker_model, "query": query, "documents": documents,
        "top_n": len(documents),
    })
    try:
        scores = {row["index"]: float(row["relevance_score"]) for row in response["results"]}
        if set(scores) != set(range(len(documents))) or not all(math.isfinite(score) for score in scores.values()):
            raise ValueError
        return scores
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ModelAPIError("invalid_reranker_response") from None

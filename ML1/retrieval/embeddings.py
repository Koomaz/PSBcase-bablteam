from __future__ import annotations

import json
import math
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import Settings


class ModelAPIError(RuntimeError):
    """Safe to expose in diagnostics. Contains no credentials or payloads."""


class NoModelRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ModelAPI:
    def __init__(self, settings: Settings):
        self.settings = settings

    def post(self, path: str, payload: dict, *, building: bool = False) -> dict:
        if not self.settings.configured:
            raise ModelAPIError("proxy_not_configured")
        base = self.settings.base_url.rstrip("/").removesuffix("/chat/completions")
        endpoint = "/" + path.lstrip("/")
        try:
            parsed = urlsplit(base)
        except ValueError:
            raise ModelAPIError("invalid_proxy_url") from None
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
            raise ModelAPIError("invalid_proxy_url")
        # A bare proxy host follows the same /v1 convention as the generation client.
        if not parsed.path and endpoint in {"/embeddings", "/rerank"}:
            base += "/v1"
        url = base + endpoint
        req = Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.settings.api_key}"},
            method="POST",
        )
        timeout = self.settings.build_timeout if building else self.settings.api_timeout
        for attempt in range(self.settings.retries + 1):
            try:
                with build_opener(NoModelRedirect()).open(req, timeout=timeout) as response:
                    result = json.load(response)
                if not isinstance(result, dict):
                    raise ModelAPIError("invalid_response")
                return result
            except HTTPError as exc:
                reason = f"http_{exc.code}"
                retry = exc.code in {408, 429, 500, 502, 503, 504}
                exc.close()
            except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError):
                reason, retry = "network_or_timeout", True
            except (ValueError, UnicodeError):
                reason, retry = "invalid_response", False
            if not retry or attempt == self.settings.retries:
                raise ModelAPIError(reason) from None
            time.sleep(0.20 * (2 ** attempt))
        raise ModelAPIError("request_failed")

    def embed(self, texts: list[str], *, building: bool = False) -> list[list[float]]:
        result = self.post(self.settings.embedding_path, {
            "model": self.settings.embedding_model, "input": texts, "encoding_format": "float",
        }, building=building)
        try:
            entries = sorted(result["data"], key=lambda row: row["index"])
            if [row["index"] for row in entries] != list(range(len(texts))):
                raise ValueError
            vectors = [row["embedding"] for row in entries]
            dimensions = {len(v) for v in vectors}
            if len(dimensions) != 1 or 0 in dimensions:
                raise ValueError
            normalized = []
            for vector in vectors:
                values = [float(value) for value in vector]
                norm = math.sqrt(sum(value * value for value in values))
                if not math.isfinite(norm) or norm == 0:
                    raise ValueError
                normalized.append([value / norm for value in values])
            return normalized
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ModelAPIError("invalid_embedding_response") from None

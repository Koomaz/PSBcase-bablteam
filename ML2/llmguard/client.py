"""Клиент OpenAI-совместимого API (AITUNNEL / Platform Proxy) без внешних зависимостей."""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


def load_dotenv(path: str | None = None) -> None:
    """Читает ML2/.env, не перетирая уже заданные переменные окружения."""
    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LLMError(RuntimeError):
    pass


class ChatClient:
    last_finish_reason: str | None = None

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 45.0,
        retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(0, retries)

    @classmethod
    def from_env(cls) -> "ChatClient":
        base = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
        key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME") or "deepseek-v4.1-flash"
        timeout = float(os.getenv("LLM_TIMEOUT", "45"))
        retries = int(os.getenv("LLM_RETRIES", "2"))
        return cls(model=model, base_url=base, api_key=key, timeout=timeout, retries=retries)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return self.base_url + "/chat/completions"
        return self.base_url + "/v1/chat/completions"

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 450,
    ) -> str:
        if not self.configured:
            raise LLMError("LLM_BASE_URL/LLM_API_KEY не заданы")

        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self._endpoint(), data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                self.last_finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
                return self._extract(data)
            except urllib.error.HTTPError as exc:  # 4xx/5xx
                detail = exc.read().decode("utf-8", "replace")[:400]
                last_error = LLMError(f"HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 403, 404):
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = LLMError(f"network: {exc}")
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = LLMError(f"bad response: {exc}")
            if attempt < self.retries:
                time.sleep(min(8.0, 1.5 * (2 ** attempt)) + random.uniform(0, 0.4))

        raise last_error or LLMError("неизвестная ошибка вызова LLM")

    @staticmethod
    def _extract(data: dict[str, Any]) -> str:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None:
            content = choice.get("text", "")
        if isinstance(content, list):  # некоторые прокси отдают массив блоков
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        if not isinstance(content, str) or not content.strip():
            raise KeyError("пустой content в ответе модели")
        return content.strip()


class MockClient:
    """Детерминированная заглушка: извлекает ответ из доказательств.

    Нужна, чтобы прогонять весь конвейер и тесты без доступа к прокси.
    """

    configured = True
    model = "mock-extractive"
    last_finish_reason = "stop"

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 450,
    ) -> str:
        if len(messages) > 3:  # повторный проход после валидатора
            return self._first_answer(messages)
        return self._first_answer(messages)

    @staticmethod
    def _first_answer(messages: Sequence[dict[str, str]]) -> str:
        user = messages[-1]["content"] if messages else ""
        question = ""
        match = re.search(r"<question>(.*?)</question>", user, re.S)
        if match:
            question = match.group(1).strip()

        blocks = re.findall(r"\[(E\d+)\].*?\ntext:\s*(.+?)(?=\n\n\[E|\Z)", user, re.S)
        if not blocks:
            return (
                "В предоставленных документах ответа нет: отсутствуют релевантные положения "
                "по этому вопросу. Для ответа нужен соответствующий нормативный документ."
            )

        eid, text = blocks[0]
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.;])\s+", text)
        body = " ".join(sentences[:2])[:320]
        return f"{body} [{eid}]."


def make_client():
    """Возвращает боевой клиент либо заглушку, если доступ к модели не настроен."""
    load_dotenv()
    client = ChatClient.from_env()
    if client.configured and os.getenv("MOCK_LLM", "").lower() not in ("1", "true", "yes"):
        return client
    return MockClient()

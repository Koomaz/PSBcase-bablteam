"""Shared question pipeline for the submission and the web API."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError, Future
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Awaitable, Callable, Literal
from urllib.parse import urlsplit

import httpx

LOG = logging.getLogger(__name__)

FLAT_RUNTIME = Path(__file__).with_name("retrieval_runtime.py").is_file()
if FLAT_RUNTIME:
    from retrieval_runtime import Chunk, RetrievedChunk, Retriever, Settings, build_indexes
else:
    from ML1.retrieval import RetrievedChunk, Retriever, Settings
    from ML1.retrieval.artifacts import build_indexes
    from ML1.retrieval.schemas import Chunk

ANSWER_LIMIT = 10_000
TECHNICAL_ANSWER = "Не удалось обработать вопрос из-за технического сбоя."
ROLE_LINE = re.compile(r"^[ \t]*(Пользователь|Ассистент):[ \t]*", re.MULTILINE)
AnswerKind = Literal["answer", "partial", "clarification", "insufficient", "refusal"]
KINDS = {"answer", "partial", "clarification", "insufficient", "refusal"}
ERROR_CODES = {"unavailable", "timeout", "upstream", "invalid_response"}


class BackendError(Exception):
    """Stable public error code. Exception details stay within the adapter."""

    def __init__(self, code: str):
        super().__init__(code if code in ERROR_CODES else "internal")


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Question:
    text: str
    history: tuple[Turn, ...] = ()

    @classmethod
    def from_platform(cls, value: str) -> Question:
        value = value.strip()
        markers = list(ROLE_LINE.finditer(value))
        if not markers or markers[0].start() != 0:
            return cls(value)
        turns = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(value)
            role = "user" if marker[1] == "Пользователь" else "assistant"
            turns.append(Turn(role, value[marker.end():end].strip()))
        for index in range(len(turns) - 1, -1, -1):
            if turns[index].role == "user":
                return cls(turns[index].content, tuple(turn for turn in turns[:index] if turn.content))
        return cls("")


@dataclass(frozen=True)
class Generated:
    kind: AnswerKind
    answer: str
    source_ids: tuple[str, ...] = ()


Generator = Callable[[Question, list[RetrievedChunk], dict], Awaitable[Generated]]

SOURCE_FIELDS = (
    "doc_id", "doc_title", "doc_type", "doc_number", "doc_date", "file_path", "article", "part",
    "clause", "subclause", "section", "appendix", "amends_clause", "edition_date",
    "effective_date", "validity_period", "snapshot_date", "version_status", "temporal_status",
    "parent_doc_id", "provision_effective_date", "amendment_application_date", "amendment_application_relation",
    "char_start", "char_end",
)


def source_record(chunk_id: str, reference: str, metadata: dict) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_reference": reference,
        **{field: metadata.get(field) for field in SOURCE_FIELDS},
    }


@dataclass(frozen=True)
class Answer:
    kind: AnswerKind
    answer: str
    sources: tuple[dict, ...]

    def as_csv_text(self) -> str:
        result = self.answer
        if self.sources:
            result += "\n\nИсточники:\n" + "\n".join(
                f"[{index}] {source['source_reference']}" for index, source in enumerate(self.sources, 1)
            )
        if len(result) > ANSWER_LIMIT:
            raise BackendError("invalid_response")
        return result


class Service:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        top_k: int = 5,
        timeout: float = 25,
        chunks: dict[str, Chunk] | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.timeout = timeout
        self.chunks = chunks if chunks is not None else {}

    async def ask(self, question: Question) -> Answer:
        if not question.text.strip():
            return Answer("clarification", "Уточните, пожалуйста, ваш вопрос.", ())
        history = [{"role": turn.role, "content": turn.content} for turn in question.history]
        try:
            chunks, diagnostics = self.retriever.search_with_diagnostics(
                question.text, top_k=self.top_k, history=history,
            )
        except Exception as exc:
            LOG.error("Retrieval failed (%s)", type(exc).__name__)
            raise BackendError("unavailable") from None
        citations = {
            chunk.chunk_id: source_record(chunk.chunk_id, chunk.source_reference, chunk.citation_payload)
            for chunk in chunks
        }
        for chunk in chunks:
            for neighbor in chunk.neighboring_context:
                neighbor_id = neighbor["chunk_id"]
                if neighbor_id not in citations and neighbor_id in self.chunks:
                    raw = self.chunks[neighbor_id]
                    metadata = {**raw.metadata, "temporal_status": neighbor.get("temporal_status")}
                    citations[neighbor_id] = source_record(neighbor_id, raw.source_reference, metadata)
        try:
            result = await asyncio.wait_for(
                self.generator(question, chunks, diagnostics), timeout=self.timeout,
            )
        except (asyncio.TimeoutError, httpx.TimeoutException):
            raise BackendError("timeout") from None
        if not isinstance(result, Generated) or not isinstance(result.kind, str) or result.kind not in KINDS:
            raise BackendError("invalid_response")
        if not isinstance(result.answer, str) or not result.answer.strip() or len(result.answer) > ANSWER_LIMIT:
            raise BackendError("invalid_response")
        if not isinstance(result.source_ids, tuple) or any(
            not isinstance(source_id, str) or source_id not in citations for source_id in result.source_ids
        ):
            raise BackendError("invalid_response")
        if result.kind in {"answer", "partial"} and not result.source_ids:
            raise BackendError("invalid_response")
        sources = tuple(citations[source_id] for source_id in dict.fromkeys(result.source_ids))
        return Answer(result.kind, result.answer.strip(), sources)

    def fragment(self, chunk_id: str) -> dict | None:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return None
        return {**source_record(chunk.chunk_id, chunk.source_reference, chunk.metadata), "text": chunk.text}


def env_number(name: str, default: int | float, lower: int | float, upper: int | float):
    raw = os.getenv(name, "")
    try:
        value = type(default)(raw)
    except ValueError:
        return default
    return value if lower <= value <= upper else default


def data_directory() -> Path:
    return Path(os.getenv("DATA_DIR") or "data")


def first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return ""


class ModelProxy:
    """OpenAI-compatible transport without ML2's CLI fallback or dotenv loading."""

    def __init__(self, client: httpx.AsyncClient, extract: Callable[[dict], str]):
        base = first_env("LLM_BASE_URL", "OPENAI_BASE_URL")
        self.key = first_env("LLM_API_KEY", "OPENAI_API_KEY")
        self.model = first_env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL")
        try:
            url = urlsplit(base)
            if (
                url.scheme not in {"http", "https"} or not url.hostname
                or url.username is not None or url.query or url.fragment or url.port == 0
            ):
                raise ValueError
        except ValueError:
            raise BackendError("unavailable") from None
        if not self.key or not self.model:
            raise BackendError("unavailable")
        self.url = base.rstrip("/") + "/chat/completions"
        self.client = client
        self.extract = extract

    async def complete(self, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, str | None]:
        payload = {
            "model": self.model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False,
        }
        response = await self._request(payload)
        try:
            data = response.json()
            return self.extract(data), data["choices"][0].get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise BackendError("invalid_response") from None

    async def _request(self, payload: dict) -> httpx.Response:
        error = "upstream"
        for attempt in range(2):
            try:
                response = await self.client.post(
                    self.url, headers={"Authorization": f"Bearer {self.key}"}, json=payload,
                )
                if response.is_success:
                    return response
                if response.status_code != 429 and response.status_code < 500:
                    raise BackendError("upstream")
            except httpx.TimeoutException:
                error = "timeout"
            except httpx.RequestError:
                error = "upstream"
            else:
                error = "upstream"
            if attempt == 0:
                await asyncio.sleep(0.3)
        raise BackendError(error)


class _BlockingModel:
    """Connect ML2's synchronous calls to cancellable I/O on the main event loop."""

    def __init__(self, proxy: ModelProxy):
        self.proxy = proxy
        self.loop = asyncio.get_running_loop()
        self.lock = Lock()
        self.cancelled = False
        self.pending: Future | None = None
        self.last_finish_reason: str | None = None

    def complete(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        with self.lock:
            if self.cancelled:
                raise BackendError("timeout")
            pending = asyncio.run_coroutine_threadsafe(
                self.proxy.complete(messages, temperature, max_tokens), self.loop,
            )
            self.pending = pending
        try:
            text, self.last_finish_reason = pending.result()
            return text
        except CancelledError:
            raise BackendError("timeout") from None

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            if self.pending is not None:
                self.pending.cancel()


class ML2Generator:
    def __init__(self, client: httpx.AsyncClient):
        try:
            if FLAT_RUNTIME:
                from client import ChatClient
                from ml2_solution import LegalAssistant, load_system_prompt
                from validator import CITE_RE
            else:
                from ML2.llmguard.client import ChatClient
                from ML2.llmguard.validator import CITE_RE
                from ML2.solution import LegalAssistant, load_system_prompt

            self.system_prompt = load_system_prompt()
        except (ImportError, OSError):
            raise BackendError("unavailable") from None
        self.assistant_class = LegalAssistant
        self.cite_re = CITE_RE
        self.proxy = ModelProxy(client, ChatClient._extract)
        self.max_tokens = env_number("MAX_TOKENS", 1024, 1, 8192)
        self.temperature = env_number("TEMPERATURE", 0.0, 0.0, 2.0)

    async def __call__(self, question: Question, chunks: list[RetrievedChunk], diagnostics: dict) -> Generated:
        model = _BlockingModel(self.proxy)
        assistant = self.assistant_class(
            retriever=None, llm=model, max_tokens=self.max_tokens, system_prompt=self.system_prompt,
            temperature=self.temperature,
        )
        history = [{"role": turn.role, "content": turn.content} for turn in question.history]
        try:
            answer = await asyncio.to_thread(
                assistant.answer_prepared, "", question.text, chunks, diagnostics, history=history,
            )
        finally:
            model.cancel()
        if not answer.validated or re.search(r"\[\s*\d+\s*\]", answer.text):
            raise BackendError("invalid_response")
        if answer.is_refusal:
            kind = "refusal" if answer.citations else "insufficient"
        else:
            kind = "answer"
        source_ids = tuple(dict.fromkeys(answer.citation_map[eid] for eid in answer.citations))
        numbers = {source_id: index for index, source_id in enumerate(source_ids, 1)}

        def citation(match: re.Match) -> str:
            source_id = answer.citation_map.get("E" + match[1])
            return f"[{numbers[source_id]}]" if source_id in numbers else ""

        return Generated(kind, self.cite_re.sub(citation, answer.text), source_ids)


def create_service(
    directory: Path,
    client: httpx.AsyncClient,
    *,
    generator: Generator | None = None,
) -> Service:
    if generator is None:
        generator = ML2Generator(client)
    # In-memory index from the supplied archive, without optional network models.
    settings = Settings()
    indexes = build_indexes(directory / "knowledge-base.zip", settings, dense=False)
    retriever = Retriever(indexes, settings)
    return Service(
        retriever,
        generator,
        top_k=env_number("TOP_K", 5, 1, 12),
        timeout=env_number("PSB_REQUEST_TIMEOUT", 25.0, 0.1, 90.0),
        chunks={chunk.chunk_id: chunk for chunk in indexes.chunks},
    )

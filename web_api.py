"""HTTP adapter for the frontend. Start with python -m uvicorn web_api:app."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal
from uuid import uuid4

from starlette.exceptions import HTTPException as StarletteHTTPException

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dialog_store import DialogStore, DialogError
from psb_backend import REQUEST_ID, AnswerKind, BackendError, Generator, Question, Turn, create_service, data_directory, env_number

LOG = logging.getLogger(__name__)
ERRORS = {
    "unavailable": (503, "Сервис ответа пока недоступен."),
    "timeout": (504, "Истекло время ожидания ответа."),
    "upstream": (502, "Не удалось получить ответ от сервиса генерации."),
    "invalid_response": (502, "Сервис генерации вернул некорректный ответ."),
}


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)

    @field_validator("history")
    @classmethod
    def check_history_size(cls, history: list[ChatTurn]) -> list[ChatTurn]:
        if sum(len(turn.content) for turn in history) > 12_000:
            raise ValueError("History exceeds 12000 characters")
        return history


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)


class RenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=120)


class Source(BaseModel):
    chunk_id: str
    source_reference: str
    doc_id: str
    doc_title: str
    doc_type: str
    doc_number: str | None
    doc_date: str | None
    file_path: str
    article: str | None
    part: str | None
    clause: str | None
    subclause: str | None
    section: str | None
    appendix: str | None
    amends_clause: str | None
    edition_date: str | None
    effective_date: str | None
    validity_period: dict | None
    snapshot_date: str | None
    version_status: str | None
    temporal_status: str | None
    parent_doc_id: str | None
    provision_effective_date: str | None
    amendment_application_date: str | None
    amendment_application_relation: str | None
    char_start: int
    char_end: int


class Fragment(Source):
    text: str
    request_id: str


class ChatResponse(BaseModel):
    kind: AnswerKind
    answer: str
    sources: list[Source]
    request_id: str


def http_error(code: str, status: int, message: str) -> HTTPException:
    return HTTPException(status, detail={"code": code, "message": message, "reason": "component_unavailable" if code == "unavailable" else "internal_error" if code == "internal" else "not_found"})


def create_app(*, generator: Generator | None = None) -> FastAPI:
    for logger_name in (__name__, "psb_backend"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
            logger.addHandler(handler)
            logger.propagate = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        workers = env_number("MAX_WORKERS", 2, 1, 8)
        async with httpx.AsyncClient(
            trust_env=False, timeout=httpx.Timeout(10.0, read=90.0),
            limits=httpx.Limits(max_connections=workers),
        ) as client:
            try:
                app.state.service = create_service(
                    data_directory(), client, generator=generator,
                )
            except Exception as exc:
                LOG.error("Backend initialization failed (%s)", type(exc).__name__)
                app.state.service = None
            app.state.dialogs = DialogStore(Path(os.getenv("PSB_DB_PATH", "state/dialogs.sqlite3")))
            yield

    app = FastAPI(title="PSB assistant API", version="0.1.0", lifespan=lifespan)
    origins = [origin.strip() for origin in os.getenv("PSB_CORS_ORIGINS", "").split(",") if origin.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "PATCH", "DELETE"],
                           allow_headers=["Content-Type"])

    @app.middleware("http")
    async def identify(request: Request, call_next):
        request.state.request_id = uuid4().hex
        token = REQUEST_ID.set(request.state.request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        finally:
            REQUEST_ID.reset(token)

    def failure(request: Request, status: int, code: str, message: str, reason: str):
        return JSONResponse(status_code=status, content={"request_id": request.state.request_id,
            "detail": {"code": code, "message": message, "reason": reason}})

    @app.exception_handler(StarletteHTTPException)
    async def http_failure(request: Request, error: StarletteHTTPException):
        detail = error.detail if isinstance(error.detail, dict) else {}
        return failure(request, error.status_code, detail.get("code", "not_found"),
                       detail.get("message", "Запрос не выполнен."), detail.get("reason", "not_found"))

    @app.exception_handler(DialogError)
    async def dialog_failure(request: Request, error: DialogError):
        busy = str(error) == "busy"
        return failure(request, 409 if busy else 404, "dialog_busy" if busy else "dialog_not_found",
                       "Диалог уже обрабатывается. Повторите запрос позже." if busy else "Диалог не найден.",
                       "concurrent_request" if busy else "not_found")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        LOG.error("request_id=%s internal_type=%s", request.state.request_id, type(error).__name__)
        response = failure(request, 500, "internal", "Не удалось обработать запрос.", "internal_error")
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, _error: RequestValidationError):
        return failure(request, 422, "invalid_request", "Проверьте формат вопроса и истории.", "validation_error")

    def service_for(request: Request):
        service = request.app.state.service
        if service is None:
            raise http_error("unavailable", *ERRORS["unavailable"])
        return service

    @app.get("/health/ready")
    async def ready(request: Request):
        service_for(request)
        return {"status": "ready", "request_id": request.state.request_id}

    async def ask(request: Request, question: Question):
        service = service_for(request)
        try:
            return ChatResponse.model_validate({**asdict(await service.ask(question)), "request_id": request.state.request_id})
        except BackendError as exc:
            code = str(exc)
            if code in ERRORS:
                status, message = ERRORS[code]
                LOG.info("request_id=%s error=%s reason=%s", request.state.request_id, code, exc.reason)
                raise HTTPException(status, detail={"code": code, "message": message, "reason": exc.reason}) from None
            LOG.error("Unexpected adapter error code")
        except Exception as exc:
            LOG.error("request_id=%s processing_failed=%s", request.state.request_id, type(exc).__name__)
        raise http_error("internal", 500, "Не удалось обработать запрос.") from None

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: Request, payload: ChatRequest):
        return await ask(request, Question(payload.question, tuple(Turn(t.role, t.content) for t in payload.history)))

    @app.get("/api/dialogs")
    async def dialogs(request: Request):
        return {"dialogs": request.app.state.dialogs.list(), "request_id": request.state.request_id}

    @app.post("/api/dialogs", status_code=201)
    async def new_dialog(request: Request):
        return {"id": request.app.state.dialogs.create(), "request_id": request.state.request_id}

    @app.get("/api/dialogs/{dialog_id}")
    async def dialog(request: Request, dialog_id: str):
        return {"id": dialog_id, "messages": request.app.state.dialogs.get(dialog_id), "request_id": request.state.request_id}

    @app.post("/api/dialogs/{dialog_id}/messages")
    async def message(request: Request, dialog_id: str, payload: MessageRequest):
        service = service_for(request)
        store = request.app.state.dialogs
        token, messages = store.claim(dialog_id, service.timeout + 30)
        try:
            history = []
            total = 0
            for previous in reversed(messages):
                content = previous["content"][:4000]
                if len(history) == 8 or total + len(content) > 12000:
                    break
                history.append(Turn(previous["role"], content))
                total += len(content)
            answer = (await ask(request, Question(payload.question, tuple(reversed(history))))).model_dump()
            answer["created_at"] = store.complete(dialog_id, token, payload.question, answer)
            return answer
        finally:
            store.release(dialog_id, token)

    @app.patch("/api/dialogs/{dialog_id}")
    async def rename(request: Request, dialog_id: str, payload: RenameRequest):
        request.app.state.dialogs.rename(dialog_id, payload.title)
        return {"id": dialog_id, "request_id": request.state.request_id}

    @app.delete("/api/dialogs/{dialog_id}")
    async def delete_dialog(request: Request, dialog_id: str):
        request.app.state.dialogs.delete(dialog_id)
        return {"deleted": True, "request_id": request.state.request_id}

    @app.delete("/api/dialogs")
    async def delete_dialogs(request: Request):
        request.app.state.dialogs.delete()
        return {"deleted": True, "request_id": request.state.request_id}

    @app.get("/api/documents/{doc_id}")
    async def document(request: Request, doc_id: str):
        result = service_for(request).documents.get(doc_id)
        if result is None:
            raise http_error("document_not_found", 404, "Документ не найден.")
        return {**result, "request_id": request.state.request_id}

    @app.get("/api/sources/{chunk_id}", response_model=Fragment)
    async def source(request: Request, chunk_id: str):
        fragment = service_for(request).fragment(chunk_id)
        if fragment is None:
            raise http_error("source_not_found", 404, "Фрагмент не найден.")
        return {**fragment, "request_id": request.state.request_id}

    static = Path(__file__).parent / "frontend" / "dist"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app


app = create_app()

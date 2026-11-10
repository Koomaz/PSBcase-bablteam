"""HTTP adapter for the frontend. Start with python -m uvicorn web_api:app."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from psb_backend import AnswerKind, BackendError, Generator, Question, Turn, create_service, data_directory, env_number

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


class ChatResponse(BaseModel):
    kind: AnswerKind
    answer: str
    sources: list[Source]


def http_error(code: str, status: int, message: str) -> HTTPException:
    return HTTPException(status, detail={"code": code, "message": message})


def create_app(*, generator: Generator | None = None) -> FastAPI:
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
            yield

    app = FastAPI(title="PSB assistant API", version="0.1.0", lifespan=lifespan)
    origins = [origin.strip() for origin in os.getenv("PSB_CORS_ORIGINS", "").split(",") if origin.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"])

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _error: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "invalid_request", "message": "Проверьте формат вопроса и истории."}},
        )

    def service_for(request: Request):
        service = request.app.state.service
        if service is None:
            raise http_error("unavailable", *ERRORS["unavailable"])
        return service

    @app.get("/health/ready")
    async def ready(request: Request):
        service_for(request)
        return {"status": "ready"}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: Request, payload: ChatRequest):
        service = service_for(request)
        question = Question(payload.question, tuple(Turn(turn.role, turn.content) for turn in payload.history))
        try:
            return ChatResponse.model_validate(asdict(await service.ask(question)))
        except BackendError as exc:
            code = str(exc)
            if code in ERRORS:
                raise http_error(code, *ERRORS[code]) from None
            LOG.error("Unexpected adapter error code")
        except Exception as exc:
            LOG.error("Question processing failed (%s)", type(exc).__name__)
        raise http_error("internal", 500, "Не удалось обработать запрос.") from None

    @app.get("/api/sources/{chunk_id}", response_model=Fragment)
    async def source(request: Request, chunk_id: str):
        fragment = service_for(request).fragment(chunk_id)
        if fragment is None:
            raise http_error("source_not_found", 404, "Фрагмент не найден.")
        return fragment

    return app


app = create_app()

"""Grounded legal assistant: standalone, deadline-aware platform entry point."""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from contender_context import ContextEngine
from contender_answer import build_messages, inspect_answer, render_answer, repair_message


def env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def number(name: str, default, lower, upper):
    try:
        value = type(default)(env(name))
    except (ValueError, TypeError):
        return default
    return max(lower, min(upper, value))


def read_questions(directory: Path) -> list[tuple[str, str]]:
    roots = list(dict.fromkeys([directory, Path.cwd(), Path(__file__).resolve().parent / directory]))
    path = next((root / name for root in roots for name in ("questions.csv", "demo_questions.csv")
                 if (root / name).is_file()), None)
    if path is None:
        raise FileNotFoundError("questions.csv missing from DATA_DIR")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("Expected CSV columns id,question")
        rows = [(str(row["id"] or "").strip(), str(row["question"] or "").strip()) for row in reader]
    if not rows or any(not case_id for case_id, _ in rows):
        raise ValueError("Input must contain nonempty IDs")
    if len({case_id for case_id, _ in rows}) != len(rows):
        raise ValueError("Input IDs must be unique")
    return rows


def publish(rows: list[tuple[str, str]], answers: list[str]) -> None:
    output = Path("submission.csv")
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False) as f:
            temporary = Path(f.name)
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(["id", "answer"])
            for (case_id, _), answer in zip(rows, answers):
                writer.writerow([case_id, answer.strip() or "Уточните, пожалуйста, вопрос и применимый документ."])
        temporary.replace(output)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def fallback(prepared: dict | None) -> str:
    if not prepared or not prepared.get("question", "").strip():
        return "Уточните, пожалуйста, вопрос, договор или нормативный документ и обстоятельства ситуации."
    warnings = prepared.get("warnings") or []
    if "document_identifier_not_found" in warnings:
        return "Указанный документ в предоставленной базе не найден. Уточните его номер или предоставьте текст документа, чтобы проверить применимые положения."
    if "insufficient_retrieval_evidence" in warnings or not prepared.get("evidence"):
        return "В предоставленных материалах недостаточно подтверждающих положений для достоверного ответа на этот вопрос. Требуется документ, регулирующий указанные обстоятельства, либо уточнение применимого договора и периода."
    snippets = []
    for item in prepared["evidence"]:
        meta = item.get("metadata") or {}
        if meta.get("packet_type") == "complete_document":
            continue
        if meta.get("temporal_status") in {"outside_period", "superseded", "historical_version_missing", "historical_version_unverified"}:
            continue
        text = re.sub(r"\s+", " ", item.get("text", "")).strip()
        if len(text) > 650:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            text = sentences[0] if len(sentences[0]) <= 650 else ""
        if not text:
            continue
        snippets.append(f'В источнике установлено: «{text}» [{item["id"]}].')
        if len(snippets) == 2:
            break
    if snippets:
        return render_answer(" ".join(snippets) + " Для полного вывода по указанным обстоятельствам требуется уточнение применимости этих положений.", prepared["evidence"])
    return "Для достоверного вывода требуется редакция документа и подтверждающие положения, применимые к указанной дате и обстоятельствам. В переданном контексте достаточных оснований для такого вывода нет."


def salvage(text: str, prepared: dict, fact_context: str) -> str | None:
    pieces = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])|\n+", text)
    accepted = []
    for piece in pieces:
        if not re.search(r"\[(?:S|E|Е|С)\s*\d+", piece, re.I):
            continue
        if not inspect_answer(piece, prepared["evidence"], fact_context):
            accepted.append(piece)
    if accepted:
        candidate = " ".join(accepted)
        if not inspect_answer(candidate, prepared["evidence"], fact_context):
            return render_answer(candidate, prepared["evidence"])
    return None


class UpstreamFailure(Exception):
    pass


class Model:
    def __init__(self, client: httpx.AsyncClient, deadline: float):
        self.client = client
        self.deadline = deadline
        base = env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/")
        self.key = env("LLM_API_KEY", "OPENAI_API_KEY")
        self.model = env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", "MODEL_NAME")
        if not base or not self.key or not self.model:
            raise ValueError("Model endpoint, API key and model ID must be supplied in environment")
        parsed = httpx.URL(base)
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("Invalid model endpoint")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.compat_endpoint = base + "/v1/chat/completions" if not base.endswith(("/v1", "/chat/completions")) else None
        self.temperature = number("TEMPERATURE", 0.0, 0.0, 1.5)
        self.tokens = number("MAX_TOKENS", 1800, 256, 8192)
        self.timeout = number("LLM_TIMEOUT", 85.0, 5.0, 150.0)
        self.calls = 0

    async def complete(self, messages: list[dict], tokens: int | None = None) -> tuple[str, str]:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": tokens or self.tokens, "stream": False}
        endpoint = self.endpoint
        reason = "upstream"
        for attempt in range(3):
            remaining = self.deadline - time.monotonic() - 3
            if remaining <= 1:
                raise UpstreamFailure("batch_budget")
            timeout = min(self.timeout, remaining)
            try:
                self.calls += 1
                response = await asyncio.wait_for(self.client.post(
                    endpoint, json=payload,
                    headers={"Authorization": "Bearer " + self.key},
                    timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
                ), timeout=timeout + 0.2)
                if response.status_code == 404 and self.compat_endpoint and endpoint != self.compat_endpoint:
                    endpoint = self.compat_endpoint
                    continue
                if response.status_code in {408, 429, 500, 502, 503, 504}:
                    reason = "http_" + str(response.status_code)
                elif not response.is_success:
                    raise UpstreamFailure("http_" + str(response.status_code))
                else:
                    data = response.json()
                    choice = data["choices"][0]
                    content = choice.get("message", {}).get("content", "")
                    if isinstance(content, list):
                        content = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
                    if isinstance(content, str) and content.strip():
                        return content.strip(), str(choice.get("finish_reason") or "stop")
                    if choice.get("finish_reason") == "length":
                        return "", "length"
                    reason = "empty_content"
            except (httpx.TimeoutException, httpx.RequestError, asyncio.TimeoutError):
                reason = "network_timeout"
            except (ValueError, KeyError, IndexError, TypeError):
                reason = "invalid_response"
            if attempt < 2 and self.deadline - time.monotonic() > 3:
                await asyncio.sleep(min(2.0, 0.5 * (2 ** attempt)))
        raise UpstreamFailure(reason)


async def answer_one(model: Model, prepared: dict, stats: Counter) -> str:
    messages = build_messages(prepared)
    evidence = prepared["evidence"]
    facts = prepared["question"] + "\n" + "\n".join(
        x["content"] for x in prepared.get("history", []) if x.get("role") == "user"
    )
    first = ""
    try:
        first, finish = await model.complete(messages)
        issues = inspect_answer(first, evidence, facts)
        if finish == "length":
            issues.append("Ответ достиг лимита токенов: выдай законченный краткий ответ с источниками.")
        if not issues:
            result = render_answer(first, evidence)
            if 0 < len(result) <= 10000:
                stats["accepted_first"] += 1
                return result
        if model.deadline - time.monotonic() > min(45, model.timeout):
            stats["repair_requested"] += 1
            if first.strip():
                messages.append({"role": "assistant", "content": first})
            messages.append(repair_message(issues))
            second, finish = await model.complete(messages, min(8192, model.tokens * 2) if finish == "length" else None)
            second_issues = inspect_answer(second, evidence, facts)
            if not second_issues and finish != "length":
                result = render_answer(second, evidence)
                if 0 < len(result) <= 10000:
                    stats["accepted_repair"] += 1
                    return result
            recovered = salvage(second, prepared, facts)
            if recovered and len(recovered) <= 10000:
                stats["salvaged"] += 1
                return recovered
        recovered = salvage(first, prepared, facts)
        if recovered and len(recovered) <= 10000:
            stats["salvaged"] += 1
            return recovered
    except UpstreamFailure as exc:
        stats[str(exc)] += 1
    except Exception as exc:
        stats["answer_error_" + type(exc).__name__] += 1
    if first:
        recovered = salvage(first, prepared, facts)
        if recovered and len(recovered) <= 10000:
            stats["salvaged"] += 1
            return recovered
    stats["source_fallback"] += 1
    return fallback(prepared)


async def run() -> None:
    started = time.monotonic()
    deadline = started + number("BATCH_SECONDS", 1160.0, 30.0, 1170.0)
    directory = Path(env("DATA_DIR", "INPUT_DIR", default="data"))
    rows = read_questions(directory)
    engine = ContextEngine(directory)
    answers = ["Для ответа требуется уточнить применимый документ и подтверждающие положения по указанным обстоятельствам."] * len(rows)
    publish(rows, answers)
    workers = number("MAX_WORKERS", 6, 1, 8)
    stats = Counter()
    pending = iter(enumerate(rows))
    completed = 0
    last_flush = time.monotonic()
    trace_path = Path("run_diagnostics.jsonl")
    async with httpx.AsyncClient(trust_env=False, limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers)) as client:
        model = Model(client, deadline)
        async def worker():
            nonlocal completed, last_flush
            for index, (case_id, question) in pending:
                if time.monotonic() > deadline - 3:
                    stats["batch_deadline"] += 1
                    return
                tick = time.monotonic()
                prepared = None
                try:
                    prepared = engine.prepare(question)
                    answers[index] = await answer_one(model, prepared, stats)
                except Exception as exc:
                    stats["case_error_" + type(exc).__name__] += 1
                    answers[index] = fallback(prepared)
                completed += 1
                diagnostic = {"id": case_id, "elapsed_seconds": round(time.monotonic()-tick, 3),
                              "source_count": len(prepared["evidence"]) if prepared else 0,
                              "context_chars": prepared.get("diagnostics", {}).get("context_chars") if prepared else 0}
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
                if completed % 8 == 0 or time.monotonic()-last_flush > 15:
                    publish(rows, answers)
                    last_flush = time.monotonic()
        try:
            await asyncio.wait_for(asyncio.gather(*(worker() for _ in range(min(workers,len(rows))))),
                                   timeout=max(1, deadline-time.monotonic()))
        except asyncio.TimeoutError:
            stats["batch_deadline"] += 1
        finally:
            publish(rows, answers)
        print(json.dumps({"rows": len(rows), "completed": completed, "model_calls": model.calls,
                          "seconds": round(time.monotonic()-started, 3), "stats": dict(stats)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as exc:
        print("Submission startup error: " + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)

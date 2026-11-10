"""ML2: точка входа слоя генерации. Читает вопросы, ищет доказательства через ML1,
строит ответ по строгому промпту, валидирует цитаты и пишет submission.csv.

Запуск на платформе:
    python solution.py            # DATA_DIR берется из окружения или из текущего каталога

Локальная проверка без прокси:
    MOCK_LLM=1 python solution.py --questions ../ML1/demo_questions.csv --out submission.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmguard.client import LLMError, make_client
from llmguard.csvio import find_questions_file, read_questions, write_submission
from llmguard.evidence import build_evidence, parse_question, premise_warnings, summarize_diagnostics
from llmguard.prompt_builder import build_messages, build_repair_message, load_system_prompt
from llmguard.validator import fallback_refusal, validate


def _import_retriever():
    """На платформе рядом лежит retrieval_runtime.py; локально пакет ML1/retrieval."""
    try:
        from retrieval_runtime import Retriever  # type: ignore

        return Retriever
    except ImportError:
        pass

    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent / "ML1"):
        if (candidate / "retrieval").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from retrieval import Retriever  # type: ignore

        return Retriever
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Не найден модуль retrieval. Положите retrieval_runtime.py рядом с solution.py "
            "или добавьте каталог ML1 в PYTHONPATH."
        ) from exc


@dataclass
class Answer:
    qid: str
    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    attempts: int = 1
    issues: list[str] = field(default_factory=list)
    evidence_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0


class LegalAssistant:
    def __init__(
        self,
        retriever: Any,
        llm: Any = None,
        top_k: int = 6,
        max_evidence: int = 6,
        max_words: int = 180,
        max_attempts: int = 2,
        max_tokens: int = 420,
        system_prompt: str | None = None,
        trace_path: str | Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or make_client()
        self.top_k = top_k
        self.max_evidence = max_evidence
        self.max_words = max_words
        self.max_attempts = max(1, max_attempts)
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or load_system_prompt()
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    # --- поиск -----------------------------------------------------------------
    def retrieve(self, question: str, history: list[dict[str, str]]) -> tuple[list[Any], dict[str, Any]]:
        try:
            if hasattr(self.retriever, "search_with_diagnostics"):
                chunks, diagnostics = self.retriever.search_with_diagnostics(
                    question, top_k=self.top_k, history=history or None
                )
                return list(chunks or []), dict(diagnostics or {})
            chunks = self.retriever.search(question, top_k=self.top_k, history=history or None)
            return list(chunks or []), {}
        except Exception as exc:  # retrieval не должен ронять весь прогон
            return [], {"warnings": [f"retrieval_error:{exc}"], "confidence": "low"}

    # --- один вопрос -----------------------------------------------------------
    def answer(self, qid: str, raw_question: str) -> Answer:
        started = time.perf_counter()
        question, history = parse_question(raw_question)
        chunks, diagnostics = self.retrieve(question, history)
        items = build_evidence(chunks, max_items=self.max_evidence)
        warnings = premise_warnings(question, items)
        diag_note = summarize_diagnostics(diagnostics)

        messages = build_messages(
            question=question,
            items=items,
            warnings=warnings,
            diagnostics=diag_note,
            system_prompt=self.system_prompt,
        )

        attempts = 0
        result = validate("", items, self.max_words)
        for attempt in range(self.max_attempts):
            attempts = attempt + 1
            try:
                raw = self.llm.complete(messages, temperature=0.0, max_tokens=self.max_tokens)
            except LLMError as exc:
                result = validate("", items, self.max_words)
                result.issues.append(f"сбой вызова модели: {exc}")
                break

            result = validate(raw, items, self.max_words)
            if result.ok:
                break

            messages.append({"role": "assistant", "content": raw})
            messages.append(build_repair_message(result.issues))

        if result.ok:
            text = result.cleaned
        elif result.citations and result.words >= 4:
            text = result.cleaned  # частично годный ответ лучше заглушки
        else:
            text = fallback_refusal(items)

        answer = Answer(
            qid=str(qid),
            question=raw_question,
            text=text,
            citations=result.citations,
            attempts=attempts,
            issues=result.issues,
            evidence_count=len(items),
            diagnostics=diagnostics,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        self._trace(answer)
        return answer

    def _trace(self, answer: Answer) -> None:
        if not self.trace_path:
            return
        row = {
            "id": answer.qid,
            "question": answer.question,
            "answer": answer.text,
            "citations": answer.citations,
            "attempts": answer.attempts,
            "issues": answer.issues,
            "evidence_count": answer.evidence_count,
            "confidence": answer.diagnostics.get("confidence"),
            "warnings": answer.diagnostics.get("warnings"),
            "elapsed_ms": answer.elapsed_ms,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.getenv("DATA_DIR") or os.getenv("INPUT_DIR")
    return Path(env) if env else Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PSB Legal Assistant — генерация ответов (ML2)")
    parser.add_argument("--questions", help="CSV с вопросами (id,question)")
    parser.add_argument("--data-dir", help="каталог данных платформы")
    parser.add_argument("--out", default="submission.csv", help="куда писать результат")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "6")))
    parser.add_argument("--limit", type=int, default=0, help="обработать только первые N вопросов")
    parser.add_argument("--trace", default="answers_trace.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    data_dir = _resolve_data_dir(args.data_dir)
    questions_path = Path(args.questions) if args.questions else find_questions_file(data_dir)
    if not questions_path or not Path(questions_path).is_file():
        print(f"Не найден файл вопросов в {data_dir}", file=sys.stderr)
        return 2

    rows = read_questions(questions_path)
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        print("Файл вопросов пуст", file=sys.stderr)
        return 2

    print(f"Вопросов: {len(rows)} · файл: {questions_path}", file=sys.stderr)

    Retriever = _import_retriever()
    started = time.perf_counter()
    kb_zip = data_dir / "knowledge-base.zip"
    kb_dir = data_dir / "knowledge-base"
    try:
        if kb_zip.is_file() or kb_dir.is_dir():
            retriever = Retriever.from_data_dir(str(data_dir))
        else:
            retriever = Retriever.from_artifacts(str(data_dir / "artifacts"))
    except Exception as exc:
        print(f"Не удалось загрузить индекс ({exc}), пробуем data_dir", file=sys.stderr)
        retriever = Retriever.from_data_dir(str(data_dir))
    print(f"Индекс загружен за {time.perf_counter() - started:.1f} c", file=sys.stderr)

    assistant = LegalAssistant(
        retriever=retriever,
        top_k=args.top_k,
        trace_path=args.trace,
    )

    answers: list[tuple[str, str]] = []
    for index, (qid, question) in enumerate(rows, start=1):
        answer = assistant.answer(qid, question)
        answers.append((qid, answer.text))
        if args.verbose:
            flag = "OK " if not answer.issues else "FIX"
            print(f"[{flag}] {qid}: {answer.evidence_count} док. · {answer.elapsed_ms} мс · "
                  f"{answer.citations} {('| ' + '; '.join(answer.issues)) if answer.issues else ''}",
                  file=sys.stderr)

    out_path = write_submission(args.out, answers)
    print(f"Готово: {out_path} · {len(answers)} строк · {time.perf_counter() - started:.1f} c", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

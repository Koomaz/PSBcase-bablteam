"""Organizer entry point. Run from the directory that will receive submission.csv."""

from __future__ import annotations

import asyncio
import csv
import sys
import traceback
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import httpx

from psb_backend import BackendError, Question, TECHNICAL_ANSWER, create_service, data_directory, env_number

# Leave one minute for platform overhead, cancellation and CSV publication.
BATCH_BUDGET = 19 * 60


def read_questions(path: Path) -> list[tuple[str, Question]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if sorted(header) != ["id", "question"]:
            raise ValueError("questions.csv must have the columns id and question")
        id_col, question_col = header.index("id"), header.index("question")
        rows = []
        ids = set()
        for row in reader:
            if len(row) != 2 or not row[id_col].strip() or row[id_col] in ids:
                raise ValueError(f"Invalid row or duplicate ID at line {reader.line_num}")
            ids.add(row[id_col])
            rows.append((row[id_col], Question.from_platform(row[question_col])))
        if not rows:
            raise ValueError("questions.csv has no cases")
        return rows


def write_submission(rows: list[tuple[str, Question]], answers: list[str]) -> None:
    path = Path("submission.csv")
    temp_path = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", newline="", delete=False) as stream:
            temp_path = Path(stream.name)
            writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(("id", "answer"))
            writer.writerows((case_id, answer) for (case_id, _), answer in zip(rows, answers, strict=True))
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


async def run() -> None:
    started = asyncio.get_running_loop().time()
    directory = data_directory()
    rows = read_questions(directory / "questions.csv")
    answers = [TECHNICAL_ANSWER] * len(rows)
    failures: Counter[str] = Counter()
    work = iter(enumerate(rows))
    workers = env_number("MAX_WORKERS", 2, 1, 8)
    async with httpx.AsyncClient(
        trust_env=False, timeout=httpx.Timeout(10.0, read=90.0),
        limits=httpx.Limits(max_connections=workers),
    ) as client:
        service = create_service(directory, client)

        async def process() -> None:
            for index, (_, question) in work:
                if asyncio.get_running_loop().time() - started >= BATCH_BUDGET:
                    failures["batch_deadline"] += 1
                    return
                try:
                    answers[index] = (await service.ask(question)).as_csv_text()
                except BackendError as exc:
                    failures[str(exc)] += 1
                except Exception as exc:
                    failures["internal"] += 1
                    frame = traceback.extract_tb(exc.__traceback__)[-1]
                    print(f"Internal failure: {type(exc).__name__} at {Path(frame.filename).name}:{frame.lineno}", file=sys.stderr)

        remaining = max(0.0, BATCH_BUDGET - (asyncio.get_running_loop().time() - started))
        try:
            await asyncio.wait_for(
                asyncio.gather(*(process() for _ in range(min(workers, len(rows))))),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            failures["batch_deadline"] = 1
    write_submission(rows, answers)
    print(f"submission.csv: {len(rows)} rows; failures: {dict(sorted(failures.items()))}", flush=True)


def main() -> int:
    try:
        asyncio.run(run())
    except (OSError, ValueError, csv.Error, BadZipFile, BackendError) as exc:
        print(f"Submission startup failed: {type(exc).__name__}. Check input files and model configuration.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

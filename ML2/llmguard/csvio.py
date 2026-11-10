"""Чтение вопросов и запись submission.csv (id,answer) без сюрпризов с кодировкой."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Iterable, Sequence

ID_KEYS = ("id", "ID", "Id", "question_id", "qid")
Q_KEYS = ("question", "Question", "query", "text", "Вопрос")
CANDIDATE_NAMES = ("questions.csv", "test.csv", "input.csv", "queries.csv",
                   "demo_questions.csv", "public_questions.csv")


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ";" if sample.count(";") > sample.count(",") else ","


def read_questions(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    delimiter = _sniff_delimiter(raw[:4096])
    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return []

    header = [cell.strip() for cell in rows[0]]
    id_index = next((header.index(k) for k in ID_KEYS if k in header), None)
    q_index = next((header.index(k) for k in Q_KEYS if k in header), None)

    if q_index is None:  # файл без заголовка
        body = rows
        id_index, q_index = 0, 1
    else:
        body = rows[1:]

    out: list[tuple[str, str]] = []
    for row in body:
        if not row or len(row) <= q_index:
            continue
        qid = row[id_index].strip() if id_index is not None and len(row) > id_index else str(len(out) + 1)
        question = row[q_index].strip()
        if question:
            out.append((qid, question))
    return out


def find_questions_file(data_dir: str | Path) -> Path | None:
    data_dir = Path(data_dir)
    for name in CANDIDATE_NAMES:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    for candidate in sorted(data_dir.glob("*.csv")):
        if "submission" in candidate.name.lower() or "answer" in candidate.name.lower():
            continue
        return candidate
    return None


def normalize_for_csv(answer: str) -> str:
    text = re.sub(r"\s+", " ", (answer or "").strip())
    return text.strip().strip('"').strip()


def write_submission(path: str | Path, rows: Iterable[Sequence[str]], encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for qid, answer in rows:
            writer.writerow([str(qid), normalize_for_csv(answer)])
    return path


def read_submission(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    if header[:2] == ["id", "answer"]:
        rows = rows[1:]
    return [(row[0], row[1] if len(row) > 1 else "") for row in rows if row]

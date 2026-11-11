"""Локальный LLM-судья: сравнивает ответы с эталоном по шкале регламента.

Регламент платформы: для каждого вопроса судья дважды сравнивает ответ команды
с baseline и выставляет 0 / 0.25 / 0.5 / 0.75 / 1. Итог — среднее. Здесь тот же
подход применяется к эталонному ответу (golden): 0.5 означает «сопоставимо
с эталоном», 1.0 — «значительно лучше».

Запуск из корня репозитория:
    python -m ML2.scripts.judge_answers \
        --questions ML1/demo_questions.csv \
        --golden ML2/.demo_output.csv \
        --candidate ML2/baseline_submission.csv \
        --trace ML2/baseline_trace.jsonl \
        --out ML2/reports/judge_baseline.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML2.llmguard.client import ChatClient, LLMError, load_dotenv  # noqa: E402

SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)

SYSTEM = (
    "Ты — независимый LLM-судья юридического хакатона. Оцениваешь ответ участника "
    "относительно эталонного ответа. Будь строг к фактам и ссылкам, но не штрафуй за "
    "краткость или за полезные верные уточнения. Отвечай только JSON."
)

PROMPT = """Сравни ответ участника с эталонным ответом на юридический вопрос.

Шкала (выбери ровно одно значение):
1.00 — ответ участника значительно лучше эталона: полнее и точнее, все ключевые элементы эталона раскрыты, ошибок нет.
0.75 — немного лучше: ключевые факты те же, есть полезное уточнение или более точная формулировка.
0.50 — сопоставим: те же ключевые факты и вывод, различия только в стиле и порядке.
0.25 — немного хуже: упущена второстепенная деталь или условие, либо добавлены нерелевантные детали; ключевой вывод верен.
0.00 — значительно хуже: упущен ключевой факт или вывод, неверная ссылка, противоречие эталону, отказ вместо ответа (или наоборот).

Жёсткие правила:
- Любой выдуманный факт (число, срок, статья, пункт, название, которого нет в доказательствах) → 0.00.
- Неверная ссылка на пункт/статью, если она подтверждает ключевой вывод → 0.00.
- Если участник отказывается, а эталон отвечает по существу (или наоборот) → 0.00.
- Лишние верные и уместные детали не штрафуй. Штрафуй нерелевантную «воду», повторение вопроса и общие рассуждения.
- Оценивай смысл, а не длину. Не требуй дословного совпадения с эталоном.
- Если эталон сам содержит спорное утверждение, оценивай участника по доказательствам, а не по эталону.

<question>
{question}
</question>

<golden>
{golden}
</golden>

<evidence>
{evidence}
</evidence>

<candidate>
{candidate}
</candidate>

Верни строго один JSON-объект без markdown и без пояснений вокруг:
{{"score": 0.0, "missing": ["что упущено"], "errors": ["фактические ошибки или выдумки"], "noise": ["нерелевантное"], "reason": "одно предложение"}}
"""


def read_answers(path: str | Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["id"]).strip(): (row.get("answer") or "").strip() for row in csv.DictReader(handle)}


def read_questions(path: str | Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["id"]).strip(): (row.get("question") or "").strip() for row in csv.DictReader(handle)}


def read_evidence(path: str | Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path or not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        blocks = []
        for item in row.get("evidence") or []:
            meta = item.get("meta") or {}
            meta_line = ", ".join(f"{key}={value}" for key, value in meta.items())
            block = f"[{item.get('eid', '?')}] {item.get('source', '')}"
            if meta_line:
                block += f"\nmetadata: {meta_line}"
            block += f"\n{item.get('text', '')}"
            blocks.append(block)
        out[str(row.get("id", "")).strip()] = "\n\n".join(blocks)
    return out


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_score(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(SCORES, key=lambda allowed: abs(allowed - number))


def judge_one(
    client: ChatClient,
    question: str,
    golden: str,
    candidate: str,
    evidence: str,
    *,
    max_tokens: int = 700,
) -> dict:
    prompt = PROMPT.format(
        question=question,
        golden=golden,
        evidence=evidence[:9000] or "(не переданы)",
        candidate=candidate,
    )
    last_error = ""
    for _ in range(2):
        try:
            raw = client.complete(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            data = _parse_json(raw)
            return {
                "score": _normalize_score(data.get("score")),
                "missing": [str(x) for x in (data.get("missing") or [])][:8],
                "errors": [str(x) for x in (data.get("errors") or [])][:8],
                "noise": [str(x) for x in (data.get("noise") or [])][:8],
                "reason": str(data.get("reason") or ""),
            }
        except (LLMError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"score": 0.0, "missing": [], "errors": [f"сбой судьи: {last_error}"], "noise": [], "reason": ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pairwise LLM-судья ответов ML2")
    parser.add_argument("--questions", default="ML1/demo_questions.csv")
    parser.add_argument("--golden", default="ML2/.demo_output.csv")
    parser.add_argument("--candidate", default="ML2/baseline_submission.csv")
    parser.add_argument("--trace", default="ML2/baseline_trace.jsonl")
    parser.add_argument("--out", default="ML2/reports/judge_baseline.jsonl")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv()
    questions = read_questions(args.questions)
    golden = read_answers(args.golden)
    candidate = read_answers(args.candidate)
    evidence = read_evidence(args.trace)
    ids = [qid for qid in questions if qid in golden and qid in candidate]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        print("Нет общих ID в вопросах, эталоне и ответах", file=sys.stderr)
        return 2

    client = ChatClient.from_env()
    client.model = args.model

    def run(qid: str) -> tuple[str, dict]:
        result = judge_one(
            client, questions[qid], golden[qid], candidate[qid], evidence.get(qid, "")
        )
        return qid, result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(run, ids))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for qid, result in results:
            handle.write(json.dumps({"id": qid, **result}, ensure_ascii=False) + "\n")

    scores = [result["score"] for _, result in results]
    mean = sum(scores) / len(scores)
    wins = sum(1 for score in scores if score > 0.5)
    losses = sum(1 for score in scores if score < 0.5)
    ties = len(scores) - wins - losses
    win_rate = wins / (wins + losses) if wins + losses else 0.5
    final = 0.7 * mean + 0.3 * win_rate

    for qid, result in results:
        if args.verbose or result["score"] < 0.5:
            details = "; ".join(filter(None, [
                ("missing: " + ", ".join(result["missing"])) if result["missing"] else "",
                ("errors: " + ", ".join(result["errors"])) if result["errors"] else "",
                ("noise: " + ", ".join(result["noise"])) if result["noise"] else "",
            ]))
            print(f"[{result['score']:.2f}] {qid}: {details or result['reason']}")

    print(
        f"\nMatch Rate (vs golden): {mean:.3f} · побед {wins} / ничьих {ties} / поражений {losses} "
        f"· Win Rate {win_rate:.3f} · итог 0.7*MR+0.3*WR = {final:.3f}\nОтчёт: {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

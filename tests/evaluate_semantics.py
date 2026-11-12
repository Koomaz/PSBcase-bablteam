"""Four-way integration ablation using real ZIP backend and scripted model/reviewer.
This measures enforcement given a verdict, NOT semantic accuracy of a real LLM.
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile
import httpx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--review", choices=("0", "1"), default="1")
    p.add_argument("--mode", choices=("baseline", "adaptive"), default="adaptive")
    a = p.parse_args()
    os.environ.update(LLM_BASE_URL="http://mock.invalid/v1", LLM_API_KEY="mock-key",
                      LLM_MODEL="mock-model", ANSWER_REVIEW=a.review, PSB_EVIDENCE_MODE=a.mode)
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(a.zip) as z:
            z.extractall(tmp)
        sys.path.insert(0, tmp)
        from retrieval_runtime import Retriever, Settings
        from evidence import build_evidence
        from psb_backend import ML2Generator, Question, BackendError, Service, Generated
        r = Retriever.from_source(a.corpus, settings=Settings(reranker_top_n=0), dense=False)
        query = "Какова гарантия по П-17/2024?"
        hits, diag = r.search_with_diagnostics(query)
        if hasattr(r, "evidence_packet"):
            hits, diag = r.evidence_packet(query, hits, diag)
        items = build_evidence(hits)
        eid = next(x["eid"] for x in items if x["meta"].get("clause") == "5.2")
        cases = [
            ("correct", f"Гарантия составляет 24 месяца с даты поставки [{eid}].", True),
            ("obligation", f"Поставщик обязан оплачивать товар в течение 24 месяцев с даты поставки [{eid}].", False),
            ("negation", f"Гарантия не действует в течение 24 месяцев с даты поставки [{eid}].", False),
            ("trigger", f"Гарантия действует 24 месяца с даты оплаты товара [{eid}].", False),
            ("uncited", f"Гарантия составляет 24 месяца с даты поставки [{eid}]. Покупатель вправе самовольно отменить все платежи поставщику.", False),
            ("unit", f"Гарантия составляет 24 рабочих дня с даты поставки [{eid}].", False),
        ]
        rows = []
        async def run():
            for name, draft, supported in cases:
                calls = []
                def handler(req):
                    body = json.loads(req.content)
                    calls.append(body)
                    is_review = any("Проверь черновик" in x["content"] for x in body["messages"])
                    output = json.dumps({"verdict": "ok" if supported else "revise",
                                         "unsupported": [] if supported else [draft]}) if is_review else draft
                    return httpx.Response(200, json={"choices": [{"message": {"content": output}, "finish_reason": "stop"}]})
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    try:
                        answer = await ML2Generator(client)(Question(query), hits, diag)
                        status, final = answer.kind, answer.answer
                    except BackendError as exc:
                        status, final = str(exc), ""
                accepted = "Гарантия" in final if supported else any(x in final for x in ("обязан оплачивать", "не действует", "с даты оплаты", "самовольно", "рабочих дня"))
                rows.append({"id": name, "calls": len(calls), "status": status, "pass": accepted if supported else not accepted, "answer": final})
            if hasattr(r, "evidence_packet"):
                q = Question("По П-17/2024 какой гарантийный срок?")
                original, d = r.search_with_diagnostics(q.text)
                packet, _ = r.evidence_packet(q.text, original, d)
                original_ids = {x.chunk_id for x in original}
                extra = next((x for x in packet if x.chunk_id not in original_ids), None)
                if a.mode == "adaptive":
                    assert extra is not None
                    async def generator(*_):
                        return Generated("answer", "Проверка ссылки [1].", (extra.chunk_id,))
                    result = await Service(r, generator).ask(q)
                    assert result.sources[0]["chunk_id"] == extra.chunk_id
                async def invalid(*_):
                    return Generated("answer", "Проверка ссылки [1].", ("unknown-id",))
                try:
                    await Service(r, invalid).ask(q)
                    raise AssertionError("unknown citation accepted")
                except BackendError:
                    pass
        asyncio.run(run())
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Passed {sum(x['pass'] for x in rows)}/{len(rows)} scripted checks; calls {[x['calls'] for x in rows]}")


if __name__ == "__main__":
    main()

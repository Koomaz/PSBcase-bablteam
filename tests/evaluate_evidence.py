"""Offline source coverage ablation, no model or external legal data.

Run each ZIP in its own process so imported baseline modules cannot contaminate it.
Gold is clause/span identity, not an answer template. This does not measure private score.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("cases", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--mode", choices=("baseline", "adaptive"), default="adaptive")
    args = p.parse_args()
    os.environ["PSB_EVIDENCE_MODE"] = args.mode
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(args.zip) as archive:
            archive.extractall(tmp)
        sys.path.insert(0, tmp)
        from retrieval_runtime import Retriever, Settings
        from evidence import build_evidence
        from prompt_builder import render_evidence
        r = Retriever.from_source(args.corpus, settings=Settings(reranker_top_n=0), dense=False)
        lookup = {c.chunk_id: c for c in r._index.chunks}
        def match(c, target):
            return all(v.lower() in c.text.lower() if k == "contains" else c.metadata.get(k) == v
                       for k, v in target.items())
        results = []
        for case in json.loads(args.cases.read_text(encoding="utf-8")):
            for target in case["gold"]:
                assert any(match(c, target) for c in lookup.values()), (case["id"], target)
            started = time.perf_counter()
            hits, diag = r.search_with_diagnostics(case["question"], top_k=5)
            if hasattr(r, "evidence_packet"):
                hits, diag = r.evidence_packet(case["question"], hits, diag)
            items = build_evidence(hits)
            chars = len(render_evidence(items))
            assert chars <= 7000
            coverage = [any(match(lookup[i["chunk_id"]], t) for i in items) for t in case["gold"]]
            results.append({"id": case["id"], "coverage": coverage,
                            "negative_pass": not items if not case["gold"] else None,
                            "chars": chars, "items": len(items), "packet_count": len(hits),
                            "route": diag.get("packet_route", "baseline"),
                            "ms": (time.perf_counter()-started)*1000})
        positive = [x for x in results if x["coverage"]]
        summary = {"macro_coverage": statistics.mean(sum(x["coverage"])/len(x["coverage"]) for x in positive),
                   "complete": sum(all(x["coverage"]) for x in positive), "positive": len(positive),
                   "negative_pass": sum(x["negative_pass"] is True for x in results),
                   "negative": sum(x["negative_pass"] is not None for x in results),
                   "max_chars": max(x["chars"] for x in results),
                   "median_ms": statistics.median(x["ms"] for x in results)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()

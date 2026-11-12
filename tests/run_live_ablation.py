"""Run A/B/C/D against the organizer proxy using identical questions and environment.
Does not submit anything to the scoring platform. Never writes keys to reports.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("questions", type=Path)
p.add_argument("knowledge_base", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
if not ((os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")) and
        (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))):
    p.error("Set organizer model/proxy credentials in the environment")
a.output.mkdir(parents=True, exist_ok=True)
for label, path, mode, review in (("A", a.baseline, "baseline", "0"),
                                  ("B", a.candidate, "baseline", "1"),
                                  ("C", a.candidate, "adaptive", "0"),
                                  ("D", a.candidate, "adaptive", "1")):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with ZipFile(path) as z:
            z.extractall(root)
        data = root / "data"
        data.mkdir()
        shutil.copyfile(a.questions, data / "questions.csv")
        shutil.copyfile(a.knowledge_base, data / "knowledge_base.zip")
        env = dict(os.environ, DATA_DIR=str(data), TEMPERATURE="0", PSB_EVIDENCE_MODE=mode, ANSWER_REVIEW=review)
        started = time.monotonic()
        result = subprocess.run([sys.executable, "-c", "import logging,runpy; logging.basicConfig(level=logging.INFO); runpy.run_path('solution.py',run_name='__main__')"], cwd=root, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=1200)
        (a.output / f"{label}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        (a.output / f"{label}.json").write_text(json.dumps({"elapsed_seconds": time.monotonic()-started,
                                                           "returncode": result.returncode}), encoding="utf-8")
        if (root / "submission.csv").exists():
            shutil.copyfile(root / "submission.csv", a.output / f"{label}.csv")
        if result.returncode:
            raise SystemExit(f"Variant {label} failed; inspect its log")

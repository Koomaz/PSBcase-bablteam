"""Unpack a first-submission ZIP and exercise it with the official demo and a local fake proxy."""

import argparse
import csv
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zipfile import ZipFile

EXPECTED_FILES = {
    "solution.py", "psb_backend.py", "retrieval_runtime.py", "requirements.txt",
    "ml2_solution.py", "client.py", "csvio.py", "evidence.py",
    "prompt_builder.py", "validator.py", "system_prompt.py",
}
FAKE_ANSWER = (
    'Посылка не подтверждается, "проверка" требует сверить условия с найденным положением '
    'и учитывать его применимость к указанным в вопросе обстоятельствам [E1].'
)
NO_EVIDENCE = (
    "В предоставленных документах ответа нет: отсутствуют релевантные положения. "
    "Для ответа нужен документ, регулирующий указанные обстоятельства."
)


def verify(zip_path: Path, questions: Path, knowledge_base: Path) -> None:
    requests = []
    failures = []

    class Proxy(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                assert self.path == "/v1/chat/completions"
                assert self.headers["Authorization"] == "Bearer local-placeholder"
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert body["model"] == "local-model"
                messages = body["messages"]
                content = next(message["content"] for message in messages if message["content"].startswith("<evidence>"))
                question = re.search(r"<question>(.*?)</question>", content, re.S)[1].strip()
                history = json.loads(messages[1]["content"])["history"] if messages[1]["content"].startswith("{") else []
                requests.append((question, history))
                answer = FAKE_ANSWER if "[E1]" in content else NO_EVIDENCE
                raw = json.dumps({"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as exc:
                failures.append(type(exc).__name__)
                self.send_error(500)

        def log_message(self, *_):
            pass

    with tempfile.TemporaryDirectory() as temporary, ThreadingHTTPServer(("127.0.0.1", 0), Proxy) as server:
        root = Path(temporary)
        with ZipFile(zip_path) as archive:
            assert len(archive.infolist()) == len(EXPECTED_FILES)
            assert set(archive.namelist()) == EXPECTED_FILES and archive.testzip() is None
            assert archive.read("requirements.txt").splitlines() == [b"httpx==0.28.1"]
            assert zip_path.stat().st_size <= 128_000_000
            archive.extractall(root)

        # The working directory deliberately differs from the script and input directories.
        work, data = root / "work", root / "input"
        work.mkdir()
        data.mkdir()
        shutil.copyfile(questions, data / "questions.csv")
        shutil.copyfile(knowledge_base, data / "knowledge-base.zip")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        env = {
            "PATH": os.environ["PATH"],
            "DATA_DIR": str(data),
            "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            "LLM_API_KEY": "local-placeholder",
            "LLM_MODEL": "local-model",
            "MAX_WORKERS": "2",
        }
        started = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-E", "-s", str(root / "solution.py")],
                cwd=work,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            elapsed = time.monotonic() - started
        finally:
            server.shutdown()
            worker.join()
        print(result.stdout, end="")
        if result.returncode:
            raise RuntimeError(f"Submission failed: {result.stderr}")
        assert not failures, failures
        assert "failures: {}" in result.stdout

        with questions.open(encoding="utf-8-sig", newline="") as stream:
            expected = [row["id"] for row in csv.DictReader(stream)]
        output = work / "submission.csv"
        assert not output.read_bytes().startswith(b"\xef\xbb\xbf")
        with output.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ["id", "answer"]
            rows = list(reader)
        assert [row["id"] for row in rows] == expected
        assert len(expected) == len(set(expected)) == 18
        assert all(set(row) == {"id", "answer"} and 0 < len(row["answer"]) <= 10_000 for row in rows)
        generated = [row for row in rows if "Источники:" in row["answer"]]
        assert generated and len(requests) >= 18
        assert all('"проверка"' in row["answer"] and "[1]" in row["answer"] for row in generated)
        assert len({question for question, history in requests if history}) == 5
        assert all(not question.startswith("Пользователь:") for question, _ in requests)
        peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak_mib = peak_rss / (1024 * 1024 if sys.platform == "darwin" else 1024)
        print(f"Verified {len(rows)}/18 rows, {len(requests)} mock model calls, {elapsed:.3f} s, peak RSS {peak_mib:.1f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("questions", type=Path)
    parser.add_argument("knowledge_base", type=Path)
    options = parser.parse_args()
    verify(options.zip_path, options.questions, options.knowledge_base)

"""Проверка входных файлов через реальный ZIP и штатную точку входа."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_submission import build


class TestSubmissionInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with ZipFile(build()) as archive:
            cls.modules = {name: archive.read(name) for name in archive.namelist()}

    def test_default_entrypoint_with_knowledge_base_names(self):
        document = (
            "ДОГОВОР ПОСТАВКИ № Т-01/2026\n"
            "1. Порядок расчётов\n"
            "1.1. Покупатель оплачивает товар в течение 30 календарных дней "
            "с даты поставки. Размер предоплаты указывается в счёте поставщика.\n"
        )
        for name in (
            "knowledge-base.zip", "knowledge_base.zip", "kb.zip",
            "knowledge-base", "knowledge_base",
        ):
            with self.subTest(source=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                for filename, content in self.modules.items():
                    (root / filename).write_bytes(content)
                data = root / "inputs"
                data.mkdir()
                with (data / "questions.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["id", "question"])
                    writer.writerow(["7", "Какой срок оплаты по договору Т-01/2026?"])
                    writer.writerow(["11", "Как определяется размер предоплаты по договору Т-01/2026?"])
                if name.endswith(".zip"):
                    with ZipFile(data / name, "w") as archive:
                        archive.writestr("договор.txt", document)
                else:
                    (data / name).mkdir()
                    (data / name / "договор.txt").write_text(document, encoding="utf-8")
                env = os.environ.copy()
                env.update({
                    "DATA_DIR": str(data), "MOCK_LLM": "1",
                    "LLM_BASE_URL": "", "OPENAI_BASE_URL": "",
                    "LLM_API_KEY": "", "OPENAI_API_KEY": "",
                    "TOP_K": "6", "MAX_WORKERS": "1",
                    "TEMPERATURE": "0", "MAX_TOKENS": "700",
                    "RERANKER_TOP_N": "0",
                })
                result = subprocess.run(
                    [sys.executable, "solution.py"], cwd=root, env=env,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                with (root / "submission.csv").open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, ["id", "answer"])
                    rows = list(reader)
                self.assertEqual([row["id"] for row in rows], ["7", "11"])
                self.assertTrue(all(row["answer"].strip() for row in rows))


if __name__ == "__main__":
    unittest.main()

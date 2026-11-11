import asyncio
import csv
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import httpx
from fastapi.testclient import TestClient

from psb_backend import Answer, BackendError, Generated, ML2Generator, Question, Service, TECHNICAL_ANSWER, create_service
from solution import read_questions, run
from web_api import create_app

SAMPLE_QUERY = "Что сказано в п. 4.2 договора № ПТ-05/2023?"
DEMO_DOCUMENT = """ДОГОВОР ПОСТАВКИ № ПТ-05/2023
от 20.06.2023

4. Оплата
4.1. Стоимость указана в спецификации.
4.2. Оплата производится в течение десяти рабочих дней.
"""
MOCK_CONFIG = {
    "OPENAI_BASE_URL": "http://localhost/v1",
    "OPENAI_API_KEY": "local-placeholder",
    "GEN_MODEL": "model-from-env",
}


@contextmanager
def workspace(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def write_questions(path, rows, header=("id", "question")):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


async def fake_generator(question, chunks, diagnostics):
    if not chunks:
        return Generated("insufficient", "В тестовом корпусе нет нужного фрагмента.")
    return Generated("answer", 'Тестовый ответ, "кавычки".\nВторая строка.', (chunks[0].chunk_id,))


class Fixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        with ZipFile(self.root / "knowledge-base.zip", "w") as archive:
            archive.writestr("документы компании__Договор поставки ПТ-05-2023 Альфа-Трейдинг.txt", DEMO_DOCUMENT)
        config = patch.dict(os.environ, {"DATA_DIR": str(self.root)}, clear=True)
        config.start()
        self.addCleanup(config.stop)


class BatchChecks(Fixture):
    def test_csv_round_trip_dialogue_and_independent_ids(self):
        rows = [
            ("001", f"Пользователь: {SAMPLE_QUERY}\nАссистент: Уточните пункт.\nПользователь: А что сказано в п. 4.2?"),
            ("2", SAMPLE_QUERY),
            ("3", 'Вопрос с "кавычками", запятой\nи новой строкой.'),
            ("4", "  "),
        ]
        write_questions(self.root / "questions.csv", rows)
        seen = {}

        async def generate(question, chunks, diagnostics):
            seen[question.text] = ([turn.role for turn in question.history], [c.metadata["clause"] for c in chunks])
            if question.text.startswith("Вопрос с"):
                raise BackendError("upstream")
            return await fake_generator(question, chunks, diagnostics)

        def load(directory, client):
            return create_service(directory, client, generator=generate)

        with workspace(self.root), patch("solution.create_service", load):
            asyncio.run(run())
        output = (self.root / "submission.csv").read_bytes()
        self.assertFalse(output.startswith(b"\xef\xbb\xbf"))
        with (self.root / "submission.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["id", "answer"])
            answers = list(reader)
        self.assertEqual([row["id"] for row in answers], ["001", "2", "3", "4"])
        self.assertEqual(seen["А что сказано в п. 4.2?"][0], ["user", "assistant"])
        self.assertIn("4.2", seen["А что сказано в п. 4.2?"][1])
        self.assertEqual(seen[SAMPLE_QUERY][0], [])
        self.assertIn(rows[2][1].replace("\r\n", "\n"), seen)
        self.assertIn('"кавычки".\nВторая строка.', answers[0]["answer"])
        self.assertIn("п. 4.2", answers[1]["answer"])
        self.assertEqual(answers[2]["answer"], TECHNICAL_ANSWER)
        self.assertIn("Уточните", answers[3]["answer"])

    def test_bad_headers_missing_fields_and_duplicate_ids(self):
        path = self.root / "questions.csv"
        for header, rows in [
            (("id", "answer"), [("1", "вопрос")]),
            (("id", "question"), []),
            (("id", "question"), [("1",)]),
            (("id", "question"), [("", "вопрос")]),
            (("id", "question"), [("1", "один"), ("1", "два")]),
        ]:
            with self.subTest(header=header, rows=rows):
                write_questions(path, rows, header)
                with self.assertRaises(ValueError):
                    read_questions(path)
        path.write_text('id,question\n1,"незакрыто', encoding="utf-8")
        with self.assertRaises(csv.Error):
            read_questions(path)

    def test_batch_deadline_still_writes_all_ids(self):
        write_questions(self.root / "questions.csv", [("1", SAMPLE_QUERY), ("2", SAMPLE_QUERY)])

        async def stalled(*_):
            await asyncio.sleep(1)

        def load(directory, client):
            return create_service(directory, client, generator=stalled)

        with workspace(self.root), patch("solution.create_service", load), patch("solution.BATCH_BUDGET", 0.02):
            asyncio.run(run())
        with (self.root / "submission.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(list(csv.reader(stream)), [["id", "answer"], ["1", TECHNICAL_ANSWER], ["2", TECHNICAL_ANSWER]])

    def test_csv_limit_includes_canonical_citations(self):
        self.assertEqual(len(Answer("insufficient", "я" * 10_000, ()).as_csv_text()), 10_000)
        with self.assertRaisesRegex(BackendError, "^invalid_response$"):
            Answer("answer", "я" * 10_000, ({"source_reference": "п. 4.2"},)).as_csv_text()


MODEL_TEXT = (
    'Проверяемое положение содержит описание применимого порядка оплаты, '
    'которое следует использовать при проверке условий договора по вопросу пользователя [E1].'
)
NO_EVIDENCE_TEXT = (
    'В предоставленных документах ответа нет: отсутствуют релевантные положения. '
    'Для ответа нужен документ, регулирующий указанные обстоятельства.'
)


def completion(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


class ModelChecks(Fixture):
    def ask(self, handler, question=SAMPLE_QUERY):
        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await create_service(self.root, client).ask(Question(question))

        with patch.dict(os.environ, MOCK_CONFIG):
            return asyncio.run(execute())

    def test_request_uses_env_and_passes_evidence(self):
        calls = 0

        async def proxy(request):
            nonlocal calls
            calls += 1
            self.assertEqual(str(request.url), "http://localhost/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer local-placeholder")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "model-from-env")
            user_data = body["messages"][-1]["content"]
            self.assertIn(SAMPLE_QUERY, user_data)
            self.assertIn("Оплата производится", user_data)
            self.assertIn("clause=4.2", user_data)
            return completion(MODEL_TEXT)

        answer = self.ask(proxy)
        self.assertEqual(calls, 1)
        self.assertIn("[1]", answer.answer)
        self.assertNotIn("[E1]", answer.answer)
        self.assertEqual(answer.sources[0]["clause"], "4.2")

    def test_no_evidence_returns_ml2_insufficient_without_sources(self):
        def proxy(request):
            self.assertIn("не найдено ни одного релевантного фрагмента", json.loads(request.content)["messages"][-1]["content"])
            return completion(NO_EVIDENCE_TEXT)

        answer = self.ask(proxy, "рецепт яблочного пирога")
        self.assertEqual(answer.kind, "insufficient")
        self.assertEqual(answer.sources, ())

    def test_retry_server_error_and_reject_unknown_sources(self):
        calls = 0

        def proxy(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(503)
            return completion(MODEL_TEXT.replace("[E1]", "[E99]"))

        with self.assertRaisesRegex(BackendError, "^invalid_response$"):
            self.ask(proxy)
        self.assertEqual(calls, 3)

    def test_raw_numeric_citation_cannot_bypass_source_mapping(self):
        def proxy(_):
            return completion(MODEL_TEXT + " Дополнительная ссылка [99].")

        with self.assertRaisesRegex(BackendError, "^invalid_response$"):
            self.ask(proxy)

    def test_uncited_fabricated_quantity_is_rejected(self):
        def proxy(_):
            return completion(MODEL_TEXT + " За просрочку покупатель уплачивает штраф 999999 рублей.")
        with self.assertRaisesRegex(BackendError, "^invalid_response$"):
            self.ask(proxy)

    def test_neighbor_citation_uses_full_metadata(self):
        def proxy(request):
            content = json.loads(request.content)["messages"][-1]["content"]
            self.assertIn("[E2]", content)
            return completion(MODEL_TEXT.replace("[E1]", "[E2]"))

        answer = self.ask(proxy)
        self.assertEqual(answer.sources[0]["clause"], "4.1")
        self.assertIn("[1]", answer.answer)

    def test_auth_error_and_total_timeout(self):
        calls = 0

        def auth_failure(_):
            nonlocal calls
            calls += 1
            return httpx.Response(401, text="internal details")

        with self.assertRaisesRegex(BackendError, "^upstream$"):
            self.ask(auth_failure)
        self.assertEqual(calls, 1)

        async def slow(_):
            await asyncio.sleep(1)

        with patch.dict(os.environ, {"PSB_REQUEST_TIMEOUT": "0.1"}):
            with self.assertRaisesRegex(BackendError, "^timeout$"):
                self.ask(slow)

    def test_missing_configuration(self):
        with self.assertRaisesRegex(BackendError, "^unavailable$"):
            ML2Generator(None)

    def test_retrieval_failure_is_unavailable_without_internal_details(self):
        class BrokenRetriever:
            def search_with_diagnostics(self, *_args, **_kwargs):
                raise RuntimeError("private-path")

        async def execute():
            service = Service(BrokenRetriever(), fake_generator)
            await service.ask(Question(SAMPLE_QUERY))

        with self.assertRaisesRegex(BackendError, "^unavailable$") as caught:
            asyncio.run(execute())
        self.assertNotIn("private-path", str(caught.exception))


class ApiChecks(Fixture):
    def test_answer_opens_exact_fragment(self):
        seen = []

        async def generator(question, chunks, diagnostics):
            seen.append(question)
            return await fake_generator(question, chunks, diagnostics)

        with TestClient(create_app(generator=generator)) as client:
            self.assertEqual(client.get("/health/ready").json(), {"status": "ready"})
            answer = client.post("/api/chat", json={
                "question": SAMPLE_QUERY,
                "history": [{"role": "user", "content": "Речь о ПТ-05/2023."}],
            })
            self.assertEqual(answer.status_code, 200)
            source = answer.json()["sources"][0]
            self.assertEqual(source["clause"], "4.2")
            fragment = client.get("/api/sources/" + source["chunk_id"])
            self.assertEqual(fragment.status_code, 200)
            body = fragment.json()
            self.assertEqual(body["doc_id"], source["doc_id"])
            self.assertEqual(body["text"], DEMO_DOCUMENT[source["char_start"]:source["char_end"]])
            self.assertEqual(client.get("/api/sources/unknown").status_code, 404)
            self.assertEqual(client.get("/api/sources/..%2F..%2F.env").status_code, 404)
            self.assertEqual(client.post("/api/chat", json={"question": SAMPLE_QUERY}).status_code, 200)
        self.assertEqual(len(seen[0].history), 1)
        self.assertEqual(seen[1].history, ())

    def test_rejects_empty_invalid_role_and_oversized_history(self):
        payloads = [
            {"question": "  "},
            {"question": SAMPLE_QUERY, "history": [{"role": "system", "content": "private-marker"}]},
            {"question": SAMPLE_QUERY, "history": [{"role": "user", "content": "я" * 4000}] * 4},
        ]
        with TestClient(create_app(generator=fake_generator)) as client:
            for payload in payloads:
                with self.subTest(payload_type=list(payload)):
                    response = client.post("/api/chat", json=payload)
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()["detail"]["code"], "invalid_request")
                    self.assertNotIn("private-marker", response.text)

    def test_missing_ml2_is_unavailable_and_never_falls_back(self):
        with patch.dict(os.environ, MOCK_CONFIG), patch.dict(sys.modules, {"ML2.solution": None}):
            with TestClient(create_app()) as client:
                self.assertEqual(client.get("/health/ready").status_code, 503)
                response = client.post("/api/chat", json={"question": SAMPLE_QUERY})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["detail"]["code"], "unavailable")

    def test_clean_upstream_errors(self):
        for code, status in [("upstream", 502), ("timeout", 504), ("hidden-api-detail", 500)]:
            with self.subTest(code=code):
                async def generator(*_):
                    raise BackendError(code)

                with TestClient(create_app(generator=generator)) as client:
                    self.assertEqual(client.get("/health/ready").status_code, 200)
                    response = client.post("/api/chat", json={"question": SAMPLE_QUERY})
                    self.assertEqual(response.status_code, status)
                    self.assertNotIn("hidden-api-detail", response.text)


if __name__ == "__main__":
    unittest.main()

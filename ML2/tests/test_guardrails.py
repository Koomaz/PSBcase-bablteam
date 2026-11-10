"""Тесты guardrails-слоя. Запуск: python -m unittest discover -s tests -v (из каталога ML2)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmguard.citations import human_reference, render_citations
from llmguard.csvio import normalize_for_csv, read_questions, read_submission, write_submission
from llmguard.evidence import build_evidence, parse_question, premise_warnings
from llmguard.prompt_builder import build_messages, load_system_prompt
from llmguard.validator import clean_text, fallback_refusal, validate
from solution import LegalAssistant


def chunk(cid, text, source="ДОГОВОР № А-08/2024, п. 3.1", **meta):
    payload = {"doc_type": "contract", "doc_number": "А-08/2024"}
    payload.update(meta)
    return SimpleNamespace(
        chunk_id=cid,
        text=text,
        source_reference=source,
        metadata=payload,
        score=0.9,
    )


class StubRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def search_with_diagnostics(self, query, top_k=5, history=None):
        return self._chunks[:top_k], {"warnings": ["reranker_not_configured"], "confidence": "review"}


class ScriptedLLM:
    """Отдаёт заранее заданные ответы по очереди."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def complete(self, messages, temperature=0.0, max_tokens=450):
        self.calls += 1
        return self.answers.pop(0) if self.answers else ""


class TestQuestionParsing(unittest.TestCase):
    def test_plain(self):
        question, history = parse_question("Какой срок испытания?")
        self.assertEqual(question, "Какой срок испытания?")
        self.assertEqual(history, [])

    def test_dialog(self):
        raw = (
            "Пользователь: Какой гарантийный срок по договору с «Альфа-Трейд»?\n"
            "Ассистент: 24 месяца с даты поставки (п. 5.2).\n"
            "Пользователь: А за сколько дней проверить качество?"
        )
        question, history = parse_question(raw)
        self.assertEqual(question, "А за сколько дней проверить качество?")
        self.assertEqual([turn["role"] for turn in history], ["user", "assistant", "user"])

    def test_dialog_without_final_user_line(self):
        raw = "Пользователь: Вопрос один\nАссистент: Ответ один"
        question, history = parse_question(raw)
        self.assertEqual(question, "Вопрос один")
        self.assertEqual(len(history), 2)


class TestEvidence(unittest.TestCase):
    def test_numbering_and_dedupe(self):
        items = build_evidence(
            [
                chunk("c1", "3.1. Арендная плата 450 000 рублей."),
                chunk("c1", "3.1. Арендная плата 450 000 рублей."),
                chunk("c2", "3.2. Плата вносится до 5-го числа."),
            ]
        )
        self.assertEqual([item["eid"] for item in items], ["E1", "E2"])
        self.assertEqual(items[0]["text"], "3.1. Арендная плата 450 000 рублей.")

    def test_max_items(self):
        items = build_evidence([chunk(f"c{i}", f"текст {i}") for i in range(10)], max_items=3)
        self.assertEqual(len(items), 3)

    def test_premise_number(self):
        items = build_evidence([chunk("c1", "3.1. Арендная плата составляет 450 000 рублей в месяц.")])
        warnings = premise_warnings("Почему арендная плата 500 000 рублей в месяц?", items)
        self.assertTrue(any("500 000" in w for w in warnings), warnings)

    def test_premise_counterparty(self):
        items = build_evidence([chunk("c1", "Поставка по договору с ООО «Альфа-Трейд».")])
        warnings = premise_warnings("Что с гарантией у «Альфа-Трейдинг»?", items)
        self.assertTrue(any("Альфа-Трейдинг" in w for w in warnings), warnings)

    def test_no_warnings_when_all_matches(self):
        items = build_evidence([chunk("c1", "3.1. Арендная плата 450 000 рублей.")])
        self.assertEqual(premise_warnings("Какая арендная плата 450 000?", items), [])


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.items = build_evidence(
            [chunk("c1", "3.1. Арендная плата составляет 450 000 рублей в месяц.")]
        )

    def test_good_answer(self):
        result = validate("Арендная плата составляет 450 000 рублей в месяц [E1].", self.items)
        self.assertTrue(result.ok, result.issues)
        self.assertEqual(result.citations, ["E1"])

    def test_unknown_citation_is_stripped_and_flagged(self):
        result = validate("Плата 450 000 рублей в месяц [E7].", self.items)
        self.assertFalse(result.ok)
        self.assertIn("E7", " ".join(result.issues))
        self.assertNotIn("[E7]", result.cleaned)

    def test_missing_citation(self):
        result = validate("Арендная плата составляет 450 000 рублей в месяц.", self.items)
        self.assertFalse(result.ok)
        self.assertTrue(any("нет ни одной ссылки" in issue for issue in result.issues))

    def test_false_refusal(self):
        result = validate("В документах нет данных об арендной плате.", self.items)
        self.assertTrue(result.is_refusal)
        self.assertTrue(any("необоснованный отказ" in issue for issue in result.issues))

    def test_hallucinated_number(self):
        result = validate("Плата 500 000 рублей [E1].", self.items)
        self.assertTrue(any("500000" in issue for issue in result.issues), result.issues)

    def test_markdown_stripped(self):
        result = validate("**Ответ:** плата 450 000 рублей [E1].", self.items)
        self.assertNotIn("**", result.cleaned)
        self.assertTrue(result.cleaned.startswith("Ответ:"))

    def test_long_answer(self):
        result = validate("слово " * 200 + "450 000 рублей [E1].", self.items)
        self.assertTrue(any("длиннее лимита" in issue for issue in result.issues))

    def test_empty_evidence_requires_refusal(self):
        result = validate("Арендная плата 450 000 рублей.", [])
        self.assertTrue(any("доказательств нет" in issue for issue in result.issues))

    def test_no_evidence_refusal_ok(self):
        result = validate(
            "В предоставленных документах ответа нет: отсутствует нужное положение. "
            "Для ответа требуется регламент.",
            [],
        )
        self.assertTrue(result.ok, result.issues)

    def test_forbidden_generalisation(self):
        result = validate("Как правило, плата 450 000 рублей [E1].", self.items)
        self.assertTrue(any("как правило" in issue for issue in result.issues))

    def test_truncated_answer_flagged(self):
        result = validate("Плата 450 000 рублей, см. редакцию регламента от 10 [E1]", self.items)
        self.assertTrue(any("обрывается" in issue for issue in result.issues), result.issues)

    def test_false_premise_is_not_a_refusal(self):
        answer = (
            "Указанная в вопросе сумма в договоре не закреплена: плата 450 000 рублей [E1]. "
            "Сведений об изменении размера платы в документах нет."
        )
        result = validate(answer, self.items)
        self.assertFalse(result.is_refusal, "разбор ложной посылки не должен считаться отказом")
        self.assertFalse(any("отказ" in issue for issue in result.issues), result.issues)

    def test_clean_collapses_newlines(self):
        self.assertEqual(clean_text("а\n\nб   в"), "а б в")

    def test_fallback_refusal_has_citation(self):
        text = fallback_refusal(self.items)
        self.assertIn("[E1]", text)
        self.assertIn("450 000", text)


class TestCitations(unittest.TestCase):
    def setUp(self):
        self.items = [
            {"eid": "E1", "source": "ДОГОВОР № А-08/2024, п. 3.1",
             "meta": {"doc_type": "contract", "doc_number": "А-08/2024", "clause": "3.1"}},
            {"eid": "E2", "source": "ТК РФ, ст. 70",
             "meta": {"doc_type": "law", "doc_number": "ТК РФ", "article": "70"}},
            {"eid": "E3", "source": "ФЗ 152",
             "meta": {"doc_type": "law", "doc_number": "152-ФЗ", "article": "9", "part": "1"}},
            {"eid": "E4", "source": "РЕГЛАМЕНТ, п. 2.4",
             "meta": {"doc_type": "regulation", "edition_date": "2023-01-10", "clause": "2.4"}},
        ]

    def test_contract_reference(self):
        self.assertEqual(human_reference(self.items[0]), "п. 3.1 договора № А-08/2024")

    def test_law_reference(self):
        self.assertEqual(human_reference(self.items[1]), "ст. 70 ТК РФ")
        self.assertEqual(human_reference(self.items[2]), "ч. 1 ст. 9 Закона № 152-ФЗ")

    def test_regulation_keeps_edition(self):
        self.assertEqual(human_reference(self.items[3]), "п. 2.4 регламента в ред. от 10.01.2023")

    def test_merge_adjacent(self):
        text = render_citations("Плата 450 000 [E1]. Срок [E2][E3].", self.items)
        self.assertIn("(п. 3.1 договора № А-08/2024)", text)
        self.assertIn("(ст. 70 ТК РФ; ч. 1 ст. 9 Закона № 152-ФЗ)", text)
        self.assertNotIn("[E", text)

    def test_style_id_keeps_tokens(self):
        text = render_citations("Плата [E1].", self.items, style="id")
        self.assertEqual(text, "Плата [E1].")

    def test_unknown_id_removed(self):
        self.assertEqual(render_citations("Плата [E9].", self.items), "Плата.")

    def test_both_style(self):
        text = render_citations("Плата [E1].", self.items, style="both")
        self.assertIn("(п. 3.1 договора № А-08/2024)", text)
        self.assertIn("[E1]", text)


class TestCsv(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            rows = [("1", 'Ответ с запятой, кавычкой "x" и точкой.'), ("2", "Второй")]
            write_submission(path, rows)
            self.assertEqual(read_submission(path), [(rows[0][0], normalize_for_csv(rows[0][1])), rows[1]])

    def test_read_questions_with_bom_and_commas(self):
        import tempfile

        content = '\ufeff"id","question"\n"1","Вопрос, с запятой"\n"2","Второй"\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.csv"
            path.write_text(content, encoding="utf-8")
            rows = read_questions(path)
        self.assertEqual(rows[0], ("1", "Вопрос, с запятой"))
        self.assertEqual(len(rows), 2)


class TestPipeline(unittest.TestCase):
    def test_end_to_end_with_mock_llm(self):
        chunks = [chunk("c1", "3.1. Арендная плата составляет 450 000 рублей в месяц.")]
        assistant = LegalAssistant(retriever=StubRetriever(chunks), llm=ScriptedLLM("Плата 450 000 рублей в месяц [E1]."))
        answer = assistant.answer("1", "Какая арендная плата?")
        self.assertEqual(answer.text, "Плата 450 000 рублей в месяц (п. 3.1 договора № А-08/2024).")
        self.assertEqual(answer.citations, ["E1"])

    def test_retry_after_bad_citation(self):
        chunks = [chunk("c1", "3.1. Арендная плата составляет 450 000 рублей в месяц.")]
        llm = ScriptedLLM("Плата 450 000 [E9].", "Плата 450 000 рублей [E1].")
        answer = LegalAssistant(retriever=StubRetriever(chunks), llm=llm).answer("1", "Арендная плата?")
        self.assertEqual(llm.calls, 2)
        self.assertEqual(answer.citations, ["E1"])

    def test_empty_retrieval_gives_refusal(self):
        answer = LegalAssistant(retriever=StubRetriever([]), llm=ScriptedLLM("что-то")).answer("1", "Вопрос")
        self.assertIn("ответа нет", answer.text.lower())

    def test_repeated_failure_falls_back(self):
        chunks = [chunk("c1", "3.1. Арендная плата составляет 450 000 рублей в месяц.")]
        llm = ScriptedLLM("", "", "")
        answer = LegalAssistant(retriever=StubRetriever(chunks), llm=llm, max_attempts=2).answer("1", "Арендная плата?")
        self.assertIn("(п. 3.1 договора № А-08/2024)", answer.text)

    def test_prompt_contains_hierarchy_and_evidence(self):
        items = build_evidence([chunk("c1", "текст положения")])
        messages = build_messages("вопрос", items, ["сигнал"], "confidence=review", system_prompt=load_system_prompt())
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("ИЕРАРХИЯ ИНСТРУКЦИЙ", messages[0]["content"])
        self.assertIn("[E1]", messages[1]["content"])
        self.assertIn("<premise_warnings>\n- сигнал", messages[1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

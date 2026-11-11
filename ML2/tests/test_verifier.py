"""Тесты редакторского прохода и подмешивания действующей редакции."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmguard.editions import augment_editions, base_doc_title, question_is_dated
from llmguard.citations import collapse_duplicate_refs
from llmguard.citations import human_reference
from llmguard.postprocess import (
    ensure_capitalized, strip_meta_and_redundant, strip_unsupported_inferences,
)
from llmguard.validator import ensure_premise_gap
from llmguard.validator import _numbers
from llmguard.verifier import (
    Review, _as_list, _parse_json, build_review_repair_message, prune_reviewed_answer,
    review_minimality,
)


def chunk(cid, text, title, *, edition=None, clause="2.4", doc_type="regulation"):
    metadata = {"doc_title": title, "doc_type": doc_type, "clause": clause}
    if edition:
        metadata["edition_date"] = edition
        metadata["validity_period"] = {"from": None, "to_exclusive": "2025-03-01"}
        metadata["version_status"] = "superseded"
    return SimpleNamespace(
        chunk_id=cid,
        text=text,
        source_reference=f"{title}, п. {clause}",
        metadata=metadata,
        neighboring_context=[],
    )


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries: list[str] = []

    def search(self, query, top_k=5, history=None, filters=None):
        self.queries.append(query)
        return self.chunks[:top_k]


class TestEditions(unittest.TestCase):
    def test_base_title_strips_edition(self):
        self.assertEqual(
            base_doc_title("РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ ООО «МЕРИДИАН» (ред. от 10.01.2023)"),
            "РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ ООО «МЕРИДИАН»",
        )
        self.assertEqual(base_doc_title("Договор № 1 ред. от 01.03.2025"), "Договор № 1")

    def test_question_is_dated(self):
        self.assertTrue(question_is_dated("Договор заключали в августе 2024 года"))
        self.assertTrue(question_is_dated("Что действовало 01.03.2025?"))
        self.assertFalse(question_is_dated("Кто запускает согласование?"))

    def test_current_edition_is_inserted_after_historical(self):
        old = chunk("old", "порог 10 млн", "РЕГЛАМЕНТ (ред. от 10.01.2023)", edition="2023-01-10")
        new = chunk("new", "порог 15 млн", "РЕГЛАМЕНТ (ред. от 01.03.2025)", edition="2025-03-01")
        retriever = FakeRetriever([new])
        result, note = augment_editions(
            retriever, "Договор заключали в августе 2024 года. Нужно ли одобрение?", [old]
        )
        self.assertEqual([item.chunk_id for item in result], ["old", "new"])
        self.assertIn("edition_comparison_added", note)
        self.assertIn("действующая редакция", retriever.queries[0])

    def test_undated_question_is_untouched(self):
        old = chunk("old", "порог 10 млн", "РЕГЛАМЕНТ (ред. от 10.01.2023)", edition="2023-01-10")
        result, note = augment_editions(FakeRetriever([]), "Кто запускает согласование?", [old])
        self.assertEqual(result, [old])
        self.assertEqual(note, "")

    def test_other_document_is_not_mixed_in(self):
        old = chunk("old", "порог 10 млн", "РЕГЛАМЕНТ А (ред. от 10.01.2023)", edition="2023-01-10")
        other = chunk("other", "другое", "РЕГЛАМЕНТ Б (ред. от 01.03.2025)", edition="2025-03-01")
        result, note = augment_editions(FakeRetriever([other]), "в августе 2024", [old])
        self.assertEqual(result, [old])
        self.assertEqual(note, "")


class TestReviewParsing(unittest.TestCase):
    def test_json_with_fence(self):
        data = _parse_json('```json\n{"verdict":"revise","missing":["п. 3.3"]}\n```')
        self.assertEqual(data["verdict"], "revise")

    def test_as_list_filters_empty(self):
        self.assertEqual(_as_list([" п. 3.3 ", "", None, "нет"]), ["п. 3.3"])

    def test_issues_and_repair(self):
        review = Review(missing=["коммунальные платежи"], drop=["адрес склада"])
        self.assertEqual(len(review.issues), 2)
        message = build_review_repair_message(review)
        self.assertEqual(message["role"], "user")
        self.assertIn("коммунальные платежи", message["content"])
        self.assertIn("адрес склада", message["content"])

    def test_ok_review(self):
        review = Review()
        self.assertTrue(review.ok)
        self.assertEqual(review.issues, [])


class ReviewLLM:
    review_capable = True
    last_finish_reason = "stop"
    model = "review-test"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, temperature=0.0, max_tokens=450):
        self.calls += 1
        user = messages[-1]["content"] if messages else ""
        if "Проверь черновик" in user:
            return json.dumps(
                {
                    "verdict": "revise",
                    "missing": ["отдельная оплата коммунальных платежей"],
                    "unsupported": [],
                    "noise": [],
                    "wrong_refs": [],
                },
                ensure_ascii=False,
            )
        if "В ответе не хватает" in user:
            return (
                "Арендная плата по договору составляет 450 000 рублей в месяц, включая НДС, "
                "и вносится ежемесячно не позднее пятого числа (п. 3.1) [E1]. "
                "Коммунальные платежи и электроэнергия оплачиваются отдельно (п. 3.3) [E2]."
            )
        return (
            "Арендная плата по договору составляет 450 000 рублей в месяц, включая НДС, "
            "и вносится ежемесячно не позднее пятого числа (п. 3.1) [E1]."
        )


class TestReviewPipeline(unittest.TestCase):
    def test_review_repairs_missing_condition(self):
        from solution import LegalAssistant

        items = [
            SimpleNamespace(
                chunk_id="c1", text="3.1. Арендная плата 450 000 рублей в месяц.",
                source_reference="Договор № 1, п. 3.1",
                metadata={"doc_type": "contract", "doc_number": "1", "clause": "3.1"},
                citation_payload={"doc_type": "contract", "doc_number": "1", "clause": "3.1"},
                neighboring_context=[],
            ),
            SimpleNamespace(
                chunk_id="c2", text="3.3. Коммунальные платежи оплачиваются отдельно.",
                source_reference="Договор № 1, п. 3.3",
                metadata={"doc_type": "contract", "doc_number": "1", "clause": "3.3"},
                citation_payload={"doc_type": "contract", "doc_number": "1", "clause": "3.3"},
                neighboring_context=[],
            ),
        ]
        assistant = LegalAssistant(retriever=SimpleNamespace(), llm=ReviewLLM())
        answer = assistant.answer_prepared(
            "1", "Какая арендная плата и что с коммунальными платежами?", items, {}
        )
        self.assertIn("Коммунальные платежи", answer.text)
        self.assertIn("коммунальных платежей", " ".join(answer.review_issues).lower())
        self.assertEqual(answer.attempts, 1)


class TestPruning(unittest.TestCase):
    def test_drops_noise_sentence_and_keeps_answer(self):
        draft = (
            "Оплата вносится до 5-го числа (п. 3.2). "
            "Помещение находится по адресу: г. Химки, ул. Заводская, д. 15 (п. 1.1)."
        )
        review = Review(drop=["Помещение находится по адресу"])
        result = prune_reviewed_answer(draft, review)
        self.assertIn("Оплата вносится", result)
        self.assertNotIn("Химки", result)

    def test_unsupported_inference_is_removed(self):
        draft = (
            "К претензии прикладывается акт (п. 6.2). "
            "Поэтому дефекты должны быть зафиксированы в пятидневный срок (п. 6.1)."
        )
        review = Review(unsupported=["дефекты должны быть зафиксированы"])
        result = prune_reviewed_answer(draft, review)
        self.assertIn("акт", result)
        self.assertNotIn("зафиксированы", result)

    def test_does_not_return_empty_answer(self):
        draft = "К претензии прикладывается акт (п. 6.2)."
        review = Review(drop=["К претензии прикладывается акт"])
        self.assertEqual(prune_reviewed_answer(draft, review), draft)

    def test_skipped_review_keeps_text(self):
        draft = "Ответ (п. 1)."
        self.assertEqual(prune_reviewed_answer(draft, Review(skipped=True)), draft)


class TestNumberExtraction(unittest.TestCase):
    def test_dates_are_not_quantitative_claims(self):
        numbers = _numbers("Редакция от 10.01.2023 действует до 01.03.2025.")
        self.assertNotIn("1001", numbers)
        self.assertNotIn("103", numbers)

    def test_millions_are_expanded(self):
        self.assertIn("12000000", _numbers("Договор на 12 млн рублей."))
        self.assertIn("12000000", _numbers("Договор на 12 000 000 рублей."))


class TestPostprocess(unittest.TestCase):
    def test_meta_conclusion_is_removed(self):
        text = (
            "Оплата производится в течение 45 дней с даты поставки (п. 4.1). "
            "То есть без предоплаты оплата происходит после поставки."
        )
        result = strip_meta_and_redundant(text)
        self.assertIn("45 дней", result)
        self.assertNotIn("То есть", result)

    def test_repeated_sentence_is_removed(self):
        text = (
            "Арендодатель обязан обеспечить беспрепятственный проезд транспорта круглосуточно "
            "(п. 4.3 договора № А-08/2024). "
            "Это обязанность арендодателя по п. 4.3 договора № А-08/2024."
        )
        result = strip_meta_and_redundant(text)
        self.assertIn("круглосуточно", result)
        self.assertNotIn("Это обязанность", result)

    def test_short_answer_is_not_lost(self):
        text = "Гарантия — 24 месяца (п. 5.2)."
        self.assertEqual(strip_meta_and_redundant(text), text)

    def test_duplicate_gap_sentence_is_removed(self):
        text = (
            "Плата составляет 450 000 рублей (п. 3.1). "
            "Документов об изменении суммы нет. "
            "Арендодатель вправе повышать плату на 7% (п. 3.4). "
            "Сведений о таком уведомлении не представлено."
        )
        result = strip_meta_and_redundant(text)
        self.assertEqual(result.count("нет"), 1)
        self.assertNotIn("не представлено", result)



class TestReferenceCleanup(unittest.TestCase):
    def test_duplicate_parenthetical_reference_is_collapsed(self):
        text = "К претензии прикладывается акт (п. 6.2) (п. 6.2 договора № П-17/2024)."
        self.assertEqual(
            collapse_duplicate_refs(text),
            "К претензии прикладывается акт (п. 6.2 договора № П-17/2024).",
        )

    def test_different_references_are_kept(self):
        text = "Срок (п. 6.2) (п. 6.1 договора № П-17/2024)."
        self.assertEqual(collapse_duplicate_refs(text), text)


class TestInferenceStripping(unittest.TestCase):
    def test_quality_acceptance_is_not_fixation_deadline(self):
        text = (
            "К претензии прикладывается акт о недостатках (п. 6.2). "
            "Недостатки качества должны быть выявлены в срок приёмки — 5 рабочих дней (п. 6.1). "
            "Иных документов договор не предусматривает."
        )
        result = strip_unsupported_inferences(
            text, "Приёмка по качеству — в течение 5 рабочих дней с даты поставки."
        )
        self.assertIn("акт", result)
        self.assertNotIn("должны быть выявлены", result)


class MinimalityLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete(self, messages, temperature=0.0, max_tokens=450):
        self.calls += 1
        return self.payload


class TestMinimality(unittest.TestCase):
    def test_returns_drop_phrases(self):
        draft = (
            "Согласование запускает руководитель подразделения (п. 1.3). "
            "Срок согласования юридическим отделом — 3 рабочих дня (п. 2.2). "
            "Согласование оформляется в СЭД (п. 3.1). "
            "Лист согласования на бумажном носителе не требуется (п. 3.1)."
        )
        client = MinimalityLLM('{"drop": ["Срок согласования юридическим отделом — 3 рабочих дня (п. 2.2)."]}')
        review = review_minimality(client, "Кто запускает согласование?", draft)
        self.assertFalse(review.ok)
        self.assertEqual(len(review.drop), 1)

    def test_short_draft_is_skipped(self):
        client = MinimalityLLM('{"drop": ["что-то"]}')
        review = review_minimality(client, "Вопрос", "Короткий ответ (п. 1).")
        self.assertTrue(review.skipped)
        self.assertEqual(client.calls, 0)


class TestFinalFixes(unittest.TestCase):
    def test_capitalizes_after_removing_leading_phrase(self):
        self.assertEqual(ensure_capitalized("подписанный оригинал передан."), "Подписанный оригинал передан.")
        self.assertEqual(ensure_capitalized("Уже с заглавной."), "Уже с заглавной.")

    def test_memo_reference_includes_date(self):
        item = {
            "source": "СЛУЖЕБНАЯ ЗАПИСКА № ЮО-7",
            "meta": {"doc_type": "memo", "doc_number": "ЮО-7", "doc_date": "2025-05-12"},
        }
        self.assertIn("12.05.2025", human_reference(item))
        self.assertIn("№ ЮО-7", human_reference(item))

    def test_premise_gap_is_added_only_when_missing(self):
        text = "Посылка не подтверждается: плата составляет 450 000 рублей (п. 3.1)."
        result = ensure_premise_gap(text, ["500 000"])
        self.assertIn("500 000", result)
        self.assertIn("материалах нет", result)
        self.assertEqual(ensure_premise_gap(result, ["500 000"]), result)


if __name__ == "__main__":
    unittest.main()

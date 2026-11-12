"""Semantic gate contract tests; scripted verdicts are not an LLM quality estimate."""
import json
import unittest
from test_guardrails import chunk, ScriptedLLM, StubRetriever
from solution import LegalAssistant
from llmguard.verifier import Review, prune_reviewed_answer, review_answer

GOOD = "Гарантия — 24 месяца с даты поставки [E1]."
BAD = "Покупатель обязан оплатить товар в течение 24 месяцев с даты поставки [E1]."
OK = json.dumps({"verdict": "ok", "answerability": "sufficient", "claims": []})
REJECT = json.dumps({"verdict": "revise", "unsupported": [BAD]})


class ReviewedLLM(ScriptedLLM):
    review_capable = True


class SemanticGate(unittest.TestCase):
    def run_answer(self, *responses):
        llm = ReviewedLLM(*responses)
        source = chunk("warranty", "Гарантия — 24 месяца с даты поставки.")
        answer = LegalAssistant(StubRetriever([source]), llm=llm).answer("1", "Какова гарантия и когда начинается?")
        return answer, llm.calls

    def test_short_complete_needs_no_length_repair(self):
        answer, calls = self.run_answer(GOOD, OK)
        self.assertEqual(calls, 2)
        self.assertEqual(answer.citations, ["E1"])
        self.assertFalse(answer.is_refusal)

    def test_rejected_draft_never_returns(self):
        answer, calls = self.run_answer(BAD, REJECT, BAD, REJECT)
        self.assertEqual(calls, 4)
        self.assertTrue(answer.is_refusal)
        self.assertNotIn("оплатить", answer.text)
        self.assertIn("semantic_recheck_rejected", answer.review_issues)

    def test_repair_is_rechecked(self):
        answer, calls = self.run_answer(BAD, REJECT, GOOD, OK)
        self.assertEqual(calls, 4)
        self.assertIn("Гарантия", answer.text)
        self.assertNotIn("оплатить", answer.text)

    def test_review_failure_closes_gate(self):
        answer, calls = self.run_answer(BAD, "not JSON")
        self.assertEqual(calls, 2)
        self.assertTrue(answer.is_refusal)
        self.assertEqual(answer.citations, [])

    def test_related_evidence_allows_grounded_gap(self):
        refusal = "Не удалось подтвердить срок оплаты найденными положениями. Нужен пункт об оплате."
        answer, calls = self.run_answer(refusal, OK)
        self.assertEqual(calls, 2)
        self.assertTrue(answer.is_refusal)
        self.assertTrue(answer.validated)

    def test_false_refusal_can_be_repaired(self):
        answer, calls = self.run_answer("В документах ответа нет: отсутствует условие гарантии. Нужен пункт о гарантии.",
            json.dumps({"verdict": "revise", "missing": ["гарантия"]}), GOOD, OK)
        self.assertEqual(calls, 4)
        self.assertFalse(answer.is_refusal)

    def test_all_unsupported_prunes_to_empty(self):
        self.assertEqual(prune_reviewed_answer(BAD, Review(unsupported=[BAD])), "")

    def test_unknown_claim_citation_rejects_review(self):
        payload = {"verdict": "ok", "claims": [{"claim": GOOD, "status": "supported", "evidence_ids": ["E99"]}]}
        result = review_answer(ScriptedLLM(json.dumps(payload)), "Гарантия?", [{"eid": "E1", "text": GOOD}], GOOD)
        self.assertTrue(result.skipped)

    def test_no_evidence_needs_no_model(self):
        llm = ReviewedLLM()
        answer = LegalAssistant(StubRetriever([]), llm=llm).answer("2", "Вопрос")
        self.assertEqual(llm.calls, 0)
        self.assertTrue(answer.is_refusal)


if __name__ == "__main__":
    unittest.main()

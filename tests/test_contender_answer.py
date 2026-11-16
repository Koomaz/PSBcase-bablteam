"""Seven focused regressions from the improved 2026-11-15 submission."""
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contender_answer import inspect_answer, render_answer
from solution import Model, UpstreamFailure, answer_one

RENT = [{'id':'S1','source':'Договор аренды, п. 3.1',
         'text':'Арендная плата составляет 450 000 рублей в месяц.'}]


class AnswerChecks(unittest.TestCase):
    def test_uncited_invented_quantity_after_valid_claim(self):
        text = 'Арендная плата 450 000 рублей [S1]. Штраф составляет 9 990 007 рублей.'
        self.assertTrue(inspect_answer(text, RENT, 'Каковы платежи?'))

    def test_generic_gap_does_not_hide_uncited_invention(self):
        text = 'Арендная плата составляет 9 990 007 рублей. Другие сведения отсутствуют.'
        self.assertTrue(inspect_answer(text, RENT, 'Какова арендная плата?'))

    def test_incompatible_unit(self):
        evidence = [{'id':'S1','source':'Договор, п. 5.2','text':'Гарантия составляет 24 месяца.'}]
        self.assertTrue(inspect_answer('Гарантия — 24 дня [S1].', evidence, 'Какой гарантийный срок?'))
        self.assertFalse(inspect_answer('Гарантия — 2 года [S1].', evidence, 'Какой гарантийный срок?'))

    def test_packet_routing_marker_removed_from_published_reference(self):
        evidence = [{'id':'S1','source':'Договор № А-08/2024 [полный текст; пункты и условия приведены ниже]',
                     'text':'Аренда составляет 450 000 рублей.',
                     'metadata':{'packet_type':'complete_document'}}]
        answer = render_answer('Плата — 450 000 рублей [S1].', evidence)
        self.assertIn('Договор № А-08/2024', answer)
        self.assertNotIn('полный текст', answer)

    def test_valid_word_numeral_and_user_fact_controls(self):
        evidence = [{'id':'S1','source':'ТК РФ, ст. 70','text':'Для главных бухгалтеров — шести месяцев.'}]
        self.assertFalse(inspect_answer('Максимум — 6 месяцев [S1].', evidence, 'Каков максимум?'))
        self.assertFalse(inspect_answer('Уточните дату события: в переданных материалах недостаточно сведений.', RENT, 'Как применить норму?'))


class RuntimeChecks(unittest.IsolatedAsyncioTestCase):
    async def test_empty_length_returns_length_for_budget_increase(self):
        attempts = []
        def handle(request):
            attempts.append(json.loads(request.content)['max_tokens'])
            return httpx.Response(200,json={'choices':[{'message':{'content':''},'finish_reason':'length'}]})
        with patch.dict(os.environ, {'LLM_BASE_URL':'https://mock.invalid/v1','LLM_API_KEY':'placeholder','LLM_MODEL':'local-model'}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                model = Model(client, time.monotonic()+60)
                text, reason = await model.complete([{'role':'user','content':'test'}])
        self.assertEqual((text, reason), ('','length'))
        self.assertEqual(len(attempts), 1)

    async def test_salvage_first_when_repair_transport_fails(self):
        class RepairFails:
            deadline = time.monotonic()+500
            timeout = 85
            tokens = 1000
            calls = 0
            async def complete(self,*args,**kwargs):
                self.calls += 1
                if self.calls == 1:
                    return 'Плата — 450 000 рублей [S1]. Неподтверждённый срок — 30 дней [S99].', 'stop'
                raise UpstreamFailure('network_timeout')
        prepared = {'question':'Какой размер платы?','history':[],'evidence':RENT,'warnings':[]}
        answer = await answer_one(RepairFails(), prepared, Counter())
        self.assertTrue(answer.startswith('Плата — 450 000 рублей'), answer)
        self.assertNotIn('S99', answer)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Verify shared batch/web semantics and live source contracts with a local model transport."""
import asyncio
from collections import Counter
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import httpx
from fastapi.testclient import TestClient

from contender_context import split_dialog
from contender_service import ContenderService
from psb_backend import Question, Turn, create_service
from solution import Model, answer_one, publish, read_questions
from web_api import create_app

DOCUMENT = """ДОГОВОР ПОСТАВКИ № ПТ-05/2023
от 20.06.2023

4. Оплата
4.1. Стоимость указана в спецификации.
4.2. Оплата производится в течение десяти рабочих дней.
"""
QUERY = "Что сказано в п. 4.2 договора № ПТ-05/2023?"


class SharedServiceChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        with ZipFile(self.root / 'knowledge_base.zip', 'w') as z:
            z.writestr('Договор поставки ПТ-05-2023 Альфа-Трейдинг.txt', DOCUMENT)
        config = patch.dict(os.environ, {'DATA_DIR':str(self.root), 'LLM_BASE_URL':'http://model.invalid/v1',
            'LLM_API_KEY':'test-only', 'LLM_MODEL':'test-model', 'PSB_DB_PATH':str(self.root/'dialogs.sqlite3')}, clear=True)
        config.start();self.addCleanup(config.stop)

    def proxy(self, request):
        data=json.loads(request.content)
        self.assertEqual(str(request.url), 'http://model.invalid/v1/chat/completions')
        self.assertEqual(data['model'], 'test-model')
        content=json.loads(data['messages'][1]['content'])
        source=next(e for e in content['knowledge_evidence'] if e['metadata'].get('clause')=='4.2')
        return httpx.Response(200, json={'choices':[{'message':{'content':f"Оплата производится в течение 10 рабочих дней [{source['id']}]."},'finish_reason':'stop'}]})

    def test_default_service_uses_exact_batch_pipeline_and_clickable_sources(self):
        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(self.proxy)) as client:
                service=create_service(self.root,client)
                self.assertIsInstance(service,ContenderService)
                prepared=service.engine.prepare(QUERY)
                batch=await answer_one(Model(client,asyncio.get_running_loop().time()+90),prepared,Counter())
                web=await service.ask(Question(QUERY))
                self.assertEqual(web.kind,'answer')
                self.assertEqual(len(web.sources),1)
                self.assertEqual(web.as_csv_text().split('\n\nИсточники:')[0],batch)
                source=web.sources[0]
                fragment=service.fragment(source['chunk_id'])
                self.assertEqual(fragment['text'],DOCUMENT[source['char_start']:source['char_end']])
                self.assertEqual(source['clause'],'4.2')
                self.assertEqual(service.documents[source['doc_id']]['text'],DOCUMENT)
                restarted=ContenderService(self.root,client)
                self.assertEqual(restarted.fragment(source['chunk_id']),fragment)
        asyncio.run(execute())

    def test_complete_document_packet_links_survive_restart(self):
        def packet_proxy(request):
            content=json.loads(json.loads(request.content)['messages'][1]['content'])
            source=next(e for e in content['knowledge_evidence'] if e['metadata'].get('packet_type')=='complete_document')
            return httpx.Response(200,json={'choices':[{'message':{'content':f"Стоимость указана в спецификации [{source['id']}]."},'finish_reason':'stop'}]})
        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(packet_proxy)) as client:
                service=create_service(self.root,client)
                answer=await service.ask(Question(QUERY))
                self.assertEqual(len(answer.sources),1)
                source=answer.sources[0]
                self.assertTrue(source['chunk_id'].startswith('packet_'))
                fragment=service.fragment(source['chunk_id'])
                self.assertEqual(fragment['text'],DOCUMENT)
                self.assertEqual((source['char_start'],source['char_end']),(0,len(DOCUMENT)))
                self.assertEqual(ContenderService(self.root,client).fragment(source['chunk_id']),fragment)
                self.assertNotIn('полный текст',answer.as_csv_text())
        asyncio.run(execute())

    def test_grounded_insufficiency_keeps_no_fabricated_source(self):
        def gap_proxy(request):
            return httpx.Response(200,json={'choices':[{'message':{'content':'В предоставленной базе отсутствуют сведения о пособии по беременности и родам. Для ответа требуется соответствующий нормативный документ.'},'finish_reason':'stop'}]})
        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(gap_proxy)) as client:
                service=create_service(self.root,client)
                answer=await service.ask(Question('Каков размер пособия по беременности и родам?'))
                self.assertEqual(answer.kind,'insufficient')
                self.assertEqual(answer.sources,())
                self.assertIn('отсутствуют сведения',answer.answer)
        asyncio.run(execute())

    def test_shared_law_article_reference_contains_cited_paragraph_and_stable_span(self):
        law = """ТРУДОВОЙ КОДЕКС РОССИЙСКОЙ ФЕДЕРАЦИИ
Статья 70. Испытание при приеме на работу
Срок испытания не может превышать трех месяцев.

Для руководителей организаций и главных бухгалтеров срок испытания не может превышать шести месяцев.

Испытание при приеме на работу не устанавливается для беременных женщин.
"""
        with ZipFile(self.root/'knowledge_base.zip','w') as z:
            z.writestr('законы__ТК_ст70_test.txt',law)
        def law_proxy(request):
            evidence=json.loads(json.loads(request.content)['messages'][1]['content'])['knowledge_evidence']
            item=next(x for x in evidence if 'главных бухгалтеров' in x['text'] and x['metadata'].get('packet_type')!='complete_document')
            return httpx.Response(200,json={'choices':[{'message':{'content':f"Максимальный срок испытания главного бухгалтера составляет 6 месяцев [{item['id']}]."},'finish_reason':'stop'}]})
        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(law_proxy)) as client:
                service=create_service(self.root,client)
                answer=await service.ask(Question('Какой максимальный срок испытания главного бухгалтера по ТК РФ, ст. 70?'))
                self.assertEqual(len(answer.sources),1)
                source=answer.sources[0]
                self.assertTrue(source['chunk_id'].startswith('provision_'))
                fragment=service.fragment(source['chunk_id'])
                self.assertIn('шести месяцев',fragment['text'])
                self.assertEqual(fragment['text'],law[source['char_start']:source['char_end']])
                self.assertEqual(ContenderService(self.root,client).fragment(source['chunk_id']),fragment)
        asyncio.run(execute())

    def test_waiting_for_busy_slot_counts_towards_request_timeout(self):
        from psb_backend import BackendError
        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(self.proxy)) as client:
                service=create_service(self.root,client)
                service.timeout=0.02
                service._slots=asyncio.Semaphore(0)
                with self.assertRaisesRegex(BackendError,'^timeout$'):
                    await asyncio.wait_for(service.ask(Question(QUERY)),timeout=0.15)
        asyncio.run(execute())

    def test_dialog_history_is_prepared_with_same_stateless_splitter(self):
        question,history=split_dialog('Пользователь: '+QUERY+'\nАссистент: Ответ.\nПользователь: А сколько дней?')
        self.assertEqual(question,'А сколько дней?')
        self.assertEqual([x['role'] for x in history],['user','assistant'])
        self.assertEqual(split_dialog(QUERY),(QUERY,[]))

    def test_publish_quotes_multiline_answers_and_preserves_ids(self):
        path=self.root/'questions.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['id','question']);w.writerows([['001',QUERY],['2','']])
        rows=read_questions(self.root)
        self.assertEqual([x[0] for x in rows],['001','2'])
        previous=Path.cwd()
        try:
            os.chdir(self.root)
            publish(rows,['Ответ, "цитата"\nСтрока 2','Уточните вопрос.'])
        finally:
            os.chdir(previous)
        with (self.root/'submission.csv').open(encoding='utf-8',newline='') as f:
            result=list(csv.DictReader(f))
        self.assertEqual(result[0],{'id':'001','answer':'Ответ, "цитата"\nСтрока 2'})
        path.write_text('id,question\n1,один\n1,два\n',encoding='utf-8')
        with self.assertRaises(ValueError):read_questions(self.root)

    def test_batch_deadline_publishes_every_input_id(self):
        from solution import number, run
        path=self.root/'questions.csv'
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['id','question']);w.writerows([['001',QUERY],['2',QUERY]])
        async def stalled(*args):
            await asyncio.sleep(1)
        def short_budget(name,*args):
            return 0.01 if name=='BATCH_SECONDS' else number(name,*args)
        previous=Path.cwd()
        try:
            os.chdir(self.root)
            with patch('solution.number',short_budget),patch('solution.answer_one',stalled):
                asyncio.run(run())
        finally:
            os.chdir(previous)
        with (self.root/'submission.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual([row['id'] for row in rows],['001','2'])
        self.assertTrue(all(row['answer'].strip() for row in rows))

    def test_input_contract_rejects_invalid_ids_and_preserves_quoted_multiline(self):
        path=self.root/'questions.csv'
        for header,rows in [
            (['id','answer'],[['1','ответ']]),
            (['id','question'],[]),
            (['id','question'],[['','вопрос']]),
            (['id','question'],[['1','первый'],['1','второй']]),
        ]:
            with self.subTest(header=header,rows=rows):
                with path.open('w',encoding='utf-8',newline='') as f:
                    writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)
                with self.assertRaises(ValueError):read_questions(self.root)
        question='Пользователь: Вопрос с "кавычками", запятой\nПользователь: Уточнение?'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['id','question']);writer.writerows([['001',question],['2','']])
        self.assertEqual(read_questions(self.root),[('001',question),('2','')])

    def test_default_web_timeout_allows_slow_generation_budget(self):
        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(self.proxy)) as client:
                service=create_service(self.root,client)
                self.assertEqual(service.timeout,300)
        asyncio.run(execute())

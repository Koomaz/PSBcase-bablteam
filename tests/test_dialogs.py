import asyncio

import httpx
from unittest.mock import patch
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from dialog_store import DialogError, DialogStore
from psb_backend import BackendError, REQUEST_ID, ModelProxy
from ML2.llmguard.client import ChatClient
from test_backend import Fixture, SAMPLE_QUERY, fake_generator
from web_api import create_app


class DialogChecks(Fixture):
    def test_crud_history_restart_and_document(self):
        seen = []

        async def generator(question, chunks, diagnostics):
            seen.append((question, REQUEST_ID.get()))
            return await fake_generator(question, chunks, diagnostics)

        with TestClient(create_app(generator=generator)) as client:
            identifier = client.post('/api/dialogs').json()['id']
            other = client.post('/api/dialogs').json()['id']
            first = client.post(f'/api/dialogs/{identifier}/messages', json={'question': SAMPLE_QUERY})
            self.assertEqual(first.status_code, 200)
            body = first.json()
            self.assertEqual(body['request_id'], first.headers['x-request-id'])
            self.assertEqual(seen[0][1], body['request_id'])
            self.assertTrue(body['created_at'])
            source = body['sources'][0]
            document = client.get('/api/documents/' + source['doc_id']).json()
            fragment = client.get('/api/sources/' + source['chunk_id']).json()
            self.assertEqual(document['text'][source['char_start']:source['char_end']], fragment['text'])
            self.assertEqual(document['offset_encoding'], 'unicode_codepoints')
            self.assertEqual(client.get('/api/documents/unknown').status_code, 404)
            second = client.post(f'/api/dialogs/{identifier}/messages', json={'question': SAMPLE_QUERY})
            self.assertEqual(second.status_code, 200)
            self.assertEqual([t.role for t in seen[1][0].history], ['user', 'assistant'])
            self.assertEqual(client.post(f'/api/dialogs/{other}/messages', json={'question': SAMPLE_QUERY}).status_code, 200)
            self.assertEqual(seen[2][0].history, ())
            self.assertEqual(client.patch(f'/api/dialogs/{identifier}', json={'title': 'Проверка'}).status_code, 200)
            listing = client.get('/api/dialogs').json()['dialogs']
            self.assertEqual(listing[0]['id'], identifier)
            self.assertEqual(listing[0]['message_count'], 4)
        with TestClient(create_app(generator=generator)) as client:
            self.assertEqual(len(client.get(f'/api/dialogs/{identifier}').json()['messages']), 4)
            self.assertEqual(client.delete(f'/api/dialogs/{identifier}').status_code, 200)
            self.assertEqual(client.get(f'/api/dialogs/{identifier}').status_code, 404)
            self.assertEqual(client.delete('/api/dialogs').status_code, 200)
            self.assertEqual(client.get('/api/dialogs').json()['dialogs'], [])

    def test_failed_generation_does_not_persist_and_can_retry(self):
        failed = [True]

        async def generator(question, chunks, diagnostics):
            if failed[0]:
                raise BackendError('invalid_response', 'validation_failed')
            return await fake_generator(question, chunks, diagnostics)

        with TestClient(create_app(generator=generator)) as client:
            identifier = client.post('/api/dialogs').json()['id']
            path = f'/api/dialogs/{identifier}/messages'
            response = client.post(path, json={'question': SAMPLE_QUERY})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()['detail']['reason'], 'validation_failed')
            self.assertEqual(response.headers['x-request-id'], response.json()['request_id'])
            self.assertEqual(client.get(f'/api/dialogs/{identifier}').json()['messages'], [])
            failed[0] = False
            self.assertEqual(client.post(path, json={'question': SAMPLE_QUERY}).status_code, 200)
            self.assertEqual(len(client.get(f'/api/dialogs/{identifier}').json()['messages']), 2)

    def test_concurrent_message_and_delete_do_not_resurrect_dialog(self):
        started, release = threading.Event(), threading.Event()

        async def generator(question, chunks, diagnostics):
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            return await fake_generator(question, chunks, diagnostics)

        with TestClient(create_app(generator=generator)) as client, ThreadPoolExecutor() as pool:
            identifier = client.post('/api/dialogs').json()['id']
            path = f'/api/dialogs/{identifier}/messages'
            pending = pool.submit(client.post, path, json={'question': SAMPLE_QUERY})
            self.assertTrue(started.wait(3))
            try:
                self.assertEqual(client.post(path, json={'question': SAMPLE_QUERY}).status_code, 409)
                self.assertEqual(client.delete(f'/api/dialogs/{identifier}').status_code, 200)
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=5).status_code, 404)
            self.assertEqual(client.get('/api/dialogs').json()['dialogs'], [])

    def test_db_claim_shared_between_instances_and_expired_lease_recovers(self):
        path = self.root / 'store.sqlite3'
        first, second = DialogStore(path), DialogStore(path)
        identifier = first.create()
        token, _ = first.claim(identifier, 30)
        with self.assertRaisesRegex(DialogError, 'busy'):
            second.claim(identifier, 30)
        second.release(identifier, token)
        expired, _ = first.claim(identifier, -1)
        replacement, _ = second.claim(identifier, 30)
        first.release(identifier, expired)
        with self.assertRaisesRegex(DialogError, 'busy'):
            first.claim(identifier, 30)
        second.release(identifier, replacement)

    def test_error_reasons_and_request_ids_are_stable_and_safe(self):
        for reason in ['empty_content', 'numeric_citation', 'validation_failed']:
            async def fail(*_):
                raise BackendError('invalid_response', reason)
            with TestClient(create_app(generator=fail)) as client:
                response = client.post('/api/chat', json={'question': SAMPLE_QUERY})
                self.assertEqual(response.json()['detail']['reason'], reason)
                self.assertEqual(response.json()['request_id'], response.headers['x-request-id'])
                bad = client.post('/api/chat', json={'question': ''})
                self.assertEqual(bad.status_code, 422)
                self.assertNotEqual(bad.json()['request_id'], response.json()['request_id'])
                self.assertEqual(client.post('/api/dialogs/not-found/messages', json={'question': SAMPLE_QUERY}).status_code, 404)

    def test_transport_distinguishes_empty_content_and_malformed_payload(self):
        async def probe(payload):
            with patch.dict('os.environ', {'LLM_BASE_URL': 'http://localhost/v1', 'LLM_API_KEY': 'test-placeholder', 'LLM_MODEL': 'test-model'}):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
                    proxy = ModelProxy(client, ChatClient._extract)
                    try:
                        await proxy.complete([], 0, 20)
                    except BackendError as error:
                        return error.reason
        self.assertEqual(asyncio.run(probe({'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}]})), 'empty_content')
        self.assertEqual(asyncio.run(probe({'choices': []})), 'malformed_response')

    def test_unexpected_failure_hides_message(self):
        async def broken(*_):
            raise RuntimeError('private-configuration-detail')
        with TestClient(create_app(generator=broken)) as client:
            response = client.post('/api/chat', json={'question': SAMPLE_QUERY})
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('private-configuration-detail', response.text)

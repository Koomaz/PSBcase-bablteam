"""Persistent single-workspace dialogs. Model calls never hold a DB transaction."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import time
from uuid import uuid4


class DialogError(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DialogStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS dialogs (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, updated_at TEXT NOT NULL,
                    lease TEXT, expires REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    dialog_id TEXT NOT NULL REFERENCES dialogs(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, kind TEXT,
                    sources TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_dialog ON messages(dialog_id, seq);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self) -> str:
        identifier = uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO dialogs(id,title,updated_at) VALUES (?,?,?)',
                       (identifier, 'Новый диалог', now()))
        return identifier

    def list(self) -> list[dict]:
        with self.connection() as db:
            return [dict(row) for row in db.execute('''
                SELECT d.id,d.title,d.updated_at,COUNT(m.seq) AS message_count
                FROM dialogs d LEFT JOIN messages m ON m.dialog_id=d.id
                GROUP BY d.id ORDER BY d.updated_at DESC,d.id
            ''')]

    def _messages(self, db, identifier) -> list[dict]:
        result = []
        for row in db.execute('SELECT role,content,kind,sources,created_at FROM messages WHERE dialog_id=? ORDER BY seq', (identifier,)):
            message = dict(row)
            message['sources'] = json.loads(message['sources'])
            result.append(message)
        return result

    def get(self, identifier: str) -> list[dict]:
        with self.connection() as db:
            db.execute('BEGIN')
            if db.execute('SELECT 1 FROM dialogs WHERE id=?', (identifier,)).fetchone() is None:
                raise DialogError('not_found')
            return self._messages(db, identifier)

    def claim(self, identifier: str, seconds: float) -> tuple[str, list[dict]]:
        token = uuid4().hex
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT lease,expires FROM dialogs WHERE id=?', (identifier,)).fetchone()
            if row is None:
                raise DialogError('not_found')
            if row['lease'] and row['expires'] > time():
                raise DialogError('busy')
            db.execute('UPDATE dialogs SET lease=?,expires=? WHERE id=?', (token, time()+seconds, identifier))
            return token, self._messages(db, identifier)

    def release(self, identifier: str, token: str):
        with self.connection() as db:
            db.execute('UPDATE dialogs SET lease=NULL,expires=0 WHERE id=? AND lease=?', (identifier, token))

    def complete(self, identifier: str, token: str, question: str, answer: dict) -> str:
        created = now()
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT title,lease FROM dialogs WHERE id=?', (identifier,)).fetchone()
            if row is None:
                raise DialogError('not_found')
            if row['lease'] != token:
                raise DialogError('busy')
            first = db.execute('SELECT 1 FROM messages WHERE dialog_id=? LIMIT 1', (identifier,)).fetchone() is None
            for role, content, kind, sources in [
                ('user', question, None, []),
                ('assistant', answer['answer'], answer['kind'], answer['sources']),
            ]:
                db.execute('INSERT INTO messages(dialog_id,role,content,kind,sources,created_at) VALUES (?,?,?,?,?,?)',
                           (identifier,role,content,kind,json.dumps(sources,ensure_ascii=False),created))
            title = question[:80] if first and row['title'] == 'Новый диалог' else row['title']
            db.execute('UPDATE dialogs SET title=?,updated_at=?,lease=NULL,expires=0 WHERE id=?', (title,created,identifier))
        return created

    def rename(self, identifier: str, title: str):
        with self.connection() as db:
            if db.execute('UPDATE dialogs SET title=?,updated_at=? WHERE id=?', (title,now(),identifier)).rowcount == 0:
                raise DialogError('not_found')

    def delete(self, identifier: str | None = None):
        with self.connection() as db:
            if identifier is None:
                db.execute('DELETE FROM dialogs')
            elif db.execute('DELETE FROM dialogs WHERE id=?', (identifier,)).rowcount == 0:
                raise DialogError('not_found')

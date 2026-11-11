"""Локальная раздача фронтенда с проксированием API бэкенда.

Запуск из каталога frontend/:

    python serve.py

Страница будет доступна на http://127.0.0.1:8080, запросы /api/* и /health/*
проксируются на бэкенд (по умолчанию http://127.0.0.1:8000). Проксирование
избавляет от настройки CORS: фронт и API выглядят для браузера одним origin.

Статика отдаётся из собранного каталога dist/ (npm run build), а при его
отсутствии — из текущего каталога.

Переменные окружения:
    PSB_API_URL        адрес бэкенда (web_api.py, uvicorn)
    PSB_FRONTEND_PORT  порт статического сервера
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / "dist" if (BASE / "dist").is_dir() else BASE
API_URL = os.getenv("PSB_API_URL", "http://127.0.0.1:8000").rstrip("/")
PORT = int(os.getenv("PSB_FRONTEND_PORT", "8080"))

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "PsbFrontend/0.1"

    # --- API proxy -----------------------------------------------------

    def _is_api(self) -> bool:
        return self.path.startswith("/api/") or self.path.startswith("/health")

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {}
        if body:
            headers["Content-Type"] = self.headers.get(
                "Content-Type", "application/json"
            )
        request = urllib.request.Request(
            API_URL + self.path, data=body, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                self._relay(response.status, response.headers.get("Content-Type"), response.read())
        except urllib.error.HTTPError as error:
            self._relay(error.code, error.headers.get("Content-Type"), error.read())
        except Exception:
            payload = json.dumps(
                {
                    "detail": {
                        "code": "unavailable",
                        "message": "Сервис ответа пока недоступен.",
                    }
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self._relay(502, "application/json; charset=utf-8", payload)

    def _relay(self, status: int, content_type: str | None, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # --- static files --------------------------------------------------

    def _static(self) -> None:
        requested = self.path.split("?", 1)[0].split("#", 1)[0]
        if requested in ("/", ""):
            requested = "/index.html"
        target = (ROOT / requested.lstrip("/")).resolve()
        if ROOT.resolve() not in target.parents or not target.is_file():
            # SPA-фолбэк: неизвестные пути отдаём index.html
            index = ROOT / "index.html"
            if index.is_file() and "." not in requested.rsplit("/", 1)[-1]:
                self._relay(200, CONTENT_TYPES[".html"], index.read_bytes())
                return
            self._relay(404, "text/plain; charset=utf-8", "Не найдено".encode("utf-8"))
            return
        payload = target.read_bytes()
        self._relay(
            200, CONTENT_TYPES.get(target.suffix, "application/octet-stream"), payload
        )

    # --- verbs ---------------------------------------------------------

    def do_GET(self) -> None:
        if self._is_api():
            self._proxy("GET")
        else:
            self._static()

    def do_POST(self) -> None:
        self._proxy("POST")

    def log_message(self, fmt: str, *args) -> None:  # компактный лог в консоль
        print("%s %s" % (self.address_string(), fmt % args))


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("Фронтенд: http://127.0.0.1:%d  ->  API: %s" % (PORT, API_URL))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

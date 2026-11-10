"""Exercise Compose with the real ML pipeline and a temporary fake model proxy."""
import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from verify_submission import FAKE_ANSWER, NO_EVIDENCE


def main():
    calls = []
    mode = ["answer"]

    class Proxy(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = body["messages"]
            calls.append(messages)
            if mode[0] == "error":
                self.send_error(401)
                return
            if mode[0] == "slow":
                time.sleep(1)
            evidence = next(m["content"] for m in messages if m["content"].startswith("<evidence>"))
            text = FAKE_ANSWER if "[E1]" in evidence else NO_EVIDENCE
            data = json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    with ThreadingHTTPServer(("0.0.0.0", 0), Proxy) as proxy:
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        env = {**os.environ, "PSB_PORT": str(port), "PSB_DATA_DIR": os.path.abspath("data"),
               "LLM_BASE_URL": f"http://host.docker.internal:{proxy.server_port}/v1",
               "LLM_API_KEY": "local-test-placeholder", "LLM_MODEL": "local-test-model",
               "PSB_REQUEST_TIMEOUT": "0.3"}
        command = ["docker", "compose", "-p", "psb-backend-verification"]
        try:
            subprocess.run(command + ["up", "-d", "--wait", "--wait-timeout", "60"], env=env, check=True)
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client:
                assert client.get("/health/ready").status_code == 200
                question = "Что сказано в п. 4.2 договора ПТ-05/2023?"
                response = client.post("/api/chat", json={"question": question})
                response.raise_for_status()
                source = response.json()["sources"][0]
                fragment = client.get("/api/sources/" + source["chunk_id"])
                fragment.raise_for_status()
                assert fragment.json()["text"] and fragment.json()["doc_id"] == source["doc_id"]
                history = [{"role": "user", "content": question}]
                followup = client.post("/api/chat", json={"question": "А что сказано в п. 4.1?", "history": history})
                followup.raise_for_status()
                assert json.loads(calls[-1][1]["content"])["history"] == history
                assert client.post("/api/chat", json={"question": question}).status_code == 200
                assert calls[-1][1]["content"].startswith("<evidence>")
                empty = client.post("/api/chat", json={"question": "рецепт яблочного пирога"})
                assert empty.status_code == 200 and empty.json()["kind"] == "insufficient"
                mode[0] = "error"
                assert client.post("/api/chat", json={"question": question}).status_code == 502
                mode[0] = "slow"
                assert client.post("/api/chat", json={"question": question}).status_code == 504
                assert client.get("/api/sources/..%2F.env").status_code == 404
                print("Compose passed: readiness, chat, source, history, independent requests, no evidence, 401, timeout, path rejection")
        finally:
            subprocess.run(command + ["down"], env=env, check=True)
            proxy.shutdown()


if __name__ == "__main__":
    main()

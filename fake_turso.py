"""A stand-in for Turso's SQL-over-HTTP endpoint, backed by in-memory SQLite.

It follows the request/response format in https://docs.turso.tech/sdk/http/reference.
Used by the tests (no internet or Turso account needed). It is NOT a replacement for testing against real Turso.
Run as a server for manual tests:  python tests/fake_turso.py 8766
"""
import base64
import json
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = "test-token"


class FakeTurso:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()

    @staticmethod
    def _arg(a):
        t = a["type"]
        if t == "null":
            return None
        if t == "integer":
            return int(a["value"])
        if t == "float":
            return float(a["value"])
        if t == "blob":
            return base64.b64decode(a["base64"])
        return a["value"]

    @staticmethod
    def _cell(v):
        if v is None:
            return {"type": "null"}
        if isinstance(v, int):
            return {"type": "integer", "value": str(v)}
        if isinstance(v, float):
            return {"type": "float", "value": v}
        if isinstance(v, bytes):
            return {"type": "blob", "base64": base64.b64encode(v).decode()}
        return {"type": "text", "value": v}

    def pipeline(self, payload: dict) -> dict:
        results = []
        for req in payload["requests"]:
            if req["type"] == "close":
                results.append({"type": "ok", "response": {"type": "close"}})
                continue
            stmt = req["stmt"]
            try:
                with self.lock:
                    cur = self.conn.execute(stmt["sql"], [self._arg(a) for a in stmt.get("args", [])])
                    rows = cur.fetchall()
                    cols = [{"name": d[0], "decltype": None} for d in (cur.description or [])]
                    last = str(cur.lastrowid) if stmt["sql"].lstrip().upper().startswith("INSERT") else None
                results.append({"type": "ok", "response": {"type": "execute", "result": {
                    "cols": cols, "rows": [[self._cell(v) for v in r] for r in rows],
                    "affected_row_count": max(cur.rowcount, 0), "last_insert_rowid": last}}})
            except sqlite3.Error as e:
                results.append({"type": "error", "error": {"message": f"SQLite error: {e}", "code": "SQLITE_ERROR"}})
        return {"baton": None, "base_url": None, "results": results}


def serve(port: int):
    fake = FakeTurso()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Authorization") != f"Bearer {TOKEN}" or self.path != "/v2/pipeline":
                self.send_response(401); self.end_headers(); return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps(fake.pipeline(body)).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
        def log_message(self, *a): pass

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8766)

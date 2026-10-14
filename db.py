"""Tiny Turso (libSQL) client that talks to the documented SQL-over-HTTP endpoint.

Docs: https://docs.turso.tech/sdk/http/reference  (POST {url}/v2/pipeline, Bearer token)
Only needs `httpx`, so there is no native library to build on Render.
"""
import base64
import os

import httpx


class TursoError(Exception):
    pass


_client: httpx.AsyncClient | None = None


def configured() -> bool:
    return bool(os.getenv("TURSO_DATABASE_URL", "").strip() and os.getenv("TURSO_AUTH_TOKEN", "").strip())


def _base_url() -> str:
    url = os.getenv("TURSO_DATABASE_URL", "").strip().rstrip("/")
    for prefix in ("libsql://", "turso://", "wss://", "ws://"):
        if url.startswith(prefix):
            url = "https://" + url[len(prefix):]
    return url


def _arg(v) -> dict:
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    return {"type": "text", "value": str(v)}


def _val(cell: dict):
    t = cell.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(cell["value"])
    if t == "float":
        return float(cell["value"])
    if t == "blob":
        return base64.b64decode(cell.get("base64", ""))
    return cell.get("value")


async def execute(sql: str, args: tuple | list = ()) -> list[dict]:
    """Run one parameterised statement and return the rows as dictionaries."""
    global _client
    if not configured():
        raise TursoError("Turso is not configured")
    if _client is None:
        _client = httpx.AsyncClient(timeout=20)
    body = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": [_arg(a) for a in args]}},
            {"type": "close"},
        ]
    }
    try:
        r = await _client.post(
            _base_url() + "/v2/pipeline",
            headers={"Authorization": "Bearer " + os.getenv("TURSO_AUTH_TOKEN", "").strip()},
            json=body,
        )
    except httpx.HTTPError as e:
        raise TursoError(f"Could not reach Turso: {e.__class__.__name__}")
    if r.status_code != 200:
        raise TursoError(f"Turso returned HTTP {r.status_code}")
    first = r.json()["results"][0]
    if first.get("type") == "error":
        raise TursoError(first["error"].get("message", "Turso error"))
    result = first["response"]["result"]
    names = [c["name"] for c in result.get("cols", [])]
    return [dict(zip(names, (_val(c) for c in row))) for row in result.get("rows", [])]

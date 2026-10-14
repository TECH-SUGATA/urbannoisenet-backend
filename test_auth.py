import json
import os
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_turso import TOKEN, FakeTurso  # noqa: E402

import auth  # noqa: E402
import db  # noqa: E402
import main  # noqa: E402

GOOD = {"name": "Asha Roy", "email": "Asha@Example.com", "password": "correct-horse"}


@pytest.fixture()
def client(monkeypatch):
    fake = FakeTurso()

    def handler(request: httpx.Request):
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401)
        return httpx.Response(200, json=fake.pipeline(json.loads(request.content)))

    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://fake-db.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("JWT_SECRET", "x" * 40)
    monkeypatch.setenv("OFFICER_CODE", "CITY-2026")
    monkeypatch.setattr(db, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(auth, "_schema_ready", False)
    auth._fails.clear()
    c = TestClient(main.app)
    c.fake = fake
    return c


def signup(c, **kw):
    return c.post("/auth/signup", json={**GOOD, **kw})


def test_health_reports_auth(client):
    assert client.get("/health").json()["auth"] is True


def test_signup_login_me(client):
    r = signup(client)
    assert r.status_code == 201
    assert r.json()["user"] == {"id": 1, "name": "Asha Roy", "email": "asha@example.com", "role": "citizen"}
    r = client.post("/auth/login", json={"email": " ASHA@example.com ", "password": "correct-horse"})
    assert r.status_code == 200
    me = client.get("/auth/me", headers={"Authorization": "Bearer " + r.json()["token"]})
    assert me.status_code == 200 and me.json()["user"]["email"] == "asha@example.com"


def test_password_is_hashed_in_database(client):
    signup(client)
    row = client.fake.conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert row.startswith("scrypt$") and "correct-horse" not in row


def test_duplicate_email(client):
    signup(client)
    assert signup(client, email="asha@example.com").status_code == 409


@pytest.mark.parametrize("bad,msg", [({"password": "short"}, "8 to 128"), ({"email": "nope"}, "valid email"), ({"name": "  "}, "name")])
def test_validation_messages_are_plain_text(client, bad, msg):
    r = signup(client, **bad)
    assert r.status_code == 422 and isinstance(r.json()["detail"], str) and msg in r.json()["detail"]


def test_missing_field(client):
    r = client.post("/auth/signup", json={"email": "a@b.co"})
    assert r.status_code == 422 and "required" in r.json()["detail"]


def test_officer_needs_the_code(client):
    assert signup(client, role="officer").status_code == 403
    assert signup(client, role="officer", officer_code="wrong").status_code == 403
    r = signup(client, role="officer", officer_code="CITY-2026")
    assert r.status_code == 201 and r.json()["user"]["role"] == "officer"


def test_officer_signup_off_when_no_code_configured(client, monkeypatch):
    monkeypatch.delenv("OFFICER_CODE")
    assert signup(client, role="officer", officer_code="anything").status_code == 403


def test_wrong_password_and_unknown_email_look_the_same(client):
    signup(client)
    a = client.post("/auth/login", json={"email": "asha@example.com", "password": "nope-nope"})
    b = client.post("/auth/login", json={"email": "ghost@example.com", "password": "nope-nope"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


def test_login_rate_limit(client):
    signup(client)
    codes = [client.post("/auth/login", json={"email": "asha@example.com", "password": "bad-bad-bad"}).status_code for _ in range(9)]
    assert codes[:8] == [401] * 8 and codes[8] == 429
    # even the right password is blocked while locked out
    assert client.post("/auth/login", json={"email": "asha@example.com", "password": "correct-horse"}).status_code == 429


def test_bad_and_tampered_tokens(client):
    r = signup(client)
    t = r.json()["token"]
    assert client.get("/auth/me").status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer abc"}).status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer " + t[:-3] + "xyz"}).status_code == 401


def test_token_stops_working_if_user_is_deleted(client):
    t = signup(client).json()["token"]
    client.fake.conn.execute("DELETE FROM users")
    assert client.get("/auth/me", headers={"Authorization": "Bearer " + t}).status_code == 401


def test_not_configured_fails_closed(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "short")
    assert client.get("/health").json()["auth"] is False
    assert signup(client).status_code == 503


def test_database_down_gives_503_not_500(client, monkeypatch):
    def boom(request):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(db, "_client", httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    assert signup(client).status_code == 503


def test_cors_allows_authorization_header(client):
    r = client.options("/auth/me", headers={"Origin": "https://x.app", "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization,content-type"})
    assert r.status_code == 200 and "authorization" in r.headers["access-control-allow-headers"].lower()


def test_classify_still_works_without_login(client):
    import numpy as np, soundfile as sf, io
    buf = io.BytesIO(); sf.write(buf, (0.3 * np.sin(np.arange(22050 * 2) / 20)).astype("float32"), 22050, format="WAV")
    r = client.post("/classify", files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200 and "classification" in r.json()

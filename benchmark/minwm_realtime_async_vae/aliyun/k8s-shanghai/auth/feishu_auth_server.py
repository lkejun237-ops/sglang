from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

logger = logging.getLogger("zing.feishu_auth")

AUTHORIZE_URL = os.getenv(
    "FEISHU_AUTHORIZE_URL",
    "https://accounts.feishu.cn/open-apis/authen/v1/authorize",
)
TOKEN_URL = os.getenv(
    "FEISHU_TOKEN_URL",
    "https://open.feishu.cn/open-apis/authen/v2/oauth/token",
)
USER_INFO_URL = os.getenv(
    "FEISHU_USER_INFO_URL",
    "https://open.feishu.cn/open-apis/authen/v1/user_info",
)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


APP_ID = _required_env("FEISHU_APP_ID")
APP_SECRET = _required_env("FEISHU_APP_SECRET")
COOKIE_SECRET = _required_env("AUTH_COOKIE_SECRET").encode("utf-8")
REDIS_URL = _required_env("AUTH_REDIS_URL")
PUBLIC_BASE_URL = _required_env("AUTH_PUBLIC_BASE_URL").rstrip("/")
REDIRECT_PATH = os.getenv("AUTH_REDIRECT_PATH", "/auth/callback")
REDIRECT_URI = os.getenv("AUTH_REDIRECT_URI", PUBLIC_BASE_URL + REDIRECT_PATH)
COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "zing_webui_session")
STATE_COOKIE_NAME = os.getenv("AUTH_STATE_COOKIE_NAME", "zing_webui_oauth_state")
COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN") or None
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "true").lower() not in {"0", "false"}
SESSION_TTL_SECONDS = int(os.getenv("AUTH_SESSION_TTL_SECONDS", "86400"))
STATE_TTL_SECONDS = int(os.getenv("AUTH_STATE_TTL_SECONDS", "300"))
REDIS_KEY_PREFIX = os.getenv("AUTH_REDIS_KEY_PREFIX", "zing:shanghai:webui-auth")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("AUTH_REQUEST_TIMEOUT_SECONDS", "10"))
FEISHU_SCOPE = os.getenv("FEISHU_SCOPE", "").strip()
ALLOWED_TENANT_KEYS = {
    item.strip()
    for item in os.getenv("AUTH_ALLOWED_TENANT_KEYS", os.getenv("AUTH_ALLOWED_TENANT_KEY", "")).split(",")
    if item.strip()
}
ALLOWED_EMAIL_DOMAINS = {
    item.strip().lower().lstrip("@")
    for item in os.getenv("AUTH_ALLOWED_EMAIL_DOMAINS", "").split(",")
    if item.strip()
}
ALLOWED_USERS = {
    item.strip().lower()
    for item in os.getenv("AUTH_ALLOWED_USERS", "").split(",")
    if item.strip()
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64url(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _signature(value: str) -> str:
    return _b64url(hmac.new(COOKIE_SECRET, value.encode("utf-8"), hashlib.sha256).digest())


def _sign_json(payload: dict[str, Any]) -> str:
    raw = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return raw + "." + _signature(raw)


def _open_signed_json(value: str) -> dict[str, Any]:
    try:
        raw, sig = value.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError("malformed signed value") from exc
    expected = _signature(raw)
    if not hmac.compare_digest(sig, expected):
        raise ValueError("invalid signature")
    payload = json.loads(_unb64url(raw))
    if not isinstance(payload, dict):
        raise ValueError("signed value is not an object")
    exp = float(payload.get("exp") or 0)
    if exp and exp < time.time():
        raise ValueError("signed value expired")
    return payload


def _safe_return_to(value: str | None) -> str:
    if not value:
        return "/"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return "/"
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    if value.startswith("/auth/"):
        return "/"
    return value


def _split_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class RedisAddress:
    host: str
    port: int
    username: str | None
    password: str | None
    db: int
    use_ssl: bool


def _parse_redis_url(url: str) -> RedisAddress:
    parsed = urlsplit(url)
    if parsed.scheme not in {"redis", "rediss"}:
        raise ValueError("AUTH_REDIS_URL must start with redis:// or rediss://")
    db = 0
    path = parsed.path.lstrip("/")
    if path:
        db = int(path)
    return RedisAddress(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        username=parsed.username,
        password=parsed.password,
        db=db,
        use_ssl=parsed.scheme == "rediss",
    )


class RedisStore:
    def __init__(self, url: str, key_prefix: str):
        self.address = _parse_redis_url(url)
        self.key_prefix = key_prefix.rstrip(":")
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    def key(self, sid: str) -> str:
        return f"{self.key_prefix}:session:{sid}"

    async def close(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            writer.close()
            await writer.wait_closed()

    async def _connect(self) -> None:
        if self._reader is not None and self._writer is not None:
            return
        self._reader, self._writer = await asyncio.open_connection(
            self.address.host,
            self.address.port,
            ssl=self.address.use_ssl,
        )
        if self.address.password:
            if self.address.username:
                await self._command_unlocked("AUTH", self.address.username, self.address.password)
            else:
                await self._command_unlocked("AUTH", self.address.password)
        if self.address.db:
            await self._command_unlocked("SELECT", str(self.address.db))

    async def command(self, *parts: str | bytes | int) -> Any:
        async with self._lock:
            for attempt in range(2):
                try:
                    await self._connect()
                    return await self._command_unlocked(*parts)
                except Exception:
                    await self.close()
                    if attempt:
                        raise

    async def _command_unlocked(self, *parts: str | bytes | int) -> Any:
        if self._reader is None or self._writer is None:
            raise RuntimeError("Redis connection is not open")
        encoded_parts = []
        for part in parts:
            if isinstance(part, bytes):
                encoded = part
            else:
                encoded = str(part).encode("utf-8")
            encoded_parts.append(encoded)
        wire = [f"*{len(encoded_parts)}\r\n".encode("ascii")]
        for part in encoded_parts:
            wire.append(f"${len(part)}\r\n".encode("ascii"))
            wire.append(part)
            wire.append(b"\r\n")
        self._writer.write(b"".join(wire))
        await self._writer.drain()
        return await self._read_response()

    async def _read_line(self) -> bytes:
        if self._reader is None:
            raise RuntimeError("Redis connection is not open")
        line = await self._reader.readline()
        if not line:
            raise RuntimeError("Redis connection closed")
        if not line.endswith(b"\r\n"):
            raise RuntimeError("malformed Redis response")
        return line[:-2]

    async def _read_response(self) -> Any:
        if self._reader is None:
            raise RuntimeError("Redis connection is not open")
        prefix = await self._reader.readexactly(1)
        if prefix == b"+":
            return (await self._read_line()).decode("utf-8", errors="replace")
        if prefix == b"-":
            message = (await self._read_line()).decode("utf-8", errors="replace")
            raise RuntimeError(f"Redis error: {message}")
        if prefix == b":":
            return int(await self._read_line())
        if prefix == b"$":
            size = int(await self._read_line())
            if size < 0:
                return None
            data = await self._reader.readexactly(size)
            await self._reader.readexactly(2)
            return data
        if prefix == b"*":
            count = int(await self._read_line())
            if count < 0:
                return None
            return [await self._read_response() for _ in range(count)]
        raise RuntimeError(f"unsupported Redis response prefix: {prefix!r}")

    async def set_session(self, sid: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        await self.command(
            "SETEX",
            self.key(sid),
            ttl_seconds,
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        )

    async def get_session(self, sid: str) -> dict[str, Any] | None:
        value = await self.command("GET", self.key(sid))
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        payload = json.loads(value)
        return payload if isinstance(payload, dict) else None

    async def delete_session(self, sid: str) -> None:
        await self.command("DEL", self.key(sid))


store = RedisStore(REDIS_URL, REDIS_KEY_PREFIX)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await store.close()


app = FastAPI(title="Zing Feishu Auth", lifespan=lifespan)


def _cookie_session_id(request: Request) -> str | None:
    value = request.cookies.get(COOKIE_NAME)
    if not value:
        return None
    try:
        payload = _open_signed_json(value)
    except ValueError:
        return None
    sid = payload.get("sid")
    return sid if isinstance(sid, str) and sid else None


def _set_cookie(response: Response, name: str, value: str, max_age: int) -> None:
    response.set_cookie(
        name,
        value,
        max_age=max_age,
        path="/",
        domain=COOKIE_DOMAIN,
        secure=COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )


def _clear_cookie(response: Response, name: str) -> None:
    response.delete_cookie(
        name,
        path="/",
        domain=COOKIE_DOMAIN,
        secure=COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )


async def _current_session(request: Request) -> dict[str, Any] | None:
    sid = _cookie_session_id(request)
    if not sid:
        return None
    try:
        session = await store.get_session(sid)
    except Exception:
        logger.exception("failed to read auth session")
        return None
    if not session:
        return None
    if float(session.get("expires_at") or 0) < time.time():
        await store.delete_session(sid)
        return None
    return session


def _login_url(request: Request) -> str:
    original = (
        request.headers.get("x-original-uri")
        or request.headers.get("x-forwarded-uri")
        or str(request.url.path)
    )
    rd = quote(_safe_return_to(original), safe="")
    return f"/auth/login?rd={rd}"


def _user_headers(session: dict[str, Any]) -> dict[str, str]:
    user = session.get("user") if isinstance(session.get("user"), dict) else {}
    headers = {}
    for source, target in {
        "open_id": "X-Auth-Feishu-Open-Id",
        "union_id": "X-Auth-Feishu-Union-Id",
        "user_id": "X-Auth-Feishu-User-Id",
        "email": "X-Auth-Feishu-Email",
        "name": "X-Auth-Feishu-Name",
    }.items():
        value = user.get(source)
        if value:
            headers[target] = str(value)
    return headers


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/check")
async def check(request: Request) -> Response:
    session = await _current_session(request)
    if not session:
        response = RedirectResponse(_login_url(request), status_code=302)
        response.headers["Cache-Control"] = "no-store"
        return response
    return Response(status_code=204, headers=_user_headers(session))


@app.get("/auth/login")
async def login(request: Request) -> Response:
    rd = _safe_return_to(request.query_params.get("rd"))
    state = _sign_json(
        {
            "rd": rd,
            "nonce": secrets.token_urlsafe(18),
            "exp": time.time() + STATE_TTL_SECONDS,
        }
    )
    params = {
        "client_id": APP_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "state": state,
    }
    if FEISHU_SCOPE:
        params["scope"] = FEISHU_SCOPE
    response = RedirectResponse(AUTHORIZE_URL + "?" + urlencode(params), status_code=302)
    _set_cookie(response, STATE_COOKIE_NAME, state, STATE_TTL_SECONDS)
    response.headers["Cache-Control"] = "no-store"
    return response


def _extract_feishu_payload(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("code") not in (None, 0):
        raise HTTPException(status_code=502, detail="feishu api rejected request")
    payload = data.get("data", data)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="malformed feishu api response")
    return payload


async def _exchange_code(code: str) -> str:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(
            TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": APP_ID,
                "client_secret": APP_SECRET,
                "code": code,
                "redirect_uri": REDIRECT_URI,
            },
        )
        response.raise_for_status()
        payload = _extract_feishu_payload(response.json())
    token = payload.get("access_token") or payload.get("user_access_token")
    if not token:
        raise HTTPException(status_code=502, detail="feishu token response missing access token")
    return str(token)


async def _fetch_user(access_token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(
            USER_INFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        payload = _extract_feishu_payload(response.json())
    return payload


def _normalized_user(info: dict[str, Any]) -> dict[str, Any]:
    email = info.get("email") or info.get("enterprise_email")
    return {
        "open_id": info.get("open_id"),
        "union_id": info.get("union_id"),
        "user_id": info.get("user_id"),
        "tenant_key": info.get("tenant_key"),
        "email": email,
        "name": info.get("name") or info.get("en_name") or email or info.get("open_id"),
    }


def _enforce_allowlist(user: dict[str, Any]) -> None:
    tenant_key = str(user.get("tenant_key") or "")
    if ALLOWED_TENANT_KEYS and tenant_key not in ALLOWED_TENANT_KEYS:
        raise HTTPException(status_code=403, detail="tenant is not allowed")
    identifiers = {
        str(user.get("open_id") or "").lower(),
        str(user.get("union_id") or "").lower(),
        str(user.get("user_id") or "").lower(),
        str(user.get("email") or "").lower(),
    }
    if ALLOWED_USERS and not (identifiers & ALLOWED_USERS):
        raise HTTPException(status_code=403, detail="user is not allowed")
    email = str(user.get("email") or "").lower()
    if ALLOWED_EMAIL_DOMAINS:
        domain = email.rsplit("@", 1)[-1] if "@" in email else ""
        if domain not in ALLOWED_EMAIL_DOMAINS:
            raise HTTPException(status_code=403, detail="email domain is not allowed")


@app.get(REDIRECT_PATH)
async def callback(request: Request) -> Response:
    error = request.query_params.get("error")
    if error:
        raise HTTPException(status_code=403, detail="feishu login was not authorized")
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    state_cookie = request.cookies.get(STATE_COOKIE_NAME)
    if not code or not state or not state_cookie or not hmac.compare_digest(state, state_cookie):
        raise HTTPException(status_code=400, detail="invalid oauth state")
    try:
        state_payload = _open_signed_json(state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid oauth state") from exc

    access_token = await _exchange_code(code)
    user = _normalized_user(await _fetch_user(access_token))
    _enforce_allowlist(user)

    sid = secrets.token_urlsafe(32)
    now = time.time()
    await store.set_session(
        sid,
        {
            "sid": sid,
            "user": user,
            "created_at": now,
            "expires_at": now + SESSION_TTL_SECONDS,
        },
        SESSION_TTL_SECONDS,
    )
    session_cookie = _sign_json({"sid": sid, "exp": now + SESSION_TTL_SECONDS})
    response = RedirectResponse(_safe_return_to(str(state_payload.get("rd") or "/")), status_code=302)
    _set_cookie(response, COOKIE_NAME, session_cookie, SESSION_TTL_SECONDS)
    _clear_cookie(response, STATE_COOKIE_NAME)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/auth/logout")
async def logout(request: Request) -> Response:
    sid = _cookie_session_id(request)
    if sid:
        try:
            await store.delete_session(sid)
        except Exception:
            logger.exception("failed to delete auth session")
    response = RedirectResponse("/auth/login", status_code=302)
    _clear_cookie(response, COOKIE_NAME)
    _clear_cookie(response, STATE_COOKIE_NAME)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/auth/whoami")
async def whoami(request: Request) -> Response:
    session = await _current_session(request)
    if not session:
        return JSONResponse({"authenticated": False}, status_code=401)
    return JSONResponse({"authenticated": True, "user": session.get("user")})


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    uvicorn.run(
        app,
        host=os.getenv("AUTH_HOST", "0.0.0.0"),
        port=int(os.getenv("AUTH_PORT", "18082")),
        access_log=os.getenv("AUTH_ACCESS_LOG", "false").lower() in {"1", "true"},
    )


if __name__ == "__main__":
    main()

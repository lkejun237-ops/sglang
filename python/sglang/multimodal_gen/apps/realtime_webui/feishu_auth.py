# SPDX-License-Identifier: Apache-2.0

"""Optional Feishu OAuth gate for the realtime WebUI."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

logger = logging.getLogger("sglang.realtime_webui.feishu_auth")

PUBLIC_BACKEND_PATH_RE = re.compile(
    r"^/backends/[^/]+/v1/(models|slots|realtime_video/authorized_generate)$"
)
PUBLIC_PATHS = {
    "/healthz",
    "/v1/models",
    "/v1/slots",
    "/v1/realtime_video/authorized_generate",
}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required when WEBUI_AUTH_ENABLED=true")
    return value


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64url(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


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


def _header_value(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return quote(text, safe="")
    return text


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
                await self._command_unlocked(
                    "AUTH", self.address.username, self.address.password
                )
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

    async def set_session(
        self, sid: str, payload: dict[str, Any], ttl_seconds: int
    ) -> None:
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


class FeishuAuth:
    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.http_session: ClientSession | None = None
        if not enabled:
            self.store = None
            return

        self.authorize_url = os.getenv(
            "FEISHU_AUTHORIZE_URL",
            "https://accounts.feishu.cn/open-apis/authen/v1/authorize",
        )
        self.token_url = os.getenv(
            "FEISHU_TOKEN_URL",
            "https://open.feishu.cn/open-apis/authen/v2/oauth/token",
        )
        self.user_info_url = os.getenv(
            "FEISHU_USER_INFO_URL",
            "https://open.feishu.cn/open-apis/authen/v1/user_info",
        )
        self.app_id = _required_env("FEISHU_APP_ID")
        self.app_secret = _required_env("FEISHU_APP_SECRET")
        self.cookie_secret = _required_env("AUTH_COOKIE_SECRET").encode("utf-8")
        self.redis_url = os.getenv("AUTH_REDIS_URL") or _required_env("REDIS_URL")
        self.public_base_url = _required_env("AUTH_PUBLIC_BASE_URL").rstrip("/")
        self.redirect_path = os.getenv("AUTH_REDIRECT_PATH", "/auth/callback")
        self.redirect_uri = os.getenv(
            "AUTH_REDIRECT_URI", self.public_base_url + self.redirect_path
        )
        self.cookie_name = os.getenv("AUTH_COOKIE_NAME", "zing_webui_session")
        self.state_cookie_name = os.getenv(
            "AUTH_STATE_COOKIE_NAME", "zing_webui_oauth_state"
        )
        self.cookie_domain = os.getenv("AUTH_COOKIE_DOMAIN") or None
        self.cookie_secure = os.getenv("AUTH_COOKIE_SECURE", "true").lower() not in {
            "0",
            "false",
        }
        self.session_ttl_seconds = int(os.getenv("AUTH_SESSION_TTL_SECONDS", "86400"))
        self.state_ttl_seconds = int(os.getenv("AUTH_STATE_TTL_SECONDS", "300"))
        self.redis_key_prefix = os.getenv(
            "AUTH_REDIS_KEY_PREFIX", "zing:webui-auth"
        )
        self.request_timeout_seconds = float(
            os.getenv("AUTH_REQUEST_TIMEOUT_SECONDS", "10")
        )
        self.feishu_scope = os.getenv("FEISHU_SCOPE", "").strip()
        self.allowed_tenant_keys = {
            item.strip()
            for item in os.getenv(
                "AUTH_ALLOWED_TENANT_KEYS",
                os.getenv("AUTH_ALLOWED_TENANT_KEY", ""),
            ).split(",")
            if item.strip()
        }
        self.allowed_email_domains = {
            item.strip().lower().lstrip("@")
            for item in os.getenv("AUTH_ALLOWED_EMAIL_DOMAINS", "").split(",")
            if item.strip()
        }
        self.allowed_users = {
            item.strip().lower()
            for item in os.getenv("AUTH_ALLOWED_USERS", "").split(",")
            if item.strip()
        }
        self.store = RedisStore(self.redis_url, self.redis_key_prefix)

    @classmethod
    def from_env(cls) -> "FeishuAuth":
        return cls(enabled=_truthy(os.getenv("WEBUI_AUTH_ENABLED")))

    async def cleanup_context(self, _app):
        if self.enabled:
            self.http_session = ClientSession(
                timeout=ClientTimeout(total=None),
                connector=TCPConnector(ttl_dns_cache=300),
            )
        try:
            yield
        finally:
            if self.http_session is not None:
                await self.http_session.close()
            if self.store is not None:
                await self.store.close()

    def register(self, app: web.Application) -> None:
        if not self.enabled:
            return
        app.middlewares.append(self.middleware)
        app.router.add_get("/auth/check", self.check)
        app.router.add_get("/auth/login", self.login)
        app.router.add_get(self.redirect_path, self.callback)
        app.router.add_get("/auth/logout", self.logout)
        app.router.add_get("/auth/whoami", self.whoami)

    @web.middleware
    async def middleware(self, request: web.Request, handler):
        if self._is_public_path(request.path):
            return await handler(request)
        session = await self.current_session(request)
        if session:
            request["feishu_user"] = session.get("user") or {}
            return await handler(request)
        if self._should_return_unauthorized(request):
            return web.json_response(
                {"error": "authentication required", "login_url": self.login_url(request)},
                status=401,
                headers={"Cache-Control": "no-store"},
            )
        response = web.HTTPFound(self.login_url(request))
        response.headers["Cache-Control"] = "no-store"
        raise response

    def _is_public_path(self, path: str) -> bool:
        return (
            path in PUBLIC_PATHS
            or path.startswith("/auth/")
            or PUBLIC_BACKEND_PATH_RE.fullmatch(path) is not None
        )

    @staticmethod
    def _should_return_unauthorized(request: web.Request) -> bool:
        if request.headers.get("upgrade", "").lower() == "websocket":
            return True
        path = request.path
        accept = request.headers.get("accept", "")
        return (
            path.startswith("/api/")
            or path.startswith("/studio-api/")
            or path.startswith("/v1/")
            or path.startswith("/backends/")
            or "application/json" in accept
        )

    def _signature(self, value: str) -> str:
        return _b64url(
            hmac.new(self.cookie_secret, value.encode("utf-8"), hashlib.sha256).digest()
        )

    def sign_json(self, payload: dict[str, Any]) -> str:
        raw = _b64url(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        return raw + "." + self._signature(raw)

    def open_signed_json(self, value: str) -> dict[str, Any]:
        try:
            raw, sig = value.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError("malformed signed value") from exc
        expected = self._signature(raw)
        if not hmac.compare_digest(sig, expected):
            raise ValueError("invalid signature")
        payload = json.loads(_unb64url(raw))
        if not isinstance(payload, dict):
            raise ValueError("signed value is not an object")
        exp = float(payload.get("exp") or 0)
        if exp and exp < time.time():
            raise ValueError("signed value expired")
        return payload

    def cookie_session_id(self, request: web.Request) -> str | None:
        value = request.cookies.get(self.cookie_name)
        if not value:
            return None
        try:
            payload = self.open_signed_json(value)
        except ValueError:
            return None
        sid = payload.get("sid")
        return sid if isinstance(sid, str) and sid else None

    def set_cookie(
        self, response: web.StreamResponse, name: str, value: str, max_age: int
    ) -> None:
        response.set_cookie(
            name,
            value,
            max_age=max_age,
            path="/",
            domain=self.cookie_domain,
            secure=self.cookie_secure,
            httponly=True,
            samesite="Lax",
        )

    def clear_cookie(self, response: web.StreamResponse, name: str) -> None:
        response.del_cookie(
            name,
            path="/",
            domain=self.cookie_domain,
        )

    async def current_session(self, request: web.Request) -> dict[str, Any] | None:
        sid = self.cookie_session_id(request)
        if not sid or self.store is None:
            return None
        try:
            session = await self.store.get_session(sid)
        except Exception:
            logger.exception("failed to read auth session")
            return None
        if not session:
            return None
        if float(session.get("expires_at") or 0) < time.time():
            await self.store.delete_session(sid)
            return None
        return session

    def login_url(self, request: web.Request) -> str:
        original = (
            request.headers.get("x-original-uri")
            or request.headers.get("x-forwarded-uri")
            or str(request.rel_url)
        )
        rd = quote(_safe_return_to(original), safe="")
        return f"/auth/login?rd={rd}"

    def user_headers(self, session: dict[str, Any]) -> dict[str, str]:
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
                headers[target] = _header_value(value)
        return headers

    async def check(self, request: web.Request) -> web.Response:
        session = await self.current_session(request)
        if not session:
            response = web.HTTPFound(self.login_url(request))
            response.headers["Cache-Control"] = "no-store"
            raise response
        return web.Response(status=204, headers=self.user_headers(session))

    async def login(self, request: web.Request) -> web.Response:
        rd = _safe_return_to(request.query.get("rd"))
        state = self.sign_json(
            {
                "rd": rd,
                "nonce": secrets.token_urlsafe(18),
                "exp": time.time() + self.state_ttl_seconds,
            }
        )
        params = {
            "client_id": self.app_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state,
        }
        if self.feishu_scope:
            params["scope"] = self.feishu_scope
        response = web.HTTPFound(self.authorize_url + "?" + urlencode(params))
        self.set_cookie(response, self.state_cookie_name, state, self.state_ttl_seconds)
        response.headers["Cache-Control"] = "no-store"
        raise response

    async def callback(self, request: web.Request) -> web.Response:
        if request.query.get("error"):
            raise web.HTTPForbidden(text="feishu login was not authorized")
        code = request.query.get("code")
        state = request.query.get("state")
        state_cookie = request.cookies.get(self.state_cookie_name)
        if (
            not code
            or not state
            or not state_cookie
            or not hmac.compare_digest(state, state_cookie)
        ):
            raise web.HTTPBadRequest(text="invalid oauth state")
        try:
            state_payload = self.open_signed_json(state)
        except ValueError as exc:
            raise web.HTTPBadRequest(text="invalid oauth state") from exc

        access_token = await self.exchange_code(code)
        user = self.normalized_user(await self.fetch_user(access_token))
        self.enforce_allowlist(user)

        sid = secrets.token_urlsafe(32)
        now = time.time()
        if self.store is None:
            raise web.HTTPInternalServerError(text="auth store is not configured")
        await self.store.set_session(
            sid,
            {
                "sid": sid,
                "user": user,
                "created_at": now,
                "expires_at": now + self.session_ttl_seconds,
            },
            self.session_ttl_seconds,
        )
        session_cookie = self.sign_json({"sid": sid, "exp": now + self.session_ttl_seconds})
        response = web.HTTPFound(_safe_return_to(str(state_payload.get("rd") or "/")))
        self.set_cookie(response, self.cookie_name, session_cookie, self.session_ttl_seconds)
        self.clear_cookie(response, self.state_cookie_name)
        response.headers["Cache-Control"] = "no-store"
        raise response

    async def logout(self, request: web.Request) -> web.Response:
        sid = self.cookie_session_id(request)
        if sid and self.store is not None:
            try:
                await self.store.delete_session(sid)
            except Exception:
                logger.exception("failed to delete auth session")
        response = web.HTTPFound("/auth/login")
        self.clear_cookie(response, self.cookie_name)
        response.headers["Cache-Control"] = "no-store"
        raise response

    async def whoami(self, request: web.Request) -> web.Response:
        session = await self.current_session(request)
        if not session:
            raise web.HTTPUnauthorized(text="authentication required")
        return web.json_response(
            {"user": session.get("user") or {}},
            headers={"Cache-Control": "no-store"},
        )

    async def exchange_code(self, code: str) -> str:
        session = self._http()
        async with session.post(
            self.token_url,
            json={
                "grant_type": "authorization_code",
                "client_id": self.app_id,
                "client_secret": self.app_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
            },
            timeout=ClientTimeout(total=self.request_timeout_seconds),
        ) as response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                raise web.HTTPBadGateway(text="feishu token request failed")
            data = self.extract_feishu_payload(payload)
        token = data.get("access_token") or data.get("user_access_token")
        if not token:
            raise web.HTTPBadGateway(text="feishu token response missing access token")
        return str(token)

    async def fetch_user(self, access_token: str) -> dict[str, Any]:
        session = self._http()
        async with session.get(
            self.user_info_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=ClientTimeout(total=self.request_timeout_seconds),
        ) as response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                raise web.HTTPBadGateway(text="feishu user info request failed")
            return self.extract_feishu_payload(payload)

    def _http(self) -> ClientSession:
        if self.http_session is None:
            raise web.HTTPInternalServerError(text="auth http client is not ready")
        return self.http_session

    @staticmethod
    def extract_feishu_payload(data: dict[str, Any]) -> dict[str, Any]:
        if data.get("code") not in (None, 0):
            raise web.HTTPBadGateway(text="feishu api rejected request")
        payload = data.get("data", data)
        if not isinstance(payload, dict):
            raise web.HTTPBadGateway(text="malformed feishu api response")
        return payload

    @staticmethod
    def normalized_user(info: dict[str, Any]) -> dict[str, Any]:
        email = info.get("email") or info.get("enterprise_email")
        return {
            "open_id": info.get("open_id"),
            "union_id": info.get("union_id"),
            "user_id": info.get("user_id"),
            "tenant_key": info.get("tenant_key"),
            "email": email,
            "name": info.get("name") or info.get("en_name") or email or info.get("open_id"),
        }

    def enforce_allowlist(self, user: dict[str, Any]) -> None:
        tenant_key = str(user.get("tenant_key") or "")
        if self.allowed_tenant_keys and tenant_key not in self.allowed_tenant_keys:
            raise web.HTTPForbidden(text="tenant is not allowed")
        identifiers = {
            str(user.get("open_id") or "").lower(),
            str(user.get("union_id") or "").lower(),
            str(user.get("user_id") or "").lower(),
            str(user.get("email") or "").lower(),
        }
        if self.allowed_users and not (identifiers & self.allowed_users):
            raise web.HTTPForbidden(text="user is not allowed")
        email = str(user.get("email") or "").lower()
        if self.allowed_email_domains:
            domain = email.rsplit("@", 1)[-1] if "@" in email else ""
            if domain not in self.allowed_email_domains:
                raise web.HTTPForbidden(text="email domain is not allowed")

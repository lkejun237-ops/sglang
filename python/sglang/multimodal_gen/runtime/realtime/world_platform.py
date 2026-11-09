# SPDX-License-Identifier: Apache-2.0
"""World platform ingress: session-token verification and lifecycle callbacks.

The authorized_generate route used with world-service, the business backend,
works as follows:
  - Credential: a compact Ed25519 JWS. The gateway only holds the public key,
    while world-service remains the sole token authority. Identity sources such
    as Clerk or biz-core can sit in front without changing this module.
  - Session payload: world-service seals the init message with AES-GCM and asks
    the browser to forward it. The gateway unseals it here. The browser never
    sees prompts, timelines, or skill instructions. GCM provides integrity, so a
    one-byte tamper fails without a separate hash binding.
  - Callbacks: started, first-frame, ended, aborted, and recorded callbacks use
    HMAC signing compatible with the biz-core tracking ingest path.

Legacy showcase routes do not reference this module. If world platform settings
are absent, behavior is unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac as hmac_mod
import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from uuid import UUID

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)
_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
FINISH_TAIL_SECONDS = 5


class TokenError(Exception):
    """Session-token verification failed; externally surfaced as close code 1008."""


@dataclass(frozen=True)
class Principal:
    """Session principal carried by the token."""

    user_id: str  # per-run pseudonym, not linkable back to the account
    run_id: str
    play_seconds: int
    max_lifetime_s: int
    allow_free_prompt: bool
    record_video: bool
    jti: str = ""  # one-time nonce to prevent two sessions from one token
    exp: int = 0  # expiry timestamp in seconds, used to age out replay cache
    # Account-level pseudonym: stable across runs for the same player, unlike
    # user_id which changes with every run. The coordinator's one-lease-per-user
    # admission must key on this. Keyed on the per-run pseudonym instead, a
    # player whose business-side ledger is cleared once (session declared dead,
    # or the player left) gets a new run and therefore a new identity, so the
    # same person can take a second seat while the old session still holds a
    # GPU. Older tokens do not carry this claim and fall back to user_id.
    account_id: str = ""
    # Missing role is the backwards-compatible primary token shape. Comparison
    # is explicit so a token cannot acquire the LingBot route by omission.
    role: str = "primary"
    # Verified deployment profile selected from the signed JWS ``kid``. It is
    # used only inside the gateway to isolate coordinator identities, callback
    # credentials, payload keys, and recording storage between environments.
    profile_id: str = "legacy"

    @property
    def lease_key(self) -> str:
        """Identity key for the coordinator lease."""
        return self.account_id or self.user_id


def session_payload_aad(run_id: str, role: str) -> str:
    """Return the role-scoped AES-GCM associated data for a session payload.

    Primary keeps the original run-id-only AAD for backwards compatibility.
    Comparison adds a fixed domain separator so the two opaque payloads issued
    for one run cannot be swapped between the Zing and LingBot routes.
    """
    if role == "primary":
        return run_id
    if role == "comparison":
        return f"{run_id}:comparison"
    raise ValueError("session role must be primary or comparison")


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def load_public_key(pub_b64: str) -> Ed25519PublicKey:
    """Build a verifier from a base64 public key; fail fast during startup."""
    raw = base64.b64decode(pub_b64)
    if len(raw) != 32:
        raise ValueError("Ed25519 public key must be 32 base64-encoded bytes")
    return Ed25519PublicKey.from_public_bytes(raw)


def token_key_id(token: str) -> str | None:
    """Read and validate the signed-key selector from a compact JWS header.

    This only selects a candidate public key. The caller must still verify the
    signature before trusting the returned value.
    """
    try:
        header_b64, _, _ = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
    except (
        ValueError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
        binascii.Error,
    ) as exc:
        raise TokenError("invalid token header") from exc
    if not isinstance(header, dict) or header.get("alg") != "EdDSA":
        raise TokenError("only EdDSA is supported")
    key_id = header.get("kid")
    if key_id is None:
        return None
    if not isinstance(key_id, str) or not _PROFILE_ID_RE.fullmatch(key_id):
        raise TokenError("invalid token key ID")
    return key_id


def verify_session_token(
    token: str,
    public_key: Ed25519PublicKey,
    *,
    expected_issuer: str = "world-service",
    expected_key_id: str | None = None,
    profile_id: str = "legacy",
) -> Principal:
    """Verify the session token and return its principal."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if not isinstance(header, dict):
            raise TokenError("invalid token header")
        if header.get("alg") != "EdDSA":
            raise TokenError("only EdDSA is supported")
        if expected_key_id is not None and header.get("kid") != expected_key_id:
            raise TokenError("kid mismatch")
        public_key.verify(
            _b64url_decode(sig_b64),
            f"{header_b64}.{payload_b64}".encode(),
        )
        claims = json.loads(_b64url_decode(payload_b64))
        if not isinstance(claims, dict):
            raise TokenError("invalid token claims")
    except TokenError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        InvalidSignature,
        binascii.Error,
    ) as exc:
        raise TokenError(f"invalid token: {exc}") from exc

    if claims.get("aud") != "world-session":
        raise TokenError("aud mismatch")
    if expected_issuer and claims.get("iss") != expected_issuer:
        raise TokenError("iss mismatch")
    exp = int(claims.get("exp") or 0)
    if exp and time.time() > exp + 30:  # 30s clock skew tolerance
        raise TokenError("token expired")
    session = claims.get("session")
    if not isinstance(session, dict):
        raise TokenError("session claim is required")
    max_lifetime = session.get("max_lifetime_s")
    play_seconds = session.get("play_seconds")
    allow_free_prompt = session.get("allow_free_prompt")
    record_video = session.get("record_video")
    role = session.get("role", "primary")
    # bool is a subclass of int in Python, so exact type checks are required.
    # Coercion would turn strings such as "false" into an enabled policy.
    if type(max_lifetime) is not int or max_lifetime <= 0:
        raise TokenError("missing session lifetime")
    if type(play_seconds) is not int or play_seconds <= 0:
        raise TokenError("play_seconds must be a positive integer")
    if max_lifetime - play_seconds - FINISH_TAIL_SECONDS <= 0:
        raise TokenError("session startup budget must be positive")
    if type(allow_free_prompt) is not bool:
        raise TokenError("allow_free_prompt must be boolean")
    if type(record_video) is not bool:
        raise TokenError("record_video must be boolean")
    if type(role) is not str or role not in {"primary", "comparison"}:
        raise TokenError("session role must be primary or comparison")
    if role == "comparison" and record_video:
        raise TokenError("comparison sessions must disable recording")
    raw_jti = claims.get("jti")
    if not isinstance(raw_jti, str):
        raise TokenError("jti must be a UUID")
    try:
        UUID(raw_jti)
    except (ValueError, AttributeError) as exc:
        raise TokenError("jti must be a UUID") from exc
    return Principal(
        user_id=str(claims.get("sub") or ""),
        run_id=str(claims.get("sid") or ""),
        play_seconds=play_seconds,
        max_lifetime_s=max_lifetime,
        allow_free_prompt=allow_free_prompt,
        record_video=record_video,
        role=role,
        # Callbacks must return the token jti byte-for-byte. UUID parsing only
        # validates it and must not silently normalize the attempt_id here.
        jti=raw_jti,
        exp=int(claims.get("exp") or 0),
        account_id=str(claims.get("uid") or ""),
        profile_id=profile_id,
    )


class SessionPayloadSealer:
    """Unseal session payloads produced by world-service pkg/seal.

    The AES-GCM key is derived from the service-to-service shared secret with
    HKDF-SHA256. The info label separates this key from HMAC callback signing.
    The associated data is supplied by the authorized route. Primary uses the
    run id unchanged; comparison uses ``run_id + ':comparison'``. This binds a
    payload to both its run and role, so cross-run and cross-route copies fail.
    """

    _HKDF_INFO = b"world-service/session-payload/v1"

    def __init__(self, shared_secret: str) -> None:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        self._key = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None, info=self._HKDF_INFO
        ).derive(shared_secret.encode())

    def open(self, sealed: bytes, associated_data: str) -> dict:
        """Unseal and return the init message; failures raise TokenError."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        # Lazy import keeps this module light and avoids loading the full sglang
        # dependency chain at import time.
        from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
            decode_message,
        )

        nonce_size = 12  # standard AES-GCM nonce length, matching the Go side
        if len(sealed) <= nonce_size:
            raise TokenError("invalid session payload format")
        try:
            plaintext = AESGCM(self._key).decrypt(
                sealed[:nonce_size], sealed[nonce_size:], associated_data.encode()
            )
        except Exception as exc:  # Normalize InvalidTag and related failures.
            raise TokenError("session payload verification failed") from exc
        message = decode_message(plaintext)
        if not isinstance(message, dict) or message.get("type") != "init":
            raise TokenError("session payload is not an init message")
        return message


class TokenReplayGuard:
    """One-use token guarantee using jti.

    The in-process cache is naturally bounded because the gateway is a single
    instance and tokens expire quickly. consume returns False when the token has
    already been used to create a session.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], float] = {}
        # ASGI normally has one event loop, but tests and embedded servers may
        # still enter from different threads. A check-then-write race would let
        # two connections for one attempt enter the Coordinator as owner.
        self._lock = threading.Lock()

    def consume(self, principal: Principal, now: float | None = None) -> bool:
        if not principal.jti:
            return False
        now = time.time() if now is None else now
        replay_key = (principal.profile_id, principal.jti)
        with self._lock:
            # Lazy cleanup: records past exp plus skew can no longer verify.
            expired = [j for j, until in self._seen.items() if until <= now]
            for j in expired:
                del self._seen[j]
            if replay_key in self._seen:
                return False
            # Keep until exp+60s and for at least five minutes, leaving room beyond
            # the 30s verification skew so retention cannot be shorter than validity.
            self._seen[replay_key] = max(float(principal.exp) + 60.0, now + 300.0)
            return True


@dataclass(frozen=True)
class WorldPlatformConfig:
    """Configuration for the authorized_generate route."""

    public_key: Ed25519PublicKey
    callback_url: str  # world-service root, e.g. http://world-service.world-model
    callback_app_id: str
    callback_key_id: str
    callback_secret: str
    recording_bucket: str
    recording_region: str
    recording_endpoint_url: str = ""
    recording_uri_scheme: str = "s3"
    recording_access_key_id: str = ""
    recording_secret_access_key: str = ""
    recording_session_token: str = ""
    profile_id: str = "legacy"
    token_issuer: str = "world-service"


class WorldPlatformProfiles:
    """Immutable key-ID registry for per-session platform isolation."""

    def __init__(
        self,
        profiles: list[WorldPlatformConfig],
        *,
        legacy_profile_id: str = "",
    ) -> None:
        if not profiles:
            raise ValueError("at least one world platform profile is required")
        by_id: dict[str, WorldPlatformConfig] = {}
        for profile in profiles:
            if not _PROFILE_ID_RE.fullmatch(profile.profile_id):
                raise ValueError("world platform profile ID is invalid")
            if profile.profile_id in by_id:
                raise ValueError("world platform profile IDs must be unique")
            by_id[profile.profile_id] = profile
        if legacy_profile_id and legacy_profile_id not in by_id:
            raise ValueError("legacy world platform profile does not exist")
        self._profiles = by_id
        self.legacy_profile_id = legacy_profile_id

    @classmethod
    def from_legacy(cls, profile: WorldPlatformConfig) -> WorldPlatformProfiles:
        """Wrap the original single-profile configuration for rolling upgrades."""
        return cls([profile], legacy_profile_id=profile.profile_id)

    @property
    def profiles(self) -> tuple[WorldPlatformConfig, ...]:
        return tuple(self._profiles.values())

    def verify(self, token: str) -> tuple[Principal, WorldPlatformConfig]:
        """Select a profile from signed ``kid`` and verify the whole token."""
        key_id = token_key_id(token)
        profile_id = key_id or self.legacy_profile_id
        if not profile_id:
            raise TokenError("token key ID is required")
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise TokenError("unknown token key ID")
        principal = verify_session_token(
            token,
            profile.public_key,
            expected_issuer=profile.token_issuer,
            expected_key_id=key_id,
            profile_id=profile.profile_id,
        )
        return principal, profile


class WorldCallbacks:
    """Lifecycle callback client with fire-and-forget bounded retries.

    Callback failures never affect the session; world-service has a deadline
    fallback scan.
    """

    def __init__(self, cfg: WorldPlatformConfig) -> None:
        self._cfg = cfg
        self._client = httpx.AsyncClient(timeout=5.0)
        # Rewrites get their own client with a bounded connection pool. Sharing
        # the pool with lifecycle callbacks lets one rewrite flood fill the
        # default 100 connections and starve started/ended/aborted for *every
        # session in this process*, spreading the blast radius from the player
        # who is spamming to everyone. Eight is plenty: the coordinator collapses
        # a burst into a single in-flight rewrite per session.
        self._rewrite_client = httpx.AsyncClient(
            timeout=25.0, limits=httpx.Limits(max_connections=8)
        )
        self._pending: set[asyncio.Task] = set()  # strong task refs; see fire()

    def _sign(self, method: str, path: str, body: bytes) -> dict[str, str]:
        # Canonical string/signature compatible with biz-core tracking ingest.
        ts = str(int(time.time() * 1000))
        nonce = secrets.token_hex(16)
        canonical = "\n".join(
            [method.upper(), path, ts, nonce, hashlib.sha256(body).hexdigest()]
        )
        sig = base64.b64encode(
            hmac_mod.new(
                self._cfg.callback_secret.encode(), canonical.encode(), hashlib.sha256
            ).digest()
        ).decode()
        return {
            "Content-Type": "application/json",
            "X-Track-App-Id": self._cfg.callback_app_id,
            "X-Track-Key-Id": self._cfg.callback_key_id,
            "X-Track-Timestamp": ts,
            "X-Track-Nonce": nonce,
            "X-Track-Signature": sig,
            "X-Track-Signature-Version": "v1",
        }

    async def _post(self, path: str, payload: dict) -> bool:
        body = json.dumps(payload).encode()
        url = self._cfg.callback_url.rstrip("/") + path
        for attempt in range(3):
            try:
                resp = await self._client.post(
                    url, content=body, headers=self._sign("POST", path, body)
                )
                if resp.status_code < 300:
                    return True
                logger.warning(
                    "world callback returned %s (%s)", resp.status_code, path
                )
                # A 4xx rejects the callback contract itself, so retrying the
                # same request cannot change the result. Only network failures
                # and 5xx responses enter the bounded retry below.
                if resp.status_code < 500:
                    return False
            except Exception as exc:  # noqa: BLE001 - callback failures are logged.
                logger.warning(
                    "world callback failed (%s), attempt %d: %s",
                    path,
                    attempt + 1,
                    exc,
                )
            await asyncio.sleep(0.5 * (attempt + 1))
        return False

    def fire(self, path: str, payload: dict) -> asyncio.Task:
        """Dispatch asynchronously without blocking the session path.

        Keep a strong task reference as asyncio recommends, because the event
        loop only keeps weak references. Done callbacks remove completed tasks.
        """
        task = asyncio.get_running_loop().create_task(self._post(path, payload))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def rewrite(
        self,
        run_id: str,
        event_id: int | None,
        role: str,
        instruction: str,
        baseline: str,
    ) -> dict:
        """Rewrite a player instruction synchronously for the interaction path.

        world-service owns rewrite-model credentials and prompt assets. The
        gateway sends only the raw user instruction and current baseline, then
        receives a new full prompt. The 25s timeout leaves room for retries even
        though the model call itself is normally much shorter.
        """
        path = "/internal/v1/rewrite"
        payload = {
            "run_id": run_id,
            "instruction": instruction,
            "baseline": baseline,
        }
        if event_id is not None:
            # Paired Web sessions use run_id + event_id as their shared
            # idempotency key. Legacy primary clients predate event_id; keep
            # their original request shape so other primary traffic is not
            # coupled to the comparison rollout.
            payload.update({"event_id": event_id, "role": role})
        body = json.dumps(payload).encode()
        url = self._cfg.callback_url.rstrip("/") + path
        resp = await self._rewrite_client.post(
            url, content=body, headers=self._sign("POST", path, body)
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data.get("prompt"):
            raise ValueError("rewrite response missing prompt")
        return data

    def started(
        self,
        run_id: str,
        attempt_id: str,
        trace_id: str,
        max_lifetime_s: int,
        admitted_at_ms: int,
    ) -> None:
        self.fire(
            "/internal/v1/runs/started",
            {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "trace_id": trace_id,
                "admitted_at_ms": admitted_at_ms,
                "max_lifetime_s": max_lifetime_s,
            },
        )

    def first_frame(
        self,
        run_id: str,
        attempt_id: str,
        trace_id: str,
        first_media_at_ms: int,
    ) -> None:
        self.fire(
            "/internal/v1/runs/first-frame",
            {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "trace_id": trace_id,
                "first_media_at_ms": first_media_at_ms,
            },
        )

    def ended(
        self,
        run_id: str,
        attempt_id: str,
        trace_id: str,
        reason: str,
        *,
        ended_at_ms: int,
    ) -> None:
        self.fire(
            "/internal/v1/runs/ended",
            {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "trace_id": trace_id,
                "ended_at_ms": ended_at_ms,
                "reason": reason,
            },
        )

    def recorded(self, run_id: str, attempt_id: str, video_uri: str) -> asyncio.Task:
        return self.fire(
            "/internal/v1/runs/recorded",
            {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "video_uri": video_uri,
            },
        )

    def aborted(
        self,
        run_id: str,
        attempt_id: str,
        trace_id: str,
        fault: str,
        reason: str,
        established: bool,
        *,
        disconnect_kind: str | None = None,
        disconnect_phase: str | None = None,
        websocket_close_code: int | None = None,
        disconnected_at_ms: int | None = None,
    ) -> None:
        """Report session termination.

        established=False means this session never actually came up (it never
        reported started). The business side uses that to ignore "a connection
        attempt that never established wants to terminate a run that is already
        live" - such callbacks come from the same token landing on another
        gateway replica, or from another run of the same account hitting the
        engine's single-lease admission. The rejected party is the newcomer;
        the live session is an innocent bystander.
        """
        payload = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "trace_id": trace_id,
            "fault": fault,
            "reason": reason,
            "established": established,
        }
        optional_diagnostics = {
            "disconnect_kind": disconnect_kind,
            "disconnect_phase": disconnect_phase,
            "websocket_close_code": websocket_close_code,
            "disconnected_at_ms": disconnected_at_ms,
        }
        payload.update(
            {
                name: value
                for name, value in optional_diagnostics.items()
                if value is not None
            }
        )
        self.fire("/internal/v1/runs/aborted", payload)

    async def close(self) -> None:
        if self._pending:
            _, pending = await asyncio.wait(tuple(self._pending), timeout=5.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        await self._client.aclose()
        await self._rewrite_client.aclose()

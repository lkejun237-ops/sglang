# SPDX-License-Identifier: Apache-2.0
"""Session token parsing: the account-level pseudonym and its fallback."""

import asyncio
import base64
import concurrent.futures
import json
import time
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sglang.multimodal_gen.runtime.realtime.world_platform import (
    Principal,
    TokenError,
    TokenReplayGuard,
    WorldCallbacks,
    load_public_key,
    verify_session_token,
)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _sign(claims: dict) -> tuple[str, str]:
    """Mint a token in the world-service compact JWS shape; returns (token, pub)."""
    key = Ed25519PrivateKey.generate()
    signing = (
        _b64(json.dumps({"alg": "EdDSA", "typ": "JWT"}).encode())
        + "."
        + _b64(json.dumps(claims).encode())
    )
    token = signing + "." + _b64(key.sign(signing.encode()))
    return token, base64.b64encode(key.public_key().public_bytes_raw()).decode()


def _claims(**extra) -> dict:
    now = int(time.time())
    base = {
        "iss": "world-service",
        "aud": "world-session",
        "sub": "s_abc123",
        "sid": "11111111-1111-1111-1111-111111111111",
        "jti": "22222222-2222-2222-2222-222222222222",
        "iat": now,
        "exp": now + 90,
        "session": {"max_lifetime_s": 90, "allow_free_prompt": True},
    }
    base.update(extra)
    return base


def test_lease_key_prefers_account_pseudonym():
    """One lease per user must key on the account pseudonym, not the per-run one."""
    token, pub = _sign(_claims(uid="u_deadbeef"))
    principal = verify_session_token(token, load_public_key(pub))
    assert principal.account_id == "u_deadbeef"
    assert principal.user_id == "s_abc123"  # per-run pseudonym stays, for logs
    assert principal.lease_key == "u_deadbeef"


def test_lease_key_falls_back_for_tokens_without_account():
    """Older tokens carry no uid, so the lease falls back to the per-run pseudonym.

    That fallback is what decouples the rollout order of the two repositories.
    Without it, deploying the gateway first would turn every in-flight token's
    lease key into an empty string, letting only one player in at a time.
    """
    token, pub = _sign(_claims())
    principal = verify_session_token(token, load_public_key(pub))
    assert principal.account_id == ""
    assert principal.lease_key == "s_abc123"


def test_lease_key_is_never_empty_for_a_valid_token():
    """An empty lease key would make the coordinator treat everyone as one user."""
    principal = Principal(
        user_id="s_x", run_id="r", max_lifetime_s=1, allow_free_prompt=False
    )
    assert principal.lease_key == "s_x"


def test_attempt_id_is_required_and_must_be_a_uuid():
    for value in (None, "", "not-a-uuid"):
        claims = _claims()
        if value is None:
            claims.pop("jti")
        else:
            claims["jti"] = value
        token, pub = _sign(claims)

        with pytest.raises(TokenError, match="jti must be a UUID"):
            verify_session_token(token, load_public_key(pub))


def test_attempt_id_is_returned_exactly_as_signed():
    attempt_id = "22222222-2222-2222-2222-22222222222A"
    token, pub = _sign(_claims(jti=attempt_id))

    assert verify_session_token(token, load_public_key(pub)).jti == attempt_id


def test_replay_guard_has_exactly_one_concurrent_attempt_owner():
    guard = TokenReplayGuard()
    principal = Principal(
        user_id="s_x",
        run_id="run-1",
        max_lifetime_s=90,
        allow_free_prompt=False,
        jti="22222222-2222-2222-2222-222222222222",
        exp=int(time.time()) + 90,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda _: guard.consume(principal), range(64)))

    assert results.count(True) == 1
    assert results.count(False) == 63


def test_aborted_payload_carries_established_flag():
    """Terminal callbacks must state whether the session ever came up.

    The business side uses the flag to refuse "a connection attempt that never
    established wants to terminate a run that is already live" - such callbacks
    come from the same token landing on another gateway replica, or another run
    of the same account hitting single-lease admission. Without the flag, a
    player mid-session is kicked off by somebody else's failed connection.
    """
    fired = []

    class _Stub(WorldCallbacks):
        def __init__(self):  # no http client needed
            pass

        def fire(self, path, payload):
            fired.append((path, payload))

    stub = _Stub()
    stub.aborted(
        "run-1",
        "attempt-1",
        "t",
        fault="ours",
        reason="USER_SESSION_LIMIT",
        established=False,
    )
    stub.aborted(
        "run-2",
        "attempt-2",
        "t",
        fault="ours",
        reason="engine crashed",
        established=True,
    )

    assert fired[0][0] == "/internal/v1/runs/aborted"
    assert fired[0][1]["attempt_id"] == "attempt-1"
    assert fired[0][1]["established"] is False
    assert fired[1][1]["attempt_id"] == "attempt-2"
    assert fired[1][1]["established"] is True


def test_lifecycle_callbacks_carry_the_token_attempt_id():
    fired = []

    class _Stub(WorldCallbacks):
        def __init__(self):
            pass

        def fire(self, path, payload):
            fired.append((path, payload))

    stub = _Stub()
    stub.started("run-1", "attempt-1", "trace-1", 90)
    stub.ended("run-1", "attempt-1", "trace-1", "completed")

    assert [path for path, _ in fired] == [
        "/internal/v1/runs/started",
        "/internal/v1/runs/ended",
    ]
    assert all(payload["attempt_id"] == "attempt-1" for _, payload in fired)


def test_callback_retries_5xx_but_treats_4xx_as_terminal(monkeypatch):
    class Client:
        def __init__(self, statuses):
            self.statuses = iter(statuses)
            self.calls = 0

        async def post(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(status_code=next(self.statuses))

    async def no_sleep(_delay):
        pass

    async def run():
        callbacks = WorldCallbacks.__new__(WorldCallbacks)
        callbacks._cfg = SimpleNamespace(
            callback_url="http://world-service.invalid",
            callback_app_id="gateway",
            callback_key_id="k1",
            callback_secret="secret",
        )
        monkeypatch.setattr(
            "sglang.multimodal_gen.runtime.realtime.world_platform.asyncio.sleep",
            no_sleep,
        )

        terminal = Client([409, 200])
        callbacks._client = terminal
        assert await callbacks._post("/callback", {"attempt_id": "a"}) is False
        assert terminal.calls == 1

        retryable = Client([503, 200])
        callbacks._client = retryable
        assert await callbacks._post("/callback", {"attempt_id": "a"}) is True
        assert retryable.calls == 2

    asyncio.run(run())

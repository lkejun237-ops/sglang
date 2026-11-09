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
    WorldPlatformConfig,
    WorldPlatformProfiles,
    load_public_key,
    session_payload_aad,
    verify_session_token,
)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _sign(
    claims: dict,
    *,
    kid: str | None = None,
    key: Ed25519PrivateKey | None = None,
) -> tuple[str, str]:
    """Mint a token in the world-service compact JWS shape; returns (token, pub)."""
    key = key or Ed25519PrivateKey.generate()
    header = {"alg": "EdDSA", "typ": "JWT"}
    if kid is not None:
        header["kid"] = kid
    signing = (
        _b64(json.dumps(header).encode()) + "." + _b64(json.dumps(claims).encode())
    )
    token = signing + "." + _b64(key.sign(signing.encode()))
    return token, base64.b64encode(key.public_key().public_bytes_raw()).decode()


def _profile(profile_id: str, key: Ed25519PrivateKey) -> WorldPlatformConfig:
    return WorldPlatformConfig(
        profile_id=profile_id,
        public_key=key.public_key(),
        callback_url=f"https://{profile_id}.example",
        callback_app_id="zing-gateway",
        callback_key_id="k1",
        callback_secret=f"secret-{profile_id}",
        recording_bucket=f"bucket-{profile_id}",
        recording_region="us-east-2",
    )


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
        "session": {
            "play_seconds": 52,
            "max_lifetime_s": 90,
            "allow_free_prompt": True,
            "record_video": True,
        },
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


def test_signed_key_id_selects_the_complete_platform_profile():
    us_key = Ed25519PrivateKey.generate()
    cn_key = Ed25519PrivateKey.generate()
    profiles = WorldPlatformProfiles(
        [_profile("world-us-k1", us_key), _profile("world-cn-k1", cn_key)],
        legacy_profile_id="world-us-k1",
    )
    token, _ = _sign(_claims(), kid="world-cn-k1", key=cn_key)

    principal, profile = profiles.verify(token)

    assert principal.profile_id == "world-cn-k1"
    assert profile.callback_secret == "secret-world-cn-k1"
    assert profile.recording_bucket == "bucket-world-cn-k1"


def test_profile_registry_rejects_unknown_key_id_and_wrong_issuer():
    key = Ed25519PrivateKey.generate()
    profiles = WorldPlatformProfiles([_profile("world-us-k1", key)])
    unknown, _ = _sign(_claims(), kid="world-cn-k1", key=key)
    wrong_issuer, _ = _sign(_claims(iss="another-service"), kid="world-us-k1", key=key)

    with pytest.raises(TokenError, match="unknown token key ID"):
        profiles.verify(unknown)
    with pytest.raises(TokenError, match="iss mismatch"):
        profiles.verify(wrong_issuer)


def test_legacy_profile_accepts_only_missing_or_matching_key_id():
    key = Ed25519PrivateKey.generate()
    profile = _profile("world-us-k1", key)
    profiles = WorldPlatformProfiles.from_legacy(profile)
    no_kid, _ = _sign(_claims(), key=key)
    matching, _ = _sign(_claims(), kid="world-us-k1", key=key)

    assert profiles.verify(no_kid)[0].profile_id == "world-us-k1"
    assert profiles.verify(matching)[0].profile_id == "world-us-k1"


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
        user_id="s_x",
        run_id="r",
        play_seconds=1,
        max_lifetime_s=7,
        allow_free_prompt=False,
        record_video=False,
    )
    assert principal.lease_key == "s_x"


def test_recording_policy_is_read_from_the_verified_session_claim():
    token, pub = _sign(_claims())

    principal = verify_session_token(token, load_public_key(pub))

    assert principal.record_video is True

    claims = _claims()
    claims["session"]["record_video"] = False
    token, pub = _sign(claims)
    assert verify_session_token(token, load_public_key(pub)).record_video is False


@pytest.mark.parametrize("play_seconds", [52, 60])
def test_dynamic_play_seconds_are_preserved_from_the_verified_claim(play_seconds):
    claims = _claims()
    claims["session"]["play_seconds"] = play_seconds
    token, pub = _sign(claims)

    principal = verify_session_token(token, load_public_key(pub))

    assert principal.play_seconds == play_seconds


@pytest.mark.parametrize("invalid_value", [None, True, False, 0, -1, "52", 52.0])
def test_play_seconds_is_required_and_must_be_a_positive_integer(invalid_value):
    claims = _claims()
    if invalid_value is None:
        claims["session"].pop("play_seconds")
    else:
        claims["session"]["play_seconds"] = invalid_value
    token, pub = _sign(claims)

    with pytest.raises(TokenError, match="play_seconds must be a positive integer"):
        verify_session_token(token, load_public_key(pub))


def test_session_token_requires_positive_startup_budget_after_finish_tail():
    claims = _claims()
    claims["session"].update({"play_seconds": 52, "max_lifetime_s": 57})
    token, pub = _sign(claims)

    with pytest.raises(TokenError, match="session startup budget must be positive"):
        verify_session_token(token, load_public_key(pub))


def test_missing_role_is_a_backwards_compatible_primary_session():
    token, pub = _sign(_claims())

    principal = verify_session_token(token, load_public_key(pub))

    assert principal.role == "primary"


def test_session_payload_aad_preserves_primary_and_separates_comparison():
    run_id = "11111111-1111-1111-1111-111111111111"

    assert session_payload_aad(run_id, "primary") == run_id
    assert session_payload_aad(run_id, "comparison") == f"{run_id}:comparison"
    with pytest.raises(ValueError, match="session role must be primary or comparison"):
        session_payload_aad(run_id, "secondary")


def test_comparison_role_allows_signed_free_prompt_without_recording():
    claims = _claims()
    claims["session"].update(
        {
            "role": "comparison",
            "allow_free_prompt": True,
            "record_video": False,
        }
    )
    token, pub = _sign(claims)

    principal = verify_session_token(token, load_public_key(pub))

    assert principal.role == "comparison"
    assert principal.allow_free_prompt is True
    assert principal.record_video is False


@pytest.mark.parametrize("invalid_role", [None, "", "secondary", 1, True])
def test_session_role_rejects_unknown_or_non_string_values(invalid_role):
    claims = _claims()
    claims["session"]["role"] = invalid_role
    token, pub = _sign(claims)

    with pytest.raises(TokenError, match="session role must be primary or comparison"):
        verify_session_token(token, load_public_key(pub))


def test_comparison_role_still_rejects_recording():
    claims = _claims()
    claims["session"].update(
        {
            "role": "comparison",
            "allow_free_prompt": True,
            "record_video": True,
        }
    )
    token, pub = _sign(claims)

    with pytest.raises(TokenError, match="comparison sessions must disable recording"):
        verify_session_token(token, load_public_key(pub))


@pytest.mark.parametrize("invalid_value", [None, "false", 0, 1])
def test_recording_policy_is_required_and_must_be_a_boolean(invalid_value):
    claims = _claims()
    if invalid_value is None:
        claims["session"].pop("record_video")
    else:
        claims["session"]["record_video"] = invalid_value
    token, pub = _sign(claims)

    with pytest.raises(TokenError, match="record_video must be boolean"):
        verify_session_token(token, load_public_key(pub))


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
        play_seconds=52,
        max_lifetime_s=90,
        allow_free_prompt=False,
        record_video=False,
        jti="22222222-2222-2222-2222-222222222222",
        exp=int(time.time()) + 90,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(lambda _: guard.consume(principal), range(64)))

    assert results.count(True) == 1
    assert results.count(False) == 63


def test_replay_guard_scopes_nonce_by_verified_profile():
    guard = TokenReplayGuard()
    common = dict(
        user_id="s_x",
        run_id="run-1",
        play_seconds=52,
        max_lifetime_s=90,
        allow_free_prompt=False,
        record_video=False,
        jti="22222222-2222-2222-2222-222222222222",
        exp=int(time.time()) + 90,
    )

    assert guard.consume(Principal(profile_id="world-us-k1", **common))
    assert guard.consume(Principal(profile_id="world-cn-k1", **common))
    assert not guard.consume(Principal(profile_id="world-us-k1", **common))


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
    assert "disconnect_kind" not in fired[0][1]
    assert "websocket_close_code" not in fired[0][1]
    assert fired[1][1]["attempt_id"] == "attempt-2"
    assert fired[1][1]["established"] is True


def test_aborted_disconnect_diagnostics_are_optional_and_signed_in_the_body():
    posted = []
    signed = []

    class Client:
        async def post(self, url, *, content, headers):
            posted.append((url, content, headers))
            return SimpleNamespace(status_code=200)

    async def run():
        callbacks = WorldCallbacks.__new__(WorldCallbacks)
        callbacks._cfg = SimpleNamespace(
            callback_url="http://world-service.invalid",
        )
        callbacks._client = Client()
        callbacks._pending = set()

        def sign(method, path, body):
            signed.append((method, path, body))
            return {"X-Track-Signature": "signed"}

        callbacks._sign = sign
        callbacks.aborted(
            "run-1",
            "attempt-1",
            "trace-1",
            fault="client",
            reason="USER_LEFT",
            established=True,
            disconnect_kind="connection_lost",
            disconnect_phase="playing",
            websocket_close_code=1006,
            disconnected_at_ms=1_788_406_622_000,
        )
        await asyncio.gather(*tuple(callbacks._pending))

    asyncio.run(run())

    assert len(posted) == 1
    url, body, headers = posted[0]
    assert url == "http://world-service.invalid/internal/v1/runs/aborted"
    assert headers == {"X-Track-Signature": "signed"}
    assert signed == [("POST", "/internal/v1/runs/aborted", body)]
    assert json.loads(body) == {
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "trace_id": "trace-1",
        "fault": "client",
        "reason": "USER_LEFT",
        "established": True,
        "disconnect_kind": "connection_lost",
        "disconnect_phase": "playing",
        "websocket_close_code": 1006,
        "disconnected_at_ms": 1_788_406_622_000,
    }


def test_lifecycle_callbacks_carry_the_token_attempt_id():
    fired = []
    scheduled = object()

    class _Stub(WorldCallbacks):
        def __init__(self):
            pass

        def fire(self, path, payload):
            fired.append((path, payload))
            return scheduled

    stub = _Stub()
    stub.started("run-1", "attempt-1", "trace-1", 90, 500)
    stub.first_frame("run-1", "attempt-1", "trace-1", 1_000)
    stub.ended(
        "run-1",
        "attempt-1",
        "trace-1",
        "completed",
        ended_at_ms=53_000,
    )
    result = stub.recorded("run-1", "attempt-1", "s3://videos/runs/run-1/video.mp4")

    assert result is scheduled
    assert [path for path, _ in fired] == [
        "/internal/v1/runs/started",
        "/internal/v1/runs/first-frame",
        "/internal/v1/runs/ended",
        "/internal/v1/runs/recorded",
    ]
    assert all(payload["attempt_id"] == "attempt-1" for _, payload in fired)
    assert fired[0][1]["admitted_at_ms"] == 500
    assert fired[1][1]["first_media_at_ms"] == 1_000
    assert fired[2][1]["ended_at_ms"] == 53_000
    assert fired[3][1] == {
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "video_uri": "s3://videos/runs/run-1/video.mp4",
    }


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


def test_rewrite_signs_coordinated_fields_and_preserves_legacy_primary_body():
    calls = []
    signed = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"prompt": "a rainy world", "change_type": "persistent"}

    class Client:
        async def post(self, url, *, content, headers):
            calls.append((url, content, headers))
            return Response()

    async def run():
        callbacks = WorldCallbacks.__new__(WorldCallbacks)
        callbacks._cfg = SimpleNamespace(
            callback_url="http://world-service.invalid",
        )
        callbacks._rewrite_client = Client()

        def sign(method, path, body):
            signed.append((method, path, body))
            return {"X-Track-Signature": "signed"}

        callbacks._sign = sign

        result = await callbacks.rewrite(
            "run-1",
            42,
            "comparison",
            "make it rain",
            "a quiet world",
        )
        legacy_result = await callbacks.rewrite(
            "run-legacy",
            None,
            "primary",
            "make it sunny",
            "a cloudy world",
        )

        assert result == {
            "prompt": "a rainy world",
            "change_type": "persistent",
        }
        assert legacy_result == result
        assert len(calls) == 2
        url, body, headers = calls[0]
        assert url == "http://world-service.invalid/internal/v1/rewrite"
        assert json.loads(body) == {
            "run_id": "run-1",
            "event_id": 42,
            "role": "comparison",
            "instruction": "make it rain",
            "baseline": "a quiet world",
        }
        assert headers == {"X-Track-Signature": "signed"}
        legacy_url, legacy_body, legacy_headers = calls[1]
        assert legacy_url == url
        assert json.loads(legacy_body) == {
            "run_id": "run-legacy",
            "instruction": "make it sunny",
            "baseline": "a cloudy world",
        }
        assert legacy_headers == headers
        assert signed == [
            ("POST", "/internal/v1/rewrite", body),
            ("POST", "/internal/v1/rewrite", legacy_body),
        ]

    asyncio.run(run())

"""Cleanup failures must retain the original owner and capacity for diagnosis."""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSessionCache,
    RealtimeSessionCapacityError,
)


class Lease:
    def __init__(self):
        self.calls = 0

    def release(self):
        self.calls += 1


class FailingState(BaseRealtimeState):
    def __init__(self):
        self.fail = True
        self.calls = 0

    def dispose(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("decoder abort unconfirmed")


class HealthyState(BaseRealtimeState):
    def __init__(self):
        self.calls = 0

    def dispose(self):
        self.calls += 1


def request(id="a", block=0):
    return SimpleNamespace(realtime_session_id=id, block_idx=block, session=None)


def test_dispose_failure_retains_state_and_pool_then_retries_only_failed_owner():
    cache = RealtimeSessionCache(max_sessions=1)
    first = request()
    cache.attach(first)
    bad = first.session.get_or_create_state(FailingState)
    good = first.session.get_or_create_state(HealthyState)
    first.session.zing_kv_lease = lease = Lease()
    with pytest.raises(RuntimeError, match="unconfirmed"):
        cache.release("a")
    assert cache._sessions["a"] is first.session
    assert first.session.get_state(FailingState) is bad
    assert first.session.get_state(HealthyState) is None
    assert lease.calls == 0 and good.calls == 1
    with pytest.raises(RealtimeSessionCapacityError):
        cache.attach(request("b"))
    with pytest.raises(RuntimeError, match="cleanup is pending"):
        first.session.get_or_create_state(HealthyState)
    bad.fail = False
    assert cache.release("a") is True
    assert lease.calls == 1 and good.calls == 1 and bad.calls == 2
    assert cache.release("a") is False
    cache.attach(request("b"))


def test_stale_cleanup_failure_does_not_admit_another_player():
    now = [0.0]
    cache = RealtimeSessionCache(max_sessions=1, stale_after_s=5, clock=lambda: now[0])
    first = request()
    cache.attach(first)
    bad = first.session.get_or_create_state(FailingState)
    now[0] = 10.0
    with pytest.raises(RuntimeError, match="unconfirmed"):
        cache.attach(request("b"))
    assert list(cache._sessions) == ["a"]
    bad.fail = False
    cache.attach(request("b"))
    assert list(cache._sessions) == ["b"]


def test_reset_cannot_replace_an_owner_whose_cleanup_failed():
    cache = RealtimeSessionCache(max_sessions=1)
    first = request()
    cache.attach(first)
    first.session.get_or_create_state(FailingState)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        cache.attach(request("a"))
    assert cache._sessions["a"] is first.session

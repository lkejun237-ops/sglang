"""Capture the diagnostic output boundary without touching business owners."""

import pytest


@pytest.fixture
def player_log_capture(caplog, monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox import diagnostics

    monkeypatch.setattr(diagnostics, "_emit_player_failure", caplog.handler.handle)
    return caplog

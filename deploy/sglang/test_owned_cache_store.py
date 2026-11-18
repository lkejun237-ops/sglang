import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import owned_cache_store as store


def test_default_does_not_start_or_stop_any_store(monkeypatch, tmp_path):
    monkeypatch.setattr(
        store.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("spawn")
    )
    with store.owned_store({}, tmp_path) as check:
        check()


@pytest.mark.parametrize("failure", [None, "early_exit", "runtime_exit", "identity"])
def test_owned_store_waits_for_its_receipt_and_reaps_only_its_child(
    monkeypatch, tmp_path, failure
):
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps(
            dict(
                store_memory_bytes=128 << 20, tiers=dict(state_directory=str(tmp_path))
            )
        )
    )
    child = type("Child", (), {})()
    child.pid = 123456
    child.code = 1 if failure == "early_exit" else None
    child.poll = lambda: child.code
    waits, kills = [], []
    child.wait = lambda **kw: waits.append(kw)

    def spawn(argv, **kwargs):
        assert argv[1] == str(tmp_path / "deploy/whitebox/cache_store.py")
        assert kwargs["start_new_session"] is True
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["MC_STORE_PIN_MEMORY_MAX_BYTES"] == str(128 << 20)
        assert kwargs["env"]["MC_STORE_MEMCPY"] == "0"
        (tmp_path / "store-ready.json").write_text(
            json.dumps(
                dict(
                    pid=child.pid,
                    memory_bytes=1 if failure == "identity" else 128 << 20,
                )
            )
        )
        return child

    monkeypatch.setattr(store.subprocess, "Popen", spawn)
    monkeypatch.setattr(store.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    config = dict(
        manage_cache_store=True,
        dit_options=dict(zing_block0_mooncake_config=str(config_path)),
    )

    def run():
        with store.owned_store(config, tmp_path) as check:
            if failure == "runtime_exit":
                child.code = 1
            check()

    if failure:
        with pytest.raises(RuntimeError, match="Owned cache Store"):
            run()
    else:
        run()
    assert [pid for pid, _ in kills] == [child.pid]
    assert waits == [{"timeout": 30}]

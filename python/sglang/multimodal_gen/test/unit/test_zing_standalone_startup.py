# SPDX-License-Identifier: Apache-2.0
"""Run the original launch path with process boundaries replaced, without CUDA.

Real role selection and launch orchestration execute here. Model equivalence
still requires the existing real-input GPU replays.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

RUNTIME = Path(__file__).resolve().parents[2] / "runtime"


def _functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(  # noqa: S102 - execute only the repository's own launch functions
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    spawned = []

    def unexpected_role(*args, **kwargs):
        pytest.fail("An externally managed role must not be spawned by DiT")

    class Pipe:
        def recv(self):
            return {"status": "ready"}

        def close(self):
            pass

    class Process:
        def __init__(self, **kwargs):
            self.args = kwargs["args"]

        def start(self):
            self.visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            spawned.append(self)

    namespace = {
        "os": os,
        "sys": SimpleNamespace(platform="linux"),
        "subprocess": SimpleNamespace(Popen=unexpected_role),
        "atexit": SimpleNamespace(register=unexpected_role),
        "mp": SimpleNamespace(Pipe=lambda **kwargs: (Pipe(), Pipe()), Process=Process),
        "configure_logger": lambda args: None,
        "logger": logging.getLogger(__name__),
        "run_scheduler_process": object(),
    }
    _functions(
        RUNTIME / "disaggregation/role_launcher.py",
        [
            "resolve_role_devices",
            "assign_denoiser_visible_devices",
            "start_role_workers",
        ],
        namespace,
    )
    _functions(RUNTIME / "launch_server.py", ["_launch_server"], namespace)
    args = SimpleNamespace(
        encoder_gpus=None,
        decoder_gpus=None,
        denoiser_gpus="1,3",
        disagg_encoder_port=31003,
        disagg_decoder_port=31004,
        disagg_encoder_ports=[31003, 31005],
        disagg_decoder_ports=[31004, 31006],
        num_gpus=2,
        nnodes=1,
        node_rank=0,
        master_port=32000,
    )
    return namespace["_launch_server"], args, spawned


@pytest.mark.parametrize(
    "visible,expected",
    [
        (None, "1,3"),
        ("4,5,6,7", "5,7"),
        ("GPU-a,GPU-b,GPU-c,GPU-d", "GPU-b,GPU-d"),
    ],
)
def test_external_roles_preserve_ports_and_select_dit_devices(
    launcher, monkeypatch, visible, expected
):
    launch, args, spawned = launcher
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    processes = launch(args, launch_http_server=False)
    assert processes == spawned and len(spawned) == 2
    assert [process.visible for process in spawned] == [expected, expected]
    assert [process.args[:2] for process in spawned] == [(0, 0), (1, 1)]
    assert args.disagg_encoder_ports == [31003, 31005]
    assert args.disagg_decoder_ports == [31004, 31006]
    assert args.disagg_encoder_port == 31003
    assert args.disagg_decoder_port == 31004


def test_external_dit_invalid_device_fails_before_spawn(launcher, monkeypatch):
    launch, args, spawned = launcher
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    with pytest.raises(ValueError, match="not visible"):
        launch(args, launch_http_server=False)
    assert not spawned


def test_external_dit_rejects_cuda_already_initialized(launcher, monkeypatch):
    launch, args, spawned = launcher
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA"):
        launch(args, launch_http_server=False)
    assert not spawned


def test_unpartitioned_dit_keeps_callers_visibility(launcher, monkeypatch):
    launch, args, spawned = launcher
    args.denoiser_gpus = None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c,GPU-a")
    launch(args, launch_http_server=False)
    assert [process.visible for process in spawned] == ["GPU-c,GPU-a"] * 2

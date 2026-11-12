"""Exercise the manual builder's actual CLI-to-Docker argv without Docker/network."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release

SOURCE = "a" * 40
SDK = "b" * 40


@pytest.fixture
def builder(monkeypatch, tmp_path):
    calls, exports, gates = [], [], []
    sdk_checkout = tmp_path / "sdk"

    def inputs(args):
        assert args.source == SOURCE
        assert args.sdk_checkout == sdk_checkout
        return SOURCE, SDK

    monkeypatch.setattr(release, "release_inputs", inputs)
    monkeypatch.setattr(release, "export", lambda *a, **kw: exports.append((a, kw)))
    monkeypatch.setattr(
        release.subprocess, "run", lambda *a, **kw: calls.append((a, kw))
    )
    monkeypatch.setattr(release, "check_headroom", lambda *a: gates.append(a))
    monkeypatch.setattr(release, "inspect_local", lambda *a: {"checked": True})

    def run(component, *extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "release.py",
                "build",
                "--source",
                SOURCE,
                "--sdk-checkout",
                str(sdk_checkout),
                "--component",
                component,
                "--tag",
                "fixture:release",
                *extra,
            ],
        )
        release.main()
        return calls, exports, gates

    return run


@pytest.mark.parametrize("component", ["cpu", "room"])
@pytest.mark.parametrize("mirrors", [False, True])
def test_build_mirrors_only_add_explicit_network_args(
    builder, capsys, component, mirrors
):
    extra = (
        [
            "--pip-index-url",
            "https://mirrors.aliyun.com/pypi/simple/",
            "--go-proxy",
            "https://goproxy.cn",
        ]
        if mirrors
        else []
    )
    calls, exports, gates = builder(component, *extra)
    assert len(calls) == 1
    (argv,), options = calls[0]
    expected = [
        "docker",
        "build",
        "--file",
        "docker/whitebox.Dockerfile",
        "--target",
        "room" if component == "room" else "release",
        "--tag",
        "fixture:release",
        "--build-arg",
        "WHITEBOX_SOURCE_SHA=" + SOURCE,
        "--build-arg",
        "WHITEBOX_SOURCE_BRANCH=main",
        "--build-arg",
        "WHITEBOX_SDK_SHA=" + SDK,
        "--label",
        "org.opencontainers.image.revision=" + SOURCE,
        "--label",
        "com.seedleap.worldkit.revision=" + SDK,
    ]
    if mirrors:
        expected += [
            "--build-arg",
            "PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/",
            "--build-arg",
            "GOPROXY=https://goproxy.cn",
        ]
    assert argv == expected + ["--platform", "linux/amd64", "."]
    assert options["check"] is True
    assert exports[0][0][:2] == (release.REPO, SOURCE)
    assert exports[1][0][1] == SDK
    assert exports[1][1] == {"room_sdk": component == "room"}
    assert gates == [(6 if component == "room" else 1, 15)]
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "source_branch": "main",
        "source_sha": SOURCE,
        "sdk_sha": SDK,
        "local_image_tag": "fixture:release",
        "component": component,
        "storage": {"checked": True},
        "deployed": False,
    }


@pytest.mark.parametrize(
    "flag,name,value",
    [
        ("--pip-index-url", "PIP_INDEX_URL", "https://pypi.org/simple"),
        ("--go-proxy", "GOPROXY", "https://goproxy.cn"),
    ],
)
def test_one_mirror_does_not_override_other_download_defaults(
    builder, flag, name, value
):
    calls, _, _ = builder("room", flag, value)
    argv = calls[0][0][0]
    network_args = [
        arg for arg in argv if arg.startswith(("PIP_INDEX_URL=", "GOPROXY="))
    ]
    assert network_args == [name + "=" + value]


def test_explicit_build_hosts_are_repeatable_and_keep_release_identity(builder):
    hosts = ["mirrors.aliyun.com:100.100.2.148", "mirror.example:[2001:db8::1]"]
    calls, exports, _ = builder(
        "room", "--build-host", hosts[0], "--build-host", hosts[1]
    )
    argv = calls[0][0][0]
    assert argv[-7:] == [
        "--add-host",
        hosts[0],
        "--add-host",
        hosts[1],
        "--platform",
        "linux/amd64",
        ".",
    ]
    assert "WHITEBOX_SOURCE_SHA=" + SOURCE in argv
    assert "WHITEBOX_SDK_SHA=" + SDK in argv
    assert exports[0][0][1] == SOURCE
    assert exports[1][0][1] == SDK
    assert not any(arg.startswith(("PIP_INDEX_URL=", "GOPROXY=")) for arg in argv)


@pytest.mark.parametrize(
    "value",
    [
        "mirror.example:host-gateway",
        "https://mirror.example:127.0.0.1",
        "mirror.example",
        "mirror.example=127.0.0.1",
        "*.example:127.0.0.1",
        "bad_host:127.0.0.1",
        "mirror.example:999.1.1.1",
        "mirror.example:[::1",
        "mirror.example:127.0.0.1 --build-arg WHITEBOX_SOURCE_SHA=other",
    ],
)
def test_invalid_build_host_rejected_before_build(builder, value):
    with pytest.raises(SystemExit) as error:
        builder("room", "--build-host", value)
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--pip-index-url", "--go-proxy"])
@pytest.mark.parametrize(
    "url",
    [
        "http://mirror.example/simple",
        "https://user:secret@mirror.example/simple",
        "https://mirror.example/simple?token=secret",
        "https://mirror.example/#secret",
        "https://mirror.example,direct",
        "https://mirror.example|direct",
        "https://mirror.example/\npath",
        "https://[invalid",
        "direct",
    ],
)
def test_invalid_mirror_is_rejected_before_any_build(builder, flag, url):
    with pytest.raises(SystemExit) as error:
        builder("room", flag, url)
    assert error.value.code == 2


def test_model_build_uses_pinned_base_image_without_kernels_checkout(
    builder, capsys, monkeypatch, tmp_path
):
    base = "registry.example/base@sha256:" + "0" * 64
    calls, exports, gates = builder("model", "--base", base)
    assert len(calls) == 1
    (argv,), options = calls[0]
    expected = [
        "docker",
        "build",
        "--file",
        "docker/whitebox-model.Dockerfile",
        "--target",
        "model",
        "--tag",
        "fixture:release",
        "--build-arg",
        "WHITEBOX_SOURCE_SHA=" + SOURCE,
        "--build-arg",
        "WHITEBOX_SOURCE_BRANCH=main",
        "--build-arg",
        "WHITEBOX_SDK_SHA=" + SDK,
        "--label",
        "org.opencontainers.image.revision=" + SOURCE,
        "--label",
        "com.seedleap.worldkit.revision=" + SDK,
        "--build-arg",
        "MODEL_BASE_IMAGE=" + base,
        "--platform",
        "linux/amd64",
        ".",
    ]
    assert argv == expected
    assert options["check"] is True
    assert gates == []
    assert len(exports) == 2
    assert exports[0][0][:2] == (release.REPO, SOURCE)
    assert exports[1][0][:2] == (tmp_path / "sdk", SDK)
    assert exports[1][1] == {"room_sdk": False}
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "source_branch": "main",
        "source_sha": SOURCE,
        "sdk_sha": SDK,
        "local_image_tag": "fixture:release",
        "component": "model",
        "storage": None,
        "deployed": False,
    }

    with pytest.raises(SystemExit) as error:
        builder(
            "model", "--base", base, "--kernels-checkout", str(tmp_path / "kernels")
        )
    assert error.value.code == 2

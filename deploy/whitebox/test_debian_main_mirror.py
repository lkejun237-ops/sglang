"""Build-only repository selection; all network and signature work is mocked."""

import subprocess

import debian_main_mirror as mirror
import pytest

SOURCE = b"""Types: deb deb-src
URIs: http://deb.debian.org/debian
Suites: bookworm bookworm-updates
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: http://deb.debian.org/debian-security
Suites: bookworm-security
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
"""
RELEASE = b"signed release\nCodename: bookworm\n"


def test_equal_verified_release_changes_only_main_and_preserves_source_types():
    selected, receipt = mirror.select_sources(
        SOURCE, fetch=lambda _: RELEASE, verify=lambda *_: True
    )
    blocks = selected.decode().strip().split("\n\n")
    assert len(blocks) == 3
    assert "URIs: https://mirrors.aliyun.com/debian\nSuites: bookworm\n" in blocks[0]
    assert (
        "URIs: https://deb.debian.org/debian\nSuites: bookworm-updates\n" in blocks[1]
    )
    assert blocks[2] == SOURCE.decode().strip().split("\n\n")[1].replace(
        "http://deb.debian.org/", "https://deb.debian.org/"
    )
    assert all(
        "Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg" in b for b in blocks
    )
    assert blocks[0].startswith("Types: deb deb-src\n")
    assert receipt["selected"] == "aliyun-main"
    assert receipt["reason"] == "identical-signed-release"


@pytest.mark.parametrize("failure", ["stale", "download", "signature", "codename"])
def test_untrusted_or_unavailable_mirror_uses_same_official_sources_over_https(failure):
    def fetch(url):
        if failure == "download":
            raise TimeoutError()
        if failure == "stale" and url == mirror.MIRROR_RELEASE:
            return RELEASE + b"older"
        return b"wrong distribution" if failure == "codename" else RELEASE

    selected, receipt = mirror.select_sources(
        SOURCE, fetch=fetch, verify=lambda *_: failure != "signature"
    )
    assert selected == SOURCE.replace(
        b"http://deb.debian.org/", b"https://deb.debian.org/"
    )
    assert receipt["selected"] == "official"
    assert receipt["reason"] != "identical-signed-release"


def test_snapshot_or_unrecognized_sources_do_not_trigger_network():
    original = SOURCE.replace(
        b"http://deb.debian.org/debian",
        b"https://snapshot.debian.org/archive/debian/20261020",
    )
    selected, receipt = mirror.select_sources(
        original, fetch=lambda _: pytest.fail("unsupported source fetched")
    )
    assert selected == original
    assert receipt["reason"] == "unsupported-source-layout"


@pytest.mark.parametrize("raises", [False, True])
def test_build_command_runs_once_and_restores_sources_on_failure(tmp_path, raises):
    source = tmp_path / "debian.sources"
    source.write_bytes(SOURCE)
    calls = []

    def command(args):
        calls.append(args)
        assert b"mirrors.aliyun.com" in source.read_bytes()
        assert b"URIs: https://deb.debian.org/debian-security" in source.read_bytes()
        assert b"URIs: http://deb.debian.org" not in source.read_bytes()
        if raises:
            raise OSError("fixture failure")
        return 23

    if raises:
        with pytest.raises(OSError):
            mirror.run(
                ["original-command"],
                source,
                fetch=lambda _: RELEASE,
                verify=lambda *_: True,
                execute=command,
            )
    else:
        assert (
            mirror.run(
                ["original-command"],
                source,
                fetch=lambda _: RELEASE,
                verify=lambda *_: True,
                execute=command,
            )
            == 23
        )
    assert calls == [["original-command"]]
    assert source.read_bytes() == SOURCE


def test_release_download_uses_a_hard_subprocess_deadline(monkeypatch):
    def run(args, **kwargs):
        assert args[-2:] == ["--fetch", mirror.ORIGIN_RELEASE]
        assert kwargs["timeout"] == mirror.FETCH_SECONDS
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(mirror.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        mirror.fetch_release(mirror.ORIGIN_RELEASE)


def test_official_build_wraps_every_apt_consumer_without_copying_helper():
    from pathlib import Path

    dockerfile = Path(__file__).resolve().parents[2] / "docker/whitebox.Dockerfile"
    text = dockerfile.read_text()
    apt_consumers = text.count("apt-get update") + text.count(
        "playwright install --with-deps"
    )
    assert apt_consumers > 0
    assert text.count("source=deploy/whitebox/debian_main_mirror.py") == apt_consumers
    assert text.count("python /tmp/debian_main_mirror.py -- sh -ec") == apt_consumers
    assert "COPY deploy/whitebox/debian_main_mirror.py" not in text
    assert "FROM python:3.14-slim-bookworm AS base" in text


def test_https_upgrade_does_not_rewrite_unknown_repository_or_comments():
    raw = b"# http://deb.debian.org/debian\nURIs: http://private.example/debian\n"
    assert mirror.official_https(raw) == raw

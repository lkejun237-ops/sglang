"""Build-input invariants, independent of model and image dependencies."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from project_runtime_source import project_runtime_source as project


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        value = (
            os.readlink(path)
            if path.is_symlink()
            else (path.read_bytes() if path.is_file() else None)
        )
        result[str(path.relative_to(root))] = (
            stat.S_IFMT(info.st_mode),
            stat.S_IMODE(info.st_mode),
            value,
            info.st_mtime_ns,
        )
    return result


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for name, content in {
        "__init__.py": b"",
        "multimodal_gen/test/unit/realtime/test_contract.py": b"assert True\n",
        "multimodal_gen/runtime/demo.py": b"runtime demo\n",
        "multimodal_gen/runtime/assets/pattern.bin": b"\x00\xff\x01",
        "apps/plugin/testset/input.json": b'{"runtime":true}',
        "test/utils.py": b"retained until separately reviewed\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "multimodal_gen/runtime/demo.py").chmod(0o751)
    (root / "asset-link").symlink_to("multimodal_gen/runtime/assets/pattern.bin")
    (root / "asset-dir").symlink_to(
        "multimodal_gen/runtime/assets", target_is_directory=True
    )
    return root


def test_only_reviewed_tests_are_excluded_and_source_is_unchanged(source, tmp_path):
    before = inventory(source)
    destination = tmp_path / "projected"
    project(source, destination)
    assert inventory(source) == before
    assert not (destination / "multimodal_gen/test").exists()
    assert (
        destination / "multimodal_gen/runtime/demo.py"
    ).read_bytes() == b"runtime demo\n"
    assert (
        stat.S_IMODE((destination / "multimodal_gen/runtime/demo.py").stat().st_mode)
        == 0o751
    )
    assert (destination / "apps/plugin/testset/input.json").is_file()
    assert (destination / "test/utils.py").is_file()
    assert (destination / "asset-link").is_symlink()
    assert (
        os.readlink(destination / "asset-link")
        == "multimodal_gen/runtime/assets/pattern.bin"
    )
    assert (destination / "asset-dir/pattern.bin").read_bytes() == b"\x00\xff\x01"
    assert all(item[3] == 0 for item in inventory(destination).values())


def test_test_only_changes_preserve_output_but_runtime_changes_do_not(source, tmp_path):
    project(source, tmp_path / "first")
    expected = inventory(tmp_path / "first")
    (source / "multimodal_gen/test/unit/realtime/test_contract.py").write_bytes(
        b"assert False\n"
    )
    (source / "multimodal_gen/test/new-fixture.json").write_text("new fixture")
    project(source, tmp_path / "second")
    assert inventory(tmp_path / "second") == expected
    (source / "multimodal_gen/runtime/demo.py").write_bytes(b"changed runtime\n")
    project(source, tmp_path / "third")
    assert inventory(tmp_path / "third") != expected


@pytest.mark.parametrize(
    "target",
    [
        "../outside",
        "/etc/passwd",
        "multimodal_gen/test/unit/realtime/test_contract.py",
        "multimodal_gen/test/../runtime/demo.py",
        "missing",
        "loop",
    ],
)
def test_invalid_links_fail_without_changing_source_or_destination(
    source, tmp_path, target
):
    (tmp_path / "outside").write_text("outside")
    (source / "loop").symlink_to(target)
    before = inventory(source)
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        project(source, destination)
    assert inventory(source) == before
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize(
    "kind",
    ["nonempty", "symlink", "ancestor-symlink", "inside-source", "source-itself"],
)
def test_unsafe_destination_is_rejected_without_source_mutation(source, tmp_path, kind):
    destination = tmp_path / "destination"
    if kind == "nonempty":
        destination.mkdir()
        (destination / "keep").write_text("preserve")
    elif kind == "symlink":
        destination.symlink_to(source, target_is_directory=True)
    elif kind == "ancestor-symlink":
        destination.symlink_to(tmp_path, target_is_directory=True)
        destination = destination / "nested"
    elif kind == "inside-source":
        destination = source / "projected"
    else:
        destination = source
    before = inventory(source)
    with pytest.raises(ValueError):
        project(source, destination)
    assert inventory(source) == before
    if kind == "nonempty":
        assert (destination / "keep").read_text() == "preserve"


def test_missing_source_does_not_create_destination(tmp_path):
    with pytest.raises(ValueError, match="[Ss]ource"):
        project(tmp_path / "missing", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_cli_supports_read_only_source_and_reports_english_errors(source, tmp_path):
    helper = Path(__file__).with_name("project_runtime_source.py")
    paths = [source, *(p for p in source.rglob("*") if not p.is_symlink())]
    modes = {path: stat.S_IMODE(path.stat().st_mode) for path in paths}
    try:
        for path in paths:
            path.chmod(modes[path] & ~0o222)
        before = inventory(source)
        result = subprocess.run(
            [
                sys.executable,
                str(helper),
                "--source",
                str(source),
                "--destination",
                str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
        )
        assert inventory(source) == before
    finally:
        for path, mode in modes.items():
            path.chmod(mode)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--source",
            str(tmp_path / "missing"),
            "--destination",
            str(tmp_path / "unused"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Source" in result.stderr
    assert result.stderr.isascii()


@pytest.mark.parametrize("relative", [".", "multimodal_gen/runtime"])
def test_private_source_directory_cannot_publish_an_unreadable_runtime(
    source, tmp_path, relative
):
    private = source / relative
    private.chmod(0o700)
    destination = tmp_path / "runtime"
    with pytest.raises(ValueError, match="not readable by its runtime user"):
        project(source, destination)
    assert not destination.exists()
    assert private.stat().st_mode & 0o777 == 0o700

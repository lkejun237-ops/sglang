"""Pinned real lint tools must check eligible files without changing source."""

import subprocess
import sys

import pytest
import yaml
from zing_ci_fixtures import cli, commit, make_repo, result_json, write

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, stage="base-a", runner_config="cpu")


def lint(repo, base, head):
    output = repo.parent / "lint.json"
    process = cli(
        "check_python.py", repo, "--base", base, "--head", head, "--output", output
    )
    return process, output


@pytest.mark.parametrize(
    "source,tool,diagnostic",
    [
        ("import os\n", "ruff", "F401"),
        ("value = unknown_name\n", "ruff", "F821"),
        (
            'from __future__ import annotations\n\ndef accept(value: "int"):\n    return value\n',
            "ruff",
            "UP037",
        ),
        ("value=  [1,2,3]\n", "ruff-format", ""),
        ("import sys\nimport os\n\nprint(sys.version, os.name)\n", "isort", ""),
    ],
)
def test_real_violations_fail_and_source_is_unchanged(
    tmp_path, source, tool, diagnostic
):
    repo = make_repo(tmp_path)
    base = commit(repo)
    path = "test/lint fixture.py"
    file = write(repo, path, source)
    head = commit(repo)
    before = file.read_bytes()
    # Establish that the fixture actually fails the pinned external tool.
    commands = {
        "ruff": [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select=F401,F821,UP037",
            path,
        ],
        "ruff-format": [sys.executable, "-m", "ruff", "format", "--check", path],
        "isort": [sys.executable, "-m", "isort", "--check-only", "--diff", path],
    }
    baseline = subprocess.run(
        commands[tool], cwd=repo, capture_output=True, text=True, timeout=15
    )
    assert baseline.returncode != 0, baseline.stdout + baseline.stderr
    if diagnostic:
        assert diagnostic in baseline.stdout + baseline.stderr
    process, output = lint(repo, base, head)
    assert process.returncode != 0
    assert file.read_bytes() == before, "CI lint must never apply fixes"
    receipt = result_json(output)
    assert receipt["eligible_files"] == [path]
    assert receipt["tools"][tool]["exit_code"] != 0
    assert diagnostic in receipt["tools"][tool]["output"]
    assert receipt["versions"]["ruff"] == "0.15.1"
    assert receipt["versions"]["isort"] == "7.0.0"


@pytest.mark.parametrize(
    "valid_import_order", [False, True], ids=["violation", "valid"]
)
def test_leading_option_filename_is_checked_as_a_file(tmp_path, valid_import_order):
    repo = make_repo(tmp_path)
    base = commit(repo)
    path = "--extension=.py"
    imports = (
        "import os\nimport sys\n" if valid_import_order else "import sys\nimport os\n"
    )
    file = write(repo, path, imports + "\nprint(sys.version, os.name)\n")
    ordinary = write(repo, "valid.py", "import os\n\nprint(os.name)\n")
    head = commit(repo)
    before = {path: file.read_bytes(), "valid.py": ordinary.read_bytes()}
    baseline = subprocess.run(
        [
            sys.executable,
            "-m",
            "isort",
            "--check-only",
            "--diff",
            "--settings-path",
            str(repo / ".isort.cfg"),
            "--",
            path,
            "valid.py",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert (baseline.returncode == 0) is valid_import_order, (
        baseline.stdout + baseline.stderr
    )
    process, output = lint(repo, base, head)
    receipt = result_json(output)
    assert receipt["eligible_files"] == [path, "valid.py"]
    assert file.read_bytes() == before[path]
    assert ordinary.read_bytes() == before["valid.py"]
    if not valid_import_order:
        assert process.returncode != 0, (
            "The option-shaped file's isort violation was ignored"
        )
        assert receipt["status"] == "error"
        assert receipt["tools"]["isort"]["exit_code"] != 0
        assert path in receipt["tools"]["isort"]["output"]
    else:
        assert process.returncode == 0, process.stderr
        assert receipt["status"] == "success"
        assert all(tool["exit_code"] == 0 for tool in receipt["tools"].values())
    command = receipt["tools"]["isort"]["command"]
    assert "--" in command, "isort must receive an explicit end-of-options delimiter"
    assert command[command.index("--") + 1 :] == [path, "valid.py"]


def test_valid_changed_file_passes_all_three_tools(tmp_path):
    repo = make_repo(tmp_path)
    base = commit(repo)
    path = "test/valid.py"
    file = write(repo, path, "import os\n\nprint(os.name)\n")
    head = commit(repo)
    process, output = lint(repo, base, head)
    assert process.returncode == 0, process.stderr
    receipt = result_json(output)
    assert receipt["eligible_files"] == [path]
    assert all(
        receipt["tools"][tool]["exit_code"] == 0
        for tool in ("ruff", "ruff-format", "isort")
    )
    assert file.read_text() == "import os\n\nprint(os.name)\n"


@pytest.mark.parametrize("kind", ["empty", "deleted", "non-python"])
def test_empty_eligible_set_never_invokes_tools_on_the_entire_repo(tmp_path, kind):
    repo = make_repo(tmp_path)
    write(repo, "test/old_debt.py", "not formatted = ????\n")
    deleted = write(repo, "test/deleted.py", "import unused\n")
    base = commit(repo)
    if kind == "deleted":
        deleted.unlink()
    elif kind == "non-python":
        write(repo, "notes/new.md", "docs\n")
    head = commit(repo)
    process, output = lint(repo, base, head)
    assert process.returncode == 0, process.stderr
    receipt = result_json(output)
    assert receipt["eligible_files"] == []
    assert receipt["file_count"] == 0
    assert all(record["invoked"] is False for record in receipt["tools"].values())


@pytest.mark.parametrize(
    "path",
    [
        "python/sglang/multimodal_gen/csrc/excluded.py",
        "python/sglang/kernels/ops/diffusion/render/excluded.py",
        "python/sglang/kernels/ops/attention/flash_attn/cute/excluded.py",
        "python/sglang/srt/grpc/example_pb2.py",
        "python/sglang/srt/grpc/example_pb2_grpc.py",
    ],
)
def test_repository_and_hook_excludes_are_preserved(tmp_path, path):
    repo = make_repo(tmp_path)
    base = commit(repo)
    write(repo, path, "bad syntax= ???\n")
    head = commit(repo)
    process, output = lint(repo, base, head)
    assert process.returncode == 0
    assert result_json(output)["eligible_files"] == []


def test_ruff_only_exclude_does_not_disable_format_and_isort(tmp_path):
    repo = make_repo(tmp_path)
    base = commit(repo)
    path = "python/sglang/example/__init__.py"
    write(repo, path, "import os\n")
    head = commit(repo)
    process, output = lint(repo, base, head)
    assert process.returncode == 0, process.stderr
    tools = result_json(output)["tools"]
    assert path not in tools["ruff"]["files"]
    assert path in tools["ruff-format"]["files"]
    assert path in tools["isort"]["files"]


@pytest.mark.parametrize("config", [".isort.cfg", ".pre-commit-config.yaml"])
def test_config_change_checks_full_eligible_tree_and_catches_historical_debt(
    tmp_path, config
):
    repo = make_repo(tmp_path)
    path = "test/existing_debt.py"
    write(repo, path, "import os\n")
    base = commit(repo)
    write(repo, config, (repo / config).read_text() + "\n# changed configuration\n")
    head = commit(repo)
    process, output = lint(repo, base, head)
    assert process.returncode != 0
    receipt = result_json(output)
    assert receipt["mode"] == "full"
    assert path in receipt["eligible_files"]


@pytest.mark.parametrize("config_name", ["ruff.toml", ".ruff.toml", "pyproject.toml"])
def test_nested_lint_config_checks_existing_eligible_files(tmp_path, config_name):
    repo = make_repo(tmp_path)
    path = "test/nested/example.py"
    file = write(repo, path, 'value = "hello"\n')
    base = commit(repo)
    section = "tool.ruff.format" if config_name == "pyproject.toml" else "format"
    write(repo, "test/nested/" + config_name, f'[{section}]\nquote-style = "single"\n')
    head = commit(repo)
    before = file.read_bytes()
    direct = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "--", path],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert direct.returncode == 1, direct.stdout + direct.stderr
    process, output = lint(repo, base, head)
    assert process.returncode != 0, (
        "Nested config changed old-file formatting without checking it"
    )
    receipt = result_json(output)
    assert receipt["mode"] == "full"
    assert path in receipt["eligible_files"]
    assert receipt["tools"]["ruff-format"]["exit_code"] != 0
    assert file.read_bytes() == before


@pytest.mark.parametrize("action", ["delete", "rename"])
def test_removed_nested_lint_config_checks_old_directory_files(tmp_path, action):
    repo = make_repo(tmp_path)
    path = "test/nested/example.py"
    file = write(repo, path, "value = 'hello'\n")
    config = write(repo, "test/nested/ruff.toml", '[format]\nquote-style = "single"\n')
    base = commit(repo)
    if action == "delete":
        config.unlink()
    else:
        destination = repo / "test/other/ruff.toml"
        destination.parent.mkdir()
        config.rename(destination)
    head = commit(repo)
    before = file.read_bytes()
    direct = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "--", path],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert direct.returncode == 1, direct.stdout + direct.stderr
    process, output = lint(repo, base, head)
    assert process.returncode != 0, (
        "Removing nested config must check files under its old path"
    )
    receipt = result_json(output)
    assert receipt["mode"] == "full"
    assert path in receipt["eligible_files"]
    assert receipt["tools"]["ruff-format"]["exit_code"] != 0
    assert file.read_bytes() == before


@pytest.mark.parametrize("config_name", [".isort.cfg", "pyproject.toml", "setup.cfg"])
@pytest.mark.parametrize(
    "scenario", ["existing-file", "changed-file", "valid-isolation"]
)
def test_nested_isort_configuration_applies_per_file(tmp_path, config_name, scenario):
    repo = make_repo(tmp_path)
    path = "test/nested/example.py"
    combined = "from pathlib import Path, PurePath\n\nprint(Path, PurePath)\n"
    separated = "from pathlib import Path\nfrom pathlib import PurePath\n\nprint(Path, PurePath)\n"
    file = write(repo, path, combined if scenario == "existing-file" else separated)
    others = ["root_example.py", "test/other/example.py"]
    for other in others:
        write(repo, other, combined)
    section = {
        ".isort.cfg": "settings",
        "pyproject.toml": "tool.isort",
        "setup.cfg": "isort",
    }[config_name]
    config_path = "test/nested/" + config_name
    config_source = (
        f'[{section}]\nprofile = "black"\nforce_single_line = true\n'
        if config_name == "pyproject.toml"
        else f"[{section}]\nprofile = black\nforce_single_line = true\n"
    )
    if scenario != "existing-file":
        write(repo, config_path, config_source)
    base = commit(repo)
    if scenario == "existing-file":
        write(repo, config_path, config_source)
    elif scenario == "changed-file":
        file.write_text(combined)
    else:
        # Change all three files without changing their import convention.
        for changed in [path, *others]:
            candidate = repo / changed
            candidate.write_text(candidate.read_text() + "value = 1\n")
    head = commit(repo)
    before = {name: (repo / name).read_bytes() for name in [path, *others]}
    direct = subprocess.run(
        [
            sys.executable,
            "-m",
            "isort",
            "--check-only",
            "--diff",
            "--resolve-all-configs",
            "--config-root",
            str(repo),
            "--",
            *others,
            path,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    valid = scenario == "valid-isolation"
    assert (direct.returncode == 0) is valid, direct.stdout + direct.stderr
    if not valid:
        assert "test/nested/example.py" in direct.stdout + direct.stderr
        for other in others:
            assert f"ERROR: {repo / other}" not in direct.stdout + direct.stderr
    process, output = lint(repo, base, head)
    assert (process.returncode == 0) is valid, (
        "Each file must use its nearest isort configuration"
    )
    receipt = result_json(output)
    assert path in receipt["tools"]["isort"]["files"]
    assert (receipt["tools"]["isort"]["exit_code"] == 0) is valid
    if scenario == "existing-file":
        assert receipt["mode"] == "full"
    if valid:
        assert all(other in receipt["tools"]["isort"]["files"] for other in others)
        assert receipt["status"] == "success"
    for name, content in before.items():
        assert (repo / name).read_bytes() == content


@pytest.mark.parametrize(
    "scenario",
    [
        "direct",
        "environment-path",
        "transitive-pyproject",
        "symlink-target",
        "delete",
        "rename",
        "cycle",
        "bad-toml",
    ],
)
def test_ruff_extended_configuration_inputs_cannot_be_ignored(
    tmp_path, monkeypatch, scenario
):
    repo = make_repo(tmp_path)
    nested = scenario == "transitive-pyproject"
    path = "test/nested/example.py" if nested else "test/example.py"
    source = write(repo, path, 'value = "hello"\n')
    leaf = "config/style.rules"
    if nested:
        write(
            repo,
            "test/nested/pyproject.toml",
            '[tool.ruff]\nextend = "../../config/base.rules"\n',
        )
        write(repo, "config/base.rules", 'extend = "deeper/policy.data"\n')
        leaf = "config/deeper/policy.data"
    else:
        write(repo, "ruff.toml", 'extend = "config/style.rules"\n')
    if scenario == "environment-path":
        monkeypatch.setenv("ISSUE277_RUFF_STYLE", "config/style.rules")
        write(repo, "ruff.toml", 'extend = "$ISSUE277_RUFF_STYLE"\n')
    if scenario == "symlink-target":
        leaf = "config/actual.policy"
        write(repo, leaf, '[format]\nquote-style = "double"\n')
        (repo / "config/style.rules").symlink_to("actual.policy")
    else:
        write(repo, leaf, '[format]\nquote-style = "double"\n')
    base = commit(repo)
    if scenario == "delete":
        (repo / leaf).unlink()
    elif scenario == "rename":
        (repo / leaf).rename(repo / "config/renamed.policy")
    elif scenario == "cycle":
        write(repo, leaf, 'extend = "../ruff.toml"\n')
    elif scenario == "bad-toml":
        write(repo, leaf, "[format\nquote-style =\n")
    else:
        write(repo, leaf, '[format]\nquote-style = "single"\n')
    head = commit(repo)
    before = source.read_bytes()
    direct = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "--", path],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    valid_change = scenario in {
        "direct",
        "environment-path",
        "transitive-pyproject",
        "symlink-target",
    }
    assert direct.returncode == (1 if valid_change else 2), (
        direct.stdout + direct.stderr
    )
    process, output = lint(repo, base, head)
    assert process.returncode != 0, (
        "A changed Ruff configuration dependency must not be ignored"
    )
    receipt = result_json(output)
    assert receipt["status"] == "error"
    if valid_change:
        assert receipt["mode"] == "full"
        assert path in receipt["eligible_files"]
        assert receipt["tools"]["ruff-format"]["exit_code"] != 0
    else:
        assert receipt.get("reason_code") == "CONFIG_DRIFT" or any(
            tool["exit_code"] != 0 for tool in receipt["tools"].values()
        )
    assert source.read_bytes() == before


def test_unrelated_malformed_toml_is_not_a_ruff_configuration_error(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, "test/example.py", 'value = "hello"\n')
    base = commit(repo)
    write(repo, "config/unrelated.toml", "[not-toml\n")
    head = commit(repo)
    direct = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "--", "test/example.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert direct.returncode == 0, direct.stdout + direct.stderr
    process, output = lint(repo, base, head)
    assert process.returncode == 0, process.stderr
    assert result_json(output)["status"] == "success"


def test_symlinked_isort_configuration_target_is_an_input(tmp_path):
    repo = make_repo(tmp_path)
    path = "test/example.py"
    source = write(
        repo, path, "from pathlib import Path, PurePath\n\nprint(Path, PurePath)\n"
    )
    config = repo / ".isort.cfg"
    content = config.read_text()
    write(repo, "config/import.policy", content)
    config.unlink()
    config.symlink_to("config/import.policy")
    base = commit(repo)
    write(repo, "config/import.policy", content + "\nforce_single_line = true\n")
    head = commit(repo)
    before = source.read_bytes()
    direct = subprocess.run(
        [
            sys.executable,
            "-m",
            "isort",
            "--check-only",
            "--diff",
            "--resolve-all-configs",
            "--config-root",
            str(repo),
            "--",
            path,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert direct.returncode == 1, direct.stdout + direct.stderr
    process, output = lint(repo, base, head)
    assert process.returncode != 0, (
        "A symlinked isort configuration target is a real input"
    )
    receipt = result_json(output)
    assert receipt["mode"] == "full"
    assert path in receipt["eligible_files"]
    assert receipt["tools"]["isort"]["exit_code"] != 0
    assert source.read_bytes() == before


def test_growing_directory_symlink_configuration_finishes_like_real_tools(tmp_path):
    repo = make_repo(tmp_path)
    path = "test/example.py"
    source = write(repo, path, "value = 1\n")
    base = commit(repo)
    (repo / ".ruff.toml").symlink_to("cycle/config.toml")
    (repo / "cycle").symlink_to("cycle/child", target_is_directory=True)
    head = commit(repo)
    before = source.read_bytes()
    commands = [
        [sys.executable, "-m", "ruff", "format", "--check", "--", path],
        [
            sys.executable,
            "-m",
            "isort",
            "--check-only",
            "--diff",
            "--resolve-all-configs",
            "--config-root",
            str(repo),
            "--",
            path,
        ],
    ]
    for command in commands:
        direct = subprocess.run(
            command, cwd=repo, capture_output=True, text=True, timeout=5
        )
        assert direct.returncode == 0, direct.stdout + direct.stderr
    output = repo.parent / "lint.json"
    try:
        process = cli(
            "check_python.py",
            repo,
            "--base",
            base,
            "--head",
            head,
            "--output",
            output,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "Configuration symlink traversal must terminate when pinned tools finish normally"
        )
    assert process.returncode == 0, process.stderr
    receipt = result_json(output)
    assert receipt["status"] == "success"
    assert all(tool["exit_code"] == 0 for tool in receipt["tools"].values())
    assert source.read_bytes() == before


def test_nested_editorconfig_checks_existing_python_imports(tmp_path):
    repo = make_repo(tmp_path)
    path = "test/nested/example.py"
    source = write(
        repo, path, "from pathlib import Path, PurePath\n\nprint(Path, PurePath)\n"
    )
    base = commit(repo)
    write(
        repo,
        "test/nested/.editorconfig",
        "root = true\n\n[*.py]\nprofile = black\nforce_single_line = true\n",
    )
    head = commit(repo)
    before = source.read_bytes()
    direct = subprocess.run(
        [
            sys.executable,
            "-m",
            "isort",
            "--check-only",
            "--diff",
            "--resolve-all-configs",
            "--config-root",
            str(repo),
            "--",
            path,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert direct.returncode == 1, direct.stdout + direct.stderr
    process, output = lint(repo, base, head)
    assert process.returncode != 0, (
        "isort's .editorconfig input must check existing Python files"
    )
    receipt = result_json(output)
    assert receipt["mode"] == "full"
    assert path in receipt["eligible_files"]
    assert receipt["tools"]["isort"]["exit_code"] != 0
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "hook,arguments,source",
    [
        (
            "isort",
            ["--force-single-line-imports"],
            "from pathlib import Path, PurePath\n\nprint(Path, PurePath)\n",
        ),
        ("ruff-format", ["--line-length=20"], "values = [1, 2, 3, 4, 5, 6]\n"),
    ],
)
def test_unreviewed_hook_arguments_fail_closed(tmp_path, hook, arguments, source):
    repo = make_repo(tmp_path)
    path = "test/example.py"
    file = write(repo, path, source)
    base = commit(repo)
    config = repo / ".pre-commit-config.yaml"
    payload = yaml.safe_load(config.read_text())
    for repository in payload["repos"]:
        for item in repository["hooks"]:
            if item["id"] == hook:
                item["args"] = arguments
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    head = commit(repo)
    before = file.read_bytes()
    command = (
        [
            sys.executable,
            "-m",
            "isort",
            "--check-only",
            "--diff",
            "--settings-path",
            str(repo / ".isort.cfg"),
        ]
        if hook == "isort"
        else [sys.executable, "-m", "ruff", "format", "--check"]
    )
    direct = subprocess.run(
        command + arguments + ["--", path],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert direct.returncode == 1, direct.stdout + direct.stderr
    process, output = lint(repo, base, head)
    assert process.returncode != 0, (
        "Configured hook arguments must not be silently discarded"
    )
    receipt = result_json(output)
    assert receipt["reason_code"] == "CONFIG_DRIFT"
    assert receipt["status"] == "error"
    assert file.read_bytes() == before


@pytest.mark.parametrize(
    "old,new",
    [
        ("rev: v0.15.1", "rev: v0.14.0"),
        ("rev: 7.0.0", "rev: 6.0.0"),
        ("--select=F401,F821,UP037", "--select=F401"),
    ],
)
def test_tool_version_or_rule_drift_fails_closed(tmp_path, old, new):
    repo = make_repo(tmp_path)
    base = commit(repo)
    config = repo / ".pre-commit-config.yaml"
    assert old in config.read_text()
    config.write_text(config.read_text().replace(old, new))
    head = commit(repo)
    process, output = lint(repo, base, head)
    assert process.returncode != 0
    assert result_json(output)["reason_code"] == "CONFIG_DRIFT"


def test_rename_only_checks_existing_new_python_path(tmp_path):
    repo = make_repo(tmp_path)
    old = write(repo, "test/old.py", "import os\n\nprint(os.name)\n")
    base = commit(repo)
    new = "test/new space.py"
    old.rename(repo / new)
    head = commit(repo)
    process, output = lint(repo, base, head)
    assert process.returncode == 0
    assert result_json(output)["eligible_files"] == [new]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

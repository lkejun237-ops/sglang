"""Check changed Python using the repository's pinned hook configuration."""

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml
from common import changed_paths, git, sha256, write_json

VERSIONS = {"ruff": "0.15.1", "isort": "7.0.0"}
CONFIG_NAMES = {
    ".pre-commit-config.yaml",
    ".isort.cfg",
    ".editorconfig",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "ruff.toml",
    ".ruff.toml",
}


# pyproject.toml also carries packaging metadata; only these keys can change what the
# pinned hooks do (Ruff infers its target version from requires-python).
PYPROJECT_LINT_KEYS = (
    ("tool", "ruff"),
    ("tool", "isort"),
    ("tool", "black"),
    ("project", "requires-python"),
)


def pyproject_lint_view(data):
    config = tomllib.loads(data.decode("utf-8"))
    view = {}
    for keys in PYPROJECT_LINT_KEYS:
        node = config
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        view[".".join(keys)] = node
    return view


def lint_neutral_pyproject(repo, path, base, head):
    """True only if base, head and worktree all parse and agree on every lint key."""
    if Path(path).name != "pyproject.toml":
        return False
    try:
        views = [
            pyproject_lint_view(git(repo, "show", f"{ref}:{path}"))
            for ref in (base, head)
        ]
        views.append(pyproject_lint_view((repo / path).read_bytes()))
    except Exception:
        return False
    return views[0] == views[1] == views[2]


def configuration_inputs(repo, ref=None):
    """Collect config dependencies without judging tool configuration validity.

    Snapshot blobs preserve old extend and symlink edges after deletion or
    rename. Ambiguous/malformed graphs trigger full checks; the pinned tools
    decide whether the configuration actually applies and is valid.
    """
    tree = {}
    if ref:
        for entry in git(repo, "ls-tree", "-rz", "--full-tree", ref).split(b"\0"):
            if entry:
                metadata, path = entry.split(b"\t", 1)
                mode, _, object_id = metadata.decode().split()
                tree[os.fsdecode(path)] = (mode, object_id)
        names = tree
    else:
        names = [
            os.fsdecode(path)
            for path in git(repo, "ls-files", "-z").split(b"\0")
            if path
        ]
    inputs, visited, active = set(), set(), set()
    uncertain = False

    def relative(path):
        normalized = Path(os.path.abspath(path))
        try:
            return normalized.relative_to(repo).as_posix()
        except ValueError:
            return str(normalized)

    def read(path):
        nonlocal uncertain
        links = set()
        while True:
            inputs.add(path)
            if path in links or len(links) >= 100:
                uncertain = True
                return path, None
            links.add(path)
            parts = Path(path).parts
            redirected = False
            for index in range(1, len(parts) + 1):
                prefix = Path(*parts[:index]).as_posix()
                if ref:
                    linked = tree.get(prefix, (None,))[0] == "120000"
                    target = (
                        os.fsdecode(git(repo, "cat-file", "blob", tree[prefix][1]))
                        if linked
                        else None
                    )
                else:
                    linked = (repo / prefix).is_symlink()
                    target = os.readlink(repo / prefix) if linked else None
                if linked:
                    inputs.add(prefix)
                    path = relative(
                        repo / Path(prefix).parent / target / Path(*parts[index:])
                    )
                    redirected = True
                    break
            if not redirected:
                break
        try:
            if ref:
                return path, git(repo, "cat-file", "blob", tree[path][1])
            return path, (repo / path).read_bytes()
        except (KeyError, OSError):
            uncertain = True
            return path, None

    def visit(path, ruff=False):
        nonlocal uncertain
        key = (path, ruff)
        if key in active or len(active) >= 100:
            uncertain = True
            return
        if key in visited:
            return
        visited.add(key)
        active.add(key)
        resolved, data = read(path)
        if ruff and data is not None:
            try:
                config = tomllib.loads(data.decode("utf-8"))
                if Path(path).name == "pyproject.toml":
                    config = config.get("tool", {}).get("ruff", {})
                extension = config.get("extend")
                if extension is not None:
                    if not isinstance(extension, str):
                        uncertain = True
                    else:
                        extension = os.path.expanduser(os.path.expandvars(extension))
                        # Include both possible locations for linked configs;
                        # this is conservative and never changes tool semantics.
                        for parent in {Path(path).parent, Path(resolved).parent}:
                            visit(relative(repo / parent / extension), True)
            except (UnicodeError, tomllib.TOMLDecodeError, AttributeError):
                uncertain = True
        active.remove(key)

    for path in names:
        if Path(path).name in CONFIG_NAMES:
            visit(
                path, Path(path).name in {"ruff.toml", ".ruff.toml", "pyproject.toml"}
            )
    return inputs, uncertain


def read_config(repo):
    config = yaml.safe_load((repo / ".pre-commit-config.yaml").read_text())
    hooks = {}
    for source in config["repos"]:
        for hook in source["hooks"]:
            if hook["id"] in {"ruff", "ruff-format", "isort"}:
                hooks[hook["id"]] = {**hook, "rev": source["rev"]}
    if set(hooks) != {"ruff", "ruff-format", "isort"}:
        raise ValueError("CONFIG_DRIFT: missing required lint hooks")
    if (
        hooks["ruff"]["rev"] != "v" + VERSIONS["ruff"]
        or hooks["ruff-format"]["rev"] != "v" + VERSIONS["ruff"]
        or hooks["isort"]["rev"] != VERSIONS["isort"]
    ):
        raise ValueError("CONFIG_DRIFT: pinned hook versions differ")
    for name in ("isort", "ruff-format"):
        if hooks[name].get("args", []) != []:
            raise ValueError("CONFIG_DRIFT: unreviewed " + name + " options")
    if "--select=F401,F821,UP037" not in hooks["ruff"].get("args", []):
        raise ValueError("CONFIG_DRIFT: required Ruff rules differ")
    # Preserve all configured checks but strip source-changing fix flags.
    if any(
        argument not in {"--select=F401,F821,UP037", "--fix"}
        for argument in hooks["ruff"].get("args", [])
    ):
        raise ValueError("CONFIG_DRIFT: unreviewed Ruff options")
    return config, hooks


def eligible(path, config, hook):
    if not path.endswith((".py", ".pyi")):
        return False
    for expression in (config.get("exclude"), hook.get("exclude")):
        if expression and re.search(expression, path):
            return False
    return not hook.get("files") or bool(re.search(hook["files"], path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = {
        "status": "error",
        "eligible_files": [],
        "file_count": 0,
        "tools": {},
        "versions": {},
        "mode": "changed",
    }
    try:
        change = changed_paths(repo, args.base, args.head)
        result.update(change)
        config, hooks = read_config(repo)
        for package in VERSIONS:
            version = subprocess.run(
                [sys.executable, "-m", package, "--version"],
                cwd=repo,
                text=True,
                capture_output=True,
                timeout=15,
            )
            if version.returncode or not re.search(
                r"(?<!\d)" + re.escape(VERSIONS[package]) + r"(?!\d)", version.stdout
            ):
                raise ValueError(
                    "CONFIG_DRIFT: installed " + package + " differs from hook version"
                )
            result["versions"][package] = VERSIONS[package]
        inputs, uncertain = set(), False
        for ref in (change["merge_base_sha"], change["head_sha"], None):
            dependencies, ambiguous = configuration_inputs(repo, ref)
            inputs.update(dependencies)
            uncertain |= ambiguous
        result["configuration_inputs"] = sorted(inputs)
        neutral = {
            path
            for path in change["changed_paths"]
            if lint_neutral_pyproject(
                repo, path, change["merge_base_sha"], change["head_sha"]
            )
        }
        result["lint_neutral_configs"] = sorted(neutral)
        relevant = set(change["changed_paths"]) - neutral
        full = bool(
            relevant & inputs
            or any(Path(path).name in CONFIG_NAMES for path in relevant)
            or (uncertain and change["changed_paths"])
        )
        result["mode"] = "full" if full else "changed"
        candidates = change["current_paths"]
        if full:
            candidates = [
                path.decode()
                for path in git(repo, "ls-files", "-z").split(b"\0")
                if path
            ]
        candidates = sorted(path for path in candidates if (repo / path).is_file())
        checked = set()
        for name, hook in hooks.items():
            files = [path for path in candidates if eligible(path, config, hook)]
            checked.update(files)
            tool = {
                "files": files,
                "invoked": bool(files),
                "exit_code": 0,
                "output": "",
            }
            if files:
                if name == "ruff":
                    command = [
                        sys.executable,
                        "-m",
                        "ruff",
                        "check",
                        "--select=F401,F821,UP037",
                        "--",
                        *files,
                    ]
                elif name == "ruff-format":
                    command = [
                        sys.executable,
                        "-m",
                        "ruff",
                        "format",
                        "--check",
                        "--",
                        *files,
                    ]
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "isort",
                        "--check-only",
                        "--diff",
                        "--resolve-all-configs",
                        "--config-root",
                        str(repo),
                        "--",
                        *files,
                    ]
                before = {path: sha256(repo / path) for path in files}
                process = subprocess.run(
                    command, cwd=repo, capture_output=True, text=True, timeout=300
                )
                tool.update(
                    exit_code=process.returncode,
                    output=process.stdout + process.stderr,
                    command=command,
                )
                if any(
                    sha256(repo / path) != digest for path, digest in before.items()
                ):
                    raise ValueError(
                        "Lint tool modified source during check-only execution"
                    )
            result["tools"][name] = tool
        result["eligible_files"] = sorted(checked)
        result["file_count"] = len(checked)
        result["status"] = (
            "success"
            if all(tool["exit_code"] == 0 for tool in result["tools"].values())
            else "error"
        )
        result["config_sha256"] = {
            path: sha256(repo / path)
            for path in (".pre-commit-config.yaml", ".isort.cfg")
        }
    except (
        OSError,
        ValueError,
        KeyError,
        yaml.YAMLError,
        subprocess.TimeoutExpired,
    ) as error:
        result["error"] = str(error)
        if "CONFIG_DRIFT" in str(error):
            result["reason_code"] = "CONFIG_DRIFT"
    write_json(args.output, result)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute the Jenkins registry verification payload against a fake registry."""

import hashlib
import io
import json
import re
import types
from pathlib import Path

import pytest

JENKINSFILE = Path(__file__).with_name("Jenkinsfile")


def payload():
    source = JENKINSFILE.read_text().split("void waitForRegistryImageDigest", 1)[1]
    script = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    return re.sub(r"\\([\\'])", r"\1", script)


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("target", ["release", "room", "other"])
def test_cpu_callback_does_not_swallow_failed_image_storage_gate(monkeypatch, target):
    import ast
    import os

    text = JENKINSFILE.read_text().split("void callbackPlatform", 1)[1]
    script = text.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    script = re.sub(r"\\([\\'])", r"\1", script)
    tree = ast.parse(script)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "status == 'success'"
    )
    monkeypatch.setenv("BUILD_TARGET", target)
    monkeypatch.setenv("CALLBACK_THAILAND_ACR_ONLY", "false")

    def invalid(_):
        raise ValueError("Hidden SDK layer")

    scope = dict(
        os=os,
        status="success",
        payload={"commitSha": "a" * 40},
        COMMIT_SHA_PATTERN=re.compile(r"[a-f0-9]{40}"),
        DIGEST_PATTERN=re.compile(r"sha256:[a-f0-9]{64}"),
        resolve_ecr_image=lambda _: {"registryDigest": "sha256:" + "b" * 64},
        inspect_ecr_image_config=invalid,
    )
    code = compile(ast.Module(body=[node], type_ignores=[]), "actual-callback", "exec")
    if target in {"release", "room"}:
        with pytest.raises(SystemExit):
            exec(code, scope)
    else:
        exec(code, scope)  # Existing non-CPU callback behavior is preserved.


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "source",
        "platform",
        "manifest_bytes",
        "config_bytes",
        "header",
        "scope",
        "realm",
        "sdk_layers",
        "missing_history",
    ],
)
@pytest.mark.parametrize("target", ["release", "room"])
def test_acr_artifact_verification(monkeypatch, tmp_path, failure, target):
    import base64
    import urllib.error
    import urllib.request

    registry = "lwdp-codex-th-registry.ap-southeast-7.cr.aliyuncs.com"
    revision = "a" * 40
    monkeypatch.setenv("BUILD_TARGET", target)
    history = [{"created_by": "COPY /runtime-source/sglang /app/python/sglang"}]
    if target == "room":
        history.append({"created_by": "COPY /opt/worldkit /opt/worldkit"})
    if failure == "sdk_layers":
        history.insert(0, {"created_by": "COPY old /opt/worldkit"})
        history.insert(1, {"created_by": "RUN rm -rf /opt/worldkit"})
    config = json.dumps(
        {
            "os": "linux",
            "architecture": "arm64" if failure == "platform" else "amd64",
            "history": [] if failure == "missing_history" else history,
            "config": {
                "Labels": {
                    "org.opencontainers.image.revision": "b" * 40
                    if failure == "source"
                    else revision
                },
                "Entrypoint": ["python"],
                "Cmd": ["run_service.py"],
            },
        }
    ).encode()
    manifest = json.dumps(
        {"schemaVersion": 2, "config": {"digest": digest(config)}}
    ).encode()
    expected = digest(manifest)
    for key, value in {
        "REPLICATION_TARGET_REGISTRY": registry,
        "REPLICATION_TARGET_REPOSITORY": "lwdp/whitebox-runtime",
        "REPLICATION_TARGET_TAG": "test",
        "REPLICATION_EXPECTED_DIGEST": expected,
        "REPLICATION_TIMEOUT_SECONDS": "1",
        "REPLICATION_SOURCE_REVISION": revision,
    }.items():
        monkeypatch.setenv(key, value)
    auth_file = tmp_path / "config.json"
    auth_file.write_text(
        json.dumps(
            {
                "auths": {
                    registry: {"auth": base64.b64encode(b"test:temporary").decode()}
                }
            }
        )
    )
    output = tmp_path / "verified-acr-image.json"
    paths = {
        "/build-secrets/docker/config.json": auth_file,
        "/build-secrets/verified-acr-image.json": output,
    }
    fake_pathlib = types.SimpleNamespace(Path=lambda name: paths[name])
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        return fake_pathlib if name == "pathlib" else real_import(name, *args, **kwargs)

    class Response(io.BytesIO):
        def __init__(self, raw=b"", headers=None):
            super().__init__(raw)
            self.headers = headers or {}

    calls = []

    def open_request(request, timeout):
        calls.append((request.full_url, request.get_method()))
        auth = request.get_header("Authorization")
        if "/manifests/" in request.full_url and auth is None:
            realm = (
                "https://attacker.example/token"
                if failure == "realm"
                else "https://dockerauth.ap-southeast-7.aliyuncs.com/auth"
            )
            scope = (
                "repository:other/private:pull"
                if failure == "scope"
                else "repository:lwdp/whitebox-runtime:pull"
            )
            challenge = f'Bearer realm="{realm}",service="acr",scope="{scope}"'
            raise urllib.error.HTTPError(
                request.full_url, 401, "auth", {"WWW-Authenticate": challenge}, None
            )
        if "dockerauth." in request.full_url:
            assert auth.startswith("Basic ")
            return Response(b'{"token":"temporary-bearer"}')
        assert auth == "Bearer temporary-bearer"
        if request.get_method() == "HEAD":
            return Response(
                headers={
                    "Docker-Content-Digest": "sha256:" + "0" * 64
                    if failure == "header"
                    else expected
                }
            )
        if "/manifests/" in request.full_url:
            return Response(manifest + (b" " if failure == "manifest_bytes" else b""))
        if "/blobs/" in request.full_url:
            return Response(config + (b" " if failure == "config_bytes" else b""))
        raise AssertionError(request.full_url)

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    namespace = {
        "__builtins__": dict(vars(__import__("builtins")), __import__=fake_import)
    }
    if failure:
        with pytest.raises(SystemExit):
            exec(compile(payload(), "jenkins-registry-verifier", "exec"), namespace)
        assert not output.exists()
    else:
        exec(compile(payload(), "jenkins-registry-verifier", "exec"), namespace)
        result = json.loads(output.read_text())
        assert result == {
            "digest": expected,
            "revision": revision,
            "imageConfig": {"entrypoint": ["python"], "cmd": ["run_service.py"]},
        }
        assert output.stat().st_mode & 0o777 == 0o600
        assert len(calls) == 5


def cache_payload():
    source = JENKINSFILE.read_text().split("void discoverWhiteboxCacheCandidates", 1)[1]
    script = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    return re.sub(r"\\([\\'])", r"\1", script)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "platform",
        "source",
        "branch",
        "target",
        "recipe",
        "missing_labels",
        "manifest_bytes",
        "config_bytes",
        "scope",
        "registry",
        "digest",
        "unavailable",
    ],
)
def test_ecr_cache_uses_verified_digest_not_moved_tag(monkeypatch, capsys, failure):
    import subprocess
    import urllib.request

    registry = "829115578968.dkr.ecr.us-east-2.amazonaws.com"
    repository = "leap-world/minwm-realtime"
    labels = {
        "org.opencontainers.image.revision": "b" * 40,
        "org.worldkit.whitebox.cache.branch": "main",
        "org.worldkit.whitebox.cache.target": "release",
        "org.worldkit.whitebox.cache.recipe-sha256": "c" * 64,
    }
    field = {
        "source": "org.opencontainers.image.revision",
        "branch": "org.worldkit.whitebox.cache.branch",
        "target": "org.worldkit.whitebox.cache.target",
        "recipe": "org.worldkit.whitebox.cache.recipe-sha256",
    }.get(failure)
    if field:
        labels[field] = "wrong"
    if failure == "missing_labels":
        labels = {}
    config = json.dumps(
        {
            "os": "linux",
            "architecture": "arm64" if failure == "platform" else "amd64",
            "config": {"Labels": labels},
        }
    ).encode()
    manifest = json.dumps(
        {"schemaVersion": 2, "config": {"digest": digest(config)}}
    ).encode()
    image_digest = digest(manifest)
    monkeypatch.setenv("WHITEBOX_CACHE_RECIPE", "c" * 64)
    monkeypatch.setenv("WHITEBOX_CACHE_BRANCH", "main")
    monkeypatch.setenv("BUILD_TARGET", "release")
    monkeypatch.setenv("ECR_IMAGE_REPOSITORY", registry + "/" + repository)
    monkeypatch.setenv("ECR_REPOSITORY", repository)
    monkeypatch.setenv("ECR_REGISTRY", registry)
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_ACCOUNT_ID", "829115578968")
    calls = []

    def aws(args, **kwargs):
        calls.append(args)
        if "describe-images" in args:
            return json.dumps(
                {
                    "imageDetails": [
                        {"imageDigest": image_digest, "imageTags": ["moved-tag"]}
                    ]
                }
            )
        if "batch-get-image" in args:
            assert "imageDigest=" + image_digest in args
            if failure == "unavailable":
                return '{"images":[]}'
            return json.dumps(
                {
                    "images": [
                        {
                            "registryId": "000000000000"
                            if failure == "registry"
                            else "829115578968",
                            "repositoryName": "other/repository"
                            if failure == "scope"
                            else repository,
                            "imageId": {
                                "imageDigest": "sha256:" + "0" * 64
                                if failure == "digest"
                                else image_digest
                            },
                            "imageManifest": manifest.decode()
                            + (" " if failure == "manifest_bytes" else ""),
                        }
                    ]
                }
            )
        if "get-download-url-for-layer" in args:
            assert digest(config) in args
            return '{"downloadUrl":"https://existing-ecr-signed-blob.example/config"}'
        raise AssertionError(args)

    monkeypatch.setattr(subprocess, "check_output", aws)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            config + (b" " if failure == "config_bytes" else b"")
        ),
    )
    exec(compile(cache_payload(), "jenkins-ecr-cache-verifier", "exec"), {})
    records = json.loads(capsys.readouterr().out)
    if failure:
        assert records == []
    else:
        assert records == [
            {
                "repository": registry + "/" + repository,
                "digest": image_digest,
                "sourceRevision": "b" * 40,
                "branch": "main",
                "target": "release",
                "recipeSha256": "c" * 64,
                "platform": "linux/amd64",
                "integrityVerified": True,
            }
        ]
    assert not any("imageTag=moved-tag" in call for call in calls)


@pytest.mark.parametrize("eligible", [False, True])
def test_ecr_reader_bounds_artifact_inspection_and_accepted_digests(
    monkeypatch, capsys, eligible
):
    import subprocess
    import urllib.request

    registry = "829115578968.dkr.ecr.us-east-2.amazonaws.com"
    repository = "leap-world/minwm-realtime"
    for name, value in {
        "ECR_REGISTRY": registry,
        "ECR_REPOSITORY": repository,
        "ECR_IMAGE_REPOSITORY": registry + "/" + repository,
        "AWS_ACCOUNT_ID": "829115578968",
        "AWS_REGION": "us-east-2",
        "WHITEBOX_CACHE_BRANCH": "main",
        "WHITEBOX_CACHE_RECIPE": "c" * 64,
        "BUILD_TARGET": "release",
    }.items():
        monkeypatch.setenv(name, value)
    manifests, configs = {}, {}
    for number in range(1, 22):
        config = json.dumps(
            {
                "os": "linux",
                "architecture": "amd64",
                "config": {
                    "Labels": {
                        "org.opencontainers.image.revision": f"{number:040x}",
                        "org.worldkit.whitebox.cache.branch": "main"
                        if eligible
                        else "wrong",
                        "org.worldkit.whitebox.cache.target": "release",
                        "org.worldkit.whitebox.cache.recipe-sha256": "c" * 64,
                    }
                },
            }
        ).encode()
        manifest = json.dumps({"config": {"digest": digest(config)}}).encode()
        manifests[digest(manifest)] = manifest
        configs[digest(config)] = config
    inspected = []

    def aws(args, **kwargs):
        if "describe-images" in args:
            return json.dumps([{"imageDigest": key} for key in manifests])
        if "batch-get-image" in args:
            key = args[args.index("--image-ids") + 1].split("=", 1)[1]
            inspected.append(key)
            return json.dumps(
                {
                    "images": [
                        {
                            "registryId": "829115578968",
                            "repositoryName": repository,
                            "imageId": {"imageDigest": key},
                            "imageManifest": manifests[key].decode(),
                        }
                    ]
                }
            )
        key = args[args.index("--layer-digest") + 1]
        return json.dumps({"downloadUrl": key})

    monkeypatch.setattr(subprocess, "check_output", aws)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, **kwargs: io.BytesIO(configs[url])
    )
    exec(compile(cache_payload(), "jenkins-ecr-cache-verifier", "exec"), {})
    records = json.loads(capsys.readouterr().out)
    assert len(records) == (3 if eligible else 0)
    assert len(inspected) == (3 if eligible else 20)
    assert len({record["digest"] for record in records}) == len(records)


def test_ecr_reader_rejects_repository_outside_existing_scope(monkeypatch, capsys):
    import subprocess

    for key, value in {
        "ECR_REGISTRY": "829115578968.dkr.ecr.us-east-2.amazonaws.com",
        "ECR_REPOSITORY": "attacker/repository",
        "ECR_IMAGE_REPOSITORY": "attacker.example/repository",
        "AWS_ACCOUNT_ID": "829115578968",
        "AWS_REGION": "us-east-2",
    }.items():
        monkeypatch.setenv(key, value)

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Out-of-scope cache lookup must not contact a registry")

    monkeypatch.setattr(subprocess, "check_output", forbidden)
    exec(compile(cache_payload(), "jenkins-ecr-cache-verifier", "exec"), {})
    assert json.loads(capsys.readouterr().out) == []
    assert calls == []

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import launch
import select_gpu
import source


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def assets(self, role="dit"):
        model = self.root / "model"
        model.mkdir()
        (model / "zing_config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"fixture weights")
        pipeline = self.root / "pipeline.json"
        pipeline.write_text('{"zing_adaln_mode":"resident"}')
        decoder = self.root / "decoder.json"
        decoder.write_text('{"type":"lynn_streaming"}')
        files = {
            str(p): source.digest(p)
            for p in [
                model / "zing_config.json",
                model / "model.safetensors",
                pipeline,
                decoder,
            ]
        }
        receipt = self.root / "assets.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "model_id": launch.MODEL_ID,
                    "component": "generator",
                    "ema": False,
                    "model": str(model),
                    "files": files,
                    "adaln": {"mode": "resident"},
                }
            )
        )
        c = {
            "role": role,
            "schema": 1,
            "model_id": launch.MODEL_ID,
            "model": str(model),
            "asset_receipt": str(receipt),
            "asset_receipt_sha256": source.digest(receipt),
        }
        if role == "dit":
            c["pipeline"] = str(pipeline)
        elif role == "decoder":
            c["decoder_config"] = str(decoder)
        return c

    def test_archive_excludes_dirty_and_rejects_partial_commit(self):
        repo = self.root / "repo"
        repo.mkdir()
        for cmd in [
            ["init", "-q"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "test"],
        ]:
            subprocess.run(["git", "-C", str(repo), *cmd], check=True)
        (repo / "file").write_text("committed")
        subprocess.run(["git", "-C", str(repo), "add", "file"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        (repo / "file").write_text("dirty")
        (repo / "secret").write_text("untracked")
        m = source.prepare(
            repo, commit, self.root / "release", "oss://example-bucket/source"
        )
        source.unpack(
            self.root / "release/source.tar", m["sha256"], self.root / "unpacked"
        )
        self.assertEqual((self.root / "unpacked/file").read_text(), "committed")
        self.assertFalse((self.root / "unpacked/secret").exists())
        with self.assertRaises(ValueError):
            source.prepare(
                repo, commit[:12], self.root / "bad", "oss://example-bucket/source"
            )
        with self.assertRaises(ValueError):
            source.unpack(
                self.root / "release/source.tar", "0" * 64, self.root / "corrupt"
            )
        self.assertFalse((self.root / "corrupt").exists())

    def test_archive_rejects_escaping_paths_and_links(self):
        for i, (name, link) in enumerate(
            [("../escape", None), ("/absolute", None), ("link", "../../escape")]
        ):
            archive = self.root / (f"bad{i}.tar")
            with tarfile.open(archive, "w") as tar:
                info = tarfile.TarInfo(name)
                if link:
                    info.type, info.linkname = tarfile.SYMTYPE, link
                    tar.addfile(info)
                else:
                    info.size = 1
                    tar.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                source.unpack(archive, source.digest(archive), self.root / (f"bad{i}"))

    def test_asset_hash_and_model_mismatch_fail_closed(self):
        c = self.assets()
        launch.verify_assets(c)
        Path(c["pipeline"]).write_text('{"zing_adaln_mode":"weights_free"}')
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            launch.verify_assets(c)
        c["model_id"] = "old-model"
        path = self.root / "role.json"
        path.write_text(json.dumps(c))
        with self.assertRaises(ValueError):
            launch.load_config(path)

    def test_configuration_rejects_env_injection(self):
        c = self.assets("encoder")
        path = self.root / "role.json"
        for env in [
            {"PYTHONPATH": "/image"},
            {"AWS_SECRET_ACCESS_KEY": "secret"},
            {"CUDA_VISIBLE_DEVICES": "all"},
            {"WORLD_MODEL_METRIC_SERVICE": "wrong-role"},
            {"PROMETHEUS_MULTIPROC_DIR": "/shared/stale"},
        ]:
            c["environment"] = env
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                launch.load_config(path)

    def test_adaln_manifest_and_schedule_are_frozen(self):
        c = self.assets()
        table = self.root / "table"
        table.mkdir()
        metadata = {"schedule": {"solver": "few_step"}, "arithmetic": {"cuda": "13.0"}}
        (table / "manifest.json").write_text(json.dumps(metadata))
        (table / "table.safetensors").write_bytes(b"fixture table")
        Path(c["pipeline"]).write_text(
            json.dumps(
                {"zing_adaln_mode": "weights_free", "zing_adaln_table_path": str(table)}
            )
        )
        receipt_path = Path(c["asset_receipt"])
        a = json.loads(receipt_path.read_text())
        a["adaln"] = {
            "mode": "weights_free",
            "table": str(table),
            "schema": 1,
            "model_id": launch.MODEL_ID,
            "schedule": metadata["schedule"],
            "cuda_provenance": metadata["arithmetic"],
        }
        for path in [
            Path(c["pipeline"]),
            table / "manifest.json",
            table / "table.safetensors",
        ]:
            a["files"][str(path)] = source.digest(path)
        receipt_path.write_text(json.dumps(a))
        c["asset_receipt_sha256"] = source.digest(receipt_path)
        launch.verify_assets(c)
        a["adaln"]["schedule"] = {"solver": "different"}
        receipt_path.write_text(json.dumps(a))
        c["asset_receipt_sha256"] = source.digest(receipt_path)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            launch.verify_assets(c)

    def test_fetch_checks_download_before_extracting(self):
        archive = self.root / "fixture.tar"
        with tarfile.open(archive, "w") as tar:
            member = tarfile.TarInfo("file")
            member.size = 7
            tar.addfile(member, io.BytesIO(b"payload"))
        m = {
            "schema": 1,
            "tools": {},
            "commit": "a" * 40,
            "sha256": source.digest(archive),
            "uri": "oss://example-bucket/"
            + "a" * 40
            + "/"
            + source.digest(archive)
            + "/source.tar",
        }

        def fake_download(argv, **kwargs):
            import shutil

            self.assertEqual(argv[:3], ["ossutil", "cp", m["uri"]])
            shutil.copyfile(archive, argv[3])

        with patch.object(source.subprocess, "run", side_effect=fake_download):
            source.fetch(m, self.root / "release/source")
        self.assertEqual((self.root / "release/source/file").read_bytes(), b"payload")
        self.assertEqual(
            json.loads((self.root / "release/verified-source.json").read_text())[
                "commit"
            ],
            m["commit"],
        )
        m["sha256"] = "0" * 64
        with (
            patch.object(source.subprocess, "run", side_effect=fake_download),
            self.assertRaises(ValueError),
        ):
            source.fetch(m, self.root / "invalid/source")
        self.assertFalse((self.root / "invalid/source").exists())

    def test_role_commands_use_native_cli_and_clean_env(self):
        for role in launch.MODULES:
            with self.subTest(role=role):
                if (self.root / "model").exists():
                    import shutil

                    shutil.rmtree(self.root / "model")
                c = self.assets(role)
                with patch.dict(
                    os.environ,
                    {
                        "PYTHONPATH": "/image/source",
                        "TOKEN": "private",
                        "WORLD_MODEL_METRIC_SERVICE": "wrong-role",
                        "PROMETHEUS_MULTIPROC_DIR": "/shared/stale",
                    },
                    clear=True,
                ):
                    argv, env = launch.command(
                        c, self.root, "10.0.0.2", "worker-fixture"
                    )
                with self.assertRaisesRegex(ValueError, "Unknown configuration field"):
                    launch.validate_config({**c, "role_launcher": "native"})
                with self.assertRaisesRegex(ValueError, "Unknown configuration field"):
                    launch.validate_config({**c, "role_launcher": "profile"})
                self.assertIn(
                    "sglang.multimodal_gen.tools.zing_h3_profile_launcher", argv
                )
                self.assertEqual(argv[argv.index("--mode") + 1], "independent")
                self.assertEqual(argv[argv.index("--component") + 1], role)
                self.assertEqual(env["PYTHONPATH"], str(self.root / "python"))
                self.assertNotIn("TOKEN", env)
                self.assertEqual(env["WORLD_MODEL_METRIC_SERVICE"], role)
                self.assertEqual(env["WORLD_MODEL_METRIC_LANE"], "default")
                self.assertEqual(
                    env["PROMETHEUS_MULTIPROC_DIR"],
                    str(self.root.parent / "prometheus" / "worker-fixture"),
                )
                self.assertEqual(argv[argv.index("--service-count") + 1], "1")
                if role == "decoder":
                    self.assertNotIn("--pipeline-config-path", argv)
                if role == "dit":
                    self.assertEqual(argv[argv.index("--sp-degree") + 1], "1")
                # Check emitted options against actual target source without GPU imports.
                module_path = (
                    Path(__file__).resolve().parents[2]
                    / "python"
                    / (launch.MODULES[role].replace(".", "/") + ".py")
                )
                import ast

                tree = ast.parse(module_path.read_text())
                options = {
                    a.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)
                }
                index = argv.index("--")
                for arg in argv[index + 1 :: 2]:
                    self.assertIn(arg, options)

    def test_pending_pod_claim_is_not_selected(self):
        gpus = [{"uuid": "GPU-a", "memory": 0}, {"uuid": "GPU-b", "memory": 0}]
        pods = {
            "items": [
                {
                    "spec": {
                        "nodeName": "world-model-b27mys7b-0000",
                        "containers": [
                            {
                                "env": [
                                    {"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-a"}
                                ]
                            }
                        ],
                    },
                    "status": {"phase": "Pending"},
                }
            ]
        }
        self.assertEqual(select_gpu.select(gpus, [], pods), "GPU-b")
        with self.assertRaises(ValueError):
            select_gpu.select(gpus, ["GPU-b"], pods)

    def test_render_never_submits_and_shell_is_valid(self):
        c = self.assets("encoder")
        site = {
            "schema": 1,
            "image": "example.invalid/model@sha256:" + "a" * 64,
            "namespace": "sglang",
            "volumes": [],
            "volume_mounts": [],
        }
        manifest = {
            "schema": 1,
            "commit": "b" * 40,
            "sha256": "c" * 64,
            "uri": "oss://example-bucket/" + "b" * 40 + "/" + "c" * 64 + "/source.tar",
            "tools": {
                n: source.digest(Path(__file__).with_name(n))
                for n in ("launch.py", "source.py", "select_gpu.py")
            },
        }
        with patch(
            "subprocess.run", side_effect=AssertionError("render must be offline")
        ):
            launch.render(c, manifest, site, self.root / "render")
        subprocess.run(["sh", "-n", str(self.root / "render/h100-node.sh")], check=True)
        request = json.loads((self.root / "render/eflo.json").read_text())
        self.assertEqual(request["NodeIdList"], ["e01-cn-2zd4xi8m505"])
        self.assertIn("flock -n", request["CommandContent"])
        c["role"], c["pipeline"] = "dit", str(self.root / "pipeline.json")
        site = {
            "schema": 1,
            "image": site["image"],
            "job_spec": {
                "Type": "Worker",
                "QuotaId": "quota3l8bnv9a35n",
                "RestartPolicy": "Never",
                "PodCount": 1,
                "ResourceConfig": {"GPU": "1"},
            },
            "data_sources": [],
        }
        launch.render(c, manifest, site, self.root / "pai")
        job = json.loads((self.root / "pai/pai.json").read_text())
        self.assertEqual(job["WorkspaceId"], "1453237")
        self.assertEqual(job["JobSpecs"][0]["ResourceConfig"]["GPU"], "1")
        site["job_spec"]["ResourceConfig"]["GPU"] = "2"
        with self.assertRaises(ValueError):
            launch.render(c, manifest, site, self.root / "wrong-gpu")

    def test_worker_startup_failure_is_nonzero_and_reaped(self):
        c = self.assets()
        source.write_json(
            self.root / "verified-source.json",
            {"commit": "a" * 40, "archive_sha256": "b" * 64},
        )
        child = unittest.mock.Mock()
        child.pid = 1234
        child.poll.return_value = 7
        with (
            patch.object(launch, "reachable_ip", return_value="10.0.0.2"),
            patch.object(launch, "probe_source", return_value={}),
            patch.object(launch.subprocess, "check_output", return_value="GPU-fixture"),
            patch.object(launch.subprocess, "Popen", return_value=child),
            patch.object(launch.os, "killpg"),
            self.assertRaisesRegex(RuntimeError, "before becoming healthy"),
        ):
            launch.run(c, self.root / "source", "a" * 40, "b" * 64)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        child.wait.assert_called()


if __name__ == "__main__":
    unittest.main()


@pytest.fixture
def live_assets():
    fixture = LaunchTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def runtime_manifest():
    return dict(
        schema=1,
        commit="a" * 40,
        sha256="b" * 64,
        uri="oss://leap-world-shanghai/releases/"
        + "a" * 40
        + "/"
        + "b" * 64
        + "/source.tar",
        tools={
            n: source.digest(Path(__file__).with_name(n))
            for n in ("launch.py", "source.py", "select_gpu.py")
        },
    )


@pytest.mark.parametrize("location", ["mount", "cache"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_fetch_prefers_local_archive_and_still_checks_hash(
    tmp_path, monkeypatch, location, corrupt
):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("file")
        member.size = 7
        tar.addfile(member, io.BytesIO(b"payload"))
    manifest = runtime_manifest()
    checksum = source.digest(archive)
    manifest.update(
        sha256=checksum,
        uri="oss://leap-world-shanghai/" + "a" * 40 + "/" + checksum + "/source.tar",
    )
    mount, cache = tmp_path / "mount", tmp_path / "cache"
    monkeypatch.setattr(source, "OSS_MOUNT", mount)
    monkeypatch.setattr(source, "SOURCE_CACHE", cache)
    candidate = (
        mount / manifest["uri"].split("/", 3)[3]
        if location == "mount"
        else cache / checksum / "source.tar"
    )
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"corrupt" if corrupt else archive.read_bytes())
    with patch.object(
        source.subprocess,
        "run",
        side_effect=AssertionError("No ossutil for local archives"),
    ) as download:
        if corrupt:
            with pytest.raises(ValueError, match="SHA256 mismatch"):
                source.fetch(manifest, tmp_path / "release/source")
        else:
            source.fetch(manifest, tmp_path / "release/source")
            assert (tmp_path / "release/source/file").read_bytes() == b"payload"
            assert (
                json.loads((tmp_path / "release/verified-source.json").read_text())[
                    "archive_sha256"
                ]
                == checksum
            )
        download.assert_not_called()


@pytest.mark.parametrize(
    "routes,pod_ip,expected",
    [
        (["10.0.0.122"], "22.69.1.2", "10.0.0.122"),
        (["22.69.1.2", "10.0.0.122"], "22.69.1.2", "10.0.0.122"),
        ([None, None], "22.69.1.2", "22.69.1.2"),
        ([None, None], "127.0.0.1", None),
        (["22.69.1.2", "22.69.1.2"], "22.69.1.2", None),
        (["22.69.1.2", "22.69.1.2"], "10.0.0.123", "10.0.0.123"),
    ],
)
def test_reachable_ip_prefers_vpc_routes(monkeypatch, routes, pod_ip, expected):
    from unittest.mock import MagicMock, Mock

    sockets = []
    for address in routes:
        sock = MagicMock()
        connection = sock.__enter__.return_value
        if address is None:
            connection.connect.side_effect = OSError("No route")
        else:
            connection.getsockname.return_value = (address, 1234)
        sockets.append(sock)
    monkeypatch.setenv("POD_IP", pod_ip)
    monkeypatch.setattr(launch.socket, "socket", Mock(side_effect=sockets))
    if expected is None:
        with pytest.raises(ValueError, match="usable advertise IP"):
            launch.reachable_ip()
    else:
        assert launch.reachable_ip() == expected
    sockets[0].__enter__.return_value.connect.assert_called_once_with(("10.0.1.37", 9))
    if len(sockets) == 2:
        sockets[1].__enter__.return_value.connect.assert_called_once_with(
            ("10.0.0.121", 9)
        )


def test_verify_assets_follows_directory_symlinks_and_skips_sized_weights(
    live_assets, monkeypatch
):
    fixture = live_assets
    c = fixture.assets("encoder")
    external = fixture.root / "external"
    external.mkdir()
    weights = external / "weights.safetensors"
    weights.write_bytes(b"external weights")
    alias = Path(c["model"]) / "text_encoder"
    alias.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="weight files must be covered"):
        launch.verify_assets(c)
    receipt = json.loads(Path(c["asset_receipt"]).read_text())
    receipt["files"][str(alias / weights.name)] = dict(
        bytes=weights.stat().st_size, sha256=source.digest(weights)
    )
    Path(c["asset_receipt"]).write_text(json.dumps(receipt))
    c["asset_receipt_sha256"] = source.digest(c["asset_receipt"])
    original = launch.digest

    def checked_digest(path):
        assert Path(path) != alias / weights.name, "Sized cold weights must not be read"
        return original(path)

    monkeypatch.setattr(launch, "digest", checked_digest)
    launch.verify_assets(c)
    weights.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="size mismatch"):
        launch.verify_assets(c)


def test_sized_config_entries_still_require_hash_match(live_assets):
    c = live_assets.assets("encoder")
    config = Path(c["model"]) / "zing_config.json"
    receipt = json.loads(Path(c["asset_receipt"]).read_text())
    receipt["files"][str(config)] = dict(bytes=2, sha256=source.digest(config))
    Path(c["asset_receipt"]).write_text(json.dumps(receipt))
    c["asset_receipt_sha256"] = source.digest(c["asset_receipt"])
    launch.verify_assets(c)
    config.write_text("[]")  # Same size; must not bypass the config hash.
    with pytest.raises(ValueError, match="digest mismatch"):
        launch.verify_assets(c)


@pytest.mark.parametrize(
    "entry",
    [
        {"bytes": True, "sha256": "a" * 64},
        {"bytes": 15, "sha256": "invalid"},
        {"bytes": -1, "sha256": "a" * 64},
    ],
)
def test_sized_asset_metadata_is_validated(live_assets, entry):
    c = live_assets.assets("encoder")
    receipt = json.loads(Path(c["asset_receipt"]).read_text())
    receipt["files"][str(Path(c["model"]) / "model.safetensors")] = entry
    Path(c["asset_receipt"]).write_text(json.dumps(receipt))
    c["asset_receipt_sha256"] = source.digest(c["asset_receipt"])
    with pytest.raises(ValueError):
        launch.verify_assets(c)


def test_stage_assets_copies_hot_files_and_links_peripherals(tmp_path):
    src, dst = tmp_path / "mounted", tmp_path / "local"
    contents = {
        "model/transformer/part.safetensors": b"hot weights",
        "model/zing_config.json": b"{}",
        "model/model_index.json": b"{}",
        "model/zing_realtime_profile.json": b'{"source_attention_window":{"window_chunks":99,"sink_chunks":2},"other":7}',
        "model/EXPORT_COMPLETE.json": b"{}",
        "model/text_encoder/weights.bin": b"cold weights",
        "model/tokenizer/config.json": b"{}",
        "lynn/weights.bin": b"lynn",
        "adaln-table-unsplit/table.safetensors": b"table",
        "adaln-table-unsplit/manifest.json": b"{}",
    }
    for name, data in contents.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    c = dict(stage_from=str(src), model=str(dst / "model"))
    from concurrent.futures import ThreadPoolExecutor

    with patch.object(launch, "ThreadPoolExecutor", wraps=ThreadPoolExecutor) as pool:
        launch.stage_assets(c)
        pool.assert_called_once_with(max_workers=4)
    for name in (
        "model/transformer/part.safetensors",
        "model/zing_config.json",
        "model/model_index.json",
        "model/EXPORT_COMPLETE.json",
        "adaln-table-unsplit/table.safetensors",
        "adaln-table-unsplit/manifest.json",
    ):
        assert (dst / name).read_bytes() == contents[name]
        assert not (dst / name).is_symlink()
    for name in ("model/text_encoder", "model/tokenizer", "lynn"):
        assert (dst / name).is_symlink() and (dst / name).resolve() == src / name
    profile = json.loads((dst / "model/zing_realtime_profile.json").read_text())
    assert profile == dict(
        source_attention_window=dict(window_chunks=6, sink_chunks=0), other=7
    )
    assert (src / "model/zing_realtime_profile.json").read_bytes() == contents[
        "model/zing_realtime_profile.json"
    ]
    launch.stage_assets(
        c
    )  # Repeated startup preserves the source and the same symlink targets.


def test_stage_assets_runs_before_verification(live_assets):
    c = live_assets.assets("encoder")
    c["stage_from"] = "/mounted/assets"
    events = []

    def verify(config):
        events.append("verify")
        raise ValueError("Stop before GPU startup")

    with (
        patch.object(
            launch, "stage_assets", side_effect=lambda config: events.append("stage")
        ),
        patch.object(launch, "verify_assets", side_effect=verify),
    ):
        with pytest.raises(ValueError, match="Stop before GPU"):
            launch.run(c, live_assets.root / "source", "a" * 40, "b" * 64)
    assert events == ["stage", "verify"]


@pytest.mark.parametrize("role", ["encoder", "decoder"])
@pytest.mark.parametrize("policy", ["Never", "IfNotPresent", "Always"])
def test_h100_render_image_policy_and_source_cache(live_assets, policy, role):
    import shlex

    c = live_assets.assets(role)
    site = dict(
        schema=1,
        image="test@sha256:" + "c" * 64,
        namespace="test",
        volumes=[],
        volume_mounts=[],
        image_pull_policy=policy,
    )
    output = live_assets.root / "render"
    if policy == "Always":
        with pytest.raises(ValueError, match="pull policy"):
            launch.render(c, runtime_manifest(), site, output)
        return
    launch.render(c, runtime_manifest(), site, output)
    script = (output / "h100-node.sh").read_text()
    tag_command = (
        "/var/lib/rancher/k3s/data/current/bin/ctr -a /run/k3s/containerd/containerd.sock -n k8s.io images tag "
        + shlex.quote(site["image"])
        + " "
        + shlex.quote("test:cached-" + "c" * 12)
        + " >/dev/null 2>&1 || true\n"
    )
    assert script.count(tag_command) == 1
    assert (
        script.index("flock -n 9")
        < script.index(tag_command)
        < script.index("GPU_UUID=")
    )
    assert script.index(tag_command) < script.index("kubectl create")
    line = next(line for line in script.splitlines() if line.startswith("POD_JSON="))
    pod = json.loads(shlex.split(line[len("POD_JSON=") :])[0])
    assert pod["spec"]["hostNetwork"] is True
    assert pod["spec"]["dnsPolicy"] == "ClusterFirstWithHostNet"
    container = pod["spec"]["containers"][0]
    assert container["image"] == site["image"]
    assert container["imagePullPolicy"] == policy
    assert {
        "name": "sglang-source-cache",
        "mountPath": "/var/lib/sglang-orchestrate/cache",
        "readOnly": True,
    } in container["volumeMounts"]
    assert (
        pod["spec"]["volumes"][0]["hostPath"]["path"]
        == "/var/lib/sglang-orchestrate/cache"
    )


def test_pai_image_config_is_preserved(live_assets):
    c = live_assets.assets("dit")
    image_config = {
        "DockerRegistry": "registry.example",
        "Username": "platform-reference",
    }
    site = dict(
        schema=1,
        image="test@sha256:" + "c" * 64,
        job_spec=dict(
            Type="Worker",
            PodCount=1,
            QuotaId="quota3l8bnv9a35n",
            RestartPolicy="Never",
            ResourceConfig={"GPU": "1"},
            ImageConfig=image_config,
        ),
        data_sources=[],
    )
    output = live_assets.root / "render"
    launch.render(c, runtime_manifest(), site, output)
    assert (
        json.loads((output / "pai.json").read_text())["JobSpecs"][0]["ImageConfig"]
        == image_config
    )


def test_live_environment_and_dit_options_are_forwarded(live_assets):
    c = live_assets.assets("dit")
    names = [
        "NCCL_NET",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "XDG_CACHE_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "SGLANG_DIFFUSION_STAGE_LOGGING",
        "SGLANG_H3_QWEN_TEST",
        "MINIMAX_H3_PROGRESSIVE_TEST",
        "SGLANG_ZING_TEST",
    ]
    c["environment"] = {name: "1" for name in names}
    c["dit_options"] = {
        "zing_block0_cache_host_total_gb": 12,
        "zing_block0_cache_budget_root": "/cache",
        "zing_block0_cache_min_free_gb": 1,
        "zing_block0_cache_inflight_gb": 2,
        "zing_block0_cache_max_writers": 2,
        "zing_block0_cache_gc_high_watermark": 0.9,
        "zing_block0_cache_gc_low_watermark": 0.8,
        "zing_block0_cache_capacity_dir": "/capacity",
        "zing_source_joint_forward": "source_first",
        "timeout_seconds": 300,
    }
    launch.validate_config(c)
    argv, env = launch.command(c, live_assets.root, "10.0.0.122", "owned")
    assert all(env[name] == "1" for name in names)
    for key, value in c["dit_options"].items():
        assert argv[argv.index("--" + key.replace("_", "-")) + 1] == str(value)


def test_stage_assets_never_writes_through_a_shared_model_symlink(tmp_path):
    src, dst = tmp_path / "mounted", tmp_path / "local"
    (src / "model").mkdir(parents=True)
    dst.mkdir()
    (dst / "model").symlink_to(src / "model", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        launch.stage_assets(dict(stage_from=str(src), model=str(dst / "model")))
    assert not (src / "model/zing_realtime_profile.json").exists()


def test_populated_h100_payload_stays_below_eflo_command_limit(live_assets):
    import ast
    import base64
    import shlex
    import zlib

    c = live_assets.assets("encoder")
    # Populate every serialized section to exercise the attachment payload budget.
    c.update(
        worker_config=dict(
            schema=1,
            role="encoder",
            service_count=1,
            coordinator_url="http://10.0.1.37:18181",
            pool_id="adaptive-test-20261109",
            control_epoch="test-epoch",
            compatibility="model1027-test",
            network_domain="shanghai-test",
            capacity={"slots": 1},
            profiles={"480": {"slots": 1}},
            profile_limits={
                "480": dict(
                    source_sizes=[[832, 480]],
                    target_sizes=[[832, 480]],
                    max_fps=24,
                    max_references=6,
                    max_reference_bytes=1000000,
                    max_reference_pixels=1000000,
                    max_prompt_chars=10000,
                    max_preview_pixels=1000000,
                )
            },
            hardware_ids=["__ALLOCATED_GPU_UUID__"],
            signing_secret_env="ROLE_POOL_SIGNING_KEY",
            role_token_env="ROLE_POOL_REGISTRATION_TOKEN",
        ),
        environment={
            "NCCL_MIN_NCHANNELS": "2",
            "NCCL_MIN_CTAS": "2",
            "NCCL_GRAPH_REGISTER": "0",
            "SGLANG_ZING_SOURCE_PROMPT_JOINT": "1",
            "SGLANG_H3_QWEN_TEST": "1",
        },
        dit_options={
            "mem_fraction_static": 0.8,
            "zing_denoising_step_list": "1000,500",
            "zing_block0_cache_budget_root": "/cache",
        },
        secret_files={
            "ROLE_POOL_SIGNING_KEY": "/secrets/signing-key",
            "ROLE_POOL_REGISTRATION_TOKEN": "/secrets/registration-token",
        },
    )
    site = dict(
        schema=1,
        image="test@sha256:" + "c" * 64,
        namespace="test",
        volumes=[],
        volume_mounts=[],
    )
    output = live_assets.root / "render"
    launch.render(c, runtime_manifest(), site, output)
    script = (output / "h100-node.sh").read_text()
    assert len(script) < 12000
    assert len(script.encode()) < 12000
    subprocess.run(["sh", "-n", str(output / "h100-node.sh")], check=True)
    pod_line = next(
        line for line in script.splitlines() if line.startswith("POD_JSON=")
    )
    pod_text = shlex.split(pod_line[len("POD_JSON=") :])[0]
    pod = json.loads(pod_text)
    assert pod_text == json.dumps(pod, separators=(",", ":"))
    command = pod["spec"]["containers"][0]["command"]
    config = json.loads(command[command.index("--config") + 1])
    for key in ("worker_config", "environment", "dit_options", "secret_files"):
        assert config[key] == c[key]
    selector_line = next(
        line for line in script.splitlines() if line.startswith("GPU_UUID=")
    )
    selector = shlex.split(selector_line[len("GPU_UUID=$(") : -1])[2]
    for code, filename in ((command[2], "source.py"), (selector, "select_gpu.py")):
        tree = ast.parse(code, feature_version=(3, 6))
        payload = next(
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "b64decode"
        )
        assert (
            zlib.decompress(base64.b64decode(payload))
            == Path(launch.__file__).with_name(filename).read_bytes()
        )
        namespace = {"__name__": "payload_test"}
        exec(
            code, namespace
        )  # Decode/compile the real payload without running either entrypoint.
        assert ("fetch" if filename == "source.py" else "select") in namespace


@pytest.fixture
def exited_launcher(live_assets):
    """Exercise real run()/save() without loading CUDA or starting a worker."""
    from unittest.mock import Mock

    source.write_json(
        live_assets.root / "verified-source.json",
        dict(commit="a" * 40, archive_sha256="b" * 64),
    )

    def run(config, before_spawn=lambda: None, runtime=None):
        child = Mock(pid=1234)
        child.poll.return_value = 7

        def spawn(*args, **kwargs):
            before_spawn()
            return child

        with (
            patch.object(launch, "reachable_ip", return_value="10.0.0.122"),
            patch.object(launch, "probe_source", return_value=runtime or {}),
            patch.object(launch.subprocess, "check_output", return_value="GPU-fixture"),
            patch.object(launch.subprocess, "Popen", side_effect=spawn),
            patch.object(launch.socket, "socket"),
            patch.object(launch.signal, "signal"),
            patch.object(launch.os, "killpg"),
        ):
            with pytest.raises(RuntimeError, match="before becoming healthy"):
                launch.run(config, live_assets.root / "source", "a" * 40, "b" * 64)
        return json.loads((live_assets.root / "receipt.json").read_text())

    return run


@pytest.mark.parametrize(
    "role,storage,created",
    [
        ("dit", "local", True),
        ("dit", "mounted", False),
        ("dit", None, False),
        ("encoder", "local", False),
    ],
)
def test_local_dit_cache_directories_created_before_spawn(
    live_assets, exited_launcher, role, storage, created
):
    c = live_assets.assets(role)
    keys = (
        "zing_block0_cache_dir",
        "zing_block0_cache_budget_root",
        "zing_block0_cache_capacity_dir",
    )
    c["dit_options"] = {key: str(live_assets.root / "cache" / key) for key in keys}
    if storage is not None:
        c["dit_options"]["zing_block0_cache_storage"] = storage

    def check_directories():
        assert all(Path(c["dit_options"][key]).is_dir() == created for key in keys)

    exited_launcher(c, check_directories)
    check_directories()


def test_large_receipt_keeps_full_disk_copy_and_parseable_log_line(
    live_assets, exited_launcher, capsys
):
    import orchestrate

    c = live_assets.assets("dit")
    c["environment"] = {"SGLANG_ZING_TEST": "x" * 4000}
    full = exited_launcher(c)
    lines = capsys.readouterr().out.splitlines()
    records = [json.loads(line) for line in lines]
    assert records[-2] == full
    assert len(lines[-2]) > 1800
    expected = {
        key: value
        for key, value in full.items()
        if key not in {"adaln", "argv", "environment"}
    }
    assert records[-1] == expected
    assert lines[-1] == json.dumps(expected, separators=(",", ":"))
    assert len(lines[-1].encode()) < 2048
    assert all(key in full for key in ("adaln", "argv", "environment"))
    # Reproduce PAI's truncation of each individual line, not of the entire log response.
    truncated = "\n".join(
        line.encode()[:2048].decode(errors="ignore") for line in lines
    )
    assert orchestrate.last_receipt(truncated, "dit") == expected


@pytest.mark.parametrize("role", ["encoder", "dit", "decoder"])
def test_preflight_enables_address_reuse_before_binding(
    live_assets, exited_launcher, role
):
    from unittest.mock import call

    c = live_assets.assets(role)

    def check_preflight():
        listener = launch.socket.socket.return_value.__enter__.return_value
        expected = []
        for port in launch.role_ports(c).values():
            expected.extend(
                [
                    call.setsockopt(
                        launch.socket.SOL_SOCKET, launch.socket.SO_REUSEADDR, 1
                    ),
                    call.bind(("0.0.0.0", port)),
                ]
            )
        assert listener.method_calls == expected

    exited_launcher(c, check_preflight)


def test_managed_receipt_log_stays_under_pai_line_limit(
    live_assets, exited_launcher, capsys
):
    import orchestrate

    c = live_assets.assets("dit")
    c["worker_config"] = dict(
        schema=1,
        role="dit",
        service_count=1,
        coordinator_url="http://10.0.1.37:18181",
        pool_id="adaptive-test-20261109",
        control_epoch="test-control-epoch",
        compatibility="model1027-test",
        network_domain="shanghai-test",
        capacity={"slots": 1},
        profiles={"480": {"slots": 1}},
        profile_limits={
            "480": dict(
                source_sizes=[[832, 480]],
                target_sizes=[[832, 480]],
                max_fps=24,
                max_references=6,
                max_reference_bytes=1000000,
                max_reference_pixels=1000000,
                max_prompt_chars=10000,
                max_preview_pixels=1000000,
            )
        },
        hardware_ids=["__ALLOCATED_GPU_UUID__"],
        signing_secret_env="ROLE_POOL_SIGNING_KEY",
        role_token_env="ROLE_POOL_REGISTRATION_TOKEN",
    )
    runtime = {
        "sglang": {
            "path": "/tmp/verified-source/python/sglang/__init__.py",
            "sha256": "a" * 64,
        },
        "python": "v" * 500,
        "torch": "2.9",
        "cuda": "13.0",
    }
    full = exited_launcher(c, runtime=runtime)
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[-2]) == full
    old_compact = {
        key: value
        for key, value in full.items()
        if key not in {"adaln", "argv", "environment"}
    }
    assert len(json.dumps(old_compact, separators=(",", ":")).encode()) > 2048
    omitted = {
        "adaln",
        "argv",
        "environment",
        "runtime",
        "launcher_sha256",
        "release_id",
        "image_requested",
        "image_actual",
        "pipeline",
        "decoder_config",
        "model_path",
    }
    expected = {key: value for key, value in full.items() if key not in omitted}
    assert json.loads(lines[-1]) == expected
    assert len(lines[-1].encode()) < 1800
    assert expected["worker_config"]["hardware_ids"] == ["GPU-fixture"]
    assert all(key in full for key in omitted)
    truncated = "\n".join(
        line.encode()[:2048].decode(errors="ignore") for line in lines
    )
    assert orchestrate.last_receipt(truncated, "dit") == expected


def test_stage_assets_skips_already_staged_weight_files_with_matching_size(tmp_path):
    src, dst = tmp_path / "mounted", tmp_path / "local"
    (src / "model/transformer").mkdir(parents=True)
    (src / "model/transformer/part.safetensors").write_bytes(b"0123456789")
    (src / "model/zing_config.json").write_bytes(b'{"v":1}')
    c = dict(stage_from=str(src), model=str(dst / "model"))
    launch.stage_assets(c)
    copied = []
    orig_copyfile = launch.shutil.copyfile

    def tracked_copy(s, d, *args, **kwargs):
        copied.append(Path(s).name)
        return orig_copyfile(s, d, *args, **kwargs)

    (src / "model/zing_config.json").write_bytes(b'{"v":2}')
    with patch.object(launch.shutil, "copyfile", side_effect=tracked_copy):
        launch.stage_assets(c)
    assert "part.safetensors" not in copied
    assert "zing_config.json" in copied
    assert (dst / "model/zing_config.json").read_bytes() == b'{"v":2}'


def test_b300_and_multi_service_role_contracts(live_assets, exited_launcher):
    import shutil
    from unittest.mock import call

    for role in ("encoder", "decoder"):
        if (live_assets.root / "model").exists():
            shutil.rmtree(live_assets.root / "model")
        c = live_assets.assets(role)
        c.update(profile="b300", service_count=2)
        launch.validate_config(c)
        argv, _ = launch.command(c, live_assets.root, "10.0.0.121", f"{role}-w")
        assert argv[argv.index("--profile") + 1] == "b300"
        assert argv[argv.index("--service-count") + 1] == "2"
        site = dict(
            schema=1,
            image="test@sha256:" + "c" * 64,
            workspace_id="th-workspace",
            job_spec=dict(
                Type="Worker",
                PodCount=1,
                QuotaId="quota-b300",
                RestartPolicy="Never",
                ResourceConfig={"GPU": "1"},
            ),
            data_sources=[],
        )
        out = live_assets.root / f"b300-{role}"
        launch.render(c, runtime_manifest(), site, out)
        rendered_role = json.loads((out / "role.json").read_text())
        assert rendered_role["hardware"] == {
            "capability": [10, 3],
            "minimum_memory_gib": 240,
        }
        pai = json.loads((out / "pai.json").read_text())
        assert pai["WorkspaceId"] == "th-workspace"
        assert pai["ResourceId"] == "quota-b300"

    # Multi-service preflight binds every expanded port in the range.
    shutil.rmtree(live_assets.root / "model")
    enc = live_assets.assets("encoder")
    enc["service_count"] = 2

    def check_multi_preflight():
        listener = launch.socket.socket.return_value.__enter__.return_value
        expected = []
        for port in (30000, 30001, 31000, 31001, 32000, 32001):
            expected.extend(
                [
                    call.setsockopt(
                        launch.socket.SOL_SOCKET, launch.socket.SO_REUSEADDR, 1
                    ),
                    call.bind(("0.0.0.0", port)),
                ]
            )
        assert listener.method_calls == expected

    exited_launcher(enc, check_multi_preflight)

    # DiT rejects service_count != 1 and profile="h100".
    shutil.rmtree(live_assets.root / "model")
    dit = live_assets.assets("dit")
    with pytest.raises(ValueError, match="service_count=1"):
        launch.validate_config({**dit, "service_count": 2})
    with pytest.raises(ValueError, match="DiT profile"):
        launch.validate_config({**dit, "profile": "h100"})
    # Overlapping expanded port ranges fail closed.
    with pytest.raises(ValueError, match="overlapping"):
        launch.validate_config(
            {
                **enc,
                "service_count": 4,
                "ports": {"port": 30000, "http_port": 30002, "rendezvous_port": 32000},
            }
        )


def test_examples_reuse_profile_launcher_without_duplicate_acceleration_defaults():
    examples_dir = Path(__file__).parent / "examples"
    for name in ("encoder.json", "dit.json", "decoder.json"):
        data = json.loads((examples_dir / name).read_text())
        assert "role_launcher" not in data
        assert data["profile"] == "auto"
        assert "quantization" not in data
        assert "dit_options" not in data
    orch_data = json.loads((examples_dir / "orchestration.json").read_text())
    for role, data in orch_data["roles"].items():
        assert "role_launcher" not in data
        assert data["profile"] == "auto"
        assert "quantization" not in data
        assert "dit_options" not in data


def test_metrics_boot_isolation_and_receipt(live_assets, exited_launcher):
    config = live_assets.assets("encoder")
    config["service_count"] = 3
    config["environment"] = {"WORLD_MODEL_METRIC_LANE": "test-lane"}
    launch.validate_config(config)
    first = exited_launcher(config)["metrics"]
    directory = Path(first["multiprocess_dir"])
    assert directory.is_dir()
    marker = directory / "counter_123.db"
    marker.write_bytes(b"live counter")
    second = exited_launcher(config)["metrics"]
    assert first["multiprocess_dir"] != second["multiprocess_dir"]
    assert marker.read_bytes() == b"live counter"
    assert first["service"] == "encoder"
    assert first["lane"] == "test-lane"
    assert first["enabled"] is True
    # Shared-model services expose the same registry; advertise only the base HTTP.
    assert first["url"] == "http://10.0.0.122:31000/metrics"


def test_stage_assets_by_role_stages_only_role_hot_weights_and_links_rest(tmp_path):
    src = tmp_path / "mounted"
    contents = {
        "model/transformer/part.safetensors": b"dit-hot-weights",
        "model/video_vae/vae.safetensors": b"encoder-vae-weights",
        "model/processor/tokenizer.json": b'{"tok":1}',
        "model/text_encoder/qwen.safetensors": b"cold-text-encoder",
        "model/zing_config.json": b"{}",
        "model/model_index.json": b"{}",
        "model/EXPORT_COMPLETE.json": b"{}",
        "lynn/decoder.pt": b"decoder-lynn-weights",
        "adaln-table-unsplit/table.safetensors": b"dit-adaln-table",
        "adaln-table-unsplit/manifest.json": b"{}",
    }
    for rel, raw in contents.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)

    # 1. Encoder copies video_vae + processor, symlinks transformer / adaln / lynn.
    enc_dst = tmp_path / "enc_local"
    launch.stage_assets(
        {"role": "encoder", "stage_from": str(src), "model": str(enc_dst / "model")}
    )
    assert not (enc_dst / "model/video_vae/vae.safetensors").is_symlink()
    assert (
        enc_dst / "model/video_vae/vae.safetensors"
    ).read_bytes() == b"encoder-vae-weights"
    assert not (enc_dst / "model/processor/tokenizer.json").is_symlink()
    assert (enc_dst / "model/transformer").is_symlink()
    assert (enc_dst / "adaln-table-unsplit").is_symlink()
    assert (enc_dst / "lynn").is_symlink()

    # 2. Decoder copies lynn, symlinks transformer / video_vae / adaln.
    dec_dst = tmp_path / "dec_local"
    launch.stage_assets(
        {"role": "decoder", "stage_from": str(src), "model": str(dec_dst / "model")}
    )
    assert not (dec_dst / "lynn").is_symlink()
    assert not (dec_dst / "lynn/decoder.pt").is_symlink()
    assert (dec_dst / "lynn/decoder.pt").read_bytes() == b"decoder-lynn-weights"
    assert (dec_dst / "model/transformer").is_symlink()
    assert (dec_dst / "model/video_vae").is_symlink()
    assert (dec_dst / "adaln-table-unsplit").is_symlink()

    # 3. DiT copies transformer + adaln-table-unsplit, symlinks video_vae / lynn.
    dit_dst = tmp_path / "dit_local"
    launch.stage_assets(
        {"role": "dit", "stage_from": str(src), "model": str(dit_dst / "model")}
    )
    assert not (dit_dst / "model/transformer/part.safetensors").is_symlink()
    assert not (dit_dst / "adaln-table-unsplit/table.safetensors").is_symlink()
    assert (dit_dst / "model/video_vae").is_symlink()
    assert (dit_dst / "lynn").is_symlink()


def test_encoder_defaults_zing_source_prepare_env_without_overriding_explicit(
    live_assets,
):
    enc = live_assets.assets("encoder")
    _, default_env = launch.command(enc, live_assets.root, "10.0.0.122", "enc-w")
    assert default_env["SGLANG_ZING_SOURCE_PREPARE"] == "1"

    enc_override = {**enc, "environment": {"SGLANG_ZING_SOURCE_PREPARE": "0"}}
    launch.validate_config(enc_override)
    _, explicit_env = launch.command(
        enc_override, live_assets.root, "10.0.0.122", "enc-w"
    )
    assert explicit_env["SGLANG_ZING_SOURCE_PREPARE"] == "0"


def test_pai_render_resolves_image_auth_file_without_storing_inline_secret(
    live_assets, tmp_path
):
    c = live_assets.assets("dit")
    auth_file = tmp_path / "ACR_IMAGE_AUTH"
    auth_file.write_text("secret-acr-auth-token\n")
    site = dict(
        schema=1,
        image="test@sha256:" + "c" * 64,
        image_auth_file=str(auth_file),
        job_spec=dict(
            Type="Worker",
            PodCount=1,
            QuotaId="quota3l8bnv9a35n",
            RestartPolicy="Never",
            ResourceConfig={"GPU": "1"},
            ImageConfig={"DockerRegistry": "registry.example", "Username": "user"},
        ),
        data_sources=[],
    )
    out = tmp_path / "rendered"
    launch.render(c, runtime_manifest(), site, out)
    pai = json.loads((out / "pai.json").read_text())
    assert pai["JobSpecs"][0]["ImageConfig"]["Auth"] == "secret-acr-auth-token"
    assert "Auth" not in site["job_spec"]["ImageConfig"]

    with pytest.raises(ValueError, match="image_auth_file"):
        launch.render(
            c,
            runtime_manifest(),
            {**site, "image_auth_file": "relative/secret"},
            tmp_path / "bad-rel",
        )


def test_assets_cli_sized_weights_records_bytes_and_sha256(
    live_assets, tmp_path, monkeypatch
):
    c = live_assets.assets("dit")
    cfg_path = tmp_path / "dit-config.json"
    cfg_path.write_text(json.dumps(c))
    out_receipt = tmp_path / "sized-assets.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "assets",
            "--config",
            str(cfg_path),
            "--output",
            str(out_receipt),
            "--sized-weights",
        ],
    )
    launch.main()
    data = json.loads(out_receipt.read_text())
    weight_key = str(Path(c["model"]) / "model.safetensors")
    cfg_key = str(Path(c["model"]) / "zing_config.json")
    assert isinstance(data["files"][weight_key], dict)
    assert set(data["files"][weight_key]) == {"bytes", "sha256"}
    assert isinstance(data["files"][cfg_key], str)

    c["asset_receipt"] = str(out_receipt)
    c["asset_receipt_sha256"] = source.digest(out_receipt)
    launch.verify_assets(c)


def test_select_gpu_ignores_void_and_none_nvidia_visible_devices():
    import select_gpu

    gpus = [
        {"uuid": "GPU-free-0", "memory": 0, "name": "H100", "total_memory_mib": 81559}
    ]
    pods = {
        "items": [
            {
                "spec": {
                    "containers": [
                        {
                            "env": [
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
                                {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    assert select_gpu.select(gpus, [], pods) == "GPU-free-0"


def test_run_records_warmed_up_and_stage_timings_in_receipt(live_assets):
    from unittest.mock import MagicMock, Mock

    c = live_assets.assets("encoder")
    source.write_json(
        live_assets.root / "verified-source.json",
        dict(commit="a" * 40, archive_sha256="b" * 64),
    )
    polls = [None, 0]
    child = Mock(pid=4321)
    child.poll.side_effect = lambda: polls.pop(0) if polls else 0
    child.wait.return_value = 0

    health_resp = MagicMock()
    health_resp.__enter__.return_value.read.return_value = json.dumps(
        {
            "ready": True,
            "role": "encoder",
            "warmed_up": True,
            "warmup": {"role": "encoder", "warmed_up": True, "elapsed_s": 3.21},
        }
    ).encode()

    opener = Mock()
    opener.open.return_value = health_resp

    with (
        patch.object(launch, "reachable_ip", return_value="10.0.0.122"),
        patch.object(launch, "probe_source", return_value={}),
        patch.object(launch.subprocess, "check_output", return_value="GPU-fixture"),
        patch.object(launch.subprocess, "Popen", return_value=child),
        patch.object(launch.urllib.request, "build_opener", return_value=opener),
        patch.object(launch.socket, "socket"),
        patch.object(launch.signal, "signal"),
        patch.object(launch.time, "sleep"),
    ):
        rc = launch.run(c, live_assets.root / "source", "a" * 40, "b" * 64)
    assert rc == 0
    saved = json.loads((live_assets.root / "receipt.json").read_text())
    assert saved["status"] == "exited"
    assert saved["warmed_up"] is True
    assert saved["warmup"]["elapsed_s"] == 3.21
    for k in ("stage_assets_s", "verify_assets_s", "worker_ready_s", "total_boot_s"):
        assert k in saved["timings_s"]


@pytest.mark.parametrize("missing_key", ["model_id", "component", "ema", "model"])
def test_verify_assets_rejects_incomplete_receipt_metadata_with_value_error(
    live_assets, missing_key
):
    c = live_assets.assets("encoder")
    receipt = json.loads(Path(c["asset_receipt"]).read_text())
    receipt.pop(missing_key, None)
    Path(c["asset_receipt"]).write_text(json.dumps(receipt))
    c["asset_receipt_sha256"] = source.digest(c["asset_receipt"])
    with pytest.raises(ValueError, match="Model identity or component mismatch"):
        launch.verify_assets(c)

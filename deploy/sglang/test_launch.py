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
                    {"PYTHONPATH": "/image/source", "TOKEN": "private"},
                    clear=True,
                ):
                    argv, env = launch.command(
                        c, self.root, "10.0.0.2", "worker-fixture"
                    )
                self.assertIn(launch.MODULES[role], argv)
                self.assertEqual(env["PYTHONPATH"], str(self.root / "python"))
                self.assertNotIn("TOKEN", env)
                self.assertEqual(argv[argv.index("--service-count") + 1], "1")
                if role == "decoder":
                    self.assertIn("torch.distributed.run", argv)
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
                index = argv.index(launch.MODULES[role])
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

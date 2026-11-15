"""Offline contracts: no GPU, credentials or cloud network access required."""

import asyncio
import base64
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import launch
import orchestrate as orch
import source


@pytest.fixture
def spec(tmp_path):
    roles = {}
    for role in orch.ROLES:
        roles[role] = dict(
            schema=1,
            role=role,
            model_id="model1027",
            model="/assets/model1027",
            asset_receipt="/assets/assets.json",
            asset_receipt_sha256="a" * 64,
            environment={},
        )
    roles["dit"].update(
        pipeline="/assets/dit.json",
        dit_options={
            "mem_fraction_static": 0.8,
            "zing_denoising_step_list": "1000,500",
        },
    )
    roles["decoder"]["decoder_config"] = "/assets/decoder.json"
    limits = dict(
        source_sizes=[[832, 480]],
        target_sizes=[[832, 480]],
        max_fps=24,
        max_references=6,
        max_reference_bytes=1000,
        max_reference_pixels=1000,
        max_prompt_chars=1000,
        max_preview_pixels=1000,
    )
    return dict(
        schema=1,
        target=copy.deepcopy(orch.TARGET),
        model_id="model1027",
        roles=roles,
        sites=dict(
            h100=dict(
                schema=1,
                image="test@sha256:" + "c" * 64,
                namespace="sglang-test",
                volumes=[],
                volume_mounts=[],
            ),
            h200=dict(
                schema=1,
                image="test@sha256:" + "c" * 64,
                job_spec=dict(
                    Type="Worker",
                    PodCount=1,
                    QuotaId=orch.TARGET["quota_id"],
                    RestartPolicy="Never",
                    ResourceConfig={"GPU": "1"},
                ),
                data_sources=[],
            ),
        ),
        route="test-480",
        pool=dict(
            control_epoch="epoch-test",
            compatibility="model1027-test",
            network_domain="shanghai-test",
            capacity={r: {"slots": 1} for r in orch.ROLES},
            profiles={r: {"480": {"slots": 1}} for r in orch.ROLES},
            profile_limits={"480": limits},
        ),
    )


@pytest.fixture
def state(spec, tmp_path):
    manifest = dict(
        schema=1,
        commit="b" * 40,
        sha256="c" * 64,
        uri="oss://leap-world-shanghai/test/"
        + "b" * 40
        + "/"
        + "c" * 64
        + "/source.tar",
        tools={
            n: source.digest(Path(__file__).with_name(n))
            for n in ("launch.py", "source.py", "select_gpu.py", "orchestrate.py")
        },
    )
    state = dict(
        schema=1,
        run_id="d" * 32,
        target=copy.deepcopy(orch.TARGET),
        phase="planned",
        generation=0,
        source=manifest,
        spec=spec,
        resources={},
    )
    orch.render_roles(state, tmp_path)
    for role in orch.ROLES:
        name = orch.read(tmp_path / (role + "-0/render.json"))["name"]
        c = orch.read(tmp_path / (role + "-0/role.json"))
        receipt = dict(
            schema=1,
            role=role,
            status="ready",
            commit=manifest["commit"],
            archive_sha256=manifest["sha256"],
            model_id="model1027",
            asset_receipt_sha256="a" * 64,
            worker_id=role + "-worker",
            role_config_sha256=hashlib.sha256(
                json.dumps(c, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            http="http://"
            + ("10.0.0.122" if role == "dit" else "10.0.0.121")
            + ":"
            + str(launch.role_ports(spec["roles"][role])["http_port"]),
            gpu="GPU-" + role + ", fixture, 140000, 0",
        )
        resource = dict(
            run_id=state["run_id"],
            generation=0,
            name=name,
            status="submitted",
            receipt=receipt,
        )
        if role == "dit":
            resource["job_id"] = "dlc-owned"
        else:
            resource.update(namespace="sglang-test", uid=role + "-uid")
        state["resources"][role] = resource
    orch.save(tmp_path / "state.json", state)
    return state


def write_spec(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_spec_freezes_nccl_and_host_ports(spec, tmp_path):
    checked = orch.load_spec(write_spec(tmp_path, spec))
    assert all(
        "NCCL_MIN_NCHANNELS" not in c["environment"]
        and "NCCL_MIN_CTAS" not in c["environment"]
        for c in checked["roles"].values()
    )
    with_topo = copy.deepcopy(spec)
    with_topo["target"]["topology_profile"] = "cross_host_sm90_h100_h200"
    checked_topo = orch.load_spec(write_spec(tmp_path, with_topo))
    assert all(
        c["environment"]["NCCL_MIN_NCHANNELS"] == "2"
        and c["environment"]["NCCL_MIN_CTAS"] == "2"
        for c in checked_topo["roles"].values()
    )
    spec["roles"]["decoder"]["ports"] = {"port": 30000}
    with pytest.raises(ValueError, match="disjoint"):
        orch.load_spec(write_spec(tmp_path, spec))


@pytest.mark.parametrize(
    "field,value",
    [
        ("region", "ap-southeast-7"),
        ("h100_ip", "10.0.1.33"),
        ("coordinator_url", "http://10.0.1.33:18181"),
        ("pool_id", "prod"),
        ("oss_prefix", "oss://other/test"),
        ("workspace_id", "other"),
    ],
)
def test_pinned_copy_target_identity_cannot_drift(spec, tmp_path, field, value):
    spec["target"][field] = value
    with pytest.raises(ValueError, match="pinned"):
        orch.load_spec(write_spec(tmp_path, spec))


@pytest.mark.parametrize(
    "url,role",
    [
        ("http://10.0.1.33:28110", "encoder"),
        ("http://10.0.0.121:33000", "dit"),
        ("http://127.0.0.1:33000", "dit"),
        ("http://10.0.0.122", "dit"),
        ("http://user:password@10.0.0.122:33000", "dit"),
        ("http://10.0.0.121:31000/path", "encoder"),
    ],
)
def test_endpoint_guard(url, role):
    with pytest.raises(ValueError):
        orch.validate_endpoint(url, role)


@pytest.mark.parametrize(
    "environment", ["prod", "production", "prod-shanghai", "prod-thailand"]
)
@pytest.mark.parametrize("h100_ip", ["10.0.0.145", "10.0.1.31", "10.0.1.33"])
def test_production_spec_accepts_environment_pool_nodes_and_ports(
    spec, tmp_path, environment, h100_ip
):
    spec["target"].update(
        environment=environment,
        pool_id="prod-model-pool",
        h100_ip=h100_ip,
        coordinator_url="http://10.0.1.31:28110",
        room_url="http://10.0.1.33:28210",
        wtp_url="https://prod.example.com",
    )
    spec["roles"]["encoder"]["ports"] = {"http_port": 28110}
    spec["roles"]["decoder"]["ports"] = {"http_port": 28210}
    checked = orch.load_spec(write_spec(tmp_path, spec))
    assert checked["target"]["environment"] == environment
    assert checked["target"]["pool_id"] == "prod-model-pool"
    orch.validate_endpoint(
        f"http://{h100_ip}:28110", "encoder", target=checked["target"]
    )
    orch.validate_endpoint(
        f"http://{h100_ip}:28210", "decoder", target=checked["target"]
    )


@pytest.mark.parametrize("host", ["10.0.0.145", "10.0.1.31", "10.0.1.33"])
@pytest.mark.parametrize("port", [28110, 28210])
def test_pai_endpoint_accepts_production_addresses_and_ports(host, port):
    orch.validate_endpoint(f"http://{host}:{port}", "dit")


def test_production_reused_service_and_saved_state_are_accepted(state, tmp_path):
    state["target"].update(
        environment="prod-shanghai", pool_id="prod-model-pool", h100_ip="10.0.1.33"
    )
    state["spec"]["target"] = copy.deepcopy(state["target"])
    state["spec"]["roles"]["encoder"] = {
        "mode": "reuse",
        "http": "http://10.0.1.33:28110",
        "worker_id": "production-encoder",
    }
    state["spec"] = orch.validate_spec(state["spec"])
    state["resources"]["encoder"] = orch.reused_resource(
        state["run_id"],
        state["spec"]["model_id"],
        "encoder",
        state["spec"]["existing_services"]["encoder"],
    )
    state_path = tmp_path / "production-state.json"
    orch.save(state_path, state)
    restored = orch.load_state(state_path)
    assert restored["target"]["environment"] == "prod-shanghai"
    assert orch.owned_roles(restored["spec"]) == ("dit", "decoder")


@pytest.mark.parametrize(
    "ports",
    [
        {"port": 0},
        {"port": True},
        {"port": 65536},
        {"port": 31000},
        {"output_port": 38000},
    ],
)
def test_role_port_validation(spec, ports):
    c = spec["roles"]["encoder"]
    c["ports"] = ports
    with pytest.raises(ValueError):
        launch.validate_config(c)


def test_native_argv_custom_ports_dit_options_and_profile(spec, tmp_path):
    c = spec["roles"]["dit"]
    c["ports"] = dict(port=41001, http_port=41002, rendezvous_port=41003)
    argv, _ = launch.command(c, tmp_path, "10.0.0.122", "owned")
    for option, value in (
        ("--port", "41001"),
        ("--rendezvous-port", "41003"),
        ("--mem-fraction-static", "0.8"),
        ("--zing-denoising-step-list", "1000,500"),
    ):
        assert argv[argv.index(option) + 1] == value
    c["role_launcher"] = "profile"
    argv, _ = launch.command(c, tmp_path, "10.0.0.122", "owned")
    assert "sglang.multimodal_gen.tools.zing_h3_profile_launcher" in argv
    assert argv[argv.index("--mode") + 1] == "independent"
    assert "--port" in argv


@pytest.mark.parametrize("option", sorted(launch.UNSUPPORTED_DIT_OPTIONS))
def test_missing_native_cache_flags_are_not_silently_ignored(spec, option):
    spec["roles"]["dit"]["dit_options"][option] = "1"
    with pytest.raises(ValueError, match="Unsupported native"):
        launch.validate_config(spec["roles"]["dit"])


def test_environment_allows_agent_and_zing_but_keeps_source_isolation(
    spec, tmp_path, monkeypatch
):
    env = dict(
        ROLE_POOL_SIGNING_KEY="sensitive",
        ROLE_POOL_REGISTRATION_TOKEN="private",
        NCCL_GRAPH_REGISTER="0",
        SGLANG_ZING_FA4_FIXED="0",
        PYTHONPATH="/untrusted",
        PYTHONHOME="/wrong",
        LD_PRELOAD="/wrong",
    )
    monkeypatch.setattr(os, "environ", env)
    c = spec["roles"]["encoder"]
    _, actual = launch.command(c, tmp_path, "10.0.0.121", "owned")
    assert actual["ROLE_POOL_SIGNING_KEY"] == "sensitive"
    assert actual["SGLANG_ZING_FA4_FIXED"] == "0"
    assert actual["PYTHONPATH"] == str(tmp_path / "python")
    assert "LD_PRELOAD" not in actual and "PYTHONHOME" not in actual
    c["environment"] = {"ROLE_POOL_SIGNING_KEY": "inline-secret"}
    with pytest.raises(ValueError, match="Secrets"):
        launch.validate_config(c)


def test_model1027_asset_generation_and_tamper(tmp_path, spec):
    model = tmp_path / "model1027"
    model.mkdir()
    (model / "zing_config.json").write_text("{}")
    weights = model / "weights.safetensors"
    weights.write_bytes(b"weights")
    c = {**spec["roles"]["encoder"], "model": str(model)}
    result = launch.generate_assets(c, [], tmp_path / "assets.json")
    c.update(result)
    assert launch.verify_assets(c)["model_id"] == "model1027"
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        launch.verify_assets(c)
    with pytest.raises(FileExistsError):
        launch.generate_assets(c, [], tmp_path / "assets.json")


def test_receipt_parsing_ignores_other_roles_and_accepts_pai_prefix(state):
    receipt = state["resources"]["dit"]["receipt"]
    assert (
        orch.last_receipt("noise\n2026-11-10 10:00:00 " + json.dumps(receipt), "dit")
        == receipt
    )
    with pytest.raises(RuntimeError):
        orch.last_receipt(json.dumps(receipt), "encoder")


def test_ready_requires_runtime_and_live_health(state, tmp_path, monkeypatch):
    cloud = Mock(timeout=1)
    cloud.receipt.side_effect = lambda r, role: r["receipt"]
    monkeypatch.setattr(
        orch,
        "http",
        lambda url, cloud=None: dict(
            ready=True,
            role=next(
                r
                for r in orch.ROLES
                if state["resources"][r]["receipt"]["http"] + "/health" == url
            ),
        ),
    )
    orch.wait_ready(state, tmp_path, cloud)
    assert state["phase"] == "ready" and not state["waiting"]
    state["resources"]["dit"]["receipt"]["commit"] = "f" * 40
    with pytest.raises(ValueError, match="identity mismatch"):
        orch.wait_ready(state, tmp_path, cloud)


def test_ready_timeout_keeps_resource_ids(state, tmp_path, monkeypatch):
    cloud = Mock(timeout=1)
    cloud.receipt.side_effect = RuntimeError("No receipt yet")
    monkeypatch.setattr(orch.time, "sleep", lambda _: None)
    monkeypatch.setattr(orch.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    with pytest.raises(TimeoutError):
        orch.wait_ready(state, tmp_path, cloud)
    assert (
        orch.read(tmp_path / "state.json")["resources"]["dit"]["job_id"] == "dlc-owned"
    )


def job(resource, status="Running"):
    return dict(
        JobId=resource["job_id"],
        DisplayName=resource["name"],
        WorkspaceId="1453237",
        ResourceId="quota3l8bnv9a35n",
        Status=status,
        JobSpecs=[dict(PodCount=1, ResourceConfig={"GPU": "1"})],
    )


def test_down_checks_every_identity_before_any_mutation(state, tmp_path):
    cloud = Mock()
    cloud.pod.side_effect = lambda r: {"metadata": {"uid": "wrong"}}
    with pytest.raises(ValueError, match="UID changed"):
        orch.down(state, tmp_path, cloud)
    cloud.cli.assert_not_called()
    cloud.eflo.assert_not_called()


def test_down_uid_precondition_and_single_exact_job(state, tmp_path):
    cloud = Mock(timeout=1)
    cloud.pod.side_effect = lambda r: {"metadata": {"uid": r["uid"]}}
    cloud.get_job.side_effect = lambda _: job(state["resources"]["dit"], "Stopped")
    orch.down(state, tmp_path, cloud)
    cloud.cli.assert_called_once_with("pai-dlc", "StopJob", JobId="dlc-owned")
    assert state["phase"] == "stopped"
    assert len(cloud.eflo.call_args_list) == 2
    for call in cloud.eflo.call_args_list:
        assert "preconditions" in call.args[0] and "uid" in call.args[0]
        assert "--all" not in call.args[0]
    cloud.reset_mock()
    orch.down(state, tmp_path, cloud)
    cloud.cli.assert_not_called()


def test_unknown_create_cannot_be_deleted_or_resubmitted(state, tmp_path):
    state["resources"] = {"dit": {**state["resources"]["dit"]}}
    state["resources"]["dit"].pop("job_id")
    cloud = Mock()
    with pytest.raises(ValueError, match="UNKNOWN"):
        orch.down(state, tmp_path, cloud)
    state["phase"] = "submitting"
    with pytest.raises(ValueError, match="never retry"):
        orch.submit(state, tmp_path, cloud)
    cloud.assert_not_called()


def test_state_refuses_foreign_run_name_or_target(state, tmp_path):
    assert orch.load_state(tmp_path / "state.json")["run_id"] == state["run_id"]
    state["resources"]["dit"]["name"] = "other-job"
    orch.save(tmp_path / "state.json", state)
    with pytest.raises(ValueError, match="name"):
        orch.load_state(tmp_path / "state.json")


def test_attach_plan_is_local_and_has_no_secret_values(state, tmp_path, monkeypatch):
    state["phase"] = "ready"
    state["selftest"] = {
        "status": "passed",
        "source": copy.deepcopy(state["source"]),
        "worker_ids": {
            r: state["resources"][r]["receipt"]["worker_id"] for r in orch.ROLES
        },
    }
    monkeypatch.setattr(
        orch, "http", Mock(side_effect=AssertionError("No network in attach plan"))
    )
    cloud = Mock()
    configs = orch.attach(state, tmp_path, cloud)
    assert set(configs) == set(orch.ROLES)
    assert configs["dit"]["hardware_ids"] == ["GPU-dit"]
    assert configs["encoder"]["role_token_env"] == "ROLE_POOL_REGISTRATION_TOKEN"
    assert "secret" not in configs["dit"].keys()
    cloud.assert_not_called()
    assert state["generation"] == 0


def test_attach_cannot_skip_selftest(state, tmp_path):
    with pytest.raises(ValueError, match="selftest"):
        orch.attach(state, tmp_path, Mock())


def test_pool_config_rejects_shared_gpu_and_unmeasured_limits(state):
    state["resources"]["decoder"]["receipt"]["gpu"] = "GPU-encoder, fixture"
    with pytest.raises(ValueError, match="distinct"):
        orch.worker_configs(state)
    state["resources"]["decoder"]["receipt"]["gpu"] = "GPU-decoder, fixture"
    state["spec"]["pool"]["profile_limits"]["480"].pop("max_fps")
    with pytest.raises(ValueError, match="INVALID_PROFILE"):
        orch.worker_configs(state)


def test_eflo_uses_real_nested_node_schema_and_checks_exit():
    cloud = orch.Cloud(timeout=1)
    response = dict(
        Invocations=dict(
            Invocation=[
                dict(
                    InvokeId="inv-1",
                    InvokeNodes=dict(
                        InvokeNode=[
                            dict(
                                NodeId=orch.TARGET["node_id"],
                                InvocationStatus="Success",
                                ExitCode=0,
                                Dropped=0,
                                Output="result",
                            )
                        ]
                    ),
                )
            ]
        )
    )
    cloud.cli = Mock(side_effect=[{"InvokeId": "inv-1"}, response])
    submitted = Mock()
    assert cloud.eflo("hostname", submitted=submitted) == "result"
    submitted.assert_called_once_with("inv-1")
    response["Invocations"]["Invocation"][0]["InvokeNodes"]["InvokeNode"][0][
        "ExitCode"
    ] = 1
    cloud.cli = Mock(side_effect=[{"InvokeId": "inv-1"}, response])
    with pytest.raises(RuntimeError, match="failed"):
        cloud.eflo("hostname")


def test_pai_receipt_logs_are_array_and_scoped_to_exact_job(state):
    cloud = orch.Cloud()
    resource = state["resources"]["dit"]
    observed = job(resource)
    observed["Pods"] = [{"PodId": "owned-worker-0"}]
    cloud.get_job = Mock(return_value=observed)
    cloud.cli = Mock(
        return_value=dict(
            JobId="dlc-owned",
            PodId="owned-worker-0",
            Logs=[json.dumps(resource["receipt"])],
        )
    )
    assert cloud.receipt(resource, "dit") == resource["receipt"]
    cloud.cli.assert_called_once_with(
        "pai-dlc", "GetPodLogs", JobId="dlc-owned", PodId="owned-worker-0", MaxLines=100
    )


def test_cli_mutations_default_to_offline(state, tmp_path, monkeypatch, capsys):
    for action in ("down", "selftest"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["orchestrate.py", action, "--state", str(tmp_path / "state.json")],
        )
        with patch.object(orch, "Cloud") as cloud:
            orch.main()
            cloud.return_value.cli.assert_not_called()
            cloud.return_value.eflo.assert_not_called()
        assert json.loads(capsys.readouterr().out)["status"] == "not-executed"


def test_native_group_selftest_uses_exact_three_workers_and_destroys(
    state, tmp_path, monkeypatch
):
    state["phase"] = "ready"
    calls = []
    descriptors = {
        r: dict(
            role=r,
            worker_id=state["resources"][r]["receipt"]["worker_id"],
            service_id=r + "/0",
            endpoints={"http": state["resources"][r]["receipt"]["http"]},
        )
        for r in orch.ROLES
    }

    def request(address, path, **kwargs):
        role = next(
            r for r in orch.ROLES if state["resources"][r]["receipt"]["http"] == address
        )
        calls.append((role, path))
        if path == "/health":
            return dict(ready=True, role=role)
        if path == "/get_connection_status":
            return dict(
                state="available",
                reservation_token="private-token",
                service=descriptors[role],
            )
        if path == "/build_group":
            return dict(
                success=True,
                state="connected",
                worker_group_id=kwargs["payload"]["worker_group_id"],
            )
        if path == "/quiesce_group":
            return dict(
                success=True,
                state="quiesced",
                worker_group_id=kwargs["payload"]["worker_group_id"],
            )
        if path == "/destroy_group":
            return dict(success=True, state="disconnected")
        raise AssertionError(path)

    original = orch.importlib.util.module_from_spec

    def module_from_spec(spec):
        module = original(spec)
        loader = spec.loader
        execute = loader.exec_module

        def load(target):
            execute(target)
            target.http_request = request

        loader.exec_module = load
        return module

    monkeypatch.setattr(orch.importlib.util, "module_from_spec", module_from_spec)
    asyncio.run(orch.selftest(state, tmp_path))
    assert state["selftest"]["status"] == "passed"
    assert all((r, "/destroy_group") in calls for r in orch.ROLES)
    assert "private-token" not in (tmp_path / "selftest.json").read_text()


def test_submit_writes_identity_before_create_and_keeps_unknown(
    state, tmp_path, monkeypatch
):
    (tmp_path / "source").mkdir()
    archive = tmp_path / "source/source.tar"
    archive.write_bytes(b"archive")
    state["source"]["sha256"] = source.digest(archive)
    state["resources"] = {}
    cloud = Mock()

    def fail_create(script, submitted=None):
        saved = orch.read(tmp_path / "state.json")
        assert saved["resources"]["encoder"]["status"] == "submitting"
        submitted("inv-owned")
        raise TimeoutError("Simulated unknown create")

    cloud.eflo.side_effect = fail_create
    monkeypatch.setattr(orch.subprocess, "run", Mock())
    with pytest.raises(TimeoutError):
        orch.submit(state, tmp_path, cloud)
    saved = orch.read(tmp_path / "state.json")
    assert saved["resources"]["encoder"]["invoke_id"] == "inv-owned"
    assert saved["phase"] == "submitting"
    cloud.submit_job.assert_not_called()


def test_submit_checks_all_payloads_before_upload(state, tmp_path, monkeypatch):
    (tmp_path / "dit-0/pai.json").write_text("{}")
    upload = Mock()
    monkeypatch.setattr(orch.subprocess, "run", upload)
    with pytest.raises(ValueError, match="payload changed"):
        orch.submit(state, tmp_path, Mock())
    upload.assert_not_called()


def test_up_prepares_uploads_and_submits_all_three_once(state, tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    archive = tmp_path / "source/source.tar"
    archive.write_bytes(b"archive")
    state["source"]["sha256"] = source.digest(archive)
    state["resources"] = {}
    cloud = Mock()

    def create(script, submitted=None):
        role = "encoder" if "sglang-encoder-" in script else "decoder"
        resource = state["resources"][role]
        return json.dumps(
            dict(
                metadata=dict(
                    name=resource["name"],
                    namespace=resource["namespace"],
                    uid=role + "-uid",
                )
            )
        )

    cloud.eflo.side_effect = create
    cloud.submit_job.return_value = "dlc-new"
    upload = Mock()
    monkeypatch.setattr(orch.subprocess, "run", upload)
    wait = Mock()
    monkeypatch.setattr(orch, "wait_ready", wait)
    orch.submit(state, tmp_path, cloud)
    assert set(state["resources"]) == set(orch.ROLES)
    assert state["resources"]["dit"]["job_id"] == "dlc-new"
    assert cloud.eflo.call_count == 2 and cloud.submit_job.call_count == 1
    cloud.upload_source.assert_called_once_with(archive, state["source"]["uri"])
    cloud.stage_h100_source.assert_called_once_with(state["source"])
    wait.assert_called_once_with(state, tmp_path, cloud)


@pytest.mark.parametrize(
    "registration_error",
    [
        None,
        RuntimeError("Process registration is pending"),
        OSError("Coordinator is unavailable"),
        ValueError("Manifest mismatch"),
    ],
)
def test_attach_execute_only_recreates_owned_idle_roles(
    state, tmp_path, monkeypatch, registration_error
):
    state["phase"] = "ready"
    state["selftest"] = {
        "status": "passed",
        "source": copy.deepcopy(state["source"]),
        "worker_ids": {
            r: state["resources"][r]["receipt"]["worker_id"] for r in orch.ROLES
        },
    }
    for c in state["spec"]["roles"].values():
        c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "operator")
    monkeypatch.setattr(
        orch,
        "http",
        lambda url, cloud=None: (
            dict(backend="role_pool", profile="480")
            if "/v1/capacity" in url
            else dict(state="available")
        ),
    )
    stopped = []

    def stop(old, directory, cloud):
        stopped.append(copy.deepcopy(old["resources"]))

    monkeypatch.setattr(orch, "down", stop)
    monkeypatch.setattr(orch, "submit", Mock())
    verify = Mock(
        side_effect=[registration_error, None] if registration_error else [None]
    )
    monkeypatch.setattr(orch, "verify_pool", verify)
    sleep = Mock()
    monkeypatch.setattr(orch.time, "sleep", sleep)
    if isinstance(registration_error, ValueError):
        with pytest.raises(ValueError, match="Manifest mismatch"):
            orch.attach(state, tmp_path, Mock(timeout=1), execute=True)
        assert verify.call_count == 1
        sleep.assert_not_called()
    else:
        orch.attach(state, tmp_path, Mock(timeout=1), execute=True)
        assert verify.call_count == (2 if registration_error else 1)
        if registration_error:
            sleep.assert_called_once_with(2)
        else:
            sleep.assert_not_called()
    assert len(stopped) == 1 and set(stopped[0]) == set(orch.ROLES)
    assert state["generation"] == 1
    assert state["resources"] == {}
    assert all(
        c["worker_config"]["hardware_ids"] == ["__ALLOCATED_GPU_UUID__"]
        for c in state["spec"]["roles"].values()
    )
    assert (tmp_path / "dit-1/pai.json").exists()


def test_probe_only_reads_and_redacts_unrelated_pod_env(spec, tmp_path):
    inventory = dict(
        gpus="GPU-a, H100, 81559, 0\nGPU-b, H100, 81559, 0",
        processes="",
        busy_ports=[],
        models={"/assets/model1027": True},
        images={"images": [{"repoDigests": [spec["sites"]["h100"]["image"]]}]},
        pods={
            "items": [
                {
                    "metadata": {"name": "other"},
                    "status": {"phase": "Running"},
                    "spec": {
                        "nodeName": "other-node",
                        "containers": [
                            {"env": [{"name": "TOKEN", "value": "sensitive"}]}
                        ],
                    },
                }
            ]
        },
    )
    cloud = Mock()
    cloud.eflo.return_value = json.dumps(inventory)
    cloud.cli.return_value = dict(Jobs=[], TotalCount=0)
    result = orch.probe(spec, tmp_path / "probe", cloud)
    assert result["idle_gpu_uuids"] == ["GPU-a", "GPU-b"]
    cloud.cli.assert_called_once_with(
        "pai-dlc",
        "ListJobs",
        WorkspaceId="1453237",
        ShowOwn="false",
        PageNumber=1,
        PageSize=100,
    )
    script = cloud.eflo.call_args.args[0]
    import shlex

    assert "['kubectl','get'" in shlex.split(script)[2]
    assert "create" not in script and "delete" not in script
    assert "sensitive" not in (tmp_path / "probe/probe.json").read_text()


def test_managed_worker_binds_gpu_and_redacts_credentials(tmp_path, monkeypatch):
    # Use the existing fixture builder to exercise the real run() receipt and failure cleanup.
    from test_launch import LaunchTests

    fixture = LaunchTests()
    fixture.setUp()
    try:
        c = fixture.assets("dit")
        c["worker_config"] = dict(
            role="dit", service_count=1, hardware_ids=["__ALLOCATED_GPU_UUID__"]
        )
        source.write_json(
            fixture.root / "verified-source.json",
            dict(commit="a" * 40, archive_sha256="b" * 64),
        )
        child = Mock(pid=1234)
        child.poll.return_value = 1
        monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "never-log-this")
        monkeypatch.setenv("ROLE_POOL_REGISTRATION_TOKEN", "nor-this-token")
        with (
            patch.object(launch, "reachable_ip", return_value="10.0.0.122"),
            patch.object(launch, "probe_source", return_value={}),
            patch.object(
                launch.subprocess,
                "check_output",
                return_value="GPU-actual, H200, 140000, 0",
            ),
            patch.object(launch.subprocess, "Popen", return_value=child),
            patch.object(launch.os, "killpg"),
        ):
            with pytest.raises(RuntimeError):
                launch.run(c, fixture.root / "source", "a" * 40, "b" * 64)
        worker = orch.read(fixture.root / "role_pool_worker.json")
        assert worker["hardware_ids"] == ["GPU-actual"]
        receipt = (fixture.root / "receipt.json").read_text()
        assert "never-log-this" not in receipt and "nor-this-token" not in receipt
    finally:
        fixture.doCleanups()


def test_pool_verification_requires_exact_owned_manifest_not_aggregate_capacity(
    state, tmp_path, monkeypatch
):
    configs = orch.worker_configs(state)
    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "operator")
    for role in orch.ROLES:
        state["resources"][role]["receipt"]["worker_config"] = configs[role]
    import time

    def request(url, payload=None, token=None, cloud=None):
        if "/v1/capacity" in url:
            return dict(
                backend="role_pool",
                profile="480",
                roles={"model_group": {"free_slots": 9}},
            )
        if "/managed/identity" in url:
            role = next(
                r
                for r in orch.ROLES
                if state["resources"][r]["receipt"]["http"] + "/managed/identity" == url
            )
            return dict(
                service=dict(
                    worker_id=role + "-worker", role=role, service_id=role + "/0"
                ),
                boot_epoch=role + "-boot",
            )
        role = payload["process_id"].removesuffix("-worker")
        return dict(
            process=dict(
                manifest=copy.deepcopy(configs[role]),
                state="ready",
                seen_at=time.time(),
            )
        )

    monkeypatch.setattr(orch, "http", request)
    orch.verify_pool(state, tmp_path)
    assert set(state["attachment"]["identities"]) == set(orch.ROLES)
    assert "not deterministic" in state["attachment"]["routing"]
    configs["dit"]["pool_id"] = "wrong-pool"
    # Frozen runtime config and reported manifest must not share mutable test objects.
    state["resources"]["dit"]["receipt"]["worker_config"] = {
        **configs["dit"],
        "pool_id": orch.TARGET["pool_id"],
    }
    with pytest.raises(ValueError, match="manifest"):
        orch.verify_pool(state, tmp_path)


def test_role_endpoint_requires_ip_and_disallows_redirects():
    with pytest.raises(ValueError):
        orch.validate_endpoint("http://10.evil.example:33000", "dit")
    with pytest.raises(ValueError):
        orch.NoRedirect().redirect_request(
            None, None, 302, "redirect", {}, "http://10.0.0.145:33000"
        )


def test_stale_selftest_cannot_authorize_attach(state, tmp_path):
    state["phase"] = "ready"
    state["selftest"] = dict(status="passed", source=state["source"], worker_ids={})
    with pytest.raises(ValueError, match="stale"):
        orch.attach(state, tmp_path, Mock())


def test_eflo_run_command_uses_rpc_fields_not_body():
    cloud = orch.Cloud(timeout=1)
    cloud.cli = Mock(
        side_effect=[
            {"InvokeId": "inv-1"},
            {
                "Invocations": {
                    "Invocation": [
                        {
                            "InvokeId": "inv-1",
                            "InvokeNodes": {
                                "InvokeNode": [
                                    {
                                        "NodeId": orch.TARGET["node_id"],
                                        "InvocationStatus": "Success",
                                        "ExitCode": 0,
                                        "Output": "ok",
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        ]
    )
    script = "printf '%s\\n' '{\"message\":\"测试\"}'\npython3 -c 'print(1)'"
    assert cloud.eflo(script) == "ok"
    operation = cloud.cli.call_args_list[0]
    assert operation.args == ("eflo-controller", "RunCommand")
    assert operation.kwargs == dict(
        NodeIdList=[orch.TARGET["node_id"]],
        Name=operation.kwargs["Name"],
        RepeatMode="Once",
        Timeout=180,
        ContentEncoding="Base64",
        CommandContent=base64.b64encode(script.encode()).decode(),
    )
    assert operation.kwargs["Name"].startswith("sglang-observe-")

    assert base64.b64decode(operation.kwargs["CommandContent"]).decode() == script
    query = cloud.cli.call_args_list[1]
    assert query.args == ("eflo-controller", "DescribeInvocations")
    assert query.kwargs == dict(
        InvokeId="inv-1", IncludeOutput="true", ContentEncoding="PlainText"
    )


def test_upload_source_pins_shanghai_endpoint(tmp_path, monkeypatch):
    upload = Mock()
    monkeypatch.setattr(orch.subprocess, "run", upload)
    uri = "oss://leap-world-shanghai/releases/source.tar"
    archive = tmp_path / "source.tar"
    orch.Cloud(aliyun_profile="test-operator").upload_source(archive, uri)
    upload.assert_called_once_with(
        [
            "aliyun",
            "--profile",
            "test-operator",
            "--endpoint",
            "oss-cn-shanghai.aliyuncs.com",
            "oss",
            "cp",
            "--force",
            str(archive),
            uri,
        ],
        check=True,
        timeout=300,
    )


def test_h100_staging_signs_internal_url_and_hash_checks_before_publish(
    state, monkeypatch
):
    import shlex

    signed_url = "https://leap-world-shanghai.oss-cn-shanghai-internal.aliyuncs.com/source.tar?Signature=private&Expires=123"
    sign = Mock(return_value=signed_url + "\n")
    monkeypatch.setattr(orch.subprocess, "check_output", sign)
    cloud = orch.Cloud(aliyun_profile="test-operator")
    cloud.eflo = Mock(return_value="")
    cloud.stage_h100_source(state["source"])
    sign.assert_called_once_with(
        [
            "aliyun",
            "--profile",
            "test-operator",
            "--endpoint",
            "oss-cn-shanghai-internal.aliyuncs.com",
            "oss",
            "sign",
            state["source"]["uri"],
            "--timeout",
            "3600",
        ],
        text=True,
        timeout=60,
    )
    script = cloud.eflo.call_args.args[0]
    assert (
        "/var/lib/sglang-orchestrate/cache/" + state["source"]["sha256"] + "/source.tar"
        in script
    )
    assert "sha256sum -c -" in script and "flock" in script
    curl = shlex.split(
        next(line for line in script.splitlines() if line.startswith("curl "))
    )
    assert curl == ["curl", "-fsSL", "-o", "$TMP", "--", signed_url]
    assert script.index("then exit 0; fi") < script.index("curl ")
    assert script.rindex("sha256sum -c -") < script.index('mv "$TMP"')
    cloud.eflo.side_effect = RuntimeError("CLI error includes " + signed_url)
    with pytest.raises(RuntimeError, match="staging failed") as error:
        cloud.stage_h100_source(state["source"])
    assert "Signature" not in str(error.value)


def test_h100_staging_failure_prevents_all_pod_and_job_creates(state, tmp_path):
    (tmp_path / "source").mkdir()
    archive = tmp_path / "source/source.tar"
    archive.write_bytes(b"archive")
    state["source"]["sha256"] = source.digest(archive)
    state["resources"] = {}
    cloud = Mock()
    cloud.stage_h100_source.side_effect = RuntimeError("Source staging failed")
    with pytest.raises(RuntimeError, match="staging failed"):
        orch.submit(state, tmp_path, cloud)
    cloud.upload_source.assert_called_once()
    cloud.eflo.assert_not_called()
    cloud.submit_job.assert_not_called()
    assert not orch.read(tmp_path / "state.json")["resources"]


def test_single_pai_master_receipt_does_not_require_log_pod_id(state):
    cloud = orch.Cloud()
    resource = state["resources"]["dit"]
    observed = job(resource)
    observed["Pods"] = [
        {"PodId": "dlc-owned-aimaster-0"},
        {"PodId": "dlc-owned-master-0"},
    ]
    cloud.get_job = Mock(return_value=observed)
    cloud.cli = Mock(
        return_value=dict(JobId="dlc-owned", Logs=[json.dumps(resource["receipt"])])
    )
    assert cloud.receipt(resource, "dit") == resource["receipt"]
    cloud.cli.assert_called_once_with(
        "pai-dlc",
        "GetPodLogs",
        JobId="dlc-owned",
        PodId="dlc-owned-master-0",
        MaxLines=100,
    )
    cloud.cli.return_value["JobId"] = "different-job"
    with pytest.raises(ValueError, match="log identity"):
        cloud.receipt(resource, "dit")


@pytest.mark.parametrize("image_source", ["spec", "status", "status-digest-only"])
def test_probe_matches_running_pod_images_without_crictl_entries(
    spec, tmp_path, image_source
):
    import shlex

    image = spec["sites"]["h100"]["image"]
    pod = dict(
        spec=dict(
            nodeName="other-node",
            containers=[dict(image=image if image_source == "spec" else "test:tag")],
        ),
        status={},
    )
    if image_source != "spec":
        pod["status"]["containerStatuses"] = [
            dict(
                imageID="containerd://"
                + (image if image_source == "status" else image.split("@", 1)[1])
            )
        ]
    inventory = dict(
        gpus="GPU-a, H100, 81559, 0\nGPU-b, H100, 81559, 0",
        processes="",
        busy_ports=[],
        models={"/assets/model1027": True},
        images={"images": []},
        pods={"items": [pod]},
    )
    cloud = Mock()
    cloud.eflo.return_value = json.dumps(inventory)
    cloud.cli.return_value = dict(Jobs=[], TotalCount=0)
    result = orch.probe(spec, tmp_path / "probe", cloud)
    assert "Pinned H100 image digest is not present" not in result["blockers"]
    code = shlex.split(cloud.eflo.call_args.args[0])[2]
    assert (
        "['crictl','-r','unix:///run/k3s/containerd/containerd.sock','images'" in code
    )
    assert result["h100"]["observed_images"]


def test_http_vpc_fallback_preserves_post_auth_and_python36_compatibility(monkeypatch):
    import ast
    import contextlib
    import io
    import shlex
    import urllib.request

    direct = Mock()
    direct.open.side_effect = OSError("Laptop has no VPC route")
    monkeypatch.setattr(orch, "build_opener", Mock(return_value=direct))
    relay = Mock()
    forwarded = []

    def execute(script):
        argv = shlex.split(script)
        assert argv[:2] == ["python3", "-c"]
        ast.parse(argv[2], feature_version=(3, 6))
        remote_opener = Mock()
        remote_opener.open.return_value = io.StringIO('{"ready": true}')
        capture = io.StringIO()
        with (
            patch.object(urllib.request, "build_opener", return_value=remote_opener),
            contextlib.redirect_stdout(capture),
        ):
            exec(argv[2], {})
        forwarded.append(remote_opener.open.call_args.args[0])
        return capture.getvalue()

    relay.eflo.side_effect = execute
    result = orch.http(
        "http://10.0.1.37:18181/v1/role-pool/process-status",
        {"process_id": "owned"},
        "private-token",
        cloud=relay,
    )
    assert result == {"ready": True}
    assert json.loads(forwarded[0].data) == {"process_id": "owned"}
    assert forwarded[0].get_method() == "POST"
    assert forwarded[0].get_header("Authorization") == "Bearer private-token"
    assert forwarded[0].full_url == "http://10.0.1.37:18181/v1/role-pool/process-status"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/health",
        "http://172.16.0.1/health",
        "http://10.example.com/health",
    ],
)
def test_http_network_fallback_is_vpc_only(monkeypatch, url):
    direct = Mock()
    direct.open.side_effect = OSError("No route")
    monkeypatch.setattr(orch, "build_opener", Mock(return_value=direct))
    cloud = Mock()
    with pytest.raises(OSError):
        orch.http(url, cloud=cloud)
    cloud.eflo.assert_not_called()


def test_http_success_or_business_rejection_never_replays_on_h100(monkeypatch):
    import io
    from urllib.error import HTTPError

    direct = Mock()
    direct.open.return_value = io.StringIO('{"ok": true}')
    monkeypatch.setattr(orch, "build_opener", Mock(return_value=direct))
    cloud = Mock()
    assert orch.http("http://10.0.0.121:31000/health", cloud=cloud) == {"ok": True}
    cloud.eflo.assert_not_called()
    direct.open.side_effect = HTTPError(
        "http://10.0.0.121:31000/health", 403, "Forbidden", {}, None
    )
    with pytest.raises(HTTPError):
        orch.http("http://10.0.0.121:31000/health", cloud=cloud)
    cloud.eflo.assert_not_called()


def test_selftest_falls_back_to_owned_encoder_verified_source(
    state, tmp_path, monkeypatch
):
    import contextlib
    import io
    import shlex
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    state["phase"] = "ready"
    root = tmp_path / "release/source"
    (root / "deploy/sglang").mkdir(parents=True)
    (root / "deploy/sglang/orchestrate.py").write_bytes(
        Path(orch.__file__).read_bytes()
    )
    orch.save(
        root.parent / "verified-source.json",
        dict(
            commit=state["source"]["commit"], archive_sha256=state["source"]["sha256"]
        ),
    )
    encoder = state["resources"]["encoder"]
    encoder["receipt"]["runtime"] = {
        "sglang": {"path": str(root / "python/sglang/__init__.py")}
    }
    summary = dict(
        worker_group_id="owned-test-group",
        topology={
            r: dict(worker_id=state["resources"][r]["receipt"]["worker_id"])
            for r in orch.ROLES
        },
        reused=False,
        connections_kept=False,
    )
    native = AsyncMock(side_effect=OSError("Laptop cannot connect to VPC"))
    monkeypatch.setattr(orch, "native_selftest", native)
    inside_pod = AsyncMock(return_value=summary)
    cloud = Mock()
    cloud.pod.return_value = dict(metadata=dict(uid=encoder["uid"]))

    def execute(script):
        argv = shlex.split(script)
        assert argv[:8] == [
            "kubectl",
            "-n",
            encoder["namespace"],
            "exec",
            encoder["name"],
            "--",
            "/opt/sglang/bin/python",
            "-c",
        ]
        capture = io.StringIO()
        with (
            patch.dict(
                sys.modules,
                {"orchestrate": SimpleNamespace(native_selftest=inside_pod)},
            ),
            patch.object(sys, "path", list(sys.path)),
            contextlib.redirect_stdout(capture),
        ):
            exec(argv[8], {})
        return capture.getvalue()

    cloud.eflo.side_effect = execute
    asyncio.run(orch.selftest(state, tmp_path, cloud=cloud))
    inside_pod.assert_awaited_once()
    addresses, workers, _ = inside_pod.call_args.args
    assert set(addresses) == set(orch.ROLES)
    assert workers["encoder"] == encoder["receipt"]["worker_id"]
    assert state["selftest"]["execution"] == "owned-encoder-pod"
    assert orch.read(tmp_path / "selftest.json")["group"] == summary
    assert orch.read(tmp_path / "state.json")["selftest"]["status"] == "passed"


def test_selftest_does_not_replay_failed_build_or_teardown(
    state, tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock

    state["phase"] = "ready"
    monkeypatch.setattr(
        orch,
        "native_selftest",
        AsyncMock(side_effect=RuntimeError("Build failed after reservation")),
    )
    cloud = Mock()
    with pytest.raises(RuntimeError, match="Build failed"):
        asyncio.run(orch.selftest(state, tmp_path, cloud=cloud))
    cloud.pod.assert_not_called()
    cloud.eflo.assert_not_called()
    assert orch.read(tmp_path / "state.json")["selftest"]["status"] != "passed"


def test_remote_selftest_rejects_replaced_encoder_pod(state):
    cloud = Mock()
    cloud.pod.return_value = dict(metadata=dict(uid="replacement-uid"))
    with pytest.raises(ValueError, match="UID changed"):
        orch.remote_selftest(state, {}, {}, cloud)
    cloud.eflo.assert_not_called()


def test_native_selftest_network_failure_fails_fast_before_reservation(
    state, tmp_path, monkeypatch
):
    from urllib.error import URLError

    addresses = {
        role: state["resources"][role]["receipt"]["http"] for role in orch.ROLES
    }
    workers = {
        role: state["resources"][role]["receipt"]["worker_id"] for role in orch.ROLES
    }
    calls = []
    original = orch.importlib.util.module_from_spec

    def load_spec(spec):
        module = original(spec)
        execute = spec.loader.exec_module

        def load(target):
            execute(target)

            def request(address, path, **kwargs):
                calls.append(path)
                raise URLError("No VPC route")

            target.http_request = request

        spec.loader.exec_module = load
        return module

    monkeypatch.setattr(orch.importlib.util, "module_from_spec", load_spec)
    with pytest.raises(orch.VPCUnreachable):
        asyncio.run(orch.native_selftest(addresses, workers, tmp_path))
    assert calls and set(calls) == {"/health"}


def test_cloud_cli_does_not_expose_relay_credentials(monkeypatch):
    import subprocess

    def fail(argv, **kwargs):
        assert kwargs["stderr"] == subprocess.PIPE
        raise subprocess.CalledProcessError(1, argv, stderr="private-token")

    monkeypatch.setattr(orch.subprocess, "check_output", fail)
    with pytest.raises(RuntimeError) as error:
        orch.Cloud(aliyun_profile="test-operator").cli(
            "eflo-controller", "RunCommand", CommandContent="private-token"
        )
    assert str(error.value) == "Cloud CLI failed: eflo-controller RunCommand"


def test_probe_node_output_trims_historical_pods_before_eflo(spec, tmp_path):
    import ast
    import contextlib
    import io
    import shlex

    import select_gpu

    image = spec["sites"]["h100"]["image"]
    active = {
        "metadata": {"annotations": {"discarded": "large metadata"}},
        "spec": {
            "nodeName": "world-model-b27mys7b-0000",
            "runtimeClassName": "nvidia",
            "containers": [
                {
                    "image": "test:tag",
                    "env": [
                        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-a"},
                        {"name": "TOKEN", "value": "discarded-secret"},
                    ],
                    "command": ["discarded-command"],
                }
            ],
            "initContainers": [
                {
                    "image": "test:init",
                    "env": [
                        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-b"},
                    ],
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {"imageID": "containerd://" + image, "containerID": "discarded-id"}
            ],
            "initContainerStatuses": [
                {"imageID": "containerd://init@sha256:" + "d" * 64}
            ],
        },
    }
    raw_pods = json.dumps(
        {
            "items": [
                active,
                {
                    "metadata": {"discarded": "x" * (12 * 1024 * 1024)},
                    "status": {"phase": "Succeeded"},
                },
                {"status": {"phase": "Failed"}},
            ]
        }
    )
    assert len(raw_pods.encode()) > 11 * 1024 * 1024
    observed = []
    calls = []

    def node_command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "kubectl":
            return raw_pods
        if argv[0] == "crictl":
            assert argv == [
                "crictl",
                "--config",
                "/dev/null",
                "-r",
                "unix:///run/k3s/containerd/containerd.sock",
                "images",
                "-o",
                "json",
            ]
            return json.dumps(
                {"images": [{"repoDigests": [image], "discarded": "x" * 100000}]}
            )
        if argv[1] == "--query-compute-apps=gpu_uuid":
            return ""
        return "\n".join("GPU-" + gpu + ", H100, 81559, 0" for gpu in "abcd")

    def run_node(script):
        code = shlex.split(script)[2]
        ast.parse(code, feature_version=(3, 6))
        capture = io.StringIO()
        with (
            patch.object(orch.subprocess, "check_output", side_effect=node_command),
            patch("socket.socket"),
            patch("os.path.isdir", return_value=True),
            contextlib.redirect_stdout(capture),
        ):
            exec(code, {})
        text = capture.getvalue()
        assert len(text.encode()) < 24000
        observed.append(json.loads(text))
        return text

    cloud = Mock()
    cloud.eflo.side_effect = run_node
    cloud.cli.return_value = dict(Jobs=[], TotalCount=0)
    result = orch.probe(spec, tmp_path / "probe", cloud)
    assert result["idle_gpu_uuids"] == ["GPU-c", "GPU-d"]
    compact = observed[0]
    assert len(compact["pods"]["items"]) == 1
    pod = compact["pods"]["items"][0]
    assert set(pod) == {"spec", "status"}
    assert pod["spec"] == {
        key: active["spec"][key] for key in ("nodeName", "runtimeClassName")
    } | {
        "containers": [
            {
                "image": "test:tag",
                "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-a"}],
            }
        ],
        "initContainers": active["spec"]["initContainers"],
    }
    assert pod["status"] == {
        "phase": "Running",
        "containerStatuses": [{"imageID": "containerd://" + image}],
        "initContainerStatuses": active["status"]["initContainerStatuses"],
    }
    assert compact["images"] == {"images": [{"repoDigests": [image]}]}
    assert "discarded" not in json.dumps(compact)
    # The compact form must still fail closed for ambiguous NVIDIA allocations.
    pod["spec"]["initContainers"][0]["env"] = [{"name": "NVIDIA_VISIBLE_DEVICES"}]
    with pytest.raises(ValueError, match="ambiguous GPU allocation"):
        select_gpu.select([{"uuid": "GPU-c", "memory": 0}], [], compact["pods"])


@pytest.mark.parametrize("status", ["Stopped", "Succeeded", "Failed"])
def test_down_accepts_stop_error_only_after_confirming_terminal_job(
    state, tmp_path, status
):
    state["resources"] = {"dit": state["resources"]["dit"]}
    resource = state["resources"]["dit"]
    cloud = Mock(timeout=1)
    cloud.cli.side_effect = RuntimeError("Job is already terminal")
    cloud.get_job.return_value = job(resource, status)
    orch.down(state, tmp_path, cloud)
    cloud.cli.assert_called_once_with("pai-dlc", "StopJob", JobId=resource["job_id"])
    assert (
        cloud.get_job.call_count == 2
    )  # Ownership preflight, then terminal confirmation.
    assert resource["status"] == "deleted"
    assert orch.read(tmp_path / "state.json")["phase"] == "stopped"


def test_down_preserves_stop_error_when_owned_job_is_still_running(state, tmp_path):
    state["resources"] = {"dit": state["resources"]["dit"]}
    resource = state["resources"]["dit"]
    cloud = Mock(timeout=1)
    error = RuntimeError("Stop request denied")
    cloud.cli.side_effect = error
    cloud.get_job.return_value = job(resource, "Running")
    with pytest.raises(RuntimeError) as failure:
        orch.down(state, tmp_path, cloud)
    assert failure.value is error
    assert resource["status"] != "deleted"
    assert state["phase"] != "stopped"


@pytest.mark.parametrize("status", ["Stopped", "Succeeded", "Failed", "Running"])
def test_pai_missing_receipt_fails_fast_only_for_terminal_job(state, status):
    resource = state["resources"]["dit"]
    cloud = orch.Cloud()
    observed = job(resource, status)
    observed["Pods"] = [{"PodId": "dlc-owned-master-0"}]
    cloud.get_job = Mock(return_value=observed)
    cloud.cli = Mock(
        return_value={
            "JobId": resource["job_id"],
            "Logs": ['{"schema":1,"role":"dit","archive_sha256":"truncated'],
        }
    )
    if status == "Running":
        with pytest.raises(RuntimeError, match="receipt not available"):
            cloud.receipt(resource, "dit")
    else:
        with pytest.raises(
            ValueError, match="Owned PAI job terminated before ready receipt"
        ):
            cloud.receipt(resource, "dit")
    # A terminal Job with a complete receipt must still be inspectable.
    cloud.cli.return_value["Logs"] = [json.dumps(resource["receipt"])]
    assert cloud.receipt(resource, "dit") == resource["receipt"]


@pytest.mark.parametrize(
    "existing_gpu,worker_config",
    [
        (None, {"hardware_ids": ["GPU-one", "GPU-two"]}),
        (None, None),
        ("GPU-existing", {"hardware_ids": ["GPU-other"]}),
    ],
)
def test_truncated_runtime_receipt_recovers_complete_prefix(
    state, existing_gpu, worker_config
):
    receipt = {
        key: value
        for key, value in state["resources"]["dit"]["receipt"].items()
        if key != "gpu"
    }
    if existing_gpu is not None:
        receipt["gpu"] = existing_gpu
    receipt["worker_config"] = worker_config
    complete = {**receipt, "runtime": {"version": "x" * 3000}}
    if existing_gpu is None:
        complete["gpu"] = "unread tail"
    line = json.dumps(complete, separators=(",", ":"))
    candidate = line.encode()[:2048].decode()
    assert ',"runtime":{' in candidate
    with pytest.raises(ValueError):
        json.loads(candidate)
    parsed = orch.last_receipt("PAI log prefix " + candidate, "dit")
    if existing_gpu is None and isinstance(worker_config, dict):
        receipt["gpu"] = "GPU-one, GPU-two"
    assert parsed == receipt
    # Complete lines retain the normal parsing behavior.
    assert orch.last_receipt(line, "dit")["gpu"] == complete["gpu"]


@pytest.mark.parametrize(
    "candidate",
    [
        '{broken,"runtime":{',
        '{"schema":2,"role":"dit","archive_sha256":"a","runtime":{',
        '{"schema":1,"role":"encoder","archive_sha256":"a","runtime":{',
    ],
)
def test_truncated_receipt_still_rejects_invalid_prefix_schema_or_role(candidate):
    with pytest.raises(RuntimeError, match="receipt not available"):
        orch.last_receipt(candidate, "dit")


@pytest.mark.parametrize(
    "plan_ids,hash_kind,accepted",
    [
        (["__ALLOCATED_GPU_UUID__"], "original", True),
        (["__ALLOCATED_GPU_UUID__"], "allocated", True),
        (["GPU-pinned"], "allocated", False),
        (["__ALLOCATED_GPU_UUID__"], "other-change", False),
    ],
)
def test_wait_ready_accepts_only_original_or_allocated_gpu_config_hash(
    state, tmp_path, monkeypatch, plan_ids, hash_kind, accepted
):
    path = tmp_path / "encoder-0/role.json"
    plan = orch.read(path)
    plan["worker_config"] = {
        "hardware_ids": plan_ids,
        "role": "encoder",
        "service_count": 1,
    }
    orch.save(path, plan)
    before = path.read_bytes()
    receipt = state["resources"]["encoder"]["receipt"]
    receipt["gpu"] = "GPU-b, GPU-a, GPU-b"
    hashed = copy.deepcopy(plan)
    if hash_kind != "original":
        hashed["worker_config"]["hardware_ids"] = ["GPU-a", "GPU-b"]
    if hash_kind == "other-change":
        hashed["startup_timeout"] = 42
    receipt["role_config_sha256"] = hashlib.sha256(
        json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cloud = Mock(timeout=1)
    cloud.receipt.side_effect = lambda resource, role: resource["receipt"]
    monkeypatch.setattr(
        orch,
        "http",
        lambda url, cloud=None: dict(
            ready=True,
            role=next(
                r
                for r in orch.ROLES
                if state["resources"][r]["receipt"]["http"] + "/health" == url
            ),
        ),
    )
    if accepted:
        orch.wait_ready(state, tmp_path, cloud)
        assert state["phase"] == "ready"
    else:
        with pytest.raises(ValueError, match="role config identity mismatch"):
            orch.wait_ready(state, tmp_path, cloud)
    assert path.read_bytes() == before


def test_down_retires_attached_pool_processes_after_stop(state, tmp_path, monkeypatch):
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "o" * 64)
    state["attachment"] = {
        "identities": {
            role: {
                "process_id": f"{role}-proc",
                "boot_epoch": f"{role}-epoch",
                "service_id": f"{role}-proc/0",
            }
            for role in orch.ROLES
        }
    }
    calls = []
    monkeypatch.setattr(
        orch,
        "http",
        lambda url, body=None, token=None, cloud=None: (
            calls.append((url, body, token)) or {"ok": True}
        ),
    )
    cloud = Mock(timeout=1)
    cloud.pod.side_effect = lambda r: {"metadata": {"uid": r["uid"]}}
    cloud.get_job.side_effect = lambda _: job(state["resources"]["dit"], "Stopped")
    orch.down(state, tmp_path, cloud)
    assert len(calls) == 9
    for idx, role in enumerate(orch.ROLES):
        drain_call, stop_call, retire_call = calls[idx * 3 : idx * 3 + 3]
        assert drain_call[0].endswith("/v1/role-pool/drain")
        assert drain_call[1] == {
            "process_id": f"{role}-proc",
            "boot_epoch": f"{role}-epoch",
        }
        assert stop_call[0].endswith("/v1/role-pool/confirm-stopped")
        assert stop_call[1]["operator_id"] == "orchestrate-down"
        assert retire_call[0].endswith("/v1/role-pool/retire-process")
        assert retire_call[1]["stop_receipt"] == stop_call[1]["stop_receipt"]


def test_attach_in_place_activates_workers_without_recreating_pods_or_jobs(
    state, tmp_path, monkeypatch
):
    state["phase"] = "ready"
    state["selftest"] = {
        "status": "passed",
        "source": copy.deepcopy(state["source"]),
        "worker_ids": {
            r: state["resources"][r]["receipt"]["worker_id"] for r in orch.ROLES
        },
    }
    for c in state["spec"]["roles"].values():
        c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "operator")
    attached = []

    def request(url, payload=None, token=None, cloud=None):
        if "/v1/capacity" in url:
            return dict(backend="role_pool", profile="480")
        if "/get_connection_status" in url:
            return dict(state="available")
        if url.endswith("/managed/attach"):
            role = payload["worker_config"]["role"]
            attached.append((role, payload, token))
            return dict(
                service=dict(
                    worker_id=state["resources"][role]["receipt"]["worker_id"],
                    role=role,
                    service_id=role + "/0",
                ),
                boot_epoch=role + "-boot",
            )
        if url.endswith("/health"):
            role = next(
                r
                for r in orch.ROLES
                if state["resources"][r]["receipt"]["http"] + "/health" == url
            )
            return dict(ready=True, role=role)
        raise AssertionError(url)

    monkeypatch.setattr(orch, "http", request)
    down_mock = Mock()
    submit_mock = Mock()
    verify_mock = Mock()
    monkeypatch.setattr(orch, "down", down_mock)
    monkeypatch.setattr(orch, "submit", submit_mock)
    monkeypatch.setattr(orch, "verify_pool", verify_mock)
    orch.attach(state, tmp_path, Mock(timeout=1), execute=True, in_place=True)
    down_mock.assert_not_called()
    submit_mock.assert_not_called()
    verify_mock.assert_called_once()
    assert state["generation"] == 0
    assert state["attachment_mode"] == "in-place"
    assert [r for r, _, _ in attached] == list(orch.ROLES)
    for role in orch.ROLES:
        assert state["resources"][role]["receipt"]["worker_config"]["hardware_ids"] == [
            "GPU-" + role
        ]
    # Subsequent inspect (wait_ready) preserves in-place worker_config when raw log receipt is unchanged.
    cloud = Mock(timeout=1)

    def raw_receipt(resource, role):
        raw = copy.deepcopy(resource["receipt"])
        raw.pop("worker_config", None)
        return raw

    cloud.receipt.side_effect = raw_receipt
    orch.wait_ready(state, tmp_path, cloud)
    for role in orch.ROLES:
        assert state["resources"][role]["receipt"]["worker_config"]["hardware_ids"] == [
            "GPU-" + role
        ]


def test_reusing_existing_encoder_decoder_services_and_multi_service(
    spec, tmp_path, monkeypatch
):
    # Configure standalone DiT reusing existing multi-service Encoder & Decoder.
    spec["existing_services"] = {
        "encoder": {
            "http": "http://10.0.0.121:31001",
            "worker_id": "encoder-shared",
            "service_index": 1,
            "service_count": 2,
            "gpu": "GPU-shared-encoder",
            "boot_epoch": "enc-epoch",
        },
        "decoder": {
            "http": "http://10.0.0.121:36001",
            "worker_id": "decoder-shared",
            "service_index": 1,
            "service_count": 2,
            "gpu": "GPU-shared-decoder",
            "boot_epoch": "dec-epoch",
        },
    }
    spec["roles"] = {"dit": spec["roles"]["dit"]}
    spec["roles"]["dit"]["secret_files"] = {
        k: "/secrets/" + k for k in launch.SECRET_ENV
    }
    checked = orch.validate_spec(spec)
    assert orch.owned_roles(checked) == ("dit",)

    manifest = dict(
        schema=1,
        commit="b" * 40,
        sha256="c" * 64,
        uri="oss://leap-world-shanghai/test/"
        + "b" * 40
        + "/"
        + "c" * 64
        + "/source.tar",
        tools={
            n: source.digest(Path(__file__).with_name(n))
            for n in ("launch.py", "source.py", "select_gpu.py", "orchestrate.py")
        },
    )
    run_state = dict(
        schema=1,
        run_id="e" * 32,
        target=copy.deepcopy(orch.TARGET),
        phase="planned",
        generation=0,
        source=manifest,
        spec=checked,
        resources={
            role: orch.reused_resource("e" * 32, checked["model_id"], role, entry)
            for role, entry in checked["existing_services"].items()
        },
    )
    orch.render_roles(run_state, tmp_path)
    assert (tmp_path / "dit-0/pai.json").exists()
    assert not (tmp_path / "encoder-0").exists()
    assert not (tmp_path / "decoder-0").exists()

    (tmp_path / "source").mkdir()
    archive = tmp_path / "source/source.tar"
    archive.write_bytes(b"archive")
    run_state["source"]["sha256"] = source.digest(archive)
    dit_role_cfg = orch.read(tmp_path / "dit-0/role.json")
    dit_receipt = dict(
        schema=1,
        role="dit",
        status="ready",
        commit=manifest["commit"],
        archive_sha256=run_state["source"]["sha256"],
        model_id="model1027",
        asset_receipt_sha256="a" * 64,
        worker_id="dit-owned",
        role_config_sha256=hashlib.sha256(
            json.dumps(dit_role_cfg, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        http="http://10.0.0.122:33000",
        gpu="GPU-dit-owned, H200, 140000, 0",
    )
    cloud = Mock(timeout=1)
    cloud.submit_job.return_value = "dlc-dit-only"
    cloud.receipt.return_value = dit_receipt

    def http_handler(url, payload=None, token=None, cloud=None):
        if url.endswith("/health"):
            role = (
                "encoder"
                if ":31001/" in url
                else ("decoder" if ":36001/" in url else "dit")
            )
            return dict(ready=True, role=role)
        if "/get_connection_status" in url:
            if ":31001/" in url:
                return dict(
                    state="available",
                    service=dict(worker_id="encoder-shared", role="encoder"),
                )
            if ":36001/" in url:
                return dict(
                    state="available",
                    service=dict(worker_id="decoder-shared", role="decoder"),
                )
            return dict(
                state="available",
                service=dict(worker_id="dit-owned", role="dit"),
            )
        if "/v1/capacity" in url:
            return dict(
                backend="role_pool",
                profile="480",
                roles={"model_group": {"free_slots": 2}},
            )
        if url.endswith("/managed/attach") or url.endswith("/managed/identity"):
            assert "10.0.0.122:33000" in url
            return dict(
                service=dict(
                    worker_id="dit-owned", role="dit", service_id="dit-owned/0"
                ),
                boot_epoch="dit-epoch",
            )
        if url.endswith("/v1/role-pool/process-status"):
            assert payload["process_id"] == "dit-owned"
            return dict(
                process=dict(
                    manifest=copy.deepcopy(
                        run_state["resources"]["dit"]["receipt"]["worker_config"]
                    ),
                    state="ready",
                    seen_at=orch.time.time(),
                )
            )
        raise AssertionError(url)

    monkeypatch.setattr(orch, "http", http_handler)
    orch.submit(run_state, tmp_path, cloud)
    # No H100 staging or K3s Pod creation occurs when Encoder & Decoder are reused.
    cloud.stage_h100_source.assert_not_called()
    cloud.eflo.assert_not_called()
    cloud.submit_job.assert_called_once()
    assert run_state["phase"] == "ready"
    assert orch.load_state(tmp_path / "state.json")["run_id"] == run_state["run_id"]

    # Attach only attaches owned DiT and never restarts or mutates reused Encoder/Decoder.
    run_state["selftest"] = dict(
        status="passed",
        source=copy.deepcopy(run_state["source"]),
        worker_ids={
            "encoder": "encoder-shared",
            "dit": "dit-owned",
            "decoder": "decoder-shared",
        },
    )
    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "operator")
    configs = orch.attach(run_state, tmp_path, cloud, execute=False)
    assert set(configs) == {"dit"}
    orch.attach(run_state, tmp_path, cloud, execute=True, in_place=True)
    assert run_state["attachment"]["identities"]["encoder"]["reused"] is True
    assert run_state["attachment"]["identities"]["decoder"]["reused"] is True
    assert run_state["attachment"]["identities"]["dit"]["process_id"] == "dit-owned"

    # Down only stops the owned DiT job and never touches reused Encoder/Decoder.
    cloud.get_job.side_effect = lambda _: {
        **job(run_state["resources"]["dit"], "Stopped"),
        **run_state["resources"]["dit"]["expected_job"],
    }
    monkeypatch.delenv("ROLE_POOL_OPERATOR_TOKEN", raising=False)
    orch.down(run_state, tmp_path, cloud)
    cloud.cli.assert_called_once_with("pai-dlc", "StopJob", JobId="dlc-dit-only")
    cloud.eflo.assert_not_called()
    assert (
        run_state["resources"]["encoder"]["reused"] is True
        and run_state["resources"]["encoder"]["status"] == "ready"
    )
    assert (
        run_state["resources"]["decoder"]["reused"] is True
        and run_state["resources"]["decoder"]["status"] == "ready"
    )
    assert run_state["resources"]["dit"]["status"] == "deleted"


def test_b300_three_role_and_multi_service_orchestration(spec, tmp_path):
    for role in orch.ROLES:
        spec["roles"][role]["profile"] = "b300"
        if role != "dit":
            spec["roles"][role]["service_count"] = 2
            spec["pool"]["capacity"][role] = {"slots": 2}
    spec["sites"] = {
        "b300": dict(
            schema=1,
            image="test@sha256:" + "c" * 64,
            workspace_id="b300-ws",
            job_spec=dict(
                Type="Worker",
                PodCount=1,
                QuotaId="quota-b300",
                RestartPolicy="Never",
                ResourceConfig={"GPU": "1"},
            ),
            data_sources=[],
        )
    }
    checked = orch.validate_spec(spec)
    manifest = dict(
        schema=1,
        commit="b" * 40,
        sha256="c" * 64,
        uri="oss://leap-world-shanghai/test/"
        + "b" * 40
        + "/"
        + "c" * 64
        + "/source.tar",
        tools={
            n: source.digest(Path(__file__).with_name(n))
            for n in ("launch.py", "source.py", "select_gpu.py", "orchestrate.py")
        },
    )
    run_state = dict(
        schema=1,
        run_id="f" * 32,
        target=copy.deepcopy(orch.TARGET),
        phase="planned",
        generation=0,
        source=manifest,
        spec=checked,
        resources={},
    )
    orch.render_roles(run_state, tmp_path)
    for role in orch.ROLES:
        assert (tmp_path / f"{role}-0/pai.json").exists()
        role_cfg = orch.read(tmp_path / f"{role}-0/role.json")
        assert role_cfg["hardware"] == {
            "capability": [10, 3],
            "minimum_memory_gib": 240,
        }
        run_state["resources"][role] = dict(
            run_id=run_state["run_id"],
            generation=0,
            name=f"sglang-{role}-" + "f" * 12 + "-0",
            job_id=f"dlc-{role}",
            status="ready",
            receipt=dict(
                schema=1,
                role=role,
                status="ready",
                worker_id=f"{role}-b300",
                http=f"http://10.0.0.19{orch.ROLES.index(role) + 1}:31000",
                gpu=f"GPU-b300-{role}, B300, 274113, 0",
            ),
        )
    configs = orch.worker_configs(run_state)
    assert configs["encoder"]["service_count"] == 2
    assert configs["decoder"]["service_count"] == 2
    assert configs["dit"]["service_count"] == 1


def test_down_retries_coordinator_confirm_and_retire_after_pod_deleted(
    state, tmp_path, monkeypatch
):
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "operator-secret-token")
    state["attachment"] = dict(
        identities={
            role: dict(
                process_id=f"{role}-worker",
                boot_epoch=f"{role}-boot",
                service_id=f"{role}-worker/0",
            )
            for role in orch.ROLES
        }
    )
    cloud = Mock(timeout=1)
    deleted_pods = set()

    def pod_lookup(resource):
        if resource["name"] in deleted_pods:
            raise RuntimeError("Error from server (NotFound): pods not found")
        return {"metadata": {"uid": resource["uid"]}}

    def eflo_exec(script):
        for role in ("encoder", "decoder"):
            name = state["resources"][role]["name"]
            if f"pods/{name}" in script and "kubectl delete --raw" in script:
                if name in deleted_pods:
                    raise RuntimeError("Pod already deleted")
                deleted_pods.add(name)
                return ""
        raise AssertionError(script)

    cloud.pod.side_effect = pod_lookup
    cloud.eflo.side_effect = eflo_exec
    cloud.get_job.side_effect = lambda _: job(state["resources"]["dit"], "Stopped")

    calls = []
    fail_stage = {"mode": "confirm"}

    def coordinator_http(url, payload=None, token=None, cloud=None):
        calls.append((url, payload["process_id"]))
        if (
            fail_stage["mode"] == "confirm"
            and url.endswith("/v1/role-pool/confirm-stopped")
            and payload["process_id"] == "encoder-worker"
        ):
            raise RuntimeError("transient confirm-stopped failure")
        if (
            fail_stage["mode"] == "retire"
            and url.endswith("/v1/role-pool/retire-process")
            and payload["process_id"] == "encoder-worker"
        ):
            raise RuntimeError("transient retire-process failure")
        return {"ok": True}

    monkeypatch.setattr(orch, "http", coordinator_http)

    # Attempt 1: encoder Pod is deleted, then confirm-stopped fails.
    with pytest.raises(RuntimeError, match="transient confirm-stopped failure"):
        orch.down(state, tmp_path, cloud)
    persisted = orch.read(tmp_path / "state.json")
    assert persisted["resources"]["encoder"]["status"] == "stopped-unretired"
    assert not persisted["resources"]["encoder"].get("stop_confirmed")
    assert state["resources"]["encoder"]["name"] in deleted_pods

    # Attempt 2: encoder Pod is already deleted; confirm-stopped succeeds, retire-process fails.
    fail_stage["mode"] = "retire"
    with pytest.raises(RuntimeError, match="transient retire-process failure"):
        orch.down(state, tmp_path, cloud)
    persisted = orch.read(tmp_path / "state.json")
    assert persisted["resources"]["encoder"]["status"] == "stopped-unretired"
    assert persisted["resources"]["encoder"]["stop_confirmed"] is True
    assert not persisted["resources"]["encoder"].get("retired")

    # Attempt 3: encoder skips drain/delete/confirm-stopped, completes retire-process, and remaining roles finish.
    fail_stage["mode"] = None
    orch.down(state, tmp_path, cloud)
    persisted = orch.read(tmp_path / "state.json")
    assert persisted["phase"] == "stopped"
    for role in orch.ROLES:
        assert persisted["resources"][role]["status"] == "deleted"
        assert persisted["resources"][role]["retired"] is True


def test_in_place_attach_partial_success_idempotent_retry_and_reconciliation(
    state, tmp_path, monkeypatch
):
    state["phase"] = "ready"
    state["selftest"] = dict(
        status="passed",
        source=state["source"],
        worker_ids={
            r: state["resources"][r]["receipt"]["worker_id"] for r in orch.ROLES
        },
    )
    for role in orch.ROLES:
        state["spec"]["roles"][role]["secret_files"] = {
            k: "/secrets/" + k for k in launch.SECRET_ENV
        }
    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "operator-secret")

    attached_roles = set()
    fail_dit_attach = {"enabled": True}
    tamper_boot_epoch = {"enabled": False}

    def http_handler(url, payload=None, token=None, cloud=None):
        if "/v1/capacity" in url:
            return dict(
                backend="role_pool",
                profile="480",
                roles={"model_group": {"free_slots": 1}},
            )
        if "/get_connection_status" in url:
            role = (
                "encoder"
                if ":31000/" in url
                else ("dit" if ":33000/" in url else "decoder")
            )
            if role in attached_roles:
                raise RuntimeError("HTTP 403 MANAGED_CONTROL_REQUIRED")
            return dict(state="available")
        if url.endswith("/managed/observe"):
            role = (
                "encoder"
                if ":31000/" in url
                else ("dit" if ":33000/" in url else "decoder")
            )
            assert role in attached_roles
            assert payload["boot_epoch"] == f"{role}-boot"
            return dict(state="idle", owner=None, restart_required=False)
        if url.endswith("/managed/attach"):
            role = payload["worker_config"]["role"]
            if role == "dit" and fail_dit_attach["enabled"]:
                raise RuntimeError("dit attach network timeout")
            attached_roles.add(role)
            return dict(
                service=dict(
                    worker_id=role + "-worker",
                    role=role,
                    service_id=role + "-worker/0",
                ),
                boot_epoch=(
                    "tampered-boot"
                    if tamper_boot_epoch["enabled"] and role == "encoder"
                    else role + "-boot"
                ),
                control_epoch=payload["control_epoch"],
                deployment=dict(hardware_ids=payload["worker_config"]["hardware_ids"]),
            )
        if url.endswith("/managed/identity"):
            role = (
                "encoder"
                if ":31000/" in url
                else ("dit" if ":33000/" in url else "decoder")
            )
            return dict(
                service=dict(
                    worker_id=role + "-worker",
                    role=role,
                    service_id=role + "-worker/0",
                ),
                boot_epoch=(
                    "wrong-boot"
                    if tamper_boot_epoch["enabled"] and role in ("encoder", "dit")
                    else role + "-boot"
                ),
                control_epoch=state["spec"]["pool"]["control_epoch"],
            )
        if url.endswith("/v1/role-pool/process-status"):
            role = payload["process_id"].removesuffix("-worker")
            manifest = copy.deepcopy(
                state["resources"][role]["receipt"]["worker_config"]
            )
            manifest.update(
                process_id=payload["process_id"],
                boot_epoch=payload["boot_epoch"],
                role=role,
            )
            return dict(
                process=dict(
                    manifest=manifest,
                    state="ready",
                    seen_at=orch.time.time(),
                )
            )
        raise AssertionError(url)

    monkeypatch.setattr(orch, "http", http_handler)

    # Attempt 1: encoder attaches in-place and persists identity; dit fails.
    with pytest.raises(RuntimeError, match="dit attach network timeout"):
        orch.attach(state, tmp_path, Mock(timeout=1), execute=True, in_place=True)

    persisted = orch.read(tmp_path / "state.json")
    assert persisted["attachment_mode"] == "in-place"
    assert persisted["attachment"]["identities"]["encoder"] == dict(
        process_id="encoder-worker",
        boot_epoch="encoder-boot",
        service_id="encoder-worker/0",
    )
    assert "worker_config" in persisted["resources"]["encoder"]["receipt"]
    assert "dit" not in persisted["attachment"]["identities"]

    # If a retried attach sees a changed boot_epoch on already-attached encoder, fail closed.
    tamper_boot_epoch["enabled"] = True
    with pytest.raises(ValueError, match="mismatched worker identity"):
        orch.attach(state, tmp_path, Mock(timeout=1), execute=True, in_place=True)

    # Attempt 2: idempotent retry uses /managed/observe for encoder, attaches dit & decoder, and reconciles pool.
    tamper_boot_epoch["enabled"] = False
    fail_dit_attach["enabled"] = False
    orch.attach(state, tmp_path, Mock(timeout=1), execute=True, in_place=True)
    assert set(state["attachment"]["identities"]) == set(orch.ROLES)

    # Reconciliation rejects boot_epoch drift between attach and verify_pool.
    tamper_boot_epoch["enabled"] = True
    with pytest.raises(ValueError, match="Managed identity does not match"):
        orch.verify_pool(state, tmp_path)


def test_generic_non_production_target_and_minimal_probe(tmp_path):
    b300_target = {
        "environment": "test-b300-thailand",
        "region": "ap-southeast-7",
        "workspace_id": "ws-th-b300",
        "quota_id": "quota-th-b300",
        "coordinator_url": "http://10.20.0.10:18181",
        "pool_id": "test-b300-pool",
        "room_url": "http://10.20.0.11:8092",
        "wtp_url": "https://test-b300.example.com",
        "oss_prefix": "oss://leap-worldmodel-thailand-single-az/sglang/orchestration",
    }
    minimal = {
        "schema": 1,
        "target": b300_target,
        "topology": "b300",
        "model_id": "model1027",
        "roles": {
            "encoder": {
                "mode": "reuse",
                "http": "http://10.20.0.50:31000",
                "worker_id": "b300-enc-shared",
                "service_index": 0,
                "service_count": 2,
                "gpu": "GPU-b300-enc",
                "boot_epoch": "enc-b300-boot",
                "pai": True,
            },
            "decoder": {"mode": "create", "service_count": 2},
            "dit": {"mode": "create"},
        },
    }
    cloud = Mock()
    cloud.cli.return_value = {
        "Jobs": [
            {
                "JobId": "dlc-b300-1",
                "DisplayName": "sglang-dit",
                "Status": "Running",
                "UserId": "1001",
                "ResourceId": "quota-th-b300",
            }
        ],
        "TotalCount": 1,
    }
    probe_out = tmp_path / "b300-probe"
    evidence = orch.probe(minimal, probe_out, cloud)
    cloud.eflo.assert_not_called()
    assert evidence["target"]["environment"] == "test-b300-thailand"
    assert len(evidence["h200_jobs"]) == 1
    frozen_spec = orch.read(probe_out / "spec.json")
    assert orch.owned_roles(frozen_spec) == ("dit", "decoder")
    assert "encoder" in frozen_spec["existing_services"]
    assert frozen_spec["roles"]["decoder"]["service_count"] == 2


def test_per_role_mode_reuse_and_lease_only_release_on_down(
    spec, tmp_path, monkeypatch
):
    spec["roles"]["encoder"] = {
        "mode": "reuse",
        "http": "http://10.0.0.121:31000",
        "worker_id": "enc-reused-worker",
        "service_index": 0,
        "service_count": 2,
        "gpu": "GPU-enc-reused",
        "boot_epoch": "enc-reused-epoch",
    }
    spec["roles"]["decoder"]["mode"] = "create"
    spec["roles"]["dit"]["mode"] = "create"
    checked = orch.validate_spec(spec)
    assert orch.owned_roles(checked) == ("dit", "decoder")
    assert "encoder" in checked["existing_services"]

    manifest = dict(
        schema=1,
        commit="b" * 40,
        sha256="c" * 64,
        uri="oss://leap-world-shanghai/test/source.tar",
        tools={
            n: source.digest(Path(__file__).with_name(n))
            for n in ("launch.py", "source.py", "select_gpu.py", "orchestrate.py")
        },
    )
    run_state = dict(
        schema=1,
        run_id="a" * 32,
        target=copy.deepcopy(orch.TARGET),
        phase="ready",
        generation=0,
        source=manifest,
        spec=checked,
        resources={
            "encoder": {
                **orch.reused_resource(
                    "a" * 32,
                    "model1027",
                    "encoder",
                    checked["existing_services"]["encoder"],
                ),
                "active_lease": {"lease_id": "lease-run-123"},
            },
            "decoder": {
                "run_id": "a" * 32,
                "generation": 0,
                "name": "sglang-decoder-" + "a" * 12 + "-0",
                "namespace": "sglang-test",
                "uid": "dec-uid",
                "status": "ready",
                "receipt": {
                    "schema": 1,
                    "role": "decoder",
                    "status": "ready",
                    "worker_id": "dec-owned",
                    "http": "http://10.0.0.121:36000",
                    "gpu": "GPU-dec-owned",
                },
            },
            "dit": {
                "run_id": "a" * 32,
                "generation": 0,
                "name": "sglang-dit-" + "a" * 12 + "-0",
                "job_id": "dlc-dit-owned",
                "status": "ready",
                "receipt": {
                    "schema": 1,
                    "role": "dit",
                    "status": "ready",
                    "worker_id": "dit-owned",
                    "http": "http://10.0.0.122:33000",
                    "gpu": "GPU-dit-owned",
                },
            },
        },
    )
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "op-token")
    http_calls = []
    monkeypatch.setattr(
        orch,
        "http",
        lambda url, payload=None, token=None, cloud=None: (
            http_calls.append((url, payload, token)) or {"ok": True}
        ),
    )
    cloud = Mock(timeout=1)
    cloud.pod.return_value = {"metadata": {"uid": "dec-uid"}}
    cloud.get_job.return_value = job(run_state["resources"]["dit"], "Stopped")
    orch.down(run_state, tmp_path, cloud)
    # Reused encoder only had its lease released; never drained, stopped, or retired.
    assert any(
        url.endswith("/v1/role-pool/release-lease")
        and payload == {"lease_id": "lease-run-123", "run_id": "a" * 32}
        for url, payload, _ in http_calls
    )
    assert not any(
        payload and payload.get("process_id") == "enc-reused-worker"
        for _, payload, _ in http_calls
    )
    assert run_state["resources"]["encoder"]["lease_released"] is True
    assert run_state["resources"]["encoder"]["status"] == "reused"
    assert run_state["resources"]["decoder"]["status"] == "deleted"
    assert run_state["resources"]["dit"]["status"] == "deleted"


def test_confirm_execution_requires_yes_when_non_interactive(monkeypatch):
    fake_stdin = Mock()
    fake_stdin.isatty.return_value = False
    monkeypatch.setattr(orch.sys, "stdin", fake_stdin)
    with pytest.raises(ValueError, match="explicit --yes"):
        orch.confirm_execution("up", {"deployment_id": "deploy-1"}, yes=False)
    assert orch.confirm_execution("up", {"deployment_id": "deploy-1"}, yes=True) is True


def test_deployment_summary_history_show_diff_and_rollback(
    state, tmp_path, monkeypatch
):
    history_dir = tmp_path / "shared-history"
    state["target"]["history_dir"] = str(history_dir)
    state["deployment_id"] = "deploy-20261110T010000Z-aaaa1111"
    for c in state["spec"]["roles"].values():
        c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
    state["phase"] = "ready"
    summary_v1 = orch.write_deployment_summary(state, tmp_path / "run1")
    orch.save_history_record(state, tmp_path / "run1", history_dir=history_dir)
    assert (tmp_path / "run1/deployment.json").exists()
    assert (tmp_path / "run1/DEPLOYMENT.md").exists()
    assert summary_v1["deployment_id"] == "deploy-20261110T010000Z-aaaa1111"
    assert summary_v1["roles"]["dit"]["launch_profile"]["effective_argv"] == "unknown"
    assert summary_v1["roles"]["dit"]["launch_profile"]["argv_status"] == "unknown"
    assert (
        "sglang.multimodal_gen.tools.zing_h3_profile_launcher"
        in summary_v1["roles"]["dit"]["launch_profile"]["planned_argv"]
    )

    # Create a second deployment record with reused encoder and updated commit.
    state_v2 = copy.deepcopy(state)
    state_v2["deployment_id"] = "deploy-20261110T020000Z-bbbb2222"
    state_v2["run_id"] = "e" * 32
    state_v2["source"]["commit"] = "9" * 40
    state_v2["spec"]["existing_services"] = {
        "encoder": {
            "http": "http://10.0.0.121:31000",
            "worker_id": "encoder-shared",
            "service_index": 0,
            "service_count": 1,
            "gpu": "GPU-encoder",
            "boot_epoch": "enc-boot",
        }
    }
    state_v2["spec"]["roles"].pop("encoder", None)
    state_v2["resources"]["encoder"] = orch.reused_resource(
        "e" * 32,
        "model1027",
        "encoder",
        state_v2["spec"]["existing_services"]["encoder"],
    )
    for role in ("dit", "decoder"):
        state_v2["resources"][role]["run_id"] = "e" * 32
        state_v2["resources"][role]["name"] = f"sglang-{role}-" + "e" * 12 + "-0"
    run2_dir = tmp_path / "run2"
    orch.save(run2_dir / "state.json", state_v2)
    orch.save_history_record(state_v2, run2_dir, history_dir=history_dir)

    items = orch.list_history(history_dir)
    assert [i["deployment_id"] for i in items] == [
        "deploy-20261110T010000Z-aaaa1111",
        "deploy-20261110T020000Z-bbbb2222",
    ]
    diff = orch.diff_deployments(
        history_dir,
        "deploy-20261110T010000Z-aaaa1111",
        "deploy-20261110T020000Z-bbbb2222",
    )
    assert diff["source_changed"] is True
    assert "encoder" in diff["roles"]
    assert diff["roles"]["encoder"]["from"]["mode"] == "create"
    assert diff["roles"]["encoder"]["to"]["mode"] == "reuse"

    # Receipt-only or incomplete records must be rejected for rollback.
    with pytest.raises(ValueError, match="full rollback_bundle is required"):
        orch.validate_rollback_record({"schema": 1, "deployment_id": "only-summary"})
    tampered = orch.show_deployment(history_dir, "deploy-20261110T010000Z-aaaa1111")
    tampered["rollback_bundle"]["images"]["h100"] = "unpinned:latest"
    with pytest.raises(ValueError, match="pinned digest"):
        orch.validate_rollback_record(tampered)

    # Plaintext secrets in role environment must be rejected when saving history.
    bad_state = copy.deepcopy(state)
    bad_state["spec"]["roles"]["dit"]["environment"]["ROLE_POOL_SIGNING_KEY"] = "leaked"
    with pytest.raises(ValueError, match="Plaintext secrets are forbidden"):
        orch.build_history_record(bad_state)

    # Offline rollback plan protects reused encoder in current state and links rollback_of.
    rb_dir = tmp_path / "rollback-run"
    cloud = Mock(timeout=1)
    planned_rb = orch.rollback_deployment(
        history_dir,
        "deploy-20261110T010000Z-aaaa1111",
        rb_dir,
        cloud,
        execute=False,
        current_state_path=run2_dir / "state.json",
    )
    assert planned_rb["rollback_of"] == "deploy-20261110T010000Z-aaaa1111"
    assert planned_rb["rollback_plan"]["current_reused_roles_protected"] == ["encoder"]
    assert set(planned_rb["rollback_plan"]["current_owned_roles_to_stop"]) == {
        "dit",
        "decoder",
    }
    assert planned_rb["rollback_plan"]["preserves_shared_caches"] is True
    assert planned_rb["rollback_plan"]["requires_previous_gpu_or_pod_uid"] is False

    # Execute rollback with partial failure recovery:
    # Attempt 1 stops current owned resources, then fails during submit.
    down_calls = []
    monkeypatch.setattr(
        orch,
        "down",
        lambda st, d, c: down_calls.append(st["run_id"]),
    )
    # Simulate an interrupted partial source directory to verify rollback prepares cleanly.
    (rb_dir / "source").mkdir(parents=True, exist_ok=True)
    archive_bytes = b"pinned-archive"
    expected_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    planned_rb["source"]["sha256"] = expected_sha256
    orch.save(rb_dir / "state.json", planned_rb)
    rec1 = orch.show_deployment(history_dir, "deploy-20261110T010000Z-aaaa1111")
    rec1["rollback_bundle"]["source"]["sha256"] = expected_sha256
    orch.save(history_dir / "deploy-20261110T010000Z-aaaa1111.json", rec1)

    def fake_prepare(repo_arg, commit_arg, out_dir, oss_prefix_arg):
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "source.tar").write_bytes(archive_bytes)
        return {"sha256": expected_sha256}

    monkeypatch.setattr(orch, "prepare", fake_prepare)

    fail_submit = {"once": True}

    def fake_submit(st, d, c):
        if fail_submit["once"]:
            fail_submit["once"] = False
            raise RuntimeError("transient submit failure during rollback")
        st["phase"] = "ready"
        orch.save(d / "state.json", st)

    monkeypatch.setattr(orch, "submit", fake_submit)
    with pytest.raises(RuntimeError, match="transient submit failure"):
        orch.rollback_deployment(
            history_dir,
            "deploy-20261110T010000Z-aaaa1111",
            rb_dir,
            cloud,
            execute=True,
            yes=True,
            current_state_path=run2_dir / "state.json",
        )
    assert down_calls == ["e" * 32]
    assert orch.read(rb_dir / "state.json")["rollback_stopped_current"] is True

    # Attempt 2 resumes rollback without re-running down on current_state, and completes.
    completed_rb = orch.rollback_deployment(
        history_dir,
        "deploy-20261110T010000Z-aaaa1111",
        rb_dir,
        cloud,
        execute=True,
        yes=True,
        current_state_path=run2_dir / "state.json",
    )
    assert down_calls == ["e" * 32]
    assert completed_rb["phase"] == "ready"
    assert completed_rb["rollback_of"] == "deploy-20261110T010000Z-aaaa1111"
    persisted_rb = orch.show_deployment(history_dir, completed_rb["deployment_id"])
    assert persisted_rb["rollback_of"] == "deploy-20261110T010000Z-aaaa1111"


def test_managed_reused_services_coordinator_selftest_and_lease_release(
    spec, tmp_path, monkeypatch
):
    spec["roles"]["dit"]["secret_files"] = {
        name: "/secrets/" + name for name in launch.SECRET_ENV
    }
    spec["roles"]["encoder"] = {
        "mode": "reuse",
        "http": "http://10.0.0.121:31000",
        "worker_id": "encoder-managed",
        "gpu": "GPU-enc-managed",
        "boot_epoch": "enc-boot-1",
    }
    spec["roles"]["decoder"] = {
        "mode": "reuse",
        "http": "http://10.0.0.121:29210",
        "worker_id": "decoder-managed",
        "gpu": "GPU-dec-managed",
        "boot_epoch": "dec-boot-1",
    }
    checked = orch.validate_spec(spec)
    manifest = dict(
        schema=1,
        commit="b" * 40,
        sha256="c" * 64,
        uri="oss://leap-world-shanghai/test/"
        + "b" * 40
        + "/"
        + "c" * 64
        + "/source.tar",
        tools={
            n: source.digest(Path(__file__).with_name(n))
            for n in ("launch.py", "source.py", "select_gpu.py", "orchestrate.py")
        },
    )
    run_state = dict(
        schema=1,
        run_id="f" * 32,
        target=copy.deepcopy(orch.TARGET),
        phase="planned",
        generation=0,
        source=manifest,
        spec=checked,
        resources={
            role: orch.reused_resource("f" * 32, checked["model_id"], role, entry)
            for role, entry in checked["existing_services"].items()
        },
    )
    orch.render_roles(run_state, tmp_path)
    (tmp_path / "source").mkdir()
    archive = tmp_path / "source/source.tar"
    archive.write_bytes(b"archive")
    run_state["source"]["sha256"] = source.digest(archive)
    dit_role_cfg = orch.read(tmp_path / "dit-0/role.json")
    dit_receipt = dict(
        schema=1,
        role="dit",
        status="ready",
        commit=manifest["commit"],
        archive_sha256=run_state["source"]["sha256"],
        model_id="model1027",
        asset_receipt_sha256="a" * 64,
        worker_id="dit-owned",
        role_config_sha256=hashlib.sha256(
            json.dumps(dit_role_cfg, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        http="http://10.0.0.122:33000",
        gpu="GPU-dit-owned, H200, 140000, 0",
    )
    cloud = Mock(timeout=1)
    cloud.submit_job.return_value = "dlc-dit-managed-test"
    cloud.receipt.return_value = dit_receipt

    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "o" * 32)
    monkeypatch.setenv("WHITEBOX_ROOM_TOKEN", "r" * 32)

    admit_calls = []
    close_calls = []

    def http_handler(url, payload=None, token=None, cloud=None):
        if url.endswith("/health"):
            role = (
                "encoder"
                if ":31000/" in url
                else ("decoder" if ":29210/" in url else "dit")
            )
            return dict(ready=True, role=role)
        if "/get_connection_status" in url:
            if ":31000/" in url or ":29210/" in url:
                raise RuntimeError("HTTP 403 MANAGED_CONTROL_REQUIRED")
            return dict(
                state="available",
                service=dict(worker_id="dit-owned", role="dit"),
            )
        if url.endswith("/managed/identity") or url.endswith("/managed/attach"):
            if ":31000/" in url:
                return dict(
                    service=dict(
                        worker_id="encoder-managed",
                        role="encoder",
                        service_id="encoder-managed/0",
                    ),
                    boot_epoch="enc-boot-1",
                    control_epoch=checked["pool"]["control_epoch"],
                )
            if ":29210/" in url:
                return dict(
                    service=dict(
                        worker_id="decoder-managed",
                        role="decoder",
                        service_id="decoder-managed/0",
                    ),
                    boot_epoch="dec-boot-1",
                    control_epoch=checked["pool"]["control_epoch"],
                )
            return dict(
                service=dict(
                    worker_id="dit-owned", role="dit", service_id="dit-owned/0"
                ),
                boot_epoch="dit-boot-1",
                control_epoch=checked["pool"]["control_epoch"],
            )
        if url.endswith("/v1/role-pool/process-status"):
            pid = payload["process_id"]
            if pid == "dit-owned":
                manifest_cfg = copy.deepcopy(
                    run_state["resources"]["dit"]["receipt"]["worker_config"]
                )
                manifest_cfg["process_id"] = "dit-owned"
                manifest_cfg["boot_epoch"] = "dit-boot-1"
            else:
                manifest_cfg = dict(process_id=pid, boot_epoch=payload["boot_epoch"])
            return dict(
                process=dict(
                    manifest=manifest_cfg,
                    state="ready",
                    seen_at=orch.time.time(),
                )
            )
        if "/v1/capacity" in url:
            return dict(
                backend="role_pool",
                profile="480",
                roles={"model_group": {"free_slots": 2}},
            )
        if url.endswith("/v1/role-pool/admit"):
            admit_calls.append(payload)
            dit_wid = "other-dit" if len(admit_calls) == 1 else "dit-owned"
            return dict(
                state="bound",
                allocation_id=f"alloc-{len(admit_calls)}",
                group_id=f"group-{len(admit_calls)}",
                topology=dict(
                    encoder=dict(
                        worker_id="encoder-managed",
                        service_id="encoder-managed/0",
                        endpoints={},
                    ),
                    dit=dict(
                        worker_id=dit_wid,
                        service_id=f"{dit_wid}/0",
                        endpoints={},
                    ),
                    decoder=dict(
                        worker_id="decoder-managed",
                        service_id="decoder-managed/0",
                        endpoints={},
                    ),
                ),
            )
        if url.endswith("/v1/role-pool/close"):
            close_calls.append(payload)
            return dict(
                state="released",
                roles=dict(encoder="released", dit="released", decoder="released"),
            )
        raise AssertionError(url)

    monkeypatch.setattr(orch, "http", http_handler)
    orch.submit(run_state, tmp_path, cloud)
    assert run_state["phase"] == "ready"
    assert run_state["resources"]["encoder"]["receipt"]["managed"] is True
    assert run_state["resources"]["decoder"]["receipt"]["managed"] is True

    import asyncio

    asyncio.run(orch.selftest(run_state, tmp_path, cloud=cloud))
    assert run_state["selftest"]["status"] == "passed"
    assert run_state["selftest"]["scope"] == "coordinator-lease-build-quiesce-release"
    assert len(admit_calls) == 2
    assert len(close_calls) == 2

    # Test down releasing an active Coordinator lease on a reused service via /v1/role-pool/close.
    run_state["resources"]["encoder"]["active_lease"] = dict(
        admission_id="adm-active", allocation_id="alloc-active"
    )
    cloud.get_job.side_effect = lambda _: {
        **job(run_state["resources"]["dit"], "Stopped"),
        **run_state["resources"]["dit"]["expected_job"],
    }
    monkeypatch.delenv("ROLE_POOL_OPERATOR_TOKEN", raising=False)
    orch.down(run_state, tmp_path, cloud)
    assert close_calls[-1]["admission_id"] == "adm-active"
    assert close_calls[-1]["allocation_id"] == "alloc-active"
    assert run_state["resources"]["encoder"]["lease_released"] is True


def test_custom_aliyun_profile_used_across_eflo_pai_oss_and_no_shengdong_default(
    state, tmp_path, monkeypatch
):
    monkeypatch.delenv("ALIYUN_PROFILE", raising=False)
    unconfigured = orch.Cloud()
    with pytest.raises(ValueError, match="explicit Aliyun CLI profile is required"):
        unconfigured.cli("pai-dlc", "GetJob", JobId="dlc-1")
    with pytest.raises(ValueError, match="explicit Aliyun CLI profile is required"):
        unconfigured.upload_source(
            tmp_path / "source.tar", "oss://leap-world-shanghai/test/source.tar"
        )
    with pytest.raises(ValueError, match="explicit Aliyun CLI profile is required"):
        unconfigured.stage_h100_source(state["source"])
    with pytest.raises(ValueError, match="explicit Aliyun CLI profile is required"):
        unconfigured.submit_job(state["spec"]["sites"]["h200"]["job_spec"])

    recorded_check_output = []
    recorded_run = []

    def fake_check_output(argv, **kwargs):
        recorded_check_output.append(list(argv))
        if "oss" in argv and "sign" in argv:
            return "https://leap-world-shanghai.oss-cn-shanghai-internal.aliyuncs.com/source.tar?Signature=ok\n"
        if "RunCommand" in argv:
            return json.dumps({"InvokeId": "inv-custom"})
        if "DescribeInvocations" in argv:
            return json.dumps(
                {
                    "Invocations": {
                        "Invocation": [
                            {
                                "InvokeId": "inv-custom",
                                "InvocationStatus": "Finished",
                                "InvokeNodes": {
                                    "InvokeNode": [
                                        {
                                            "NodeId": orch.TARGET["node_id"],
                                            "InvocationStatus": "Success",
                                            "ExitCode": 0,
                                            "Output": "ok\n",
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                }
            )
        if "/api/v1/jobs" in argv:
            return json.dumps({"JobId": "dlc-custom-1"})
        if "GetJob" in argv:
            expected_job = state["resources"]["dit"].get(
                "expected_job", state["spec"]["sites"]["h200"]["job_spec"]
            )
            return json.dumps(
                {
                    "JobId": "dlc-custom-1",
                    "DisplayName": state["resources"]["dit"]["name"],
                    "WorkspaceId": orch.TARGET["workspace_id"],
                    "Status": "Running",
                    **expected_job,
                }
            )
        return "{}"

    def fake_run(argv, **kwargs):
        recorded_run.append(list(argv))

    monkeypatch.setattr(orch.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(orch.subprocess, "run", fake_run)

    custom_profile = "custom-operator-profile"
    parser = orch.build_arg_parser()
    parsed = parser.parse_args(
        [
            "probe",
            "--aliyun-profile",
            custom_profile,
            "--spec",
            "/tmp/spec.json",
            "--output",
            "/tmp/out",
        ]
    )
    assert parsed.aliyun_profile == custom_profile

    cloud = orch.Cloud(aliyun_profile=custom_profile)
    cloud.eflo("echo ok")
    cloud.upload_source(
        tmp_path / "source.tar", "oss://leap-world-shanghai/test/source.tar"
    )
    cloud.stage_h100_source(state["source"])
    state["resources"]["dit"]["expected_job"] = orch.read(tmp_path / "dit-0/pai.json")
    assert cloud.submit_job(state["resources"]["dit"]["expected_job"]) == "dlc-custom-1"

    all_argvs = recorded_check_output + recorded_run
    assert len(all_argvs) >= 5
    for argv in all_argvs:
        assert argv[0] == "aliyun"
        idx = argv.index("--profile")
        assert argv[idx + 1] == custom_profile
        assert "shengdong" not in argv


def test_rollback_uses_target_historical_version_render_tools_and_prepares_before_down(
    state, tmp_path, monkeypatch
):
    import io
    import tarfile

    history_dir = tmp_path / "history"
    for c in state["spec"]["roles"].values():
        c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
    state["phase"] = "ready"

    # Build a synthetic historical source.tar whose launch.py/source.py/select_gpu.py
    # differ from the current checkout (like commit 7cd0c632a278c0b48e7cf944c38d99e823f1a7e8,
    # where launch.CONFIG_KEYS did not include 'profile' or 'service_count').
    hist_launch_code = (
        Path(launch.__file__)
        .read_text()
        .replace(
            '"profile",\n    "resolution",\n    "service_count",\n',
            '"role_launcher",\n    "resolution",\n',
        )
        + "\n# historical-launch-version\n"
    )
    hist_source_code = (
        Path(source.__file__).read_text() + "\n# historical-source-version\n"
    )
    hist_select_code = (
        Path(launch.__file__).with_name("select_gpu.py").read_text()
        + "\n# historical-select-gpu-version\n"
    )
    hist_orch_code = Path(orch.__file__).read_text() + "\n# historical-orch-version\n"

    tar_path = tmp_path / "historical-source.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name, text in (
            ("launch.py", hist_launch_code),
            ("source.py", hist_source_code),
            ("select_gpu.py", hist_select_code),
            ("orchestrate.py", hist_orch_code),
        ):
            raw = text.encode()
            info = tarfile.TarInfo("deploy/sglang/" + name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))

    hist_commit = "7cd0c632a278c0b48e7cf944c38d99e823f1a7e8"
    hist_sha256 = source.digest(tar_path)
    hist_tools = {
        "launch.py": hashlib.sha256(hist_launch_code.encode()).hexdigest(),
        "source.py": hashlib.sha256(hist_source_code.encode()).hexdigest(),
        "select_gpu.py": hashlib.sha256(hist_select_code.encode()).hexdigest(),
        "orchestrate.py": hashlib.sha256(hist_orch_code.encode()).hexdigest(),
    }
    assert hist_tools["launch.py"] != source.digest(Path(launch.__file__))

    old_state = copy.deepcopy(state)
    old_state["deployment_id"] = "deploy-20261110T001000Z-old7cd0c"
    old_state["source"] = {
        "schema": 1,
        "commit": hist_commit,
        "sha256": hist_sha256,
        "uri": f"oss://leap-world-shanghai/sglang/orchestration/{hist_commit}/{hist_sha256}/source.tar",
        "tools": hist_tools,
    }
    for role_cfg in old_state["spec"]["roles"].values():
        role_cfg.pop("profile", None)
        role_cfg.pop("service_count", None)
        role_cfg["role_launcher"] = "profile"
    orch.save_history_record(old_state, tmp_path / "old-run", history_dir=history_dir)

    current_dir = tmp_path / "current-active"
    current_state = copy.deepcopy(state)
    current_state["deployment_id"] = "deploy-20261110T020000Z-active"
    orch.save(current_dir / "state.json", current_state)

    down_calls = []
    monkeypatch.setattr(
        orch, "down", lambda st, d, c: down_calls.append(st["deployment_id"])
    )

    # 1. If target source prepare fails, down(current_state) must NEVER be called.
    def fail_prepare(repo_arg, commit_arg, out_dir, oss_prefix_arg):
        raise RuntimeError("git archive failed: commit missing locally")

    monkeypatch.setattr(orch, "prepare", fail_prepare)
    with pytest.raises(RuntimeError, match="commit missing locally"):
        orch.rollback_deployment(
            history_dir,
            "deploy-20261110T001000Z-old7cd0c",
            tmp_path / "rb-fail-prepare",
            Mock(timeout=1),
            execute=True,
            yes=True,
            current_state_path=current_dir / "state.json",
        )
    assert down_calls == []

    # 2. If prepared archive SHA256 mismatches rollback record, down(current_state) must NEVER be called.
    def mismatch_prepare(repo_arg, commit_arg, out_dir, oss_prefix_arg):
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "source.tar").write_bytes(b"corrupted-archive")
        return {"sha256": hashlib.sha256(b"corrupted-archive").hexdigest()}

    monkeypatch.setattr(orch, "prepare", mismatch_prepare)
    with pytest.raises(ValueError, match="SHA256 does not match rollback record"):
        orch.rollback_deployment(
            history_dir,
            "deploy-20261110T001000Z-old7cd0c",
            tmp_path / "rb-mismatch-sha",
            Mock(timeout=1),
            execute=True,
            yes=True,
            current_state_path=current_dir / "state.json",
        )
    assert down_calls == []

    # 3. With valid historical archive, both dry-run (execute=False) and execute=True
    # load the target version's render tools from the verified archive and succeed
    # without weakening launch.render()'s tool digest check.
    def ok_prepare(repo_arg, commit_arg, out_dir, oss_prefix_arg):
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "source.tar").write_bytes(tar_path.read_bytes())
        return {
            "schema": 1,
            "commit": commit_arg,
            "sha256": hist_sha256,
            "uri": old_state["source"]["uri"],
            "tools": hist_tools,
        }

    monkeypatch.setattr(orch, "prepare", ok_prepare)
    monkeypatch.setattr(
        orch,
        "submit",
        lambda st, d, c: st.update(phase="ready") or orch.save(d / "state.json", st),
    )

    rb_plan_dir = tmp_path / "rb-cross-version"
    planned = orch.rollback_deployment(
        history_dir,
        "deploy-20261110T001000Z-old7cd0c",
        rb_plan_dir,
        Mock(timeout=1),
        execute=False,
        current_state_path=current_dir / "state.json",
    )
    assert planned["rollback_of"] == "deploy-20261110T001000Z-old7cd0c"
    assert (rb_plan_dir / "dit-0/pai.json").exists()
    assert down_calls == []

    executed = orch.rollback_deployment(
        history_dir,
        "deploy-20261110T001000Z-old7cd0c",
        rb_plan_dir,
        Mock(timeout=1),
        execute=True,
        yes=True,
        current_state_path=current_dir / "state.json",
    )
    assert down_calls == ["deploy-20261110T020000Z-active"]
    assert executed["phase"] == "ready"


def test_in_place_attach_multi_service_second_endpoint_failure_and_lost_response_reconciliation(
    state, tmp_path, monkeypatch
):
    state["phase"] = "ready"
    state["spec"]["roles"]["encoder"]["service_count"] = 2
    state["spec"]["pool"]["capacity"]["encoder"] = {"slots": 2}
    state["selftest"] = dict(
        status="passed",
        source=copy.deepcopy(state["source"]),
        worker_ids={
            r: state["resources"][r]["receipt"]["worker_id"] for r in orch.ROLES
        },
    )
    for role in orch.ROLES:
        state["spec"]["roles"][role]["secret_files"] = {
            k: "/secrets/" + k for k in launch.SECRET_ENV
        }
    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "operator-secret")
    monkeypatch.setattr(orch, "verify_pool", Mock())

    server_managed_endpoints = set()
    fail_second_encoder_ep = {"enabled": True}
    lose_dit_attach_response = {"enabled": True}
    attach_calls = []
    observe_calls = []
    identity_calls = []

    def endpoint_meta(url):
        if ":31000/" in url:
            return "encoder", "http://10.0.0.121:31000", 0
        if ":31001/" in url:
            return "encoder", "http://10.0.0.121:31001", 1
        if ":33000/" in url:
            return "dit", "http://10.0.0.122:33000", 0
        return "decoder", "http://10.0.0.121:36000", 0

    def http_handler(url, payload=None, token=None, cloud=None):
        if "/v1/capacity" in url:
            return dict(
                backend="role_pool",
                profile="480",
                roles={"model_group": {"free_slots": 2}},
            )
        role, base_ep, idx = endpoint_meta(url)
        if "/get_connection_status" in url:
            if base_ep in server_managed_endpoints:
                raise RuntimeError("HTTP 403 MANAGED_CONTROL_REQUIRED")
            return dict(state="available")
        if url.endswith("/managed/identity"):
            assert base_ep in server_managed_endpoints
            identity_calls.append(base_ep)
            return dict(
                service=dict(
                    worker_id=f"{role}-worker",
                    role=role,
                    service_id=f"{role}-worker/{idx}",
                ),
                boot_epoch=f"{role}-boot",
                control_epoch=state["spec"]["pool"]["control_epoch"],
                deployment=dict(hardware_ids=[f"GPU-{role}"]),
            )
        if url.endswith("/managed/observe"):
            assert base_ep in server_managed_endpoints
            assert payload["boot_epoch"] == f"{role}-boot"
            observe_calls.append(base_ep)
            return dict(state="idle", owner=None, restart_required=False)
        if url.endswith("/managed/attach"):
            attach_calls.append(base_ep)
            if (
                base_ep == "http://10.0.0.121:31001"
                and fail_second_encoder_ep["enabled"]
            ):
                raise RuntimeError("encoder second service endpoint timeout")
            server_managed_endpoints.add(base_ep)
            if (
                base_ep == "http://10.0.0.122:33000"
                and lose_dit_attach_response["enabled"]
            ):
                raise RuntimeError("connection reset after server attached dit")
            return dict(
                service=dict(
                    worker_id=f"{role}-worker",
                    role=role,
                    service_id=f"{role}-worker/{idx}",
                ),
                boot_epoch=f"{role}-boot",
                control_epoch=payload["control_epoch"],
                deployment=dict(hardware_ids=payload["worker_config"]["hardware_ids"]),
            )
        raise AssertionError(url)

    monkeypatch.setattr(orch, "http", http_handler)

    # Attempt 1: encoder :31000 succeeds and is persisted in endpoint_identities;
    # encoder :31001 fails before role-level identities["encoder"] is written.
    with pytest.raises(RuntimeError, match="encoder second service endpoint timeout"):
        orch.attach(state, tmp_path, Mock(timeout=1), execute=True, in_place=True)

    persisted_1 = orch.read(tmp_path / "state.json")
    assert "encoder" not in persisted_1["attachment"].get("identities", {})
    assert (
        persisted_1["attachment"]["endpoint_identities"]["encoder"][
            "http://10.0.0.121:31000"
        ]["service_id"]
        == "encoder-worker/0"
    )
    assert (
        "http://10.0.0.121:31001"
        in persisted_1["attachment"]["endpoint_intents"]["encoder"]
    )

    # Attempt 2: encoder :31000 is reconciled via /managed/identity + /managed/observe
    # (no 403 on /get_connection_status and no duplicate /managed/attach on :31000);
    # encoder :31001 attaches; dit :33000 attaches on server but HTTP response is lost!
    fail_second_encoder_ep["enabled"] = False
    with pytest.raises(
        RuntimeError, match="connection reset after server attached dit"
    ):
        orch.attach(state, tmp_path, Mock(timeout=1), execute=True, in_place=True)

    persisted_2 = orch.read(tmp_path / "state.json")
    assert "encoder" in persisted_2["attachment"]["identities"]
    assert "dit" not in persisted_2["attachment"]["identities"]
    assert (
        "http://10.0.0.122:33000"
        in persisted_2["attachment"]["endpoint_intents"]["dit"]
    )
    assert attach_calls.count("http://10.0.0.121:31000") == 1

    # Attempt 3 (via _attach_owned_in_place as well): dit :33000 returns 403 on
    # /get_connection_status, reconciles via /managed/identity + /managed/observe
    # without re-calling /managed/attach, and decoder :36000 completes attach.
    lose_dit_attach_response["enabled"] = False
    orch._attach_owned_in_place(state, tmp_path, Mock(timeout=1))
    assert set(state["attachment"]["identities"]) == set(orch.ROLES)
    assert attach_calls.count("http://10.0.0.122:33000") == 1
    assert "http://10.0.0.122:33000" in identity_calls
    assert "http://10.0.0.122:33000" in observe_calls


def test_effective_argv_never_uses_local_sys_executable_and_recovers_container_receipt(
    state, tmp_path
):
    # 1. When receipt is compact and omits argv, effective_argv is "unknown",
    # never reconstructed from local sys.executable.
    for role in orch.ROLES:
        state["resources"][role]["receipt"].pop("argv", None)
    summary = orch.write_deployment_summary(state, tmp_path / "compact-summary")
    md_text = (tmp_path / "compact-summary/DEPLOYMENT.md").read_text()
    for role in orch.ROLES:
        lp = summary["roles"][role]["launch_profile"]
        assert lp["effective_argv"] == "unknown"
        assert lp["argv_status"] == "unknown"
        assert lp["planned_argv"][0] == "/opt/sglang/bin/python"
        assert sys.executable not in str(lp["planned_argv"])
    assert "Effective runtime `argv`: `unknown`" in md_text
    assert sys.executable not in md_text

    # 2. Cloud.receipt recovers container-internal receipt.json via kubectl exec for K3s Pods.
    enc_res = state["resources"]["encoder"]
    compact_line = json.dumps(enc_res["receipt"], separators=(",", ":"))
    full_container_argv = [
        "/opt/sglang/bin/python",
        "-m",
        "sglang.multimodal_gen.tools.zing_h3_profile_launcher",
        "--mode",
        "independent",
        "--component",
        "encoder",
    ]
    full_disk_receipt = {**enc_res["receipt"], "argv": full_container_argv}
    cloud = orch.Cloud(aliyun_profile="test-operator")
    cloud.pod = Mock(return_value={"metadata": {"uid": enc_res["uid"]}})

    def fake_eflo(script):
        if " logs " in script:
            return compact_line + "\n"
        if " exec " in script:
            return json.dumps(full_disk_receipt)
        raise AssertionError(script)

    cloud.eflo = Mock(side_effect=fake_eflo)
    recovered = cloud.receipt(enc_res, "encoder")
    assert recovered["argv"] == full_container_argv
    enc_res["receipt"] = recovered
    summary_obs = orch.write_deployment_summary(state, tmp_path / "observed-summary")
    assert (
        summary_obs["roles"]["encoder"]["launch_profile"]["effective_argv"]
        == full_container_argv
    )
    assert (
        summary_obs["roles"]["encoder"]["launch_profile"]["argv_status"] == "observed"
    )


def test_skill_and_orchestration_doc_cli_examples_parse_cleanly():
    import re
    import shlex

    parser = orch.build_arg_parser()
    doc_paths = [
        orch.REPO / ".agents/skills/sglang-env-deploy/SKILL.md",
        orch.REPO / "deploy/sglang/ORCHESTRATION.md",
    ]
    parsed_commands = []
    for doc_path in doc_paths:
        text = doc_path.read_text()
        blocks = re.findall(r"```(?:bash|sh)\n(.*?)```", text, flags=re.DOTALL)
        for block in blocks:
            normalized = block.replace("\\\n", " ")
            for raw_line in normalized.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = shlex.split(line)
                if (
                    len(tokens) >= 2
                    and tokens[0] in ("python", "python3")
                    and tokens[1].endswith("deploy/sglang/orchestrate.py")
                ):
                    ns = parser.parse_args(tokens[2:])
                    parsed_commands.append((doc_path.name, ns.action, tokens[2:]))
    assert len(parsed_commands) >= 16
    actions = {action for _, action, _ in parsed_commands}
    assert actions == {
        "probe",
        "up",
        "inspect",
        "selftest",
        "attach",
        "history",
        "show",
        "diff",
        "rollback",
        "down",
    }


def test_all_h200_topology_minimal_spec_expansion_and_default_cpu_16():
    minimal = {
        "schema": 1,
        "topology": "h200",
        "model_id": "model1027",
        "model": "/opt/whitebox/assets/model",
        "stage_from": "/mnt/serving-assets",
        "asset_receipt": "/mnt/oss/assets.json",
        "asset_receipt_sha256": "a" * 64,
        "image": "registry.example/sglang@sha256:" + "c" * 64,
        "target": copy.deepcopy(orch.TARGET),
    }
    expanded = orch.expand_minimal_spec(minimal)
    assert set(expanded["sites"]) == {"h200"}
    rc = expanded["sites"]["h200"]["job_spec"]["ResourceConfig"]
    assert rc == {"GPU": "1", "CPU": "16", "Memory": "128Gi", "SharedMemory": "64Gi"}
    for role in orch.ROLES:
        assert orch.is_pai_role(expanded, role) is True
        assert expanded["roles"][role]["profile"] == "h200"
        assert expanded["roles"][role]["stage_from"] == "/mnt/serving-assets"


def test_probe_h200_inventory_detects_gpu_and_cpu_starvation_without_h100_blockers(
    tmp_path,
):
    minimal = {
        "schema": 1,
        "topology": "h200",
        "model_id": "model1027",
        "model": "/opt/whitebox/assets/model",
        "asset_receipt": "/mnt/oss/assets.json",
        "asset_receipt_sha256": "a" * 64,
        "image": "registry.example/sglang@sha256:" + "c" * 64,
        "target": copy.deepcopy(orch.TARGET),
    }
    cloud = Mock(timeout=1)
    # H100 has zero idle GPUs, which must NOT block an all-H200 topology!
    cloud.eflo.return_value = json.dumps(
        {
            "gpus": "",
            "processes": "",
            "pods": {"items": []},
            "images": {"images": []},
            "models": {},
            "busy_ports": [],
        }
    )

    def make_jobs(count, cpu_each):
        jobs = [
            {
                "JobId": f"dlc-busy-{i}",
                "DisplayName": f"busy-{i}",
                "Status": "Running",
                "UserId": "other",
                "ResourceId": "quota3l8bnv9a35n",
            }
            for i in range(count)
        ]
        details = {
            f"dlc-busy-{i}": {
                "JobId": f"dlc-busy-{i}",
                "Status": "Running",
                "JobSpecs": [
                    {
                        "PodCount": 1,
                        "Image": "registry.example/sglang@sha256:" + "c" * 64,
                        "ResourceConfig": {"GPU": "1", "CPU": str(cpu_each)},
                    }
                ],
            }
            for i in range(count)
        }
        return jobs, details

    # Case 1: 5 GPUs running * 14 CPUs = 70 CPUs used; 3 new H200 roles * 16 CPUs = 48 -> fits (8 GPUs, 118 CPUs <= 120), zero H100 needed.
    jobs_ok, details_ok = make_jobs(5, 14)
    cloud.cli.return_value = {"Jobs": jobs_ok, "TotalCount": len(jobs_ok)}
    cloud.get_job.side_effect = lambda jid: details_ok[jid]
    ev_ok = orch.probe(copy.deepcopy(minimal), tmp_path / "probe-ok", cloud)
    assert ev_ok["blockers"] == []
    assert ev_ok["h200_inventory"]["running_gpus"] == 5
    assert ev_ok["h200_inventory"]["requested_pai_gpus"] == 3
    assert ev_ok["h200_inventory"]["requested_pai_cpus"] == 48

    # Case 2: 5 GPUs running * 32 CPUs = 160 CPUs used; +48 = 208 > 192 -> CPU starvation blocker!
    jobs_cpu, details_cpu = make_jobs(5, 32)
    cloud.cli.return_value = {"Jobs": jobs_cpu, "TotalCount": len(jobs_cpu)}
    cloud.get_job.side_effect = lambda jid: details_cpu[jid]
    ev_cpu = orch.probe(copy.deepcopy(minimal), tmp_path / "probe-cpu", cloud)
    assert any("CPU" in b for b in ev_cpu["blockers"])


def test_submit_all_pai_roles_runs_in_parallel_and_records_timings_sla(
    spec, tmp_path, monkeypatch
):
    import threading

    for role in orch.ROLES:
        spec["roles"][role]["profile"] = "h200"
    spec["sites"] = {"h200": spec["sites"]["h200"]}
    checked = orch.validate_spec(spec)
    manifest = dict(
        schema=1,
        commit="b" * 40,
        sha256="c" * 64,
        uri="oss://leap-world-shanghai/test/"
        + "b" * 40
        + "/"
        + "c" * 64
        + "/source.tar",
        tools={
            n: source.digest(Path(__file__).with_name(n))
            for n in ("launch.py", "source.py", "select_gpu.py", "orchestrate.py")
        },
    )
    run_state = dict(
        schema=1,
        run_id="d" * 32,
        target=copy.deepcopy(orch.TARGET),
        phase="planned",
        generation=0,
        source=manifest,
        spec=checked,
        resources={},
    )
    orch.render_roles(run_state, tmp_path)
    (tmp_path / "source").mkdir()
    arc = tmp_path / "source/source.tar"
    arc.write_bytes(b"archive")
    run_state["source"]["sha256"] = source.digest(arc)

    barrier = threading.Barrier(3, timeout=5)
    persisted_before_submit = []

    def parallel_submit(job_payload):
        # Verify write-ahead state.json already has all 3 roles before any submit finishes!
        on_disk = orch.read(tmp_path / "state.json")
        persisted_before_submit.append(set(on_disk["resources"]))
        barrier.wait()
        name = job_payload["DisplayName"]
        role = next(r for r in orch.ROLES if f"sglang-{r}-" in name)
        return f"dlc-par-{role}"

    def fake_receipt(resource, role):
        cfg = orch.read(tmp_path / f"{role}-0/role.json")
        http_port = launch.role_ports(cfg)["http_port"]
        return dict(
            schema=1,
            role=role,
            status="ready",
            commit=manifest["commit"],
            archive_sha256=run_state["source"]["sha256"],
            model_id="model1027",
            asset_receipt_sha256="a" * 64,
            worker_id=f"{role}-par",
            role_config_sha256=hashlib.sha256(
                json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            http=f"http://10.0.0.15{orch.ROLES.index(role)}:{http_port}",
            gpu=f"GPU-{role}-h200, H200, 140000, 0",
            warmed_up=True,
            warmup={"role": role, "warmed_up": True, "elapsed_s": 2.5},
        )

    cloud = Mock(timeout=1)
    cloud.submit_job.side_effect = parallel_submit
    cloud.receipt.side_effect = fake_receipt
    monkeypatch.setattr(
        orch,
        "http",
        lambda url, payload=None, token=None, cloud=None: {
            "ready": True,
            "role": next(
                r for r in orch.ROLES if f"10.0.0.15{orch.ROLES.index(r)}" in url
            ),
            "warmed_up": True,
        },
    )

    orch.submit(run_state, tmp_path, cloud)
    assert len(persisted_before_submit) == 3
    assert all(s == set(orch.ROLES) for s in persisted_before_submit)
    assert run_state["phase"] == "ready"
    assert run_state["timings_sec"]["within_10min_sla"] is True
    assert "submit_sec" in run_state["timings_sec"]
    assert "wait_ready_sec" in run_state["timings_sec"]
    summary = orch.write_deployment_summary(run_state, tmp_path)
    assert summary["acceptance_gates"]["within_10min_sla"] is True
    assert summary["acceptance_gates"]["all_roles_warmed_up"] is True


def test_eflo_http_forwards_structured_http_error_and_remote_selftest_uses_eflo_relay(
    state, monkeypatch
):
    from urllib.error import HTTPError

    cloud = Mock(timeout=1)
    cloud.eflo.return_value = json.dumps(
        {
            "__eflo_http_error__": 403,
            "reason": "Forbidden",
            "body": '{"error":"MANAGED_CONTROL_REQUIRED"}',
        }
    )
    monkeypatch.setattr(
        orch,
        "build_opener",
        Mock(
            return_value=Mock(
                open=Mock(side_effect=OSError("Network unreachable from laptop"))
            )
        ),
    )
    with pytest.raises(HTTPError) as exc_info:
        orch.http("http://10.0.0.150:31000/get_connection_status", cloud=cloud)
    assert exc_info.value.code == 403
    assert "MANAGED_CONTROL_REQUIRED" in exc_info.value.read().decode()

    # All-PAI deployment (no K3s pod UIDs) runs remote_selftest via Eflo /usr/bin/python3.12 relay.
    for r in orch.ROLES:
        state["resources"][r].pop("uid", None)
        state["resources"][r].pop("namespace", None)
        state["resources"][r]["job_id"] = f"dlc-{r}"
    captured_scripts = []

    def relay_eflo(script):
        captured_scripts.append(script)
        return '{"status":"passed","frames":4}\n'

    cloud.eflo.side_effect = relay_eflo
    addrs = {r: state["resources"][r]["receipt"]["http"] for r in orch.ROLES}
    wids = {r: state["resources"][r]["receipt"]["worker_id"] for r in orch.ROLES}
    res = orch.remote_selftest(state, addrs, wids, cloud)
    assert res == {"status": "passed", "frames": 4}
    assert "/opt/sglang/bin/python" in captured_scripts[-1]
    assert "run --rm --net-host" in captured_scripts[-1]
    assert "PY=/usr/bin/python3.12" in captured_scripts[-1]
    assert "kubectl" not in captured_scripts[-1]
    # Ensure the generated Python snippet passed to -c is syntactically valid!
    import shlex as _shlex

    tokens = _shlex.split(captured_scripts[-1])
    c_idx = tokens.index("-c")
    compile(tokens[c_idx + 1], "<selftest_relay>", "exec")


def test_reusing_pai_pinned_encoder_decoder_validates_job_and_compatibility_domain(
    spec, tmp_path, monkeypatch
):
    spec["existing_services"] = {
        "encoder": {
            "http": "http://10.0.0.151:31000",
            "worker_id": "enc-pai",
            "gpu": "GPU-enc-pai",
            "boot_epoch": "enc-be",
            "pai": True,
            "job_id": "dlc-enc-1",
            "job_name": "sglang-encoder-fixed",
        },
        "decoder": {
            "http": "http://10.0.0.152:36000",
            "worker_id": "dec-pai",
            "gpu": "GPU-dec-pai",
            "boot_epoch": "dec-be",
            "pai": True,
            "job_id": "dlc-dec-1",
            "job_name": "sglang-decoder-fixed",
        },
    }
    spec["roles"] = {"dit": spec["roles"]["dit"]}
    spec["sites"] = {"h200": spec["sites"]["h200"]}
    checked = orch.validate_spec(spec)
    manifest = dict(
        schema=1,
        commit="b" * 40,
        sha256="c" * 64,
        uri="oss://leap-world-shanghai/test/"
        + "b" * 40
        + "/"
        + "c" * 64
        + "/source.tar",
        tools={
            n: source.digest(Path(__file__).with_name(n))
            for n in ("launch.py", "source.py", "select_gpu.py", "orchestrate.py")
        },
    )
    st = dict(
        schema=1,
        run_id="1" * 32,
        target=copy.deepcopy(orch.TARGET),
        phase="submitted",
        generation=0,
        source=manifest,
        spec=checked,
        resources={
            r: orch.reused_resource("1" * 32, checked["model_id"], r, entry)
            for r, entry in checked["existing_services"].items()
        },
    )
    orch.render_roles(st, tmp_path)
    dit_cfg = orch.read(tmp_path / "dit-0/role.json")
    st["resources"]["dit"] = dict(
        run_id="1" * 32,
        generation=0,
        name="sglang-dit-owned",
        job_id="dlc-dit-1",
        status="submitted",
        expected_job=orch.read(tmp_path / "dit-0/pai.json"),
    )

    cloud = Mock(timeout=1)
    cloud.get_job.side_effect = lambda jid: {
        "JobId": jid,
        "WorkspaceId": orch.TARGET["workspace_id"],
        "ResourceId": orch.TARGET["quota_id"],
        "DisplayName": (
            "sglang-encoder-fixed"
            if jid == "dlc-enc-1"
            else ("sglang-decoder-fixed" if jid == "dlc-dec-1" else "sglang-dit-owned")
        ),
        "Status": "Running",
    }
    cloud.receipt.return_value = dict(
        schema=1,
        role="dit",
        status="ready",
        commit=manifest["commit"],
        archive_sha256=manifest["sha256"],
        model_id=checked["model_id"],
        asset_receipt_sha256="a" * 64,
        worker_id="dit-owned",
        role_config_sha256=hashlib.sha256(
            json.dumps(dit_cfg, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        http="http://10.0.0.153:33000",
        gpu="GPU-dit-pai",
        warmed_up=True,
    )

    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "op-token")
    wrong_compat = {"enabled": True}

    def fake_http(url, payload=None, token=None, cloud=None):
        if url.endswith("/health"):
            role = (
                "encoder"
                if ":31000/" in url
                else ("decoder" if ":36000/" in url else "dit")
            )
            return {"ready": True, "role": role, "warmed_up": True}
        if "/get_connection_status" in url:
            raise RuntimeError("HTTP 403 MANAGED_CONTROL_REQUIRED")
        if url.endswith("/managed/identity"):
            role = "encoder" if ":31000/" in url else "decoder"
            wid = "enc-pai" if role == "encoder" else "dec-pai"
            be = "enc-be" if role == "encoder" else "dec-be"
            return {
                "service": {"worker_id": wid, "role": role, "service_id": f"{wid}/0"},
                "boot_epoch": be,
                "control_epoch": checked["pool"]["control_epoch"],
            }
        if url.endswith("/v1/role-pool/process-status"):
            return {
                "process": {
                    "state": "ready",
                    "manifest": {
                        "compatibility": (
                            "other-tenant-group"
                            if wrong_compat["enabled"]
                            else checked["pool"]["compatibility"]
                        ),
                        "network_domain": checked["pool"]["network_domain"],
                    },
                }
            }
        raise AssertionError(url)

    monkeypatch.setattr(orch, "http", fake_http)
    with pytest.raises(
        ValueError, match="compatibility or network_domain does not match"
    ):
        orch.wait_ready(st, tmp_path, cloud)

    wrong_compat["enabled"] = False
    orch.wait_ready(st, tmp_path, cloud)
    assert st["phase"] == "ready"


def test_verify_pool_checks_exact_compatibility_group_capacity_when_room_token_present(
    state, tmp_path, monkeypatch
):
    w_configs = orch.worker_configs(state)
    for r in orch.ROLES:
        state["resources"][r]["receipt"]["worker_config"] = w_configs[r]
    state["attachment"] = {
        "identities": {
            r: {
                "process_id": f"{r}-worker",
                "boot_epoch": f"{r}-boot",
                "service_id": f"{r}-worker/0",
            }
            for r in orch.ROLES
        }
    }
    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "op-secret")
    monkeypatch.setenv("WHITEBOX_ROOM_TOKEN", "room-secret")
    free_slots_holder = {"slots": 0}

    def fake_http(url, payload=None, token=None, cloud=None):
        if "/v1/capacity?" in url:
            # Simulate shared route showing another tenant's busy group (free_slots=0)
            return {
                "backend": "role_pool",
                "profile": "480",
                "roles": {"model_group": {"free_slots": 0}},
            }
        if url.endswith("/v1/role-pool/capacity"):
            assert token == "room-secret"
            assert payload["compatibility"] == state["spec"]["pool"]["compatibility"]
            return {"free_slots": free_slots_holder["slots"]}
        if url.endswith("/managed/identity"):
            role = (
                "encoder"
                if ":31000/" in url
                else ("dit" if ":33000/" in url else "decoder")
            )
            return {
                "service": {
                    "worker_id": f"{role}-worker",
                    "role": role,
                    "service_id": f"{role}-worker/0",
                },
                "boot_epoch": f"{role}-boot",
                "control_epoch": state["spec"]["pool"]["control_epoch"],
            }
        if url.endswith("/v1/role-pool/process-status"):
            role = payload["process_id"].removesuffix("-worker")
            return {
                "process": {
                    "state": "ready",
                    "seen_at": orch.time.time(),
                    "manifest": state["resources"][role]["receipt"]["worker_config"],
                }
            }
        raise AssertionError(url)

    monkeypatch.setattr(orch, "http", fake_http)
    cloud = Mock(timeout=0.05)

    with pytest.raises(OSError, match="No ready player group in compatibility pool"):
        orch.verify_pool(state, tmp_path, cloud=cloud)

    free_slots_holder["slots"] = 1
    orch.verify_pool(state, tmp_path, cloud=cloud)
    assert state["attachment"]["compatibility_capacity"]["480"]["free_slots"] == 1


def test_history_strips_inline_image_auth_and_rollback_previous_syncs_oss(
    state, tmp_path, monkeypatch
):
    history_dir = tmp_path / "history"
    auth_file = tmp_path / "ACR_AUTH"
    auth_file.write_text("super-secret-acr-auth\n")
    monkeypatch.setenv("SGLANG_PAI_IMAGE_AUTH_FILE", str(auth_file))

    for c in state["spec"]["roles"].values():
        c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
    state["spec"]["sites"]["h200"]["image_auth_file"] = str(auth_file)
    state["spec"]["sites"]["h200"]["job_spec"]["ImageConfig"] = {
        "DockerRegistry": "registry.example",
        "Username": "user",
        "Auth": "inline-secret-must-never-persist",
    }

    # Save V1 (ready) and V2 (ready, active)
    v1 = copy.deepcopy(state)
    v1["deployment_id"] = "deploy-20261114T100000Z-v1stable"
    v1["created_at"] = "2026-11-14T10:00:00Z"
    v1["phase"] = "ready"
    rec_v1 = orch.save_history_record(v1, tmp_path / "v1", history_dir=history_dir)
    serialized_v1 = json.dumps(rec_v1)
    assert "inline-secret-must-never-persist" not in serialized_v1
    assert rec_v1["rollback_bundle"]["site_secret_references"]["h200"][
        "image_auth_file"
    ] == str(auth_file)
    assert (
        "Auth"
        not in rec_v1["rollback_bundle"]["spec"]["sites"]["h200"]["job_spec"][
            "ImageConfig"
        ]
    )

    v2 = copy.deepcopy(state)
    v2["deployment_id"] = "deploy-20261114T110000Z-v2canary"
    v2["created_at"] = "2026-11-14T11:00:00Z"
    v2["phase"] = "ready"
    orch.save(tmp_path / "v2/state.json", v2)
    orch.save_history_record(v2, tmp_path / "v2", history_dir=history_dir)

    resolved_prev = orch.resolve_previous_deployment_id(
        history_dir, current_deployment_id=v2["deployment_id"]
    )
    assert resolved_prev == "deploy-20261114T100000Z-v1stable"

    def fake_prepare(repo_arg, commit_arg, out_dir, oss_prefix_arg):
        out_dir.mkdir(parents=True, exist_ok=False)
        payload = b"verified-v1-archive"
        (out_dir / "source.tar").write_bytes(payload)
        return {
            "schema": 1,
            "commit": commit_arg,
            "sha256": v1["source"]["sha256"],
            "uri": v1["source"]["uri"],
            "tools": v1["source"]["tools"],
        }

    v1_arc = tmp_path / "v1_arc.tar"
    v1_arc.write_bytes(b"verified-v1-archive")
    v1["source"]["sha256"] = source.digest(v1_arc)
    v1["source"]["uri"] = (
        f"oss://leap-world-shanghai/test/{v1['source']['commit']}/{v1['source']['sha256']}/source.tar"
    )
    orch.save_history_record(v1, tmp_path / "v1", history_dir=history_dir)

    monkeypatch.setattr(orch, "prepare", fake_prepare)
    monkeypatch.setattr(orch, "down", lambda st, d, c: None)
    monkeypatch.setattr(
        orch,
        "submit",
        lambda st, d, c: st.update(phase="ready") or orch.save(d / "state.json", st),
    )
    rb_dir = tmp_path / "rb-out"
    rb_state = orch.rollback_deployment(
        history_dir,
        resolved_prev,
        rb_dir,
        Mock(timeout=1),
        execute=True,
        yes=True,
        current_state_path=tmp_path / "v2/state.json",
    )
    assert rb_state["rollback_of"] == "deploy-20261114T100000Z-v1stable"
    rendered_pai = orch.read(rb_dir / "dit-0/pai.json")
    assert rendered_pai["JobSpecs"][0]["ImageConfig"]["Auth"] == "super-secret-acr-auth"

    # Verify Cloud.sync_history_artifacts uploads via aliyun oss cp with top-level --endpoint.
    oss_cmds = []
    monkeypatch.setattr(
        orch.subprocess, "run", lambda argv, **kw: oss_cmds.append(list(argv))
    )
    orch.write_deployment_summary(rb_state, rb_dir)
    orch.save_history_record(rb_state, rb_dir, history_dir=history_dir)
    cloud = orch.Cloud(target=rb_state["target"], aliyun_profile="test-op")
    synced = cloud.sync_history_artifacts(
        history_dir, rb_state["deployment_id"], summary_dir=rb_dir
    )
    assert len(synced) >= 3
    assert all("--endpoint" in cmd for cmd in oss_cmds)


def test_verify_traffic_quality_decode_psnr_ssim_and_deployment_summary_gate(
    state, tmp_path
):
    import verify_traffic_quality as vtq

    w, h = 64, 48
    frames_base = []
    frames_near = []
    frames_black = []
    for idx in range(16):
        pixels = bytearray()
        near_pixels = bytearray()
        for y in range(h):
            for x in range(w):
                r = (x * 4 + idx * 5) % 256
                g = (y * 5 + idx * 3) % 256
                b = ((x + y) * 3 + idx * 7) % 256
                pixels.extend((r, g, b))
                near_pixels.extend((min(255, r + (idx % 2)), g, b))
        frames_base.append(bytes(pixels))
        frames_near.append(bytes(near_pixels))
        frames_black.append(b"\x00" * (w * h * 3))

    batches = [
        {
            "output_frame_start": i,
            "num_frames": 1,
            "width": w,
            "height": h,
            "encoding": "raw",
            "payload": f,
        }
        for i, f in enumerate(frames_base)
    ]
    decoded = vtq.decode_frame_batches_to_rgb(
        batches, expected_width=w, expected_height=h
    )
    assert decoded == frames_base

    # Without a pinned baseline, intrinsic quality may pass, but quality_unchanged MUST fail-closed to False!
    base_eval = vtq.evaluate_visual_quality(
        frames_base, width=w, height=h, min_frames=16
    )
    assert base_eval["intrinsic_quality"]["passed"] is True
    assert base_eval["quality_unchanged"] is False
    assert base_eval["is_baseline"] is True

    # Even with baseline_frames, missing baseline_identity MUST fail-closed to False!
    unbound_eval = vtq.evaluate_visual_quality(
        frames_near,
        width=w,
        height=h,
        min_frames=16,
        baseline_frames=frames_base,
        baseline_identity=None,
    )
    assert unbound_eval["quality_unchanged"] is False

    base_dir = tmp_path / "pinned-baseline"
    vtq.save_frames_png(frames_base, base_dir / "frames", width=w, height=h)
    base_manifest = vtq.write_baseline_manifest(
        base_dir,
        frames=frames_base,
        width=w,
        height=h,
        case_id="shanghai_drive_baseline",
        seed=886410,
        inference_steps=2,
    )

    near_eval = vtq.evaluate_visual_quality(
        frames_near,
        width=w,
        height=h,
        min_frames=16,
        baseline_frames=frames_base,
        baseline_identity=base_manifest,
    )
    assert near_eval["quality_unchanged"] is True
    assert near_eval["baseline_comparison"]["mean_psnr_db"] > 45.0
    assert near_eval["baseline_comparison"]["mean_ssim"] > 0.99

    black_eval = vtq.evaluate_visual_quality(
        frames_black,
        width=w,
        height=h,
        min_frames=16,
        baseline_frames=frames_base,
        baseline_identity=base_manifest,
    )
    assert black_eval["quality_unchanged"] is False

    # Bind to deployment state and verify acceptance_gates in deployment.json.
    run_dir = tmp_path / "dep-qual"
    run_dir.mkdir()
    state["phase"] = "ready"
    orch.save(run_dir / "state.json", state)
    with pytest.raises(
        ValueError, match="Baseline recording output cannot be attached"
    ):
        vtq.attach_verification_to_deployment(run_dir / "state.json", base_eval)
    summary = vtq.attach_verification_to_deployment(run_dir / "state.json", near_eval)
    assert summary["acceptance_gates"]["quality_unchanged"] is True


def test_last_receipt_prefers_gpu_uuid_over_nested_runtime_gpu_in_truncated_pai_log():
    truncated_line = (
        '{"schema":1,"role":"dit","archive_sha256":"'
        + ("a" * 64)
        + '","worker_id":"dit-test","status":"ready","runtime":{"torch":"2.13.0","gpu":"NVIDIA L20X"},'
        '"gpu":"GPU-31ae5499-2973-a77a-2b80-61fe52b8e01a, NVIDIA L20X, 143771 MiB, 1 MiB","pod":"dlc-'
    )
    parsed = orch.last_receipt(truncated_line, "dit")
    assert (
        parsed["gpu"]
        == "GPU-31ae5499-2973-a77a-2b80-61fe52b8e01a, NVIDIA L20X, 143771 MiB, 1 MiB"
    )


def test_check_role_warmup_and_pool_capacity_reads_nested_model_group_free_slots(
    monkeypatch,
):
    import verify_traffic_quality as vtq

    def fake_http_json(url, payload=None, token=None, timeout=10.0):
        if url.endswith("/health"):
            role = (
                "encoder"
                if "31000" in url
                else ("dit" if "33000" in url else "decoder")
            )
            return {"ready": True, "role": role, "warmed_up": True}
        if "/v1/capacity?" in url:
            return {"backend": "role_pool", "profile": "480"}
        if url.endswith("/v1/role-pool/capacity"):
            return {
                "backend": "role_pool",
                "profile": "480",
                "roles": {"model_group": {"free_slots": 1, "active_sessions": 0}},
            }
        raise AssertionError(url)

    monkeypatch.setattr(vtq, "_http_json", fake_http_json)
    report = vtq.check_role_warmup_and_pool_capacity(
        role_urls={
            "encoder": "http://10.0.0.10:31000",
            "dit": "http://10.0.0.11:33000",
            "decoder": "http://10.0.0.12:36000",
        },
        coordinator_url="http://10.0.0.121:28181",
        routing_revision="whitebox-copy-model1027-480",
        compatibility="h200-deploy-verify-20261114",
        profile="480",
        network_domain="shanghai-vpc-h100-h200",
        room_token="t" * 64,
    )
    assert report["all_roles_warmed_up"] is True
    assert (
        report["capacity"]["exact_group_capacity"]["roles"]["model_group"]["free_slots"]
        == 1
    )


def test_infer_denoising_steps_from_state_handles_csv_and_list():
    import verify_traffic_quality as vtq

    assert (
        vtq.infer_denoising_steps_from_state(
            {
                "spec": {
                    "roles": {
                        "dit": {"dit_options": {"zing_denoising_step_list": "1000,500"}}
                    }
                }
            }
        )
        == 2
    )
    assert (
        vtq.infer_denoising_steps_from_state(
            {
                "spec": {
                    "roles": {
                        "dit": {
                            "dit_options": {
                                "zing_denoising_step_list": [1000, 750, 500, 250]
                            }
                        }
                    }
                }
            }
        )
        == 4
    )
    assert vtq.infer_denoising_steps_from_state({}) is None


def test_decode_mp4_to_rgb_uses_ffmpeg_rawvideo_pipeline(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import verify_traffic_quality as vtq

    fake_mp4 = tmp_path / "model.mp4"
    fake_mp4.write_bytes(b"mp4")
    w, h = 8, 6
    expected = [bytes([10 + i]) * (w * h * 3) for i in range(4)]

    monkeypatch.setattr(vtq.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        vtq.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout=b"".join(expected), stderr=b""
        ),
    )
    decoded = vtq.decode_mp4_to_rgb(fake_mp4, expected_width=w, expected_height=h)
    assert decoded == expected


def test_resolve_previous_deployment_id_does_not_skip_cleanly_stopped_v1_over_older_v0(
    state, tmp_path
):
    history_dir = tmp_path / "history"
    base_state = copy.deepcopy(state)
    base_state["spec"]["target"] = copy.deepcopy(orch.TARGET)

    for dep_id, ts, phase in (
        ("deploy-20261114T090000Z-v0full", "2026-11-14T09:00:00Z", "ready"),
        ("deploy-20261114T100000Z-v1reuse", "2026-11-14T10:00:00Z", "stopped"),
        ("deploy-20261114T110000Z-v2reuse", "2026-11-14T11:00:00Z", "ready"),
    ):
        st = copy.deepcopy(base_state)
        st["deployment_id"] = dep_id
        st["created_at"] = ts
        st["phase"] = phase
        d = tmp_path / dep_id
        orch.save(d / "state.json", st)
        orch.save_history_record(st, d, history_dir=history_dir)

    assert (
        orch.resolve_previous_deployment_id(
            history_dir, current_deployment_id="deploy-20261114T110000Z-v2reuse"
        )
        == "deploy-20261114T100000Z-v1reuse"
    )


def test_submit_sends_image_auth_to_pai_without_persisting_auth_in_state_json(
    state, tmp_path, monkeypatch
):
    auth_file = tmp_path / "acr_auth.secret"
    auth_file.write_text("top-secret-acr-auth-value\n")
    out_dir = tmp_path / "auth-run"
    out_dir.mkdir()
    (out_dir / "source").mkdir()
    (out_dir / "source/source.tar").write_bytes(b"fake-source-archive")
    st = copy.deepcopy(state)
    new_sha = source.digest(out_dir / "source/source.tar")
    st["source"]["sha256"] = new_sha
    st["source"]["uri"] = (
        f"oss://leap-world-shanghai/test/{st['source']['commit']}/{new_sha}/source.tar"
    )
    st["resources"] = {}
    st["spec"]["target"] = copy.deepcopy(orch.TARGET)
    st["spec"]["existing_services"] = {
        "encoder": {"http": "http://10.0.0.10:31000", "worker_id": "enc-0"},
        "decoder": {"http": "http://10.0.0.11:36000", "worker_id": "dec-0"},
    }
    st["spec"]["roles"]["encoder"]["mode"] = "reuse"
    st["spec"]["roles"]["decoder"]["mode"] = "reuse"
    st["spec"]["sites"]["h200"]["image_auth_file"] = str(auth_file)
    orch.render_roles(st, out_dir)
    orch.save(out_dir / "state.json", st)

    submitted_payloads = []
    cloud = Mock(timeout=1)
    cloud.upload_source = Mock()
    cloud.submit_job = Mock(
        side_effect=lambda payload: (
            submitted_payloads.append(copy.deepcopy(payload)) or "dlc-auth-check-1"
        )
    )
    monkeypatch.setattr(orch, "wait_ready", lambda s, d, c: None)

    orch.submit(st, out_dir, cloud)

    assert len(submitted_payloads) == 1
    assert (
        submitted_payloads[0]["JobSpecs"][0]["ImageConfig"]["Auth"]
        == "top-secret-acr-auth-value"
    )
    persisted_text = (out_dir / "state.json").read_text()
    assert "top-secret-acr-auth-value" not in persisted_text
    persisted_state = orch.read(out_dir / "state.json")
    assert "Auth" not in persisted_state["resources"]["dit"]["expected_job"][
        "JobSpecs"
    ][0].get("ImageConfig", {})


def test_fixed_environment_oss_history_prefixes_and_legacy_thailand_bucket_guard():
    assert (
        orch.resolve_oss_history_prefix(
            {"environment": "selftest", "region": "cn-shanghai"}
        )
        == "oss://leap-world-shanghai/sglang/deployments/selftest"
    )
    assert (
        orch.resolve_oss_history_prefix(
            {"environment": "selftest", "region": "ap-southeast-7"}
        )
        == "oss://leap-worldmodel-thailand-single-az/sglang/deployments/selftest"
    )
    assert (
        orch.resolve_oss_history_prefix(
            {"environment": "dev-h200-smoke", "region": "cn-shanghai"}
        )
        == "oss://leap-world-shanghai/sglang/deployments/selftest/dev-h200-smoke"
    )
    assert (
        orch.resolve_oss_history_prefix({"environment": "test-shanghai-copy"})
        == "oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy"
    )
    assert (
        orch.resolve_oss_history_prefix({"environment": "prod-shanghai"})
        == "oss://leap-world-shanghai/sglang/deployments/prod-shanghai"
    )
    assert (
        orch.resolve_oss_history_prefix({"environment": "prod-thailand"})
        == "oss://leap-worldmodel-thailand-single-az/sglang/deployments/prod-thailand"
    )

    bad_thailand = {
        "environment": "prod-thailand",
        "region": "ap-southeast-7",
        "workspace_id": "ws-th",
        "quota_id": "quota-th",
        "coordinator_url": "http://10.2.0.10:18181",
        "pool_id": "prod-th-pool",
        "room_url": "http://10.2.0.11:8092",
        "wtp_url": "https://wtp-th.example.com",
        "oss_prefix": "oss://leap-worldmodel-thailand/sglang/orchestration",
    }
    with pytest.raises(ValueError, match="leap-worldmodel-thailand-single-az"):
        orch.check_target(bad_thailand)

    good_thailand = {
        **bad_thailand,
        "oss_prefix": "oss://leap-worldmodel-thailand-single-az/sglang/orchestration",
    }
    orch.check_target(good_thailand)


def test_default_oss_archiving_merges_remote_index_and_optional_local_mirror(
    state, tmp_path, monkeypatch, capsys
):
    parser = orch.build_arg_parser()
    for argv in (
        ["up", "--spec", "s.json", "--output", "out", "--commit", "a" * 40],
        ["selftest", "--state", "st.json"],
        ["attach", "--state", "st.json"],
        ["down", "--state", "st.json"],
        ["rollback", "--previous", "--output", "rb"],
    ):
        ns = parser.parse_args(argv)
        assert ns.sync_oss is True
        assert ns.history_dir is None

    ns_disabled = parser.parse_args(
        [
            "attach",
            "--state",
            "st.json",
            "--no-sync-oss",
            "--local-history-dir",
            "/tmp/h",
        ]
    )
    assert ns_disabled.sync_oss is False
    assert ns_disabled.history_dir == Path("/tmp/h")

    # Simulate an in-memory OSS bucket backing fixed environment prefixes.
    fake_oss_objects = {}
    prefix = "oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy"
    legacy_index_bytes = json.dumps(
        {
            "schema": 1,
            "deployments": [
                {
                    "deployment_id": "deploy-20261114T080000Z-existing",
                    "run_id": "0" * 32,
                    "created_at": "2026-11-14T08:00:00Z",
                    "phase": "ready",
                    "environment": "test-shanghai-copy",
                }
            ],
        }
    ).encode()
    fake_oss_objects[f"{prefix}/index.json"] = legacy_index_bytes

    def fake_upload(self, local_path, uri):
        fake_oss_objects[uri] = Path(local_path).read_bytes()

    def fake_download(self, uri, local_path):
        if uri not in fake_oss_objects:
            return False
        p = Path(local_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(fake_oss_objects[uri])
        return True

    def fake_list(self, uri_prefix):
        norm = uri_prefix.rstrip("/") + "/"
        return sorted(
            k for k in fake_oss_objects if k.startswith(norm) and not k.endswith("/")
        )

    monkeypatch.setattr(orch.Cloud, "upload_source", fake_upload)
    monkeypatch.setattr(orch.Cloud, "download_oss_optional", fake_download)
    monkeypatch.setattr(orch.Cloud, "list_oss_objects", fake_list)

    st = copy.deepcopy(state)
    st["deployment_id"] = "deploy-20261115T090000Z-newoss"
    st["created_at"] = "2026-11-15T09:00:00Z"
    st["phase"] = "ready"
    for c in st["spec"]["roles"].values():
        c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
    run_dir = tmp_path / "run-default-oss"
    orch.write_deployment_summary(st, run_dir)
    orch.save_history_record(st, run_dir)
    # Without --local-history-dir, only staging under run_dir/history exists (never repo .sglang-deployments).
    assert (run_dir / "history" / f"{st['deployment_id']}.json").is_file()
    assert orch.optional_local_history_dir(st) is None

    cloud = orch.Cloud(target=st["target"], aliyun_profile="test-op")
    orch._maybe_sync_oss_history(st, run_dir, cloud)
    assert f"{prefix}/{st['deployment_id']}.json" in fake_oss_objects
    assert f"{prefix}/{st['deployment_id']}/deployment.json" in fake_oss_objects
    assert f"{prefix}/{st['deployment_id']}/DEPLOYMENT.md" in fake_oss_objects
    assert any(
        k.startswith(f"{prefix}/entries/{st['deployment_id']}/") and k.endswith(".json")
        for k in fake_oss_objects
    )
    assert any(
        k.startswith(f"{prefix}/records/{st['deployment_id']}/") and k.endswith(".json")
        for k in fake_oss_objects
    )
    # Legacy index.json is NEVER overwritten by writers; read-side merges legacy + immutable shards.
    assert fake_oss_objects[f"{prefix}/index.json"] == legacy_index_bytes

    # CLI `history` and `show` read directly from OSS by default without --local-history-dir!
    orch.main(
        [
            "history",
            "--environment",
            "test-shanghai-copy",
            "--aliyun-profile",
            "test-op",
        ]
    )
    hist_out = json.loads(capsys.readouterr().out)
    assert [d["deployment_id"] for d in hist_out] == [
        "deploy-20261114T080000Z-existing",
        "deploy-20261115T090000Z-newoss",
    ]

    orch.main(
        [
            "show",
            "--environment",
            "test-shanghai-copy",
            "--deployment-id",
            st["deployment_id"],
            "--aliyun-profile",
            "test-op",
        ]
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["deployment_id"] == st["deployment_id"]

    # Optional --local-history-dir mirrors artifacts locally as well.
    mirror_dir = tmp_path / "optional-local-mirror"
    orch._maybe_sync_oss_history(st, run_dir, cloud, history_dir=mirror_dir)
    assert (mirror_dir / f"{st['deployment_id']}.json").is_file()
    assert (mirror_dir / "index.json").is_file()


def test_verify_traffic_quality_stdlib_png_roundtrip_without_pillow(
    tmp_path, monkeypatch
):
    import verify_traffic_quality as vtq

    monkeypatch.setitem(sys.modules, "PIL", None)
    w, h = 32, 24
    frames = []
    for idx in range(3):
        buf = bytearray()
        for y in range(h):
            for x in range(w):
                buf.extend(((x * 7 + idx) % 256, (y * 11 + idx) % 256, (x ^ y) % 256))
        frames.append(bytes(buf))
    out_dir = tmp_path / "png-frames"
    saved = vtq.save_frames_png(frames, out_dir, width=w, height=h)
    assert len(saved) == 3
    loaded, lw, lh = vtq.load_frames_png(out_dir)
    assert (lw, lh) == (w, h)
    assert loaded == frames


def test_fail_closed_baseline_manifest_and_cli_mode_separation(tmp_path):
    import verify_traffic_quality as vtq

    parser = vtq.build_arg_parser()
    # Candidate `run` and `compare-dirs` MUST require --baseline-dir (cannot omit it).
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "compare-dirs",
                "--candidate-dir",
                str(tmp_path / "cand"),
                "--output",
                str(tmp_path / "cmp.json"),
            ]
        )
    # `record-baseline` does not accept `--state` so it can never mutate a candidate deployment.
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "record-baseline",
                "--state",
                str(tmp_path / "state.json"),
                "--output-dir",
                str(tmp_path / "base"),
            ]
        )

    w, h = 32, 24
    frames = []
    for idx in range(16):
        buf = bytearray()
        for y in range(h):
            for x in range(w):
                buf.extend(
                    ((x * 5 + idx * 7) % 256, (y * 9 + idx * 3) % 256, (x + y) % 256)
                )
        frames.append(bytes(buf))

    unpinned_dir = tmp_path / "unpinned-base"
    vtq.save_frames_png(frames, unpinned_dir / "frames", width=w, height=h)
    # Missing baseline_manifest.json fails closed.
    with pytest.raises(ValueError, match="missing required baseline_manifest.json"):
        vtq.load_verified_baseline(unpinned_dir, expected_width=w, expected_height=h)

    manifest = vtq.write_baseline_manifest(
        unpinned_dir,
        frames=frames,
        width=w,
        height=h,
        case_id="shanghai_drive_baseline",
        seed=886410,
        inference_steps=2,
    )
    loaded, lw, lh, loaded_manifest = vtq.load_verified_baseline(
        unpinned_dir,
        expected_width=w,
        expected_height=h,
        expected_case_id="shanghai_drive_baseline",
        expected_seed=886410,
        expected_inference_steps=2,
    )
    assert loaded == frames
    assert (lw, lh) == (w, h)
    assert loaded_manifest["frames_sha256"] == manifest["frames_sha256"]

    # Seed / case_id / inference_steps mismatch fails closed!
    with pytest.raises(ValueError, match="Baseline seed"):
        vtq.load_verified_baseline(
            unpinned_dir, expected_width=w, expected_height=h, expected_seed=999999
        )
    with pytest.raises(ValueError, match="Baseline case_id"):
        vtq.load_verified_baseline(
            unpinned_dir,
            expected_width=w,
            expected_height=h,
            expected_case_id="wrong_case",
        )
    with pytest.raises(ValueError, match="Baseline inference_steps"):
        vtq.load_verified_baseline(
            unpinned_dir,
            expected_width=w,
            expected_height=h,
            expected_inference_steps=4,
        )

    # Tampering a frame invalidates frames_sha256 and fails closed!
    vtq.save_frames_png(
        [b"\x10" * (w * h * 3)] + frames[1:], unpinned_dir / "frames", width=w, height=h
    )
    with pytest.raises(ValueError, match="do not match pinned baseline_manifest.json"):
        vtq.load_verified_baseline(unpinned_dir, expected_width=w, expected_height=h)


def test_traffic_ready_gate_requires_traffic_accepted_and_fails_on_failed_session(
    state, tmp_path
):
    import verify_traffic_quality as vtq

    st = copy.deepcopy(state)
    st["phase"] = "ready"
    for c in st["spec"]["roles"].values():
        c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
    st["attachment"] = {
        "identities": {
            r: {
                "process_id": f"{r}-worker",
                "boot_epoch": f"{r}-boot",
                "service_id": f"{r}-worker/0",
            }
            for r in orch.ROLES
        }
    }
    run_dir = tmp_path / "dep-traffic-gate"
    run_dir.mkdir()
    orch.save(run_dir / "state.json", st)

    # 1. Pool attached alone without live traffic verification MUST have traffic_ready == False.
    pre_summary = orch.write_deployment_summary(st, run_dir)
    assert pre_summary["pool_attached"] is True
    assert pre_summary["acceptance_gates"]["traffic_ready"] is False

    # 2. Failed live traffic verification (e.g. lease not released or insufficient frames) MUST keep traffic_ready == False!
    failed_traffic_report = {
        "schema": 1,
        "deployment_id": st.get("deployment_id"),
        "traffic_accepted": False,
        "warmup_verified": True,
        "quality_unchanged": False,
        "is_baseline": False,
        "baseline_identity": None,
        "intrinsic_quality": {"passed": False},
        "baseline_comparison": None,
    }
    failed_summary = vtq.attach_verification_to_deployment(
        run_dir / "state.json", failed_traffic_report
    )
    assert failed_summary["pool_attached"] is True
    assert failed_summary["acceptance_gates"]["traffic_ready"] is False
    assert failed_summary["acceptance_gates"]["quality_unchanged"] is False

    # 3. Only when pool_attached is True AND traffic_accepted is True does traffic_ready become True.
    passed_traffic_report = {
        **failed_traffic_report,
        "traffic_accepted": True,
    }
    passed_summary = vtq.attach_verification_to_deployment(
        run_dir / "state.json", passed_traffic_report
    )
    assert passed_summary["acceptance_gates"]["traffic_ready"] is True


def test_reused_roles_missing_or_false_warmed_up_fail_closed_in_precheck_and_summary(
    state, tmp_path, monkeypatch
):
    import verify_traffic_quality as vtq

    # 1. Precheck: reused encoder missing `warmed_up` field or reporting `warmed_up=False` fails closed.
    encoder_health_payload = {"ready": True, "role": "encoder"}  # No warmed_up key!

    def fake_http_json(url, payload=None, token=None, timeout=10.0):
        if url.endswith("/health"):
            if "31000" in url:
                return dict(encoder_health_payload)
            role = "dit" if "33000" in url else "decoder"
            return {"ready": True, "role": role, "warmed_up": True}
        raise AssertionError(url)

    monkeypatch.setattr(vtq, "_http_json", fake_http_json)
    urls = {
        "encoder": "http://10.0.0.10:31000",
        "dit": "http://10.0.0.11:33000",
        "decoder": "http://10.0.0.12:36000",
    }
    rep_missing = vtq.check_role_warmup_and_pool_capacity(
        role_urls=urls,
        coordinator_url=None,
        routing_revision=None,
        compatibility=None,
        profile="480",
        network_domain="shanghai-vpc-h100-h200",
        room_token=None,
    )
    assert rep_missing["all_roles_warmed_up"] is False
    assert rep_missing["roles"]["encoder"]["warmed_up"] is False
    assert rep_missing["roles"]["encoder"]["explicit_warmup_flag"] is False

    encoder_health_payload["warmed_up"] = False
    rep_false = vtq.check_role_warmup_and_pool_capacity(
        role_urls=urls,
        coordinator_url=None,
        routing_revision=None,
        compatibility=None,
        profile="480",
        network_domain="shanghai-vpc-h100-h200",
        room_token=None,
    )
    assert rep_false["all_roles_warmed_up"] is False
    assert rep_false["roles"]["encoder"]["warmed_up"] is False

    # 2. Deployment summary: reused encoder/decoder without explicit warmed_up=True MUST NOT default to True!
    st = copy.deepcopy(state)
    st["phase"] = "ready"
    st["spec"]["existing_services"] = {
        "encoder": {
            "http": "http://10.0.0.10:31000",
            "worker_id": "enc-legacy",
            "gpu": "GPU-enc",
            "boot_epoch": "be-enc",
        },
        "decoder": {
            "http": "http://10.0.0.12:36000",
            "worker_id": "dec-warmed",
            "gpu": "GPU-dec",
            "boot_epoch": "be-dec",
        },
    }
    st["spec"]["roles"] = {"dit": st["spec"]["roles"]["dit"]}
    st["resources"]["encoder"] = orch.reused_resource(
        st["run_id"], "model1027", "encoder", st["spec"]["existing_services"]["encoder"]
    )
    st["resources"]["decoder"] = orch.reused_resource(
        st["run_id"], "model1027", "decoder", st["spec"]["existing_services"]["decoder"]
    )
    # Dit and decoder explicitly report warmed_up=True, but reused encoder omits warmed_up:
    st["resources"]["dit"]["receipt"]["warmed_up"] = True
    st["resources"]["decoder"]["receipt"]["warmed_up"] = True
    assert "warmed_up" not in st["resources"]["encoder"]["receipt"]

    sum_missing = orch.write_deployment_summary(st, tmp_path / "sum-missing")
    assert sum_missing["roles"]["encoder"]["warmup"]["warmed_up"] is False
    assert sum_missing["roles"]["dit"]["warmup"]["warmed_up"] is True
    assert sum_missing["roles"]["decoder"]["warmup"]["warmed_up"] is True
    assert sum_missing["acceptance_gates"]["all_roles_warmed_up"] is False

    # Reused decoder explicitly reporting warmed_up=False also fails closed:
    st["resources"]["encoder"]["receipt"]["warmed_up"] = True
    st["resources"]["decoder"]["receipt"]["warmed_up"] = False
    sum_false = orch.write_deployment_summary(st, tmp_path / "sum-false")
    assert sum_false["roles"]["decoder"]["warmup"]["warmed_up"] is False
    assert sum_false["acceptance_gates"]["all_roles_warmed_up"] is False

    # Only when all three roles (owned + reused) explicitly report warmed_up=True does the gate pass:
    st["resources"]["decoder"]["receipt"]["warmed_up"] = True
    sum_all_ok = orch.write_deployment_summary(st, tmp_path / "sum-ok")
    assert sum_all_ok["acceptance_gates"]["all_roles_warmed_up"] is True


@pytest.mark.parametrize("override", [False, True])
def test_quality_cli_reads_coordinator_from_state_target(
    state, tmp_path, monkeypatch, override
):
    import verify_traffic_quality as vtq

    state_path = tmp_path / "state.json"
    orch.save(state_path, state)
    monkeypatch.setattr(
        vtq, "load_verified_baseline", lambda *a, **kw: ([], 832, 480, {})
    )
    expected = (
        "http://override:18181" if override else state["target"]["coordinator_url"]
    )

    def precheck(**kwargs):
        assert kwargs["coordinator_url"] == expected
        raise RuntimeError("precheck reached expected Coordinator")

    monkeypatch.setattr(vtq, "check_role_warmup_and_pool_capacity", precheck)
    argv = [
        "run",
        "--state",
        str(state_path),
        "--output-dir",
        str(tmp_path / "quality"),
        "--baseline-dir",
        str(tmp_path / "baseline"),
    ]
    if override:
        argv += ["--coordinator-url", expected]
    with pytest.raises(RuntimeError, match="precheck reached expected Coordinator"):
        vtq.main(argv)


@pytest.mark.parametrize("difference", ["environment", "model_id", "route"])
def test_previous_rollback_skips_unrelated_deployments(state, tmp_path, difference):
    store = tmp_path / "scope-history"
    for i, dep_id in enumerate(("stable", "unrelated", "current")):
        st = copy.deepcopy(state)
        st.update(
            deployment_id=dep_id, created_at=f"2026-11-15T0{i}:00:00Z", phase="ready"
        )
        if dep_id == "unrelated":
            if difference == "environment":
                st["target"]["environment"] = "other-environment"
            else:
                st["spec"][difference] = "other-" + difference
        orch.save_history_record(st, history_dir=store)
    assert orch.resolve_previous_deployment_id(store, "current") == "stable"
    (store / "stable.json").unlink()
    # With no matching stable version, fail instead of falling back to an unrelated entry.
    index = orch.read(store / "index.json")
    index["deployments"] = [
        e for e in index["deployments"] if e["deployment_id"] != "stable"
    ]
    orch.save(store / "index.json", index)
    with pytest.raises(ValueError, match="No previous stable deployment matches"):
        orch.resolve_previous_deployment_id(store, "current")
    with pytest.raises(ValueError, match="absent from history"):
        orch.resolve_previous_deployment_id(store, "missing")


def test_explicit_rollback_scope_mismatch_cannot_stop_current(
    state, tmp_path, monkeypatch
):
    state["deployment_id"] = "current"
    current_path = tmp_path / "current.json"
    orch.save(current_path, state)
    target = copy.deepcopy(state)
    target["deployment_id"] = "unrelated"
    target["target"]["environment"] = "other-environment"
    store = tmp_path / "scope-history"
    orch.save_history_record(target, history_dir=store)
    down = Mock()
    monkeypatch.setattr(orch, "down", down)
    with pytest.raises(ValueError, match="does not match the current deployment"):
        orch.rollback_deployment(
            store,
            "unrelated",
            tmp_path / "rollback",
            Mock(),
            current_state_path=current_path,
            execute=True,
            yes=True,
        )
    down.assert_not_called()
    assert not (tmp_path / "rollback").exists()


def test_rollback_sla_includes_stopping_previous_deployment(
    state, tmp_path, monkeypatch
):
    state["phase"] = "ready"
    state["selftest"] = dict(
        status="passed",
        source=state["source"],
        worker_ids={
            r: state["resources"][r]["receipt"]["worker_id"] for r in orch.ROLES
        },
    )
    state["timings_sec"] = dict(up_total_sec=500, selftest_sec=20, stop_previous_sec=40)
    monkeypatch.setenv("ROLE_POOL_SIGNING_KEY", "s" * 32)
    monkeypatch.setenv("ROLE_POOL_OPERATOR_TOKEN", "test")
    monkeypatch.setattr(orch, "worker_configs", lambda st: {})
    monkeypatch.setattr(
        orch, "http", lambda *a, **kw: dict(backend="role_pool", profile="480")
    )
    monkeypatch.setattr(orch, "_preflight_or_reconcile_owned_endpoints", Mock())
    monkeypatch.setattr(orch, "_attach_owned_endpoints_in_place", Mock())
    monkeypatch.setattr(orch, "verify_pool", Mock())
    ticks = iter((0, 0, 50))
    monkeypatch.setattr(orch.time, "monotonic", lambda: next(ticks))
    orch.attach(state, tmp_path, Mock(timeout=100), execute=True, in_place=True)
    assert state["timings_sec"]["end_to_end_sec"] == 610
    assert state["timings_sec"]["within_10min_sla"] is False


def test_oss_history_distinguishes_not_found_from_errors_and_never_overwrites_index(
    state, tmp_path, monkeypatch, capsys
):
    from types import SimpleNamespace

    cloud = orch.Cloud(target=copy.deepcopy(orch.TARGET), aliyun_profile="test-op")

    # 1. Explicit 404 NoSuchKey returns False for download_oss_optional and [] for empty list_oss_objects.
    monkeypatch.setattr(
        orch.subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(
            returncode=1,
            stdout="FinishWithError: Scanned 1 objects. Error num: 1.\n",
            stderr='\x1b[1;31mERROR: oss: service returned error: StatusCode=404, ErrorCode=NoSuchKey, ErrorMessage="The specified key does not exist."\x1b[0m\n',
        ),
    )
    assert (
        cloud.download_oss_optional(
            "oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy/missing.json",
            tmp_path / "missing.json",
        )
        is False
    )
    assert not (tmp_path / "missing.json").exists()

    monkeypatch.setattr(
        orch.subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(
            returncode=0,
            stdout="Object Number is: 0\n\n0.12(s) elapsed\n",
            stderr="",
        ),
    )
    assert (
        cloud.list_oss_objects(
            "oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy/entries/"
        )
        == []
    )

    # 2. Non-NotFound CLI errors (403 AccessDenied, 429 SlowDown, 503 ServiceUnavailable,
    #    404 NoSuchBucket, network timeout, unknown silent rc=1) MUST fail-closed with RuntimeError!
    for bad_stderr in (
        "ERROR: oss: service returned error: StatusCode=403, ErrorCode=AccessDenied",
        "ERROR: oss: service returned error: StatusCode=429, ErrorCode=SlowDown",
        "ERROR: oss: service returned error: StatusCode=503, ErrorCode=ServiceUnavailable",
        "ERROR: oss: service returned error: StatusCode=404, ErrorCode=NoSuchBucket",
        "ERROR: dial tcp 10.0.0.1:443: i/o timeout",
        "",
    ):
        monkeypatch.setattr(
            orch.subprocess,
            "run",
            lambda argv, _err=bad_stderr, **kw: SimpleNamespace(
                returncode=1, stdout="", stderr=_err
            ),
        )
        with pytest.raises(RuntimeError, match="OSS download failed"):
            cloud.download_oss_optional(
                "oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy/index.json",
                tmp_path / "err.json",
            )
        with pytest.raises(RuntimeError, match="OSS list failed"):
            cloud.list_oss_objects(
                "oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy/entries/"
            )

    # 3. End-to-end simulated OSS with legacy index.json (v1) + shard (v2):
    #    Transient read/list failure MUST NEVER overwrite remote index.json and MUST fail-closed
    #    on `history` and `rollback --previous` rather than treating history as empty or picking wrong version!
    fake_oss = {}
    prefix = "oss://leap-world-shanghai/sglang/deployments/test-shanghai-copy"
    read_failure = {"mode": None}

    def _cli_sim(argv, **kw):
        op_idx = argv.index("oss")
        subcmd = argv[op_idx + 1]
        if subcmd == "cp":
            src, dst = argv[op_idx + 3], argv[op_idx + 4]
            if src.startswith("oss://"):
                if read_failure["mode"] == "download":
                    return SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr="ERROR: oss: service returned error: StatusCode=503, ErrorCode=ServiceUnavailable",
                    )
                if src not in fake_oss:
                    return SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr='ERROR: oss: service returned error: StatusCode=404, ErrorCode=NoSuchKey, ErrorMessage="The specified key does not exist."',
                    )
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_bytes(fake_oss[src])
                return SimpleNamespace(returncode=0, stdout="Succeed\n", stderr="")
            else:
                fake_oss[dst] = Path(src).read_bytes()
                return SimpleNamespace(returncode=0, stdout="Succeed\n", stderr="")
        if subcmd == "ls":
            if read_failure["mode"] == "list":
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="ERROR: oss: service returned error: StatusCode=403, ErrorCode=AccessDenied",
                )
            uri_prefix = argv[op_idx + 2]
            matches = sorted(
                k for k in fake_oss if k.startswith(uri_prefix) and not k.endswith("/")
            )
            lines = "\n".join(matches)
            body = (
                lines + "\n" if lines else ""
            ) + f"Object Number is: {len(matches)}\n"
            return SimpleNamespace(returncode=0, stdout=body, stderr="")
        raise AssertionError(f"Unexpected oss command: {argv}")

    monkeypatch.setattr(orch.subprocess, "run", _cli_sim)

    def _make_dep(dep_id, created_at):
        st = copy.deepcopy(state)
        st["deployment_id"] = dep_id
        st["created_at"] = created_at
        st["phase"] = "ready"
        for c in st["spec"]["roles"].values():
            c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
        d = tmp_path / dep_id
        orch.save(d / "state.json", st)
        orch.write_deployment_summary(st, d)
        rec = orch.save_history_record(st, d)
        return st, d, rec

    st_v1, d_v1, rec_v1 = _make_dep(
        "deploy-20261115T010000Z-v1", "2026-11-15T01:00:00Z"
    )
    # Seed v1 into legacy remote index.json + direct record to verify legacy preservation!
    legacy_index_raw = json.dumps(
        {"schema": 1, "deployments": [orch.build_history_index_entry(rec_v1)]},
        indent=2,
        sort_keys=True,
    ).encode()
    fake_oss[f"{prefix}/index.json"] = legacy_index_raw
    fake_oss[f"{prefix}/{st_v1['deployment_id']}.json"] = (
        d_v1 / "history" / f"{st_v1['deployment_id']}.json"
    ).read_bytes()

    st_v2, d_v2, _ = _make_dep("deploy-20261115T020000Z-v2", "2026-11-15T02:00:00Z")
    cloud.sync_history_artifacts(
        d_v2 / "history", st_v2["deployment_id"], summary_dir=d_v2
    )

    # Inject transient 503 on OSS downloads while archiving v3:
    # Remote legacy `index.json` MUST NOT be overwritten!
    read_failure["mode"] = "download"
    st_v3, d_v3, _ = _make_dep("deploy-20261115T030000Z-v3", "2026-11-15T03:00:00Z")
    cloud.sync_history_artifacts(
        d_v3 / "history", st_v3["deployment_id"], summary_dir=d_v3
    )
    assert fake_oss[f"{prefix}/index.json"] == legacy_index_raw

    # While OSS download or list is failing, `history` and `rollback --previous` MUST fail-closed!
    for mode in ("download", "list"):
        read_failure["mode"] = mode
        with pytest.raises(RuntimeError, match="OSS (download|list) failed"):
            orch.main(
                [
                    "history",
                    "--environment",
                    "test-shanghai-copy",
                    "--aliyun-profile",
                    "test-op",
                ]
            )
        with pytest.raises(RuntimeError, match="OSS (download|list) failed"):
            orch.main(
                [
                    "rollback",
                    "--previous",
                    "--current-state",
                    str(d_v3 / "state.json"),
                    "--output",
                    str(tmp_path / f"rb-fail-{mode}"),
                    "--aliyun-profile",
                    "test-op",
                ]
            )

    # Once transient failure clears, all 3 deployments [v1, v2, v3] are intact and `rollback --previous` picks v2!
    read_failure["mode"] = None
    orch.main(
        [
            "history",
            "--environment",
            "test-shanghai-copy",
            "--aliyun-profile",
            "test-op",
        ]
    )
    recovered = json.loads(capsys.readouterr().out)
    assert [e["deployment_id"] for e in recovered] == [
        "deploy-20261115T010000Z-v1",
        "deploy-20261115T020000Z-v2",
        "deploy-20261115T030000Z-v3",
    ]
    orch.main(
        [
            "rollback",
            "--previous",
            "--current-state",
            str(d_v3 / "state.json"),
            "--output",
            str(tmp_path / "rb-plan-ok"),
            "--aliyun-profile",
            "test-op",
        ]
    )
    rb_plan_out = json.loads(capsys.readouterr().out)
    assert rb_plan_out["rollback_of"] == "deploy-20261115T020000Z-v2"


def test_concurrent_oss_history_writers_preserve_all_entries_and_rollback_previous_is_exact(
    state, tmp_path, monkeypatch, capsys
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    oss_lock = threading.Lock()
    fake_oss = {}

    def fake_upload(self, local_path, uri):
        data = Path(local_path).read_bytes()
        with oss_lock:
            fake_oss[uri] = data

    def fake_download(self, uri, local_path):
        with oss_lock:
            data = fake_oss.get(uri)
        if data is None:
            return False
        p = Path(local_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return True

    def fake_list(self, uri_prefix):
        norm = uri_prefix.rstrip("/") + "/"
        with oss_lock:
            return sorted(
                k for k in fake_oss if k.startswith(norm) and not k.endswith("/")
            )

    monkeypatch.setattr(orch.Cloud, "upload_source", fake_upload)
    monkeypatch.setattr(orch.Cloud, "download_oss_optional", fake_download)
    monkeypatch.setattr(orch.Cloud, "list_oss_objects", fake_list)

    specs = [
        ("deploy-20261115T010000Z-v1", "2026-11-15T01:00:00Z", "ready", None),
        ("deploy-20261115T020000Z-v2", "2026-11-15T02:00:00Z", "ready", None),
        (
            "deploy-20261115T023000Z-other-model",
            "2026-11-15T02:30:00Z",
            "ready",
            "other-model-id",
        ),
        ("deploy-20261115T030000Z-v3", "2026-11-15T03:00:00Z", "ready", None),
    ]
    prepared_runs = {}
    for dep_id, created_at, phase, custom_model in specs:
        st = copy.deepcopy(state)
        st["deployment_id"] = dep_id
        st["created_at"] = created_at
        st["phase"] = phase
        if custom_model:
            st["spec"]["model_id"] = custom_model
        for c in st["spec"]["roles"].values():
            c["secret_files"] = {k: "/secrets/" + k for k in launch.SECRET_ENV}
        d = tmp_path / dep_id
        orch.save(d / "state.json", st)
        orch.write_deployment_summary(st, d)
        orch.save_history_record(st, d)
        prepared_runs[dep_id] = (st, d)

    # Also prepare a lifecycle update for v1 transitioning ready -> stopped in its own staging dir!
    st_v1_stopped = copy.deepcopy(prepared_runs["deploy-20261115T010000Z-v1"][0])
    st_v1_stopped["phase"] = "stopped"
    d_v1_stopped = tmp_path / "v1-stopped-stage"
    orch.save(d_v1_stopped / "state.json", st_v1_stopped)
    orch.write_deployment_summary(st_v1_stopped, d_v1_stopped)
    orch.save_history_record(st_v1_stopped, d_v1_stopped)

    tasks = [
        (st["deployment_id"], d, st["target"]) for st, d in prepared_runs.values()
    ] + [(st_v1_stopped["deployment_id"], d_v1_stopped, st_v1_stopped["target"])]

    barrier = threading.Barrier(len(tasks))

    def _concurrent_writer(item):
        dep_id, run_dir, target = item
        c = orch.Cloud(target=target, aliyun_profile="test-op")
        barrier.wait(timeout=10)
        return c.sync_history_artifacts(
            run_dir / "history", dep_id, summary_dir=run_dir
        )

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        results = list(pool.map(_concurrent_writer, tasks))
    assert len(results) == 5

    # Verify zero lost updates: all 4 deployment_ids are present, and v1 merged deterministically to `stopped`.
    orch.main(
        [
            "history",
            "--environment",
            "test-shanghai-copy",
            "--aliyun-profile",
            "test-op",
        ]
    )
    merged_hist = json.loads(capsys.readouterr().out)
    assert [e["deployment_id"] for e in merged_hist] == [
        "deploy-20261115T010000Z-v1",
        "deploy-20261115T020000Z-v2",
        "deploy-20261115T023000Z-other-model",
        "deploy-20261115T030000Z-v3",
    ]
    by_id = {e["deployment_id"]: e for e in merged_hist}
    assert by_id["deploy-20261115T010000Z-v1"]["phase"] == "stopped"

    # Verify `rollback --previous` from v3 skips `other-model` and selects exact predecessor `v2` (never misses v2!).
    _, d_v3 = prepared_runs["deploy-20261115T030000Z-v3"]
    orch.main(
        [
            "rollback",
            "--previous",
            "--current-state",
            str(d_v3 / "state.json"),
            "--output",
            str(tmp_path / "rb-concurrent-ok"),
            "--aliyun-profile",
            "test-op",
        ]
    )
    rb_plan = json.loads(capsys.readouterr().out)
    assert rb_plan["rollback_of"] == "deploy-20261115T020000Z-v2"


def test_resolve_previous_deployment_id_fails_closed_when_predecessor_record_missing_or_corrupt(
    state, tmp_path
):
    store = tmp_path / "corrupt-pred-store"
    for dep_id, ts in (
        ("deploy-20261115T010000Z-v0", "2026-11-15T01:00:00Z"),
        ("deploy-20261115T020000Z-v1", "2026-11-15T02:00:00Z"),
        ("deploy-20261115T030000Z-v2", "2026-11-15T03:00:00Z"),
    ):
        st = copy.deepcopy(state)
        st["deployment_id"] = dep_id
        st["created_at"] = ts
        st["phase"] = "ready"
        orch.save_history_record(st, history_dir=store)

    # 1. If immediate predecessor v1 lacks rollback_bundle, MUST fail-closed (never skip v1 to pick stale v0!).
    v1_file = store / "deploy-20261115T020000Z-v1.json"
    v1_rec = orch.read(v1_file)
    del v1_rec["rollback_bundle"]
    orch.save(v1_file, v1_rec)
    with pytest.raises(ValueError, match="missing a valid rollback_bundle"):
        orch.resolve_previous_deployment_id(store, "deploy-20261115T030000Z-v2")

    # 2. If immediate predecessor v1 record file is missing while indexed, MUST raise FileNotFoundError!
    v1_file.unlink()
    with pytest.raises(FileNotFoundError, match="deploy-20261115T020000Z-v1"):
        orch.resolve_previous_deployment_id(store, "deploy-20261115T030000Z-v2")

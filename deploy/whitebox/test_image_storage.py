import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_storage import inspect_local, validate_history

SOURCE = {"created_by": "COPY /runtime-source/sglang /app/python/sglang # buildkit"}
SDK = {"created_by": "COPY /opt/worldkit /opt/worldkit # buildkit"}
DELETE = {"created_by": "RUN /bin/sh -c rm -rf /app /opt/worldkit # buildkit"}


@pytest.fixture(autouse=True)
def simulated_export_capacity(monkeypatch):
    """Metadata archives are synthetic; runner free space is not their input.

    The dedicated low-space test overrides this value and still exercises the
    real pre-save guard. Production checks and their 15 GiB reserve are unchanged.
    """
    import shutil
    from types import SimpleNamespace

    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1 << 40))


@pytest.mark.parametrize("component", ["cpu", "room"])
def test_visible_deletion_cannot_hide_old_sdk_layer(component):
    history = [SDK, SOURCE, DELETE, SOURCE]
    if component == "room":
        history.append(SDK)
    with pytest.raises(ValueError, match="SDK COPY"):
        validate_history({"history": history}, component)


@pytest.mark.parametrize("component", ["cpu", "room"])
def test_clean_dependency_base_keeps_one_source_and_room_only_sdk(component):
    history = [{"created_by": "RUN pip install -r requirements.txt"}, SOURCE]
    if component == "room":
        history.extend(
            [SDK, {"created_by": "ENV CLI=/opt/worldkit/cli", "empty_layer": True}]
        )
    result = validate_history({"history": history}, component)
    assert result["sdk_copies"] == (component == "room")


def test_legacy_docker_history_sdk_copy_is_recognized():
    old = {"created_by": "/bin/sh -c #(nop) COPY dir:123 in /opt/worldkit "}
    with pytest.raises(ValueError, match="SDK COPY"):
        validate_history({"history": [old, DELETE, SOURCE]}, "cpu")


def test_native_codec_copy_is_not_a_second_source_tree():
    native = {
        "created_by": "COPY /tmp/libwhitebox_x264.so /app/python/sglang/multimodal_gen/runtime/realtime/whitebox/native/libwhitebox_x264.so # buildkit"
    }
    result = validate_history({"history": [SOURCE, SDK, native]}, "room")
    assert result["source_copies"] == 1


@pytest.mark.parametrize("history", [[], None, [SOURCE, SOURCE], [SDK]])
def test_missing_or_duplicate_source_is_not_attested(history):
    with pytest.raises(ValueError):
        validate_history({"history": history}, "room")


def test_local_inspection_fences_a_moving_tag(tmp_path):
    image, config, files = saved_image([SOURCE], layout="classic")
    calls = []

    def run(args, **kwargs):
        import json

        calls.append(args)
        if args[1:3] == ["image", "inspect"]:
            return json.dumps(
                [{"Id": image, "Size": 123, "Os": "linux", "Architecture": "amd64"}]
            )
        assert args[1:3] == ["image", "save"] and args[-1] == image
        write_archive(args[args.index("--output") + 1], files)
        return ""

    result = inspect_local("registry/cpu:tag", "cpu", run=run)
    assert result["image_id"] == image
    assert len(calls) == 2


def test_browser_repacking_does_not_inherit_deleted_dependency_layers():
    import shlex

    recipe = (
        Path(__file__).resolve().parents[2]
        / "docker/whitebox-browser-dependencies.Dockerfile"
    )
    stages = []
    for line in recipe.read_text().replace("\\\n", " ").splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == "FROM":
            stages.append(parts[1:])
    assert stages[-1] == ["scratch", "AS", "browser-root"]
    # A normal final Room image consumes this clean root and adds its source/SDK
    # once. Inheriting dependency-files would retain the negative-control SDK.
    clean_root = {"created_by": "COPY / / # buildkit"}
    assert (
        validate_history({"history": [clean_root, SOURCE, SDK]}, "room")["sdk_copies"]
        == 1
    )


# The containerd image store can report Size=0B for real COPY/RUN layers.
# Use the saved config's immutable history/empty_layer metadata instead.
def saved_image(history=None, *, layout="oci", config=None):
    import hashlib
    import json

    def seal(value):
        raw = json.dumps(value, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        return digest, raw

    if config is None:
        config = {
            "os": "linux",
            "architecture": "amd64",
            "history": history,
            "rootfs": {
                "type": "layers",
                "diff_ids": [
                    "sha256:" + "d" * 64
                    for row in history
                    if not row.get("empty_layer")
                ],
            },
        }
    digest, raw = seal(config)
    if layout == "classic":
        return (
            "sha256:" + digest,
            config,
            {
                digest + ".json": raw,
                "manifest.json": json.dumps(
                    [{"Config": digest + ".json", "Layers": []}]
                ).encode(),
            },
        )
    files = {"blobs/sha256/" + digest: raw}
    manifest_digest, manifest = seal(
        {"schemaVersion": 2, "config": {"digest": "sha256:" + digest}, "layers": []}
    )
    files["blobs/sha256/" + manifest_digest] = manifest
    index_digest, index = seal(
        {
            "schemaVersion": 2,
            "manifests": [
                {
                    "digest": "sha256:" + manifest_digest,
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ],
        }
    )
    files["blobs/sha256/" + index_digest] = index
    return "sha256:" + index_digest, config, files


def write_archive(path, files):
    import io
    import tarfile

    with tarfile.open(path, "w") as archive:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


@pytest.mark.parametrize("layout", ["classic", "oci"])
@pytest.mark.parametrize("component", ["cpu", "room"])
def test_real_config_history_survives_containerd_zero_byte_display(layout, component):
    import json

    history = [SOURCE, {"created_by": "ENV PATH=/opt/worldkit", "empty_layer": True}]
    if component == "room":
        history.append(SDK)
    image, _, files = saved_image(history, layout=layout)

    def run(args, **kwargs):
        if args[1:3] == ["image", "inspect"]:
            return json.dumps(
                [{"Id": image, "Size": 123, "Os": "linux", "Architecture": "amd64"}]
            )
        assert args[1:3] == ["image", "save"], (
            "human-readable history Size is not authoritative"
        )
        write_archive(args[args.index("--output") + 1], files)
        return ""

    result = inspect_local("registry/candidate:tag", component, run=run)
    assert result["source_copies"] == 1
    assert result["sdk_copies"] == (component == "room")


@pytest.mark.parametrize("layout", ["classic", "oci"])
def test_saved_history_keeps_hidden_old_sdk_rejection(layout):
    import json

    image, _, files = saved_image([SDK, DELETE, SOURCE], layout=layout)

    def run(args, **kwargs):
        if args[1:3] == ["image", "inspect"]:
            return json.dumps(
                [{"Id": image, "Size": 123, "Os": "linux", "Architecture": "amd64"}]
            )
        write_archive(args[args.index("--output") + 1], files)
        return ""

    with pytest.raises(ValueError, match="SDK COPY"):
        inspect_local("candidate", "cpu", run=run)


@pytest.mark.parametrize(
    "fault", ["digest", "platform", "history-layer-count", "layer-count"]
)
def test_saved_image_identity_and_layer_metadata_fail_closed(fault):
    import json

    history = [SOURCE]
    if fault == "history-layer-count":
        history = [dict(SOURCE, empty_layer=True)]
    image, _, files = saved_image(
        history,
        config={
            "os": "linux",
            "architecture": "amd64",
            "history": history,
            "rootfs": {"diff_ids": []},
        }
        if fault == "layer-count"
        else None,
    )
    if fault == "digest":
        files["blobs/sha256/" + image.removeprefix("sha256:")] += b" "
    if fault == "history-layer-count":
        # Empty-layer metadata cannot count as a source COPY.
        expected = "source COPY"
    elif fault == "layer-count":
        expected = "filesystem layers"
    else:
        expected = "identity|platform"

    def run(args, **kwargs):
        if args[1:3] == ["image", "inspect"]:
            return json.dumps(
                [
                    {
                        "Id": image,
                        "Size": 123,
                        "Os": "linux",
                        "Architecture": "arm64" if fault == "platform" else "amd64",
                    }
                ]
            )
        write_archive(args[args.index("--output") + 1], files)
        return ""

    with pytest.raises(ValueError, match=expected):
        inspect_local("candidate", "cpu", run=run)


def test_captured_containerd_config_keeps_real_zero_display_copy_layers():
    import json

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/containerd-zero-size-history.json"
        ).read_text()
    )
    assert all(row["Size"] == "0B" for row in fixture["display_history"])
    image, _, files = saved_image(config=fixture["config"])

    def run(args, **kwargs):
        if args[1:3] == ["image", "inspect"]:
            return json.dumps(
                [{"Id": image, "Size": 123, "Os": "linux", "Architecture": "amd64"}]
            )
        write_archive(args[args.index("--output") + 1], files)
        return ""

    assert inspect_local("captured-cpu", "cpu", run=run)["source_copies"] == 1


@pytest.mark.parametrize("layout", ["classic", "oci", "direct-oci-config"])
def test_final_config_platform_must_match_even_when_index_descriptor_matches(
    tmp_path, layout
):
    import json

    config = {
        "os": "windows",
        "architecture": "arm64",
        "history": [SOURCE],
        "rootfs": {"diff_ids": ["sha256:" + "d" * 64]},
    }
    image, _, files = saved_image(
        config=config, layout="classic" if layout == "classic" else "oci"
    )
    if layout == "direct-oci-config":
        import hashlib

        raw = json.dumps(config, separators=(",", ":")).encode()
        image = "sha256:" + hashlib.sha256(raw).hexdigest()
    path = tmp_path / "wrong-platform.tar"
    write_archive(path, files)
    from image_storage import saved_config

    with pytest.raises(ValueError, match="platform"):
        saved_config(path, image, ("linux", "amd64"))


def test_export_checks_its_actual_temp_filesystem_before_save(monkeypatch):
    import json
    import shutil
    from types import SimpleNamespace

    image, _, _ = saved_image([SOURCE])
    calls, measured = [], []

    def run(args, **kwargs):
        calls.append(args)
        assert args[1:3] == ["image", "inspect"], (
            "image save must not run on a full scratch disk"
        )
        return json.dumps(
            [{"Id": image, "Size": 123, "Os": "linux", "Architecture": "amd64"}]
        )

    def usage(path):
        measured.append(Path(path))
        # DockerRootDir could have ample space on a different device; only the
        # actual allocated export directory protects this write peak.
        return SimpleNamespace(free=1024**3)

    monkeypatch.setattr(shutil, "disk_usage", usage)
    with pytest.raises(ValueError, match="export.*headroom"):
        inspect_local("candidate", "room", run=run)
    assert len(calls) == 1 and len(measured) == 1
    assert measured[0].name.startswith("whitebox-image-history-")


def test_jenkins_package_import_remains_supported():
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from deploy.whitebox.image_storage import validate_history; assert callable(validate_history)",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )

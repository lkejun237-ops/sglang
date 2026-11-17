"""Build a pinned test-only FC collector update; never call cloud APIs.

The existing native collector is supplied as a downloaded deployment archive.
Only two checksum-verified sources change; dependencies and production FC
functions are untouched. Generated commands and packages contain no credentials.
"""

import argparse
import ast
import base64
import hashlib
import json
import re
import runpy
import zipfile
import zlib
from pathlib import Path

PINS = {
    "copy_host_snapshot.py": "71e26705fe9876b330eb4c038b44ffcdfcd803c6207d5b583dfb2eca59de3e27",
    "relay.py": "e49217987a574053b9940f9230388e84dcb4648b8e75f639a295991795f52e9d",
}
BASELINE_SHA256 = "d2970807683ab037bc6327814b038b57b6d5de4002f749cb1cf9157ee6184afb"
COMMAND_SHA256 = "d3e7853b98cc84a08dd908de2aac1dae4d618fbddf80f8a02d1b780eeb4db1b3"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Collector source does not match the reviewed baseline")
    return source.replace(old, new)


def patched_sources(files, stages, room, gateway):
    for name in (room, gateway):
        if not re.fullmatch(r"whitebox-sh-copy-[a-zA-Z0-9-]+", name):
            raise ValueError("Only Shanghai copy container names are supported")
    result = {}
    for name, digest in PINS.items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError("Collector baseline checksum mismatch: " + name)
        result[name] = files[name].decode()
    host = result["copy_host_snapshot.py"]
    host = replace_once(host, "whitebox-sh-copy-gateway-adaptive-pool-r5", gateway)
    host = replace_once(host, "whitebox-sh-copy-room-adaptive-017-default36-r6", room)
    host = replace_once(host, "MAX_DOCUMENT = 262144", "MAX_DOCUMENT = 1048576")
    host = replace_once(host, "len(seen)>1024", "len(seen)>4096")
    host = replace_once(host, "len(text.encode())>131072", "len(text.encode())>1048576")
    # Frozen, low-cardinality names from the same PR's serving contract.
    host += "\nSTAGES = STAGES | frozenset(" + repr(sorted(stages)) + ")\n"
    result["copy_host_snapshot.py"] = host
    relay = result["relay.py"]
    relay = replace_once(relay, "max_series: int = 2_000", "max_series: int = 4_096")
    relay += "\nRD_STAGES = RD_STAGES | frozenset(" + repr(sorted(stages)) + ")\n"
    result["relay.py"] = relay
    for name, source in result.items():
        compile(source, name, "exec")
    return {name: source.encode() for name, source in result.items()}


def host_command(files, kind, baseline_command):
    names = ("role_pool", "log_patterns", "copy_host_snapshot")
    expected = (
        COMMAND_SHA256
        if kind == "controls"
        else "70e5646c53c13027a0e29ef562528533abfaa99ea7008dd44865ab42efbe4015"
    )
    if (
        kind not in {"controls", "pool"}
        or hashlib.sha256(baseline_command).hexdigest() != expected
    ):
        raise ValueError("Host command does not match the reviewed baseline")
    packed = re.search(r"b64decode\('([^']+)'\)", baseline_command.decode())[1]
    baseline = ast.parse(zlib.decompress(base64.b64decode(packed)).decode())
    sources = ast.literal_eval(baseline.body[1].value)
    # Retain the baseline's reviewed dependency subset. No new host imports.
    required = {
        node.name
        for node in ast.parse(sources["copy_host_snapshot"]).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    tree = ast.parse(files["copy_host_snapshot.py"].decode())
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
        or node.name in required
    ]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    sources["copy_host_snapshot"] = ast.unparse(tree)
    compile(sources["copy_host_snapshot"], "copy_host_snapshot.py", "exec")
    program = "import sys,types\nsources=" + repr(sources) + "\n"
    program += "for name in " + repr(names) + ":\n"
    program += " module=types.ModuleType(name);sys.modules[name]=module\n"
    program += " exec(compile(sources[name],name,'exec'),module.__dict__)\n"
    program += "print(sys.modules['copy_host_snapshot'].snapshot(" + repr(kind) + "))\n"
    payload = base64.b64encode(zlib.compress(program.encode(), 9)).decode()
    return (
        "#!/bin/sh\nPYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'\nimport base64,zlib\nexec(compile(zlib.decompress(base64.b64decode("
        + repr(payload)
        + ")), '<test-copy-collector>', 'exec'))\nPY\n"
    )


def registered_host_commands(
    archive_bytes, baseline_command, pool_command, *, room, gateway
):
    """Rebind only the test host readers after a verified container replacement."""
    import io

    if (
        hashlib.sha256(archive_bytes).hexdigest()
        != "a80254563127d1bc76448a390b96ae4984ca7b7100e68371c88cf6056682f402"
    ):
        raise ValueError("Registered collector archive checksum mismatch")
    for name in (room, gateway):
        if not re.fullmatch(r"whitebox-sh-(?:copy|test)-[a-zA-Z0-9-]+", name):
            raise ValueError("Only Shanghai test container names are supported")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        raw = archive.read("copy_host_snapshot.py")
    if (
        hashlib.sha256(raw).hexdigest()
        != "53b6f0ffa568fdde002018ba481b9afc02c59d1dd4c85081016fae6cc252dd0a"
    ):
        raise ValueError("Registered host source checksum mismatch")
    source = replace_once(
        raw.decode(), "whitebox-sh-copy-room-pr662-81280695-timing-1", room
    )
    source = replace_once(
        source, "whitebox-sh-copy-gateway-debug-88a3744c-20261111", gateway
    )
    source = replace_once(source, "'dit':{23300,33000}", "'dit':{23300,33000,43310}")
    # The test Room has several routes. This PR observes its model1027 group;
    # the Coordinator default can still refer to an unrelated legacy group.
    source = replace_once(
        source,
        "value.get('capacity_routes',{}).get(value.get('default_capacity_route'))",
        "value.get('capacity_routes',{}).get('whitebox-copy-model1027-480')",
    )
    files = {"copy_host_snapshot.py": source.encode()}
    result = {
        kind: host_command(
            files, kind, baseline_command if kind == "controls" else pool_command
        )
        for kind in ("controls", "pool")
    }
    if any(
        len(base64.b64encode(command.encode())) > 18 * 1024
        for command in result.values()
    ):
        raise ValueError("Generated host command exceeds the cloud API budget")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-command", type=Path, required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.baseline.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError("Collector archive does not match the reviewed baseline")
    root = Path(__file__).resolve().parents[2]
    contract = runpy.run_path(
        str(
            root / "python/sglang/multimodal_gen/runtime/realtime/whitebox/telemetry.py"
        )
    )
    stages = contract["TTFF_METRIC_STAGES"] | contract["CLI_STARTUP_STAGES"]
    with zipfile.ZipFile(args.baseline) as archive:
        files = {entry.filename: archive.read(entry) for entry in archive.infolist()}
        updates = patched_sources(files, stages, args.room, args.gateway)
        files.update(updates)
        args.output.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            args.output / "collector.zip", "w", zipfile.ZIP_DEFLATED
        ) as target:
            for entry in archive.infolist():
                target.writestr(entry, files[entry.filename])
    for kind in ("controls",):
        command = host_command(files, kind, args.baseline_command.read_bytes())
        if len(base64.b64encode(command.encode())) > 18 * 1024:
            raise ValueError("Generated host command exceeds the cloud API budget")
        (args.output / (kind + ".sh")).write_text(command, encoding="utf8")
    for name, data in updates.items():
        (args.output / name).write_bytes(data)
    receipt = {
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "source_pins": PINS,
        "updated_sha256": {
            k: hashlib.sha256(v).hexdigest() for k, v in updates.items()
        },
        "room": args.room,
        "gateway": args.gateway,
        "allowed_new_stages": len(stages),
    }
    (args.output / "build.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf8"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

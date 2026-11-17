"""Prepare the Shanghai test collector for container replacements; no cloud calls.

The reviewed FC archive owns the source. Preserve its dependencies, metrics and
identity checks; only replace dated container names and update observed ports.
"""

import argparse
import ast
import base64
import hashlib
import json
import zipfile
from pathlib import Path

from patch_test_timing_collector import host_command, replace_once

BASELINE_SHA256 = "d5cd9ea9fd538061387d44451e538b358f391089ef425812e386ad48e9ff5d84"

CONTROL_RUNTIME = """
def _docker(container_id, service):
    if service not in CONTROL_PORTS or not re.fullmatch('[a-f0-9]{64}', container_id):
        raise CopyReadError('control_identity_not_allowed')
    connection = _UnixHTTP()
    try:
        connection.request('GET', '/containers/' + container_id + '/json')
        response = connection.getresponse()
        if response.status != 200:
            raise CopyReadError('container_inspect_failed')
        raw = response.read(1048577)
        if len(raw) > 1048576:
            raise CopyReadError('container_inspect_budget')
        obj = json.loads(raw)
    finally:
        connection.close()
    env = dict(row.split('=', 1) for row in obj.get('Config', {}).get('Env', []) if '=' in row)
    if (obj.get('Id') != container_id or
            not obj.get('Name', '').startswith('/whitebox-sh-copy-' + service + '-') or
            env.get('WHITEBOX_ROLE') != service or
            obj.get('State', {}).get('Running') is not True or
            obj.get('HostConfig', {}).get('NetworkMode') != 'host'):
        raise CopyReadError('control_container_identity')
    return obj


def _control_identity(service):
    # The fixed test listener selects the container, not an arbitrary name match.
    owner = _listener(CONTROL_PORTS[service])
    process = _process(owner)
    obj = _docker(process['container_id'], service)
    init = _process(obj['State']['Pid'])
    if process['container_id'] != obj['Id'] or init['container_id'] != obj['Id']:
        raise CopyReadError('listener_container_mismatch')
    return obj, {'container_id': obj['Id'], 'container_name': obj['Name'].lstrip('/'),
                 'started_at': obj['State']['StartedAt'], 'listener': process, 'init': init}
"""


def patch_host(raw):
    tree = ast.parse(raw.decode())
    replacements = {n.name: n for n in ast.parse(CONTROL_RUNTIME).body}
    found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not replacements.keys() <= found:
        raise ValueError("Reviewed control identity functions are missing")
    tree.body = [
        replacements.get(n.name, n) if isinstance(n, ast.FunctionDef) else n
        for n in tree.body
    ]
    source = ast.unparse(tree)
    source = replace_once(
        source, "'encoder': {29110, 31000}", "'encoder': {28510, 29110, 31000}"
    )
    source = replace_once(
        source, "'dit': {23300, 33000}", "'dit': {23300, 33000, 43310}"
    )
    source = replace_once(
        source,
        "value.get('capacity_routes', {}).get(value.get('default_capacity_route'))",
        "value.get('capacity_routes', {}).get('whitebox-copy-model1027-480')",
    )
    compile(source, "copy_host_snapshot.py", "exec")
    return source.encode()


def prepare(baseline, controls, pool, output):
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError("Test collector archive checksum mismatch")
    with zipfile.ZipFile(baseline) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
        if len(files) != len(archive.namelist()):
            raise ValueError("Duplicate archive entry")
        files["copy_host_snapshot.py"] = patch_host(files["copy_host_snapshot.py"])
        # The FC proof validator and the host reader enforce the same endpoints.
        source = replace_once(
            files["copy_sources.py"].decode(),
            "'encoder': {29110, 31000}",
            "'encoder': {28510, 29110, 31000}",
        )
        compile(source, "copy_sources.py", "exec")
        files["copy_sources.py"] = source.encode()
        commands = {
            kind: host_command(files, kind, path.read_bytes())
            for kind, path in [("controls", controls), ("pool", pool)]
        }
        if any(
            len(base64.b64encode(s.encode())) > 18 * 1024 for s in commands.values()
        ):
            raise ValueError("Host command exceeds cloud API budget")
        output.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(
            output / "collector.zip", "w", zipfile.ZIP_DEFLATED
        ) as target:
            for entry in archive.infolist():
                target.writestr(entry, files[entry.filename])
        for kind, command in commands.items():
            (output / (kind + ".sh")).write_text(
                command, encoding="utf-8", newline="\n"
            )
    receipt = {
        "baseline_sha256": BASELINE_SHA256,
        "environment": "test-shanghai-copy",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
        },
        "deployed": False,
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "controls", "pool", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline, args.controls, args.pool, args.output)))

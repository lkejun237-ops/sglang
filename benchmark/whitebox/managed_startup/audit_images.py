"""Audit fully built immutable test images; never builds or activates services."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import hashlib,json,os,sys
from pathlib import Path
v=json.loads(Path('/proof.json').read_text());component=sys.argv[1]
release=json.loads(Path('/app/release.json').read_text())
assert release['source_sha']==v['source'] and release['sdk_sha']==v['sdk'] and release['source_branch']==v['branch']
expected={}
for name,digest in v['files'].items():
    if any(p=='.env' or p.startswith('.env.') for p in name.split('/')):continue
    if name.startswith('python/sglang/') and not name.startswith('python/sglang/multimodal_gen/test/'):
        expected['/app/'+name]=digest
    elif name=='benchmark/whitebox/run_service.py':expected['/app/run_service.py']=digest
    elif component=='room' and name.startswith('.whitebox-sdk/'):
        expected['/opt/worldkit/'+name[len('.whitebox-sdk/'):]]=digest
for name,digest in expected.items():
    p=Path(name);data=os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==digest,name
assert Path('/app/whitebox-contracts-passed').exists()
if component=='room':
    assert Path('/app/whitebox-cli-passed').exists() and Path('/app/whitebox-immediate-codec-passed').exists()
print(json.dumps({'source':release,'component':component,'verified_files':len(expected),'source_files':sum(n.startswith('/app/')for n in expected),'sdk_files':sum(n.startswith('/opt/worldkit/')for n in expected)}))
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--tag-prefix", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.context / "deploy/whitebox"))
    from image_storage import inspect_local

    proof = a.context / "candidate-source.json"
    v = json.loads(proof.read_text())
    if not all(re.fullmatch("[a-f0-9]{40}", v[k]) for k in ("source", "sdk")):
        raise ValueError("Frozen source identity required")
    if not str(a.output).startswith("/opt/whitebox-managed-parallel-startup-20261113/"):
        raise ValueError("Expected private audit stage")
    exported = a.output.with_name("image-proof.json")
    exported.write_bytes(proof.read_bytes())
    os.chmod(str(exported), 0o644)

    def engine(argv, **kwargs):
        args = ["podman", *argv[1:]]
        if args[1:3] == ["image", "save"]:
            args[3:3] = ["--format", "docker-archive"]
        out = subprocess.check_output(args, **kwargs)
        if args[1:3] == ["image", "inspect"]:
            rows = json.loads(out)
            for row in rows:
                if re.fullmatch("[a-f0-9]{64}", row.get("Id", "")):
                    row["Id"] = "sha256:" + row["Id"]
            return json.dumps(rows)
        return out

    results = []
    for component in ("cpu", "room"):
        tag = a.tag_prefix + "-" + component
        obj = json.loads(
            engine(["docker", "image", "inspect", tag], universal_newlines=True)
        )[0]
        raw = subprocess.check_output(
            [
                "podman",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python",
                "-v",
                str(exported) + ":/proof.json:ro",
                obj["Id"],
                "-c",
                PROBE,
                component,
            ],
            universal_newlines=True,
        )
        result = json.loads(raw)
        result.update(tag=tag, storage=inspect_local(obj["Id"], component, run=engine))
        results.append(result)
        print(json.dumps(result), flush=True)
    with a.output.open("x") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

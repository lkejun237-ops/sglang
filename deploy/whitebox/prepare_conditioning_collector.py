"""Prepare, but never deploy, the conditioning extension to T051's collector.

Use the reviewed PR #662 r3 archive and relay checksum; no cloud API or
credential is used. Room host collection stays byte-identical. Registering an
authenticated, proved conditioning target remains a deployment integration step.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

PINS = {
    "relay.py": "fa97a2ea9e590e60fd00b12130e053b079f0265862a67200bf2f83ee7c73831b",
}
BASELINE_SHA256 = "a80254563127d1bc76448a390b96ae4984ca7b7100e68371c88cf6056682f402"
CONTRACT = "conditioning_metric_contract.py"
SOURCE = "conditioning_metric_source.py"
SCHEDULE = "copy_production.py"
SCHEDULE_SHA256 = "dc8980254d233e75c684b28d1be382824d384b914eb06984c4fc990a4ffdb426"
POOL_SOURCE = "copy_sources.py"
POOL_SOURCE_SHA256 = "ead6386ba5e5e29783f3a72f87fdb557135c3820f370a00cba9d032971c3f7d7"
IMPORT = (
    "from conditioning_metric_contract import (LABELS as CONDITIONING_LABELS, "
    "validate_labels as validate_conditioning_labels, is_test_conditioning_target)\n"
)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Collector source does not match the reviewed baseline")
    return source.replace(old, new)


def patched_sources(files):
    result = {}
    for name, digest in PINS.items():
        if hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError("Collector source checksum mismatch: " + name)
        result[name] = files[name].decode("utf-8")
    relay = result["relay.py"]
    relay = replace_once(
        relay,
        "from dataclasses import dataclass, fields",
        IMPORT + "from dataclasses import dataclass, fields",
    )
    relay = replace_once(
        relay,
        "        if identity['service'] != 'room' or not _allowed(name):",
        "        conditioning = is_test_conditioning_target(identity) and name in CONDITIONING_LABELS\n"
        "        if not conditioning and (identity['service'] != 'room' or not _allowed(name)):",
    )
    relay = replace_once(
        relay,
        "        if source.get('stage') not in RD_STAGES or name.endswith('_bucket') and not source.get('le'):",
        "        if conditioning:\n"
        "            try:validate_conditioning_labels(name,source)\n"
        "            except ValueError as error:raise RelayError(str(error)) from None\n"
        "        elif source.get('stage') not in RD_STAGES or name.endswith('_bucket') and not source.get('le'):",
    )
    relay = replace_once(
        relay,
        "        if len(parts) == 2:",
        "        if conditioning and (value<0 or len(parts)!=1):\n"
        "            raise RelayError('conditioning_sample_value')\n"
        "        if len(parts) == 2:",
    )
    relay = replace_once(
        relay,
        "        for key, val in source.items():",
        "        if conditioning:labels.update(source)\n"
        "        for key, val in ([] if conditioning else source.items()):",
    )
    result = {"relay.py": relay}
    for name, source in result.items():
        compile(source, name, "exec")
    encoded = {name: source.encode("utf-8") for name, source in result.items()}
    encoded[CONTRACT] = Path(__file__).with_name(CONTRACT).read_bytes()
    encoded[SOURCE] = Path(__file__).with_name(SOURCE).read_bytes()
    return encoded


def patched_schedule(raw):
    if hashlib.sha256(raw).hexdigest() != SCHEDULE_SHA256:
        raise ValueError("Collector schedule checksum mismatch")
    source = replace_once(
        raw.decode("utf-8"),
        "from relay import build_remote_write, assert_sendable",
        "from conditioning_metric_source import collect_registered\n"
        "from relay import build_remote_write, assert_sendable",
    )
    source = replace_once(
        source,
        "    try:\n        text, proof, at = source.room()",
        "    conditioning = collect_registered(pool_proof, sink,\n"
        "        target=os.environ.get('CONDITIONING_METRICS_TARGET'),\n"
        "        token=os.environ.get('CONDITIONING_METRICS_TOKEN'), clock=clock)\n"
        "    coverage['conditioning_metrics'] = conditioning['status'] == 'written'\n\n"
        "    try:\n        text, proof, at = source.room()",
    )
    source = replace_once(
        source,
        "    return {'coverage': coverage, 'room_series_written': metric_summary.get('series', 0),",
        "    return {'coverage': coverage, 'conditioning': conditioning,\n"
        "            'room_series_written': metric_summary.get('series', 0),",
    )
    compile(source, SCHEDULE, "exec")
    return source.encode("utf-8")


def patched_pool_source(raw):
    if hashlib.sha256(raw).hexdigest() != POOL_SOURCE_SHA256:
        raise ValueError("Collector pool source checksum mismatch")
    source = replace_once(
        raw.decode("utf-8"), "'dit': {23300, 33000}", "'dit': {23300, 33000, 43310}"
    )
    compile(source, POOL_SOURCE, "exec")
    return source.encode("utf-8")


def prepare(baseline, output):
    baseline_sha256 = hashlib.sha256(baseline.read_bytes()).hexdigest()
    if baseline_sha256 != BASELINE_SHA256:
        raise ValueError("Collector archive does not match the reviewed baseline")
    with zipfile.ZipFile(baseline) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or {CONTRACT, SOURCE} & set(names):
            raise ValueError(
                "Duplicate entries or conditioning extension already present"
            )
        updates = patched_sources({name: archive.read(name) for name in PINS})
        updates[SCHEDULE] = patched_schedule(archive.read(SCHEDULE))
        updates[POOL_SOURCE] = patched_pool_source(archive.read(POOL_SOURCE))
        output.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(
            output / "collector.zip", "w", zipfile.ZIP_DEFLATED
        ) as target:
            for entry in archive.infolist():
                target.writestr(entry, updates.get(entry.filename, archive.read(entry)))
            target.writestr(CONTRACT, updates[CONTRACT])
            target.writestr(SOURCE, updates[SOURCE])
    receipt = {
        "baseline_sha256": baseline_sha256,
        "source_pins": {
            **PINS,
            SCHEDULE: SCHEDULE_SHA256,
            POOL_SOURCE: POOL_SOURCE_SHA256,
        },
        "updated_sha256": {
            k: hashlib.sha256(v).hexdigest() for k, v in updates.items()
        },
        "collector_sha256": hashlib.sha256(
            (output / "collector.zip").read_bytes()
        ).hexdigest(),
        "deployed": False,
        "target_registered": False,
        "authenticated_entrypoint": "copy_production.run_round -> conditioning_metric_source.collect_registered",
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline, args.output)))

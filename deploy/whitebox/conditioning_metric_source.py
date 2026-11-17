"""Authenticated read for an already proved Shanghai test conditioning target.

Deployment must obtain a CURRENT target proof independently, bind the Secret
file, call this entry, and recheck assert_sendable immediately before transport.
This module does not discover targets, invent proofs, schedule or remote_write.
"""

import asyncio
import hashlib
import ipaddress
import time
import urllib.parse

import aiohttp
from conditioning_metric_contract import is_test_conditioning_target
from relay import assert_sendable, build_remote_write

SCRAPE_TIMEOUT_SECONDS = 3
MAX_METRICS_BYTES = 1048576


class ConditioningReadError(ValueError):
    """Fixed error categories only: no credential, target or response contents."""


async def _read_metrics(url, token):
    # One deadline spans connection, TLS, headers and all body chunks. aiohttp
    # cancels the read and closes the response/session on timeout; no worker
    # thread survives a failed collection. Both FC and CPU CI already ship it.
    timeout = aiohttp.ClientTimeout(total=SCRAPE_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(
        timeout=timeout, trust_env=False, auto_decompress=False
    ) as session:
        async with session.get(
            url,
            headers={"Authorization": "Bearer " + token, "Accept": "text/plain"},
            allow_redirects=False,
        ) as response:
            if (
                response.status != 200
                or response.content_type != "text/plain"
                or response.headers.get("Content-Encoding", "identity") != "identity"
            ):
                raise ValueError()
            raw = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                raw.extend(chunk)
                if len(raw) > MAX_METRICS_BYTES:
                    raise ValueError()
            return bytes(raw)


def prepare_batch(
    proof, token_file=None, *, token=None, scheme="http", clock=time.time
):
    """Synchronous FC entry: one 3s HTTP deadline / 1 MiB cap, no send or retry."""
    if not is_test_conditioning_target(proof.get("labels")):
        raise ConditioningReadError("conditioning_target_scope")
    target = proof.get("target", "")
    try:
        address = urllib.parse.urlsplit("//" + target)
        if (
            scheme not in {"http", "https"}
            or address.username is not None
            or address.password is not None
            or address.path
            or address.query
            or address.fragment
            or address.port is None
            or not ipaddress.ip_address(address.hostname).is_private
        ):
            raise ValueError()
    except (ValueError, TypeError):
        raise ConditioningReadError("conditioning_target_address") from None
    now = clock()
    build_remote_write(
        "", labels=proof["labels"], target=target, proof=proof, sampled_at=now, now=now
    )
    try:
        if (token_file is None) == (token is None):
            raise ValueError()
        if token_file is not None:
            with open(token_file, "rb") as source:
                raw = source.read(4097)
            token = raw.decode("ascii").strip()
        else:
            raw = token.encode("ascii")
        if len(raw) > 4096 or not token or any(not 33 <= ord(c) <= 126 for c in token):
            raise ValueError()
    except (OSError, UnicodeError, ValueError, AttributeError):
        raise ConditioningReadError("conditioning_credential_unavailable") from None
    try:
        raw = asyncio.run(_read_metrics(scheme + "://" + target + "/metrics", token))
        sampled_at = clock()
    except TimeoutError:
        raise ConditioningReadError("conditioning_scrape_timeout") from None
    except (aiohttp.ClientError, OSError, ValueError):
        raise ConditioningReadError("conditioning_scrape_failed") from None
    batch = build_remote_write(
        raw,
        labels=proof["labels"],
        target=target,
        proof=proof,
        sampled_at=sampled_at,
        now=clock(),
    )
    assert_sendable(batch, proof=proof, now=clock())
    return batch


def collect_registered(pool_proof, sink, *, target, token, clock=time.time):
    """One optional scrape in the existing fenced round, from its fresh pool proof."""
    if not target and not token:
        return {"status": "disabled", "series": 0}
    try:
        now = clock()
        if (
            not target
            or not token
            or not pool_proof
            or pool_proof.get("up") != 1
            or not pool_proof["observed_at"] <= now < pool_proof["valid_until"]
        ):
            raise ValueError()
        owners = [
            p
            for p in pool_proof["processes"]
            if p["role"] == "encoder" and target in p["targets"]
        ]
        if len(owners) != 1:
            raise ValueError()
        owner = owners[0]
        proof = dict(
            status="matched",
            complete=True,
            target=target,
            labels=dict(
                business_line_id="world-model-data",
                region="cn-shanghai",
                deployment_id="test-shanghai-copy",
                pool="test-shanghai-copy",
                service="sglang",
            ),
            observed_at=pool_proof["observed_at"],
            valid_until=pool_proof["valid_until"],
            proof_id=hashlib.sha256(
                (
                    pool_proof["proof_id"]
                    + owner["process_id"]
                    + owner["boot_epoch"]
                    + target
                ).encode()
            ).hexdigest(),
        )
        batch = prepare_batch(proof, token=token, clock=clock)
        if batch.series_count:
            assert_sendable(batch, proof=proof, now=clock())
            if sink.write(batch) is not True:
                raise ValueError()
        return {
            "status": "written" if batch.series_count else "unobserved",
            "series": batch.series_count,
        }
    except Exception:
        # Prefix diagnostics cannot suppress Room collection or disclose a Secret.
        return {"status": "unknown", "series": 0}

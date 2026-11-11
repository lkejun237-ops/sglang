"""Explicit owner enrollment/inspection/drain. Does not restart or stop services."""

import argparse
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def endpoint(value):
    p = urlsplit(value)
    if (
        p.scheme not in {"http", "https"}
        or not p.hostname
        or p.username
        or p.password
        or p.query
        or p.fragment
        or p.path not in {"", "/"}
    ):
        raise argparse.ArgumentTypeError("Use a credential-free HTTP origin")
    return value.rstrip("/")


def request(coordinator, operation, body, token):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    req = urllib.request.Request(
        coordinator + "/v1/whitebox/room-owners/" + operation,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=15) as response:
            value = response.read(2 * 1024**2 + 1)
            if len(value) > 2 * 1024**2:
                raise ValueError("Room registry response exceeded its limit")
            return json.loads(value)
    except urllib.error.HTTPError as error:
        raise ValueError(
            f"Room registry request failed with HTTP {error.code}"
        ) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True, type=endpoint)
    parser.add_argument("--token-env", default="WHITEBOX_ROOM_ADMIN_TOKEN")
    sub = parser.add_subparsers(dest="operation", required=True)
    add = sub.add_parser("enroll")
    add.add_argument("--room-url", required=True, type=endpoint)
    add.add_argument(
        "--initialize-empty-pool",
        action="store_true",
        help="First use in a new, empty control namespace only; never a recovery reset",
    )
    show = sub.add_parser("inspect")
    show.add_argument("--owner-id")
    drain = sub.add_parser("drain")
    drain.add_argument("--owner-id", required=True)
    drain.add_argument("--owner-epoch", required=True)
    remove = sub.add_parser("remove")
    remove.add_argument("--owner-id", required=True)
    remove.add_argument("--owner-epoch", required=True)
    args = parser.parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        parser.error("The operator token environment variable is missing")
    if args.operation == "enroll":
        body = dict(owner_url=args.room_url, initialize=args.initialize_empty_pool)
    else:
        body = {
            k: getattr(args, k)
            for k in ("owner_id", "owner_epoch")
            if getattr(args, k, None)
        }
    try:
        print(
            json.dumps(
                request(args.coordinator, args.operation, body, token),
                ensure_ascii=False,
                indent=2,
            )
        )
    except (ValueError, urllib.error.URLError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()

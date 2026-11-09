# SPDX-License-Identifier: Apache-2.0
"""Short-lived RTC create capabilities for browser sessions and API clients."""

import base64
import hashlib
import hmac
import json
import re
import time


class RTCCapabilities:
    def __init__(self, secret, *, now=time.time, audience="whitebox-rtc"):
        if not isinstance(secret, str) or len(secret.encode()) < 32:
            raise ValueError("RTC authorization is not configured")
        self.secret, self.now, self.used = secret.encode(), now, {}
        self.audience = audience

    def _decode(self, token):
        if not isinstance(token, str) or len(token) > 4096:
            raise ValueError()
        payload, signature = token.split(".")

        def decode(value):
            return base64.b64decode(
                value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
            )

        expected = hmac.new(
            self.secret, payload.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(decode(signature), expected):
            raise ValueError()
        claims = json.loads(decode(payload))
        if not isinstance(claims, dict):
            raise ValueError()
        return claims

    def verify(self, token, *, origin, world, operation=None, scope=None, consume=True):
        try:
            # Prefill's shared request store owns its one-compute claim, so a
            # reconnect/status read revalidates the ticket without consuming it.
            # Ordinary RTC create capabilities remain strictly one-use.
            if consume is not True and (
                consume is not False or self.audience != "whitebox-prefill"
            ):
                raise ValueError()
            claims = self._decode(token)
            if not isinstance(claims, dict) or not isinstance(world, dict):
                raise ValueError()
            version = claims.get("v")
            if type(version) is not int:
                raise ValueError()
            if version == 1:
                if (
                    not isinstance(claims.get("origin"), str)
                    or claims["origin"] != origin
                ):
                    raise ValueError()
            elif version == 2:
                # Only the trusted issuer may attest API Token authentication.
                # Such clients need no browser Origin; the signed world/user,
                # expiry and one-create identity remain mandatory below.
                if claims.get("auth_method") != "api-token" or "origin" in claims:
                    raise ValueError()
            else:
                raise ValueError()
            now = self.now()
            if (
                claims.get("aud") != self.audience
                or claims.get("world_id") != world.get("world_id")
                or claims.get("source_id") != world.get("source_id")
                or (operation is not None and claims.get("operation") != operation)
                or (scope is not None and claims.get("scope") != scope)
                or not isinstance(claims.get("sub"), str)
                or not 1 <= len(claims["sub"]) <= 256
                or type(claims.get("iat")) is not int
                or type(claims.get("exp")) is not int
                or not claims["iat"] <= now + 5
                or not now < claims["exp"] <= claims["iat"] + 120
                or not re.fullmatch(r"[a-f0-9]{32}", str(claims.get("jti", "")))
            ):
                raise ValueError()
            from .session_encoding import bind_signed_source_encoding

            if self.audience == "whitebox-rtc":
                bind_signed_source_encoding(claims, world)
            self.used = {
                key: expiry for key, expiry in self.used.items() if expiry > now
            }
            if consume:
                if claims["jti"] in self.used or len(self.used) >= 4096:
                    raise ValueError()
                self.used[claims["jti"]] = claims["exp"]
            return claims
        except (ValueError, TypeError, KeyError, UnicodeError):
            raise ValueError("RTC authorization could not be verified") from None

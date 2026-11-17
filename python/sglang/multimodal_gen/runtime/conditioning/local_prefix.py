# SPDX-License-Identifier: Apache-2.0
"""Session-owned prefix reuse for the standalone Encoder's existing Qwen model."""

import time

from .inputs import media_identity
from .worker import LocalConditioner


class SessionPrefixConditioner(LocalConditioner):
    """Used only on the Encoder's serial execution thread; never model-global."""

    def __init__(self, encode, profile_id, *, prefix_bytes, metrics, cache_bytes):
        from sglang.multimodal_gen.runtime.models.encoders.qwen_prefix import (
            QwenPrefixCache,
        )

        if type(prefix_bytes) is not int or prefix_bytes < 0:
            raise ValueError("Invalid Qwen prefix cache byte budget")
        self.prefix = QwenPrefixCache(prefix_bytes) if prefix_bytes else None
        self.metrics = metrics
        self.native_encode = encode
        self._encoded = False
        super().__init__(self._encode, profile_id, cache_bytes=cache_bytes)

    def _encode(self, prompt, media):
        self._encoded = True
        if self.prefix is None:
            return self.native_encode(prompt, media)
        return self.native_encode(
            prompt,
            media,
            prefix_cache=self.prefix,
            media_identity=media_identity(media),
        )

    def __call__(self, prompt, media, *, state=None):
        self._encoded = False
        started = time.perf_counter()
        try:
            result = super().__call__(prompt, media, state=state)
        except BaseException:
            # A failed forward can already have mutated KV in place.
            self.clear_prefix()
            raise
        if self.metrics is not None:
            if self._encoded:
                stats = None
                if self.prefix is not None:
                    stats = dict(
                        hit=self.prefix.reused_rows > 0,
                        reused_tokens=self.prefix.reused_rows,
                        encoded_tokens=self.prefix.encoded_rows,
                        total_tokens=result["context"].shape[0],
                    )
                self.metrics.encoded(
                    stats, (time.perf_counter() - started) * 1000, None
                )
            self.metrics.results.labels(
                "encoding" if self._encoded else "result_cache"
            ).inc()
        return result

    def clear_prefix(self):
        if self.prefix is not None:
            self.prefix.clear()

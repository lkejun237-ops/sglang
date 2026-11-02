# SPDX-License-Identifier: Apache-2.0
"""Small independent game/group associations; KV entities live separately."""

import base64
from pathlib import Path

from .disk_cache import Block0DiskCache, _dump_json, _read_json


def validate_labels(game_id=None, group_id=None):
    for value in (game_id, group_id):
        if value is not None and (
            not isinstance(value, str) or not value or len(value.encode()) > 128
        ):
            raise ValueError(
                "Cache labels must be nonempty strings of at most 128 UTF-8 bytes"
            )
    if group_id is not None and game_id is None:
        raise ValueError("A cache group requires a game ID")


def _component(value):
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


class CacheIndex:
    def __init__(self, root):
        self.root = Path(root).resolve() / "index"

    def bind(self, game_id, group_id, cache_key):
        validate_labels(game_id, group_id)
        if game_id is None:
            raise ValueError("A cache association requires a game ID")
        if not Block0DiskCache._safe_key(cache_key):
            raise ValueError("Invalid cache key")
        folder = (
            self.root
            / _component(game_id)
            / ("_" if group_id is None else _component(group_id))
        )
        folder.mkdir(parents=True, exist_ok=True)
        _dump_json(
            folder / f"{cache_key}.json",
            dict(game_id=game_id, group_id=group_id, cache_key=cache_key),
        )

    def select(self, game_id, group_id=None, cache_key=None):
        validate_labels(game_id, group_id)
        if game_id is None:
            raise ValueError("A cache selection requires a game ID")
        folder = self.root / _component(game_id)
        pattern = "*/*.json" if group_id is None else f"{_component(group_id)}/*.json"
        keys = set()
        for path in folder.glob(pattern):
            if not path.resolve().is_relative_to(self.root):
                continue
            try:
                value = _read_json(path)
                key = value["cache_key"]
                if (
                    value["game_id"] == game_id
                    and (group_id is None or value.get("group_id") == group_id)
                    and (cache_key is None or key == cache_key)
                    and path.name == f"{key}.json"
                ):
                    keys.add(key)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if cache_key is not None and cache_key not in keys:
            raise ValueError("The cache key does not belong to the selected game/group")
        return sorted(keys)

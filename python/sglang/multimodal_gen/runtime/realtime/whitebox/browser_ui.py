# SPDX-License-Identifier: Apache-2.0
"""Verified UI bundle handoff. This owner never launches a UI browser/encoder."""

import asyncio
import hashlib
from pathlib import Path


class BrowserUI:
    def __init__(self, descriptor, package):
        if not package or not isinstance(package.get("manifest"), dict):
            raise ValueError("The verified UI bundle is missing")
        manifest = package["manifest"]
        self.root = Path(package["root"]).resolve()
        self.files = {}
        for item in [
            manifest["module"],
            manifest["catalog"],
            manifest["definition"],
            manifest["stateSchema"],
            *manifest["styles"],
            *manifest["assets"],
        ]:
            path = Path(item["path"])
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != item["path"]
            ):
                raise ValueError("Invalid UI bundle path")
            previous = self.files.setdefault(item["path"], item["sha256"])
            if previous != item["sha256"]:
                raise ValueError("Conflicting UI asset identity")
        self.metadata = dict(
            mode="browser",
            training_eligible=True,
            runtime_abi=manifest["runtimeAbiVersion"],
            bundle_sha256=descriptor["uiBundleHash"],
            source_session_id=descriptor["sessionId"],
            source_epoch=descriptor["epoch"],
            world_build_hash=descriptor["worldBuildHash"],
            module=manifest["module"],
            styles=manifest["styles"],
            assets=manifest["assets"],
            files=self.files,
            document=descriptor["ui"]["document"],
            catalog=descriptor["ui"]["catalog"],
        )

    async def asset(self, path):
        if path not in self.files:
            raise ValueError("The UI asset is not declared")
        target = self.root / path
        if (
            self.root not in target.resolve().parents
            or target.is_symlink()
            or target.stat().st_size > 8 * 1024 * 1024
        ):
            raise ValueError("Invalid UI asset file")
        value = await asyncio.to_thread(target.read_bytes)
        if hashlib.sha256(value).hexdigest() != self.files[path]:
            raise ValueError("The UI asset changed after verification")
        return value

    async def consume_ui(self, message):
        # CLIView owns the bounded opaque event stream; the browser selects UI.
        pass

    async def render(self, *args):
        raise RuntimeError("This deployment renders UI in the browser")

    async def close(self):
        pass


def factory(*, descriptor, spec, package):
    return BrowserUI(descriptor, package)

# SPDX-License-Identifier: Apache-2.0

"""Project-local structured-output client for Doubao models on Ark."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


ROOT = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS_PATH = (
    ROOT.parent / "realtime_webui_secrets" / "doubao-ark.json"
)
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-0-mini-260428"
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _append_structure_correction(
    user_content: str | list[dict[str, Any]], correction: str
) -> str | list[dict[str, Any]]:
    text = (
        "\n\nSTRUCTURE CORRECTION FOR THE NEXT ATTEMPT:\n"
        f"{correction}\nReturn valid JSON matching the requested schema exactly."
    )
    if isinstance(user_content, str):
        return user_content + text
    copied = [dict(item) for item in user_content]
    copied.append({"type": "text", "text": text})
    return copied


class DoubaoStructuredProvider:
    """One warm AsyncOpenAI-compatible Ark client with strict JSON validation."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        credentials_path: str | Path | None = None,
        request_timeout_seconds: float | None = None,
        max_attempts: int = 2,
        client: Any | None = None,
    ) -> None:
        configured_path = credentials_path or os.environ.get(
            "DOUBAO_ARK_CREDENTIALS"
        )
        self.credentials_path = Path(
            configured_path or DEFAULT_CREDENTIALS_PATH
        ).expanduser()
        self._model_explicit = model is not None or bool(os.environ.get("DOUBAO_MODEL"))
        self.model = model or os.environ.get("DOUBAO_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self._base_url = base_url or os.environ.get("DOUBAO_ARK_BASE_URL", "")
        self.request_timeout_seconds = float(
            request_timeout_seconds
            or os.environ.get("DOUBAO_REQUEST_TIMEOUT_SECONDS", "20")
        )
        self.max_attempts = max(1, int(max_attempts))
        self._client = client
        self._client_lock = asyncio.Lock()
        self._config_loaded = False

    def _load_config(self) -> None:
        if self._config_loaded:
            return
        self._config_loaded = True
        if not self.credentials_path.is_file():
            return
        try:
            payload = json.loads(self.credentials_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Unable to read the project Doubao config") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Project Doubao config must be a JSON object")
        self._api_key = self._api_key or str(payload.get("api_key") or "")
        self._base_url = self._base_url or str(payload.get("base_url") or "")
        if not self._model_explicit and payload.get("model"):
            self.model = str(payload["model"])

    @property
    def configured(self) -> bool:
        if self._client is not None or self._api_key:
            return True
        return self.credentials_path.is_file()

    async def get_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is not None:
                return self._client
            self._load_config()
            if not self._api_key:
                raise RuntimeError("Doubao API key is not configured in this project")
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise RuntimeError("openai is required for the Doubao provider") from error
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url or DEFAULT_BASE_URL,
                timeout=self.request_timeout_seconds,
            )
            return self._client

    async def warmup(self) -> None:
        await self.get_client()

    async def generate_structured(
        self,
        *,
        system_instruction: str,
        user_content: str | list[dict[str, Any]],
        schema: type[SchemaT],
        schema_name: str,
        max_output_tokens: int = 2048,
    ) -> SchemaT:
        client = await self.get_client()
        base_content = user_content
        attempt_content = user_content
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }
        for attempt in range(1, self.max_attempts + 1):
            # API failures and timeouts are latency-sensitive: surface immediately.
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": attempt_content},
                    ],
                    max_tokens=max_output_tokens,
                    response_format=response_format,
                    extra_body={"thinking": {"type": "disabled"}},
                ),
                timeout=self.request_timeout_seconds,
            )
            content = response.choices[0].message.content or ""
            try:
                return schema.model_validate_json(content)
            except (ValidationError, ValueError) as error:
                if attempt == self.max_attempts:
                    raise RuntimeError(
                        "Doubao returned invalid structured output after "
                        f"{self.max_attempts} attempt(s)"
                    ) from error
                attempt_content = _append_structure_correction(
                    base_content, str(error)
                )
        raise AssertionError("unreachable")

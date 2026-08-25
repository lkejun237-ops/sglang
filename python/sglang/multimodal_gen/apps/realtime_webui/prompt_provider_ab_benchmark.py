# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark Gemini and Ark/Doubao for World Studio prompt tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doubao_provider import DoubaoStructuredProvider
from prompt_rewriter import PromptRewriter
from world_creator import WorldCreator


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "assets" / "presets" / "v1" / "reactor-misted-kingdom.png"
BASE_WORLD = (
    "A third-person over-the-shoulder fantasy view follows a sword-slung rider "
    "on a brown horse along a walkable stone road through a broad misty valley. "
    "The rider stays centered and faces away from the camera. Wildflower meadows, "
    "ruined arches, cottages, a river, and a many-spired castle remain visible. "
    "Cool morning light, stable scale, coherent geography, and an unobstructed "
    "forward route support continuous exploration."
)


CASES: list[dict[str, Any]] = [
    {
        "id": "rewrite_sunset",
        "kind": "rewrite",
        "instruction": "让整个山谷进入金色黄昏，天空和光照持续保持日落状态",
        "previous_prompt": BASE_WORLD,
        "expected_change_type": "persistent",
        "anchors": [["sunset", "dusk", "golden hour"], ["valley"]],
    },
    {
        "id": "rewrite_snow",
        "kind": "rewrite",
        "instruction": "让山谷持续下雪，路面和草地逐渐积雪，但道路仍然可探索",
        "previous_prompt": BASE_WORLD,
        "expected_change_type": "persistent",
        "anchors": [["snow"], ["road", "path"], ["valley"]],
    },
    {
        "id": "rewrite_castle_route",
        "kind": "rewrite",
        "instruction": "把前方道路改成一条持续通往城堡正门的林间道路",
        "previous_prompt": BASE_WORLD,
        "expected_change_type": "persistent",
        "anchors": [["castle"], ["road", "path"], ["forest", "woodland", "trees"]],
    },
    {
        "id": "rewrite_fire_breath",
        "kind": "rewrite",
        "instruction": "主角立刻向前喷出一道巨大的火焰，火星和烟雾在空中散开",
        "previous_prompt": BASE_WORLD,
        "expected_change_type": "one_time",
        "anchors": [["fire", "flame"], ["spark", "ember"], ["smoke"]],
    },
    {
        "id": "rewrite_summon_ship",
        "kind": "rewrite",
        "instruction": "召唤一艘发光飞船从云层快速降落到道路旁边",
        "previous_prompt": BASE_WORLD,
        "expected_change_type": "one_time",
        "anchors": [["ship", "spacecraft", "airship"], ["cloud"], ["descend", "land"]],
    },
    {
        "id": "rewrite_lightning",
        "kind": "rewrite",
        "instruction": "向前方释放一道蓝白色闪电，短暂照亮城堡和山谷",
        "previous_prompt": BASE_WORLD,
        "expected_change_type": "one_time",
        "anchors": [["lightning"], ["blue", "white"], ["castle", "valley"]],
    },
    {
        "id": "rule_ice_shield",
        "kind": "rule",
        "rule_kind": "skill",
        "input": "召唤一面环绕主角的透明冰盾",
        "previous_prompt": BASE_WORLD,
        "expected_change_type": "one_time",
        "anchors": [["ice", "frost"], ["shield", "barrier"]],
    },
    {
        "id": "rule_star_badge",
        "kind": "rule",
        "rule_kind": "goal",
        "input": "玩家获得一枚从天空落下的发光星形徽章",
        "previous_prompt": BASE_WORLD,
        "expected_change_type": "one_time",
        "anchors": [["star"], ["badge", "medal", "emblem"], ["glow", "light"]],
    },
    {
        "id": "world_text_desert",
        "kind": "world",
        "seed_text": "第一人称驾驶越野车穿过橙色沙漠峡谷，前方有清晰可行驶的道路",
        "expected_camera_mode": "first_person",
        "expected_external_subject": False,
        "anchors": [["desert", "canyon"], ["road", "route", "track"], ["vehicle", "dashboard", "windshield"]],
    },
    {
        "id": "world_image_rider",
        "kind": "world",
        "seed_text": "保持骑马冒险主体，扩展成可以沿道路探索的雾谷王国",
        "image_path": str(DEFAULT_IMAGE),
        "expected_camera_mode": "third_person",
        "expected_external_subject": True,
        "anchors": [["rider", "horse"], ["road", "path", "route"], ["mist", "fog", "valley"]],
    },
]


def _world_user_content(case: dict[str, Any]) -> tuple[str, bytes | None, str]:
    seed_text = str(case.get("seed_text") or "").strip()
    image_path = Path(case["image_path"]) if case.get("image_path") else None
    image_bytes = image_path.read_bytes() if image_path else None
    mime = "image/png" if image_path and image_path.suffix.lower() == ".png" else "image/jpeg"
    if seed_text and image_bytes:
        text = (
            "First inspect the supplied first frame and decide whether it has a clear "
            "external playable subject. Use third_person only when it does; otherwise "
            "use first_person. For first_person, inventory only the player body parts "
            "actually visible in the image and make the final description explicitly "
            "state whether any are visible. For third_person, make the external subject "
            "centered in the frame. Then create a complete visual world by combining "
            "the image with this seed text:\n<world_seed>\n"
            f"{seed_text}\n</world_seed>"
        )
    elif seed_text:
        text = (
            "Create the complete visual world described by this text:\n"
            f"<world_seed>\n{seed_text}\n</world_seed>"
        )
    else:
        text = (
            "Inspect the supplied first frame before writing. Use third_person only if "
            "it contains a clear external playable subject; otherwise use first_person."
        )
    return text, image_bytes, mime


@dataclass
class DoubaoProvider:
    model: str
    api_key: str

    def __post_init__(self) -> None:
        self.structured_provider = DoubaoStructuredProvider(
            api_key=self.api_key,
            model=self.model,
            request_timeout_seconds=40,
        )
        self.rewriter = PromptRewriter(
            provider="doubao",
            model=self.model,
            doubao_provider=self.structured_provider,
        )
        self.creator = WorldCreator(
            description_provider="doubao",
            doubao_provider=self.structured_provider,
        )

    async def run(self, case: dict[str, Any]) -> dict[str, Any]:
        if case["kind"] == "rewrite":
            output = await self.rewriter.rewrite(
                case["instruction"], case["previous_prompt"]
            )
        elif case["kind"] == "rule":
            output = await self.rewriter.complete_world_rule(
                case["input"], case["previous_prompt"], case["rule_kind"]
            )
        else:
            _, image_bytes, mime = _world_user_content(case)
            output = await self.creator.describe(case["seed_text"], image_bytes, mime)
        return output.model_dump(mode="json")


class GeminiProvider:
    def __init__(self) -> None:
        self.rewriter = PromptRewriter()
        self.creator = WorldCreator(gemini_client_provider=self.rewriter._get_client)
        self.model = self.rewriter.model

    async def run(self, case: dict[str, Any]) -> dict[str, Any]:
        if case["kind"] == "rewrite":
            output = await self.rewriter.rewrite(
                case["instruction"], case["previous_prompt"]
            )
        elif case["kind"] == "rule":
            output = await self.rewriter.complete_world_rule(
                case["input"], case["previous_prompt"], case["rule_kind"]
            )
        else:
            _, image_bytes, mime = _world_user_content(case)
            output = await self.creator.describe(case["seed_text"], image_bytes, mime)
        return output.model_dump(mode="json")


def _contains_anchor(text: str, alternatives: list[str]) -> bool:
    lowered = text.casefold()
    return any(item.casefold() in lowered for item in alternatives)


def _quality(case: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    if case["kind"] == "world":
        text = str(output.get("world_description") or "")
        camera_ok = output.get("camera_mode") == case["expected_camera_mode"]
        subject_ok = (
            output.get("source_image_has_clear_external_subject")
            == case["expected_external_subject"]
        )
        type_ok = camera_ok and subject_ok
    else:
        text = str(output.get("prompt") or "")
        type_ok = output.get("change_type") == case["expected_change_type"]
    words = len(text.split())
    anchor_hits = sum(_contains_anchor(text, group) for group in case["anchors"])
    anchor_score = 4 * anchor_hits / max(1, len(case["anchors"]))
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / max(1, len(text))
    if case["kind"] == "world":
        length_score = 2 if 90 <= words <= 260 else 1 if 60 <= words <= 360 else 0
    else:
        length_score = 2 if 180 <= words <= 320 else 1 if 100 <= words <= 400 else 0
    score = 2 + (2 if type_ok else 0) + length_score + anchor_score
    if ascii_ratio >= 0.9:
        score += 0.5
    if case["kind"] == "rule" and str(output.get("name") or "").strip():
        score += 0.5
    return {
        "score": round(min(10, score), 2),
        "type_ok": type_ok,
        "word_count": words,
        "anchor_hits": anchor_hits,
        "anchor_total": len(case["anchors"]),
        "english_ratio": round(ascii_ratio, 3),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doubao-model",
        action="append",
        dest="doubao_models",
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    api_key = os.environ.get("DOUBAO_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DOUBAO_API_KEY is required")
    models = args.doubao_models or ["doubao-seed-2-0-lite-260428"]
    providers: list[tuple[str, Any]] = [("gemini", GeminiProvider())]
    providers.extend((model, DoubaoProvider(model, api_key)) for model in models)
    results: list[dict[str, Any]] = []
    for case in CASES:
        for provider_name, provider in providers:
            started = time.perf_counter()
            try:
                output = await provider.run(case)
                elapsed = time.perf_counter() - started
                row = {
                    "case_id": case["id"],
                    "kind": case["kind"],
                    "provider": provider_name,
                    "model": provider.model,
                    "ok": True,
                    "seconds": round(elapsed, 3),
                    "output": output,
                }
                row.update(_quality(case, output))
            except Exception as error:
                row = {
                    "case_id": case["id"],
                    "kind": case["kind"],
                    "provider": provider_name,
                    "model": provider.model,
                    "ok": False,
                    "seconds": round(time.perf_counter() - started, 3),
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                    "score": 0,
                }
            results.append(row)
            print(json.dumps({k: row[k] for k in ("case_id", "provider", "ok", "seconds", "score")}, ensure_ascii=False), flush=True)

    summaries = []
    for provider_name, provider in providers:
        rows = [row for row in results if row["provider"] == provider_name]
        passed = [row for row in rows if row["ok"]]
        timings = [row["seconds"] for row in passed]
        summaries.append(
            {
                "provider": provider_name,
                "model": provider.model,
                "success": len(passed),
                "total": len(rows),
                "avg_seconds": round(statistics.mean(timings), 3) if timings else None,
                "median_seconds": round(statistics.median(timings), 3) if timings else None,
                "p90_seconds": round(_percentile(timings, 0.9), 3) if timings else None,
                "avg_quality": round(statistics.mean(row["score"] for row in rows), 2),
                "type_accuracy": round(
                    sum(bool(row.get("type_ok")) for row in rows) / len(rows), 3
                ),
            }
        )
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "case_count": len(CASES),
        "rubric": (
            "10-point deterministic rubric: schema 2, type/perspective 2, length 2, "
            "case anchors 4, plus English/name bonuses capped at 10."
        ),
        "summaries": summaries,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "prompt_provider_ab_results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# World Studio Prompt Provider A/B",
        "",
        f"Cases: {len(CASES)}",
        "",
        "| Provider | Model | Success | Avg s | Median s | P90 s | Quality / 10 | Type accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['provider']} | {item['model']} | {item['success']}/{item['total']} "
            f"| {item['avg_seconds']} | {item['median_seconds']} | {item['p90_seconds']} "
            f"| {item['avg_quality']} | {item['type_accuracy']:.0%} |"
        )
    lines.extend(["", "## Per-case", "", "| Case | Kind | Provider | Seconds | Score | Type OK | Words |", "|---|---|---|---:|---:|---:|---:|"])
    for row in results:
        lines.append(
            f"| {row['case_id']} | {row['kind']} | {row['provider']} | {row['seconds']} "
            f"| {row['score']} | {row.get('type_ok', False)} | {row.get('word_count', 0)} |"
        )
    (args.output_dir / "prompt_provider_ab_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summaries": summaries, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

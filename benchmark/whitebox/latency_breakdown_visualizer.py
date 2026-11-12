#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Room-SGLang 白膜+按键供帧逐帧耗时分解数据整理与可视化工具（交互式 HTML + 高清 PNG）。"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

STAGE_ROLES = ("encoder", "dit", "decoder")
STAGE_METRICS = ("wait_ms", "process_ms", "return_ms")
ROLE_LABELS = {
    "encoder": "Encoder",
    "dit": "DiT",
    "decoder": "Decoder",
}
METRIC_LABELS = {
    "wait_ms": "等待攒帧",
    "process_ms": "处理计算",
    "return_ms": "返回结果",
}

KEY_SHORT_LABELS = {
    "KeyW": "W",
    "KeyA": "A",
    "KeyS": "S",
    "KeyD": "D",
    "KeyQ": "Q",
    "KeyE": "E",
    "KeyR": "R",
    "KeyF": "F",
    "Space": "Space",
    "ShiftLeft": "Shift",
    "ShiftRight": "Shift",
    "ArrowUp": "↑",
    "ArrowDown": "↓",
    "ArrowLeft": "←",
    "ArrowRight": "→",
    "Mouse0": "M0",
    "Mouse1": "M1",
    "Mouse2": "M2",
}


def short_key_label(code: str) -> str:
    if code in KEY_SHORT_LABELS:
        return KEY_SHORT_LABELS[code]
    if code.startswith("Key") and len(code) == 4:
        return code[3:]
    if code.startswith("Digit") and len(code) == 6:
        return code[5:]
    return code


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 3)
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, 3)


def summarize_metric(values: list[float | None]) -> dict[str, Any]:
    clean = sorted(
        float(v)
        for v in values
        if type(v) in (int, float) and math.isfinite(v) and v >= 0
    )
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "min": None,
        }
    return {
        "count": len(clean),
        "mean": round(sum(clean) / len(clean), 3),
        "median": round(statistics.median(clean), 3),
        "p95": _percentile(clean, 0.95),
        "max": round(clean[-1], 3),
        "min": round(clean[0], 3),
    }


def build_key_hold_spans(
    input_events: list[dict[str, Any]],
    duration_seconds: float,
    *,
    initial_held_keys: list[str] | None = None,
    initial_held_buttons: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """根据按键事件序列还原持续按住条带（spans）与瞬时边沿事件列表（markers）。"""
    active_since: dict[str, float] = {}
    for k in initial_held_keys or []:
        active_since[str(k)] = 0.0
    for b in initial_held_buttons or []:
        active_since[f"Mouse{b}"] = 0.0

    spans: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []

    sorted_events = sorted(input_events, key=lambda x: float(x.get("rel_time_s", 0.0)))
    for item in sorted_events:
        t = max(0.0, min(duration_seconds, float(item.get("rel_time_s", 0.0))))
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        raw_evts = (
            payload.get("events") if isinstance(payload.get("events"), list) else []
        )
        held_keys = [str(k) for k in (payload.get("held_keys") or [])]
        held_btns = [f"Mouse{b}" for b in (payload.get("held_buttons") or [])]
        now_held = set(held_keys) | set(held_btns)

        for ev in raw_evts:
            if not isinstance(ev, dict):
                continue
            etype = str(ev.get("type", ""))
            code = ev.get("code")
            if code is None and "button" in ev:
                code = f"Mouse{ev.get('button')}"
            code_str = str(code or etype)
            markers.append(
                {
                    "time_s": round(t, 4),
                    "type": etype,
                    "code": code_str,
                    "label": short_key_label(code_str),
                    "sequence": item.get("sequence"),
                }
            )
            if etype in {"keydown", "pointerdown"}:
                active_since.setdefault(code_str, t)
            elif etype in {"keyup", "pointerup"}:
                start_t = active_since.pop(code_str, t)
                spans.append(
                    {
                        "code": code_str,
                        "label": short_key_label(code_str),
                        "start_s": round(start_t, 4),
                        "end_s": round(max(t, start_t + 0.025), 4),
                        "duration_ms": round(max(0.0, (t - start_t) * 1000.0), 1),
                    }
                )

        # Reconcile state with snapshot held_keys / held_buttons
        for k in list(active_since.keys()):
            if k not in now_held:
                start_t = active_since.pop(k)
                spans.append(
                    {
                        "code": k,
                        "label": short_key_label(k),
                        "start_s": round(start_t, 4),
                        "end_s": round(max(t, start_t + 0.025), 4),
                        "duration_ms": round(max(0.0, (t - start_t) * 1000.0), 1),
                    }
                )
        for k in now_held:
            active_since.setdefault(k, t)

    for code_str, start_t in sorted(active_since.items(), key=lambda kv: kv[1]):
        if duration_seconds > start_t:
            spans.append(
                {
                    "code": code_str,
                    "label": short_key_label(code_str),
                    "start_s": round(start_t, 4),
                    "end_s": round(duration_seconds, 4),
                    "duration_ms": round((duration_seconds - start_t) * 1000.0, 1),
                }
            )

    # Assign non-overlapping lane track index (0, 1, 2) for concurrent keys
    spans.sort(key=lambda s: (s["start_s"], s["end_s"], s["code"]))
    track_ends: list[float] = []
    for sp in spans:
        placed = False
        for tidx, tend in enumerate(track_ends):
            if sp["start_s"] >= tend + 0.03:
                sp["track"] = tidx
                track_ends[tidx] = sp["end_s"]
                placed = True
                break
        if not placed:
            sp["track"] = len(track_ends)
            track_ends.append(sp["end_s"])

    return spans, markers


def _latent_for_pixel(frame):
    clip, offset = divmod(frame, 17)
    return clip * 5 + (0 if offset == 0 else 1 + (offset - 1) // 4)


def build_stage_groups(selected_frames):
    """Protocol grouping over duration illustrations, not measured CUDA spans."""
    groups = {role: {} for role in STAGE_ROLES}
    batches = {}
    for row in selected_frames:
        frame = row["model_frame"]
        clip, offset = divmod(frame, 17)
        enc_group = clip * 3 + (0 if offset == 0 else 1 if offset < 9 else 2)
        keys = {
            "encoder": enc_group,
            "dit": row.get("chunk_index"),
            "decoder": _latent_for_pixel(frame),
        }
        for role, key in keys.items():
            if key is not None and row["projected"].get(role) is not None:
                groups[role].setdefault(key, []).append(row)
        start, count = row.get("output_batch_start"), row.get("output_batch_frames")
        if type(start) is int and type(count) is int and count > 0:
            batch = batches.setdefault(
                start,
                {
                    "output_frame_start": start,
                    "num_frames": count,
                    "received_s": row["output_s"],
                    "observed_frames": 0,
                },
            )
            batch["observed_frames"] += 1
            batch["received_s"] = max(batch["received_s"], row["output_s"])

    result = {role: [] for role in STAGE_ROLES}
    for role, buckets in groups.items():
        for key, rows in buckets.items():
            first = min(r["model_frame"] for r in rows)
            stop = max(r["model_frame"] for r in rows) + 1
            # Average duplicated frame attribution; never sum it as GPU work.
            start = statistics.mean(r["projected"][role]["start_s"] for r in rows)
            wait, process, ret = (
                statistics.mean(r["stages"][role][m] for r in rows) / 1000
                for m in STAGE_METRICS
            )
            latents = len({_latent_for_pixel(r["model_frame"]) for r in rows})
            packet_anchor = role == "decoder" and all(
                r.get("output_batch_start") == first
                and r.get("output_batch_frames") == len(rows)
                for r in rows
            )
            if packet_anchor:
                # Align a duration illustration with its observed packet end;
                # this does not turn an attributed chunk duration into a trace.
                start = (
                    statistics.mean(r["output_s"] for r in rows) - wait - process - ret
                )
            labels = {
                "encoder": f"E{key}: {len(rows)}px → {latents}L",
                "dit": f"C{key}: {latents}L / {len(rows)}px",
                "decoder": f"L{key}: 1L → {len(rows)}px",
            }
            result[role].append(
                {
                    "id": key,
                    "label": labels[role],
                    "frame_start": first,
                    "frame_end": stop,
                    "pixel_frames": len(rows),
                    "latent_frames": latents,
                    "start_s": start,
                    "proc_start_s": start + wait,
                    "ret_start_s": start + wait + process,
                    "end_s": start + wait + process + ret,
                    "timing_kind": "duration_illustration",
                }
            )
    return result, sorted(batches.values(), key=lambda b: b["output_frame_start"])


def build_timeline_dataset(
    *,
    case_meta: dict[str, Any],
    source_frames: list[dict[str, Any]],
    selected_frames: list[dict[str, Any]],
    frame_stage_breakdown: list[dict[str, Any]],
    input_events: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """整合白膜源帧、玩家按键、选中帧 Room 侧计时与三角色逐帧分解耗时。"""
    stage_by_output: dict[int, dict[str, Any]] = {}
    for entry in frame_stage_breakdown:
        if isinstance(entry, dict) and type(entry.get("output_frame")) is int:
            stage_by_output[entry["output_frame"]] = entry

    if source_frames:
        t_origin = float(source_frames[0]["capture_time_s"])
    elif selected_frames:
        t_origin = float(selected_frames[0]["capture_time_s"])
    else:
        t_origin = 0.0

    norm_sources: list[dict[str, Any]] = []
    for sf in source_frames:
        idx = int(sf["source_frame_index"])
        cap_s = round(max(0.0, float(sf["capture_time_s"]) - t_origin), 6)
        pts_s = round(float(sf.get("pts_s", cap_s)), 6)
        norm_sources.append(
            {
                "source_frame": idx,
                "capture_s": cap_s,
                "pts_s": pts_s,
                "sdk_frame": int(sf.get("sdk_frame", idx)),
                "held_keys": list(sf.get("held_keys") or []),
            }
        )

    duration_seconds = max(
        [row["capture_s"] for row in norm_sources]
        + [float(case_meta.get("duration_s", 0.0)), 0.1]
    )
    key_spans, key_markers = build_key_hold_spans(
        input_events,
        duration_seconds,
        initial_held_keys=case_meta.get("initial_held_keys"),
        initial_held_buttons=case_meta.get("initial_held_buttons"),
    )

    # Backfill held_keys onto source frames from key_spans if not already populated
    for sf in norm_sources:
        t = sf["capture_s"]
        active = sorted(
            {sp["label"] for sp in key_spans if sp["start_s"] <= t <= sp["end_s"]}
        )
        if active:
            sf["held_keys"] = active

    source_time_map = {sf["source_frame"]: sf["capture_s"] for sf in norm_sources}

    enriched_selected: list[dict[str, Any]] = []
    for sel in sorted(selected_frames, key=lambda r: int(r["model_frame"])):
        s_idx = int(sel["source_frame"])
        m_idx = int(sel["model_frame"])
        cap_s = source_time_map.get(
            s_idx,
            round(max(0.0, float(sel.get("capture_time_s", t_origin)) - t_origin), 6),
        )
        src_to_sel_ms = round(
            max(0.0, float(sel.get("source_to_selection_ms", 0.0))), 3
        )
        sel_to_disp_ms = round(
            max(0.0, float(sel.get("selection_to_dispatch_ms", 0.0))), 3
        )
        disp_to_out_ms = round(
            max(0.0, float(sel.get("dispatch_to_raw_output_ms", 0.0))), 3
        )
        src_to_out_ms = round(
            max(
                src_to_sel_ms + sel_to_disp_ms + disp_to_out_ms,
                float(sel.get("source_to_raw_output_ms", 0.0)),
            ),
            3,
        )
        stage_info = stage_by_output.get(m_idx, {})
        stages_out: dict[str, dict[str, float]] = {}
        for role in STAGE_ROLES:
            raw_r = (
                stage_info.get(role) if isinstance(stage_info.get(role), dict) else {}
            )
            metrics = {}
            for metric in STAGE_METRICS:
                value = raw_r.get(metric)
                metrics[metric] = (
                    round(value, 3)
                    if type(value) in (int, float)
                    and math.isfinite(value)
                    and value >= 0
                    else None
                )
            metrics["total_ms"] = (
                round(sum(metrics.values()), 3)
                if all(v is not None for v in metrics.values())
                else None
            )
            stages_out[role] = metrics

        totals = [st["total_ms"] for st in stages_out.values()]
        stage_sum_ms = (
            round(sum(totals), 3) if all(v is not None for v in totals) else None
        )
        t_sel_s = cap_s + src_to_sel_ms / 1000.0
        t_disp_s = t_sel_s + sel_to_disp_ms / 1000.0
        t_out_s = t_disp_s + disp_to_out_ms / 1000.0

        # Durations do not determine absolute stage boundaries. Keep the old
        # serial illustration explicitly separate from the measured output point.
        cursor_s = t_disp_s
        projected = {role: None for role in STAGE_ROLES}
        if stage_sum_ms is not None:
            for role in STAGE_ROLES:
                st = stages_out[role]
                s0 = cursor_s
                s_proc = s0 + st["wait_ms"] / 1000.0
                s_ret = s_proc + st["process_ms"] / 1000.0
                s1 = s_ret + st["return_ms"] / 1000.0
                projected[role] = {
                    "start_s": round(s0, 6),
                    "proc_start_s": round(s_proc, 6),
                    "ret_start_s": round(s_ret, 6),
                    "end_s": round(s1, 6),
                }
                cursor_s = s1

        active_keys = sorted(
            {sp["label"] for sp in key_spans if sp["start_s"] <= cap_s <= sp["end_s"]}
        )
        blocked_dict = sel.get("input_capacity_blocked_ms") or {}
        enriched_selected.append(
            {
                "source_frame": s_idx,
                "model_frame": m_idx,
                "chunk_index": sel.get("chunk_index"),
                "output_batch_start": sel.get("output_batch_start"),
                "output_batch_frames": sel.get("output_batch_frames"),
                "prompt_version": int(sel.get("prompt_version", 1)),
                "capture_s": round(cap_s, 6),
                "select_s": round(t_sel_s, 6),
                "dispatch_s": round(t_disp_s, 6),
                "output_s": round(t_out_s, 6),
                "source_to_selection_ms": src_to_sel_ms,
                "selection_to_dispatch_ms": sel_to_disp_ms,
                "dispatch_to_raw_output_ms": disp_to_out_ms,
                "source_to_raw_output_ms": src_to_out_ms,
                "stages_total_ms": stage_sum_ms,
                "admission_wait_ms": round(
                    float(sel.get("input_admission_wait_ms", 0.0)), 3
                ),
                "blocked_ms": {
                    k: round(float(blocked_dict.get(k, 0.0)), 3)
                    for k in (
                        "pair_capacity",
                        "encoder_credit",
                        "output_window",
                        "condition_switch",
                    )
                },
                "held_keys": active_keys,
                "stages": stages_out,
                "projected": projected,
            }
        )

    stage_groups, output_batches = build_stage_groups(enriched_selected)

    # Detect skipped-source-frame intervals (anomalies / natural frame skipping)
    skip_intervals: list[dict[str, Any]] = []
    for prev, curr in zip(enriched_selected, enriched_selected[1:]):
        skipped = curr["source_frame"] - prev["source_frame"] - 1
        gap_s = curr["capture_s"] - prev["capture_s"]
        if skipped >= 1:
            blk = curr["blocked_ms"]
            dominant_reason = (
                max(blk, key=lambda k: blk[k])
                if any(blk.values())
                else "natural_pacing"
            )
            skip_intervals.append(
                {
                    "start_s": round(prev["capture_s"], 4),
                    "end_s": round(curr["capture_s"], 4),
                    "gap_seconds": round(gap_s, 3),
                    "skipped": int(skipped),
                    "source_before": prev["source_frame"],
                    "source_after": curr["source_frame"],
                    "model_after": curr["model_frame"],
                    "wait_ms": curr["admission_wait_ms"],
                    "blocked_ms": blk,
                    "reason": dominant_reason,
                }
            )
    skip_intervals.sort(key=lambda x: (-x["skipped"], -x["gap_seconds"]))

    stage_summary: dict[str, dict[str, Any]] = {}
    for role in STAGE_ROLES:
        stage_summary[role] = {
            metric: summarize_metric(
                [row["stages"][role][metric] for row in enriched_selected]
            )
            for metric in (*STAGE_METRICS, "total_ms")
        }
        for stats in stage_summary[role].values():
            stats["missing_count"] = len(enriched_selected) - stats["count"]
            stats["coverage"] = (
                stats["count"] / len(enriched_selected) if enriched_selected else 0.0
            )

    plot_horizon_s = max(
        [duration_seconds]
        + [row["output_s"] for row in enriched_selected]
        + [
            pj["end_s"]
            for row in enriched_selected
            for pj in row["projected"].values()
            if pj
        ]
    )

    summary = {
        "mode": mode,
        "case_id": case_meta.get("case_id", "wtp-clip"),
        "world_id": case_meta.get("world_id", ""),
        "archive_recording_id": case_meta.get("archive_recording_id", ""),
        "clip_window_s": [
            round(float(case_meta.get("clip_start_s", 0.0)), 3),
            round(
                float(case_meta.get("clip_start_s", 0.0))
                + float(case_meta.get("duration_s", duration_seconds)),
                3,
            ),
        ],
        "source_duration_seconds": round(duration_seconds, 3),
        "plot_horizon_seconds": round(plot_horizon_s, 3),
        "source_frames": len(norm_sources),
        "selected_model_frames": len(enriched_selected),
        "skipped_source_frames": max(0, len(norm_sources) - len(enriched_selected)),
        "input_event_packets": len(input_events),
        "key_hold_spans": len(key_spans),
        "key_edge_events": len(key_markers),
        "source_to_selection_ms": summarize_metric(
            [r["source_to_selection_ms"] for r in enriched_selected]
        ),
        "selection_to_dispatch_ms": summarize_metric(
            [r["selection_to_dispatch_ms"] for r in enriched_selected]
        ),
        "dispatch_to_raw_output_ms": summarize_metric(
            [r["dispatch_to_raw_output_ms"] for r in enriched_selected]
        ),
        "source_to_raw_output_ms": summarize_metric(
            [r["source_to_raw_output_ms"] for r in enriched_selected]
        ),
        "stages": stage_summary,
    }

    return {
        "schema_version": 2,
        "timeline_kind": "serial_duration_illustration",
        "stage_groups": stage_groups,
        "output_batches": output_batches,
        "summary": summary,
        "source_frames": norm_sources,
        "selected_frames": enriched_selected,
        "skip_intervals": skip_intervals,
        "key_spans": key_spans,
        "key_markers": key_markers,
    }


def render_html_report(dataset: dict[str, Any], output_html: Path) -> Path:
    """渲染自包含中文交互式逐帧耗时分解与按键可视化页面。"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    embedded_json = json.dumps(dataset, ensure_ascii=False, separators=(",", ":"))
    # JSON is embedded in an HTML script element, whose parser recognizes
    # closing tags even inside JavaScript strings. Preserve JSON round trips.
    for char in ("&", "<", ">", "\u2028", "\u2029"):
        embedded_json = embedded_json.replace(char, f"\\u{ord(char):04x}")
    mode = dataset.get("summary", {}).get("mode")
    origin_label = (
        "真实服务模式（real-sglang；硬件与运行回执见实测记录）"
        if mode == "real-sglang"
        else "模拟模式（mock；非真实 GPU 实测）"
        if mode in {"mock", "inproc-mock"}
        else "来源未确认，请核对原始 JSON 与运行回执"
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SGLang 白膜自适应供帧与三角色逐帧耗时分解</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172b35;
  --muted: #52656d;
  --grid: #dce5e8;
  --supply: #a6b4bb;
  --chosen: #00866d;
  --dispatch: #2874bd;
  --wait: #f59e0b;
  --enc-proc: #2563eb;
  --dit-proc: #7c3aed;
  --dec-proc: #059669;
  --ret: #ea580c;
  --raw: #c16c15;
  --key-bg: #eef2ff;
  --key-border: #4f46e5;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: #f4f7f8;
  color: var(--ink);
  font: 14px/1.6 -apple-system, BlinkMacSystemFont, "PingFang SC", "Noto Sans CJK SC", system-ui, sans-serif;
}}
main {{ max-width: 1360px; margin: auto; padding: 24px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 18px; margin: 0 0 10px; }}
p {{ margin: 6px 0; }}
.muted {{ color: var(--muted); font-size: 13px; }}
section {{ background: #fff; border-radius: 12px; padding: 18px 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(15,23,42,0.04); }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 14px 0; }}
.stat {{ background: #fff; border-radius: 10px; padding: 12px 14px; border: 1px solid #e3ecee; }}
.stat strong {{ font-size: 22px; display: block; font-variant-numeric: tabular-nums; }}
.stat span {{ font-size: 12px; color: var(--muted); }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; margin-bottom: 8px; align-items: center; }}
.dot {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
section.chart-expanded {{ position:fixed; inset:0; z-index:1000; overflow:auto;
  margin:0; border-radius:0; background:white; }}
canvas {{ width: 100%; display: block; cursor: crosshair; }}
button {{
  font: inherit; border: 1px solid #cbd8de; border-radius: 6px;
  padding: 5px 11px; background: #fff; color: var(--ink); cursor: pointer;
}}
button:hover, button.active {{ background: #e5f3f0; border-color: var(--chosen); font-weight: 600; }}
.controls {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 10px 0 12px; }}
input[type=range] {{ flex: 1; min-width: 200px; accent-color: var(--chosen); }}
#inspection {{
  min-height: 118px; background: #f1f6f7; padding: 12px 14px;
  border-radius: 8px; font-variant-numeric: tabular-nums; margin-top: 10px;
  border-left: 4px solid var(--chosen);
}}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e6eef0; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; font-weight: 600; }}
th {{ background: #f8fbfc; color: var(--muted); font-weight: 600; }}
.skip-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 8px; }}
.skip-grid button {{ text-align: left; }}
.skip-grid small {{ display: block; color: var(--muted); font-size: 12px; }}
.pill {{ display: inline-block; padding: 1px 7px; border-radius: 999px; background: var(--key-bg); border: 1px solid var(--key-border); color: #3730a3; font-size: 12px; font-weight: 600; margin-right: 4px; }}
</style>
</head>
<body>
<main>
  <h1>白膜自适应供帧 + 玩家按键 + Encoder / DiT / Decoder 逐帧耗时分解</h1>
  <p><strong>数据来源：{origin_label}</strong></p>
  <p class="muted" id="subtitle"></p>
  <div class="stats" id="stats"></div>

  <section>
    <h2>三角色逐帧耗时分解统计（单位：ms，针对每个进入 SGLang 并产出的帧）</h2>
    <table>
      <thead>
        <tr>
          <th>角色阶段</th>
          <th>等待攒帧 (中位 / P95 / 均值)</th>
          <th>处理计算 (中位 / P95 / 均值)</th>
          <th>返回结果 (中位 / P95 / 均值)</th>
          <th>角色小计 (中位 / P95 / 均值)</th>
        </tr>
      </thead>
      <tbody id="stageTable"></tbody>
    </table>
  </section>

  <section>
    <h2>全程概览：每秒白膜供给 vs 入模产出 & 玩家按键活跃带（点击可快速定位）</h2>
    <div class="legend">
      <span class="dot"><i class="swatch" style="background:var(--supply)"></i>白膜供给 (帧/秒)</span>
      <span class="dot"><i class="swatch" style="background:var(--chosen)"></i>入模并产出 (帧/秒)</span>
      <span class="dot"><i class="swatch" style="background:var(--key-border)"></i>玩家按键按下区间</span>
      <span class="dot"><i class="swatch" style="background:#f1c78d"></i>Room 自适应跳帧区间</span>
    </div>
    <canvas id="overview" height="175"></canvas>
  </section>

  <section>
    <h2>多泳道逐帧轨迹：白膜供给+按键 → 派发 → Encoder / DiT / Decoder (等待攒帧·处理·返回) → 调用方收到输出</h2>
    <p class="muted">每个 17 pixel-frame 段：Encoder [1,8,8] px → [1,2,2] latent；Lynn 每步输入 1 latent → [1,4,4,4,4] px。DiT 黄色 = 等输入/攒 chunk/排队，紫色 = 去噪。阶段条是耗时示意；最下行是模型 WS→Room 实际收到的批次，不是浏览器显示批次。是否逐 latent 返回由实际部署配置决定；输出泳道按采集到的批次绘制，不从模型形态推断发布粒度。</p>
    <div class="controls">
      <strong id="windowLabel"></strong>
      <button id="zoomIn">放大 +</button><button id="zoomOut">缩小 −</button>
      <label>时间窗 <input id="zoom" type="range" min="0.1" step="0.05" aria-label="时间缩放"></label>
      <label>图高 <input id="heightZoom" type="range" min="1" max="2.5" step="0.1" value="1" aria-label="图高缩放"></label>
      <button id="fullScreen">全屏图表</button>
      <button data-span="0.25">0.25 秒（看 latent）</button>
      <button data-span="1">1 秒</button>
      <button data-span="3">3 秒（看单帧条块）</button>
      <button data-span="6" class="active">6 秒</button>
      <button data-span="12">12 秒</button>
      <button data-span="0">全程</button>
      <input id="position" type="range" min="0" step="0.001" value="0" aria-label="时间窗口起点">
    </div>
    <div class="legend">
      <span class="dot"><i class="swatch" style="background:var(--supply)"></i>未选中源帧</span>
      <span class="dot"><i class="swatch" style="background:var(--chosen)"></i>选中源帧</span>
      <span class="dot"><i class="swatch" style="background:var(--key-border)"></i>按键持续条带</span>
      <span class="dot"><i class="swatch" style="background:var(--wait)"></i>等待输入/攒帧/排队 (wait_ms)</span>
      <span class="dot"><i class="swatch" style="background:var(--enc-proc)"></i>Encoder 处理</span>
      <span class="dot"><i class="swatch" style="background:var(--dit-proc)"></i>DiT 处理</span>
      <span class="dot"><i class="swatch" style="background:var(--dec-proc)"></i>Decoder 处理</span>
      <span class="dot"><i class="swatch" style="background:var(--ret)"></i>返回给下游/调用方 (return_ms)</span>
    </div>
    <canvas id="detail" height="590"></canvas>
    <div id="inspection">将鼠标移到任意白膜源帧或耗时条上，查看该帧的按键状态以及在 Encoder / DiT / Decoder 的等待攒帧、处理计算与返回结果耗时。</div>
  </section>

  <section>
    <h2>自适应跳帧区间（点击跳转到对应局部窗口）</h2>
    <div class="skip-grid" id="skips"></div>
  </section>
</main>
<script>
const D = {embedded_json};
const S = D.summary;
const $ = id => document.getElementById(id);
// All dataset values interpolated into HTML must pass through this helper.
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({{
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}})[char]);
const fmt = (n, d=1) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(d) : '未测';
const horizon = Math.max(S.plot_horizon_seconds || S.source_duration_seconds || 1, 1);
let span = Math.min(3, horizon);
let heightScale = 1;
let start = 0;
let hoverSource = D.selected_frames.length ? D.selected_frames[Math.min(12, D.selected_frames.length - 1)].source_frame : null;

const selectedBySource = new Map(D.selected_frames.map(r => [r.source_frame, r]));

$('subtitle').textContent =
  `案例 ${{S.case_id}} (${{S.world_id || 'WTP'}}) · 截取窗口 ${{fmt(S.clip_window_s[0],1)}}–${{fmt(S.clip_window_s[1],1)}}s (共 ${{fmt(S.source_duration_seconds,2)}}s) · 运行模式: ${{S.mode}}。阶段条为串行耗时示意，不表示真实起止或重叠；输出点使用实测时间。`;

const statItems = [
  [S.source_frames, '白膜供给总帧数'],
  [S.selected_model_frames, '入模并产出帧数'],
  [S.skipped_source_frames, '自适应跳过源帧'],
  [S.key_hold_spans + ' 段 / ' + S.input_event_packets + ' 包', '注入按键区间 / 事件包'],
  [fmt(S.dispatch_to_raw_output_ms.median) + ' ms', '派发→收到输出 (中位)'],
  [fmt(S.dispatch_to_raw_output_ms.p95) + ' ms', '派发→收到输出 (P95)'],
];
$('stats').innerHTML = statItems.map(([v, l]) => `<div class="stat"><strong>${{escapeHtml(v)}}</strong><span>${{escapeHtml(l)}}</span></div>`).join('');

const roleNames = [['encoder', 'Encoder (VAE 编码)'], ['dit', 'DiT (AR 去噪)'], ['decoder', 'Decoder (流式解码+输出)']];
$('stageTable').innerHTML = roleNames.map(([ key, title ]) => {{
  const st = S.stages[key];
  const cell = m => `${{fmt(st[m].median)}} / ${{fmt(st[m].p95)}} / ${{fmt(st[m].mean)}} (n=${{st[m].count}}, 缺失=${{st[m].missing_count}})`;
  return `<tr><td>${{escapeHtml(title)}}</td><td>${{escapeHtml(cell('wait_ms'))}}</td><td>${{escapeHtml(cell('process_ms'))}}</td><td>${{escapeHtml(cell('return_ms'))}}</td><td><strong>${{escapeHtml(cell('total_ms'))}}</strong></td></tr>`;
}}).join('');

function setupCanvas(id, h) {{
  const el = $(id);
  const dpr = window.devicePixelRatio || 1;
  const w = el.clientWidth || 1200;
  el.width = w * dpr;
  el.height = h * dpr;
  const c = el.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  c.font = '12px -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';
  return [c, w, h];
}}

function line(c, x1, y1, x2, y2, stroke, width=1) {{
  c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2);
  c.strokeStyle = stroke; c.lineWidth = width; c.stroke();
}}

function drawOverview() {{
  const [c, w, h] = setupCanvas('overview', 175);
  const l = 56, r = w - 16;
  const bins = Math.max(1, Math.ceil(horizon));
  const sup = Array(bins).fill(0), sel = Array(bins).fill(0);
  D.source_frames.forEach(f => {{ const b = Math.min(bins - 1, Math.floor(f.capture_s)); sup[b]++; }});
  D.selected_frames.forEach(f => {{ const b = Math.min(bins - 1, Math.floor(f.capture_s)); sel[b]++; }});
  const x = t => l + (t / horizon) * (r - l);
  const y = n => 122 - n * 2.8;
  for (let n = 0; n <= 30; n += 10) {{
    line(c, l, y(n), r, y(n), '#e5ecef');
    c.fillStyle = '#52656d'; c.fillText(String(n), 24, y(n) + 4);
  }}
  c.fillStyle = '#52656d'; c.fillText('帧/秒', 12, 18);
  const bw = (r - l) / bins;
  for (let i = 0; i < bins; i++) {{
    c.fillStyle = '#a6b4bb';
    c.fillRect(x(i), y(sup[i]), Math.max(1, bw - 1), sup[i] * 2.8);
    c.fillStyle = '#00866d';
    c.fillRect(x(i), y(sel[i]), Math.max(1, bw * 0.62), sel[i] * 2.8);
  }}
  D.key_spans.forEach(sp => {{
    c.fillStyle = 'rgba(79,70,229,0.75)';
    c.fillRect(x(sp.start_s), 130, Math.max(2, x(sp.end_s) - x(sp.start_s)), 7);
  }});
  D.skip_intervals.forEach(sk => {{
    c.fillStyle = '#f1c78d';
    c.fillRect(x(sk.start_s), 141, Math.max(2, x(sk.end_s) - x(sk.start_s)), 7);
  }});
  const curSpan = span <= 0 ? horizon : Math.min(span, horizon);
  c.strokeStyle = '#2874bd'; c.lineWidth = 2;
  c.strokeRect(x(start), 22, Math.max(4, x(Math.min(horizon, start + curSpan)) - x(start)), 130);
  const stepS = horizon <= 20 ? 2 : 5;
  c.fillStyle = '#52656d';
  for (let t = 0; t <= horizon; t += stepS) c.fillText(t + 's', Math.min(r - 22, x(t) - 6), 166);
}}

function drawDetail() {{
  const [c, w, h] = setupCanvas('detail', 590 * heightScale);
  c.scale(1, heightScale);
  const l = 156, r = w - 18;
  const effSpan = span <= 0 ? horizon : Math.min(span, horizon);
  const x = t => l + ((t - start) / effSpan) * (r - l);

  const lanes = [
    {{ y: 54, label: '白膜供给' }},
    {{ y: 108, label: '按键输入 (供给)' }},
    {{ y: 168, label: '选中并派发' }},
    {{ y: 256, label: 'Encoder [1,8,8] px' }},
    {{ y: 356, label: 'DiT（latent chunk）' }},
    {{ y: 456, label: 'Lynn：1 latent/步' }},
    {{ y: 534, label: '模型 WS → Room' }},
  ];

  const ticks = Math.max(4, Math.min(12, Math.floor((r - l) / 95)));
  for (let i = 0; i <= ticks; i++) {{
    const t = start + (effSpan * i) / ticks;
    line(c, x(t), 28, x(t), 552, '#edf2f4');
    c.fillStyle = '#52656d';
    c.fillText(fmt(t, 2) + 's', Math.min(r - 38, x(t) - 14), 574);
  }}

  lanes.forEach(ln => {{
    line(c, l, ln.y, r, ln.y, '#dce5e8');
    c.fillStyle = '#172b35';
    c.font = '600 12.5px -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';
    c.fillText(ln.label, 8, ln.y + 4);
  }});
  c.font = '12px -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';
  c.fillStyle = '#52656d';
  c.fillText('(等待攒帧/处理/返回)', 8, lanes[3].y + 20);
  c.fillText('(黄色=等输入/排队)', 8, lanes[4].y + 20);
  c.fillText('(等待队列/处理/返回)', 8, lanes[5].y + 20);

  c.save();
  c.beginPath(); c.rect(l, 20, r - l, 540); c.clip();

  // Skipped intervals highlight (matches user reference screenshot)
  D.skip_intervals.filter(v => v.end_s >= start && v.start_s <= start + effSpan).forEach(v => {{
    const x0 = Math.max(l, x(v.start_s)), x1 = Math.min(r, x(v.end_s));
    c.fillStyle = '#fff0d8';
    c.fillRect(x0, 66, Math.max(2, x1 - x0), 28);
    if (x1 - x0 > 68) {{
      c.fillStyle = '#c16c15';
      c.fillText(`${{fmt(v.gap_seconds, 2)}}s / 跳过${{v.skipped}}帧`, x0 + 4, 84);
    }}
  }});

  // Player key hold spans in whitebox supply companion lane
  D.key_spans.filter(sp => sp.end_s >= start && sp.start_s <= start + effSpan).forEach(sp => {{
    const x0 = Math.max(l, x(sp.start_s)), x1 = Math.min(r, x(sp.end_s));
    const bw = Math.max(6, x1 - x0);
    const trackOffset = (sp.track % 2) * 18 - 8;
    const ky = lanes[1].y + trackOffset;
    c.fillStyle = '#e0e7ff';
    c.strokeStyle = '#4f46e5';
    c.lineWidth = 1.2;
    c.beginPath();
    c.roundRect(x0, ky - 8, bw, 16, 4);
    c.fill(); c.stroke();
    if (bw > 18) {{
      c.fillStyle = '#312e81';
      c.font = '600 11px system-ui, sans-serif';
      const txt = bw > 64 ? `${{sp.label}} (${{fmt(sp.duration_ms/1000,2)}}s)` : sp.label;
      c.fillText(txt, x0 + 4, ky + 4);
      c.font = '12px -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';
    }}
  }});

  // Source frame ticks
  D.source_frames.filter(sf => sf.capture_s >= start && sf.capture_s <= start + effSpan).forEach(sf => {{
    const isSel = selectedBySource.has(sf.source_frame);
    const isHov = hoverSource === sf.source_frame;
    const sx = x(sf.capture_s);
    line(c, sx, lanes[0].y - 10, sx, lanes[0].y + 10, isSel ? '#00866d' : '#a6b4bb', isHov ? 2.8 : 1.1);
    if (isSel) {{
      c.globalAlpha = isHov ? 0.85 : 0.25;
      line(c, sx, lanes[0].y + 10, x(selectedBySource.get(sf.source_frame).dispatch_s), lanes[2].y - 7, '#00866d', isHov ? 1.8 : 0.9);
      c.globalAlpha = 1;
    }}
  }});

  // Draw one bar per codec/AR group instead of duplicating it per pixel.
  const roleCfg = [
    ['encoder', lanes[3].y, '#2563eb'],
    ['dit', lanes[4].y, '#7c3aed'],
    ['decoder', lanes[5].y, '#059669'],
  ];
  roleCfg.forEach(([role, baseY, color]) => {{
    (D.stage_groups?.[role] || []).forEach(g => {{
      if (g.end_s < start || g.start_s > start + effSpan) return;
      const rowSlot = (Number(g.id) % 3 - 1) * 22;
      const y = baseY + rowSlot;
      const active = D.selected_frames.some(f => f.source_frame === hoverSource &&
        f.model_frame >= g.frame_start && f.model_frame < g.frame_end);
      [[g.start_s,g.proc_start_s,'#f59e0b'],[g.proc_start_s,g.ret_start_s,color],
       [g.ret_start_s,g.end_s,'#ea580c']].forEach(([a,b,col]) => {{
        c.fillStyle = col;
        c.fillRect(x(a), y-4, Math.max(1,x(b)-x(a)), 8);
      }});
      if (active) {{ c.strokeStyle='#172b35'; c.lineWidth=1.5;
        c.strokeRect(x(g.start_s)-1,y-6,Math.max(2,x(g.end_s)-x(g.start_s))+2,12); }}
      if (effSpan <= 3) {{
        c.fillStyle = '#172b35'; c.font = '11px system-ui,sans-serif';
        c.fillText(g.label, Math.max(l+2,x(g.proc_start_s)), y-8);
      }}
    }});
  }});
  D.selected_frames.forEach(f => {{
    const active = f.source_frame === hoverSource;
    line(c,x(f.dispatch_s),lanes[2].y-6,x(f.dispatch_s),lanes[2].y+6,'#2874bd',active?2.5:1);
    if (active) {{
      line(c,x(f.dispatch_s),lanes[2].y+8,x(f.output_s),lanes[6].y-8,'#64748b',1);
    }}
  }});
  if ((D.output_batches || []).length) {{
    D.output_batches.forEach((b,i) => {{
      if (b.received_s < start || b.received_s > start+effSpan) return;
      const bx = x(b.received_s), by = lanes[6].y;
      line(c,bx,by-10,bx,by+10,'#c16c15',3);
      c.fillStyle='#9a4d06'; c.font='11px system-ui,sans-serif';
      c.fillText(`${{b.num_frames}}px [${{b.output_frame_start}},${{b.output_frame_start+b.num_frames}})`,
        bx+4,by + (i%2 ? 28 : -16));
    }});
  }} else {{
    c.fillStyle='#52656d'; c.fillText('此旧数据未采集模型 WS 批次边界，不能推断为逐 latent 返回',l+6,lanes[6].y-12);
  }}
  c.restore();

  if (hoverSource !== null) {{
    const sf = D.source_frames.find(a => a.source_frame === hoverSource);
    if (sf && sf.capture_s >= start && sf.capture_s <= start + effSpan) {{
      const hx = x(sf.capture_s);
      line(c, hx, 18, hx, 552, '#172b35', 1.5);
      c.fillStyle = '#172b35';
      c.font = '600 12px -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';
      c.fillText('源帧 ' + sf.source_frame, Math.max(l + 4, Math.min(hx + 6, r - 86)), 16);
    }}
  }}

  $('windowLabel').textContent = `${{fmt(start, 2)}} – ${{fmt(Math.min(horizon, start + effSpan), 2)}} 秒`;
  $('zoom').max = horizon;
  $('zoom').value = effSpan;
  $('position').max = Math.max(0, horizon - effSpan);
  $('position').value = start;
  document.querySelectorAll('[data-span]').forEach(b => b.classList.toggle('active', Number(b.dataset.span) === span));
}}

function updateInspection(sourceIdx) {{
  const sf = D.source_frames.find(a => a.source_frame === sourceIdx);
  if (!sf) return;
  const m = selectedBySource.get(sourceIdx);
  const keysHtml = (sf.held_keys && sf.held_keys.length)
    ? sf.held_keys.map(k => `<span class="pill">${{escapeHtml(k)}}</span>`).join('')
    : '<span class="muted">无按键按下</span>';
  if (!m) {{
    $('inspection').innerHTML =
      `<strong>源帧 ${{escapeHtml(sf.source_frame)}} · 采集时刻 +${{fmt(sf.capture_s, 3)}} s</strong> &nbsp;|&nbsp; 当时白膜按键：${{keysHtml}}<br>` +
      `<span class="muted">该源帧在 Room 自适应控速下被跳过（SGLang 正在消费前序帧或等待信用窗口，恢复时自动取最新白膜帧）。</span>`;
    return;
  }}
  const enc = m.stages.encoder, dit = m.stages.dit, dec = m.stages.decoder;
  $('inspection').innerHTML =
    `<strong>源帧 ${{escapeHtml(sf.source_frame)}} → 模型输出帧 #${{escapeHtml(m.model_frame)}} (Chunk ${{escapeHtml(m.chunk_index ?? '-')}}) · 采集 +${{fmt(sf.capture_s, 3)}} s</strong> &nbsp;|&nbsp; 当时白膜按键：${{keysHtml}}<br>` +
    `模型 WS 返回批次：<strong>${{escapeHtml(m.output_batch_frames ?? '未采集')}} px</strong> · 起点 #${{escapeHtml(m.output_batch_start ?? '-')}}（Lynn 每 latent 常规 [1,4,4,4,4]，实际以本批为准）<br>` +
    `Room 侧：采集→选中 <strong>${{fmt(m.source_to_selection_ms)}} ms</strong> · 选中→派发 <strong>${{fmt(m.selection_to_dispatch_ms)}} ms</strong> · 派发→收到原始输出 <strong>${{fmt(m.dispatch_to_raw_output_ms)}} ms</strong>（端到端总计 <strong>${{fmt(m.source_to_raw_output_ms)}} ms</strong>）<br>` +
    `• <strong>Encoder</strong>：等待攒帧 <strong>${{fmt(enc.wait_ms)}} ms</strong> + 处理计算 <strong>${{fmt(enc.process_ms)}} ms</strong> + 返回给 DiT <strong>${{fmt(enc.return_ms)}} ms</strong> = <strong>${{fmt(enc.total_ms)}} ms</strong><br>` +
    `• <strong>DiT</strong>：等待攒帧 <strong>${{fmt(dit.wait_ms)}} ms</strong> + 去噪处理 <strong>${{fmt(dit.process_ms)}} ms</strong> + 返回给 Decoder <strong>${{fmt(dit.return_ms)}} ms</strong> = <strong>${{fmt(dit.total_ms)}} ms</strong><br>` +
    `• <strong>Decoder</strong>：等待队列 <strong>${{fmt(dec.wait_ms)}} ms</strong> + 解码处理 <strong>${{fmt(dec.process_ms)}} ms</strong> + 返回调用方 <strong>${{fmt(dec.return_ms)}} ms</strong> = <strong>${{fmt(dec.total_ms)}} ms</strong>`;
}}

function renderAll() {{
  const effSpan = span <= 0 ? horizon : Math.min(span, horizon);
  start = Math.max(0, Math.min(start, horizon - effSpan));
  drawOverview();
  drawDetail();
  if (hoverSource !== null) updateInspection(hoverSource);
}}

const reasonLabels = {{
  encoder_credit: 'Encoder 信用满',
  output_window: '输出窗口在途上限',
  pair_capacity: '配对容量满',
  condition_switch: '条件切换栅栏',
  natural_pacing: '自适应取新帧',
}};
$('skips').innerHTML = D.skip_intervals.length
  ? D.skip_intervals.slice(0, 8).map((v, i) =>
      `<button data-skip="${{i}}">${{fmt(v.start_s,2)}}–${{fmt(v.end_s,2)}}s · 跳过 ${{escapeHtml(v.skipped)}} 帧` +
      `<small>间隔 ${{fmt(v.gap_seconds,2)}}s · 主因: ${{escapeHtml(reasonLabels[v.reason] || v.reason)}}</small></button>`
    ).join('')
  : '<span class="muted">本段无跳帧区间。</span>';

document.querySelectorAll('[data-skip]').forEach(b => b.onclick = () => {{
  const v = D.skip_intervals[Number(b.dataset.skip)];
  span = Math.min(4, horizon);
  start = Math.max(0, v.start_s - 0.8);
  hoverSource = v.source_after;
  renderAll();
}});
document.querySelectorAll('[data-span]').forEach(b => b.onclick = () => {{
  span = Number(b.dataset.span);
  renderAll();
}});
// Zoom the shared time window around the pointer; never alter data timestamps.
function zoomTo(seconds, anchor=0.5) {{
  const old = span <= 0 ? horizon : Math.min(span,horizon);
  const next = Math.max(Math.min(0.1,horizon),Math.min(seconds,horizon));
  start += anchor * (old-next); span = next; renderAll();
}}
$('zoomIn').onclick = () => zoomTo((span || horizon)/2);
$('zoomOut').onclick = () => zoomTo((span || horizon)*2);
$('zoom').oninput = e => zoomTo(Number(e.target.value));
$('heightZoom').oninput = e => {{ heightScale=Number(e.target.value); renderAll(); }};
$('fullScreen').onclick = () => {{
  const expanded=$('detail').closest('section').classList.toggle('chart-expanded');
  document.body.style.overflow=expanded ? 'hidden' : '';
  $('fullScreen').textContent=expanded ? '退出全屏' : '全屏图表';
  renderAll();
}};
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    $('detail').closest('section').classList.remove('chart-expanded');
    document.body.style.overflow=''; $('fullScreen').textContent='全屏图表'; renderAll();
  }}
}});
$('detail').addEventListener('wheel', e => {{
  e.preventDefault();
  const rect=$('detail').getBoundingClientRect();
  const anchor=Math.max(0,Math.min(1,(e.clientX-rect.left-156)/Math.max(1,rect.width-174)));
  zoomTo((span || horizon)*Math.exp(e.deltaY*0.002),anchor);
}}, {{passive:false}});
$('position').oninput = e => {{ start = Number(e.target.value); renderAll(); }};
$('overview').onclick = e => {{
  const rect = e.currentTarget.getBoundingClientRect();
  const effSpan = span <= 0 ? horizon : Math.min(span, horizon);
  const t = ((e.clientX - rect.left - 56) / Math.max(1, rect.width - 72)) * horizon;
  start = t - effSpan / 2;
  renderAll();
}};
$('detail').onmousemove = e => {{
  const rect = e.currentTarget.getBoundingClientRect();
  const effSpan = span <= 0 ? horizon : Math.min(span, horizon);
  const t = start + ((e.clientX - rect.left - 156) / Math.max(1, rect.width - 174)) * effSpan;
  let best = null, bestDist = Infinity;
  D.source_frames.forEach(sf => {{
    const d = Math.abs(sf.capture_s - t);
    if (d < bestDist) {{ bestDist = d; best = sf; }}
  }});
  if (best) {{
    hoverSource = best.source_frame;
    drawDetail();
    updateInspection(best.source_frame);
  }}
}};
window.addEventListener('resize', renderAll);
renderAll();
</script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _load_cjk_font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    if bold:
        candidates = [
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            *candidates,
        ]
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fmt(value):
    return f"{value:.1f}" if type(value) in (int, float) else "未测"


def render_png_overview(
    dataset: dict[str, Any],
    output_png: Path,
    *,
    zoom_window_s: tuple[float, float] | None = None,
) -> Path:
    """使用 Pillow 渲染高清双视图静态总览 PNG（上半部：局部多泳道精细分解+按键；下半部：三角色耗时汇总卡片）。"""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    width, height = 2200, 1380
    img = Image.new("RGB", (width, height), (244, 247, 248))
    draw = ImageDraw.Draw(img, "RGBA")

    f_title = _load_cjk_font(30, bold=True)
    f_sub = _load_cjk_font(18)
    f_sec = _load_cjk_font(21, bold=True)
    f_lbl = _load_cjk_font(18, bold=True)
    f_txt = _load_cjk_font(16)
    f_sml = _load_cjk_font(14)

    summary = dataset["summary"]
    sources = dataset["source_frames"]
    selected = dataset["selected_frames"]
    skips = dataset["skip_intervals"]
    key_spans = dataset["key_spans"]
    sel_by_src = {r["source_frame"]: r for r in selected}

    # Header
    draw.text(
        (40, 26),
        "SGLang 白膜自适应供帧 + 按键注入 + Encoder / DiT / Decoder 逐帧耗时分解",
        fill=(23, 43, 53),
        font=f_title,
    )
    clip_w = summary.get("clip_window_s", [0, 0])
    draw.text(
        (40, 68),
        f"案例: {summary.get('case_id')}  |  截取窗口: {clip_w[0]:.1f}s–{clip_w[1]:.1f}s (共 {summary['source_duration_seconds']:.2f}s)"
        f"  |  白膜供给: {summary['source_frames']} 帧  |  入模产出: {summary['selected_model_frames']} 帧"
        f"  |  跳过: {summary['skipped_source_frames']} 帧  |  按键区间: {summary['key_hold_spans']} 段 ({summary['input_event_packets']} 包)"
        f"  |  模式: {summary['mode']}  |  阶段条为串行耗时示意，非实测起止；输出点为实测",
        fill=(82, 101, 109),
        font=f_sub,
    )

    # Choose an informative window (default: window rich in keys + steady-state chunks, ~8s, or full if short)
    horizon = max(
        float(
            summary.get("plot_horizon_seconds") or summary["source_duration_seconds"]
        ),
        1.0,
    )
    if zoom_window_s is not None:
        w_start, w_end = zoom_window_s
    elif horizon <= 10.0:
        w_start, w_end = 0.0, horizon
    else:
        # Pick 8.5s window starting near first active key span so key holds, skips, and stage bars are crisp
        first_key_t = key_spans[0]["start_s"] if key_spans else 1.0
        w_start = max(0.0, min(first_key_t - 0.6, horizon - 9.0))
        w_end = min(horizon, w_start + 9.0)
    w_span = max(0.5, w_end - w_start)

    # Pick a focus frame near the middle of the window that has a key pressed if possible
    candidates = [
        r
        for r in selected
        if w_start + 0.12 * w_span <= r["capture_s"] <= w_start + 0.65 * w_span
    ]
    with_keys = [r for r in candidates if r.get("held_keys")]
    focus = (
        with_keys[len(with_keys) // 2]
        if with_keys
        else (
            candidates[len(candidates) // 2]
            if candidates
            else (selected[0] if selected else None)
        )
    )

    # Main timeline card
    card_top, card_bot = 108, 985
    draw.rounded_rectangle(
        (32, card_top, width - 32, card_bot), radius=16, fill=(255, 255, 255)
    )
    draw.text(
        (58, card_top + 18),
        f"逐帧多泳道时间线（展示窗口 {w_start:.2f}s – {w_end:.2f}s，含白膜按键与三角色等待攒帧 / 处理 / 返回分解）",
        fill=(23, 43, 53),
        font=f_sec,
    )

    # Legend row
    legend_items = [
        ((166, 180, 187), "未选白膜帧"),
        ((0, 134, 109), "选中入模帧"),
        ((79, 70, 229), "玩家按键持续条"),
        ((245, 158, 11), "等待攒帧 (wait_ms)"),
        ((37, 99, 235), "Encoder 处理"),
        ((124, 58, 237), "DiT 处理"),
        ((5, 150, 105), "Decoder 处理"),
        ((234, 88, 12), "返回结果 (return_ms)"),
    ]
    lx = 58
    ly = card_top + 54
    for col, txt in legend_items:
        draw.rounded_rectangle((lx, ly + 3, lx + 16, ly + 19), radius=3, fill=col)
        draw.text((lx + 22, ly), txt, fill=(60, 76, 84), font=f_sml)
        lx += 22 + int(draw.textlength(txt, font=f_sml)) + 26

    plot_l, plot_r = 258, width - 64

    def x_of(t: float) -> float:
        return plot_l + ((t - w_start) / w_span) * (plot_r - plot_l)

    lanes = [
        (card_top + 128, "白膜供给", "约 24–30 FPS 源流"),
        (card_top + 206, "按键输入 (白膜供给)", "玩家实时按键/鼠标"),
        (card_top + 286, "选中并派发", "Room 选帧→派发"),
        (card_top + 398, "Encoder [1,8,8] px", "每组输出 [1,2,2] latent"),
        (card_top + 528, "DiT latent chunk", "黄色=等输入 / 攒组 / 排队"),
        (card_top + 658, "Lynn：1 latent/步", "每5步输出 [1,4,4,4,4] px"),
        (card_top + 772, "模型 WS → Room", "实际收到的帧批次"),
    ]

    # Vertical time grid
    num_ticks = 9
    for i in range(num_ticks + 1):
        t_val = w_start + (w_span * i) / num_ticks
        gx = x_of(t_val)
        draw.line(
            [(gx, card_top + 96), (gx, card_top + 806)], fill=(235, 241, 243), width=1
        )
        draw.text(
            (gx - 18, card_top + 816), f"{t_val:.2f}s", fill=(82, 101, 109), font=f_sml
        )

    for ly_pos, l_title, l_sub in lanes:
        draw.line([(plot_l, ly_pos), (plot_r, ly_pos)], fill=(220, 229, 232), width=2)
        draw.text((56, ly_pos - 14), l_title, fill=(23, 43, 53), font=f_lbl)
        draw.text((56, ly_pos + 10), l_sub, fill=(110, 126, 133), font=f_sml)

    # Skipped frame orange bands (same style as reference image)
    for sk in skips:
        if sk["end_s"] < w_start or sk["start_s"] > w_end:
            continue
        x0 = max(plot_l, x_of(sk["start_s"]))
        x1 = min(plot_r, x_of(sk["end_s"]))
        if x1 > x0:
            draw.rectangle(
                (x0, lanes[0][0] + 15, x1, lanes[0][0] + 49), fill=(255, 240, 216)
            )
            if x1 - x0 >= 60:
                sk_lbl = (
                    f"{sk['gap_seconds']:.2f}s / 跳过{sk['skipped']}帧"
                    if (x1 - x0) >= 105
                    else f"跳过{sk['skipped']}帧"
                )
                draw.text(
                    (x0 + 4, lanes[0][0] + 22),
                    sk_lbl,
                    fill=(193, 108, 21),
                    font=f_sml,
                )

    # Key hold capsules in lane 1
    for sp in key_spans:
        if sp["end_s"] < w_start or sp["start_s"] > w_end:
            continue
        x0 = max(plot_l, x_of(sp["start_s"]))
        x1 = min(plot_r, x_of(sp["end_s"]))
        bw = max(10.0, x1 - x0)
        ky = lanes[1][0] + ((sp.get("track", 0) % 2) * 24 - 11)
        draw.rounded_rectangle(
            (x0, ky - 10, x0 + bw, ky + 10),
            radius=5,
            fill=(224, 231, 255),
            outline=(79, 70, 229),
            width=2,
        )
        if bw >= 24:
            lbl = (
                f"{sp['label']} ({sp['duration_ms'] / 1000:.2f}s)"
                if bw >= 92
                else sp["label"]
            )
            draw.text((x0 + 6, ky - 8), lbl, fill=(49, 46, 129), font=f_sml)

    # Source ticks & selection lines
    for sf in sources:
        t = sf["capture_s"]
        if not (w_start <= t <= w_end):
            continue
        sx = x_of(t)
        s_idx = sf["source_frame"]
        is_sel = s_idx in sel_by_src
        is_foc = focus is not None and focus["source_frame"] == s_idx
        col = (0, 134, 109) if is_sel else (166, 180, 187)
        draw.line(
            [(sx, lanes[0][0] - 13), (sx, lanes[0][0] + 13)],
            fill=col,
            width=4 if is_foc else 2,
        )
        if is_sel:
            dx = x_of(sel_by_src[s_idx]["dispatch_s"])
            if plot_l <= dx <= plot_r:
                alpha = 225 if is_foc else 55
                draw.line(
                    [(sx, lanes[0][0] + 13), (dx, lanes[2][0] - 9)],
                    fill=(0, 134, 109, alpha),
                    width=3 if is_foc else 1,
                )

    # One bar per protocol group; the positions are duration illustrations.
    for role, lane, color in [
        ("encoder", lanes[3][0], (37, 99, 235)),
        ("dit", lanes[4][0], (124, 58, 237)),
        ("decoder", lanes[5][0], (5, 150, 105)),
    ]:
        for group in dataset.get("stage_groups", {}).get(role, []):
            if group["end_s"] < w_start or group["start_s"] > w_end:
                continue
            y = lane + (group["id"] % 3 - 1) * 24
            for begin, end, col in [
                ("start_s", "proc_start_s", (245, 158, 11)),
                ("proc_start_s", "ret_start_s", color),
                ("ret_start_s", "end_s", (234, 88, 12)),
            ]:
                x0 = max(plot_l, min(plot_r, x_of(group[begin])))
                x1 = max(plot_l, min(plot_r, x_of(group[end])))
                if x1 > x0:
                    draw.rectangle((x0, y - 3, x1, y + 3), fill=col)
            if w_span <= 3:
                draw.text(
                    (max(plot_l, x_of(group["proc_start_s"])), y - 20),
                    group["label"],
                    fill=(23, 43, 53),
                    font=f_sml,
                )
    for i, batch in enumerate(dataset.get("output_batches", [])):
        t = batch["received_s"]
        if w_start <= t <= w_end:
            x = x_of(t)
            y = lanes[6][0]
            draw.line([(x, y - 10), (x, y + 10)], fill=(193, 108, 21), width=3)
            draw.text(
                (x + 4, y + (14 if i % 2 else -30)),
                f"{batch['num_frames']}px",
                fill=(154, 77, 6),
                font=f_sml,
            )

    # Vertical cursor line on focus frame (matches "源帧 3072" in reference screenshot)
    if focus is not None and w_start <= focus["capture_s"] <= w_end:
        fx = x_of(focus["capture_s"])
        draw.line(
            [(fx, card_top + 88), (fx, card_top + 806)], fill=(23, 43, 53), width=2
        )
        draw.text(
            (min(plot_r - 130, fx + 6), card_top + 84),
            f"源帧 {focus['source_frame']} (输出#{focus['model_frame']})",
            fill=(23, 43, 53),
            font=f_lbl,
        )

    # Bottom section: Left = 3x3 Stage Summary Table, Right = Focus Frame Breakdown Card
    bot_top = 1006
    bot_bot = height - 28
    mid_x = 1230

    draw.rounded_rectangle(
        (32, bot_top, mid_x, bot_bot), radius=16, fill=(255, 255, 255)
    )
    draw.text(
        (56, bot_top + 18),
        "全程逐帧三角色耗时统计（中位数 / P95 / 均值，单位 ms）",
        fill=(23, 43, 53),
        font=f_sec,
    )
    headers = [
        "角色",
        "等待攒帧 (wait)",
        "处理计算 (process)",
        "返回结果 (return)",
        "角色小计",
    ]
    col_x = [56, 285, 535, 785, 1025]
    hy = bot_top + 64
    for cx, h_txt in zip(col_x, headers):
        draw.text((cx, hy), h_txt, fill=(82, 101, 109), font=f_lbl)
    draw.line([(56, hy + 30), (mid_x - 24, hy + 30)], fill=(224, 232, 235), width=2)

    for r_idx, (r_key, r_name) in enumerate(
        [
            ("encoder", "Encoder (VAE)"),
            ("dit", "DiT (Denoise)"),
            ("decoder", "Decoder (Lynn)"),
        ]
    ):
        ry = hy + 46 + r_idx * 56
        st = summary["stages"][r_key]
        draw.text((col_x[0], ry), r_name, fill=(23, 43, 53), font=f_lbl)
        for m_i, m_key in enumerate(
            ("wait_ms", "process_ms", "return_ms", "total_ms"), start=1
        ):
            ms = st[m_key]
            draw.text(
                (col_x[m_i], ry),
                f"{_fmt(ms['median'])} / {_fmt(ms['p95'])} / {_fmt(ms['mean'])} n={ms['count']}",
                fill=(23, 43, 53) if m_key == "total_ms" else (55, 71, 79),
                font=f_lbl if m_key == "total_ms" else f_txt,
            )
        draw.line([(56, ry + 36), (mid_x - 24, ry + 36)], fill=(238, 243, 245), width=1)

    e2e = summary["dispatch_to_raw_output_ms"]
    s2o = summary["source_to_raw_output_ms"]
    draw.text(
        (56, bot_bot - 46),
        f"端到端汇总：派发→收到原始输出 中位 {_fmt(e2e['median'])} ms (P95 {_fmt(e2e['p95'])} ms)  |  源采集→收到原始输出 中位 {_fmt(s2o['median'])} ms (P95 {_fmt(s2o['p95'])} ms)",
        fill=(0, 110, 90),
        font=f_txt,
    )

    # Right card: Focus frame exact breakdown
    draw.rounded_rectangle(
        (mid_x + 22, bot_top, width - 32, bot_bot), radius=16, fill=(255, 255, 255)
    )
    if focus is not None:
        keys_str = ", ".join(focus.get("held_keys") or []) or "无按键"
        draw.text(
            (mid_x + 46, bot_top + 18),
            f"基准线焦点帧详情：源帧 {focus['source_frame']} → 输出帧 #{focus['model_frame']}",
            fill=(23, 43, 53),
            font=f_sec,
        )
        draw.text(
            (mid_x + 46, bot_top + 54),
            f"采集 +{focus['capture_s']:.3f}s  |  当时白膜按键: [{keys_str}]  |  端到端: {focus['source_to_raw_output_ms']:.1f} ms",
            fill=(79, 70, 229),
            font=f_txt,
        )
        lines_txt = [
            f"Room 侧：采集→选中 {focus['source_to_selection_ms']:.1f} ms  |  选中→派发 {focus['selection_to_dispatch_ms']:.1f} ms  |  派发→收到输出 {focus['dispatch_to_raw_output_ms']:.1f} ms",
            f"Encoder：等待攒帧 {_fmt(focus['stages']['encoder']['wait_ms'])} ms  +  编码处理 {_fmt(focus['stages']['encoder']['process_ms'])} ms  +  传给 DiT {_fmt(focus['stages']['encoder']['return_ms'])} ms  =  {_fmt(focus['stages']['encoder']['total_ms'])} ms",
            f"DiT    ：等待攒帧 {_fmt(focus['stages']['dit']['wait_ms'])} ms  +  去噪处理 {_fmt(focus['stages']['dit']['process_ms'])} ms  +  传给 Decoder {_fmt(focus['stages']['dit']['return_ms'])} ms  =  {_fmt(focus['stages']['dit']['total_ms'])} ms",
            f"Decoder：等待队列 {_fmt(focus['stages']['decoder']['wait_ms'])} ms  +  解码处理 {_fmt(focus['stages']['decoder']['process_ms'])} ms  +  返回调用方 {_fmt(focus['stages']['decoder']['return_ms'])} ms  =  {_fmt(focus['stages']['decoder']['total_ms'])} ms",
        ]
        for i, lt in enumerate(lines_txt):
            draw.text(
                (mid_x + 46, bot_top + 96 + i * 42), lt, fill=(35, 52, 62), font=f_txt
            )

    img.save(output_png, format="PNG")
    return output_png


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="从 latency-breakdown-data.json 渲染交互式 HTML 报告与高清 PNG 概览图"
    )
    parser.add_argument(
        "--data-json", type=Path, required=True, help="输入数据 JSON 文件路径"
    )
    parser.add_argument("--output-html", type=Path, default=None, help="输出 HTML 路径")
    parser.add_argument("--output-png", type=Path, default=None, help="输出 PNG 路径")
    parser.add_argument(
        "--zoom-window",
        type=float,
        nargs=2,
        default=None,
        metavar=("START_S", "END_S"),
        help="PNG 局部展示窗口起止秒数（可选）",
    )
    args = parser.parse_args(argv)
    dataset = json.loads(args.data_json.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 2:
        raise ValueError(
            "Schema v2 is required; regenerate the dataset with the corrected replay runner"
        )
    out_dir = args.data_json.parent
    html_path = args.output_html or (out_dir / "index.html")
    png_path = args.output_png or (out_dir / "latency-breakdown-overview.png")
    render_html_report(dataset, html_path)
    render_png_overview(
        dataset,
        png_path,
        zoom_window_s=tuple(args.zoom_window) if args.zoom_window else None,  # type: ignore[arg-type]
    )
    print(
        json.dumps({"html": str(html_path), "png": str(png_path)}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Append conditioning panels to an exported, existing ARMS test dashboard.

Offline only: does not provision targets, deploy collectors or write Grafana.
The input is GET /api/dashboards/uid/sglang-whitebox-test-shanghai-copy.
"""

import argparse
import copy
import json
from pathlib import Path

UID = "sglang-whitebox-test-shanghai-copy"
MARKER = "Qwen prefix / conditioning [PR #687]"


def prepare(export, template):
    dashboard = copy.deepcopy(export["dashboard"])
    if dashboard["uid"] != UID:
        raise ValueError("Only the existing Shanghai test dashboard is supported")
    if any(p.get("title") == MARKER for p in dashboard["panels"]):
        raise ValueError("Conditioning section already exists; review before updating")
    first_id = max(p["id"] for p in dashboard["panels"]) + 1
    y = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in dashboard["panels"])
    panels = copy.deepcopy(template["panels"][6:])
    if len(panels) != 9 or panels[0]["type"] != "row":
        raise ValueError("Expected one conditioning row and eight charts")
    origin_y = panels[0]["gridPos"]["y"]
    panels[0]["title"] = MARKER
    selection = (
        'business_line_id="world-model-data",region="cn-shanghai",'
        'pool="test-shanghai-copy",service="sglang"'
    )
    for index, panel in enumerate(panels):
        panel["id"] = first_id + index
        panel["gridPos"]["y"] += y - origin_y
        if panel["type"] == "row":
            continue
        panel["datasource"] = {"type": "prometheus", "uid": "whitebox-sh-native"}
        panel["description"] = (
            "PR #662/#687；remote conditioning 或已接线的独立 Encoder。"
            "无数据表示尚未观测，不代表 miss；独立 Encoder 未上报独立 condition 队列时间。"
            "需要源服务、鉴权采集、relay 合同及 remote_write 全链路接通。"
            + panel["description"].split(" 复用既有")[0]
        )
        for target in panel["targets"]:
            target["expr"] = (
                target["expr"]
                .replace(
                    'cluster_id=~"$cluster_id",service_id=~"$service_id"', selection
                )
                .replace("job,instance", "instance")
            )
            target["legendFormat"] = target["legendFormat"].replace("{{job}} ", "")
    dashboard["panels"].extend(panels)
    # Keep the exported version: Grafana must reject a concurrent edit.
    return {
        "dashboard": dashboard,
        "folderUid": export["meta"]["folderUid"],
        "overwrite": False,
        "message": "PR #687：追加 Qwen prefix 监控；数据接入状态独立验收",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    template = json.loads(
        (root / "benchmark/whitebox/grafana-dashboard.json").read_text(encoding="utf-8")
    )
    result = prepare(json.loads(args.export.read_text(encoding="utf-8")), template)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

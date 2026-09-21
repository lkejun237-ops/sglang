# SPDX-License-Identifier: Apache-2.0
"""Build an offline-capable, synchronized native/TAEH3 evidence viewer."""

import argparse
import html
import json
import statistics
from pathlib import Path
from urllib.parse import quote


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", required=True, help="Directory containing exported run evidence"
    )
    args = p.parse_args()
    root = Path(args.root).resolve()
    rows, cases = [], []
    for path in sorted(root.rglob("sglang.json")):
        data = json.loads(path.read_text())
        backend = data.get("video_decoder_backend", path.parent.name)
        stage_rows = data.get("measured_metrics", [])

        def median_stage(name):
            values = [
                row["stages"][name] / 1000
                for row in stage_rows
                if name in row.get("stages", {})
            ]
            return f"{statistics.median(values):.3f}" if values else "—"

        if len(data.get("generation_seconds_all", [])) >= 3:
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(value))}</td>"
                    for value in (
                        str(path.parent.parent.relative_to(root)),
                        data["sp_size"],
                        backend
                        + (
                            f" ({data.get('taeh3_decode_precision', 'unspecified')})"
                            if backend == "taeh3"
                            else ""
                        ),
                        f"{data['generation_seconds']:.3f}",
                        median_stage("ZingH3Conditioning"),
                        median_stage("ZingH3Denoise"),
                        median_stage("ZingH3VideoDecode"),
                        median_stage("ZingH3AudioDecode"),
                        f"{240 / data['generation_seconds']:.2f}",
                        "≤10 秒" if data["generation_seconds"] <= 10 else ">10 秒",
                    )
                )
                + "</tr>"
            )
        if backend == "native":
            peer = path.parent.parent / "taeh3" / "sglang-0.mp4"
            native = path.parent / "sglang-0.mp4"
            if native.with_name("sglang-preview.mp4").is_file():
                native = native.with_name("sglang-preview.mp4")
            if peer.with_name("sglang-preview.mp4").is_file():
                peer = peer.with_name("sglang-preview.mp4")
            if native.is_file() and peer.is_file():
                metric = path.parent.parent / "comparison.json"
                quality = json.loads(metric.read_text()) if metric.is_file() else {}
                cases.append(
                    {
                        "name": str(path.parent.parent.relative_to(root)),
                        "native": quote(str(native.relative_to(root))),
                        "taeh3": quote(str(peer.relative_to(root))),
                        "metrics": {
                            key: quality[key]
                            for key in (
                                "latent_exact",
                                "audio_exact",
                                "psnr_global",
                                "ssim_mean",
                                "temporal_delta_mae_mean",
                                "lpips",
                            )
                            if key in quality
                        },
                    }
                )
    streams = []
    for path in sorted(root.rglob("stream-report.json")):
        report = json.loads(path.read_text())
        for run in report["runs"]:
            streams.append(dict(sp=report["sp_size"], **run))
    websocket_runs = []
    for path in sorted(root.rglob("websocket-report.json")):
        report = json.loads(path.read_text())
        websocket_runs.append({"run": str(path.parent.relative_to(root)), **report})
    chart_paths = (
        [root / "streaming-comparison.png"]
        if (root / "streaming-comparison.png").is_file()
        else sorted(root.rglob("delivery.png"))
    )
    charts = "".join(
        f'<img alt="实际接收进度与播放缓冲" style="width:100%;max-width:1000px" src="{quote(str(path.relative_to(root)))}">'
        for path in chart_paths
    )
    trace_links = "".join(
        f'<li><a href="{quote(str(path.relative_to(root)))}">{html.escape(str(path.relative_to(root)))}</a></li>'
        for path in sorted(root.rglob("*-gpu-and-scopes.trace.json.gz"))
    )
    payload = json.dumps(
        {"cases": cases, "streams": streams, "websocket": websocket_runs},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TAEH3 验收证据</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#111820;color:#eef2f6;margin:auto;padding:32px;max-width:1400px}h1{font-size:30px}h2{margin-top:32px}p{color:#bfccd8}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px;text-align:left;border-bottom:1px solid #334250}th{color:#a8c5db}button,select{padding:9px 14px;background:#223345;color:#fff;border:1px solid #536779;border-radius:5px;margin:5px}a{color:#86cfff}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}video{width:100%;background:#000}pre{white-space:pre-wrap;background:#18232e;padding:16px;overflow:auto}.status{border-left:4px solid #e8b65d;padding:12px;background:#28251e} @media(max-width:800px){.pair{grid-template-columns:1fr}body{padding:15px}table{font-size:12px}}
</style><h1>TAEH3 / native 同 latent 对照</h1>
<div class="status">验收进行中，人工画质尚未确认。decoder-only 吞吐不代表完整请求或持续流式达标。</div>
<p>新请求：832×480 · 24 FPS · 四步 [1000,750,500,250] · guidance 1 · seed 42。native 默认路径保留；TAEH3 为有损替代。</p>
<h2>Warm 完整请求（秒）</h2><p>至少 3 次独立 unprofiled 请求的中位数；profile 单独执行。API 为 video_audio 请求，包含媒体导出。表内 FPS 是生成吞吐。e2e-multi-r2 与 e2e-sp8-r5 为更新后的 main 运行；e2e-sp1-r1 是独立保留的旧 main 数据。</p>
<table><thead><tr><th>运行</th><th>SP</th><th>decoder</th><th>API</th><th>准备</th><th>DiT</th><th>视频 decode</th><th>音频 decode</th><th>FPS</th><th>10 秒请求</th></tr></thead><tbody>ROWS</tbody></table>
<h2>同步画质对比</h2><select id="case"></select><button id="play">同步播放 / 暂停</button><button id="back">上一帧</button><button id="next">下一帧</button><input id="seek" aria-label="同步播放位置" type="range" min="0" max="10" step="0.0416666667" value="0"><span id="time"></span>
<div class="pair"><section><h3>native <a id="native-link">下载</a></h3><video id="native" playsinline preload="auto"></video></section><section><h3>TAEH3 <a id="taeh3-link">下载</a></h3><video id="taeh3" muted playsinline preload="auto"></video></section></div>
<p>播放、拖动及逐帧按钮同步两侧。请分别检查人物身份、文字/细纹、快速运动、镜头切换、闪烁和尾帧。PSNR/SSIM 不替代主观验收。</p><pre id="metrics"></pre>
<h2>持续流式</h2><p>video-only 请求，实际 scheduler 与 RGB/本地 IPC；不包含 WebSocket、网络和浏览器。cold_10s 是模型已加载后的进程首个请求，磁盘编译缓存可复用，不同作业的冷请求不能直接比较。小于 1440 帧的运行不作为 60 秒验收。</p><table><thead><tr><th>SP</th><th>场景</th><th>帧数</th><th>首帧 (s)</th><th>总耗时 (s)</th><th>FPS</th><th>chunk p50/p95 (s)</th><th>卡顿 (s)</th></tr></thead><tbody id="stream-rows"></tbody></table><details><summary>持续流式原始指标</summary><pre id="streams"></pre></details>
<h2>WebSocket 服务链路</h2><p>video-only 请求，真实 FastAPI 与本机 WebSocket，原尺寸 raw RGB，不包含 WAN 或浏览器渲染。取消、重连和 prompt 更新是功能检查，不能代替持续吞吐验收。</p>CHARTS<table><thead><tr><th>运行</th><th>场景</th><th>帧数</th><th>首帧 (s)</th><th>总耗时 (s)</th><th>FPS</th><th>卡顿 (s)</th></tr></thead><tbody id="ws-rows"></tbody></table><details><summary>WebSocket 原始指标</summary><pre id="websocket"></pre></details>
<h2>Trace 查看</h2><p>以下为 Perfetto 实际打开验证过的 GPU / Zing 范围精简视图；保留事件的时间与参数不变，显示分道不代表新增 CUDA stream。原始文件另行保留并用于数值归因；原始 Python 栈与重叠区间存在 Perfetto 导入告警。</p><ul>TRACES</ul>
<script>const data=PAYLOAD;const a=document.getElementById('native'),b=document.getElementById('taeh3'),sel=document.getElementById('case');
for(const [i,c] of data.cases.entries()){let o=document.createElement('option');o.value=i;o.textContent=c.name;sel.appendChild(o)}
let wantsPlay=false;const seek=document.getElementById('seek'),clock=document.getElementById('time');
function pauseBoth(){a.pause();b.pause()}
function resume(){if(!wantsPlay||a.seeking||b.seeking||a.readyState<3||b.readyState<3)return;Promise.all([a.play(),b.play()]).catch(()=>{wantsPlay=false;pauseBoth();clock.textContent='播放失败，请重试'})}
function load(){const c=data.cases[Number(sel.value)];if(!c)return;wantsPlay=false;pauseBoth();a.src=c.native;b.src=c.taeh3;seek.value=0;clock.textContent='载入视频';document.getElementById('native-link').href=c.native;document.getElementById('taeh3-link').href=c.taeh3;document.getElementById('metrics').textContent=JSON.stringify(c.metrics,null,2)}
sel.onchange=load;load();
for(const v of [a,b]){v.onwaiting=()=>{pauseBoth();if(wantsPlay)clock.textContent='两侧同步缓冲中'};v.oncanplay=resume;v.onseeking=pauseBoth;v.onseeked=resume;v.onended=()=>{wantsPlay=false;pauseBoth()}}
a.onloadedmetadata=()=>{seek.max=a.duration};a.ontimeupdate=()=>{seek.value=a.currentTime;clock.textContent=a.currentTime.toFixed(3)+' s';if(wantsPlay&&!a.seeking&&!b.seeking&&Math.abs(a.currentTime-b.currentTime)>0.12){pauseBoth();b.currentTime=a.currentTime}};
function seekBoth(t){wantsPlay=false;pauseBoth();a.currentTime=b.currentTime=Math.max(0,Math.min(a.duration||0,t));seek.value=a.currentTime;clock.textContent=a.currentTime.toFixed(3)+' s'}
document.getElementById('play').onclick=()=>{wantsPlay=!wantsPlay;if(wantsPlay){if(a.ended)seekBoth(0);wantsPlay=true;resume()}else pauseBoth()};
seek.oninput=()=>seekBoth(Number(seek.value));document.getElementById('back').onclick=()=>seekBoth(a.currentTime-1/24);document.getElementById('next').onclick=()=>seekBoth(a.currentTime+1/24);
function tableRow(id,values){const r=document.createElement('tr');for(const v of values){const c=document.createElement('td');c.textContent=v;r.appendChild(c)}document.getElementById(id).appendChild(r)}const fmt=x=>Number.isFinite(x)?x.toFixed(3):'—';
for(const r of data.streams)tableRow('stream-rows',[r.sp,r.name,r.frames,fmt(r.first_frame_seconds),fmt(r.total_seconds),fmt(r.fps_including_startup),fmt(r.chunk_p50_seconds)+' / '+fmt(r.chunk_p95_seconds),fmt(r.playback_stall_seconds)]);
for(const run of data.websocket)for(const r of run.runs)tableRow('ws-rows',[run.run,r.name,r.frames,fmt(r.first_frame_seconds),fmt(r.total_seconds),fmt(r.fps),fmt(r.playback_stall_seconds)]);
document.getElementById('streams').textContent=data.streams.length?JSON.stringify(data.streams,null,2):'尚无已完成的持续流式记录。';
document.getElementById('websocket').textContent=data.websocket.length?JSON.stringify(data.websocket,null,2):'尚无已完成的 WebSocket 记录。';
</script></html>"""
    page = (
        page.replace(
            "ROWS", "".join(rows) or '<tr><td colspan="10">测量尚未完成</td></tr>'
        )
        .replace("CHARTS", charts)
        .replace("TRACES", trace_links or "<li>尚无已验证视图</li>")
        .replace("PAYLOAD", payload)
    )
    (root / "index.html").write_text(page)
    print(root / "index.html")


if __name__ == "__main__":
    main()

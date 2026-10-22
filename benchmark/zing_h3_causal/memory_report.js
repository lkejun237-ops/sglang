const MemoryReport = (() => {
  "use strict";
  function render(bundle) {
    validate(bundle);
    const root = document.getElementById("memory-view");
    const data = bundle.replay,
      G = 2 ** 30;
    const esc = (s) =>
      String(s ?? "").replace(
        /[&<>"']/g,
        (c) =>
          ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
          })[c],
      );
    const fmt = (n) => (Number.isFinite(n) ? (n / G).toFixed(3) : "未测"),
      colors = ["#197a62", "#a66b24", "#596cc1"];
    let summary = data.errors?.length
      ? '<p class="replay-warning">存在未完成或失败的采集：' +
        data.errors.map(esc).join("；") +
        "</p>"
      : "";
    if (bundle.hardware) {
      const hw = bundle.hardware;
      summary +=
        '<p class="tiny muted">实际设备：' +
        esc(hw.cuda.gpu) +
        "；CUDA Driver / PyTorch capability " +
        esc(hw.cuda.cc) +
        "；NVML capability " +
        esc(
          [
            ...new Set(
              hw.nvml_csv
                .trim()
                .split("\n")
                .map((v) => v.split(",")[4].trim()),
            ),
          ].join("/"),
        ) +
        "。完整原始接口读数保留在导出 JSON。</p>";
    }
    if (bundle.counters) {
      summary +=
        "<h2>先看整卡占用</h2><p>就绪基线与诊断轮观测峰值。allocated / reserved 是 allocator 计数；NVML 是整卡采样，不将不同进程的峰值相加。</p>";
      for (const [mode, cell] of Object.entries(bundle.counters)) {
        summary +=
          "<h3>" +
          esc(mode) +
          '</h3><div class="table-wrap"><table><tr><th>进程</th><th>就绪 allocated / GiB</th><th>peak allocated / GiB</th><th>就绪 reserved / GiB</th><th>peak reserved / GiB</th></tr>' +
          (cell.processes ?? [])
            .map(
              (v) =>
                "<tr><td>" +
                esc(v.role) +
                "</td>" +
                [
                  "ready_allocated",
                  "peak_allocated",
                  "ready_reserved",
                  "peak_reserved",
                ]
                  .map((k) => "<td>" + fmt(v[k]) + "</td>")
                  .join("") +
                "</tr>",
            )
            .join("") +
          "</table></div>";
        summary +=
          '<div class="table-wrap"><table><tr><th>物理卡</th><th>就绪 NVML / GiB</th><th>采样峰值 / GiB</th><th>采样增量 / GiB</th><th>覆盖</th></tr>' +
          (cell.gpus ?? [])
            .map(
              (v) =>
                "<tr><td>" +
                esc(v.role) +
                "</td>" +
                ["ready_sample", "peak_sample", "increment_sample"]
                  .map((k) => "<td>" + fmt(v[k]) + "</td>")
                  .join("") +
                "<td>" +
                (v.complete ? "诊断轮完整" : "部分记录") +
                "</td></tr>",
            )
            .join("") +
          "</table></div>";
      }
    }
    if (bundle.counter_series?.length) {
      const palette = ["#215ecb", "#188573", "#c6602b", "#9b52bc", "#677d19"];
      summary += "<details><summary>整卡与 allocator 采样曲线 · GiB</summary>";
      for (const group of bundle.counter_series) {
        const points = group.series.flatMap((v) => v.points),
          lo = Math.min(...points.map((v) => v[0])),
          hi = Math.max(...points.map((v) => v[0]));
        const top = (Math.max(1, ...points.map((v) => v[1])) / G) * 1.06,
          x = (t) => 54 + (920 * (t - lo)) / (hi - lo || 1),
          y = (v) => 208 - (180 * v) / G / top;
        let svg =
          '<svg viewBox="0 0 1000 240" role="img" aria-label="allocated reserved NVML 采样曲线">';
        for (let i = 0; i <= 4; i++) {
          const yy = 208 - (180 * i) / 4,
            xx = 54 + (920 * i) / 4;
          svg +=
            '<line x1="54" x2="974" y1="' +
            yy +
            '" y2="' +
            yy +
            '" stroke="#dce5df"/><text x="0" y="' +
            (yy + 4) +
            '">' +
            ((top * i) / 4).toFixed(1) +
            '</text><text x="' +
            xx +
            '" y="230">' +
            (((hi - lo) * i) / 4).toFixed(0) +
            "s</text>";
        }
        group.series.forEach((v, i) => {
          svg +=
            '<polyline fill="none" stroke="' +
            palette[i % palette.length] +
            '" stroke-width="1.5" points="' +
            v.points
              .map((p) => x(p[0]).toFixed(2) + "," + y(p[1]).toFixed(2))
              .join(" ") +
            '"/>';
        });
        summary +=
          "<h3>" +
          esc(group.mode + " / " + group.label) +
          "</h3>" +
          svg +
          "</svg><p>" +
          group.series
            .map(
              (v, i) =>
                '<span style="margin-right:16px;color:' +
                palette[i % palette.length] +
                '">━ ' +
                esc(v.name) +
                "</span>",
            )
            .join("") +
          "</p>";
      }
      summary +=
        '<p class="tiny muted">横轴为当前记录窗口的相对秒数。曲线保留采样极值；allocated/reserved 的区间精确 peak 另见检查点表，NVML 短尖峰可能漏采。</p></details>';
    }
    if (bundle.independent_timing) {
      const rows = bundle.independent_timing;
      summary +=
        '<h2>独立计时 · 显存采集已关闭</h2><div class="table-wrap"><table><tr><th>模式</th><th>用例</th><th>有效次数</th><th>TTFF / s</th><th>E2E / s</th><th>TTFF 后 FPS</th></tr>' +
        rows
          .map(
            (v) =>
              "<tr><td>" +
              esc(v.cell) +
              "</td><td>" +
              esc(v.case) +
              "</td><td>" +
              v.count +
              (v.partial ? "（部分）" : "") +
              "</td>" +
              ["ttff_seconds", "e2e_seconds", "post_ttff_fps"]
                .map(
                  (k) =>
                    '<td title="min–max: ' +
                    v.client[k].min.toFixed(3) +
                    "–" +
                    v.client[k].max.toFixed(3) +
                    '">' +
                    v.client[k].median.toFixed(3) +
                    "</td>",
                )
                .join("") +
              "</tr>",
          )
          .join("") +
        "</table></div>";
      const wanted = [
          ["Qwen", "Qwen"],
          ["Encoder / source_vae", "Source Encoder"],
          ["Encoder / reference_vae", "Reference Encoder"],
          ["DiT", "DiT"],
          ["Decoder", "Decoder"],
        ],
        palette = ["#2563eb", "#059669", "#d97706", "#9333ea", "#dc2626"];
      const cases = [...new Set(rows.map((v) => v.case))];
      summary +=
        '<h2>全链路：时间花在哪里？</h2><label>用例 <select id="memory-stage-case">' +
        cases.map((v) => "<option>" + esc(v) + "</option>").join("") +
        "</select></label>";
      cases.forEach((c, i) => {
        summary +=
          '<div class="memory-stage-panel" data-case="' +
          esc(c) +
          '" ' +
          (i ? "hidden" : "") +
          ' style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:20px">';
        rows
          .filter((v) => v.case === c)
          .forEach((v) => {
            if (!wanted.every(([k]) => v.stages[k])) return;
            const values = wanted.map(([k]) => v.stages[k].median),
              total = values.reduce((a, b) => a + b, 0);
            if (!total) return;
            let pos = 0;
            const stops = values.map((n, i) => {
              const before = pos;
              pos += (100 * n) / total;
              return palette[i] + " " + before + "% " + pos + "%";
            });
            summary +=
              "<article><h3>" +
              esc(v.cell) +
              " · " +
              v.count +
              ' 次</h3><div role="img" aria-label="阶段 CUDA 调用耗时构成" style="width:220px;height:220px;border-radius:50%;background:conic-gradient(' +
              stops.join(",") +
              ')"></div><ul>' +
              wanted
                .map(
                  ([, name], i) =>
                    "<li>" +
                    esc(name) +
                    "：" +
                    ((100 * values[i]) / total).toFixed(1) +
                    "% · " +
                    (values[i] / 1000).toFixed(3) +
                    "s</li>",
                )
                .join("") +
              "</ul></article>";
          });
        summary += "</div>";
      });
      summary +=
        '<p class="tiny muted">按所列阶段 CUDA 耗时中位数归一化，Qwen 不重复计入 Conditioning RPC 等待；阶段可能重叠，比例不是 E2E 时间分摊。表中悬停查看 min–max。</p>';
    }
    root.innerHTML =
      summary +
      `<h2>运行时新增：哪些分配同时活着？</h2><p>扣除所选轮次开始前已有的分配，按 alloc / free 历史重放，查看运行时新增显存及其峰值构成。</p><div class="replay-controls"><label>配置 / 轮次 / 用例 <select id="replay-case"></select></label><label>进程 <select id="replay-process"></select></label><button id="replay-peak">定位新增峰值</button></div><div id="replay-coverage"></div><div id="replay-metrics" class="replay-metrics"></div><div class="replay-legend"><span style="color:${colors[0]}">━ 新增且未请求释放</span><span style="color:${colors[1]}">━ 新增占用（含待 GPU 释放）</span><span style="color:${colors[2]}">┄ 总请求字节净增</span></div><svg id="replay-chart" viewBox="0 0 1120 350" role="img" aria-label="按分配历史重放的运行时新增显存曲线"></svg><div class="replay-stage-caption">当前进程的阶段调用轨道 · Host 时间，与上方显存共用横轴</div><svg id="replay-stages" role="img" aria-label="Encoder、DiT、Decoder 阶段调用轨道"></svg><p id="replay-stage-readout" aria-live="polite"></p><label class="replay-scrub">查看时间 <input id="replay-time" type="range" min="0" step="1" value="0"></label><p id="replay-readout" aria-live="polite"></p><h3 id="replay-stack-title"></h3><div class="replay-table"><table><thead><tr><th>分配调用栈</th><th>同时存活 / GiB</th><th>分配数</th><th>占新增峰值</th></tr></thead><tbody id="replay-stacks"></tbody></table></div><details><summary>重放口径与一致性检查</summary><p>基线取所选轮次开始时；可分别查看预热、capture 和正式轮。曲线按历史事件的 size（张量请求字节）计量；allocator 取整和块内碎片不包含在曲线中，下面以 allocated 计数器作对照。“新增”按分配身份识别：请求开始后 alloc、尚未 free_requested 的分配。相同地址释放后再次分配属于新的分配。总请求字节净增还会扣除请求前分配在运行中释放的部分，因此两条曲线可能不同。</p><p>free_requested 后、free_completed 前的分配仍占用设备空间，单列到“含待 GPU 释放”曲线。阶段轨道使用同一 CLOCK_MONOTONIC 时钟下的 Host 调用起止；CUDA event 只有耗时，没有在此轴上推算 GPU kernel 起止。轨道空白表示未落在这些已记录的 Host 调用内。曲线不包含 allocator 空闲缓存及 PyTorch 之外的占用；整卡占用见 allocated / reserved / NVML 记录，不能用请求字节曲线代替。</p><p>从最终快照反推区间开始时的存活分配，再正向重放验证闭合。显示曲线按时间桶保留各指标极值，峰值及下表按原始事件计算。</p><p id="replay-validation"></p></details>`;
    const css = document.createElement("style");
    css.textContent = `#memory-view{border-top:2px solid #287866;padding-top:20px;margin:28px 0 42px}#memory-view h2{margin-bottom:8px}.replay-controls,.replay-legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap}.replay-controls select,.replay-controls button{font:inherit;padding:6px;border:1px solid #baccc4;border-radius:5px;background:white}.replay-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.replay-metrics>div{background:#f0f5f2;padding:12px;border-radius:7px}.replay-metrics strong{display:block;font-size:27px;color:#176751}.replay-metrics small{display:block;color:#52655e}.replay-legend{font-size:14px}#replay-chart{width:100%;display:block;touch-action:pan-y}#replay-chart text{font:12px system-ui;fill:#52655e}.replay-stage-caption{font-size:13px;color:#52655e;margin-top:8px}#replay-stages{width:100%;display:block}#replay-stages text{font:11px system-ui;fill:#52655e}#replay-stage-readout{font-size:14px;min-height:24px;margin:5px 0 12px}#replay-stages rect[data-stage].active{stroke:#172e24;stroke-width:2}.replay-scrub{display:flex;gap:12px}.replay-scrub input{flex:1;min-width:0}.replay-table{overflow:auto}.replay-table td{vertical-align:top}.replay-table td:first-child{min-width:220px}.replay-table td:not(:first-child){font-variant-numeric:tabular-nums;white-space:nowrap}.replay-table pre{overflow-wrap:anywhere}.replay-warning{padding:10px 14px;background:#fff3d9;border-left:3px solid #a66b24}.replay-ok{color:#287866}.replay-controls label{display:flex;gap:6px;align-items:center}#replay-readout{font-variant-numeric:tabular-nums;background:#f6f8f7;padding:8px 12px}@media(max-width:650px){.replay-metrics{grid-template-columns:repeat(2,1fr)}.replay-metrics strong{font-size:22px}.replay-controls{gap:10px}.replay-controls label{width:100%}.replay-controls select{flex:1;min-width:0}}`;
    const previous = document.getElementById("memory-report-style");
    if (previous) previous.remove();
    css.id = "memory-report-style";
    document.head.appendChild(css);
    const stageSelect = root.querySelector("#memory-stage-case");
    if (stageSelect)
      stageSelect.addEventListener("change", () =>
        root
          .querySelectorAll(".memory-stage-panel")
          .forEach((p) => (p.hidden = p.dataset.case !== stageSelect.value)),
      );
    if (bundle.comparison) {
      const detail = document.createElement("details");
      const title = document.createElement("summary");
      title.textContent = "运行配置、Graph 路由与来源";
      const pre = document.createElement("pre");
      pre.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere";
      pre.textContent = JSON.stringify(
        { comparison: bundle.comparison, sources: bundle.sources },
        null,
        2,
      );
      detail.append(title, pre);
      root.append(detail);
    }
    const caseSelect = root.querySelector("#replay-case"),
      processSelect = root.querySelector("#replay-process"),
      slider = root.querySelector("#replay-time"),
      svg = root.querySelector("#replay-chart");
    let profile, coords;
    const labels = {
      server: "DiT rank",
      qwen: "Qwen",
      encoder: "Source Encoder",
    };
    const preferred = ["eager / memory / 5s", "graph / memory / 5s"];
    const cases = [...new Set(data.profiles.map((p) => p.case))].sort(
      (a, b) =>
        (preferred.includes(a) ? preferred.indexOf(a) : -1) -
          (preferred.includes(b) ? preferred.indexOf(b) : -1) ||
        a.localeCompare(b),
    );
    cases.sort(
      (a, b) =>
        (preferred.includes(b) ? 1 : 0) - (preferred.includes(a) ? 1 : 0),
    );
    caseSelect.innerHTML = cases
      .map((c) => `<option>${esc(c)}</option>`)
      .join("");
    function chooseCase() {
      const list = data.profiles
        .filter((p) => p.case === caseSelect.value)
        .sort(
          (a, b) =>
            (a.role === "server" ? -1 : 1) - (b.role === "server" ? -1 : 1) ||
            a.role.localeCompare(b.role) ||
            a.device - b.device,
        );
      processSelect.innerHTML = list
        .map(
          (p) =>
            `<option value="${p.pid}:${p.device}">${esc(labels[p.role] ?? p.role)}${p.role === "server" ? " " + p.device : ""}</option>`,
        )
        .join("");
      render();
    }
    function render() {
      profile = data.profiles.find(
        (p) =>
          p.case === caseSelect.value &&
          p.pid + ":" + p.device === processSelect.value,
      );
      const p = profile;
      root.querySelector("#replay-coverage").innerHTML = p.partial
        ? `<p class="replay-warning">分配历史已截断：仅重放请求第 ${p.recorded_start_seconds.toFixed(2)} s 起保留的区间。区间起点前已存活的分配作为未知基线；这里的新增峰值仅代表保留区间，不能代表完整请求。</p>`
        : `<p class="replay-ok">完整分配历史 · 已与最终快照闭合</p>`;
      const metrics = [
        ["新增存活峰值", p.new_peak_bytes, "GiB · alloc → free_requested"],
        [
          "含待 GPU 释放峰值",
          p.new_active_peak_bytes,
          "GiB · alloc → free_completed",
        ],
        [
          p.partial ? "区间起点前已存活" : "请求前已存活",
          p.baseline_bytes,
          "GiB · 从新增视图扣除",
        ],
        ["总请求字节净增峰值", p.net_peak_bytes, "GiB · 相对上述基线"],
      ];
      root.querySelector("#replay-metrics").innerHTML = metrics
        .map(
          ([a, b, c]) =>
            `<div><small>${a}</small><strong>${fmt(b)}</strong><small>${c}</small></div>`,
        )
        .join("");
      const pts = p.points,
        W = 1120,
        H = 350,
        L = 58,
        R = 20,
        T = 18,
        B = 43,
        x0 = pts[0][0],
        x1 = pts[pts.length - 1][0],
        y0 = Math.min(0, ...pts.map((v) => v[3] / G)),
        y1 =
          Math.max(...pts.map((v) => Math.max(v[1] + v[2], v[3]) / G)) * 1.08 ||
          1;
      const x = (t) => L + ((t - x0) / (x1 - x0 || 1)) * (W - L - R),
        y = (v) => T + ((y1 - v) / (y1 - y0 || 1)) * (H - T - B);
      coords = { x, y, x0, x1 };
      let content = "";
      for (let i = 0; i <= 4; i++) {
        const v = y0 + ((y1 - y0) * i) / 4,
          yy = y(v);
        content += `<line x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}" stroke="#dce5df"/><text x="${L - 8}" y="${yy + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
        const t = x0 + ((x1 - x0) * i) / 4;
        content += `<text x="${x(t)}" y="${H - 17}" text-anchor="middle">${t.toFixed(1)} s</text>`;
      }
      content += '<text x="8" y="13">GiB</text>';
      const series = [
        (v) => v[1] / G,
        (v) => (v[1] + v[2]) / G,
        (v) => v[3] / G,
      ];
      series.forEach((get, i) => {
        content += `<polyline fill="none" stroke="${colors[i]}" stroke-width="${i === 0 ? 1.8 : 1.2}" ${i === 2 ? 'stroke-dasharray="5 4"' : ""} points="${pts.map((v) => `${x(v[0]).toFixed(2)},${y(get(v)).toFixed(2)}`).join(" ")}"/>`;
      });
      content += `<line x1="${x(p.peak_seconds)}" x2="${x(p.peak_seconds)}" y1="${T}" y2="${H - B}" stroke="#197a62" stroke-dasharray="2 3"/><line id="replay-cursor" y1="${T}" y2="${H - B}" stroke="#263d34"/>`;
      svg.innerHTML = content;
      renderStages();
      root.querySelector("#replay-stack-title").textContent =
        `新增存活峰值 ${p.peak_seconds.toFixed(3)} s：同时存活的分配构成`;
      root.querySelector("#replay-stacks").innerHTML =
        p.peak_stacks
          .map((g) => {
            const f = g.frames[0];
            const title = f
              ? `${f.name} · ${f.filename.split("/").pop()}:${f.line}`
              : "无调用栈";
            return `<tr><td><details><summary>${esc(title)}</summary><pre>${g.frames.map((v) => esc(`${v.filename}:${v.line} · ${v.name}`)).join("\n")}</pre></details></td><td>${fmt(g.bytes)}</td><td>${g.count}</td><td>${((100 * g.bytes) / (p.new_peak_bytes || 1)).toFixed(1)}%</td></tr>`;
          })
          .join("") +
        (p.other_peak_stack_bytes
          ? `<tr><td>其余调用栈</td><td>${fmt(p.other_peak_stack_bytes)}</td><td>—</td><td>${((100 * p.other_peak_stack_bytes) / (p.new_peak_bytes || 1)).toFixed(1)}%</td></tr>`
          : "");
      root.querySelector("#replay-validation").textContent =
        `重放 ${p.trace_events.toLocaleString()} 个事件，最终地址/请求字节/释放状态完全匹配快照。${p.partial ? "保留区间" : "完整历史"}重放得到 请求字节峰值 ${fmt(p.reconstructed_peak_bytes)} GiB；请求期间记录到的最大 peak 计数 ${fmt(p.observed_counter_max_bytes)} GiB。原检查点 peak 读数 ${fmt(p.checkpoint_counter_peak_bytes)} GiB。${p.counter_fell_count ? ` 请求期间观察到 peak 计数 ${p.counter_fell_count} 次回落，检查点读数不能作为完整请求峰值。` : ""}`;
      slider.max = pts.length - 1;
      goPeak();
    }
    function inspect(i) {
      i = Math.max(0, Math.min(profile.points.length - 1, i));
      slider.value = i;
      const p = profile.points[i];
      root.querySelector("#replay-readout").textContent =
        `${p[0].toFixed(4)} s · 新增存活 ${fmt(p[1])} GiB · 新增待 GPU 释放 ${fmt(p[2])} GiB · 总请求字节净增 ${fmt(p[3])} GiB`;
      const line = svg.querySelector("#replay-cursor");
      line.setAttribute("x1", coords.x(p[0]));
      line.setAttribute("x2", coords.x(p[0]));
      inspectStages(p[0]);
    }
    function goPeak() {
      const pts = profile.points;
      let best = 0;
      for (let i = 1; i < pts.length; i++)
        if (
          Math.abs(pts[i][0] - profile.peak_seconds) <
          Math.abs(pts[best][0] - profile.peak_seconds)
        )
          best = i;
      inspect(best);
    }

    const stageColors = {
      Encoder: "#9b5d9a",
      DiT: "#287ab3",
      Decoder: "#cf7c24",
      Conditioning: "#888460",
      Qwen: "#62764b",
      Capture: "#b44642",
      Replay: "#4e8c91",
      Warmup: "#a48b4d",
    };
    function stageDetails(span) {
      const extra = [];
      if (span.component) extra.push(span.component);
      if (span.input_latent_range)
        extra.push("latent [" + span.input_latent_range.join(", ") + ")");
      if (span.media_id) extra.push(span.media_id);
      if (span.chunk_id !== undefined) extra.push("chunk " + span.chunk_id);
      return extra.length ? " · " + extra.join(" · ") : "";
    }
    function renderStages() {
      const rail = root.querySelector("#replay-stages"),
        spans = profile.stage_timeline || [],
        visible = spans.filter(
          (v) => v.end >= coords.x0 && v.start <= coords.x1,
        ),
        names = [
          "Conditioning",
          "Qwen",
          "Encoder",
          "DiT",
          "Decoder",
          "Capture",
          "Replay",
          "Warmup",
        ].filter((n) => visible.some((v) => v.stage === n)),
        height = names.length * 27 + 14;
      rail.setAttribute("viewBox", `0 0 1120 ${height}`);
      let content = "";
      names.forEach((name, row) => {
        const yy = 7 + row * 27;
        content += `<text x="52" y="${yy + 13}" text-anchor="end">${name === "Conditioning" ? "条件/RPC" : name}</text><line x1="58" x2="1100" y1="${yy + 8}" y2="${yy + 8}" stroke="#e4eae7"/>`;
        spans.forEach((v, i) => {
          if (v.stage !== name || v.end < coords.x0 || v.start > coords.x1)
            return;
          const left = coords.x(Math.max(v.start, coords.x0)),
            right = coords.x(Math.min(v.end, coords.x1));
          content += `<rect data-stage="${i}" x="${left}" y="${yy}" width="${Math.max(0.8, right - left)}" height="16" rx="2" fill="${stageColors[name]}"><title>${esc(name + stageDetails(v) + " · " + v.start.toFixed(4) + "–" + v.end.toFixed(4) + " s · Host")}</title></rect>`;
        });
      });
      content += `<line id="replay-stage-cursor" y1="0" y2="${height}" stroke="#263d34"/>`;
      rail.innerHTML = content;
    }
    function inspectStages(t) {
      const spans = profile.stage_timeline || [],
        active = spans
          .map((v, i) => ({ v, i }))
          .filter(({ v }) => v.start <= t && t < v.end);
      const labels = active.map(({ v }) => v.stage + stageDetails(v));
      const calls = (profile.dit_calls || []).filter(
        (v) => v.start <= t && t < v.end,
      );
      if (calls.length)
        labels.push(
          ...calls.map(
            (v) =>
              "DiT/" +
              v.phase +
              (v.step !== null && v.step !== undefined
                ? " step " + v.step
                : "") +
              (v.chunk_id !== undefined ? " · chunk " + v.chunk_id : ""),
          ),
        );
      root.querySelector("#replay-stage-readout").textContent =
        `${t.toFixed(4)} s · 当前阶段（Host）：` +
        (labels.length ? labels.join("；") : "记录间隙");
      const rail = root.querySelector("#replay-stages");
      rail.querySelectorAll("[data-stage]").forEach((el) =>
        el.classList.toggle(
          "active",
          active.some((a) => a.i === Number(el.getAttribute("data-stage"))),
        ),
      );
      const cursor = rail.querySelector("#replay-stage-cursor");
      if (cursor) {
        cursor.setAttribute("x1", coords.x(t));
        cursor.setAttribute("x2", coords.x(t));
      }
    }
    root
      .querySelector("#replay-stages")
      .addEventListener("pointerdown", (e) => {
        const rect = e.target.closest("[data-stage]");
        let time;
        if (rect) {
          const v =
            profile.stage_timeline[Number(rect.getAttribute("data-stage"))];
          time =
            (Math.max(v.start, coords.x0) + Math.min(v.end, coords.x1)) / 2;
        } else {
          const box = e.currentTarget.getBoundingClientRect();
          time =
            coords.x0 +
            ((((e.clientX - box.left) / box.width) * 1120 - 58) /
              (1120 - 58 - 20)) *
              (coords.x1 - coords.x0);
        }
        let best = 0;
        for (let i = 1; i < profile.points.length; i++)
          if (
            Math.abs(profile.points[i][0] - time) <
            Math.abs(profile.points[best][0] - time)
          )
            best = i;
        inspect(best);
      });
    caseSelect.addEventListener("change", chooseCase);
    processSelect.addEventListener("change", render);
    slider.addEventListener("input", () => inspect(Number(slider.value)));
    root.querySelector("#replay-peak").addEventListener("click", goPeak);
    svg.addEventListener("pointerdown", (e) => {
      const box = svg.getBoundingClientRect(),
        fraction =
          (((e.clientX - box.left) / box.width) * 1120 - 58) / (1120 - 58 - 20),
        time = coords.x0 + fraction * (coords.x1 - coords.x0);
      let best = 0;
      for (let i = 1; i < profile.points.length; i++)
        if (
          Math.abs(profile.points[i][0] - time) <
          Math.abs(profile.points[best][0] - time)
        )
          best = i;
      inspect(best);
    });
    chooseCase();
  }
  function validate(bundle) {
    if (
      !bundle.replay ||
      bundle.replay.schema_version !== 1 ||
      !Array.isArray(bundle.replay.profiles) ||
      !bundle.replay.profiles.length
    )
      throw new Error("Invalid memory replay");
    if (bundle.independent_timing)
      for (const row of bundle.independent_timing) {
        const requests = row.requests;
        if (
          !Array.isArray(requests) ||
          !requests.length ||
          typeof row.partial !== "boolean"
        )
          throw new Error("Missing independent timing requests");
        const stats = (values) => {
          if (values.some((n) => !Number.isFinite(n) || n < 0))
            throw new Error("Invalid independent timing");
          const sorted = [...values].sort((a, b) => a - b),
            mid = sorted.length >> 1;
          return {
            min: sorted[0],
            median:
              sorted.length % 2
                ? sorted[mid]
                : (sorted[mid - 1] + sorted[mid]) / 2,
            max: sorted.at(-1),
          };
        };
        row.partial =
          row.partial ||
          row.count !== requests.length ||
          Boolean(row.invalid?.length);
        row.count = requests.length;
        row.client = Object.fromEntries(
          ["ttff_seconds", "e2e_seconds", "post_ttff_fps"].map((k) => [
            k,
            stats(requests.map((r) => r.client[k])),
          ]),
        );
        const labels = Object.keys(requests[0].stage_cuda_ms).sort();
        if (
          requests.some(
            (r) =>
              JSON.stringify(Object.keys(r.stage_cuda_ms).sort()) !==
              JSON.stringify(labels),
          )
        )
          throw new Error("Inconsistent independent timing stage coverage");
        row.stages = Object.fromEntries(
          labels.map((k) => [
            k,
            stats(requests.map((r) => r.stage_cuda_ms[k])),
          ]),
        );
      }
    if (bundle.counter_series)
      for (const group of bundle.counter_series) {
        if (!Array.isArray(group.series) || !group.series.length)
          throw new Error("Invalid counter series");
        for (const series of group.series)
          if (
            !Array.isArray(series.points) ||
            !series.points.length ||
            series.points.some(
              (p, i) =>
                p.length !== 2 ||
                !p.every(Number.isFinite) ||
                p[1] < 0 ||
                (i && p[0] < series.points[i - 1][0]),
            )
          )
            throw new Error("Invalid counter sample");
      }
    const identities = new Set();
    for (const p of bundle.replay.profiles) {
      const key = JSON.stringify([p.case, p.pid, p.device]);
      if (identities.has(key))
        throw new Error("Duplicate memory profile identity");
      identities.add(key);
      if (
        typeof p.role !== "string" ||
        typeof p.case !== "string" ||
        !Number.isInteger(p.pid) ||
        !Number.isInteger(p.device) ||
        p.closure !== true ||
        typeof p.partial !== "boolean" ||
        !Array.isArray(p.points) ||
        !p.points.length ||
        !Array.isArray(p.peak_stacks)
      )
        throw new Error("Incomplete memory replay");
      for (const k of [
        "baseline_bytes",
        "new_peak_bytes",
        "new_active_peak_bytes",
        "net_peak_bytes",
        "reconstructed_peak_bytes",
        "peak_seconds",
        "recorded_start_seconds",
        "trace_events",
      ])
        if (!Number.isFinite(p[k]))
          throw new Error("Non-finite memory statistic");
      if (
        [
          "baseline_bytes",
          "new_peak_bytes",
          "new_active_peak_bytes",
          "net_peak_bytes",
        ].some((k) => p[k] < 0) ||
        p.new_active_peak_bytes < p.new_peak_bytes ||
        p.reconstructed_peak_bytes !== p.baseline_bytes + p.net_peak_bytes
      )
        throw new Error("Inconsistent memory peaks");
      for (const [field, label] of [
        ["stage_timeline", "stage"],
        ["dit_calls", "phase"],
      ]) {
        if (p[field] === undefined) continue;
        if (
          !Array.isArray(p[field]) ||
          p[field].some(
            (v) =>
              !v ||
              typeof v[label] !== "string" ||
              !Number.isFinite(v.start) ||
              !Number.isFinite(v.end) ||
              v.end < v.start ||
              (v.input_latent_range !== undefined &&
                (!Array.isArray(v.input_latent_range) ||
                  v.input_latent_range.some((n) => !Number.isFinite(n)))),
          )
        )
          throw new Error("Invalid stage timeline");
      }
      if (
        p.points.some(
          (v, i) =>
            !Array.isArray(v) ||
            v.length < 4 ||
            v.some((n) => !Number.isFinite(n)) ||
            (i && v[0] < p.points[i - 1][0]),
        )
      )
        throw new Error("Invalid memory timeline");
      if (
        p.peak_stacks.some(
          (g) =>
            !Number.isFinite(g.bytes) ||
            g.bytes < 0 ||
            !Number.isInteger(g.count) ||
            g.count < 0 ||
            !Array.isArray(g.frames) ||
            g.frames.some(
              (f) =>
                !f ||
                typeof f.filename !== "string" ||
                typeof f.name !== "string" ||
                !Number.isFinite(f.line),
            ),
        )
      )
        throw new Error("Invalid allocation stack");
      if (
        p.peak_stacks.reduce((n, g) => n + g.bytes, 0) +
          (p.other_peak_stack_bytes ?? 0) !==
        p.new_peak_bytes
      )
        throw new Error("Allocation stack bytes do not conserve");
    }
  }
  return { render, validate };
})();

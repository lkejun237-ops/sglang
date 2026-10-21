(function () {
  "use strict";
  const $ = (id) => document.getElementById("stream-" + id),
    esc = (v) =>
      String(v ?? "").replace(
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
  const median = (a) => {
      const v = a.filter(Number.isFinite).sort((a, b) => a - b),
        n = v.length;
      return n ? (n % 2 ? v[n >> 1] : (v[n / 2 - 1] + v[n / 2]) / 2) : null;
    },
    fmt = (v, n = 2) => (Number.isFinite(v) ? v.toFixed(n) : "未记录");
  const modes = { paced: "按 24 FPS 发送", unpaced: "尽快发送" };
  let data,
    groupIndex = 0;
  function validate(d) {
    if (
      !d?.config ||
      !Array.isArray(d.config.cases) ||
      !d.config.cases.length ||
      !Array.isArray(d.requests)
    )
      throw Error("需要包含 config.cases 和 requests 的流式结果 JSON。");
    if (
      d.server &&
      (!Array.isArray(d.server.spans) ||
        !Array.isArray(d.server.health) ||
        !d.server.validation ||
        !Array.isArray(d.server.validation.clock_domains) ||
        !d.server.validation.sessions ||
        !Array.isArray(d.server.hardware?.devices) ||
        !d.server.clients)
    )
      throw Error(
        "服务端数据需要 spans、health、validation、hardware 和 clients。",
      );
    if (d.config.transport !== "continuous_source")
      throw Error("此页读取 continuous_source 结果。");
    const labels = new Set(d.config.cases.map((c) => c.label)),
      ids = new Set();
    for (const c of d.config.cases) {
      if (
        typeof c.label !== "string" ||
        typeof c.case_id !== "string" ||
        !Number.isInteger(c.frames) ||
        c.frames < 1 ||
        (c.reference_ids != null && !Array.isArray(c.reference_ids))
      )
        throw Error("Case 需要名称、身份、有效帧数和参考 ID 数组。");
    }
    for (const r of d.requests) {
      if (
        !labels.has(r.label) ||
        typeof r.request_id !== "string" ||
        typeof r.success !== "boolean" ||
        !Number.isInteger(r.index) ||
        r.index < 0 ||
        ids.has(r.request_id) ||
        !["warmup", "measured"].includes(r.phase)
      )
        throw Error("Case、请求身份或 phase 不匹配。");
      ids.add(r.request_id);
      if (
        r.success &&
        (!Number.isFinite(r.metrics?.ttff_seconds) ||
          !Number.isFinite(r.metrics?.e2e_seconds) ||
          r.metrics.ttff_seconds < 0 ||
          r.metrics.e2e_seconds < r.metrics.ttff_seconds ||
          !Number.isInteger(r.output_frames) ||
          r.output_frames < 1)
      )
        throw Error("成功请求缺少有效交付指标。");
    }
    return d;
  }
  function label(g) {
    return (g.c.display_name ?? g.c.label) + " · " + (modes[g.mode] ?? g.mode);
  }
  function groups(d) {
    return d.config.cases
      .map((c) => ({
        c,
        mode:
          d.requests.find((r) => r.label === c.label)?.input_mode ??
          d.config.input_mode,
        rows: d.requests.filter((r) => r.label === c.label),
      }))
      .map((g) => ({
        ...g,
        formal: g.rows.filter((r) => r.phase === "measured" && r.success),
        failed: g.rows.filter((r) => r.phase === "measured" && !r.success),
        warm: g.rows.filter((r) => r.phase === "warmup"),
      }));
  }
  const stat = (g, key) => median(g.formal.map((r) => r.metrics[key]));
  function cellTable(rows, headers) {
    return (
      "<table><thead><tr>" +
      headers.map((h) => "<th>" + esc(h) + "</th>").join("") +
      "</tr></thead><tbody>" +
      rows
        .map(
          (row) =>
            "<tr>" +
            row
              .map(
                (v, i) =>
                  "<td" + (i ? ' class="num"' : "") + ">" + esc(v) + "</td>",
              )
              .join("") +
            "</tr>",
        )
        .join("") +
      "</tbody></table>"
    );
  }
  function refresh() {
    document.getElementById("report-title").textContent =
      "连续 source：场景与请求结果";
    document.getElementById("identity").textContent =
      "ZING · CONTINUOUS SOURCE";
    document.getElementById("input-description").textContent =
      "固定 source 与参考条件，逐段输入，按实际接收记录测量。";
    document.getElementById("finite-chips").hidden = true;
    const d = data,
      gs = groups(d);
    groupIndex = Math.min(groupIndex, gs.length - 1);
    $("setup").innerHTML = [
      d.config.deployment ?? "常驻服务",
      "并发 " + d.config.concurrency,
      "每组 " +
        d.config.warmup_runs +
        " 次预热 + " +
        d.config.measured_runs +
        " 次正式",
      "视频规格 24 FPS",
    ]
      .map((v) => '<span class="chip">' + esc(v) + "</span>")
      .join("");
    const passed = gs.flatMap((g) => g.formal),
      tt = passed.map((r) => r.metrics.ttff_seconds),
      fp = passed.map((r) => r.metrics.fps).filter(Number.isFinite);
    $("overview").textContent = passed.length
      ? "本批 " +
        passed.length +
        " 次正式成功请求：首批等待 " +
        fmt(Math.min(...tt)) +
        "–" +
        fmt(Math.max(...tt)) +
        " 秒" +
        (fp.length
          ? "，端到端交付 " +
            fmt(Math.min(...fp)) +
            "–" +
            fmt(Math.max(...fp)) +
            " FPS。"
          : "。")
      : "本批尚无正式成功请求。";
    const max = Math.max(
        5,
        ...gs.map((g) => stat(g, "e2e_seconds")).filter(Number.isFinite),
      ),
      axis = Math.ceil(max / 5) * 5;
    $("client-axis").textContent = axis + " s";
    $("client-bars").innerHTML = gs
      .map((g) => {
        const t = stat(g, "ttff_seconds"),
          e = stat(g, "e2e_seconds");
        return (
          '<div class="bar-row"><div class="bar-label"><span>' +
          esc(label(g)) +
          "</span><span>首批 " +
          fmt(t) +
          " s · 末批 " +
          fmt(e) +
          ' s</span></div><div class="track"><span style="width:' +
          ((t ?? 0) / axis) * 100 +
          '%;background:var(--cond)"></span><span style="width:' +
          ((e == null ? 0 : e - t) / axis) * 100 +
          '%;background:var(--accent)"></span></div></div>'
        );
      })
      .join("");
    $("comparison").innerHTML = gs
      .map(
        (g, i) =>
          '<tr class="' +
          (i === groupIndex ? "selected" : "") +
          '"><td><button data-group="' +
          i +
          '">' +
          esc(label(g)) +
          '</button><br><span class="tiny muted">成功 ' +
          g.formal.length +
          "/" +
          d.config.measured_runs +
          " · 失败 " +
          g.failed.length +
          "</span></td>" +
          ["ttff_seconds", "e2e_seconds", "fps"]
            .map((k) => '<td class="num">' + fmt(stat(g, k)) + "</td>")
            .join("") +
          "</tr>",
      )
      .join("");
    $("comparison")
      .querySelectorAll("button")
      .forEach(
        (b) =>
          (b.onclick = () => {
            groupIndex = Number(b.dataset.group);
            refresh();
          }),
      );
    $("case-select").innerHTML = gs
      .map((g, i) => '<option value="' + i + '">' + esc(label(g)) + "</option>")
      .join("");
    $("case-select").value = String(groupIndex);
    if (gs.length) detail(gs[groupIndex]);
    $("evidence-status").textContent =
      "本批 " +
      d.requests.length +
      " 次请求，正式成功 " +
      passed.length +
      " 次。预热及失败请求不进入正式统计。";
    $("stage-status").textContent = d.server
      ? "服务端阶段记录已载入，按客户端 session 关联；视频质量仍需查看生成媒体。"
      : (d.config.stage_evidence ??
        "本页展示客户端交付记录；没有已复核的 Encoder / DiT / Decoder 阶段数据。");
    $("provenance").innerHTML =
      "<p>" +
      esc(d.config.evidence_source ?? "Benchmark 导出的结果 JSON") +
      "</p><p>Job / 版本：<code>" +
      esc(d.config.job_id ?? "未记录") +
      " / " +
      esc(d.config.source_revision ?? "未记录") +
      '</code></p><p class="tiny muted">' +
      esc(
        d.config.measurement_note ??
          "起点在客户端连接与解码前；批次接收完成时记录交付。",
      ) +
      "</p>";
    $("identities").innerHTML = cellTable(
      d.requests.map((r) => [
        label(groups(d).find((g) => g.c.label === r.label)) +
          " / " +
          (r.phase === "warmup" ? "预热" : "正式") +
          "第 " +
          (r.index + 1) +
          " 次",
        r.server_session_id ?? r.request_id,
        r.output_frames,
        r.success ? "成功" : "失败",
      ]),
      ["请求", "会话身份", "输出帧数", "状态"],
    );
  }
  function detail(g) {
    const count = g.formal.length,
      duration = g.c.frames / 24;
    $("case-context").textContent =
      g.c.frames +
      " 帧 / 24 FPS" +
      (g.c.width ? " / " + g.c.width + " × " + g.c.height : "") +
      " · " +
      (g.c.reference_ids?.length ?? g.c.references?.length ?? "—") +
      " 项参考条件 · " +
      count +
      " 次正式成功";
    $("metrics").innerHTML = [
      ["首批等待", "ttff_seconds", "s"],
      ["全部收齐", "e2e_seconds", "s"],
      ["端到端交付", "fps", "FPS"],
    ]
      .map(([title, k, u]) => {
        const a = g.formal.map((r) => r.metrics[k]).filter(Number.isFinite);
        return (
          "<div><span>" +
          title +
          "</span><strong>" +
          fmt(stat(g, k)) +
          ' <span style="font-size:15px">' +
          u +
          "</span></strong><small>" +
          (a.length
            ? "范围 " + fmt(Math.min(...a)) + "–" + fmt(Math.max(...a))
            : "未记录") +
          "</small></div>"
        );
      })
      .join("");
    $("individual").innerHTML = g.rows
      .filter((r) => r.phase === "measured")
      .map(
        (r) =>
          "<tr><td>第 " +
          (r.index + 1) +
          " 次" +
          (r.success ? "" : " · 失败") +
          "</td>" +
          ["ttff_seconds", "e2e_seconds", "fps"]
            .map((k) => '<td class="num">' + fmt(r.metrics?.[k]) + "</td>")
            .join("") +
          '<td class="num">' +
          fmt(r.metrics?.icl_seconds?.p95) +
          "</td><td>" +
          ({ true: "是", false: "否" }[
            r.streaming?.first_output_before_source_end
          ] ?? "未记录") +
          "</td></tr>",
      )
      .join("");
    const promptRows = g.rows.filter((r) => r.phase === "measured").flatMap(
      (r) => (r.streaming?.prompt_updates ?? []).map((u) => ({ ...u, run: r.index + 1, successful_run: r.success })),
    );
    const promptLatencies = promptRows.filter((u) => u.successful_run && u.status === "applied")
      .map((u) => u.latency_ms).filter(Number.isFinite).sort((a, b) => a - b);
    const position = (promptLatencies.length - 1) * .95;
    const low = Math.floor(position), high = Math.ceil(position);
    const p95 = promptLatencies.length
      ? promptLatencies[low] + (promptLatencies[high] - promptLatencies[low]) * (position - low)
      : null;
    $("prompt-updates").innerHTML = promptRows.length
      ? "<p>applied 样本 " + promptLatencies.length + "；中位数 " + fmt(median(promptLatencies))
        + " ms；P95 " + fmt(p95) + " ms</p>" + cellTable(
          promptRows.map((u) => [u.run, u.event_id, u.status, fmt(u.submitted_seconds, 4),
            u.output_frame_start ?? "—", u.pts_us ?? "—", fmt(u.received_seconds, 4), fmt(u.latency_ms)]),
          ["轮次", "event_id", "状态", "提交 / s", "首帧序号", "视频PTS / µs", "收到 / s", "延迟 / ms"],
        )
      : "<p class='muted'>未记录 prompt 更新，不计为零延迟。</p>";
    $("warmups").innerHTML = cellTable(
      g.warm.map((r) => [
        "预热 " + (r.index + 1),
        fmt(r.metrics?.ttff_seconds),
        fmt(r.metrics?.e2e_seconds),
        r.success ? "成功" : (r.error ?? "失败"),
      ]),
      ["请求", "TTFF / s", "E2E / s", "状态"],
    );
    const gaps = g.formal.map((r) => r.metrics.icl_seconds),
      upper = Math.max(0.1, ...gaps.map((v) => v?.max).filter(Number.isFinite)),
      axis = Math.ceil(upper * 10) / 10;
    $("interval-axis").textContent = fmt(axis, 1) + " s";
    $("intervals").innerHTML = gaps
      .map((v, i) =>
        v && Number.isFinite(v.max)
          ? '<div class="tiny">第 ' +
            (g.formal[i].index + 1) +
            " 次 · " +
            v.count +
            " 个间隔 · P95 " +
            fmt(v.p95) +
            ' s</div><div class="interval"><span class="range" style="left:' +
            (v.min / axis) * 100 +
            "%;width:" +
            ((v.max - v.min) / axis) * 100 +
            '%"></span><span class="dot" title="中位数 ' +
            fmt(v.median, 4) +
            ' s" style="left:' +
            (v.median / axis) * 100 +
            '%"></span><span class="dot p95" title="P95 ' +
            fmt(v.p95, 4) +
            ' s" style="left:' +
            (v.p95 / axis) * 100 +
            '%"></span></div>'
          : '<p class="muted">第 ' + (i + 1) + " 次：未记录间隔分布</p>",
      )
      .join("");
    $("rhythm").innerHTML = cellTable(
      [
        ["TPOC · 平均间隔", fmt(stat(g, "tpoc_seconds"), 3) + " s"],
        ["ICL · 中位数", fmt(median(gaps.map((v) => v?.median)), 3) + " s"],
        ["ICL · P95", fmt(median(gaps.map((v) => v?.p95)), 3) + " s"],
        ["ICL · 最大间隔", fmt(median(gaps.map((v) => v?.max)), 3) + " s"],
      ],
      ["正式成功请求的中位数", "值"],
    );
    $("source-intro").textContent =
      g.mode === "paced"
        ? "按模拟的 24 FPS 采集时钟发送；credit 用尽时保留全部帧并等待，所以实际发送可能超过 " +
          duration +
          " 秒素材时长。"
        : "输入批次就绪后尽快发送；credit 用尽时等待服务端释放窗口。";
    const vals = [
      ["发送跨度", "source_send_span_seconds"],
      ["Source 至首批", "source_to_first_output_seconds"],
      ["Credit 等待", "credit_wait_seconds"],
      ["尾部排空", "tail_drain_seconds"],
    ].map(([n, k]) => [
      n,
      fmt(median(g.formal.map((r) => r.streaming?.[k]))) + " s",
    ]);
    vals.push([
      "依赖至输出 · 中位数",
      fmt(
        median(
          g.formal.map(
            (r) => r.streaming?.dependency_to_output_seconds?.median,
          ),
        ),
      ) + " s",
    ]);
    $("source-metrics").innerHTML = cellTable(vals, ["正式成功请求的中位数", "值"]);
    serverDetail(g);
  }
  const stageColor = {
    Encoder: "var(--encoder)",
    DiT: "var(--dit)",
    Decoder: "var(--decoder)",
    Conditioning: "var(--cond)",
  };
  function serverKey(r) {
    return (
      (r.worker_role === "dedicated_source_encoder"
        ? "独立 Encoder"
        : "SP rank " + r.rank) +
      " · " +
      r.stage +
      (r.stage === "Encoder" && r.component
        ? " / " +
          ({
            source_vae: "Source VAE",
            reference_vae: "参考图 VAE",
            qwen: "Qwen 条件编码",
          }[r.component] ?? r.component)
        : "")
    );
  }
  function stageGroups(rows) {
    const map = new Map();
    for (const r of rows) {
      if (!["Encoder", "DiT", "Decoder", "Conditioning"].includes(r.stage))
        continue;
      const key = serverKey(r);
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(r);
    }
    return [...map.entries()];
  }
  function serverDetail(g) {
    const server = data.server;
    if (!server) {
      $("server-status").textContent = "此结果文件没有服务端阶段记录。";
      $("server-bars").innerHTML = "";
      $("server-table").innerHTML = "";
      $("server-timeline").innerHTML = "";
      $("server-health").textContent = "";
      $("server-issues").textContent = "";
      $("server-request").innerHTML = "";
      $("request-identity").textContent = "";
      return;
    }
    const valid = server.validation,
      ids = g.formal.map((r) => r.server_session_id),
      complete =
        valid.collector_drained &&
        ids.length &&
        ids.every((id) => valid.sessions[id]?.complete);
    const spans = server.spans.filter((r) => ids.includes(r.request_id));
    $("server-status").textContent = complete
      ? "当前 case 的 " +
        ids.length +
        " 次正式请求：已记录阶段的范围及各 SP rank 完成记录通过检查，采集已排空。"
      : "当前 case 的阶段记录完整性尚未通过：" +
        [
          ...valid.issues,
          ...ids
            .filter((id) => !valid.sessions[id]?.complete)
            .map((id) => valid.sessions[id]?.error ?? "会话记录缺失"),
        ].join("；");
    $("server-issues").textContent = valid.dropped
      ? "进程累计丢弃 " +
        valid.dropped +
        " 条；对应 SP 记录缺口：" +
        valid.rank_pair_gaps
          .map(
            (v) =>
              v.label +
              " / " +
              (v.phase === "warmup" ? "预热" : "正式") +
              "-" +
              v.index +
              "（" +
              v.unmatched_spans +
              " 条）",
          )
          .join("，") +
        "。预热不进入上面的正式统计。"
      : "";
    const aggregate = stageGroups(spans).map(([key, rr]) => {
      const per = ids.map((id) => rr.filter((r) => r.request_id === id));
      return {
        key,
        stage: rr[0].stage,
        count: median(per.map((v) => v.length)),
        host: per.every(
          (v) => v.length && v.every((r) => Number.isFinite(r.host_call_ms)),
        )
          ? median(
              per.map((v) => v.reduce((a, r) => a + r.host_call_ms, 0) / 1000),
            )
          : null,
      };
    });
    const max = Math.max(
      0.01,
      ...aggregate.map((r) => r.host).filter(Number.isFinite),
    );
    $("server-bars").innerHTML = aggregate
      .map(
        (r) =>
          '<div class="bar-row"><div class="bar-label"><span>' +
          esc(r.key) +
          "</span><span>" +
          fmt(r.host, 3) +
          ' s</span></div><div class="track"><span style="width:' +
          ((r.host ?? 0) / max) * 100 +
          "%;background:" +
          stageColor[r.stage] +
          '"></span></div></div>',
      )
      .join("");
    $("server-table").innerHTML = aggregate
      .map(
        (r) =>
          "<tr><td>" +
          esc(r.key) +
          '</td><td class="num">' +
          r.count +
          '</td><td class="num">' +
          fmt(r.host, 3) +
          "</td></tr>",
      )
      .join("");
    $("server-request").innerHTML = g.formal
      .map(
        (r, i) =>
          '<option value="' +
          i +
          '">' +
          esc(label(g)) +
          " · 正式第 " +
          (r.index + 1) +
          " 次 · " +
          esc(r.server_session_id?.slice(0, 8)) +
          "</option>",
      )
      .join("");
    $("server-request").onchange = () =>
      serverTimeline(g.formal[Number($("server-request").value)], server);
    if (g.formal.length) serverTimeline(g.formal[0], server);
    else {
      $("request-identity").textContent = "";
      $("server-timeline").replaceChildren();
    }
    $("server-health").innerHTML =
      cellTable(
        server.health.map((h) => [
          h.file,
          h.written,
          h.dropped,
          h.pending,
          h.queued,
          h.error ?? "无",
        ]),
        ["采集文件", "写入", "丢弃", "待处理", "队列", "错误"],
      ) +
      "<p>时钟域：<code>" +
      esc(valid.clock_domains.join(", ")) +
      "</code></p>" +
      cellTable(
        server.hardware.devices.map((h) => [
          h.index,
          h.name,
          h.memory_MiB + " MiB",
          h.cc.join("."),
        ]),
        ["GPU", "型号", "显存", "Compute capability"],
      );
  }
  function serverTimeline(request, server) {
    const context = groups(data).find((g) => g.c.label === request.label);
    $("request-identity").innerHTML =
      "<strong>" +
      esc(label(context)) +
      " · 正式第 " +
      (request.index + 1) +
      " 次</strong><br><span>" +
      esc(context.c.frames) +
      " 帧 / 24 FPS · " +
      (context.c.reference_ids?.length ?? context.c.references?.length ?? "—") +
      ' 项参考条件</span><br><span class="tiny">Case：' +
      esc(request.case_id) +
      "<br>客户端请求：<code>" +
      esc(request.request_id) +
      "</code><br>服务端 session：<code>" +
      esc(request.server_session_id) +
      "</code></span><details><summary>查看请求内容</summary><p>Source：<code>" +
      esc(context.c.source_video ?? "未记录源文件路径") +
      "</code></p><p>" +
      esc(context.c.prompt ?? "此结果未记录 prompt") +
      "</p><p>参考 ID：<code>" +
      esc((context.c.reference_ids ?? []).join(" · ")) +
      "</code></p></details>";
    const sid = request.server_session_id,
      client = server.clients[sid],
      rr = server.spans.filter(
        (r) =>
          r.request_id === sid &&
          Number.isFinite(r.host_enter_ns) &&
          Number.isFinite(r.host_return_ns),
      );
    if (
      !client ||
      !rr.length ||
      server.validation.clock_domains.length !== 1 ||
      (!server.client_same_node_verified &&
        client.client_clock_domain !== server.validation.clock_domains[0])
    ) {
      $("server-timeline").textContent =
        "缺少可对齐的客户端记录或单节点时钟域。";
      return;
    }
    const start = client.first_output - client.ttff_seconds,
      lanes = stageGroups(rr),
      width = 1000,
      left = 215,
      right = 40,
      max = Math.max(
        client.e2e_seconds,
        ...rr.map((r) => r.host_return_ns / 1e9 - start),
      ),
      x = (t) => left + (t / max) * (width - left - right),
      height = 65 + lanes.length * 31;
    let svg =
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' +
      width +
      " " +
      height +
      '" role="img" aria-label="按客户端启动对齐的 Host 阶段时间线" style="width:100%;min-width:650px"><text x="' +
      left +
      '" y="18" font-size="12">客户端启动 · 0 s</text><text x="' +
      (width - right) +
      '" y="18" text-anchor="end" font-size="12">' +
      fmt(max) +
      " s</text>";
    for (const [i, [label, rows]] of lanes.entries()) {
      const y = 44 + i * 31;
      svg +=
        '<text x="0" y="' +
        (y + 10) +
        '" font-size="11">' +
        esc(label) +
        '</text><line x1="' +
        left +
        '" x2="' +
        (width - right) +
        '" y1="' +
        (y + 7) +
        '" y2="' +
        (y + 7) +
        '" stroke="#e4eae7"/>';
      for (const r of rows) {
        const a = r.host_enter_ns / 1e9 - start,
          b = r.host_return_ns / 1e9 - start;
        svg +=
          '<rect x="' +
          x(a) +
          '" y="' +
          y +
          '" width="' +
          Math.max(0.8, x(b) - x(a)) +
          '" height="14" fill="' +
          stageColor[r.stage] +
          '"><title>' +
          esc(r.stage + " · " + fmt(a, 3) + "–" + fmt(b, 3) + " s") +
          "</title></rect>";
      }
    }
    for (const [name, t] of [
      ["首批", client.ttff_seconds],
      ["末批", client.e2e_seconds],
    ])
      svg +=
        '<line x1="' +
        x(t) +
        '" x2="' +
        x(t) +
        '" y1="25" y2="' +
        (height - 18) +
        '" stroke="#536269" stroke-dasharray="3 3"/><text x="' +
        x(t) +
        '" y="' +
        (height - 3) +
        '" text-anchor="middle" font-size="11">' +
        name +
        " " +
        fmt(t) +
        "s</text>";
    $("server-timeline").innerHTML =
      '<div style="overflow:auto">' + svg + "</svg></div>";
  }

  $("case-select").onchange = (e) => {
    groupIndex = Number(e.target.value);
    refresh();
  };
  window.StreamingReport = {
    validate,
    render(bundle) {
      validate(bundle);
      data = bundle;
      groupIndex = 0;
      refresh();
    },
  };
})();

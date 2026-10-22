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
      fp = passed.map((r) => r.metrics.post_ttff_fps).filter(Number.isFinite);
    $("overview").textContent = passed.length
      ? "本批 " +
        passed.length +
        " 次正式成功请求：首批等待 " +
        fmt(Math.min(...tt)) +
        "–" +
        fmt(Math.max(...tt)) +
        " 秒" +
        (fp.length
          ? "，首批后生成 " +
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
          ["ttff_seconds", "e2e_seconds", "post_ttff_fps"]
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
      ["首批后生成", "post_ttff_fps", "FPS"],
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
          ["ttff_seconds", "e2e_seconds", "post_ttff_fps"]
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
    const boundaryRows = g.rows.filter(
      (r) =>
        r.phase === "measured" && (r.init_send_to_rgb || r.client_observation),
    );
    $("boundaries-section").hidden = !boundaryRows.length;
    $("client-boundaries").innerHTML = cellTable(
      boundaryRows.map((r) => [
        "第 " + (r.index + 1) + " 次" + (r.success ? "" : " · 失败"),
        fmt(r.init_send_to_rgb?.ttff_seconds),
        fmt(r.init_send_to_rgb?.e2e_seconds),
        fmt(r.init_send_to_rgb?.fps),
        fmt(r.client_observation?.elapsed_seconds),
        fmt(r.client_observation?.completion_overhead_seconds),
        r.delivery_complete == null
          ? "未记录"
          : r.delivery_complete
            ? "完整"
            : "不完整",
      ]),
      [
        "请求",
        "发送起点 TTFF / s",
        "发送起点 E2E / s",
        "发送起点 FPS",
        "客户端结束或失败 / s",
        "末批后结束开销 / s",
        "RGB 接收",
      ],
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

// Full-depth live-chain comparison, sharing the offline dataset selector.
const FullChainReport = (() => {
  const esc = (v) =>
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
  const f = (v, n = 2) => (Number.isFinite(v) ? v.toFixed(n) : "缺测");
  const distribution = (values) => {
    const a = values.filter(Number.isFinite).sort((x, y) => x - y),
      n = a.length;
    return n
      ? {
          n,
          median: n % 2 ? a[n >> 1] : (a[n / 2 - 1] + a[n / 2]) / 2,
          min: a[0],
          max: a[n - 1],
        }
      : null;
  };
  const colors = ["#287866", "#436db0", "#b16c35"];
  const stageColors = ["#8b65b5", "#d69c36", "#459da4", "#436db0", "#c96c58"];
  function componentPie(target) {
    const parts = target.parts.map((p, i) => ({
      name: p.name,
      ms: p.value?.median ?? 0,
      color: stageColors[i],
    }));
    const total = parts.reduce((sum, p) => sum + p.ms, 0);
    if (!total) return '<p class="muted">此场景未运行，暂无阶段占比。</p>';
    let angle = -Math.PI / 2;
    const wedges = parts
      .map((p) => {
        if (!p.ms) return "";
        const fraction = p.ms / total,
          end = angle + fraction * 2 * Math.PI;
        const startX = 160 + 140 * Math.cos(angle),
          startY = 160 + 140 * Math.sin(angle);
        const endX = 160 + 140 * Math.cos(end),
          endY = 160 + 140 * Math.sin(end);
        const title = esc(
          `${p.name} · ${f(fraction * 100, 1)}% · ${f(p.ms / 1000, 3)} s`,
        );
        let shape =
          fraction >= 1
            ? `<circle cx="160" cy="160" r="140" fill="${p.color}"><title>${title}</title></circle>`
            : `<path d="M160 160 L${startX} ${startY} A140 140 0 ${fraction > 0.5 ? 1 : 0} 1 ${endX} ${endY} Z" fill="${p.color}" stroke="var(--bg)" stroke-width="1.5"><title>${title}</title></path>`;
        if (fraction >= 0.06) {
          const middle = (angle + end) / 2;
          shape += `<text x="${160 + 95 * Math.cos(middle)}" y="${160 + 95 * Math.sin(middle)}" text-anchor="middle" dominant-baseline="middle" style="fill:white;font-size:17px;font-weight:650">${f(fraction * 100, 1)}%</text>`;
        }
        angle = end;
        return shape;
      })
      .join("");
    return `<figure class="component-pie" style="margin:12px 0 24px"><svg viewBox="0 0 320 320" role="img" aria-label="${esc(target.name)}各阶段累计CUDA耗时占比" style="max-width:360px;margin:0 auto">${wedges}</svg><figcaption>${parts.map((p) => `<div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:14px"><i style="width:12px;height:12px;flex-shrink:0;background:${p.color}"></i><span style="flex:1">${esc(p.name)}</span><strong>${f((p.ms / total) * 100, 1)}%</strong><span style="min-width:68px;text-align:right">${f(p.ms / 1000, 3)} s</span></div>`).join("")}</figcaption></figure>`;
  }
  function figureData(data) {
    return data.targets.map((t) => ({
      name: t.name,
      phase: t.phase_summary,
      operator: t.operator_profile
        ? {
            categories: t.operator_profile.categories,
            total_kernel_ms: t.operator_profile.total_kernel_ms,
            kernel_count: t.operator_profile.kernel_count,
          }
        : null,
    }));
  }
  function validate(data) {
    const fail = (message) => {
      throw Error("Invalid full-chain result: " + message);
    };
    const number = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0;
    const close = (a, b) =>
      Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(a), Math.abs(b));
    if (!Array.isArray(data.targets) || !data.targets.length)
      fail("no targets");
    for (const t of data.targets) {
      if (
        typeof t.name !== "string" ||
        !Array.isArray(t.cases) ||
        !t.cases.length
      )
        fail("missing target cases");
      for (const c of t.cases) {
        if (!Array.isArray(c.requests) || !c.requests.length)
          fail("no measured requests");
        for (const r of c.requests) {
          const m = r.metrics;
          if (
            !Number.isInteger(r.frames) ||
            r.frames <= 0 ||
            !m ||
            !["ttff_seconds", "e2e_seconds", "fps"].every((k) =>
              number(m[k]),
            ) ||
            m.ttff_seconds <= 0 ||
            m.e2e_seconds < m.ttff_seconds ||
            !close(m.fps, r.frames / m.e2e_seconds)
          )
            fail("inconsistent client metrics");
          if (m.first_batch_frames != null) {
            if (
              !Number.isInteger(m.first_batch_frames) ||
              m.first_batch_frames < 1 ||
              m.first_batch_frames > r.frames
            )
              fail("invalid first batch frame count");
            m.post_ttff_fps =
              m.e2e_seconds > m.ttff_seconds
                ? (r.frames - m.first_batch_frames) /
                  (m.e2e_seconds - m.ttff_seconds)
                : null;
          } else m.post_ttff_fps = null;
          if (
            !number(r.qwen_cuda_ms) ||
            !Array.isArray(r.stage_cuda_ms) ||
            r.stage_cuda_ms.some((s) => !number(s.ms))
          )
            fail("missing component timing");
          for (const rank of [0, 1])
            for (const stage of ["Conditioning", "Encoder", "DiT", "Decoder"])
              if (
                !r.stage_cuda_ms.some(
                  (s) =>
                    s.rank === rank &&
                    s.stage === stage &&
                    s.worker_role !== "dedicated_source_encoder",
                )
              )
                fail("missing main-worker stage");
          if (
            !r.stage_cuda_ms.some(
              (s) =>
                s.stage === "Encoder" &&
                s.worker_role === "dedicated_source_encoder",
            )
          )
            fail("missing source encoder");
          if (!Array.isArray(r.phase)) fail("missing phase availability");
          if (r.phase.length) {
            if (r.phase.length !== 2) fail("incomplete cycle");
            for (const [i, p] of r.phase.entries()) {
              if (
                p.chunk !== 9 + i ||
                p.latents !== [3, 2][i] ||
                !number(p.chunk_cuda_ms) ||
                !Array.isArray(p.calls) ||
                p.calls.length !== 6
              )
                fail("invalid phase chunk");
              for (const [j, call] of p.calls.entries())
                if (
                  call.phase !==
                    [
                      "source",
                      "denoise",
                      "denoise",
                      "denoise",
                      "denoise",
                      "clean_commit",
                    ][j] ||
                  !number(call.cuda_ms) ||
                  (j > 0 && j < 5 && call.step !== j - 1)
                )
                  fail("incomplete phase calls");
            }
          }
        }
        c.client = Object.fromEntries(
          ["ttff_seconds", "e2e_seconds", "fps", "post_ttff_fps"].map((k) => [
            k,
            distribution(c.requests.map((r) => r.metrics[k])),
          ]),
        );
      }
      const requests = t.cases.find((c) => c.label === "5s")?.requests ?? [];
      const measured = requests.filter((r) => r.phase.length);
      if (measured.length && measured.length !== requests.length)
        fail("partial phase repetitions");
      t.phase_summary = measured.length
        ? {
            "3_latent": distribution(
              measured.map((r) => r.phase[0].chunk_cuda_ms),
            ),
            "2_latent": distribution(
              measured.map((r) => r.phase[1].chunk_cuda_ms),
            ),
            "3_plus_2": distribution(
              measured.map(
                (r) => r.phase[0].chunk_cuda_ms + r.phase[1].chunk_cuda_ms,
              ),
            ),
            parts: Object.fromEntries(
              [3, 2].map((n, i) => [
                String(n),
                Object.fromEntries(
                  [
                    "source",
                    "denoise0",
                    "denoise1",
                    "denoise2",
                    "denoise3",
                    "clean_commit",
                  ].map((k, j) => [
                    k,
                    distribution(
                      measured.map((r) => r.phase[i].calls[j].cuda_ms),
                    ),
                  ]),
                ),
              ]),
            ),
          }
        : null;
      const p = t.operator_profile;
      if (p) {
        const categories = ["GEMM", "Attention", "KV/RoPE", "Other"];
        if (
          !p.categories ||
          categories.some(
            (k) =>
              !p.categories[k] ||
              !number(p.categories[k].cuda_ms) ||
              !Number.isInteger(p.categories[k].kernels) ||
              p.categories[k].kernels < 0,
          ) ||
          !number(p.total_kernel_ms) ||
          p.total_kernel_ms <= 0 ||
          !Number.isInteger(p.kernel_count) ||
          p.kernel_count <= 0
        )
          fail("invalid operator costs");
        if (
          !close(
            categories.reduce((s, k) => s + p.categories[k].cuda_ms, 0),
            p.total_kernel_ms,
          ) ||
          categories.reduce((s, k) => s + p.categories[k].kernels, 0) !==
            p.kernel_count
        )
          fail("operator costs do not conserve total");
        if (
          !["attributed_time_fraction", "attributed_kernel_fraction"].every(
            (k) => number(p[k]) && p[k] >= 0.95 && p[k] <= 1,
          )
        )
          fail("insufficient kernel attribution");
      }
    }
    // Published images are bound to their numeric inputs. Recomputed or edited
    // inputs remain viewable in live tables/bars without stale static figures.
    if (
      data.figures &&
      JSON.stringify(data.figure_data) !== JSON.stringify(figureData(data))
    )
      delete data.figures;
  }
  function bars(rows, value, unit) {
    const max = Math.max(1, ...rows.map((r) => value(r)?.max ?? 0));
    return rows
      .map((r, i) => {
        const v = value(r);
        return `<div class="bar-row"><div class="bar-label"><b>${esc(r.name)}</b><span>${v ? `${f(v.median)} ${unit} · ${f(v.min)}–${f(v.max)} · n=${v.n}` : "此项暂无数据"}</span></div><div class="track"><div class="bar" style="width:${v ? (v.median / max) * 100 : 0}%;background:${colors[i % 3]}"></div></div></div>`;
      })
      .join("");
  }
  function render(data) {
    validate(data);
    document.getElementById("report-title").textContent =
      "全 50 层实测：客户端与分阶段耗时";
    document.getElementById("identity").textContent =
      "ISSUE 334 · 真实输入 → Condition / Encoder → DiT → Decoder → RGB";
    document.getElementById("input-description").textContent =
      "同一原始视频与参考图，在线 Qwen 条件编码；SP2 DiT + 第三张卡运行 Qwen / Source Encoder。正式请求串行测量，旧结果仍可在上方切换。";
    const root = document.getElementById("phase-view");
    const labels = [
      ...new Set(data.targets.flatMap((t) => t.cases.map((c) => c.label))),
    ];
    root.innerHTML = `<div class="controls"><label>输入场景<select id="full-case">${labels.map((c) => `<option>${esc(c)}</option>`).join("")}</select></label></div><div id="full-content"></div>`;
    function update() {
      const label = document.getElementById("full-case").value;
      const rows = data.targets.map((t) => ({
        ...t,
        selected: t.cases.find((c) => c.label === label),
      }));
      const measured = rows.filter((t) => t.selected);
      const best = Math.min(
        ...measured.map((t) => t.selected.client.e2e_seconds.median),
      );
      const components = ["Conditioning", "Encoder", "DiT", "Decoder"];
      let html = `<div class="metrics">${rows.map((t, i) => `<article><small>${esc(t.name)}</small><strong>${f(t.selected?.client.e2e_seconds.median)} s</strong><span>客户端 E2E${t.selected ? ` · ${f(t.selected.client.e2e_seconds.median / best)}× 最快耗时` : ""}</span></article>`).join("")}</div><section><h2>首帧与完整输出</h2><p class="muted">FPS =（总输出帧数 − 首批帧数）/（末批到达时间 − 首批到达时间），不含 TTFF；仅一批输出时不计算。</p><div class="grid"><div><h3>TTFF / 秒</h3>${bars(rows, (r) => r.selected?.client.ttff_seconds, "s")}</div><div><h3>客户端 E2E / 秒</h3>${bars(rows, (r) => r.selected?.client.e2e_seconds, "s")}</div></div><div class="table-wrap"><table><thead><tr><th>实际硬件</th><th>输出帧</th><th>TTFF 中位数</th><th>E2E 中位数 [min, max]</th><th>FPS · TTFF 后</th><th>init-send → RGB E2E</th></tr></thead><tbody>${rows
        .map((t) => {
          const c = t.selected,
            s = c?.client.e2e_seconds;
          return `<tr><td>${esc(t.name)}</td><td>${c ? c.requests[0].frames : "未运行"}</td><td>${f(c?.client.ttff_seconds.median)} s</td><td>${s ? `${f(s.median)} [${f(s.min)}, ${f(s.max)}] s` : "未运行"}</td><td>${f(c?.client.post_ttff_fps?.median)}</td><td>${f(c ? distribution(c.requests.map((r) => r.init_send_to_rgb?.e2e_seconds))?.median : null)} s</td></tr>`;
        })
        .join(
          "",
        )}</tbody></table></div><p class="tiny muted">客户端从建立连接 / 启动解码前计时，到最后 RGB batch；init-send 起点另列，便于对照旧链路。同机回环 WebSocket；不含用户显示设备延迟。</p></section>`;
      const componentDefinitions = [
        ["Condition · Qwen", (r) => r.qwen_cuda_ms],
        [
          "Reference VAE",
          (r) =>
            r.stage_cuda_ms
              .filter(
                (s) =>
                  s.rank === 0 &&
                  s.stage === "Encoder" &&
                  s.worker_role !== "dedicated_source_encoder",
              )
              .reduce((a, s) => a + s.ms, 0),
        ],
        [
          "Source Encoder",
          (r) =>
            r.stage_cuda_ms
              .filter(
                (s) =>
                  s.rank === 0 &&
                  s.stage === "Encoder" &&
                  s.worker_role === "dedicated_source_encoder",
              )
              .reduce((a, s) => a + s.ms, 0),
        ],
        [
          "DiT",
          (r) =>
            r.stage_cuda_ms
              .filter((s) => s.rank === 0 && s.stage === "DiT")
              .reduce((a, s) => a + s.ms, 0),
        ],
        [
          "Decoder",
          (r) =>
            r.stage_cuda_ms
              .filter((s) => s.rank === 0 && s.stage === "Decoder")
              .reduce((a, s) => a + s.ms, 0),
        ],
      ];
      const componentRows = rows.map((t) => ({
        ...t,
        parts: componentDefinitions.map(([name, value]) => ({
          name,
          value: distribution((t.selected?.requests ?? []).map(value)),
        })),
      }));
      const componentMax = Math.max(
        1,
        ...componentRows.flatMap((t) => t.parts.map((p) => p.value?.max ?? 0)),
      );
      html += `<section><h2>全链路：时间花在哪里？</h2><p class="muted">每张卡的占比按下列五项累计 CUDA 耗时的中位数之和计算。</p><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(280px,1fr))">${componentRows.map((t) => `<div><h3>${esc(t.name)}</h3>${componentPie(t)}${t.parts.map((p, j) => `<div class="bar-row"><div class="bar-label"><span>${p.name}</span><span>${f(p.value?.median / 1000, 3)} s</span></div><div class="track"><div class="bar" style="width:${((p.value?.median ?? 0) / componentMax) * 100}%;background:${stageColors[j]}"></div></div></div>`).join("")}</div>`).join("")}</div><p class="tiny muted">Qwen 为真实在线调用；Reference VAE / DiT / Decoder 展示 rank 0，Source Encoder 在第三张卡。每项是单次请求内调用的累计 CUDA 时间，再取三次请求中位数；饼图表示所列组件的累计耗时构成，跨卡并行阶段不能相加当成 E2E。</p><details><summary>查看 Conditioning 外层与各 rank 的逐组件累计值</summary><div class="table-wrap"><table><thead><tr><th>硬件</th><th>组件 / worker</th><th>rank</th><th>累计 CUDA stream 时间 / s</th></tr></thead><tbody>${rows
        .flatMap((t) => {
          const rs = t.selected?.requests ?? [];
          const keys = new Map(
            rs
              .flatMap((r) => r.stage_cuda_ms)
              .map((s) => [
                JSON.stringify([s.worker_role, s.rank, s.stage, s.component]),
                s,
              ]),
          );
          return [...keys].map(([key, s]) => {
            const v = distribution(
              rs
                .map(
                  (r) =>
                    r.stage_cuda_ms.find(
                      (x) =>
                        JSON.stringify([
                          x.worker_role,
                          x.rank,
                          x.stage,
                          x.component,
                        ]) === key,
                    )?.ms,
                )
                .filter(Number.isFinite),
            );
            return `<tr><td>${esc(t.name)}</td><td>${esc(s.stage)} / ${esc(s.component || s.worker_role)}</td><td>${s.rank}</td><td>${f(v?.median / 1000, 3)}</td></tr>`;
          });
        })
        .join(
          "",
        )}</tbody></table></div><p>Conditioning 外层包括 Qwen RPC、参考图处理等准备工作；CUDA stream 区间中也可能有 CPU 等待。上方 Qwen 单列为服务端实际 GPU 调用。</p></details></section>`;
      if (label === "5s" && rows.some((t) => t.phase_summary)) {
        html += `<section><h2>稳定 3+2 cycle：分卡耗时对比</h2><p>取每次正式请求的 chunk 9（3 latent）与 chunk 10（2 latent），展示三次请求的中位数与 min/max。</p><div class="grid">${[
          ["3_latent", "3 latent"],
          ["2_latent", "2 latent"],
          ["3_plus_2", "3+2 cycle"],
        ]
          .map(
            ([key, title]) =>
              `<div><h3>${title}</h3>${bars(rows, (r) => r.phase_summary?.[key], "ms")}</div>`,
          )
          .join(
            "",
          )}</div>${data.figures?.cycle ? `<figure><img style="width:100%" src="${esc(data.figures.cycle)}" alt="三卡3 latent、2 latent和3+2 cycle CUDA耗时及min/max"></figure>` : ""}<p class="tiny muted">Rank 0 的独立 step CUDA stream 区间；cycle 为两个区间之和，两个 chunk 之间的 Decoder 不计入该 cycle。客户端 E2E 在上方单独展示。</p></section>`;
        html += `<section><h2>Phase 分项：Source、四步 Denoise、Clean commit</h2>${data.figures?.phases ? `<figure><img style="width:100%" src="${esc(data.figures.phases)}" alt="3与2 latent的六个调用独立CUDA耗时"></figure>` : ""}<div class="table-wrap"><table><thead><tr><th>硬件</th><th>latent</th>${["source", "denoise0", "denoise1", "denoise2", "denoise3", "clean_commit"].map((k) => `<th>${k} / ms</th>`).join("")}</tr></thead><tbody>${rows.flatMap((t) => [3, 2].map((n) => `<tr><td>${esc(t.name)}</td><td>${n}</td>${["source", "denoise0", "denoise1", "denoise2", "denoise3", "clean_commit"].map((k) => `<td>${f(t.phase_summary?.parts[n]?.[k]?.median)}</td>`).join("")}</tr>`)).join("")}</tbody></table></div></section>`;
        html += `<section><h2>算子构成与绝对成本</h2><p><strong>已有 profile：${
          rows
            .filter((t) => t.operator_profile)
            .map((t) => esc(t.name))
            .join("、") || "暂无"
        }</strong>${
          rows.some((t) => !t.operator_profile)
            ? `；${rows
                .filter((t) => !t.operator_profile)
                .map((t) => esc(t.name))
                .join("、")} 的算子 profile 待补测。`
            : ""
        }</p>${data.figures?.operators ? `<figure><img style="width:100%" src="${esc(data.figures.operators)}" alt="每卡GEMM Attention KV/RoPE Other饼图及绝对毫秒对比"></figure>` : ""}<div class="table-wrap"><table><thead><tr><th>算子类别</th>${rows.map((t) => `<th>${esc(t.name)}</th>`).join("")}</tr></thead><tbody>${[
          "GEMM",
          "Attention",
          "KV/RoPE",
          "Other",
        ]
          .map(
            (k) =>
              `<tr><td>${k}</td>${rows
                .map((t) => {
                  const p = t.operator_profile,
                    v = p?.categories[k]?.cuda_ms;
                  return `<td>${f(v)} ms${p ? ` · ${f((v / p.total_kernel_ms) * 100, 1)}%` : ""}</td>`;
                })
                .join("")}</tr>`,
          )
          .join(
            "",
          )}</tbody></table></div><p class="tiny muted">Rank 0 的一次独立诊断，区间覆盖 chunk 9、10 及中间 Decoder。类别按 kernel launch 关联互斥归属；绝对值是 kernel-work ms，饼图不是端到端墙钟占比；第三卡 Qwen / Encoder 见上方完整链路表。</p></section>`;
      }
      if (label === "5s" && !rows.some((t) => t.phase_summary))
        html += `<section><h2>Phase 与算子计时</h2><p>生产 step 路由的计时补测已通过预检，等待当前长视频采样完成后执行。客户端与完整链路数据已先展示，分项结果回收后在本页补齐。</p></section>`;
      html += `<details><summary>实验配置、缺测项与原始证据</summary><p>全50层 BF16，SP2，window6 / sink3，4步 denoise，clean commit，joint_forward=off；在线 Qwen、原始 reference/source VAE、Lynn Decoder；未使用预制 conditioning。每场景1次预热、3次正式请求；profile 单独运行，不计入正式时延。</p><p>${esc(data.notes ?? "")}</p><pre style="white-space:pre-wrap">${esc(
        JSON.stringify(
          data.targets.map((t) => ({
            name: t.name,
            status: t.status,
            evidence_root: t.evidence_root,
            phase_present: !!t.phase_summary,
            profile_present: !!t.operator_profile,
            offline_operator_validation: t.offline_operator_validation,
          })),
          null,
          2,
        ),
      )}</pre></details>`;
      document.getElementById("full-content").innerHTML = html;
    }
    document.getElementById("full-case").onchange = update;
    update();
  }
  return { validate, render };
})();

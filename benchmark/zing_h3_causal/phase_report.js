// SPDX-License-Identifier: Apache-2.0
(function () {
  "use strict";
  const kinds = {
    block: "整 chunk",
    prefix: "Prefix",
    source: "Source",
    denoise: "Denoise",
    clean_commit: "Clean commit",
  };
  const fields = [
    "mode",
    "kind",
    "chunk",
    "latent_frames",
    "step",
    "q_rows",
    "prefix_rows",
    "history_rows",
    "kv_rows",
  ];
  const number = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0;
  const integer = (v) => Number.isInteger(v) && v >= 0;
  function distribution(values) {
    const a = values.slice().sort((a, b) => a - b),
      n = a.length;
    return {
      count: n,
      median: n % 2 ? a[n >> 1] : (a[n / 2 - 1] + a[n / 2]) / 2,
      min: a[0],
      max: a[n - 1],
    };
  }
  function summarize(samples) {
    const groups = new Map(),
      blocks = new Map(),
      cycles = new Map();
    for (const r of samples.filter((r) => r.run_kind === "measured")) {
      const key = JSON.stringify(fields.map((k) => r[k]));
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
      if (r.mode === "block" && r.chunk >= 3)
        blocks.set(`${r.run}/${r.chunk}`, r);
    }
    for (const a of blocks.values()) {
      const b = blocks.get(`${a.run}/${a.chunk + 1}`);
      if (a.latent_frames !== 3 || b?.latent_frames !== 2) continue;
      if (!cycles.has(a.chunk)) cycles.set(a.chunk, []);
      cycles.get(a.chunk).push({
        run: a.run,
        chunks: [a.chunk, a.chunk + 1],
        wall_ms: a.wall_ms + b.wall_ms,
        cuda_ms: a.cuda_ms + b.cuda_ms,
      });
    }
    return {
      groups: [...groups.values()].map((rows) => ({
        ...Object.fromEntries(fields.map((k) => [k, rows[0][k]])),
        wall_ms: distribution(rows.map((r) => r.wall_ms)),
        cuda_ms: distribution(rows.map((r) => r.cuda_ms)),
      })),
      cycles: [...cycles]
        .sort((a, b) => a[0] - b[0])
        .map(([chunk, rows]) => ({
          chunks: [chunk, chunk + 1],
          samples: rows,
          wall_ms: distribution(rows.map((r) => r.wall_ms)),
          cuda_ms: distribution(rows.map((r) => r.cuda_ms)),
        })),
    };
  }
  function validate(data) {
    const fail = () => {
      throw Error(
        "Invalid DiT phase result: inconsistent runs or raw timing samples",
      );
    };
    if (!Array.isArray(data.phase_runs) || !data.phase_runs.length) fail();
    for (const run of data.phase_runs) {
      if (
        typeof run.name !== "string" ||
        !run.config ||
        !integer(run.config.layers) ||
        run.config.layers < 1 ||
        !Array.isArray(run.runs) ||
        !run.runs.length ||
        !Array.isArray(run.samples) ||
        !run.samples.length
      )
        fail();
      const identities = new Map(),
        seen = new Set();
      for (const r of run.runs) {
        const key = `${r.mode}/${r.run}`;
        if (
          !["block", "forward"].includes(r.mode) ||
          !integer(r.run) ||
          !["warmup", "measured"].includes(r.run_kind) ||
          r.finite !== true ||
          typeof r.output_sha256 !== "string" ||
          identities.has(key)
        )
          fail();
        identities.set(key, r);
      }
      if (new Set(run.runs.map((r) => r.output_sha256)).size !== 1) fail();
      for (const r of run.samples) {
        const owner = identities.get(`${r.mode}/${r.run}`),
          key = `${r.mode}/${r.run}/${r.chunk}/${r.kind}/${r.step}`;
        if (
          !owner ||
          !integer(r.run) ||
          r.run_kind !== owner.run_kind ||
          !integer(r.latent_frames) ||
          (r.latent_frames < 1 && r.kind !== "prefix") ||
          !Number.isInteger(r.chunk) ||
          r.chunk < -1 ||
          !number(r.wall_ms) ||
          !number(r.cuda_ms) ||
          seen.has(key)
        )
          fail();
        if (r.mode === "block") {
          if (r.kind !== "block" || r.step !== null || r.chunk < 0) fail();
        } else if (
          !["prefix", "source", "denoise", "clean_commit"].includes(r.kind) ||
          (r.kind === "denoise" ? !integer(r.step) : r.step !== null) ||
          !["q_rows", "prefix_rows", "history_rows", "kv_rows"].every((k) =>
            integer(r[k]),
          ) ||
          r.kv_rows !==
            r.q_rows +
              (r.kind === "prefix" ? 0 : r.prefix_rows + r.history_rows)
        )
          fail();
        seen.add(key);
      }
      // Browser imports are recomputed too; edited/stale summaries cannot set the charts.
      run.summary = summarize(run.samples);
      if (!run.summary.groups.length) fail();
      if (run.profile) {
        const p = run.profile,
          names = ["GEMM", "Attention", "KV/RoPE", "Other"];
        if (
          p.schema_version !== 1 ||
          !number(p.total_kernel_ms) ||
          p.total_kernel_ms <= 0 ||
          !integer(p.kernel_count) ||
          p.kernel_count === 0 ||
          !number(p.attributed_kernel_fraction) ||
          p.attributed_kernel_fraction < 0.95 ||
          p.attributed_kernel_fraction > 1 ||
          !number(p.attributed_time_fraction) ||
          p.attributed_time_fraction < 0.95 ||
          p.attributed_time_fraction > 1 ||
          p.output_sha256 !== run.runs[0].output_sha256 ||
          !Array.isArray(p.chunks) ||
          p.chunks.length !== 2 ||
          !run.summary.cycles.some(
            (c) => JSON.stringify(c.chunks) === JSON.stringify(p.chunks),
          ) ||
          !names.every(
            (k) =>
              number(p.categories?.[k]?.cuda_ms) &&
              integer(p.categories?.[k]?.kernels),
          )
        )
          fail();
        if (
          Math.abs(
            names.reduce((a, k) => a + p.categories[k].cuda_ms, 0) -
              p.total_kernel_ms,
          ) > 1e-6 ||
          names.reduce((a, k) => a + p.categories[k].kernels, 0) !==
            p.kernel_count
        )
          fail();
      }
    }
    return data;
  }
  const fmt = (v) => (number(v) ? v.toFixed(3) : "未测");
  function element(tag, text, parent) {
    const e = document.createElement(tag);
    if (text != null) e.textContent = text;
    if (parent) parent.append(e);
    return e;
  }
  function table(parent, headings, rows, id) {
    const wrap = element("div", null, parent);
    wrap.className = "table-wrap";
    const t = element("table", null, wrap);
    if (id) t.id = id;
    const h = element("tr", null, element("thead", null, t));
    headings.forEach((v) => element("th", v, h));
    const body = element("tbody", null, t);
    rows.forEach((values) => {
      const tr = element("tr", null, body);
      values.forEach((v) => element("td", v ?? "—", tr));
    });
    return t;
  }
  function select(parent, id, label, options, initial) {
    const l = element("label", label, parent),
      s = element("select", null, l);
    s.id = id;
    for (const [value, text] of options) {
      const o = element("option", text, s);
      o.value = value;
    }
    s.value = initial;
    return s;
  }
  const chartColors = ["#39796b", "#7962a5", "#c17b35", "#79909d"];
  const categories = ["GEMM", "Attention", "KV/RoPE", "Other"];
  // Canonical comparison data excludes host paths and runtime versions; those
  // describe the deployment being compared and remain visible in evidence.
  function comparisonSignature(run) {
    const c = run.config,
      o = c.options ?? {};
    if (!c.source_archive_sha256 || !c.selected_weights || !c.kernel_route_envs)
      return null;
    const canonical = (v) =>
      Array.isArray(v)
        ? v.map(canonical)
        : v && typeof v === "object"
          ? Object.fromEntries(
              Object.keys(v)
                .sort()
                .map((k) => [k, canonical(v[k])]),
            )
          : v;
    return JSON.stringify(
      canonical({
        source: c.source_archive_sha256,
        conditioning: c.conditioning_sha256,
        configs: [
          c.model_config_sha256,
          c.runtime_config_sha256,
          c.vae_config_sha256,
        ],
        weights: c.selected_weights,
        layers: c.layer_ids,
        sp: c.sp,
        backend: c.effective_backend,
        graph: c.graph,
        routes: c.kernel_route_envs,
        latent: c.latent_shape,
        sampling: Object.fromEntries(
          ["seed", "window_chunks", "sink_chunks", "denoising_step_list"].map(
            (k) => [k, o[k]],
          ),
        ),
        geometry: run.summary.groups.map((g) => fields.map((k) => g[k])),
      }),
    );
  }
  function bars(parent, rows, maximum, id) {
    const box = element("div", null, parent);
    box.className = "comparison-bars";
    if (id) box.id = id;
    for (const row of rows) {
      const line = element("div", null, box);
      line.className = "comparison-bar-row";
      element("span", row.label, line).className = "comparison-bar-label";
      const track = element("div", null, line);
      track.className = "comparison-bar-track";
      const fill = element("div", null, track);
      fill.className = "comparison-bar-fill";
      fill.style.width = `${(100 * row.value) / maximum}%`;
      fill.style.background = row.color;
      if (row.min != null) {
        const range = element("div", null, track);
        range.className = "comparison-bar-range";
        range.style.left = `${(100 * row.min) / maximum}%`;
        range.style.width = `${(100 * (row.max - row.min)) / maximum}%`;
      }
      line.dataset.value = row.value;
      element(
        "span",
        `${fmt(row.value)} ms${row.min == null ? "" : ` · ${fmt(row.min)}–${fmt(row.max)}`}${row.ratio == null ? "" : ` · ${row.ratio.toFixed(2)}×`}`,
        line,
      ).className = "comparison-bar-value";
    }
  }
  function renderComparison(parent, runs) {
    const groups = new Map();
    for (const run of runs) {
      const key = comparisonSignature(run);
      if (!key) continue;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(run);
    }
    const eligible = [...groups.values()].filter((a) => a.length >= 2);
    if (!eligible.length) return;
    const style = element("style", null, parent);
    style.textContent = `.comparison-section{margin-bottom:32px}.comparison-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px}.comparison-bars{display:grid;gap:12px;margin:14px 0 26px}.comparison-bar-row{display:grid;grid-template-columns:minmax(100px,180px) minmax(70px,1fr) minmax(170px,260px);gap:12px;align-items:center;font-size:13px}.comparison-bar-track{height:18px;background:#edf0ef;position:relative}.comparison-bar-fill{height:100%;min-width:1px;border-radius:3px}.comparison-bar-range{position:absolute;top:7px;height:4px;background:#202c2b;border-left:2px solid #202c2b;border-right:2px solid #202c2b}.comparison-pie{width:170px;height:170px;border-radius:50%;margin:18px auto}.comparison-legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}.comparison-legend i{display:inline-block;width:10px;height:10px;margin-right:6px}.comparison-bar-value{font-variant-numeric:tabular-nums}.comparison-grid .comparison-bar-row{grid-template-columns:minmax(100px,145px) minmax(80px,1fr) 85px;gap:8px}.comparison-headline{font-size:20px;font-weight:650}.comparison-target{border:1px solid #dce5df;border-radius:12px;padding:18px}@media(max-width:720px){.comparison-bar-row,.comparison-grid .comparison-bar-row{grid-template-columns:100px 1fr}.comparison-bar-value{grid-column:2;font-size:12px}.comparison-bar-label{grid-row:span 2}.comparison-grid{grid-template-columns:1fr}}`;
    for (const targets of eligible) {
      const section = element("section", null, parent);
      section.className = "comparison-section";
      section.id = "phase-comparison";
      const chunks = targets[0].summary.cycles.at(-1)?.chunks;
      if (
        !chunks ||
        !targets.every((r) =>
          r.summary.cycles.some(
            (c) => JSON.stringify(c.chunks) === JSON.stringify(chunks),
          ),
        )
      )
        continue;
      const baseline =
        targets.find((r) => /H20/i.test(r.config.device)) ?? targets[0];
      const cycle = (r) =>
        r.summary.cycles.find((c) => c.chunks[0] === chunks[0]);
      const fastest = targets.reduce((a, b) =>
        cycle(a).wall_ms.median < cycle(b).wall_ms.median ? a : b,
      );
      element("h2", `${targets.length} 卡耗时对比`, section);
      element(
        "p",
        `${fastest.name} 的 3+2 周期最短：${fmt(cycle(fastest).wall_ms.median)} ms`,
        section,
      ).className = "comparison-headline";
      element(
        "p",
        `同一 K=${targets[0].config.layers}、SP=${targets[0].config.sp} 工作负载 · chunk ${chunks.join(" / ")} · 柱长为中位数，黑线为 min/max · 倍速以 ${baseline.name} 为 1×`,
        section,
      ).className = "muted";
      for (const [label, frames] of [
        ["3-latent", 3],
        ["2-latent", 2],
        ["稳定 3+2 cycle", null],
      ]) {
        element("h3", label, section);
        const metric = (r) =>
          frames == null
            ? cycle(r).wall_ms
            : r.summary.groups.find(
                (g) =>
                  g.mode === "block" &&
                  chunks.includes(g.chunk) &&
                  g.latent_frames === frames,
              )?.wall_ms;
        const base = metric(baseline)?.median;
        const rows = targets
          .map((r, i) => {
            const d = metric(r);
            return (
              d && {
                label: r.name,
                value: d.median,
                min: d.min,
                max: d.max,
                ratio: d.median > 0 ? base / d.median : null,
                color: chartColors[i % 4],
              }
            );
          })
          .filter(Boolean);
        bars(
          section,
          rows,
          Math.max(...rows.map((r) => r.max)) * 1.08,
          `comparison-latency-${frames ?? "cycle"}`,
        );
      }
      element("h2", "Phase 分项：差距集中在哪个调用", section);
      const grid = element("div", null, section);
      grid.className = "comparison-grid";
      const steps = [
        ...new Set(
          targets[0].summary.groups
            .filter((g) => g.kind === "denoise")
            .map((g) => g.step),
        ),
      ].sort((a, b) => a - b);
      const calls = [
        ["source", null, "Source"],
        ...steps.map((s) => ["denoise", s, `Denoise · step ${s}`]),
        ["clean_commit", null, "Clean commit"],
      ];
      const get = (r, frames, kind, step) =>
        r.summary.groups.find(
          (g) =>
            g.mode === "forward" &&
            chunks.includes(g.chunk) &&
            g.latent_frames === frames &&
            g.kind === kind &&
            g.step === step,
        )?.wall_ms.median;
      const phaseMax =
        Math.max(
          ...targets.flatMap((r) =>
            [3, 2].flatMap((f) => calls.map(([k, s]) => get(r, f, k, s) ?? 0)),
          ),
        ) * 1.08;
      for (const frames of [3, 2]) {
        const panel = element("div", null, grid);
        element("h3", `${frames}-latent phase`, panel);
        for (const [kind, step, label] of calls) {
          element("h4", label, panel);
          bars(
            panel,
            targets.map((r, i) => ({
              label: r.name,
              value: get(r, frames, kind, step) ?? 0,
              color: chartColors[i % 4],
            })),
            phaseMax,
          );
        }
      }
      const profiles = targets.filter(
        (r) =>
          r.profile &&
          JSON.stringify(r.profile.chunks) === JSON.stringify(chunks),
      );
      element("h2", "算子构成与绝对成本", section);
      if (profiles.length !== targets.length)
        element("p", "部分目标的独立 profile 尚未完成。", section);
      if (profiles.length) {
        const legend = element("div", null, section);
        legend.className = "comparison-legend";
        categories.forEach((name, i) => {
          const item = element("span", null, legend);
          element("i", null, item).style.background = chartColors[i];
          item.append(document.createTextNode(name));
        });
        const pies = element("div", null, section);
        pies.className = "comparison-grid";
        for (const run of profiles) {
          const p = run.profile,
            card = element("div", null, pies);
          card.className = "comparison-target";
          element("h3", run.name, card);
          const pie = element("div", null, card);
          pie.className = "comparison-pie";
          pie.setAttribute("role", "img");
          let start = 0;
          const slices = categories.map((k, i) => {
            const end =
              start + (100 * p.categories[k].cuda_ms) / p.total_kernel_ms;
            const value = `${chartColors[i]} ${start}% ${end}%`;
            start = end;
            return value;
          });
          pie.style.background = `conic-gradient(${slices.join(",")})`;
          pie.setAttribute(
            "aria-label",
            categories
              .map(
                (k) =>
                  `${k} ${((100 * p.categories[k].cuda_ms) / p.total_kernel_ms).toFixed(1)}%`,
              )
              .join(", "),
          );
          table(
            card,
            ["类别", "占比", "Kernel ms"],
            categories.map((k) => [
              k,
              `${((100 * p.categories[k].cuda_ms) / p.total_kernel_ms).toFixed(1)}%`,
              fmt(p.categories[k].cuda_ms),
            ]),
          );
          element(
            "p",
            `Kernel 总计 ${fmt(p.total_kernel_ms)} ms · ${p.kernel_count} 个`,
            card,
          ).className = "muted";
        }
        const max =
          Math.max(
            ...profiles.flatMap((r) =>
              categories.map((k) => r.profile.categories[k].cuda_ms),
            ),
          ) * 1.08;
        for (const category of categories) {
          element("h3", `${category} · 绝对 ms`, section);
          bars(
            section,
            profiles.map((r) => ({
              label: r.name,
              value: r.profile.categories[category].cuda_ms,
              color: chartColors[targets.indexOf(r) % 4],
            })),
            max,
          );
        }
      }
      const details = element("details", null, section);
      element("summary", "对比口径与运行环境", details);
      element(
        "p",
        "这是部署目标对比，各目标的 Torch / FA4 版本列于下表。耗时柱来自不带 profiler 的正式重复；phase 为单独同步的 forward，不能相加代替整块。饼图为另一次连续 3+2 profile 中互斥分类的 kernel 时间之和，不能当作 wall 占比。",
        details,
      );
      table(
        details,
        [
          "目标",
          "实际 GPU",
          "Torch",
          "CUDA",
          "FA4",
          "Profile wall / ms",
          "Kernel 关联覆盖",
        ],
        targets.map((r) => [
          r.name,
          r.config.device,
          r.config.torch,
          r.config.cuda,
          r.config.versions?.["flash-attn-4"],
          fmt(r.profile?.profile_wall_ms),
          r.profile
            ? `${(100 * r.profile.attributed_time_fraction).toFixed(1)}%`
            : "未测",
        ]),
      );
    }
  }

  function render(data) {
    validate(data);
    document.getElementById("identity").textContent = "ZING · DiT 组件测速";
    document.getElementById("report-title").textContent =
      "3 / 2 phase：整块耗时与逐步拆解";
    document.getElementById("input-description").textContent =
      "先看一个完整周期需要多久，再看 Source、各步 Denoise 和 Clean commit 各自的耗时。";
    const root = document.getElementById("phase-view");
    root.replaceChildren();
    renderComparison(root, data.phase_runs);
    const top = element("div", null, root);
    top.className = "controls";
    const runSelect = select(
      top,
      "phase-run",
      "测量配置",
      data.phase_runs.map((r, i) => [String(i), r.name]),
      "0",
    );
    const content = element("div", null, root);
    function showRun() {
      content.replaceChildren();
      const run = data.phase_runs[Number(runSelect.value)],
        c = run.config,
        summary = run.summary;
      const last = summary.cycles.at(-1),
        chunks = last?.chunks ?? [];
      const chips = element("div", null, content);
      chips.className = "chips";
      for (const text of [
        c.device ?? "设备未记录",
        `K=${c.layers} · SP=${c.sp ?? "未记录"}`,
        `${c.effective_backend ?? c.attention_backend ?? "未记录"} · ${c.graph ? "Graph" : "eager"}`,
        `latent ${c.latent_shape?.join(" × ") ?? "未记录"}`,
      ])
        element("span", text, chips).className = "chip";
      const cards = element("div", null, content);
      cards.className = "metrics";
      for (const [label, value] of [
        ["一个 3+2 周期 · wall 中位数", last?.wall_ms.median],
        ...[3, 2].map((phase) => [
          `${phase}-latent 整块 · wall 中位数`,
          summary.groups.find(
            (g) =>
              g.mode === "block" &&
              chunks.includes(g.chunk) &&
              g.latent_frames === phase,
          )?.wall_ms.median,
        ]),
      ]) {
        const card = element("div", null, cards);
        element("small", label, card);
        element("strong", `${fmt(value)} ms`, card);
      }
      element("h2", "各次调用耗时", content);
      element(
        "p",
        `展示末个完整周期（chunk ${chunks.join(" / ") || "未测"}），每项为独立 forward 的中位数。`,
        content,
      ).className = "muted";
      const overviewControls = element("div", null, content);
      overviewControls.className = "controls";
      const overviewMetric = select(
        overviewControls,
        "phase-overview-metric",
        "时间",
        [
          ["wall_ms", "Wall"],
          ["cuda_ms", "CUDA event"],
        ],
        "wall_ms",
      );
      const overview = element("div", null, content);
      function showOverview() {
        overview.replaceChildren();
        const stepIds = [
          ...new Set(
            summary.groups
              .filter((g) => g.kind === "denoise")
              .map((g) => g.step),
          ),
        ].sort((a, b) => a - b);
        const calls = [
          ["source", null, "Source 写入"],
          ...stepIds.map((step) => ["denoise", step, `Denoise · step ${step}`]),
          ["clean_commit", null, "Clean commit"],
        ];
        const rows = calls.map(([kind, step, label]) => [
          label,
          ...[3, 2].map(
            (frames) =>
              summary.groups.find(
                (g) =>
                  g.mode === "forward" &&
                  chunks.includes(g.chunk) &&
                  g.latent_frames === frames &&
                  g.kind === kind &&
                  g.step === step,
              )?.[overviewMetric.value].median,
          ),
        ]);
        const max = Math.max(
          1,
          ...rows.flatMap((r) => r.slice(1)).filter(number),
        );
        const t = table(
          overview,
          ["调用", "3-latent / ms", "2-latent / ms"],
          rows.map((r) => [r[0], ...r.slice(1).map(fmt)]),
          "phase-overview",
        );
        [...t.tBodies[0].rows].forEach((tr, i) =>
          [1, 2].forEach((j) => {
            tr.cells[j].style.background =
              `linear-gradient(90deg, ${j === 1 ? "#e1eee8" : "#ece6f5"} ${((rows[i][j] ?? 0) / max) * 100}%, transparent 0)`;
          }),
        );
      }
      overviewMetric.addEventListener("change", showOverview);
      showOverview();
      const details = element("details", null, content);
      details.id = "phase-details";
      element("summary", "查看其他 chunk、逐步样本与 history", details);
      const controls = element("div", null, details);
      controls.className = "controls";
      const mode = select(
        controls,
        "phase-mode",
        "计时模式",
        [
          ["forward", "独立 forward"],
          ["block", "整 chunk baseline"],
        ],
        "forward",
      );
      const metric = select(
        controls,
        "phase-metric",
        "时间",
        [
          ["wall_ms", "Wall / ms"],
          ["cuda_ms", "CUDA event / ms"],
        ],
        "wall_ms",
      );
      const chunkIds = [...new Set(summary.groups.map((g) => g.chunk))].sort(
        (a, b) => a - b,
      );
      const chunk = select(
        controls,
        "phase-chunk",
        "Chunk",
        [
          ...(last
            ? [["pair", `末个完整 3+2 cycle · ${chunks.join(" / ")}`]]
            : []),
          ["all", "全部（含启动 / 尾块）"],
          ...chunkIds.map((i) => [
            String(i),
            i < 0 ? "Prefix 初始化" : `Chunk ${i}${i < 3 ? " · 启动" : ""}`,
          ]),
        ],
        last ? "pair" : "all",
      );
      const phase = select(
        controls,
        "phase-latents",
        "Latent phase",
        [
          ["all", "全部"],
          ["3", "3-latent"],
          ["2", "2-latent"],
          ["1", "1-latent · 启动 / 尾块"],
        ],
        "all",
      );
      const kind = select(
        controls,
        "phase-kind",
        "调用",
        [
          ["all", "全部"],
          ...Object.entries(kinds).filter(([k]) => k !== "block"),
        ],
        "all",
      );
      const detail = element("div", null, details);
      function showGroups() {
        detail.replaceChildren();
        kind.disabled = mode.value === "block";
        const groups = summary.groups.filter(
          (g) =>
            g.mode === mode.value &&
            (chunk.value === "all" ||
              (chunk.value === "pair"
                ? chunks.includes(g.chunk)
                : g.chunk === Number(chunk.value))) &&
            (phase.value === "all" ||
              g.latent_frames === Number(phase.value)) &&
            (mode.value === "block" ||
              kind.value === "all" ||
              g.kind === kind.value),
        );
        const max = Math.max(1, ...groups.map((g) => g[metric.value].median));
        element(
          "p",
          `${groups.length} 个分组 · 同 chunk / step / history 独立统计，时间单位 ms。Q / KV 为逻辑 packed rows。`,
          detail,
        ).className = "muted";
        const t = table(
          detail,
          [
            "Chunk",
            "Latent",
            "调用",
            "Step",
            "Q",
            "Prefix",
            "History",
            "KV",
            "样本",
            "Median / ms",
            "Min / ms",
            "Max / ms",
          ],
          groups.map((g) => {
            const v = g[metric.value];
            return [
              g.chunk,
              g.latent_frames,
              kinds[g.kind],
              g.step,
              g.q_rows,
              g.prefix_rows,
              g.history_rows,
              g.kv_rows,
              v.count,
              fmt(v.median),
              fmt(v.min),
              fmt(v.max),
            ];
          }),
          "phase-groups",
        );
        [...t.tBodies[0].rows].forEach((tr, i) => {
          const td = tr.cells[9];
          td.style.background = `linear-gradient(90deg, #e1eee8 ${(groups[i][metric.value].median / max) * 100}%, transparent 0)`;
        });
        if (!groups.length) element("p", "当前筛选没有样本。", detail);
      }
      for (const control of [mode, metric, chunk, phase, kind])
        control.addEventListener("change", showGroups);
      showGroups();
      const cycles = element("details", null, content);
      element("summary", "查看全部 3+2 周期与耗时波动", cycles);
      element(
        "p",
        "先按同一 run 的连续 3-latent + 2-latent chunk 求和，再统计。排除启动 1,2,2 和不完整尾块；不同 cycle 分开保留。",
        cycles,
      );
      table(
        cycles,
        [
          "Chunk 配对",
          "样本",
          "Wall median / ms",
          "Wall min / ms",
          "Wall max / ms",
          "CUDA median / ms",
        ],
        summary.cycles.map((r) => [
          r.chunks.join(" + "),
          r.wall_ms.count,
          fmt(r.wall_ms.median),
          fmt(r.wall_ms.min),
          fmt(r.wall_ms.max),
          fmt(r.cuda_ms.median),
        ]),
        "phase-cycles",
      );
      if (!summary.cycles.length) element("p", "没有完整稳定 cycle。", cycles);
      const evidence = element("details", null, content);
      element("summary", "计时口径、配置与数据来源", evidence);
      element(
        "p",
        "整块计时包含 source、全部 denoise、clean commit 和 sampler 开销。逐次 forward 单独同步测量，会改变发射节奏，不能相加代替整块耗时。CUDA event 是当前 stream 经过时间，wall 包含完成等待。",
        evidence,
      );
      element(
        "p",
        "本页为真实权重缩层 DiT 测量，不包含编解码和网络。跨设备比较需使用相同层数、权重、输入、backend 与 history；完整模型 FPS 需另测。",
        evidence,
      );
      element(
        "p",
        `预热 ${run.runs.filter((r) => r.run_kind === "warmup").length} 次，正式 ${run.runs.filter((r) => r.run_kind === "measured").length} 次（两种模式分别计数）；${run.samples.length} 条原始调用，统计排除预热。`,
        evidence,
      );
      element(
        "pre",
        JSON.stringify(
          { config: c, runs: run.runs, sources: run.sources },
          null,
          2,
        ),
        evidence,
      ).style.cssText =
        "white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px";
    }
    runSelect.addEventListener("change", showRun);
    showRun();
  }
  window.PhaseReport = { validate, render };
})();

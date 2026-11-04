// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(function () {
  "use strict";
  // Switchyard dashboard tab: live routing decisions from the switchyard sidecar, for every preset.
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;

  const React = SDK.React;
  const h = React.createElement;
  const { useState, useEffect, useCallback, useRef } = SDK.hooks;
  const { Card, CardHeader, CardTitle, CardContent, Badge, Button } = SDK.components;

  const POLL_MS = 1000;
  const API = "/api/plugins/nvidia-switchyard";
  const VGR_PHASES = [
    ["attempt", "Attempt"],
    ["type", "Type"],
    ["verify", "Verify"],
    ["serve", "Serve"],
  ];
  const ROUTE_PHASES = [
    ["route", "Route"],
    ["serve", "Serve"],
  ];
  const OUTCOMES = {
    local_committed: ["Committed local", "local"],
    local_tool_call: ["Tool call on local", "local"],
    local_after_cloud_failure: ["Local after cloud failure", "local"],
    escalated_after_verification: ["Escalated after verification", "cloud"],
    escalated_by_trajectory_judge: ["Escalated by trajectory judge", "cloud"],
    escalated_on_time_limit: ["Escalated on time limit", "cloud"],
    escalated_without_attempt: ["Escalated without attempt", "cloud"],
    shadow_tool_call_allowed: ["Shadow: tool call allowed", "local"],
    shadow_checked: ["Shadow: checked, cloud served", "muted"],
    shadow_unchecked: ["Shadow: not checked, cloud served", "muted"],
    served_local: ["Served locally", "local"],
    served_cloud: ["Served from cloud", "cloud"],
    cancelled: ["Cancelled by client", "muted"],
    failed: ["Failed", "error"],
  };

  function fmt(n) {
    if (n == null) return "-";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return String(n);
  }

  function ms(n) {
    if (n == null) return "";
    return n >= 1000 ? (n / 1000).toFixed(1) + "s" : n + "ms";
  }

  function ago(ts, now) {
    const s = Math.max(0, Math.round(now - ts));
    if (s < 60) return s + "s ago";
    if (s < 3600) return Math.round(s / 60) + "m ago";
    return Math.round(s / 3600) + "h ago";
  }

  function TierPill(props) {
    const tier = props.tier;
    return h("span", { className: "sy-tier sy-tier-" + (tier || "idle") }, tier ? tier.toUpperCase() : "IDLE");
  }

  const VERDICT_WORDS = { yes: "approved", no: "rejected", unknown: "unclear" };

  function pct(p) {
    return Math.round(p * 100) + "%";
  }

  // What one call found, in words: the trajectory judge's P(yes) is the odds the work is going wrong.
  function stageDetail(s) {
    if (s.tool_call) return "tool call";
    if (s.stage === "trajectory_judge") return s.p_yes != null ? "escalate " + pct(s.p_yes) : "no score";
    if (s.stage === "readout" || s.stage === "deliberation_verdict") return s.p_yes != null ? pct(s.p_yes) + " sure" : "no clear yes/no";
    if (s.stage === "typing") return s.verdict || "";
    return s.verdict ? (VERDICT_WORDS[s.verdict] || s.verdict) : "";
  }

  function StageChip(props) {
    const s = props.stage;
    const detail = stageDetail(s);
    return h("span", {
      className: "sy-chip sy-chip-" + s.tier + (s.status === "running" ? " sy-chip-running" : ""),
      title: s.label + " on " + s.model + " (" + s.prompt_tokens + " in / " + s.completion_tokens + " out" +
        (s.estimated ? ", estimated" : "") + ")",
    },
      h("span", { className: "sy-chip-name" }, s.label),
      detail ? h("span", { className: "sy-chip-detail" }, detail) : null,
      s.ms != null ? h("span", { className: "sy-chip-ms" }, ms(s.ms)) : null,
    );
  }

  function Stepper(props) {
    const active = props.phase;
    const seen = new Set((props.stages || []).map(function (s) { return s.phase; }));
    const phases = props.algorithm === "vgr" ? VGR_PHASES : ROUTE_PHASES;
    return h("div", { className: "sy-stepper" }, phases.map(function (p) {
      const state = p[0] === active ? "active" : seen.has(p[0]) ? "done" : "todo";
      return h("div", { key: p[0], className: "sy-step sy-step-" + state }, p[1]);
    }));
  }

  function NowCard(props) {
    const current = props.current;
    const last = props.last;
    if (!current) {
      return h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Now")),
        h(CardContent, null,
          h("div", { className: "sy-now" },
            h(TierPill, { tier: last ? last.served_tier : null }),
            h("div", null,
              h("div", { className: "sy-now-label" }, "Idle"),
              h("div", { className: "sy-muted" }, last
                ? "Last turn served by " + (last.served_tier || "-") + " tier" + (last.why ? ": " + last.why.headline : "")
                : "Waiting for the first routed turn")))));
    }
    const turn = current.turn;
    return h(Card, null,
      h(CardHeader, null, h(CardTitle, null, "Now: turn #" + turn.id + " · " + turn.preset)),
      h(CardContent, null,
        h("div", { className: "sy-now" },
          h(TierPill, { tier: current.tier }),
          h("div", null,
            h("div", { className: "sy-now-label sy-pulse" }, current.label),
            h("div", { className: "sy-muted" },
              ({ tool_step: "Tool step", auxiliary: "Auxiliary task" }[turn.kind] || "User turn") + " · " + ms(current.elapsed_ms) +
              (turn.task_type ? " · typed " + turn.task_type : "") +
              " · local " + fmt(turn.tokens.local.prompt + turn.tokens.local.completion) +
              " / cloud " + fmt(turn.tokens.cloud.prompt + turn.tokens.cloud.completion) + " tokens"))),
        h(Stepper, { phase: current.phase, stages: turn.stages, algorithm: turn.algorithm }),
        h("div", { className: "sy-chips" }, turn.stages.map(function (s, i) {
          return h(StageChip, { key: i, stage: s });
        }))));
  }

  function TierTokens(props) {
    const t = props.tokens;
    return h("div", { className: "sy-tokens sy-tokens-" + props.tier },
      h("div", { className: "sy-tokens-head" }, h(TierPill, { tier: props.tier }), h("span", { className: "sy-muted" }, props.model)),
      h("div", { className: "sy-tokens-total" }, fmt(t.prompt + t.completion)),
      h("div", { className: "sy-muted" }, fmt(t.prompt) + " in · " + fmt(t.completion) + " out · " + t.calls + " calls"));
  }

  function TokensCard(props) {
    const s = props.state;
    const totals = s.totals;
    const served = totals.served.local + totals.served.cloud;
    const localShare = served ? Math.round((100 * totals.served.local) / served) : 0;
    let verify = 0;
    Object.keys(totals.by_stage).forEach(function (k) {
      if (k !== "local_draft" && k !== "cloud_serve") {
        verify += totals.by_stage[k].prompt + totals.by_stage[k].completion;
      }
    });
    const rows = Object.keys(totals.by_stage).map(function (k) {
      const b = totals.by_stage[k];
      return h("tr", { key: k },
        h("td", null, k.replace(/_/g, " ")),
        h("td", null, b.calls),
        h("td", null, fmt(b.prompt)),
        h("td", null, fmt(b.completion)));
    });
    return h(Card, null,
      h(CardHeader, null, h(CardTitle, null, "Tokens this sidecar run" + (props.preset ? " · " + props.preset : ""))),
      h(CardContent, null,
        h("div", { className: "sy-grid2" },
          h(TierTokens, { tier: "local", model: s.local_model, tokens: totals.local }),
          h(TierTokens, { tier: "cloud", model: s.cloud_model, tokens: totals.cloud })),
        h("div", { className: "sy-share" },
          h("div", { className: "sy-share-bar" }, h("div", { className: "sy-share-local", style: { width: localShare + "%" } })),
          h("div", { className: "sy-muted" },
            localShare + "% of " + served + " turns served locally · routing overhead " + fmt(verify) + " tokens")),
        rows.length ? h("table", { className: "sy-table" },
          h("thead", null, h("tr", null, h("th", null, "Stage"), h("th", null, "Calls"), h("th", null, "In"), h("th", null, "Out"))),
          h("tbody", null, rows)) : null));
  }

  const TIER_COLOR = { local: "#22c55e", cloud: "#60a5fa" };
  const ROUTING_COLOR = "#f59e0b";
  const WORK_COLOR = "#94a3b8";
  // Stages whose output answers the task; every other stage is the router deciding where to send it.
  const WORK_STAGES = { local_draft: true, cloud_serve: true };

  // Two-part donut; the centre shows the first part's share.
  function Donut(props) {
    const first = props.first;
    const second = props.second;
    const total = first.value + second.value;
    const r = 44;
    const c = 2 * Math.PI * r;
    const frac = total ? first.value / total : 0;
    return h("div", { className: "sy-donut" },
      h("div", { className: "sy-donut-title" }, props.title),
      h("svg", { viewBox: "0 0 120 120", width: 132, height: 132, role: "img",
        "aria-label": props.title + ": " + first.name + " " + props.format(first.value) + ", " +
          second.name + " " + props.format(second.value) },
        h("circle", { cx: 60, cy: 60, r: r, fill: "none", strokeWidth: 16,
          stroke: total ? second.color : "var(--color-border)" }),
        total ? h("circle", { cx: 60, cy: 60, r: r, fill: "none", strokeWidth: 16, stroke: first.color,
          strokeDasharray: frac * c + " " + c, transform: "rotate(-90 60 60)",
          style: { transition: "stroke-dasharray 0.4s ease" } }) : null,
        h("text", { x: 60, y: 58, textAnchor: "middle", className: "sy-donut-pct" },
          total ? Math.round(frac * 100) + "%" : "-"),
        h("text", { x: 60, y: 76, textAnchor: "middle", className: "sy-donut-sub" }, first.name)),
      h("div", { className: "sy-donut-legend" },
        h("span", { className: "sy-dot", style: { background: first.color } }), first.name + " " + props.format(first.value),
        h("span", { className: "sy-dot", style: { background: second.color } }), second.name + " " + props.format(second.value)));
  }

  function tierParts(local, cloud) {
    return {
      first: { name: "local", value: local, color: TIER_COLOR.local },
      second: { name: "cloud", value: cloud, color: TIER_COLOR.cloud },
    };
  }

  function SplitCard(props) {
    const totals = props.state.totals;
    const tokens = function (t) { return t.prompt + t.completion; };
    let routingMs = 0;
    let workMs = 0;
    Object.keys(totals.by_stage).forEach(function (k) {
      if (WORK_STAGES[k]) workMs += totals.by_stage[k].ms || 0;
      else routingMs += totals.by_stage[k].ms || 0;
    });
    return h(Card, null,
      h(CardHeader, null, h(CardTitle, null, "Local vs cloud, and routing overhead")),
      h(CardContent, null,
        h("div", { className: "sy-donuts" },
          h(Donut, Object.assign({ title: "Tokens", format: fmt }, tierParts(tokens(totals.local), tokens(totals.cloud)))),
          h(Donut, Object.assign({ title: "Time in model calls", format: ms }, tierParts(totals.local.ms || 0, totals.cloud.ms || 0))),
          h(Donut, Object.assign({ title: "Turns served", format: String }, tierParts(totals.served.local, totals.served.cloud))),
          h(Donut, { title: "Time spent routing", format: ms,
            first: { name: "routing", value: routingMs, color: ROUTING_COLOR },
            second: { name: "task work", value: workMs, color: WORK_COLOR } })),
        h("div", { className: "sy-muted" },
          "Time is the summed duration of upstream calls; calls that overlap count in full. " +
          "Routing is every classifier, judge and verifier call (for VGR, task typing and its verifiers); " +
          "task work is the answers each tier produced, including VGR drafts later discarded. " +
          (props.background ? "Background tasks (titles, compression) are included."
            : "Background tasks (titles, compression) are left out, as in the timeline below."))));
  }

  // One block per stretch of consecutive calls on the same tier within one turn.
  function routingRuns(segments) {
    const runs = [];
    segments.forEach(function (s) {
      let run = runs[runs.length - 1];
      if (!run || run.tier !== s.tier || run.turn !== s.turn_id) {
        run = { tier: s.tier, turn: s.turn_id, newTurn: !!run && run.turn !== s.turn_id,
          switched: !!run && run.tier !== s.tier, ms: 0, tokens: 0, calls: 0, labels: [], parts: [], running: false };
        runs.push(run);
      }
      const routing = !WORK_STAGES[s.stage];
      let part = run.parts[run.parts.length - 1];
      if (!part || part.routing !== routing) {
        part = { routing: routing, ms: 0, tokens: 0 };
        run.parts.push(part);
      }
      part.ms += s.ms || 0;
      part.tokens += s.tokens || 0;
      run.ms += s.ms || 0;
      run.tokens += s.tokens || 0;
      run.calls += 1;
      run.running = run.running || s.running;
      run.labels.push(s.label + " " + ms(s.ms) + (s.running ? " (running)" : ""));
    });
    return runs;
  }

  function TimelineCard(props) {
    const [metric, setMetric] = useState("time");
    const background = props.background;
    const segments = (props.state.timeline || []).filter(function (s) {
      return background || s.kind !== "auxiliary";
    });
    const runs = routingRuns(segments);
    const value = function (run) { return metric === "time" ? run.ms : run.tokens; };
    const show = metric === "time" ? ms : fmt;
    const total = runs.reduce(function (sum, run) { return sum + value(run); }, 0);
    const sums = { local: 0, cloud: 0, routing: 0, work: 0 };
    runs.forEach(function (run) {
      sums[run.tier] += value(run);
      run.parts.forEach(function (p) { sums[p.routing ? "routing" : "work"] += value(p); });
    });
    const switches = runs.filter(function (run) { return run.switched; }).length;
    const toggle = function (id, label) {
      return h("button", { type: "button", className: "sy-seg" + (metric === id ? " sy-seg-on" : ""),
        onClick: function () { setMetric(id); } }, label);
    };
    return h(Card, null,
      h(CardHeader, null, h(CardTitle, null, "Routing timeline")),
      h(CardContent, null,
        h("div", { className: "sy-timeline-controls" },
          h("div", { className: "sy-segs" }, toggle("time", "Seconds"), toggle("tokens", "Tokens")),
          h("label", { className: "sy-muted sy-check" },
            h("input", { type: "checkbox", checked: background,
              onChange: function (e) { props.onBackground(e.target.checked); } }),
            "include background tasks (titles, compression)")),
        !runs.length || !total
          ? h("div", { className: "sy-muted" }, "No model calls yet.")
          : h("div", { className: "sy-timeline-scroll" },
            h("div", { className: "sy-timeline", style: { minWidth: Math.min(runs.length * 28, 4000) + "px" } },
              runs.map(function (run, i) {
                const v = value(run);
                const share = v / total;
                return h("div", {
                  key: i,
                  className: "sy-run" + (run.newTurn ? " sy-run-turn" : "") + (run.running ? " sy-pulse" : ""),
                  style: { flexGrow: Math.max(v, total * 0.004) },
                  title: run.tier + ": " + show(v) + " · " + run.calls + " call" + (run.calls > 1 ? "s" : "") +
                    (run.turn != null ? " · turn #" + run.turn : "") + "\n" + run.labels.join("\n"),
                },
                  h("div", { className: "sy-run-marks" }, run.parts.map(function (p, j) {
                    return h("div", { key: j, className: "sy-mark sy-mark-" + (p.routing ? "routing" : "work"),
                      style: { flexGrow: Math.max(value(p), v * 0.02) } });
                  })),
                  h("div", { className: "sy-run-bar sy-run-" + run.tier }, share > 0.05 ? show(v) : ""));
              }))),
        runs.length && total ? h("div", { className: "sy-muted sy-timeline-summary" },
          h("span", { className: "sy-dot", style: { background: ROUTING_COLOR } }), " routing " + show(sums.routing),
          h("span", { className: "sy-dot", style: { background: WORK_COLOR } }), " task work " + show(sums.work) +
          " (thin line above each block) · " +
          switches + " local/cloud switch" + (switches === 1 ? "" : "es") + " · gaps separate turns" +
          " · local " + show(sums.local) + " · cloud " + show(sums.cloud) +
          " · oldest on the left, newest on the right") : null));
  }

  function minus(a, b) {
    const out = {};
    Object.keys(a).forEach(function (k) { out[k] = a[k] - ((b && b[k]) || 0); });
    return out;
  }

  // Totals with the sidecar's background share (titles, compression) taken out.
  function withoutBackground(totals) {
    const bg = totals.background;
    if (!bg) return totals;
    const byStage = {};
    Object.keys(totals.by_stage || {}).forEach(function (k) {
      const b = minus(totals.by_stage[k], bg.by_stage && bg.by_stage[k]);
      if (b.calls) byStage[k] = b;
    });
    return Object.assign({}, totals, {
      local: minus(totals.local, bg.local), cloud: minus(totals.cloud, bg.cloud),
      served: minus(totals.served, bg.served), turns: totals.turns - bg.turns,
      by_stage: totals.by_stage ? byStage : undefined,
    });
  }

  // Totals for one preset: tier tokens and served counts from the sidecar, stages from recent turns.
  function presetTotals(state, preset, background) {
    if (!preset) return background ? state.totals : withoutBackground(state.totals);
    const found = state.totals.by_preset[preset] ||
      { turns: 0, served: { local: 0, cloud: 0 }, local: { prompt: 0, completion: 0, calls: 0, ms: 0 },
        cloud: { prompt: 0, completion: 0, calls: 0, ms: 0 } };
    const own = background ? found : withoutBackground(found);
    const byStage = {};
    (state.turns || []).forEach(function (t) {
      if (t.preset !== preset || (!background && t.kind === "auxiliary")) return;
      t.stages.forEach(function (s) {
        const b = byStage[s.stage] || (byStage[s.stage] = { prompt: 0, completion: 0, calls: 0, ms: 0 });
        b.prompt += s.prompt_tokens; b.completion += s.completion_tokens; b.calls += 1; b.ms += s.ms || 0;
      });
    });
    return { local: own.local, cloud: own.cloud, served: own.served, turns: own.turns, by_stage: byStage };
  }

  function PresetsCard(props) {
    const state = props.state;
    const config = props.config;
    const rows = config.presets.map(function (p) {
      const t = state.totals.by_preset[p.name] || null;
      const served = t ? t.served.local + t.served.cloud : 0;
      const tokens = function (x) { return x ? x.prompt + x.completion : 0; };
      return h("tr", { key: p.name, className: props.selected === p.name ? "sy-row-on" : "",
        onClick: function () { props.onSelect(props.selected === p.name ? "" : p.name); } },
        h("td", null, p.name, p.name === config.default_preset ? h("span", { className: "sy-muted" }, " (default)") : null),
        h("td", null, p.label, p.route.mode && p.algorithm === "vgr" ? h("span", { className: "sy-muted" }, " · " + p.route.mode) : null),
        h("td", null, t ? t.turns : 0),
        h("td", null, served ? Math.round((100 * t.served.local) / served) + "%" : "-"),
        h("td", null, fmt(tokens(t && t.local))),
        h("td", null, fmt(tokens(t && t.cloud))));
    });
    return h(Card, null,
      h(CardHeader, null, h(CardTitle, null, "Presets")),
      h(CardContent, null,
        h("table", { className: "sy-table sy-table-click" },
          h("thead", null, h("tr", null, h("th", null, "Preset"), h("th", null, "Algorithm"), h("th", null, "Turns"),
            h("th", null, "Served locally"), h("th", null, "Local tokens"), h("th", null, "Cloud tokens"))),
          h("tbody", null, rows)),
        h("div", { className: "sy-muted" }, "Click a preset to filter the cards below; click again to show all.")));
  }

  const CHECK_MARKS = { true: ["\u2713", "ok"], false: ["\u2717", "fail"], null: ["?", "unclear"] };

  // Why VGR served this turn where it did: the rule that fired and what each check found.
  function Why(props) {
    const why = props.why;
    return h("div", { className: "sy-why sy-why-" + (why.served || "muted") },
      h("div", { className: "sy-why-head" },
        h("span", { className: "sy-why-label" }, why.served === "cloud" ? "Why cloud" : "Why local"),
        h("span", { className: "sy-why-headline" }, why.headline)),
      h("div", { className: "sy-why-summary" }, why.summary),
      why.checks.length ? h("ul", { className: "sy-why-checks" }, why.checks.map(function (c, i) {
        const mark = CHECK_MARKS[String(c.ok)] || CHECK_MARKS.null;
        return h("li", { key: i },
          h("span", { className: "sy-why-mark sy-why-mark-" + mark[1] }, mark[0]),
          c.label + ": " + c.result);
      })) : null,
      why.source === "inferred"
        ? h("div", { className: "sy-muted" }, "Inferred from the recorded calls; Switchyard's own reason was not in sidecar.log.")
        : null);
  }

  function TurnRow(props) {
    const t = props.turn;
    const o = OUTCOMES[t.outcome] || [t.outcome || "-", "error"];
    const dur = t.ended ? Math.round((t.ended - t.started) * 1000) : null;
    return h("div", { className: "sy-turn" },
      h("div", { className: "sy-turn-head" },
        h("span", { className: "sy-turn-id" }, "#" + t.id),
        h(Badge, null, t.preset),
        h("span", { className: "sy-outcome sy-outcome-" + o[1] }, o[0]),
        t.task_type ? h(Badge, null, t.task_type) : null,
        h("span", { className: "sy-muted" },
          ({ tool_step: "tool step", auxiliary: "auxiliary task" }[t.kind] || "user turn") + " · " + ms(dur) + " · " + ago(t.started, props.now)),
        h("span", { className: "sy-turn-tokens" },
          "L " + fmt(t.tokens.local.prompt + t.tokens.local.completion) +
          " · C " + fmt(t.tokens.cloud.prompt + t.tokens.cloud.completion))),
      t.preview ? h("div", { className: "sy-preview" }, t.preview) : null,
      t.why ? h(Why, { why: t.why }) : null,
      h("div", { className: "sy-chips" }, t.stages.map(function (s, i) {
        return h(StageChip, { key: i, stage: s });
      })));
  }

  // How often each rule sent recent VGR turns to the cloud.
  function EscalationsCard(props) {
    const groups = {};
    let escalated = 0;
    props.turns.forEach(function (t) {
      if (!t.why || t.why.served !== "cloud") return;
      escalated += 1;
      const g = groups[t.why.headline] || (groups[t.why.headline] = { count: 0, summary: t.why.summary, last: t.id });
      g.count += 1;
    });
    const rows = Object.keys(groups).sort(function (a, b) { return groups[b].count - groups[a].count; });
    const vgrTurns = props.turns.filter(function (t) { return t.why; }).length;
    return h(Card, null,
      h(CardHeader, null, h(CardTitle, null, "Why turns went to the cloud")),
      h(CardContent, null,
        !escalated
          ? h("div", { className: "sy-muted" }, vgrTurns ? "No recent VGR turn was escalated." : "No VGR turns yet.")
          : h("div", null,
            h("div", { className: "sy-muted" }, escalated + " of the last " + vgrTurns + " VGR turns went to the cloud."),
            h("div", { className: "sy-reasons" }, rows.map(function (name) {
              const g = groups[name];
              return h("div", { key: name, className: "sy-reason", title: g.summary },
                h("div", { className: "sy-reason-bar" },
                  h("div", { className: "sy-reason-fill", style: { width: Math.round((100 * g.count) / escalated) + "%" } })),
                h("span", { className: "sy-reason-count" }, g.count),
                h("span", null, name),
                h("span", { className: "sy-muted" }, "latest #" + g.last));
            })),
            h("div", { className: "sy-muted" }, "Hover a reason for its explanation; each decision below has the details."))));
  }

  // Smart routing, or every request pinned to one tier; applies to the running sidecar at once.
  function PinControl(props) {
    const config = props.config;
    const [busy, setBusy] = useState(false);
    const options = [
      ["", "Smart routing", "Route with the selected preset"],
      ["local", "Always " + config.local.model, "Every request goes to the " + config.local.role + " model"],
      ["cloud", "Always " + config.cloud.model, "Every request goes to the " + config.cloud.role + " model"],
    ];
    const choose = function (pin) {
      setBusy(true);
      SDK.fetchJSON(API + "/pin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pin: pin }),
      })
        .then(props.onChange)
        .catch(function (e) { props.onError(String(e && e.message ? e.message : e)); })
        .finally(function () { setBusy(false); });
    };
    return h("div", { className: "sy-segs sy-pin" }, options.map(function (o) {
      return h("button", {
        key: o[0], type: "button", title: o[2], disabled: busy,
        className: "sy-seg" + (config.pin === o[0] ? " sy-seg-on" : ""),
        onClick: function () { if (config.pin !== o[0]) choose(o[0]); },
      }, o[1]);
    }));
  }

  function tierLine(tier, name) {
    return h("span", { title: tier.role + " model · " + tier.endpoint + " · " + tier.format },
      h(TierPill, { tier: name }), " " + tier.model, h("span", { className: "sy-muted" }, " · " + tier.endpoint));
  }

  function SwitchyardPage() {
    const [data, setData] = useState(null);
    const [preset, setPreset] = useState("");
    // One switch for the totals and the timeline, so the dials and the bars count the same calls.
    const [background, setBackground] = useState(false);
    const [error, setError] = useState(null);
    const [starting, setStarting] = useState(false);
    const timer = useRef(null);

    const load = useCallback(function () {
      return SDK.fetchJSON(API + "/state")
        .then(function (d) { setData(d); setError(null); })
        .catch(function (e) { setError(String(e && e.message ? e.message : e)); });
    }, []);

    useEffect(function () {
      let alive = true;
      function tick() {
        if (!alive) return;
        const next = function () { if (alive) timer.current = setTimeout(tick, POLL_MS); };
        if (document.hidden) { next(); return; }
        load().finally(next);
      }
      tick();
      return function () { alive = false; clearTimeout(timer.current); };
    }, [load]);

    const start = function () {
      setStarting(true);
      SDK.fetchJSON(API + "/start", { method: "POST" })
        .then(function (r) { if (!r.ok) setError(r.error || "Switchyard sidecar failed to start"); return load(); })
        .catch(function (e) { setError(String(e)); })
        .finally(function () { setStarting(false); });
    };

    const header = h(Card, null,
      h(CardHeader, null, h(CardTitle, null, "Switchyard Local-first Routing ", h(Badge, null, "Beta"))),
      h(CardContent, null,
        h("div", { className: "sy-muted" }, "In beta: routing and settings may change between releases."),
        !data ? h("div", { className: "sy-muted" }, error || "Loading...") :
        !data.configured ? h("div", null,
          h("div", null, "Switchyard is not configured: " + data.error),
          h("div", { className: "sy-muted" }, "Run: hermes switchyard setup --local-url ... --local-model ... --cloud-model ...")) :
        h("div", { className: "sy-config" },
          h("span", { className: "sy-status sy-status-" + (data.running ? "up" : "down") }, data.running ? "Running" : "Stopped"),
          h(Badge, null, "default " + data.config.default_preset),
          data.selected_model ? h("span", { className: "sy-muted" }, "Hermes uses " + data.selected_model) : null,
          tierLine(data.config.local, "local"),
          tierLine(data.config.cloud, "cloud"),
          h("span", { className: "sy-muted" }, data.config.endpoint),
          data.state && data.state.runtime && data.state.runtime.version
            ? h("span", { className: "sy-muted", title: data.state.runtime.location },
              "Switchyard " + data.state.runtime.version)
            : null,
          data.running ? null : h(Button, { onClick: start, disabled: starting }, starting ? "Starting..." : "Start sidecar")),
        data && data.configured ? h(PinControl, { config: data.config, onChange: load, onError: setError }) : null,
        data && data.configured && data.config.pin
          ? h("div", { className: "sy-pinned" }, "Pinned: every request goes to " + data.config[data.config.pin].model +
            "; routing decisions and the local share below reflect the pin, not the preset.")
          : null,
        data && data.configured && data.state ? ["local", "cloud"].map(function (tier) {
          const alert = (data.state.credit_alerts || {})[tier];
          if (!alert) return null;
          const role = data.config[tier].role, other = data.config[tier === "local" ? "cloud" : "local"].role;
          return h("div", { key: "credits-" + tier, className: "sy-error" },
            role.charAt(0).toUpperCase() + role.slice(1) + " model out of credits: " + data.config[tier].model +
            " refused its last call, so its turns fall back to the " + other + " model. Add credits or set another " +
            role + " model with hermes switchyard setup. Its provider said: " + alert.message);
        }) : null,
        data && data.configured ? (data.config.warnings || []).map(function (w, i) {
          return h("div", { key: i, className: "sy-warning" }, w);
        }) : null,
        error && data ? h("div", { className: "sy-error" }, error) : null));

    const state = data && data.state;
    if (!state) return h("div", { className: "sy-page" }, header);
    const allTurns = state.turns || [];
    const turns = preset ? allTurns.filter(function (t) { return t.preset === preset; }) : allTurns;
    const view = Object.assign({}, state, {
      totals: presetTotals(state, preset, background),
      timeline: preset ? (state.timeline || []).filter(function (s) { return s.preset === preset; }) : state.timeline,
    });
    return h("div", { className: "sy-page" },
      header,
      h(PresetsCard, { state: state, config: data.config, selected: preset, onSelect: setPreset }),
      h("div", { className: "sy-grid2" },
        h(NowCard, { current: state.current, last: allTurns.find(function (t) { return t.served_tier; }) }),
        h(TokensCard, { state: view, preset: preset })),
      h(SplitCard, { state: view, background: background }),
      h(TimelineCard, { state: view, background: background, onBackground: setBackground }),
      h(EscalationsCard, { turns: turns }),
      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Recent decisions" + (preset ? " · " + preset : ""))),
        h(CardContent, null, turns.length
          ? turns.map(function (t) { return h(TurnRow, { key: t.id, turn: t, now: state.now }); })
          : h("div", { className: "sy-muted" }, "No routed turns yet. Chat with Hermes to see decisions here."))));
  }

  window.__HERMES_PLUGINS__.register("nvidia-switchyard", SwitchyardPage);
})();

"use strict";

// All plots use saved complex solver samples. Animation advances their index.
const el = id => document.getElementById(id);
const state = { cases: [], caseId: null, data: null, frame: 0, amplitudes: [], peak: 0, playing: false, timer: null, job: null, loadToken: 0, customInitialized: false, preview: null, previewKey: null, previewToken: 0, previewPending: false };
const caseLabels = { profile: ["Solitary profile", "VALIDATED REFERENCE"], perturbed: ["Modest perturbation", "LOCALIZED PERTURBATION"], gaussian: ["Gaussian packet", "COMPLEX INITIAL ENVELOPE"], custom: ["Custom formula", "LATEX INITIAL ENVELOPE"] };
const formulaExamples = {
  gaussian: String.raw`A \exp(-\frac{\tau^2}{2 w^2}) \exp(i(k\tau+c\tau^2))`,
  sech: String.raw`A \sech(\tau/w) \exp(i k \tau)`
};
const glyphs = {
  profile: "M1 26 C11 26 12 25 17 7 C20 -2 23 7 26 19 C28 26 33 26 39 26",
  perturbed: "M1 26 C11 26 12 25 17 7 C20 -2 23 7 26 19 C28 27 29 18 32 23 C34 30 36 24 39 26",
  gaussian: "M1 26 C9 26 11 24 14 17 C17 6 20 2 23 12 C27 23 29 26 39 26",
  custom: "M1 25 C8 25 9 8 14 8 C19 8 18 25 24 25 C30 25 31 12 39 12 M2 3 L10 3 M6 0 L6 6"
};

function text(id, value) { el(id).textContent = value; }
function make(tag, className, content) { const node = document.createElement(tag); if (className) node.className = className; if (content !== undefined) node.textContent = content; return node; }
function finite(value) { return typeof value === "number" && Number.isFinite(value); }
function number(value, digits = 4) { if (!finite(value)) return "—"; if (value === 0) return "0"; const a = Math.abs(value); return a < .0001 || a >= 1e5 ? value.toExponential(2) : Number(value.toPrecision(digits)).toString(); }
function percent(value) { return finite(value) ? `${number(value * 100, 3)}%` : "—"; }
function showError(message) { const node = el("global-error"); node.hidden = !message; node.textContent = message || ""; }
async function request(url, options) {
  const response = await fetch(url, { cache: "no-store", ...options });
  let payload; try { payload = await response.json(); } catch (_) { throw new Error(`The local server returned an unreadable response (${response.status}).`); }
  if (!response.ok) throw new Error(payload.error || payload.message || `Request failed (${response.status}).`);
  return payload;
}
function parameterInput(name, label, fallback, min, max, step) {
  const parameters = currentCase()?.parameters || {};
  const spec = currentCase()?.parameter_fields?.find(field => field.name === name) || {};
  const wrap = make("label", "", label);
  const input = make("input"); input.type = "number"; input.name = name; input.value = parameters[name] ?? spec.default ?? fallback;
  input.min = spec.minimum ?? min; input.max = spec.maximum ?? max; input.step = spec.step ?? step; input.required = true; input.id = `param-${name}`;
  input.title = `Allowed range: ${input.min} to ${input.max}`;
  wrap.append(input); return wrap;
}
function currentCase() { return state.cases.find(item => item.id === state.caseId); }
function renderCases() {
  const list = el("case-list"); list.replaceChildren();
  for (const item of state.cases) {
    const button = make("button", "case-button"); button.type = "button"; button.dataset.caseId = item.id;
    button.setAttribute("aria-pressed", String(state.caseId === item.id));
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); svg.setAttribute("viewBox", "0 0 40 30"); svg.setAttribute("class", "case-glyph"); svg.setAttribute("aria-hidden", "true");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path"); path.setAttribute("d", glyphs[item.id] || glyphs.gaussian); svg.append(path);
    const label = make("span"); label.append(make("strong", "", caseLabels[item.id]?.[0] || item.title || item.id), make("small", "", caseLabels[item.id]?.[1] || "REFERENCE CASE"));
    button.append(svg, label); button.addEventListener("click", () => selectCase(item.id)); list.append(button);
  }
}
function renderParameterFields() {
  const node = el("parameter-fields"); node.replaceChildren();
  const item = currentCase(); if (!item) return;
  const custom = item.id === "custom";
  el("custom-fields").hidden = !custom; el("custom-preview").hidden = !custom;
  el("load-reference").hidden = custom;
  el("parameter-notice").hidden = item.id === "profile" || custom;
  text("run-help", custom ? "Run uses the exact formula, parameters and grid from your latest successful preview. Every edit requires a new preview. The server retains both complex channels and formula provenance in the download." : "Reference playback reads saved numerical data. “Run locally” executes the solver with the parameters above; playback itself performs no simulation.");
  if (custom) {
    if (!state.customInitialized) {
      const params = item.parameters || {};
      el("custom-formula").value = params.formula || formulaExamples.gaussian;
      el("custom-symbols").value = Object.entries(params.symbols || { A: 1, w: 2, k: .3, c: 0 }).map(([name, value]) => `${name}=${value}`).join(", ");
      el("custom-horizon").value = item.settings?.xi_end ?? .1;
      el("custom-grid").value = String(item.settings?.n ?? 1024);
      state.customInitialized = true;
    }
    updateRunAvailability(); return;
  }
  if (item.id === "gaussian") {
    const grid = make("div", "parameter-grid");
    grid.append(parameterInput("amplitude", "Amplitude", 1, .5, 1.5, .05), parameterInput("width", "Width parameter", 2, 1.5, 4, .1), parameterInput("wavenumber", "Phase slope", .3, -.5, .5, .05), parameterInput("chirp", "Chirp", 0, -.1, .1, .01)); node.append(grid);
  } else if (item.id === "perturbed") {
    const grid = make("div", "parameter-grid"); const field = parameterInput("delta", "Perturbation δ", .02, 0, .04, .005); field.className = "full"; grid.append(field); node.append(grid);
  } else {
    const params = item.parameters || {}; const fixed = make("div", "fixed-parameters"); fixed.append(make("span", "", `ε = ${number(params.epsilon ?? .025)}`), make("span", "", `μ = ${number(params.mu ?? 2)}`), make("span", "", `s = ${number(params.speed ?? params.s ?? 0)}`)); node.append(fixed);
  }
}
async function selectCase(id) {
  if (state.caseId === "custom" && id !== "custom") invalidatePreview("Preview again when you return to Custom formula.");
  ++state.loadToken; state.caseId = id; renderCases(); renderParameterFields(); showError("");
  const item = currentCase(); text("case-description", item.description || "Review the initial packet and its short propagation trajectory.");
  updateRunAvailability(); el("load-reference").disabled = item.precomputed_available === false;
  if (item.precomputed_available !== false) await loadReference();
  else {
    pause(); state.data = null; el("viewer-content").hidden = true; el("viewer-empty").hidden = false;
    text("source-badge", id === "custom" ? "CUSTOM · PREVIEW THEN RUN" : "NO REFERENCE AVAILABLE"); el("source-badge").classList.remove("live");
    el("viewer-empty").querySelector("h3").textContent = id === "custom" ? "Preview, then propagate." : "A packet, then its evolution.";
    el("viewer-empty").querySelector("p").textContent = id === "custom" ? "Review the initial envelope above, then use Run locally on CPU." : "Select a case to load its reference snapshots.";
    renderSetup(); clearQualification();
  }
}
async function loadReference() {
  const token = ++state.loadToken; const id = state.caseId; pause(); showError("");
  el("load-reference").disabled = true; text("source-badge", "LOADING REFERENCE…");
  try { const data = await request(`/api/data?case=${encodeURIComponent(id)}`); if (token === state.loadToken) loadData(data, "precomputed", `/api/download?case=${encodeURIComponent(id)}`); }
  catch (error) { if (token === state.loadToken) { showError(`Reference could not be loaded: ${error.message}`); text("source-badge", "LOAD FAILED"); } }
  finally { if (token === state.loadToken) el("load-reference").disabled = false; }
}
function normalizeData(data) {
  if (data.data && !data.tau) data = data.data;
  if (!data.real && data.frames) { data.real = data.frames.real; data.imag = data.frames.imag; }
  if (!Array.isArray(data.tau) || !Array.isArray(data.xi) || !Array.isArray(data.real) || !Array.isArray(data.imag) || !data.xi.length || !data.tau.length) throw new Error("Saved data does not contain the expected τ, ξ and complex envelope arrays.");
  if (data.real.length !== data.xi.length || data.imag.length !== data.xi.length || data.real.some(row => row.length !== data.tau.length) || data.imag.some(row => row.length !== data.tau.length)) throw new Error("Saved complex-array dimensions are inconsistent.");
  if (Array.isArray(data.diagnostics)) { const rows = data.diagnostics; data.diagnostics = {}; const names = { peak: "peak_amplitude", width_rms: "rms_width", tail_action_fraction: "tail_action_fraction" }; for (const key of Object.keys(rows[0] || {})) data.diagnostics[names[key] || key] = rows.map(row => row[key]); }
  return data;
}
function loadData(raw, source, downloadUrl) {
  const data = normalizeData(raw); pause(); state.data = data; state.frame = 0;
  state.amplitudes = data.real.map((row, j) => row.map((r, i) => Math.hypot(r, data.imag[j][i]))); state.peak = 0;
  for (const row of state.amplitudes) for (const a of row) if (a > state.peak) state.peak = a;
  if (data.case_id === "custom" && (!finite(diagnostic("rms_width", 0)) || finite(diagnostic("circular_concentration", 0)) && diagnostic("circular_concentration", 0) < .5)) el("view-range").value = "domain";
  el("viewer-empty").hidden = true; el("viewer-content").hidden = false;
  text("result-title", data.title || data.label || caseLabels[data.case_id]?.[0] || "Envelope trajectory");
  text("result-description", `Scaled propagation distance ξ ∈ [${number(data.xi[0])}, ${number(data.xi.at(-1))}] · ${data.xi.length} saved snapshots`);
  const labels = { amplitude: "A", width: "width", wavenumber: "phase slope", chirp: "chirp", delta: "δ" };
  text("result-parameters", Object.entries(data.parameters || {}).map(([key, value]) => `${labels[key] || key} = ${readable(value)}`).join(" · "));
  text("source-badge", data.case_id === "custom" ? "CUSTOM · EXPLORATORY PLAYBACK" : source === "live" ? "LIVE LOCAL RESULT · PLAYBACK" : "PRECOMPUTED · PLAYBACK"); el("source-badge").classList.toggle("live", source === "live");
  el("snapshot").max = data.xi.length - 1; el("snapshot").value = "0";
  el("download-data").href = downloadUrl; el("download-data").setAttribute("download", "");
  const n = data.settings?.n ?? data.settings?.N ?? data.tau.length;
  text("data-resolution", `${data.tau.length} displayed τ samples · ${number(n, 6)} solver points · full complex arrays in .npz`);
  renderSetup(); renderQualification(); renderDefinitions(); renderFrame();
}
function renderSetup() {
  const data = state.data || {}; const s = data.settings || (state.caseId === "custom" ? customSettings() : currentCase()?.settings) || {}; const node = el("setup-values"); node.replaceChildren();
  const params = data.parameters || {}; const rows = [["Model", "Spatial water-wave Dysthe"], ["ε", number(params.epsilon ?? s.epsilon ?? data.model?.epsilon ?? .025)], ["Periodic τ length", number(s.length ?? s.L)], ["Solver grid N", number(s.n ?? s.N, 6)], ["Propagation step Δξ", number(s.dxi ?? s.h)], ["Propagation interval", `0 → ${number(data.xi?.at(-1) ?? s.xi_end)}`], ["Saved snapshots", data.xi ? String(data.xi.length) : `up to ${s.samples || 21}`]];
  for (const [label, value] of rows) { const row = make("div"); row.append(make("dt", "", label), make("dd", "", value)); node.append(row); }
}
function readable(value) {
  if (value == null) return "—"; if (typeof value === "number") return number(value);
  if (Array.isArray(value)) return value.map(readable).join(" · ");
  if (typeof value === "object") return Object.entries(value).map(([k, v]) => `${k.replaceAll("_", " ")}: ${readable(v)}`).join("; ");
  return String(value);
}
function clearQualification() {
  text("qualification-badge", "NO RESULT · UNQUALIFIED");
  text("qualification-summary", "No propagation result is loaded. Custom input remains exploratory; preview screens do not provide grid, step or domain convergence evidence.");
  el("qualification-table").replaceChildren(); text("qualification-notes", "Qualification for previous reference cases does not apply to this input.");
  el("result-input-warnings").replaceChildren(); el("result-input-warnings").hidden = true;
  for (const id of ["stat-mass", "stat-boundary", "stat-phase", "stat-shape"]) text(id, "—");
}
function renderQualification() {
  const q = state.data.qualification || {}; const status = String(q.status || "unqualified");
  text("qualification-badge", q.passed === true ? "CHECKS PASSED" : q.passed === false ? "CHECKS FAILED" : "UNQUALIFIED");
  text("qualification-summary", q.summary || q.description || (status.toLowerCase().includes("unqualified") ? "This particular run has no completed grid, step and domain comparison. Treat it as exploratory, even when the conservation trace is small." : "Qualification applies to this case, its listed parameters and propagation interval. The table reports the largest discrepancy over all saved snapshots."));
  let checks = q.checks || q.comparisons || q.evidence || [];
  if (!Array.isArray(checks)) checks = Object.entries(checks).map(([kind, value]) => ({ kind, ...(typeof value === "object" ? value : { result: value }) }));
  const container = el("qualification-table"); container.replaceChildren();
  if (checks.length) {
    const table = make("table", "qualification-table"); const header = make("thead"); const headrow = make("tr"); for (const name of ["Check", "Comparison", "Field Δ / field", "Shape Δ / response"]) headrow.append(make("th", "", name)); header.append(headrow); table.append(header); const body = make("tbody");
    for (const check of checks) {
      const row = make("tr"), kind = check.kind || check.name || check.check || "Check";
      const error = check.error ?? check.relative_error ?? check.rel_l2 ?? check.difference ?? check.result;
      let comparison = check.comparison ?? check.settings ?? check.description ?? check.refinement ?? "See saved report";
      // The bounded campaign independently doubles N, halves Δξ, and doubles L at fixed Δτ.
      if (q.exact_default_settings && String(comparison).includes("→")) {
        const s = q.exact_default_settings;
        if (kind === "grid") comparison = `N: ${s.n} → ${s.n * 2}`;
        if (kind === "step") comparison = `Δξ: ${number(s.dxi)} → ${number(s.dxi / 2)}`;
        if (kind === "domain") comparison = `L: ${s.length} → ${s.length * 2}; N: ${s.n} → ${s.n * 2}`;
      }
      const checkLabels = { grid: "Retarded-time grid (τ)", step: "Propagation step (ξ)", domain: "Retarded-time window (τ)" };
      row.append(make("td", "", checkLabels[kind] || readable(kind)), make("td", "", readable(comparison)), make("td", "", readable(error)), make("td", "", readable(check.metrics?.shape_discrepancy_over_response))); body.append(row);
    }
    table.append(body); container.append(table);
    container.append(make("p", "helper", "N refines the retarded-time grid τ; Δξ refines propagation accuracy downstream; L enlarges the periodic retarded-time window at fixed τ spacing. Values are relative L² differences. Shape changes are compared after fitting translation and global phase; no amplitude rescaling. The response ratio is omitted (—) when the shape response is below 10⁻⁷ of the field norm."));
  }
  const notes = q.notes || q.caveats || []; text("qualification-notes", Array.isArray(notes) ? notes.join("\n") : readable(notes));
  const notesText = Array.isArray(notes) ? notes.join("\n") : readable(notes);
  const warningNode = el("result-input-warnings"), warnings = (state.data.initial_provenance?.warnings || []).filter(warning => !notesText.includes(typeof warning === "string" ? warning : warning.message || readable(warning)));
  warningNode.replaceChildren(); warningNode.hidden = state.data.case_id !== "custom" || !warnings.length;
  if (!warningNode.hidden) {
    warningNode.append(make("h3", "", "Initial-input warnings for this saved result")); const list = make("ul");
    for (const warning of warnings) list.append(make("li", "", typeof warning === "string" ? warning : warning.message || readable(warning)));
    warningNode.append(list);
  }
}
function renderDefinitions() {
  const definitions = state.data.diagnostic_definitions; if (!definitions || typeof definitions !== "object") return;
  const node = el("diagnostic-definitions"); node.replaceChildren();
  const labels = { peak_amplitude: "Peak amplitude", rms_width: "RMS width", displacement: "Displacement", tail_action_fraction: "Exterior action", boundary_action_fraction: "Boundary action", mass_relative_error: "Relative action drift", orbital_shape_error: "Aligned shape change", phase_shift: "Phase shift" };
  for (const [key, value] of Object.entries(definitions)) { const row = make("div"); row.append(make("dt", "", labels[key] || key.replaceAll("_", " ")), make("dd", "", readable(value))); node.append(row); }
  const caution = make("div"); caution.append(make("dt", "", "Interpretation limits"), make("dd", "", "Exterior action includes initial tails and dispersive transport; it is a radiation proxy, not a radiation measurement. Qualifying one input does not qualify all inputs. Small action drift and short persistence do not establish numerical accuracy or stability.")); node.append(caution);
}
function diagnostic(name, frame = state.frame) { return state.data?.diagnostics?.[name]?.[frame]; }
function change(name) { const first = diagnostic(name, 0), now = diagnostic(name); return finite(first) && first !== 0 && finite(now) ? `${percent((now - first) / first)} from initial` : "relative to initial packet"; }
function renderFrame() {
  if (!state.data) return;
  const i = state.frame; el("snapshot").value = i; text("snapshot-number", `${i + 1} / ${state.data.xi.length}`); text("xi-value", `ξ = ${number(state.data.xi[i])} · scaled`);
  text("stat-peak", number(diagnostic("peak_amplitude") ?? Math.max(...state.amplitudes[i]))); text("stat-peak-change", change("peak_amplitude"));
  text("stat-width", number(diagnostic("rms_width"))); text("stat-width-change", change("rms_width"));
  text("stat-displacement", number(diagnostic("displacement"))); text("stat-tail", percent(diagnostic("tail_action_fraction")));
  text("stat-mass", number(diagnostic("mass_relative_error"))); text("stat-boundary", percent(diagnostic("boundary_action_fraction"))); text("stat-shape", number(diagnostic("orbital_shape_error"))); text("stat-phase", `${number(diagnostic("phase_shift"))} rad`);
  const phase = el("field-mode").value === "phase"; el("phase-note").hidden = !phase;
  const legend = el("line-legend"); legend.replaceChildren();
  for (const [className, label] of (el("field-mode").value === "complex" ? [["real", "Current Re(A)"], ["imag", "Current Im(A)"], ["dashed", "Initial components (light)"]] : [["dashed", "Initial"], ["", "Current snapshot"]])) { const item = make("span"); item.append(make("i", `line-key ${className}`), document.createTextNode(label)); legend.append(item); }
  drawEnvelope(); drawHeatmap();
}
function canvasContext(id) {
  const canvas = el(id), rect = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (!rect.width || !rect.height) return null;
  const width = Math.round(rect.width), height = Math.round(rect.height);
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, width, height);
  ctx.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace'; ctx.fillStyle = "#595959"; ctx.lineWidth = 1;
  return { ctx, width, height };
}
function plotBounds(width, height, heatmap = false) { return { left: 55, right: width - 18, top: heatmap ? 10 : 14, bottom: height - 38 }; }
function tauBounds() {
  const tau = state.data.tau; if (el("view-range").value === "domain") return [tau[0], tau.at(-1)];
  const row = state.amplitudes[0]; let action = 0, moment = 0; for (let i = 0; i < tau.length; i++) { action += row[i] ** 2; moment += tau[i] * row[i] ** 2; }
  const center = action > 0 ? moment / action : 0, half = Math.max(12, 4 * (diagnostic("rms_width", 0) || 2));
  return [Math.max(tau[0], center - half), Math.min(tau.at(-1), center + half)];
}
function axes(ctx, box, xmin, xmax, ymin, ymax, ylabel, fill = false) {
  const { left:l, right:r, top:t, bottom:b } = box;
  ctx.save(); ctx.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace'; ctx.fillStyle = "#595959";
  for (let n = 0; n <= 4; n++) {
    const y = t + (b - t) * n / 4; ctx.strokeStyle = "#deded9"; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(r, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(number(ymax - (ymax - ymin) * n / 4, 3), l - 9, y);
  }
  for (let n = 0; n <= 4; n++) {
    const x = l + (r - l) * n / 4; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(number(xmin + (xmax - xmin) * n / 4, 3), x, b + 8);
    if (!fill) { ctx.strokeStyle = "#e8e8e3"; ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, b); ctx.stroke(); }
  }
  ctx.strokeStyle = "#9b9b95"; ctx.beginPath(); ctx.moveTo(l, t); ctx.lineTo(l, b); ctx.lineTo(r, b); ctx.stroke();
  ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.font = `${r - l < 300 ? 9 : 10}px ui-monospace, SFMono-Regular, Consolas, monospace`; ctx.fillText("Retarded time τ (moving frame)", (l + r) / 2, b + 36);
  ctx.translate(13, (t + b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(ylabel, 0, 0); ctx.restore();
}
function drawEnvelope() {
  const surface = canvasContext("envelope-chart"); if (!surface || !state.data) return;
  const { ctx, width, height } = surface, box = plotBounds(width, height), tau = state.data.tau, [xmin, xmax] = tauBounds(), mode = el("field-mode").value;
  let ymin = 0, ymax = 0; const visible = tau.map((x, i) => x >= xmin && x <= xmax ? i : -1).filter(i => i >= 0);
  for (const i of visible) ymax = Math.max(ymax, state.amplitudes[0][i], state.amplitudes[state.frame][i]);
  ymax = Math.max(ymax * 1.1, 1e-8); if (mode === "complex") ymin = -ymax; if (mode === "phase") { ymin = -Math.PI; ymax = Math.PI; }
  axes(ctx, box, xmin, xmax, ymin, ymax, mode === "amplitude" ? "Envelope |A|" : mode === "phase" ? "arg(A) · radians" : "Re(A), Im(A)");
  const sx = x => box.left + (x - xmin) / (xmax - xmin) * (box.right - box.left), sy = y => box.bottom - (y - ymin) / (ymax - ymin) * (box.bottom - box.top);
  function line(frame, component, color, dash) {
    ctx.save(); ctx.beginPath(); ctx.rect(box.left, box.top, box.right - box.left, box.bottom - box.top); ctx.clip(); ctx.strokeStyle = color; ctx.lineWidth = frame === state.frame ? 1.8 : 1.3; ctx.setLineDash(dash); let drawing = false, previous = null;
    const threshold = .01 * Math.max(...state.amplitudes[frame]); ctx.beginPath();
    for (const i of visible) {
      const value = component === "abs" ? state.amplitudes[frame][i] : component === "phase" ? Math.atan2(state.data.imag[frame][i], state.data.real[frame][i]) : state.data[component][frame][i];
      if (!finite(value) || (component === "phase" && (state.amplitudes[frame][i] < threshold || (previous !== null && Math.abs(value - previous) > Math.PI)))) { drawing = false; previous = value; continue; }
      if (drawing) ctx.lineTo(sx(tau[i]), sy(value)); else { ctx.moveTo(sx(tau[i]), sy(value)); drawing = true; } previous = value;
    }
    ctx.stroke(); ctx.restore();
  }
  if (mode === "complex") { line(0, "real", "#aaa9a2", []); line(0, "imag", "#aaa9a2", [4, 4]); line(state.frame, "real", "#202020", []); line(state.frame, "imag", "#333333", [5, 4]); }
  else { const component = mode === "phase" ? "phase" : "abs"; line(0, component, "#9b9b94", [5, 4]); line(state.frame, component, "#202020", []); }
  el("envelope-chart").setAttribute("aria-label", `${mode === "amplitude" ? "Envelope modulus |A|" : mode === "complex" ? "Real and imaginary envelope components" : "Masked principal envelope phase"}, initial versus scaled propagation distance ξ=${number(state.data.xi[state.frame])}. Horizontal axis: retarded time τ in the moving frame, range ${number(xmin)} to ${number(xmax)}. No seconds or meters conversion is configured.`);
}
function drawHeatmap() {
  const surface = canvasContext("heatmap-chart"); if (!surface || !state.data) return;
  const { ctx, width, height } = surface, box = plotBounds(width, height, true), [xmin, xmax] = tauBounds(), tau = state.data.tau, xi = state.data.xi, log = el("heatmap-scale").value === "log";
  const indices = tau.map((x, i) => x >= xmin && x <= xmax ? i : -1).filter(i => i >= 0);
  const off = document.createElement("canvas"); off.width = indices.length; off.height = xi.length; const offctx = off.getContext("2d"), pixels = offctx.createImageData(off.width, off.height);
  for (let j = 0; j < xi.length; j++) for (let k = 0; k < indices.length; k++) { const ratio = state.amplitudes[j][indices[k]] / (state.peak || 1); const value = log ? Math.max(0, 1 + Math.log10(Math.max(ratio, 1e-5)) / 5) : ratio; const gray = Math.round(239 - 207 * Math.max(0, Math.min(1, value))); const at = (j * indices.length + k) * 4; pixels.data[at] = gray; pixels.data[at + 1] = gray; pixels.data[at + 2] = gray - 3; pixels.data[at + 3] = 255; }
  offctx.putImageData(pixels, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, box.left, box.top, box.right - box.left, box.bottom - box.top);
  // ξ increases downward, matching the stored trajectory from the first frame.
  const l = box.left, r = box.right, t = box.top, b = box.bottom;
  ctx.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace'; ctx.fillStyle = "#595959"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let n = 0; n <= 4; n++) ctx.fillText(number(xi[0] + (xi.at(-1) - xi[0]) * n / 4, 3), l - 9, t + (b - t) * n / 4);
  ctx.textAlign = "center"; ctx.textBaseline = "top"; for (let n = 0; n <= 4; n++) ctx.fillText(number(xmin + (xmax - xmin) * n / 4, 3), l + (r - l) * n / 4, b + 8);
  ctx.textBaseline = "bottom"; ctx.font = `${r - l < 300 ? 9 : 10}px ui-monospace, SFMono-Regular, Consolas, monospace`; ctx.fillText("Retarded time τ (moving frame)", (l + r) / 2, b + 36); ctx.save(); ctx.translate(13, (t + b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Propagation distance ξ", 0, 0); ctx.restore();
  const cursor = t + (b - t) * state.frame / Math.max(1, xi.length - 1); ctx.strokeStyle = "#fff"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(l, cursor); ctx.lineTo(r, cursor); ctx.stroke(); ctx.strokeStyle = "#222"; ctx.lineWidth = 1; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = "#99998f"; ctx.strokeRect(l, t, r - l, b - t);
  text("heatmap-min", log ? "10⁻⁵" : "0"); text("heatmap-max", log ? "1 × peak" : number(state.peak, 3));
  el("heatmap-chart").setAttribute("aria-label", `Envelope amplitude |A(ξ,τ)|. Horizontal axis: retarded time τ in the moving frame. Vertical axis: scaled propagation distance ξ, increasing downward from ${number(xi[0])} to ${number(xi.at(-1))}. ${log ? "Log amplitude contrast clipped at one hundred thousandth of the global peak." : "Linear amplitude scale."} Dashed line marks the selected downstream station.`);
}
function pause() { state.playing = false; clearTimeout(state.timer); state.timer = null; text("play-pause", "▶"); el("play-pause").setAttribute("aria-label", "Play saved snapshots"); }
function play() {
  if (!state.data) return; if (state.playing) return pause();
  state.playing = true; text("play-pause", "Ⅱ"); el("play-pause").setAttribute("aria-label", "Pause saved snapshots");
  function advance() { if (!state.playing) return; state.frame = state.frame >= state.data.xi.length - 1 ? 0 : state.frame + 1; renderFrame(); state.timer = setTimeout(advance, Number(el("play-speed").value)); }
  state.timer = setTimeout(advance, Number(el("play-speed").value));
}
function customSettings() {
  return { n: Number(el("custom-grid").value), length: 160, dxi: .00125, xi_end: Number(el("custom-horizon").value), samples: 21 };
}
function customRequest() {
  const formula = el("custom-formula").value.trim();
  if (!formula) throw new Error("Enter an expression for the initial envelope.");
  if (formula.length > 1024) throw new Error("The formula is too long; use at most 1024 characters.");
  const symbols = {}, declaration = el("custom-symbols").value.trim();
  if (declaration.length > 512) throw new Error("The parameter declaration is too long.");
  if (declaration) {
    const entries = declaration.split(/[,;\n]/).map(value => value.trim()).filter(Boolean);
    if (entries.length > 8) throw new Error("Declare no more than 8 scalar parameters.");
    for (const entry of entries) {
      const match = /^([A-Za-z])\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$/.exec(entry);
      if (!match) throw new Error(`Use single-letter real parameters, for example A=1, w=2. Check “${entry}”.`);
      const name = match[1], value = Number(match[2]);
      if (name === "i") throw new Error("i is the imaginary unit and cannot be a parameter.");
      if (Object.hasOwn(symbols, name)) throw new Error(`Parameter ${name} is declared more than once.`);
      if (!Number.isFinite(value)) throw new Error(`Parameter ${name} must be finite.`);
      symbols[name] = value;
    }
  }
  const settings = customSettings();
  if (!Number.isFinite(settings.xi_end) || settings.xi_end < .00125 || settings.xi_end > 4 || Math.abs(settings.xi_end / settings.dxi - Math.round(settings.xi_end / settings.dxi)) > 1e-7) throw new Error("Propagation end ξ must be a multiple of 0.00125 between 0.00125 and 4.");
  if (![1024, 4096].includes(settings.n)) throw new Error("Choose a supported solver grid.");
  if (settings.n * Math.round(settings.xi_end / settings.dxi) > 8000000) throw new Error("This grid and propagation interval exceed the bounded local work limit. At N=4096, shorten ξ to 2.44125 or less.");
  return { case_id: "custom", parameters: { formula, symbols }, settings };
}
function showCustomError(message) {
  el("custom-error").hidden = !message; text("custom-error", message || "");
  el("custom-formula").setAttribute("aria-invalid", String(Boolean(message)));
}
function updateRunAvailability() {
  let valid = true;
  if (state.caseId === "custom") {
    try { valid = Boolean(state.preview?.can_run && state.preview?.fingerprint && state.previewKey === JSON.stringify(customRequest()) && !state.previewPending); }
    catch (_) { valid = false; }
  }
  el("run-live").disabled = Boolean(state.job) || !valid;
  el("preview-custom").disabled = state.previewPending || Boolean(state.job);
}
function invalidatePreview(message = "Input changed. Preview again before Run.") {
  ++state.previewToken; state.preview = null; state.previewKey = null; state.previewPending = false;
  showCustomError(""); el("custom-preview-plots").hidden = true; el("custom-warnings").replaceChildren();
  text("custom-preview-status", message); text("preview-badge", "PREVIEW REQUIRED");
  text("custom-preview-intro", "Preview the formula on the chosen periodic grid, inspect its amplitude and phase, then start a short exploratory run.");
  updateRunAvailability(); if (state.caseId === "custom" && !state.data) renderSetup();
}
function previewArrays(data) {
  const tau = data.tau, real = data.real, imag = data.imag;
  if (!Array.isArray(tau) || !Array.isArray(real) || !Array.isArray(imag) || tau.length < 2 || tau.length > 16384 || real.length !== tau.length || imag.length !== tau.length || !tau.every(finite) || !real.every(finite) || !imag.every(finite)) throw new Error("The server returned an invalid complex preview. Preview again or check the server log.");
  return { tau, real, imag, amplitude: real.map((value, i) => Math.hypot(value, imag[i])) };
}
function renderPreview(data) {
  const arrays = previewArrays(data);
  state.preview = { ...data, arrays };
  el("custom-preview-plots").hidden = false;
  text("preview-badge", data.can_run ? "READY · EXPLORATORY" : "INPUT CHECK FAILED");
  text("custom-preview-intro", data.can_run ? "The sampled initial input is ready for a bounded exploratory run. Review the warnings below before continuing." : "This input cannot run with the current settings. Correct the formula or grid and preview again.");
  const summary = data.summary || {}, node = el("custom-preview-summary"); node.replaceChildren();
  for (const [label, value] of [["Peak |u|", number(summary.peak_amplitude)], ["Filtered field change", number(summary.projection_relative_l2)], ["Boundary action", percent(summary.boundary_action_fraction)]]) {
    const item = make("div"); item.append(make("span", "", label), make("strong", "", value)); node.append(item);
  }
  const warnings = el("custom-warnings"); warnings.replaceChildren();
  const messages = Array.isArray(data.warnings) ? data.warnings : [];
  if (messages.length) {
    warnings.append(make("h3", "", "Input warnings")); const list = make("ul");
    for (const warning of messages) list.append(make("li", "", typeof warning === "string" ? warning : warning.message || readable(warning)));
    warnings.append(list);
  }
  const failures = summary.failed_run_checks;
  if (!data.can_run) {
    const reason = Array.isArray(failures) && failures.length ? failures.map(readable).join("; ") : "The server’s sampled-input checks did not permit a run.";
    showCustomError(`${reason} Change the input or grid and preview again.`);
  }
  text("custom-preview-status", data.can_run ? "Preview complete. Run is enabled for this exact input and grid." : "Preview complete; Run is blocked by input checks.");
  drawCustomPreview();
}
function drawCustomPreview() {
  if (!state.preview?.arrays || el("custom-preview").hidden) return;
  const { tau, amplitude, real, imag } = state.preview.arrays, peak = Math.max(...amplitude);
  let total = 0, moment = 0, variance = 0;
  for (let i = 0; i < tau.length; i++) { total += amplitude[i] ** 2; moment += tau[i] * amplitude[i] ** 2; }
  const center = total ? moment / total : 0;
  for (let i = 0; i < tau.length; i++) variance += (tau[i] - center) ** 2 * amplitude[i] ** 2;
  const half = Math.max(12, 4 * Math.sqrt(total ? variance / total : 0)), xmin = Math.max(tau[0], center - half), xmax = Math.min(tau.at(-1), center + half);
  for (const mode of ["amplitude", "phase"]) {
    const surface = canvasContext(`custom-${mode}-chart`); if (!surface) continue;
    const { ctx, width, height } = surface, box = { left: 48, right: width - 12, top: 12, bottom: height - 38 }, ymin = mode === "phase" ? -Math.PI : 0, ymax = mode === "phase" ? Math.PI : Math.max(peak * 1.12, 1e-8);
    axes(ctx, box, xmin, xmax, ymin, ymax, mode === "phase" ? "arg(A) · radians" : "Envelope |A|");
    ctx.save(); ctx.beginPath(); ctx.rect(box.left, box.top, box.right - box.left, box.bottom - box.top); ctx.clip(); ctx.strokeStyle = "#1e1e1e"; ctx.lineWidth = 1.7; ctx.beginPath();
    let drawing = false, previous = null;
    for (let i = 0; i < tau.length; i++) {
      if (tau[i] < xmin || tau[i] > xmax) continue;
      const value = mode === "amplitude" ? amplitude[i] : Math.atan2(imag[i], real[i]);
      if (mode === "phase" && (amplitude[i] < .01 * peak || previous !== null && Math.abs(value - previous) > Math.PI)) { drawing = false; previous = value; continue; }
      const x = box.left + (tau[i] - xmin) / (xmax - xmin) * (box.right - box.left), y = box.bottom - (value - ymin) / (ymax - ymin) * (box.bottom - box.top);
      if (drawing) ctx.lineTo(x, y); else { ctx.moveTo(x, y); drawing = true; } previous = value;
    }
    ctx.stroke(); ctx.restore();
  }
  text("preview-range", `Packet view τ ∈ [${number(xmin)}, ${number(xmax)}] of the periodic length 160. Phase is masked below 1% of peak amplitude. Plots show ${state.preview.preview_field || "the sampled formula before the numerical projector"}; boundary and resolution checks use the full grid.`);
}
async function previewCustom() {
  invalidatePreview("Preparing initial-envelope preview…");
  let body; try { body = customRequest(); } catch (error) { showCustomError(error.message); text("custom-preview-status", "Preview could not start. Check the input above."); return; }
  const token = ++state.previewToken, key = JSON.stringify(body); state.previewPending = true; updateRunAvailability();
  text("preview-badge", "CHECKING INPUT…");
  try {
    const result = await request("/api/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: key });
    if (token !== state.previewToken || state.caseId !== "custom") return;
    if (key !== JSON.stringify(customRequest())) return;
    state.previewKey = key; renderPreview(result);
  } catch (error) {
    if (token !== state.previewToken) return;
    state.preview = null; state.previewKey = null; showCustomError(error.message);
    text("custom-preview-status", "Preview failed. Correct the input and try again."); text("preview-badge", "PREVIEW FAILED");
  } finally {
    if (token === state.previewToken) { state.previewPending = false; updateRunAvailability(); }
  }
}
function parameters() {
  const result = {};
  for (const input of el("parameter-fields").querySelectorAll("input")) { if (!input.reportValidity()) return null; result[input.name] = Number(input.value); }
  return result;
}
async function runLive() {
  if (state.job) return;
  let body;
  if (state.caseId === "custom") {
    try {
      body = customRequest();
      if (!state.preview?.can_run || !state.preview?.fingerprint || state.previewKey !== JSON.stringify(body)) throw new Error("Preview this exact input and grid before Run.");
      body.preview_fingerprint = state.preview.fingerprint;
    } catch (error) { showCustomError(error.message); updateRunAvailability(); return; }
  } else { const params = parameters(); if (!params) return; body = { case_id: state.caseId, parameters: params }; }
  showError(""); pause();
  state.job = "submitting";
  const caseId = state.caseId; updateRunAvailability(); el("job-status").hidden = false; text("job-status-text", "Submitting bounded local CPU run…"); el("job-progress").hidden = true;
  try {
    const result = await request("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!result.job_id) throw new Error("The server did not return a job identifier.");
    state.job = result.job_id; await pollJob(result.job_id, caseId);
  } catch (error) { text("job-status-text", `Run failed: ${error.message}`); showError(`Local run could not complete: ${error.message}`); }
  finally { state.job = null; updateRunAvailability(); el("job-progress").hidden = true; }
}
async function pollJob(id, caseId) {
  for (;;) {
    const job = await request(`/api/jobs/${encodeURIComponent(id)}`); const progress = job.progress;
    if (job.status === "complete" || job.status === "completed") {
      const data = await request(job.result_url || `/api/results/${encodeURIComponent(id)}`);
      if (state.caseId !== caseId) { state.caseId = caseId; renderCases(); renderParameterFields(); text("case-description", currentCase()?.description || ""); }
      ++state.loadToken; loadData(data, "live", job.download_url || `/api/jobs/${encodeURIComponent(id)}/download`); text("job-status-text", caseId === "custom" ? "Custom run complete. The saved result uses the formula and settings shown with its plots; current editor changes apply to a future run. Grid, step and domain checks remain pending." : "Local run complete. Showing its saved snapshots; custom parameters need their own refinement checks."); return;
    }
    if (job.status === "error" || job.status === "failed") throw new Error(job.error || job.message || "Solver failed.");
    let label = job.status === "queued" ? "Local CPU run queued." : "Solver running locally on CPU…";
    if (progress && typeof progress === "object") {
      if (progress.phase) label = String(progress.phase).replaceAll("_", " ");
      const done = progress.completed_steps ?? progress.completed, total = progress.total_steps ?? progress.total;
      if (finite(done) && finite(total) && total > 0) { el("job-progress").hidden = false; el("job-progress").max = total; el("job-progress").value = done; label += ` · ${done} / ${total} steps`; }
    }
    text("job-status-text", label); await new Promise(resolve => setTimeout(resolve, 900));
  }
}

el("load-reference").addEventListener("click", loadReference); el("run-live").addEventListener("click", runLive); el("play-pause").addEventListener("click", play);
el("preview-custom").addEventListener("click", previewCustom);
for (const id of ["custom-formula", "custom-symbols", "custom-horizon", "custom-grid"]) el(id).addEventListener("input", () => invalidatePreview());
for (const name of ["gaussian", "sech"]) el(`example-${name}`).addEventListener("click", () => { el("custom-formula").value = formulaExamples[name]; el("custom-symbols").value = name === "sech" ? "A=1, w=2, k=0.3" : "A=1, w=2, k=0.3, c=0"; invalidatePreview("Example inserted. Preview it before Run."); el("custom-formula").focus(); });
el("snapshot").addEventListener("input", event => { pause(); state.frame = Number(event.target.value); renderFrame(); });
el("field-mode").addEventListener("change", renderFrame); el("view-range").addEventListener("change", renderFrame); el("heatmap-scale").addEventListener("change", drawHeatmap);
el("heatmap-chart").addEventListener("click", event => { if (!state.data) return; const rect = event.currentTarget.getBoundingClientRect(), box = plotBounds(rect.width, rect.height, true), y = event.clientY - rect.top; if (y < box.top || y > box.bottom) return; pause(); state.frame = Math.round((y - box.top) / (box.bottom - box.top) * (state.data.xi.length - 1)); renderFrame(); });
let resizeTimer; window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (state.data) { drawEnvelope(); drawHeatmap(); } drawCustomPreview(); }, 80); });
document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
async function init() {
  try {
    const payload = await request("/api/cases"); state.cases = Array.isArray(payload) ? payload : payload.cases || [];
    if (!state.cases.length) throw new Error("No reference cases are configured.");
    await selectCase(state.cases[0].id);
  } catch (error) { showError(`Could not connect to the local simulator: ${error.message} Start the Python server and reload this page.`); text("source-badge", "CONNECTION FAILED"); }
}
init();

function getSession() {
  try {
    return JSON.parse(localStorage.getItem("session")) || null;
  } catch {
    return null;
  }
}
function clearSession() {
  localStorage.removeItem("session");
}

function loadTasks() {
  try {
    const raw = localStorage.getItem("tasks");
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}
function saveTasks(list) {
  try {
    localStorage.setItem("tasks", JSON.stringify(list));
    localStorage.setItem("tasks_last_update", String(Date.now()));
  } catch {}
}
function upsertTask(next) {
  const list = loadTasks();
  const i = list.findIndex((t) => t.id === next.id);
  if (i === -1) list.push(next);
  else list[i] = { ...list[i], ...next };
  saveTasks(list);
}
function findTask(id) {
  return loadTasks().find((t) => t.id === id) || null;
}

function getParams() {
  const u = new URL(location.href);
  return {
    id: u.searchParams.get("id") || "",
    model: u.searchParams.get("model") || "",
  };
}
function pctFromRatio(v) {
  if (v == null || isNaN(v)) return null;
  const n = Number(v);
  return n <= 1 ? n * 100 : n;
}
function formatPct(v) {
  const n = pctFromRatio(v);
  return n == null ? "—" : `${n.toFixed(2)}%`;
}
function formatMs(v) {
  if (v == null) return "—";
  const n = Number(v);
  if (n === 0) return "0.00 ms";
  if (n < 0.01) return `${(n * 1000).toFixed(2)} µs`;
  return `${n.toFixed(2)} ms`;
}
function formatKB(v) {
  if (v == null) return "—";
  const num = Number(v);
  return num < 1024 ? `${Math.round(num)} KB` : `${(num / 1024).toFixed(1)} MB`;
}
function safeName(s) {
  return String(s || "dataset").replace(/[\\/:*?"<>|#]+/g, "_");
}

const IGNORE_IDS = new Set([
  "task_1001",
  "task_1002",
  "task_1003",
  "task_status_demo",
]);

(function ensureMockResults() {
  const { id } = getParams();
  if (id && !IGNORE_IDS.has(id)) return;

  const list = loadTasks();
  const idx = list.findIndex((t) => t.id === "task_1002");
  const demo = {
    id: "task_1002",
    datasetName: "acoustic_events_2025",
    taskType: "audio",
    device: "High-Performance SBCs",
    status: "completed",
    createdAt: Date.now() - 2 * 24 * 60 * 60 * 1e3,
    metrics: {
      accuracy: 92.03,
      latencyMsBefore: 18.9,
      latencyMsAfter: 13.5,
      sizeKBBefore: 120,
      sizeKBAfter: 98,
    },
    reportUrl: "assets/demo-report.pdf",
    modelUrl: "assets/demo-model.tflite",
    model_name: "DemoNet",
    guidance:
      "Use TFLite-Micro; allocate static tensor arena ≈ 64–128 KB;\nEnable CMSIS-NN if on ARM; quantize int8.",
  };
  if (idx === -1) {
    list.push(demo);
    saveTasks(list);
  } else {
    list[idx] = { ...demo, ...list[idx], status: "completed" };
    saveTasks(list);
  }
})();

const empty = document.getElementById("empty");
const content = document.getElementById("content");
const subTitle = document.getElementById("subTitle");
const dsNameEl = document.getElementById("dsName");
const downloadBtn = document.getElementById("downloadBtn");
const dlHint = document.getElementById("dlHint");
const reportBtn = document.getElementById("reportBtn");
const guidanceEl = document.getElementById("guidance");
const errorCard = document.getElementById("errorCard");
const errorReason = document.getElementById("errorReason");
const accOpt = document.getElementById("accOpt");
const latBeforeOpt = document.getElementById("latBeforeOpt");
const latAfterOpt = document.getElementById("latAfterOpt");
const sizeBeforeOpt = document.getElementById("sizeBeforeOpt");
const sizeAfterOpt = document.getElementById("sizeAfterOpt");

function ensureMetricRow(id, label) {
  let valEl = document.getElementById(id);
  if (valEl) return valEl;
  const table = document.querySelector("#metricsTable tbody, #metricsTable");
  if (!table) return null;

  const tr = document.createElement("tr");
  const tdL = document.createElement("td");
  const tdR = document.createElement("td");
  tdL.textContent = label;
  valEl = document.createElement("span");
  valEl.id = id;
  tdR.appendChild(valEl);
  tr.appendChild(tdL);
  tr.appendChild(tdR);
  table.appendChild(tr);
  return valEl;
}

function buildGuidance(task) {
  if (task.guidance) return task.guidance;
  const device = (task.device || task.device_family || "").toLowerCase();
  const out = [];
  if (device.includes("mcu") || device.includes("risc-v")) {
    out.push(
      "Use TFLite-Micro; allocate a static tensor arena (size depends on model)."
    );
    if (device.includes("arm"))
      out.push("Enable CMSIS-NN on ARM targets for faster kernels.");
    if (device.includes("npu") || device.includes("kpu"))
      out.push("Leverage on-die NPU/KPU operators where available.");
    out.push("Prefer int8 quantization for memory/bandwidth savings.");
  } else if (device.includes("sbc")) {
    out.push("Use TFLite or ONNX Runtime; delegate to GPU/NPU if present.");
    out.push(
      "Batch inference where possible; pin a core for deterministic latency."
    );
  } else {
    out.push(
      "Use a lightweight runtime; prefer int8 quantization where supported."
    );
  }
  return out.join("\n");
}

function apiBase() {
  return (
    localStorage.getItem("api_base") ||
    new URL(location.href).searchParams.get("api") ||
    "http://127.0.0.1:8000"
  );
}
async function ensureModelExt(task) {
  if (!task || !task.model_id || IGNORE_IDS.has(task.id)) return task;
  if (task?.export_summary?.saved_as) return task;
  try {
    const res = await fetch(
      `${apiBase()}/models/${encodeURIComponent(task.model_id)}`
    );
    if (!res.ok) return task;
    const meta = await res.json();
    if (!meta?.ext) return task;
    const updated = {
      ...task,
      export_summary: { ...(task.export_summary || {}), saved_as: meta.ext },
      model_name: task.model_name || meta.model_name,
    };
    upsertTask(updated);
    return updated;
  } catch {
    return task;
  }
}

function buildModelAssetPath(task) {
  const taskId = task.id;
  if (!taskId) return null;
  if (!task?.export_summary?.saved_as && task.status !== "completed") return null;
  
  return `${apiBase()}/tasks/${encodeURIComponent(taskId)}/download?kind=model`;
}

function buildReportPdfPath(task) {
  const taskId = task.id;
  if (!taskId) return null;
  if (task.status !== "completed") return null;

  return `${apiBase()}/tasks/${encodeURIComponent(taskId)}/download?kind=report`;
}

async function render(task) {
  if (!task) {
    if (empty) empty.hidden = false;
    if (content) content.hidden = true;
    return;
  }
  if (empty) empty.hidden = true;
  const datasetRaw = task.datasetName || "Untitled";
  if (dsNameEl) dsNameEl.textContent = datasetRaw;
  if (subTitle)
    subTitle.textContent = task.model_name ? `Model: ${task.model_name}` : "";

  const isMock = IGNORE_IDS.has(task.id);
  const isDone = task.status === "completed";
  const isFailed = task.status === "failed";

  if (errorCard) errorCard.hidden = !isFailed;
  if (content) content.hidden = isFailed; 

  if (isFailed) {
      if (errorReason) errorReason.textContent = task.error?.reason || task.errorMessage?.reason || "Unknown failure reason";
      return; 
  }

  if (!isMock && isDone && !task?.export_summary?.saved_as && task.model_id) {
    task = await ensureModelExt(task);
  }

  const modelPath = buildModelAssetPath(task);
  const reportPdf = buildReportPdfPath(task);

  const guessedExt = (task?.export_summary?.saved_as || "").replace(/^\./, "");
  const modelFileLabel =
    modelPath && guessedExt
      ? `${safeName(datasetRaw)}.${guessedExt}`
      : "pending";
  const reportFileLabel = `${safeName(datasetRaw)}.pdf`;

  if (downloadBtn)
    downloadBtn.textContent = modelPath
      ? `Download`
      : "Download (pending)";
  if (reportBtn) reportBtn.textContent = `Open Report`;

  if (isMock) {
    const swallow = (e) => {
      e.preventDefault();
      e.stopPropagation();
      return false;
    };
    if (downloadBtn) {
      downloadBtn.href = "javascript:void(0)";
      downloadBtn.onclick = swallow;
      downloadBtn.classList.add("disabled");
      downloadBtn.setAttribute("aria-disabled", "true");
      downloadBtn.textContent = "Download";
    }
    if (reportBtn) {
      reportBtn.href = "javascript:void(0)";
      reportBtn.onclick = swallow;
      reportBtn.classList.add("disabled");
      reportBtn.setAttribute("aria-disabled", "true");
      reportBtn.textContent = "Open";
    }
  } else {
    const enableModel = isDone && !!modelPath;
    const enableReport = !!reportPdf;

    if (downloadBtn) {
      downloadBtn.href = enableModel ? modelPath : "javascript:void(0)";
      downloadBtn.toggleAttribute("disabled", !enableModel);
      downloadBtn.classList.toggle("disabled", !enableModel);
      downloadBtn.setAttribute("aria-disabled", String(!enableModel));
    }
    if (reportBtn) {
      reportBtn.href = enableReport ? reportPdf : "javascript:void(0)";
      reportBtn.target = enableReport ? "_blank" : "";
      reportBtn.rel = enableReport ? "noopener" : "";
      reportBtn.classList.toggle("disabled", !enableReport);
      reportBtn.setAttribute("aria-disabled", String(!enableReport));
    }
    if (dlHint)
      dlHint.textContent = enableModel
        ? ""
        : "Model will be available once the task is completed.";
  }

  if (guidanceEl) guidanceEl.textContent = buildGuidance(task);

  const m = task.metrics || {};

  const acc = isMock ? m.accuracy ?? 92.03 : m.accuracy ?? null;

  const latBefore = isMock
    ? m.latencyMsBefore ?? 18.9
    : m.latencyMsBefore ?? null;
  const latAfter = isMock ? m.latencyMsAfter ?? 13.5 : m.latencyMsAfter ?? null;

  const sizeBeforeKB = isMock ? m.sizeKBBefore ?? 120 : m.sizeKBBefore ?? null;
  const sizeAfterKB = isMock ? m.sizeKBAfter ?? 98 : m.sizeKBAfter ?? null;

  if (accOpt) accOpt.textContent = acc == null ? "—" : formatPct(acc);
  if (latBeforeOpt)
    latBeforeOpt.textContent = latBefore == null ? "—" : formatMs(latBefore);
  if (latAfterOpt)
    latAfterOpt.textContent = latAfter == null ? "—" : formatMs(latAfter);
  if (sizeBeforeOpt)
    sizeBeforeOpt.textContent =
      sizeBeforeKB == null ? "—" : formatKB(sizeBeforeKB);
  if (sizeAfterOpt)
    sizeAfterOpt.textContent =
      sizeAfterKB == null ? "—" : formatKB(sizeAfterKB);

  const toPersistMetrics = {
    accuracy: acc,
    latencyMs: latAfter,
    latencyMsBefore: latBefore,
    latencyMsAfter: latAfter,
    sizeKBBefore: sizeBeforeKB,
    sizeKBAfter: sizeAfterKB,
    model_name: task.model_name || m.model_name || null,
  };

  upsertTask({
    ...task,
    modelUrl: modelPath || task.modelUrl,
    reportUrl: reportPdf || task.reportUrl,
    metrics: toPersistMetrics,
  });
}

let pollTimer = null,
  pollErrors = 0;

async function poll(taskId) {
  if (IGNORE_IDS.has(taskId)) {
    render(findTask(taskId));
    return;
  }

  const base = apiBase();
  const doPoll = async () => {
    const local = findTask(taskId);
    if (local && (local.status === "completed" || local.status === "failed")) {
      pollTimer && clearInterval(pollTimer);
      await render(local);
      return;
    }
    try {
      const res = await fetch(`${base}/tasks/${encodeURIComponent(taskId)}`);
      if (res.status === 404) {
        pollTimer && clearInterval(pollTimer);
        const failed = {
          ...(local || { id: taskId }),
          status: "failed",
          phase: "failed",
          errorMessage: "Task not found on server (404).",
          updatedAt: Date.now(),
        };
        upsertTask(failed);
        await render(failed);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const snap = await res.json();
      pollErrors = 0;

      const merged = {
        ...(local || { id: taskId }),
        id: taskId,
        status: snap.status?.status || snap.sched_state || snap.phase || local?.status || "queued",
        phase: snap.phase ?? local?.phase,
        progress:
          typeof snap.progress === "number" ? snap.progress : local?.progress,
        metrics: snap.metrics || local?.metrics,
        model_id: snap.model_id || local?.model_id,
        model_name: snap.model_name || local?.model_name,
        export_summary: snap.export_summary || local?.export_summary,
        asset_paths: snap.asset_paths || local?.asset_paths,
        device_family: snap.device_family || local?.device_family,
        error: snap.error || local?.error || null,
        error: snap.error || local?.error || null,
        errorMessage: local?.errorMessage || null,
        updatedAt: Date.now(),
      };

      if (merged.metrics) {
        merged.metrics = {
          accuracy: merged.metrics.accuracy ?? null,
          latencyMsBefore: merged.metrics.latencyMsBefore ?? null,
          latencyMsAfter: merged.metrics.latencyMsAfter ?? null,
          sizeKBBefore: merged.metrics.sizeKBBefore ?? null,
          sizeKBAfter: merged.metrics.sizeKBAfter ?? null,
          model_name: merged.model_name || merged.metrics.model_name || null,
        };
      }

      upsertTask(merged);
      await render(merged);

      if (merged.status === "completed" || merged.status === "failed") {
        pollTimer && clearInterval(pollTimer);
      }
    } catch {
      pollErrors += 1;
      await render(findTask(taskId));
      if (pollErrors >= 5 && pollTimer) clearInterval(pollTimer);
    }
  };

  await doPoll();
  pollTimer = setInterval(doPoll, 4000);
}

(async function init() {
  const { id } = getParams();
  const task = findTask(id);

  await render(task || findTask("task_1002"));
  poll(id);

  window.addEventListener("storage", (e) => {
    if (e.key === "tasks" || e.key === "tasks_last_update") {
      render(findTask(id));
    }
  });
})();

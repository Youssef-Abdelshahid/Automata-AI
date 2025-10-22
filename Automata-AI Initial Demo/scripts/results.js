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

const avatarBtn = document.getElementById("avatarBtn");
const avatarInitials = document.getElementById("avatarInitials");
const accountMenu = document.getElementById("accountMenu");
const menuName = document.getElementById("menuName");
const menuEmail = document.getElementById("menuEmail");
const logoutBtn = document.getElementById("logoutBtn");

(function initAccountUI() {
  const s = getSession() || { email: "demo@automata.ai", uid: "demo-uid" };
  const name = s.name || "Automata Demo";
  const email = s.email || "demo@automata.ai";
  const initials = (name || email)
    .split(/[\s._-]+/)
    .slice(0, 2)
    .map((w) => w?.[0] || "")
    .join("")
    .toUpperCase()
    .slice(0, 2);
  avatarInitials && (avatarInitials.textContent = initials || "AD");
  menuName && (menuName.textContent = name);
  menuEmail && (menuEmail.textContent = email);

  let open = false;
  function setOpen(v) {
    open = v;
    accountMenu?.classList.toggle("show", v);
    avatarBtn?.setAttribute("aria-expanded", String(v));
    accountMenu?.setAttribute("aria-hidden", String(!v));
  }
  avatarBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(!open);
  });
  document.addEventListener("click", () => open && setOpen(false));
  document.addEventListener(
    "keydown",
    (e) => e.key === "Escape" && setOpen(false)
  );
  logoutBtn?.addEventListener("click", () => {
    clearSession();
    location.href = "./auth.html";
  });
})();

function loadJobs() {
  try {
    const raw = localStorage.getItem("jobs");
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}
function saveJobs(list) {
  try {
    localStorage.setItem("jobs", JSON.stringify(list));
    localStorage.setItem("jobs_last_update", String(Date.now()));
  } catch {}
}
function upsertJob(next) {
  const list = loadJobs();
  const i = list.findIndex((j) => j.id === next.id);
  if (i === -1) list.push(next);
  else list[i] = { ...list[i], ...next };
  saveJobs(list);
}
function findJob(id) {
  return loadJobs().find((j) => j.id === id) || null;
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
  return v != null ? `${Number(v).toFixed(2)} ms` : "—";
}
function formatKB(v) {
  if (v == null) return "—";
  const num = Number(v);
  return num < 1024 ? `${Math.round(num)} KB` : `${(num / 1024).toFixed(1)} MB`;
}
function safeName(s) {
  return String(s || "dataset").replace(/[\\/:*?"<>|#]+/g, "_");
}

const IGNORE_IDS = new Set(["job_1001", "job_1002", "job_1003"]);

(function ensureMockResults() {
  const { id } = getParams();
  if (id && !IGNORE_IDS.has(id)) return;
  const list = loadJobs();
  const idx = list.findIndex((j) => j.id === "job_1002");
  const demo = {
    id: "job_1002",
    datasetName: "acoustic_events_2025.csv",
    taskType: "audio",
    device: "High-Performance SBCs",
    status: "completed",
    createdAt: Date.now() - 2 * 24 * 60 * 60 * 1e3,
    metrics: { accuracy: 92.03, latencyMs: 13.5, sizeKB: 98 },
    reportUrl: "assets/demo-report.pdf",
    modelUrl: "assets/demo-model.tflite",
    model_name: "DemoNet",
    guidance:
      "Use TFLite-Micro; allocate static tensor arena ≈ 64–128 KB;\nEnable CMSIS-NN if on ARM; quantize int8.",
  };
  if (idx === -1) {
    list.push(demo);
    saveJobs(list);
  } else {
    list[idx] = { ...demo, ...list[idx], status: "completed" };
    saveJobs(list);
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
const accVal = document.getElementById("accVal");
const latVal = document.getElementById("latVal");
const sizeVal = document.getElementById("sizeVal");

function buildGuidance(job) {
  if (job.guidance) return job.guidance;
  const device = (job.device || job.device_family || "").toLowerCase();
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
async function ensureModelExt(job) {
  if (!job || !job.model_id || IGNORE_IDS.has(job.id)) return job;
  if (job?.export_summary?.saved_as) return job;
  try {
    const res = await fetch(
      `${apiBase()}/models/${encodeURIComponent(job.model_id)}`
    );
    if (!res.ok) return job;
    const meta = await res.json();
    if (!meta?.ext) return job;
    const updated = {
      ...job,
      export_summary: { ...(job.export_summary || {}), saved_as: meta.ext },
      model_name: job.model_name || meta.model_name,
    };
    upsertJob(updated);
    return updated;
  } catch {
    return job;
  }
}

function buildModelAssetPath(job) {
  const jobId = job.id,
    modelId = job.model_id || job.modelId,
    ext = job?.export_summary?.saved_as || "";
  if (!jobId || !modelId || !ext) return null;
  return `assets/${encodeURIComponent(jobId)}/models/${encodeURIComponent(
    modelId
  )}${ext}`;
}
function buildReportPdfPath(job) {
  const jobId = job.id,
    modelId = job.model_id || job.modelId;
  if (!jobId || !modelId) return null;
  return `assets/${encodeURIComponent(jobId)}/reports/${encodeURIComponent(
    modelId
  )}.pdf`;
}

async function render(job) {
  if (!job) {
    empty && (empty.hidden = false);
    content && (content.hidden = true);
    return;
  }
  empty && (empty.hidden = true);
  content && (content.hidden = false);

  const datasetRaw = job.datasetName || "Untitled";
  dsNameEl && (dsNameEl.textContent = datasetRaw);

  subTitle &&
    (subTitle.textContent = job.model_name ? `Model: ${job.model_name}` : "");

  const isMock = IGNORE_IDS.has(job.id);
  const isDone = job.status === "completed";

  if (!isMock && isDone && !job?.export_summary?.saved_as && job.model_id) {
    job = await ensureModelExt(job);
  }

  const modelPath = buildModelAssetPath(job);
  const reportPdf = buildReportPdfPath(job);

  const guessedExt = (job?.export_summary?.saved_as || "").replace(/^\./, "");
  const modelFileLabel =
    modelPath && guessedExt ? `${job.model_name}.${guessedExt}` : "pending";
  const reportFileLabel = `Job report.pdf`;

  downloadBtn &&
    (downloadBtn.textContent = modelPath
      ? `Download ${modelFileLabel}`
      : "Download (pending)");
  reportBtn && (reportBtn.textContent = `Open ${reportFileLabel}`);

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
      downloadBtn.textContent = "Download demo-model.tflite";
    }
    if (reportBtn) {
      reportBtn.href = "javascript:void(0)";
      reportBtn.onclick = swallow;
      reportBtn.classList.add("disabled");
      reportBtn.setAttribute("aria-disabled", "true");
      reportBtn.textContent = "Open demo-report.pdf";
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
    dlHint &&
      (dlHint.textContent = enableModel
        ? ""
        : "Model will be available once the job is completed.");
  }

  guidanceEl && (guidanceEl.textContent = buildGuidance(job));

  const m = job.metrics || {};
  const acc = isMock
    ? m.accuracy ?? 92.03
    : m.accuracy ?? m.test_set_accuracy ?? null;
  const lat = isMock
    ? m.latencyMs ?? 13.5
    : m.latencyMs ?? m.latency_ms ?? null;
  const size = isMock ? m.sizeKB ?? 98 : m.sizeKB ?? m.model_size_kb ?? null;

  accVal &&
    (accVal.textContent = isMock
      ? formatPct(acc)
      : acc == null
      ? "—"
      : formatPct(acc));
  latVal &&
    (latVal.textContent = isMock
      ? formatMs(lat)
      : lat == null
      ? "—"
      : formatMs(lat));
  sizeVal &&
    (sizeVal.textContent = isMock
      ? formatKB(size)
      : size == null
      ? "—"
      : formatKB(size));

  const toPersistMetrics = isMock
    ? {
        accuracy: pctFromRatio(acc),
        latencyMs: lat,
        sizeKB: size,
        model_name: job.model_name || m.model_name,
      }
    : {
        accuracy: m.accuracy ?? m.test_set_accuracy ?? null,
        latencyMs: m.latencyMs ?? m.latency_ms ?? null,
        sizeKB: m.sizeKB ?? m.model_size_kb ?? null,
        model_name: job.model_name || m.model_name || null,
      };

  upsertJob({
    ...job,
    modelUrl: modelPath || job.modelUrl,
    reportUrl: reportPdf || job.reportUrl,
    metrics: toPersistMetrics,
  });
}

let pollTimer = null,
  pollErrors = 0;

async function poll(jobId) {
  if (IGNORE_IDS.has(jobId)) {
    render(findJob(jobId));
    return;
  }

  const base = apiBase();
  const doPoll = async () => {
    const local = findJob(jobId);
    if (local && (local.status === "completed" || local.status === "failed")) {
      pollTimer && clearInterval(pollTimer);
      await render(local);
      return;
    }
    try {
      const res = await fetch(`${base}/jobs/${encodeURIComponent(jobId)}`);
      if (res.status === 404) {
        pollTimer && clearInterval(pollTimer);
        const failed = {
          ...(local || { id: jobId }),
          status: "failed",
          phase: "failed",
          errorMessage: "Job not found on server (404).",
          updatedAt: Date.now(),
        };
        upsertJob(failed);
        await render(failed);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const snap = await res.json();
      pollErrors = 0;

      const merged = {
        ...(local || { id: jobId }),
        id: jobId,
        status: snap.status || snap.phase || local?.status || "queued",
        phase: snap.phase ?? local?.phase,
        progress:
          typeof snap.progress === "number" ? snap.progress : local?.progress,
        metrics: snap.metrics || local?.metrics,
        model_id: snap.model_id || local?.model_id,
        model_name: snap.model_name || local?.model_name,
        export_summary: snap.export_summary || local?.export_summary,
        asset_paths: snap.asset_paths || local?.asset_paths,
        device_family: snap.device_family || local?.device_family,
        updatedAt: Date.now(),
      };
      if (merged.metrics) {
        merged.metrics = {
          accuracy:
            merged.metrics.accuracy ?? merged.metrics.test_set_accuracy ?? null,
          latencyMs:
            merged.metrics.latencyMs ?? merged.metrics.latency_ms ?? null,
          sizeKB: merged.metrics.sizeKB ?? merged.metrics.model_size_kb ?? null,
          model_name: merged.model_name || merged.metrics.model_name || null,
        };
      }
      upsertJob(merged);
      await render(merged);

      if (merged.status === "completed" || merged.status === "failed") {
        pollTimer && clearInterval(pollTimer);
      }
    } catch {
      pollErrors += 1;
      await render(findJob(jobId));
      if (pollErrors >= 5 && pollTimer) clearInterval(pollTimer);
    }
  };

  await doPoll();
  pollTimer = setInterval(doPoll, 4000);
}

(async function init() {
  const { id } = getParams();
  const job = findJob(id);

  if (job && job.status === "failed") {
    location.replace("./jobs.html");
  }

  await render(job || findJob("job_1002"));
  poll(id);

  window.addEventListener("storage", (e) => {
    if (e.key === "jobs" || e.key === "jobs_last_update") {
      render(findJob(id));
    }
  });
})();

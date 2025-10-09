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

  avatarInitials.textContent = initials || "AD";
  menuName.textContent = name;
  menuEmail.textContent = email;

  let open = false;
  function setOpen(v) {
    open = v;
    accountMenu.classList.toggle("show", open);
    avatarBtn.setAttribute("aria-expanded", String(open));
    accountMenu.setAttribute("aria-hidden", String(!open));
  }

  avatarBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(!open);
  });
  document.addEventListener("click", () => open && setOpen(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setOpen(false);
  });

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
function findJob(id) {
  return loadJobs().find((j) => j.id === id) || null;
}
function getParams() {
  const u = new URL(location.href);
  return { id: u.searchParams.get("id") || "" };
}
function formatPct(v) {
  return v != null ? `${Number(v).toFixed(2)}%` : "—";
}
function formatMs(v) {
  return v != null ? `${Number(v).toFixed(1)} ms` : "—";
}
function formatKB(v) {
  if (v == null) return "—";
  const num = Number(v);
  return num < 1024 ? `${Math.round(num)} KB` : `${(num / 1024).toFixed(1)} MB`;
}

(function ensureMockResults() {
  let list = loadJobs();
  const idx = list.findIndex((j) => j.id === "job_1002");
  if (idx !== -1) {
    const job = list[idx];
    if (!job.metrics || !job.modelUrl || !job.reportUrl) {
      list[idx] = {
        ...job,
        status: "completed",
        metrics: job.metrics || {
          accuracy: 92.03,
          latencyMs: 13.5,
          sizeKB: 98,
        },
        reportUrl: job.reportUrl || "assets/demo-report.pdf",
        modelUrl: job.modelUrl || "assets/demo-model.tflite",
        guidance:
          job.guidance ||
          "Use TFLite-Micro; allocate static tensor arena ≈ 64–128 KB;\nEnable CMSIS-NN if on ARM; quantize int8.",
      };
      saveJobs(list);
    }
  } else {
    list.push({
      id: "job_1002",
      datasetName: "acoustic_events_2025.csv",
      taskType: "audio",
      device: "High-Performance SBCs",
      status: "completed",
      createdAt: Date.now() - 2 * 24 * 60 * 60 * 1e3,
      metrics: { accuracy: 92.03, latencyMs: 13.5, sizeKB: 98 },
      reportUrl: "assets/demo-report.pdf",
      modelUrl: "assets/demo-model.tflite",
      guidance:
        "Use TFLite-Micro; allocate static tensor arena ≈ 64–128 KB;\nEnable CMSIS-NN if on ARM; quantize int8.",
    });
    saveJobs(list);
  }
})();


const empty = document.getElementById("empty");
const content = document.getElementById("content");
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
  const device = (job.device || "").toLowerCase();
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


(function init() {
  const { id } = getParams();
  const job = findJob(id);

  if (!job) {
    empty.hidden = false;
    content.hidden = true;
    return;
  }

  empty.hidden = true;
  content.hidden = false;

  dsNameEl.textContent = job.datasetName || "Untitled";

  const isDone = job.status === "completed";
  const modelUrl = job.modelUrl || "assets/demo-model.tflite";
  const reportUrl = job.reportUrl || "assets/demo-report.pdf";

  downloadBtn.href = isDone ? modelUrl : "#";
  downloadBtn.toggleAttribute("disabled", !isDone);
  dlHint.textContent = isDone
    ? ""
    : "Model will be available once the job is completed.";

  reportBtn.href = reportUrl;

  guidanceEl.textContent = buildGuidance(job);

  const m = job.metrics || { accuracy: 92.03, latencyMs: 13.5, sizeKB: 98 };
  accVal.textContent = formatPct(m.accuracy);
  latVal.textContent = formatMs(m.latencyMs);
  sizeVal.textContent = formatKB(m.sizeKB);
})();

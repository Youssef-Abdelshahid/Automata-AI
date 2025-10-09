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

const seed = [
  {
    id: "job_1001",
    datasetName: "edge_device_families.csv",
    taskType: "image",
    device: "Ultra-Low-Power MCUs",
    status: "queued",
    createdAt: Date.now() - 3 * 60 * 60 * 1e3 - 10 * 60 * 1e3,
  },
  {
    id: "job_1002",
    datasetName: "acoustic_events_2025.csv",
    taskType: "audio",
    device: "High-Performance SBCs",
    status: "completed",
    createdAt: Date.now() - 2 * 24 * 60 * 60 * 1e3,
  },
  {
    id: "job_1003",
    datasetName: "sensor_combo_v2.csv",
    taskType: "sensor",
    device: "Mid-Range MCUs",
    status: "training",
    createdAt: Date.now() - 24 * 60 * 60 * 1e3 - 20 * 60 * 1e3,
  },
];

const ALLOWED_STATUS = new Set([
  "queued",
  "preprocessing",
  "training",
  "optimizing",
  "packaging",
  "completed",
  "failed",
]);
const ALLOWED_TYPES = new Set(["image", "audio", "sensor"]);

function normalizeJob(j, idx = 0) {
  const id = j?.id || `job_${Date.now().toString(36)}_${idx}`;
  const datasetName = String(j?.datasetName ?? j?.name ?? "Untitled");
  const taskType = ALLOWED_TYPES.has(j?.taskType) ? j.taskType : "image";
  const device = j?.device || "—";
  const status = ALLOWED_STATUS.has(j?.status) ? j.status : "queued";
  const createdAt =
    typeof j?.createdAt === "number" && Number.isFinite(j.createdAt)
      ? j.createdAt
      : Date.now();
  return { id, datasetName, taskType, device, status, createdAt };
}

function readJobsRaw() {
  try {
    const raw = localStorage.getItem("jobs");
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveJobs(list) {
  const normalized = list.map(normalizeJob);
  localStorage.setItem("jobs", JSON.stringify(normalized));
  localStorage.setItem("jobs_last_update", String(Date.now()));
  return normalized;
}

function loadJobs() {
  let list = readJobsRaw();
  if (!Array.isArray(list)) {
    list = seed.slice();
    return saveJobs(list);
  }
  return saveJobs(list);
}

loadJobs();

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
  avatarBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(!open);
  });
  document.addEventListener("click", () => open && setOpen(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setOpen(false);
  });

  logoutBtn.addEventListener("click", () => {
    clearSession();
    location.href = "./auth.html";
  });
})();

const RUNNING_SET = new Set([
  "queued",
  "preprocessing",
  "training",
  "optimizing",
  "packaging",
]);

function readJobs() {
  try {
    const raw = localStorage.getItem("jobs");
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function countsFrom(jobs) {
  let total = jobs.length,
    running = 0,
    completed = 0,
    failed = 0;
  for (const j of jobs) {
    const s = j?.status || "";
    if (RUNNING_SET.has(s)) running++;
    else if (s === "completed") completed++;
    else if (s === "failed") failed++;
  }
  return { total, running, completed, failed };
}

const els = {
  total: document.getElementById("statTotal"),
  running: document.getElementById("statRunning"),
  completed: document.getElementById("statCompleted"),
  failed: document.getElementById("statFailed"),
};

function render() {
  const jobs = readJobs();
  const c = countsFrom(jobs);
  els.total.textContent = c.total;
  els.running.textContent = c.running;
  els.completed.textContent = c.completed;
  els.failed.textContent = c.failed;
}

window.addEventListener("storage", (e) => {
  if (e.key === "jobs" || e.key === "jobs_last_update") render();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) render();
});

render();

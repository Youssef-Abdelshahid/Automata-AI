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
function saveSeedIfEmpty() {
  const arr = readJobs();
  if (!arr.length) {
    localStorage.setItem("jobs", JSON.stringify(seed));
    localStorage.setItem("jobs_last_update", String(Date.now()));
  }
}
saveSeedIfEmpty();

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

const RUNNING_KEYS = new Set([
  "queued",
  "preprocessing",
  "training",
  "optimizing",
  "packaging",
]);
const els = {
  total: document.getElementById("statTotal"),
  running: document.getElementById("statRunning"),
  completed: document.getElementById("statCompleted"),
  failed: document.getElementById("statFailed"),
};

function countsFrom(jobs) {
  let total = jobs.length,
    running = 0,
    completed = 0,
    failed = 0;
  for (const j of jobs) {
    const s = j?.status || "";
    if (RUNNING_KEYS.has(s)) running++;
    else if (s === "completed") completed++;
    else if (s === "failed") failed++;
  }
  return { total, running, completed, failed };
}

function render() {
  const jobs = readJobs();
  const c = countsFrom(jobs);
  els.total.textContent = c.total;
  els.running.textContent = c.running;
  els.completed.textContent = c.completed;
  els.failed.textContent = c.failed;
}

function normalizeStatus(s, fallback) {
  return ALLOWED_STATUS.has(s) ? s : fallback;
}
function mergeSnapshot(local, snap) {
  if (!snap) return local;
  const next = normalizeStatus(snap.status || snap.phase, local.status);
  return {
    ...local,
    status: next,
    phase: snap.phase ?? local.phase,
    progress:
      typeof snap.progress === "number" ? snap.progress : local.progress,
    metrics: snap.metrics || local.metrics,
    asset_paths: snap.asset_paths || local.asset_paths,
    model_id: snap.model_id || local.model_id,
    device_family: snap.device_family || local.device_family,
    updatedAt: Date.now(),
  };
}

async function pollOnce() {
  const apiBase =
    localStorage.getItem("api_base") ||
    new URL(location.href).searchParams.get("api") ||
    "http://127.0.0.1:8000";

  const list = readJobs();
  if (!list.length) return;

  const IGNORE_IDS = new Set(["job_1001", "job_1002", "job_1003"]);
  let changed = false;

  const next = await Promise.all(
    list.map(async (j) => {
      const id = j?.id;
      const s = j?.status || "queued";

      if (!id || IGNORE_IDS.has(id) || !RUNNING_SET.has(s)) return j;

      try {
        const r = await fetch(`${apiBase}/jobs/${encodeURIComponent(id)}`);
        if (!r.ok) throw new Error("http " + r.status);
        const snap = await r.json();
        const merged = mergeSnapshot(j, snap);
        if (JSON.stringify(merged) !== JSON.stringify(j)) changed = true;
        return merged;
      } catch {
        return j;
      }
    })
  );

  if (changed) {
    localStorage.setItem("jobs", JSON.stringify(next));
    localStorage.setItem("jobs_last_update", String(Date.now()));
    render();
  }
}

window.addEventListener("storage", (e) => {
  if (e.key === "jobs" || e.key === "jobs_last_update") render();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) render();
});

render();
pollOnce();
setInterval(pollOnce, 4000);

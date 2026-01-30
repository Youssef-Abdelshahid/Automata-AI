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
    id: "task_1001",
    title: "Factory Safety Production",
    datasetName: "helmet_detection_v2",
    taskType: "image",
    device: "Ultra-Low-Power MCUs",
    status: "queued",
    createdAt: Date.now() - 3 * 60 * 60 * 1e3 - 10 * 60 * 1e3,
  },
  {
    id: "task_1002",
    title: "Glass Break Sensor",
    datasetName: "acoustic_events_2025",
    taskType: "audio",
    device: "High-Performance SBCs",
    status: "completed",
    createdAt: Date.now() - 2 * 24 * 60 * 60 * 1e3,
  },
  {
    id: "task_1003",
    title: "Gesture Control V1",
    datasetName: "imu_motions_raw",
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

const ORDER = [
  "queued",
  "preprocessing",
  "training",
  "optimizing",
  "packaging",
  "completed",
  "failed",
];

function readTasks() {
  try {
    const raw = localStorage.getItem("tasks");
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}
function saveSeedIfEmpty() {
  const arr = readTasks();
  if (!arr.length) {
    localStorage.setItem("tasks", JSON.stringify(seed));
    localStorage.setItem("tasks_last_update", String(Date.now()));
  }
}
saveSeedIfEmpty();

const els = {
  total: document.getElementById("statTotal"),
  running: document.getElementById("statRunning"),
  completed: document.getElementById("statCompleted"),
  failed: document.getElementById("statFailed"),
};

function countsFrom(tasks) {
  let total = tasks.length,
    running = 0,
    completed = 0,
    failed = 0;
  for (const t of tasks) {
    const s = t?.status || "";
    if (RUNNING_SET.has(s)) running++;
    else if (s === "completed") completed++;
    else if (s === "failed") failed++;
  }
  return { total, running, completed, failed };
}

function render() {
  const tasks = readTasks();
  const c = countsFrom(tasks);
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

  const statusVal = snap.status?.status || snap.sched_state || local.status || "queued";
  const next = normalizeStatus(statusVal, local.status);

  let phase = snap.phase || local.phase;
  if (snap.status?.stage_idx !== undefined && ORDER[snap.status.stage_idx]) {
    phase = ORDER[snap.status.stage_idx];
  }

  return {
    ...local,
    status: next,
    phase: phase,
    progress:
      typeof snap.progress === "number" ? snap.progress : local.progress,
    metrics: snap.metrics || local.metrics,
    asset_paths: snap.asset_paths || local.asset_paths,
    model_id: snap.model_id || local.model_id,
    device_family: snap.device_family || local.device_family, 
    errorMessage: snap.error || local.errorMessage,
    updatedAt: Date.now(),
  };
}

async function pollOnce() {
  const apiBase =
    localStorage.getItem("api_base") ||
    new URL(location.href).searchParams.get("api") ||
    "http://127.0.0.1:8000";

  const list = readTasks();
  if (!list.length) return;

  const IGNORE_IDS = new Set([
    "task_1001",
    "task_1002",
    "task_1003",
  ]);
  let changed = false;

  const next = await Promise.all(
    list.map(async (t) => {
      const id = t?.id;
      const s = t?.status || "queued";

      if (!id || IGNORE_IDS.has(id) || !RUNNING_SET.has(s)) return t;

      try {
        const r = await fetch(`${apiBase}/tasks/${encodeURIComponent(id)}`);
        if (!r.ok) throw new Error("http " + r.status);
        const snap = await r.json();
        const merged = mergeSnapshot(t, snap);
        if (JSON.stringify(merged) !== JSON.stringify(t)) changed = true;
        return merged;
      } catch {
        return t;
      }
    })
  );

  if (changed) {
    localStorage.setItem("tasks", JSON.stringify(next));
    localStorage.setItem("tasks_last_update", String(Date.now()));
    render();
  }
}

window.addEventListener("storage", (e) => {
  if (e.key === "tasks" || e.key === "tasks_last_update") render();
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) render();
});

render();
pollOnce();
setInterval(pollOnce, 4000);

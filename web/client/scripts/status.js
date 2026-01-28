const DEFAULT_AUTO_DEMO = false;
const POLL_MS = 2500;
const STEP_MS = 1800;
const IGNORE_IDS = new Set([
  "task_1001",
  "task_1002",
  "task_1003",
  "task_status_demo",
]);

const url = new URL(location.href);
const AUTO_DEMO =
  url.searchParams.get("demo") === "1" ||
  localStorage.getItem("auto_demo") === "1" ||
  DEFAULT_AUTO_DEMO;

const ORDER = [
  "queued",
  "preprocessing",
  "training",
  "optimizing",
  "packaging",
  "completed",
  "failed",
];

const STEP_META = {
  queued: {
    title: "Queued",
    desc: "Your task is waiting in the queue.",
    icon: iconClock(),
  },
  preprocessing: {
    title: "Preprocessing",
    desc: "Your dataset is being prepared.",
    icon: iconSliders(),
  },
  training: {
    title: "Training",
    desc: "The model is being trained on your data.",
    icon: iconBolt(),
  },
  optimizing: {
    title: "Optimizing",
    desc: "The model is being optimized for the target device.",
    icon: iconWand(),
  },
  packaging: {
    title: "Packaging",
    desc: "The model is being packaged for deployment.",
    icon: iconBox(),
  },
  completed: {
    title: "Completed",
    desc: "Your model is ready for download.",
    icon: iconCheck(),
  },
  failed: {
    title: "Failed",
    desc: "The task did not complete successfully.",
    icon: iconX(),
  },
};

const dsNameEl = document.getElementById("dsName");
const emptyEl = document.getElementById("empty");
const cardEl = document.getElementById("statusCard");
const timelineEl = document.getElementById("timeline");
const summaryEl = document.getElementById("summary");
const ctaRow = document.getElementById("ctaRow");
const viewBtn = document.getElementById("viewResults");


// Header UI handled by header.js


function readTasks() {
  try {
    const raw = localStorage.getItem("tasks");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
function writeTasks(list) {
  localStorage.setItem("tasks", JSON.stringify(list));
  localStorage.setItem("tasks_last_update", String(Date.now()));
  return list;
}
function upsertTask(task) {
  const list = readTasks();
  const i = list.findIndex((j) => j.id === task.id);
  if (i >= 0) list[i] = { ...list[i], ...task };
  else list.push(task);
  return writeTasks(list);
}
function getTask(id) {
  return readTasks().find((j) => j.id === id) || null;
}

function seedIfMissing(id) {
  if (!IGNORE_IDS.has(id)) return;
  let list = readTasks();
  if (!Array.isArray(list)) list = [];
  if (!list.some((j) => j.id === id)) {
    const demo = {
      id,
      datasetName: "edge_device_families.csv",
      taskType: "image",
      device_family: "mcu_mid_dsp",
      status: "queued",
      createdAt: Date.now(),
      mock: true,
    };
    upsertTask(demo);
  }
}

function render(task) {
  if (!task) {
    emptyEl.hidden = false;
    cardEl.hidden = true;
    return;
  }
  emptyEl.hidden = true;
  cardEl.hidden = false;

  dsNameEl.textContent = task.datasetName || "—";

  summaryEl.textContent =
    task.status === "completed"
      ? "Your task has completed successfully."
      : task.status === "failed"
      ? "Your task failed."
      : "Your task is currently in progress.";

  timelineEl.innerHTML = "";
  const currentIndex = Math.max(0, ORDER.indexOf(task.status || "queued"));

  ORDER.forEach((key, idx) => {
    if (key === "failed" && task.status !== "failed") return;

    const state =
      task.status === "failed"
        ? key === "failed"
          ? "error"
          : ORDER.indexOf(key) < ORDER.indexOf("failed")
          ? "done"
          : "upcoming"
        : idx < currentIndex
        ? "done"
        : idx === currentIndex
        ? "active"
        : "upcoming";

    const li = document.createElement("li");
    li.className = `step ${state}`;
    li.setAttribute("role", "listitem");
    if (state === "active") li.setAttribute("aria-current", "step");

    const dot = document.createElement("div");
    dot.className = "dot";
    dot.innerHTML = STEP_META[key].icon;

    const content = document.createElement("div");
    const h = document.createElement("h3");
    h.className = "step-title";
    h.textContent = STEP_META[key].title;

    const p = document.createElement("p");
    p.className = "step-desc";
    p.textContent = STEP_META[key].desc;

    content.appendChild(h);
    content.appendChild(p);

    if (
      state === "active" &&
      task.status !== "completed" &&
      task.status !== "failed"
    ) {
      const pulse = document.createElement("div");
      pulse.className = "pulse";
      const sp = document.createElement("span");
      sp.className = "spinner";
      const t = document.createElement("span");
      t.textContent = "In progress…";
      pulse.appendChild(sp);
      pulse.appendChild(t);
      content.appendChild(pulse);
    }



    li.appendChild(dot);
    li.appendChild(content);
    timelineEl.appendChild(li);
  });

  ctaRow.hidden = false;
  if (task.status === "completed") {
    const href = `./results.html?id=${encodeURIComponent(task.id)}`;
    viewBtn.href = href;
    viewBtn.removeAttribute("disabled");
    viewBtn.classList.remove("disabled");
    viewBtn.textContent = "View Results";
    viewBtn.setAttribute(
      "aria-label",
      `View results for ${task.datasetName || "model"}`
    );
  } else if (task.status === "failed") {
    const href = `./results.html?id=${encodeURIComponent(task.id)}`;
    viewBtn.href = href;
    viewBtn.removeAttribute("disabled");
    viewBtn.classList.remove("disabled");
    viewBtn.textContent = "View Failure Details";
    viewBtn.setAttribute("aria-label", "View failure details");
  } else {
    viewBtn.href = "javascript:void(0)";
    viewBtn.setAttribute("disabled", "true");
    viewBtn.classList.add("disabled");
    viewBtn.textContent = "Generating…";
  }
}

let demoTimer = null;

function startDemoProgress(taskId) {
  if (!AUTO_DEMO) return;

  if (demoTimer) clearInterval(demoTimer);

  demoTimer = setInterval(() => {
    const task = getTask(taskId);
    if (!task) return;

    if (task.status === "completed" || task.status === "failed") {
      clearInterval(demoTimer);
      demoTimer = null;
      if (task.status === "completed") {
        const to = `./results.html?id=${encodeURIComponent(taskId)}`;
        setTimeout(() => {
          if (location.pathname.endsWith("status.html")) {
            location.replace(to);
          }
        }, 600);
      }
      return;
    }

    const i = ORDER.indexOf(task.status || "queued");
    const next = ORDER[i + 1] || "completed";
    task.status = next;
    upsertTask(task);
    render(task);
  }, STEP_MS);
}

let pollTimer = null;
let pollErrorCount = 0;

function apiBase() {
  return (
    localStorage.getItem("api_base") ||
    new URL(location.href).searchParams.get("api") ||
    "http://127.0.0.1:8000"
  );
}

async function pollTask(taskId) {
  if (pollTimer) clearInterval(pollTimer);

  if (IGNORE_IDS.has(taskId)) {
    if (AUTO_DEMO) startDemoProgress(taskId);
    return;
  }

  const doPoll = async () => {
    const existing = getTask(taskId);

    if (
      existing &&
      (existing.status === "completed" || existing.status === "failed")
    ) {
      if (pollTimer) clearInterval(pollTimer);
      return;
    }

    try {
      const resp = await fetch(
        `${apiBase()}/tasks/${encodeURIComponent(taskId)}`
      );
      if (resp.status === 404) {
        pollTimer && clearInterval(pollTimer);
        const failed = {
          ...(existing || { id: taskId }),
          status: "failed",
          phase: "failed",
          errorMessage: "This task no longer exists on the server (404).",
          updatedAt: Date.now(),
          mock: false,
        };
        upsertTask(failed);
        render(failed);
        setTimeout(() => {
          if (location.pathname.endsWith("status.html"))
            location.href = "./tasks.html";
        }, 4000);
        return;
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const snap = await resp.json();
      pollErrorCount = 0;

      const merged = {
        ...(existing || {}),
        id: snap.id || taskId,
        status: snap.status?.status || snap.sched_state || existing?.status || "queued",
        phase: snap.status?.stage_idx !== undefined ? ORDER[snap.status.stage_idx] : (snap.phase || existing?.phase),
        progress:
          typeof snap.progress === "number"
            ? snap.progress
            : existing?.progress,
        errorMessage: snap.error || existing?.errorMessage,
        device_family: snap.device_family || existing?.device_family,
        metrics: snap.metrics || existing?.metrics,
        asset_paths: snap.asset_paths || existing?.asset_paths,
        model_id: snap.model_id || existing?.model_id,
        updatedAt: Date.now(),
        mock: false,
      };

      upsertTask(merged);
      render(merged);

      if (merged.status === "completed" || merged.status === "failed") {
        pollTimer && clearInterval(pollTimer);
        if (merged.status === "failed") {
          setTimeout(() => {
            if (location.pathname.endsWith("status.html"))
              location.href = "./tasks.html";
          }, 4000);
        } else if (merged.status === "completed") {
          const to = `./results.html?id=${encodeURIComponent(taskId)}`;
          setTimeout(() => {
            if (location.pathname.endsWith("status.html")) location.replace(to);
          }, 800);
        }
      }
    } catch {
      pollErrorCount += 1;
      if (!existing) seedIfMissing(taskId);
      const task = getTask(taskId);
      render(task);
      if ((task?.mock || AUTO_DEMO) && !demoTimer) startDemoProgress(taskId);
      if (pollErrorCount >= 5 && pollTimer) clearInterval(pollTimer);
    }
  };

  await doPoll();
  pollTimer = setInterval(doPoll, POLL_MS);
}

window.addEventListener("storage", (e) => {
  if (e.key === "tasks" || e.key === "tasks_last_update") {
    const task = getTask(CURRENT_ID);
    render(task);
  }
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) render(getTask(CURRENT_ID));
});

const CURRENT_ID = (() => {
  const u = new URL(location.href);
  let id = u.searchParams.get("id");
  if (!id) {
    id = "task_status_demo";
    u.searchParams.set("id", id);
    history.replaceState({}, "", u.toString());
  }
  return id;
})();

seedIfMissing(CURRENT_ID);
render(getTask(CURRENT_ID));
if (AUTO_DEMO && IGNORE_IDS.has(CURRENT_ID)) startDemoProgress(CURRENT_ID);
pollTask(CURRENT_ID);

function iconCheck() {
  return `<svg class="hi" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>`;
}
function iconX() {
  return `<svg class="hi" viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>`;
}
function iconClock() {
  return `<svg class="hi" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8v4l3 3"/><path d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0"/></svg>`;
}
function iconSliders() {
  return `<svg class="hi" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 4H7"/><path d="M14 4v6"/><path d="M3 10h18"/><path d="M7 10v10"/></svg>`;
}
function iconBolt() {
  return `<svg class="hi" viewBox="0 0 24 24" aria-hidden="true"><path d="M13 3L4 14h7l-1 7 9-11h-7l1-7z"/></svg>`;
}
function iconWand() {
  return `<svg class="hi" viewBox="0 0 24 24" aria-hidden="true"><path d="m15 4 5 5-9 9H6v-5z"/><path d="M5 15l4 4"/></svg>`;
}
function iconBox() {
  return `<svg class="hi" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="M3.27 6.96 12 12.01l8.73-5.05"/><path d="M12 22.08V12"/></svg>`;
}

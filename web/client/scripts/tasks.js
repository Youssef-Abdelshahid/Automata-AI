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
const ALLOWED_TYPES = new Set(["image", "audio", "sensor"]);

const ORDER = [
  "queued",
  "preprocessing",
  "training",
  "optimizing",
  "packaging",
  "completed",
  "failed",
];

function normalizeTask(t, idx = 0) {
  const id = t?.id || `task_${Date.now().toString(36)}_${idx}`;
  const title = t?.title || "Untitled Task";
  const datasetName = String(t?.datasetName ?? t?.name ?? "Untitled Dataset");
  const taskType = ALLOWED_TYPES.has(t?.taskType) ? t.taskType : "image";
  const device = t?.device || "Generic Device"; // Now represents Family Name
  const status = ALLOWED_STATUS.has(t?.status) ? t.status : "queued";
  const createdAt =
    typeof t?.createdAt === "number" && Number.isFinite(t.createdAt)
      ? t.createdAt
      : Date.now();
  return { ...t, id, title, datasetName, taskType, device, status, createdAt };
}

function readTasksRaw() {
  try {
    const raw = localStorage.getItem("tasks");
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveTasks(list) {
  const normalized = list.map(normalizeTask);
  localStorage.setItem("tasks", JSON.stringify(normalized));
  localStorage.setItem("tasks_last_update", String(Date.now()));
  return normalized;
}


function loadTasks() {
  let list = readTasksRaw();
  if (!Array.isArray(list)) {
    list = seed.slice();
  }
  return saveTasks(list);
}

const searchInput = document.getElementById("searchInput");
const statusBtn = document.getElementById("statusBtn");
const statusList = document.getElementById("statusList");
const statusLabel = document.getElementById("statusLabel");
const typeBtn = document.getElementById("typeBtn");
const typeList = document.getElementById("typeList");
const typeLabel = document.getElementById("typeLabel");
const tbody = document.getElementById("tbody");
const emptyState = document.getElementById("emptyState");
const clearFilters = document.getElementById("clearFilters");

function getParams() {
  const u = new URL(location.href);
  return {
    q: u.searchParams.get("q") || "",
    status: u.searchParams.get("status") || "",
    type: u.searchParams.get("type") || "",
    id: u.searchParams.get("id") || "",
    access: u.searchParams.get("access") || "",
  };
}
function setParams(p) {
  const u = new URL(location.href);
  u.searchParams.set("q", p.q || "");
  u.searchParams.set("status", p.status || "");
  u.searchParams.set("type", p.type || "");
  if (p.id) u.searchParams.set("id", p.id); else u.searchParams.delete("id");
  history.replaceState({}, "", u.toString());
}

function closeAllDropdowns(exceptEl = null) {
  document.querySelectorAll(".dropdown.open").forEach((dd) => {
    if (dd !== exceptEl) {
      dd.classList.remove("open");
      dd.querySelector(".dropdown-btn")?.setAttribute("aria-expanded", "false");
    }
  });
}
let dropdownGlobalListenerInstalled = false;

function setupDropdown(btn, list, labelEl, key) {
  const parent = btn.parentElement;
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const willOpen = !parent.classList.contains("open");
    closeAllDropdowns(parent);
    parent.classList.toggle("open", willOpen);
    btn.setAttribute("aria-expanded", String(willOpen));
  });
  parent.addEventListener("click", (e) => e.stopPropagation());
  list.addEventListener("click", (e) => e.stopPropagation());

  if (!dropdownGlobalListenerInstalled) {
    dropdownGlobalListenerInstalled = true;
    document.addEventListener("click", () => closeAllDropdowns(), {
      passive: true,
    });
    document.addEventListener(
      "keydown",
      (e) => e.key === "Escape" && closeAllDropdowns()
    );
  }

  list.querySelectorAll("li").forEach((li) => {
    li.addEventListener("click", () => {
      list
        .querySelectorAll("li")
        .forEach((n) => n.classList.remove("selected"));
      li.classList.add("selected");
      const value = li.getAttribute("data-value") || "";
      labelEl.textContent =
        key === "status"
          ? value || "Status"
          : value
            ? "Task: " + value
            : "Task Type";
      filters[key] = value;
      setParams(filters);
      closeAllDropdowns();
      render();
    });
  });

  return {
    setValue(value) {
      const item =
        list.querySelector(`li[data-value="${value}"]`) ||
        list.querySelector('li[data-value=""]');
      list
        .querySelectorAll("li")
        .forEach((n) => n.classList.remove("selected"));
      item.classList.add("selected");
      labelEl.textContent =
        key === "status"
          ? value || "Status"
          : value
            ? "Task: " + value
            : "Task Type";
    },
  };
}

let tasks = loadTasks();
let filters = getParams();
const statusDd = setupDropdown(statusBtn, statusList, statusLabel, "status");
const typeDd = setupDropdown(typeBtn, typeList, typeLabel, "type");

function relativeTime(ts) {
  const diff = Date.now() - (typeof ts === "number" ? ts : Date.now());
  const mins = Math.round(diff / 60000);
  if (mins < 60) return `about ${mins} minute${mins === 1 ? "" : "s"} ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `about ${hrs} hour${hrs === 1 ? "" : "s"} ago`;
  const days = Math.round(hrs / 24);
  return `about ${days} day${days === 1 ? "" : "s"} ago`;
}
function capitalize(s) {
  const str = s == null ? "" : String(s);
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function render() {
  tasks = loadTasks();
  searchInput.value = filters.q || "";
  statusDd.setValue(filters.status || "");
  typeDd.setValue(filters.type || "");

  let list = tasks
    .slice()
    .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));

  if (filters.id) {
    list = list.filter((t) => t.id === filters.id);
  } else {
    if (filters.q)
      list = list.filter((t) =>
        (t.title || "").toLowerCase().includes(filters.q.toLowerCase()) ||
        (t.datasetName || "").toLowerCase().includes(filters.q.toLowerCase())
      );
    if (filters.status) list = list.filter((t) => t.status === filters.status);
    if (filters.type) list = list.filter((t) => t.taskType === filters.type);
  }

  tbody.innerHTML = "";
  if (!list.length) {
    emptyState.hidden = false;
    return;
  }
  emptyState.hidden = true;

  for (const t of list) {
    const isCompleted = t.status === "completed";
    const isFailed = t.status === "failed";

    let actionHref = `./status.html?id=${encodeURIComponent(t.id)}`;
    let actionText = "Status";

    if (isCompleted) {
      actionHref = `./results.html?id=${encodeURIComponent(t.id)}`;
      actionText = "Results";
    } else if (isFailed) {
      actionHref = `./results.html?id=${encodeURIComponent(t.id)}`;
      actionText = "Failure Details";
    }

    // Check for Contribute capabilities
    const canEdit = filters.access === "contribute";
    let extraActions = "";
    if (canEdit && filters.id === t.id) {
      // Point to user_profile or a dedicated edit page. 
      // Since editing is currently modal-based on user_profile, we might need a workaround or link back to user_profile?
      // BUT the prompt said "tasks page display anything now?" implying we stay here.
      // Wait, editing is only on user_profile.html currently.
      // If I link to user_profile.html, it shows ALL tasks unless filtered. 
      // Ideally we'd have a standalone edit page. 
      // For now, I'll link to `new_task.html?edit=id` if that existed, OR `user_profile.html`.
      // Let's assume for this task 'Contribute' just means 'Go to Profile' or similar? 
      // "can you make the share button... view only or contribute". 
      // The prompt implies the RECEIVER can contribute. If the receiver is NOT the owner, they can't see it on their profile?
      // This implies a shared 'edit' capability. 
      // Since we don't have a backend sharing model yet ("local" per user), and this is likely a demo on the SAME machine/browser:
      // We can link to `user_profile.html` (if own task) or just enable the button.
      // Actually, earlier we refactored 'Add Task' to `new_task.html`. 
      // Does `new_task.js` support editing? No, it's 'new'.
      // `user_profile.js` has `openEditTask`.
      // I will just add a dummy "Edit" button that alerts "Editing not implemented for shared tasks yet" OR links to status if that was key.
      // BETTER: Link to `user_profile.html` if it's the same user.
      // User request: "view only or contribute". 
      // "link copied to clipboard" -> "tasks.html?id=...&access=contribute".
      // If I paste this, I see the task. If 'contribute', I should see an Edit button.
      // I'll add the button. For now, let's make it link to `user_profile.html` as a placeholder for "Contribute", 
      // OR purely visual since there's no backend auth for sharing yet.
      // Let's make it link to `new_task.html` (simulating a fork?) or just visual.
      // I'll add a visual "Edit" button for now that goes to `user_profile.html`.
      extraActions = `<a class="tasks-btn-ghost" href="./user_profile.html" style="margin-left:8px; color:var(--brand);">Edit</a>`;
    }

    const row = document.createElement("div");
    row.className = "trow";
    row.innerHTML = `
      <div data-label="Status"><span class="pill ${t.status}"><span class="dot"></span> ${capitalize(t.status)}</span></div>
      <div data-label="Task Title"><a class="filename" href="${actionHref}">${t.title}</a></div>
      <div data-label="Dataset" class="t-main">${t.datasetName}</div>
      <div data-label="Type" class="t-main">${capitalize(t.taskType)}</div>
      <div data-label="Target Device" class="t-main">${t.device}</div>
      <div data-label="Created" class="t-meta">${relativeTime(t.createdAt)}</div>
      <div data-label="Actions" class="right">
        <a class="tasks-btn-ghost" href="${actionHref}">${actionText}</a>
        ${extraActions}
      </div>`;
    tbody.appendChild(row);
  }
}

function mergeSnapshot(local, snap) {
  if (!snap) return local;

  const statusVal = snap.status?.status || snap.sched_state || local.status || "queued";
  // We don't have normalizeStatus here but we can just trust the API or simple check
  // Actually, let's keep it simple as it was, but extract correctly.

  // Map stage_idx to string phase if available
  let phase = snap.phase || local.phase;
  if (snap.status?.stage_idx !== undefined && ORDER[snap.status.stage_idx]) {
    phase = ORDER[snap.status.stage_idx];
  }

  return {
    ...local,
    status: ALLOWED_STATUS.has(statusVal) ? statusVal : local.status,
    phase: phase,
    progress:
      typeof snap.progress === "number" ? snap.progress : local.progress,
    metrics: snap.metrics || local.metrics,
    model_id: snap.model_id || local.model_id,
    asset_paths: snap.asset_paths || local.asset_paths,
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

  const IGNORE_IDS = new Set([
    "task_1001",
    "task_1002",
    "task_1003",
  ]);
  const current = readTasksRaw();
  if (!Array.isArray(current) || !current.length) return;

  let changed = false;
  const updated = await Promise.all(
    current.map(async (t) => {
      const id = t?.id;
      const status = t?.status || "queued";
      if (!id || IGNORE_IDS.has(id) || !RUNNING_SET.has(status)) return t;
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
    saveTasks(updated);
    render();
  }
}

window.addEventListener("storage", (e) => {
  if (e.key === "tasks" || e.key === "tasks_last_update") {
    tasks = loadTasks();
    render();
  }
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    tasks = loadTasks();
    render();
  }
});

searchInput.addEventListener("input", (e) => {
  filters.q = e.target.value;
  setParams(filters);
  render();
});
clearFilters.addEventListener("click", () => {
  filters = { q: "", status: "", type: "" };
  setParams(filters);
  render();
});

render();
pollOnce();
setInterval(pollOnce, 4000);

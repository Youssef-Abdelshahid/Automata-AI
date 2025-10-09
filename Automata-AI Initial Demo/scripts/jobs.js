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
  };
}
function setParams(p) {
  const u = new URL(location.href);
  u.searchParams.set("q", p.q || "");
  u.searchParams.set("status", p.status || "");
  u.searchParams.set("type", p.type || "");
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
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeAllDropdowns();
    });
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

let jobs = loadJobs();
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
function escapeHtml(s) {
  const str = s == null ? "" : String(s);
  return str.replace(
    /[&<>"']/g,
    (m) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
        m
      ])
  );
}
function capitalize(s) {
  const str = s == null ? "" : String(s);
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function render() {
  searchInput.value = filters.q || "";
  statusDd.setValue(filters.status || "");
  typeDd.setValue(filters.type || "");

  let list = jobs.slice().sort((a, b) => {
    const diff = (b.createdAt || 0) - (a.createdAt || 0);
    return diff !== 0
      ? diff
      : String(a.datasetName || "").localeCompare(String(b.datasetName || ""));
  });

  if (filters.q)
    list = list.filter((j) =>
      (j.datasetName || "").toLowerCase().includes(filters.q.toLowerCase())
    );
  if (filters.status) list = list.filter((j) => j.status === filters.status);
  if (filters.type) list = list.filter((j) => j.taskType === filters.type);

  tbody.innerHTML = "";
  if (!list.length) {
    emptyState.hidden = false;
    return;
  }
  emptyState.hidden = true;

  for (const j of list) {
    const isCompleted = j.status === "completed";
    const actionHref = isCompleted
      ? `./results.html?id=${encodeURIComponent(j.id)}`
      : `./status.html?id=${encodeURIComponent(j.id)}`;
    const actionText = isCompleted ? "Results" : "Status";
    const actionIcon = isCompleted
      ? `<svg width="16" height="16" viewBox="0 0 24 24"><path d="M8 8h12v12H8zM4 4h12v12H4z" fill="none" stroke="currentColor" stroke-width="2"/></svg>`
      : `<svg width="16" height="16" viewBox="0 0 24 24"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>`;

    const row = document.createElement("div");
    row.className = "trow";

    const c1 = document.createElement("div");
    c1.innerHTML = `<span class="pill ${
      j.status
    }"><span class="dot"></span> ${capitalize(j.status)}</span>`;

    const c2 = document.createElement("div");
    c2.innerHTML = `
      <a class="filename" href="${actionHref}">${escapeHtml(j.datasetName)}</a>
      <div class="meta">
        <svg width="16" height="16" viewBox="0 0 24 24"><path d="M4 4h16v16H4z" fill="none" stroke="currentColor" stroke-width="2"/></svg>
        ${capitalize(j.taskType)}
      </div>`;

    const c3 = document.createElement("div");
    c3.textContent = j.device || "—";
    const c4 = document.createElement("div");
    c4.textContent = relativeTime(j.createdAt);

    const c5 = document.createElement("div");
    c5.className = "right";
    const btn = document.createElement("a");
    btn.className = "btn-ghost";
    btn.href = actionHref;
    btn.innerHTML = `${actionIcon} ${actionText}`;
    c5.appendChild(btn);

    [c1, c2, c3, c4, c5].forEach((el) => row.appendChild(el));
    tbody.appendChild(row);
  }
}

window.addEventListener("storage", (e) => {
  if (e.key === "jobs" || e.key === "jobs_last_update") {
    jobs = loadJobs();
    render();
  }
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    jobs = loadJobs();
    render();
  }
});

window.JOBS_API = {
  add(job) {
    const current = readJobsRaw();
    const list = Array.isArray(current) ? current : [];
    const added = normalizeJob(job, list.length);
    list.push(added);
    jobs = saveJobs(list); 
    render();
    return added;
  },
};


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

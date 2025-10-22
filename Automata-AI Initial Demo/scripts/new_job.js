import { DEVICE_FAMILIES } from "./util.js";

const clamp = (v, min, max) => Math.min(Math.max(v, min), max);
const formatKB = (kb) => `${Math.round(kb).toLocaleString()} KB`;
const formatMB = (mb) => `${Number(mb).toFixed(mb < 1 ? 2 : 1)} MB`;

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
  document.addEventListener(
    "keydown",
    (e) => e.key === "Escape" && setOpen(false)
  );
  logoutBtn.addEventListener("click", () => {
    clearSession();
    location.href = "./auth.html";
  });
})();

const progressEl = document.getElementById("progress");
const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");
const err1 = document.getElementById("errStep1");
const err2 = document.getElementById("errStep2");
const errFamily = document.getElementById("errFamily");

const next1 = document.getElementById("next1");
const next2 = document.getElementById("next2");
const back2 = document.getElementById("back2");
const back3 = document.getElementById("back3");
const submitJob = document.getElementById("submitJob");

const taskCards = Array.from(document.querySelectorAll(".task-card"));

const drop = document.getElementById("drop");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const fileClear = document.getElementById("fileClear");

const familyDd = document.getElementById("familyDd");
const familyBtn = document.getElementById("familyBtn");
const familyList = document.getElementById("familyList");
const familyLabel = document.getElementById("familyLabel");
const familyNote = document.getElementById("familyNote");

const accelToggle = document.getElementById("accelToggle");
const accelNote = document.getElementById("accelNote");

const ramRange = document.getElementById("ramRange");
const flashRange = document.getElementById("flashRange");
const cpuRange = document.getElementById("cpuRange");
const ramVal = document.getElementById("ramVal");
const flashVal = document.getElementById("flashVal");
const cpuVal = document.getElementById("cpuVal");
const targetColumnInput = document.getElementById("targetColumn");

const state = {
  step: 1,
  taskType: null,
  dataset: { name: null, size: 0, type: null },
  device: {
    familyId: null,
    ramKB: null,
    flashMB: null,
    cpuMHz: null,
    accelerator: false,
  },
};

function setStep(n) {
  state.step = n;
  step1.hidden = n !== 1;
  step2.hidden = n !== 2;
  step3.hidden = n !== 3;
  progressEl.style.width = n === 1 ? "33%" : n === 2 ? "66%" : "100%";
}

taskCards.forEach((card) => {
  card.addEventListener("click", () => {
    taskCards.forEach((c) => c.setAttribute("aria-checked", "false"));
    card.setAttribute("aria-checked", "true");
    state.taskType = card.dataset.type;
    err1.textContent = "";
  });
  card.addEventListener("keydown", (e) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      card.click();
    }
  });
});
next1.addEventListener("click", () => {
  if (!state.taskType) {
    err1.textContent = "Please choose a task type.";
    return;
  }
  setStep(2);
});

const MAX_SIZE = 500 * 1024 * 1024;
const ACCEPT_EXTS = [".csv"];

function getExt(name = "") {
  const lower = name.toLowerCase().trim();
  const dot = lower.lastIndexOf(".");
  return dot >= 0 ? lower.slice(dot) : "";
}
function isAllowed(file) {
  const ext = getExt(file?.name || "");
  return ACCEPT_EXTS.includes(ext);
}

function updateUploadUI(hasFile) {
  if (!hasFile) {
    fileName.textContent = "";
    fileSize.textContent = "";
    fileInfo.hidden = true;
    fileInfo.style.display = "none";
  } else {
    fileInfo.hidden = false;
    fileInfo.style.display = "";
  }
}

function clearUploadFile() {
  state.dataset = { name: null, size: 0, type: null };

  fileInput.value = "";
  try {
    const dt = new DataTransfer();
    fileInput.files = dt.files;
  } catch {}

  updateUploadUI(false);
}

function setUploadFile(file) {
  if (!file) return;
  if (!isAllowed(file)) {
    err2.textContent = "Unsupported file type. Please upload a CSV file.";
    clearUploadFile();
    return;
  }
  if (file.size > MAX_SIZE) {
    err2.textContent = "File is too large. Maximum allowed size is 500 MB.";
    clearUploadFile();
    return;
  }

  state.dataset = {
    name: file.name,
    size: file.size || 0,
    type: file.type || "application/octet-stream",
  };
  fileName.textContent = file.name;
  fileSize.textContent = `• ${(file.size / 1024).toFixed(1)} KB`;
  err2.textContent = "";
  updateUploadUI(true);
}

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("dragover");
});
drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("dragover");
  const f = e.dataTransfer?.files?.[0];
  if (f) setUploadFile(f);
});
fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) setUploadFile(f);
});
fileClear.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  clearUploadFile();
});

back2.addEventListener("click", () => setStep(1));
next2.addEventListener("click", () => {
  if (!state.dataset.name) {
    err2.textContent = "Please upload a dataset file.";
    return;
  }
  if (!targetColumnInput.value.trim()) {
    err2.textContent = "Please enter the name of the target column.";
    return;
  }
  setStep(3);
});

const getFamily = (id) => DEVICE_FAMILIES.find((f) => f.id === id) || null;
const midpoint = ([a, b]) => a + (b - a) / 2;

(function buildFamilyDropdown() {
  familyList.innerHTML = "";
  DEVICE_FAMILIES.forEach((f) => {
    const li = document.createElement("li");
    li.textContent = f.name;
    li.setAttribute("role", "option");
    li.dataset.id = f.id;
    li.addEventListener("click", () => {
      familyList
        .querySelectorAll("li")
        .forEach((n) => n.classList.remove("selected"));
      li.classList.add("selected");
      familyLabel.textContent = f.name;
      familyBtn.setAttribute("aria-expanded", "false");
      familyDd.classList.remove("open");

      state.device.familyId = f.id;
      errFamily.textContent = "";
      applyFamilyDefaults(f);
    });
    familyList.appendChild(li);
  });

  familyBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const willOpen = !familyDd.classList.contains("open");
    closeAllDropdowns(familyDd);
    familyDd.classList.toggle("open", willOpen);
    familyBtn.setAttribute("aria-expanded", String(willOpen));
  });
  document.addEventListener("click", () => closeAllDropdowns());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAllDropdowns();
  });
})();
function closeAllDropdowns(except) {
  document.querySelectorAll(".dropdown.open").forEach((dd) => {
    if (dd !== except) {
      dd.classList.remove("open");
      dd.querySelector(".dropdown-btn")?.setAttribute("aria-expanded", "false");
    }
  });
}

function getAccelCapability(fam) {
  const raw = fam?.accelerator ?? fam?.specs_hint?.accelerator;
  if (raw === true) return "builtin";
  if (raw === "optional") return "optional";
  if (raw === false) return "none";
  if (typeof raw === "string") {
    const v = raw.trim().toLowerCase();
    if (v === "true") return "builtin";
    if (v === "optional") return "optional";
  }
  return "none";
}

function applyFamilyDefaults(fam) {
  const hints = fam?.specs_hint || {
    ramKB: [32, 256],
    flashMB: [1, 8],
    cpuMHz: [80, 240],
  };

  ramRange.min = hints.ramKB[0];
  ramRange.max = hints.ramKB[1];
  const ramMid = midpoint(hints.ramKB);
  ramRange.value = clamp(
    state.device.ramKB ?? ramMid,
    hints.ramKB[0],
    hints.ramKB[1]
  );
  ramVal.textContent = formatKB(Number(ramRange.value));

  flashRange.min = hints.flashMB[0];
  flashRange.max = hints.flashMB[1];
  flashRange.step = 0.1;
  const flashMid = midpoint(hints.flashMB);
  flashRange.value = clamp(
    state.device.flashMB ?? flashMid,
    hints.flashMB[0],
    hints.flashMB[1]
  );
  flashVal.textContent = formatMB(Number(flashRange.value));

  cpuRange.min = hints.cpuMHz[0];
  cpuRange.max = hints.cpuMHz[1];
  const cpuMid = midpoint(hints.cpuMHz);
  cpuRange.value = clamp(
    state.device.cpuMHz ?? cpuMid,
    hints.cpuMHz[0],
    hints.cpuMHz[1]
  );
  cpuVal.textContent = `${Math.round(Number(cpuRange.value))} MHz`;

  const capability = getAccelCapability(fam);
  let note = "";
  if (capability === "builtin") {
    accelToggle.setAttribute("aria-checked", "true");
    accelToggle.setAttribute("disabled", "");
    state.device.accelerator = true;
    note = "This family includes a hardware accelerator.";
  } else if (capability === "optional") {
    accelToggle.removeAttribute("disabled");
    accelToggle.setAttribute(
      "aria-checked",
      String(Boolean(state.device.accelerator))
    );
    note = "Optional accelerator available — enable if your target has one.";
  } else {
    accelToggle.setAttribute("aria-checked", "false");
    accelToggle.setAttribute("disabled", "");
    state.device.accelerator = false;
    note = "Not supported on this device family.";
  }
  accelNote.textContent = note;

  familyNote.textContent = fam?.note || "";
}

accelToggle.addEventListener("click", () => {
  if (accelToggle.hasAttribute("disabled")) return;
  const on = accelToggle.getAttribute("aria-checked") === "true";
  const next = !on;
  accelToggle.setAttribute("aria-checked", String(next));
  state.device.accelerator = next;
});

ramRange.addEventListener("input", () => {
  ramVal.textContent = formatKB(Number(ramRange.value));
});
flashRange.addEventListener("input", () => {
  flashVal.textContent = formatMB(Number(flashRange.value));
});
cpuRange.addEventListener("input", () => {
  cpuVal.textContent = `${Math.round(Number(cpuRange.value))} MHz`;
});

back3.addEventListener("click", () => setStep(2));

submitJob.addEventListener("click", async () => {
  if (!state.device.familyId) {
    errFamily.textContent = "Please select a device family.";
    return;
  }
  errFamily.textContent = "";

  const fileToUpload = fileInput.files[0];
  const targetColumn = targetColumnInput.value.trim();
  if (!fileToUpload || !targetColumn) {
    alert("Missing file or target column!");
    return;
  }

  const apiBase = localStorage.getItem("api_base") || "http://127.0.0.1:8000";
  const apiUrl = `${apiBase}/train`;

  const formData = new FormData();
  formData.append("file", fileToUpload);
  formData.append("target_column", targetColumn);
  formData.append("device_family", state.device.familyId);
  formData.append("ram_kb", ramRange.value);
  formData.append("flash_mb", flashRange.value);
  formData.append("cpu_mhz", cpuRange.value);

  submitJob.disabled = true;
  submitJob.textContent = "Submitting Job...";

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let msg = "Failed to submit job.";
      try {
        const err = await response.json();
        msg = err.detail || msg;
      } catch {}
      throw new Error(msg);
    }

    const result = await response.json();
    const jobId = result.job_id || result.model_id || crypto.randomUUID();

    const fam = DEVICE_FAMILIES.find((f) => f.id === state.device.familyId);
    const job = {
      id: jobId,
      datasetName: state.dataset.name || "Untitled Dataset",
      taskType: state.taskType || "generic",
      device_family: state.device.familyId,
      device: fam?.name || "Unknown Family",
      target_column: targetColumn,
      status: "queued",
      progress: 0,
      createdAt: Date.now(),
      mock: false,
    };

    const raw = localStorage.getItem("jobs");
    const list = raw ? JSON.parse(raw) : [];
    list.push(job);
    localStorage.setItem("jobs", JSON.stringify(list));
    localStorage.setItem("jobs_last_update", String(Date.now()));

    location.href = `./status.html?id=${encodeURIComponent(jobId)}`;
  } catch (err) {
    console.error("Submit failed:", err);
    alert(`Error: ${err.message}`);
    submitJob.disabled = false;
    submitJob.textContent = "Submit Job";
  }
});

setStep(1);

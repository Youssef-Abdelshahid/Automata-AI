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
const submitTask = document.getElementById("submitTask");

const taskCards = Array.from(document.querySelectorAll(".task-card"));
const taskTitleInput = document.getElementById("taskTitle");
const taskDescInput = document.getElementById("taskDesc");
const visRadios = document.getElementsByName("visibility");

const drop = document.getElementById("drop");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const fileClear = document.getElementById("fileClear");
const uploadHint = document.getElementById("uploadHint");

const targetColGroup = document.getElementById("targetColGroup");
const targetColumnInput = document.getElementById("targetColumn");
const numClassesGroup = document.getElementById("numClassesGroup");
const numClassesInput = document.getElementById("numClasses");

const familyDd = document.getElementById("familyDd");
const familyBtn = document.getElementById("familyBtn");
const familyList = document.getElementById("familyList");
const familyLabel = document.getElementById("familyLabel");
const familyNote = document.getElementById("familyNote");

const modelExtSelect = document.getElementById("modelExtSelect");
const manualSpecsHint = document.getElementById("manualSpecsHint");

const roleToggle = document.getElementById("roleToggle");
const roleLabel = document.getElementById("roleLabel");
const powerUserControls = document.getElementById("powerUserControls");
const generalPowerControls = document.getElementById("generalPowerControls");
const dataStrategyControls = document.getElementById("dataStrategyControls");

const trainSpeedOpt = document.getElementById("trainSpeedOpt");
const accToleranceOpt = document.getElementById("accToleranceOpt");

const powerSensorPrep = document.getElementById("powerSensorPrep");
const powerImagePrep = document.getElementById("powerImagePrep");
const powerAudioPrep = document.getElementById("powerAudioPrep");

const pwrSensorRobust = document.getElementById("pwrSensorRobust");
const pwrSensorOutlier = document.getElementById("pwrSensorOutlier");

const pwrImageAug = document.getElementById("pwrImageAug");
const pwrImageFeat = document.getElementById("pwrImageFeat");
const pwrImageClean = document.getElementById("pwrImageClean");
const pwrImageNoise = document.getElementById("pwrImageNoise");

const pwrAudioNoise = document.getElementById("pwrAudioNoise");
const pwrAudioClean = document.getElementById("pwrAudioClean");

const ramRange = document.getElementById("ramRange");
const flashRange = document.getElementById("flashRange");
const cpuRange = document.getElementById("cpuRange");
const ramVal = document.getElementById("ramInput");
const flashVal = document.getElementById("flashInput");
const flashKb = document.getElementById("flashKb");
const ramMb = document.getElementById("ramMb");
const cpuVal = document.getElementById("cpuInput");
const ramUnit = document.getElementById("ramUnit");
const flashUnit = document.getElementById("flashUnit");
const cpuUnit = document.getElementById("cpuUnit");
const ramField = document.getElementById("ramField");
const flashField = document.getElementById("flashField");
const cpuField = document.getElementById("cpuField");

const quantInput = document.getElementById("quantInput");
const strategyInput = document.getElementById("strategyInput");
const epochsInput = document.getElementById("epochsInput");

const helpModal = document.getElementById("helpModal");
const helpContent = document.getElementById("helpContent");
const closeHelpBtn = document.getElementById("closeHelp");
const helpBtns = document.querySelectorAll(".help-btn");

const state = {
  step: 1,
  title: "",
  description: "",
  visibility: "private",
  taskType: null,
  isDeveloperMode: true,
  dataset: { name: null, size: 0, type: null, file: null },
  device: {
    familyId: null,
    ramKB: null,
    flashMB: null,
    cpuMHz: null,
  },
};

function updateRoleUI() {
  if (state.isDeveloperMode) {
    roleLabel.textContent = "Developer";
    roleToggle.classList.remove("new-task-btn-solid");
    roleToggle.classList.add("new-task-btn-ghost");

    powerUserControls.hidden = true;
  } else {
    roleLabel.textContent = "Power User";
    roleToggle.classList.remove("new-task-btn-ghost");
    roleToggle.classList.add("new-task-btn-solid");

    powerUserControls.hidden = false;
  }
  updateConfigUI();
}

function updateConfigUI() {
  generalPowerControls.hidden = true;
  dataStrategyControls.hidden = true; 

  powerSensorPrep.hidden = true;
  powerSensorPrep.style.display = "none";
  powerImagePrep.hidden = true;
  powerImagePrep.style.display = "none";
  powerAudioPrep.hidden = true;
  powerAudioPrep.style.display = "none";

  const type = state.taskType;
  if (!type) return; 

  if (state.isDeveloperMode) {

  } else {
    generalPowerControls.hidden = false;
    dataStrategyControls.hidden = false; 

    if (type === "sensor") {
      powerSensorPrep.hidden = false;
      powerSensorPrep.style.display = "grid";
    } else if (type === "image") {
      powerImagePrep.hidden = false;
      powerImagePrep.style.display = "grid";
    } else if (type === "audio") {
      powerAudioPrep.hidden = false;
      powerAudioPrep.style.display = "grid";
    }
  }
}

roleToggle.addEventListener("click", () => {
  state.isDeveloperMode = !state.isDeveloperMode;
  updateRoleUI();
});
updateRoleUI();

function setStep(n) {
  state.step = n;
  step1.hidden = n !== 1;
  step2.hidden = n !== 2;
  step3.hidden = n !== 3;
  progressEl.style.width = n === 1 ? "33%" : n === 2 ? "66%" : "100%";

  err1.textContent = "";
  err2.textContent = "";
  errFamily.textContent = "";
}
taskTitleInput.addEventListener("input", () => {
  err1.textContent = "";
});
targetColumnInput.addEventListener("input", () => {
  err2.textContent = "";
});
numClassesInput.addEventListener("input", () => {
  err2.textContent = "";
});

taskCards.forEach((card) => {
  card.addEventListener("click", () => {
    taskCards.forEach((c) => c.setAttribute("aria-checked", "false"));
    card.setAttribute("aria-checked", "true");
    state.taskType = card.dataset.type;
    updateConfigUI();
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
  state.title = taskTitleInput.value.trim();
  state.description = taskDescInput.value.trim();
  state.visibility =
    Array.from(visRadios).find((r) => r.checked)?.value || "private";

  if (!state.title) {
    err1.textContent = "Please enter a task title.";
    return;
  }
  if (!state.taskType) {
    err1.textContent = "Please choose a task type.";
    return;
  }

  if (state.taskType === "sensor") {
    fileInput.accept = ".csv,.tsv,.xlsx,.xls,.parquet,.json";
    uploadHint.textContent = "Allowed: CSV/TSV/XLSX/XLS/Parquet/JSON (Max 500MB)";
  } else {
    fileInput.accept = ".zip,.tar.gz,.tgz";
    uploadHint.textContent =
      "Allowed: ZIP / TAR.GZ containing folders per class";
  }

  if (state.dataset.name) {
    const ext = getExt(state.dataset.name);
    const isTabular = [".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"].includes(ext);
    const isZip = [".zip", ".tar", ".gz", ".tgz"].some((x) => ext.endsWith(x));

    if (
      (state.taskType === "sensor" && !isTabular) ||
      (state.taskType !== "sensor" && !isZip)
    ) {
      clearUploadFile();
    } else {
      updateUploadUI(true);
    }
  }

  setStep(2);
});

const MAX_SIZE = 500 * 1024 * 1024;

function getExt(name = "") {
  const lower = name.toLowerCase().trim();
  if (lower.endsWith(".tar.gz")) return ".tar.gz";
  const dot = lower.lastIndexOf(".");
  return dot >= 0 ? lower.slice(dot) : "";
}

function isAllowed(file) {
  const ext = getExt(file?.name || "");
  if (state.taskType === "sensor") {
    return [".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"].includes(ext);
  } else {
    return [".zip", ".tar", ".gz", ".tgz", ".tar.gz"].includes(ext);
  }
}

function updateUploadUI(hasFile) {
  if (!hasFile) {
    fileName.textContent = "";
    fileSize.textContent = "";
    fileInfo.hidden = true;
    fileInfo.style.display = "none";
    drop.hidden = false;
    drop.style.display = "";
    targetColGroup.hidden = true;
    numClassesGroup.hidden = true;
  } else {
    fileInfo.hidden = false;
    fileInfo.style.display = "";
    drop.hidden = true;
    drop.style.display = "none";

    if (state.taskType === "sensor") {
      targetColGroup.hidden = false;
      numClassesGroup.hidden = true;
    } else {
      targetColGroup.hidden = true;
      numClassesGroup.hidden = false;
    }
  }
}

function clearUploadFile() {
  state.dataset = { name: null, size: 0, type: null, file: null };
  fileInput.value = "";
  try {
    const dt = new DataTransfer();
    fileInput.files = dt.files;
  } catch { }
  updateUploadUI(false);
}

function setUploadFile(file) {
  if (!file) return;
  if (!isAllowed(file)) {
    err2.textContent =
      state.taskType === "sensor"
        ? "Allowed: CSV/TSV/XLSX/XLS/Parquet/JSON for Sensor tasks."
        : "Only ZIP/Archive files are allowed for Image/Audio tasks.";
    clearUploadFile();
    return;
  }
  if (file.size > MAX_SIZE) {
    err2.textContent = "File is too large. Maximum allowed size is 500 MB.";
    clearUploadFile();
    return;
  }

  const [name, extension] = file.name.split(/\.(?=[^\.]+$)/);
  state.dataset = {
    name,
    extension: extension ? `.${extension}` : "",
    size: file.size || 0,
    type: file.type || "application/octet-stream",
    file: file,
  };
  fileName.textContent = file.name;
  fileSize.textContent = `• ${(file.size / 1024 / 1024).toFixed(2)} MB`;
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
  const files = e.dataTransfer?.files;
  if (files?.length) setUploadFile(files[0]);
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

  if (state.taskType === "sensor") {
    if (!targetColumnInput.value.trim()) {
      err2.textContent = "Please enter the name of the target column.";
      return;
    }
  } else {
    if (!numClassesInput.value || numClassesInput.value < 1) {
      err2.textContent = "Please enter the number of classes (categories).";
      return;
    }
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

function enableSliders(enable) {
  ramRange.disabled = !enable;
  flashRange.disabled = !enable;
  cpuRange.disabled = !enable;

  if (enable) {
    ramField.classList.remove("disabled");
    flashField.classList.remove("disabled");
    cpuField.classList.remove("disabled");
    ramVal.disabled = false;
    flashVal.disabled = false;
    cpuVal.disabled = false;
    if (ramUnit) ramUnit.hidden = false;
    if (flashUnit) flashUnit.hidden = false;
    if (cpuUnit) cpuUnit.hidden = false;
  } else {
    ramField.classList.add("disabled");
    flashField.classList.add("disabled");
    cpuField.classList.add("disabled");
    ramVal.disabled = true;
    flashVal.disabled = true;
    cpuVal.disabled = true;
    ramVal.value = "";
    ramMb.textContent = "";
    if (ramUnit) ramUnit.hidden = true;
    flashVal.value = "";
    flashKb.textContent = "";
    if (flashUnit) flashUnit.hidden = true;
    cpuVal.value = "";
    if (cpuUnit) cpuUnit.hidden = true;
  }
}

function applyFamilyDefaults(fam) {
  enableSliders(true);
  if (manualSpecsHint) manualSpecsHint.hidden = true;

  const hints = fam?.specs_hint || {
    ramKB: [32, 256],
    flashMB: [1, 8],
    cpuMHz: [80, 240],
  };

  ramRange.min = hints.ramKB[0];
  ramRange.max = hints.ramKB[1];
  ramVal.min = hints.ramKB[0];
  ramVal.max = hints.ramKB[1];

  flashRange.min = hints.flashMB[0];
  flashRange.max = hints.flashMB[1];
  flashRange.step = 0.001;
  flashVal.min = hints.flashMB[0];
  flashVal.max = hints.flashMB[1];
  flashVal.step = "any";

  cpuRange.min = hints.cpuMHz[0];
  cpuRange.max = hints.cpuMHz[1];
  cpuVal.min = hints.cpuMHz[0];
  cpuVal.max = hints.cpuMHz[1];

  const ramMid = midpoint(hints.ramKB);
  ramRange.value = clamp(
    state.device.ramKB ?? ramMid,
    hints.ramKB[0],
    hints.ramKB[1],
  );
  ramVal.value = ramRange.value;
  ramMb.textContent = `(${formatMB(Number(ramVal.value) / 1024)})`;

  const flashMid = midpoint(hints.flashMB);
  flashRange.value = clamp(
    state.device.flashMB ?? flashMid,
    hints.flashMB[0],
    hints.flashMB[1],
  );
  flashVal.value = Number(flashRange.value);
  flashKb.textContent = `(${Math.round(Number(flashVal.value) * 1024)} KB)`;

  const cpuMid = midpoint(hints.cpuMHz);
  cpuRange.value = clamp(
    state.device.cpuMHz ?? cpuMid,
    hints.cpuMHz[0],
    hints.cpuMHz[1],
  );
  cpuVal.value = Math.round(Number(cpuRange.value));
  familyNote.textContent = fam?.note || "";

  modelExtSelect.innerHTML = "";
  modelExtSelect.removeAttribute("disabled");
  const exts = fam?.model_exts || [".tflite", ".bin"];
  const allOpt = document.createElement("option");
  allOpt.value = "all";
  allOpt.textContent = "All";
  modelExtSelect.appendChild(allOpt);

  exts.forEach((ext) => {
    const opt = document.createElement("option");
    opt.value = ext;
    opt.textContent = ext;
    modelExtSelect.appendChild(opt);
  });

  if (exts.includes(".h")) {
    modelExtSelect.value = ".h";
  } else if (exts.length > 0) {
    modelExtSelect.value = exts[0];
  } else {
    modelExtSelect.value = "all";
  }
}

function syncInputs(slider, input, isFloat = false) {
  slider.addEventListener("input", () => {
    input.value = isFloat ? Number(slider.value) : Math.round(slider.value);
  });

  input.addEventListener("change", () => {
    let val = Number(input.value);
    const min = Number(input.min);
    const max = Number(input.max);

    if (val < min) val = min;
    if (val > max) val = max;

    input.value = val;
    slider.value = val;

    if (input === flashVal) {
      flashKb.textContent = val ? `(${Math.round(val * 1024)} KB)` : "";
    } else if (input === ramVal) {
      ramMb.textContent = val ? `(${formatMB(val / 1024)})` : "";
    }
  });

  if (input === flashVal) {
    slider.addEventListener("input", () => {
      const val = Number(input.value);
      flashKb.textContent = `(${Math.round(Number(slider.value) * 1024)} KB)`;
    });
  } else if (input === ramVal) {
    slider.addEventListener("input", () => {
      ramMb.textContent = `(${formatMB(Number(slider.value) / 1024)})`;
    });
  }
}

syncInputs(ramRange, ramVal);
syncInputs(flashRange, flashVal, true);
syncInputs(cpuRange, cpuVal);

back3.addEventListener("click", () => setStep(2));

submitTask.addEventListener("click", async (e) => {
  e.preventDefault();
  if (!state.device.familyId) {
    errFamily.textContent = "Please select a device family.";
    return;
  }
  errFamily.textContent = "";

  const fileToUpload = state.dataset.file;
  const targetColumn = targetColumnInput.value.trim();
  const numClasses = numClassesInput.value.trim();
  if (!fileToUpload) {
    err2.textContent = "Please select a dataset file.";
    return;
  }
  if (state.taskType === "sensor" && !targetColumn) {
    err2.textContent = "Please enter the target column name.";
    return;
  }

  const apiBase = localStorage.getItem("api_base") || "http://127.0.0.1:8000";
  const apiUrl = `${apiBase}/tasks`;

  const formData = new FormData();
  formData.append("file", fileToUpload);
  formData.append("title", state.title);
  formData.append("description", state.description);
  formData.append("visibility", state.visibility);
  formData.append("task_type", state.taskType);
  formData.append("device_family", state.device.familyId);

  const session = getSession();
  const uid = session?.uid || "demo-user";
  formData.append("user_id", uid);
  formData.append("ram_kb", ramVal.value);
  formData.append("flash_mb", flashVal.value);
  formData.append("cpu_mhz", cpuVal.value);
  formData.append("model_extension", modelExtSelect.value);

  if (!state.isDeveloperMode) {
    formData.append("quantization", quantInput.value);
    formData.append("optimization_strategy", strategyInput.value);
    formData.append("epochs", epochsInput.value);
    formData.append("training_speed", trainSpeedOpt.value);
    formData.append("accuracy_tolerance", accToleranceOpt.value);

    if (state.taskType === "sensor") {
      formData.append("robustness", pwrSensorRobust.value);
      formData.append("outlier_removal", pwrSensorOutlier.value);
    } else if (state.taskType === "image") {
      formData.append("augmentation", pwrImageAug.value);
      formData.append("feature_handling", pwrImageFeat.value);
      formData.append("cleaning", pwrImageClean.value);
      formData.append("noise_handling", pwrImageNoise.value);
    } else if (state.taskType === "audio") {
      formData.append("noise_handling", pwrAudioNoise.value);
      formData.append("cleaning", pwrAudioClean.value);
    }
  }

  if (state.taskType === "sensor") {
    formData.append("target_column", targetColumn);
  } else {
    formData.append("num_classes", numClasses);
  }

  submitTask.disabled = true;
  submitTask.textContent = "Submitting Task...";

  try {
    console.log("=== SUBMITTING TASK ===");
    console.log("Task Type:", state.taskType);
    console.log("Submitting to:", apiUrl);

    try {
      const healthUrl = `${apiBase}/health`;
      console.log("Checking connectivity to:", healthUrl);
      const health = await fetch(healthUrl);
      if (health.ok) {
        console.log("Backend is reachable!");
      } else {
        console.warn("Backend reachable but returned error:", health.status);
      }
    } catch (e) {
      console.error("Connectivity check failed:", e);
      throw new Error(`Cannot connect to backend at ${apiBase}. Is it running?`);
    }

    console.log("FormData entries:", Array.from(formData.entries()));

    const response = await fetch(apiUrl, {
      method: "POST",
      body: formData,
    });

    console.log("Response status:", response.status);

    if (!response.ok) {
      let msg = "Failed to submit task.";
      try {
        const err = await response.json();
        console.error("Server error:", err);
        if (err && err.detail) {
          msg = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
        }
      } catch { }
      throw new Error(msg);
    }

    const result = await response.json();
    const taskId = result.task_id || result.model_id || crypto.randomUUID();
    const fam = DEVICE_FAMILIES.find((f) => f.id === state.device.familyId);

    const task = {
      id: taskId,
      title: state.title,
      datasetName: state.dataset.name || "Untitled Dataset",
      taskType: state.taskType || "generic",
      device_family: state.device.familyId,
      device: fam?.name || "Unknown Family",
      status: "queued",
      progress: 0,
      createdAt: Date.now(),
      mock: false,
    };

    const raw = localStorage.getItem("tasks");
    const list = raw ? JSON.parse(raw) : [];
    list.push(task);
    localStorage.setItem("tasks", JSON.stringify(list));
    localStorage.setItem("tasks_last_update", String(Date.now()));

    location.href = `./status.html?id=${encodeURIComponent(taskId)}`;
  } catch (err) {
    console.error("Submit failed:", err);

    let errorMsg = err.message;
    if (err.message === "Failed to fetch") {
      errorMsg = "Cannot connect to backend server. Please ensure the server is running at " + apiUrl;
    }

    alert(`Error: ${errorMsg}`);
    submitTask.disabled = false;
    submitTask.textContent = "Submit Task";
  }
});

const HELP_TEXTS = {
  step1: `
    <h3>Step 1: Task Details & Selection</h3>
    <p>Start by defining the core identity of your task.</p>
    <ul>
      <li><b>Title & Description:</b> Give your task a unique name. The description is optional but helpful for team collaboration.</li>
      <li><b>Visibility:</b>
        <ul>
          <li><b>Private:</b> Visible only to you. Best for experimentation.</li>
          <li><b>Public:</b> Visible to everyone. Use this for shared or production-ready models.</li>
        </ul>
      </li>
    </ul>
    <p><b>Select a Task Type:</b></p>
    <ul>
      <li><b>Image:</b> Classification models for visual data (e.g., identifying defects, sorting objects). Expects ZIP files with image folders.</li>
      <li><b>Audio:</b> Sound classification (e.g., glass break detection, keyword spotting). Expects ZIP files with audio folders.</li>
      <li><b>Sensor:</b> Time-series analysis (e.g., gesture recognition from IMU data). Expects a CSV file.</li>
    </ul>
  `,
  step2_sensor: `
    <h3>Uploading Sensor Data (CSV)</h3>
    <p>For sensor tasks, upload a single <b>CSV file</b>. The file must include a header row.</p>
    <p><b>Requirements:</b></p>
    <ul>
      <li><b>Target Column:</b> You must specify the exact name of the column that contains the labels (the "answer" you want to predict).</li>
      <li><b>Structure:</b> Each row represents a time step. The system will automatically detect features based on other columns.</li>
    </ul>
    <pre>timestamp, accel_x, accel_y, accel_z, label
1001,      0.12,    0.98,    0.05,    "idle"
1002,      0.11,    0.99,    0.04,    "idle"
...</pre>
  `,
  step2_image: `
    <h3>Uploading Image Data (ZIP)</h3>
    <p>Upload a standard <b>ZIP archive</b>. The internal folder structure is critical as it defines your class labels.</p>
    <p><b>Structure Requirement:</b></p>
    <pre>dataset.zip
├── class_dog/      <-- Label: "class_dog"
│   ├── image01.jpg
│   └── image02.png
└── class_cat/      <-- Label: "class_cat"
    ├── image01.jpg
    └── ...</pre>
    <ul>
      <li>Create one folder for each category (class) you want to recognize.</li>
      <li>Place all relevant images inside their respective folders.</li>
      <li>Supported formats: <b>JPG, PNG, BMP</b>.</li>
    </ul>
  `,
  step2_audio: `
    <h3>Uploading Audio Data (ZIP)</h3>
    <p>Upload a <b>ZIP archive</b> containing your audio samples. Like images, the folder names determine the class labels.</p>
    <p><b>Structure Requirement:</b></p>
    <pre>dataset.zip
├── siren_alert/    <-- Label: "siren_alert"
│   ├── sample1.wav
│   └── sample2.mp3
└── glass_break/    <-- Label: "glass_break"
    ├── sample1.wav
    └── ...</pre>
    <ul>
      <li><b>Short Clips:</b> Ideally, audio clips should be of similar length (e.g., 1-second samples for keywords).</li>

      <li>Supported formats: <b>WAV, MP3</b> (WAV recommended for quality).</li>
    </ul>
  `,
};

const EXT_DESCRIPTIONS = {
  ".tflite": "TensorFlow Lite FlatBuffer. Standard for most edge devices.",
  ".bin": "Raw binary weight file. Requires custom loader layout.",
  ".h": "C Header file. Compiles weight arrays directly into your firmware (No filesystem needed).",
  ".kmodel": "Kendryte K210 proprietary format.",
  ".onnx": "Open Neural Network Exchange. Widely supported interchange format.",
  ".engine": "NVIDIA TensorRT Engine. Optimized for Jetson GPUs.",
};

function getStep3Help(isDev) {
  const famId = state.device.familyId;
  const fam = DEVICE_FAMILIES.find((f) => f.id === famId);

  // 1. Model Format Help
  let modelFormatHelp = "";
  if (fam) {
    modelFormatHelp = `<p><b>Available Output Formats based on Device Family:</b></p><ul>`;
    modelFormatHelp += `<li><b>All:</b> Generates all compatible formats. (Default)</li>`;
    const exts = fam.model_exts || [];
    exts.forEach((ext) => {
      const desc = EXT_DESCRIPTIONS[ext] || "Standard model file.";
      modelFormatHelp += `<li><b>${ext}:</b> ${desc}</li>`;
    });
    modelFormatHelp += "</ul>";
  } else {
    modelFormatHelp = `<p><b>Model Output Format:</b> Select a Device Family first to see available specific formats.</p>`;
  }

  let prepHelp = "";
  if (state.taskType === "sensor") {
    if (!isDev) {
      prepHelp = `
                <p><b>Sensor Data Strategy (Power User):</b></p>
                <p class="muted"><i>Applying to Sensor tasks only</i></p>
                <ul>
                    <li><b>Robustness:</b> <i>Automatic</i> adapts to signal noise. <i>High</i> injection improves real-world performance.</li>
                    <li><b>Outlier Removal:</b> <i>Automatic</i> uses tailored logic. <i>Standard</i> (Z-score), <i>Robust</i> (IQR).</li>
                </ul>
             `;
    }
  } else if (state.taskType === "image") {
    if (!isDev) {
      prepHelp = `
                <p><b>Image Data Strategy (Power User):</b></p>
                <p class="muted"><i>Applying to Image tasks only</i></p>
                <ul>
                    <li><b>Augmentation:</b> <i>Automatic</i> applies standard transforms. <i>Strong</i> helps small datasets.</li>
                    <li><b>Feature Handling:</b> <i>Automatic</i> selects based on model depth. <i>Rich</i> extracts complex features.</li>
                    <li><b>Cleaning:</b> <i>Automatic</i> drops bad samples. <i>Aggressive</i> removes blurry images.</li>
                    <li><b>Noise Handling:</b> <i>Automatic</i> detects corruptions. <i>Suppress</i> removes grain.</li>
                </ul>
             `;
    }
  } else if (state.taskType === "audio") {
    if (!isDev) {
      prepHelp = `
                <p><b>Audio Data Strategy (Power User):</b></p>
                <p class="muted"><i>Applying to Audio tasks only</i></p>
                <ul>
                    <li><b>Noise Handling:</b> <i>Automatic</i> estimates background noise level. <i>Reduce</i> applies spectral subtraction.</li>
                    <li><b>Cleaning:</b> <i>Automatic</i> drops silent clips. <i>Aggressive</i> ensures quality signal.</li>
                </ul>
             `;
    }
  }

  if (isDev) {
    return `
            <h3>Target Device & Prep (Developer Mode)</h3>
            <p><b>Design Intent:</b> "Prepare my data correctly without me worrying about how."</p>
            <p><b>Manual Hardware Constraints:</b></p>
            <ul>
                <li>Set strict RAM/Flash limits to ensure the model fits your specific MCU.</li>
            </ul>
            ${modelFormatHelp}
        `;
  } else {
    return `
            <h3>Target Device & Optimization (Power User Mode)</h3>
            <p><b>Design Intent:</b> "Let me guide how the data is prepared, without micromanaging."</p>
            
            <p><b>General Strategy:</b></p>
            <ul>
                <li><b>Training Speed:</b> <i>Fast</i> uses fewer iterations; <i>Slow</i> does thorough search.</li>
                <li><b>Accuracy Tolerance:</b> <i>Efficiency Prioritized</i> accepts slight accuracy drop for major speed/size gains.</li>
            </ul>

            <hr style="border: 0; border-top: 1px solid #ddd; margin: 10px 0;">
            ${modelFormatHelp}
            
            <hr style="border: 0; border-top: 1px solid #ddd; margin: 10px 0;">
            <p><b>Advanced Training Options:</b></p>
            <ul>
                <li><b>Quantization:</b> Int8 (Standard), Float16, Float32.</li>
                <li><b>Epochs:</b> (5-40) Training cycles.</li>
            </ul>

            <hr style="border: 0; border-top: 1px solid #ddd; margin: 10px 0;">
            ${prepHelp}
        `;
  }
}

function openHelp(stepId) {
  let content = "";
  if (stepId == 1) {
    content = HELP_TEXTS.step1;
  } else if (stepId == 2) {
    if (state.taskType === "sensor") content = HELP_TEXTS.step2_sensor;
    else if (state.taskType === "audio") content = HELP_TEXTS.step2_audio;
    else content = HELP_TEXTS.step2_image;
  } else if (stepId == 3) {
    content = getStep3Help(state.isDeveloperMode);
  }

  helpContent.innerHTML = content;
  helpModal.classList.add("open");
  helpModal.setAttribute("aria-hidden", "false");
}

function closeHelp() {
  helpModal.classList.remove("open");
  helpModal.setAttribute("aria-hidden", "true");
}

helpBtns.forEach((btn) => {
  btn.addEventListener("click", (e) => {
    const step = e.currentTarget.dataset.step;
    openHelp(step);
  });
});

closeHelpBtn.addEventListener("click", closeHelp);
helpModal.addEventListener("click", (e) => {
  if (e.target === helpModal) closeHelp();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && helpModal.classList.contains("open")) closeHelp();
});

setStep(1);

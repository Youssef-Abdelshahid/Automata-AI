/* =====================
   DOM refs
===================== */

const overlay = document.getElementById("editTaskOverlay");
const closeBtn = document.getElementById("closeEditTask");
const cancelBtn = document.getElementById("cancelEdit");
const saveBtn = document.getElementById("saveTask");
const deleteTaskBtn = document.getElementById("deleteTaskBtn");

const editProfileBtn = document.getElementById("editProfileBtn");
const editProfileOverlay = document.getElementById("editProfileOverlay");
const closeEditProfile = document.getElementById("closeEditProfile");
const cancelEditProfile = document.getElementById("cancelEditProfile");
const saveProfileBtn = document.getElementById("saveProfile");

const deleteConfirmOverlay = document.getElementById("deleteConfirmOverlay");
const closeDeleteConfirm = document.getElementById("closeDeleteConfirm");
const cancelDelete = document.getElementById("cancelDelete");
const confirmDelete = document.getElementById("confirmDelete");

const userListOverlay = document.getElementById("userListOverlay");
const closeUserList = document.getElementById("closeUserList");
const userListContainer = document.getElementById("userListContainer");
const userListTitle = document.getElementById("userListTitle");

const followersBtn = document.getElementById("followersBtn");
const followingBtn = document.getElementById("followingBtn");

const tasksList = document.getElementById("tasksList");

/* =====================
   Constants
===================== */

const ALLOWED_TYPES = new Set(["image", "video", "audio"]);
const ALLOWED_STATUS = new Set([
  "queued",
  "training",
  "completed",
  "failed",
]);

/* =====================
   Storage helpers
===================== */

function normalizeTask(t, idx = 0) {
  const id = t?.id || `task_${Date.now().toString(36)}_${idx}`;
  const title = t?.title || "Untitled Task";
  const datasetName = String(t?.datasetName ?? "Untitled Dataset");
  const taskType = ALLOWED_TYPES.has(t?.taskType) ? t.taskType : "image";
  const device = t?.device || "Generic Device";
  const status = ALLOWED_STATUS.has(t?.status) ? t.status : "queued";
  const description = t?.description || "";
  const visibility = t?.visibility === "private" ? "private" : "public";
  const createdAt =
    typeof t?.createdAt === "number" && Number.isFinite(t.createdAt)
      ? t.createdAt
      : Date.now();
  const collaborators = Array.isArray(t?.collaborators) ? t.collaborators : [];

  return { ...t, id, title, datasetName, taskType, device, status, createdAt, description, visibility, collaborators };
}

function readTasksRaw() {
  try {
    const raw = localStorage.getItem("tasks");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveTasks(list) {
  const normalized = list.map(normalizeTask);
  localStorage.setItem("tasks", JSON.stringify(normalized));
  localStorage.setItem("tasks_last_update", String(Date.now()));
  return normalized;
}

/* =====================
   TASKS
===================== */

const TASKS = saveTasks(readTasksRaw());

/* =====================
   Lookup
===================== */

function getTaskById(taskId) {
  return TASKS.find((t) => t.id === taskId) || null;
}

/* =====================
   Render
===================== */

function renderTasks() {
  if (!tasksList) return;

  tasksList.innerHTML = "";

  if (!TASKS.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "You have no tasks";
    tasksList.appendChild(empty);
    return;
  }

  TASKS.forEach((task) => {
    const card = document.createElement("div");
    card.className = "task-card";

    let shareBtnHtml = "";
    if (task.visibility !== "public") {
      shareBtnHtml = `
            <button class="share-btn" data-share-id="${task.id}" data-tooltip="view link" aria-label="Copy task link">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="18" cy="5" r="3"></circle>
                    <circle cx="6" cy="12" r="3"></circle>
                    <circle cx="18" cy="19" r="3"></circle>
                    <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                    <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                </svg>
            </button>
        `;
    }

    card.innerHTML = `
      <div class="task-info">
        <h3 class="task-title">${task.title}</h3>
        <div class="task-meta">
          ${task.visibility} • ${task.description ? task.description.substring(0, 30) + (task.description.length > 30 ? "..." : "") : "No description"}
        </div>
      </div>
      <div style="display: flex; align-items: center;">
        ${shareBtnHtml}
        <button class="btn secondary edit-profile-btn" data-task-id="${task.id}" style="margin: 0;">
            Edit
        </button>
      </div>
    `;

    tasksList.appendChild(card);
  });
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast-notification";
  toast.textContent = message;

  document.body.appendChild(toast);

  // Auto cleanup matches animation duration (3s)
  setTimeout(() => {
    toast.remove();
  }, 3000);
}

// Simplified Share Logic (Direct Copy)
document.addEventListener("click", async (e) => {
  const shareBtn = e.target.closest(".share-btn");
  if (!shareBtn) return;

  const taskId = shareBtn.dataset.shareId;
  if (!taskId) return;

  // Construct link
  const url = new URL("tasks.html", location.href);
  url.searchParams.set("id", taskId);
  url.searchParams.set("access", "view"); // Default access

  try {
    await navigator.clipboard.writeText(url.toString());
    showToast("Link copied to clipboard");
  } catch (err) {
    console.error("Failed to copy", err);
    alert("Link: " + url.toString());
  }
});

/* =====================
   Overlay logic
===================== */

let currentEditingTaskId = null;

/* =====================
   Helpers for modal state
===================== */

function applyModalStyles() {
  document.documentElement.classList.add("modal-open"); // <html>
  document.body.classList.add("modal-open");            // <body>
}

function removeModalStyles() {
  document.documentElement.classList.remove("modal-open");
  document.body.classList.remove("modal-open");
}

/* =====================
   Open edit
===================== */

document.addEventListener("click", (e) => {
  // Check for the edit button using the new class
  const btn = e.target.closest(".edit-profile-btn");
  if (!btn) return;

  // Distinguish between task edit and profile edit by checking for data-task-id
  if (btn.dataset.taskId) {
    openEditTask(btn.dataset.taskId);
  } else if (btn.id === "editProfileBtn") {
    // This is handled by specific id listener below, but safe to ignore here or handle if we unified logic.
    // The specific listener for editProfileBtn will handle profile edit.
    // So we only care if it HAS a task id.
  }
});

// addTaskBtn is now a link, no JS needed
// const addTaskBtn = document.getElementById("addTaskBtn");

function openEditTask(taskId) {
  const task = getTaskById(taskId);
  if (!task) return;

  currentEditingTaskId = taskId;

  document.querySelector("#editTaskOverlay h2").textContent = "Edit Task";
  document.getElementById("taskTitle").value = task.title;
  document.getElementById("taskDescription").value = task.description;

  if (task.visibility === "private") {
    document.getElementById("visPrivate").checked = true;
  } else {
    document.getElementById("visPublic").checked = true;
  }

  // --- Collaborator Logic ---
  const collabSection = document.getElementById("collaboratorSection");
  if (task.visibility === "public") {
    collabSection.classList.add("hidden");
  } else {
    collabSection.classList.remove("hidden");
  }

  renderCollaborators(task.collaborators || []);

  overlay.classList.remove("hidden");
  applyModalStyles();
}

// Handle Visibility Toggle in Edit Overlay
document.querySelectorAll('input[name="visibility"]').forEach(radio => {
  radio.addEventListener("change", (e) => {
    const collabSection = document.getElementById("collaboratorSection");
    if (e.target.value === "public") {
      collabSection.classList.add("hidden");
    } else {
      collabSection.classList.remove("hidden");
    }
  });
});

// Collaborator State for current edit session
let currentCollaborators = [];

const collaboratorListEl = document.getElementById("collaboratorList");
const newCollaboratorInput = document.getElementById("newCollaboratorInput");
const addCollaboratorBtn = document.getElementById("addCollaboratorBtn");

function renderCollaborators(list) {
  currentCollaborators = [...list]; // Sync state
  collaboratorListEl.innerHTML = "";

  if (!list.length) {
    collaboratorListEl.innerHTML = `<div class="muted" style="font-size: 0.85rem; padding: 4px;"></div>`;
    return;
  }

  list.forEach((collab, index) => {
    const item = document.createElement("div");
    item.className = "collaborator-item";

    // Generate initials
    const initials = (collab.name || collab.email || "??").substring(0, 2).toUpperCase();

    item.innerHTML = `
      <div class="collaborator-info">
        <div class="collaborator-avatar">${initials}</div>
        <span>${collab.email || collab.handle}</span>
      </div>
      <button class="remove-collab-btn" aria-label="Remove" data-index="${index}">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
      </button>
    `;
    collaboratorListEl.appendChild(item);
  });

  // Attach remove listeners
  document.querySelectorAll(".remove-collab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.index);
      currentCollaborators.splice(idx, 1);
      renderCollaborators(currentCollaborators);
    });
  });
}

addCollaboratorBtn.addEventListener("click", () => {
  const val = newCollaboratorInput.value.trim();
  if (!val) return;

  // Simple duplicacy check
  if (currentCollaborators.some(c => c.email === val || c.handle === val)) {
    alert("User already added.");
    return;
  }

  // Mock user object
  const newUser = {
    email: val.includes("@") ? val : null,
    handle: val.startsWith("@") ? val : (val.includes("@") ? null : "@" + val),
    name: val.split("@")[0]
  };

  currentCollaborators.push(newUser);
  renderCollaborators(currentCollaborators);
  newCollaboratorInput.value = "";
});

const mentionSuggestions = document.getElementById("mentionSuggestions");

function renderMentions(list) {
  mentionSuggestions.innerHTML = "";
  if (!list.length) {
    mentionSuggestions.classList.add("hidden");
    return;
  }

  list.forEach(user => {
    const div = document.createElement("div");
    div.className = "mention-item";
    div.innerHTML = `
      <div class="mention-avatar">${user.initials}</div>
      <div class="mention-info">
        <div class="mention-name">${user.name}</div>
        <div class="mention-handle">${user.handle}</div>
      </div>
    `;
    div.addEventListener("click", () => {
      newCollaboratorInput.value = user.handle; // Auto-fill handle
      mentionSuggestions.classList.add("hidden");
      newCollaboratorInput.focus();
    });
    mentionSuggestions.appendChild(div);
  });

  mentionSuggestions.classList.remove("hidden");
}

newCollaboratorInput.addEventListener("input", (e) => {
  const val = e.target.value;
  // Check if user is typing a mention (starts with @)
  if (val.startsWith("@")) {
    const query = val.slice(1).toLowerCase(); // remove @
    // Filter followers
    const matches = MOCK_FOLLOWERS.filter(u =>
      u.handle.toLowerCase().includes(query) ||
      u.name.toLowerCase().includes(query)
    );
    renderMentions(matches);
  } else {
    mentionSuggestions.classList.add("hidden");
  }
});

// Hide mentions on outside click
document.addEventListener("click", (e) => {
  if (!e.target.closest(".collaborator-input-group")) {
    mentionSuggestions.classList.add("hidden");
  }
});

// openCreateTask removed - handled by new_task.html

/* =====================
   Close overlay
===================== */

function closeOverlay() {
  overlay.classList.add("hidden");
  currentEditingTaskId = null;
  removeModalStyles(); // ✅ REMOVE styles
}

closeBtn.onclick = closeOverlay;
cancelBtn.onclick = closeOverlay;

/* =====================
   Save handler
===================== */

saveBtn.addEventListener("click", () => {

  const title = document.getElementById("taskTitle").value.trim();
  const description = document.getElementById("taskDescription").value.trim();
  const visibility = document.querySelector('input[name="visibility"]:checked').value;

  if (!title) {
    alert("Please enter a title");
    return;
  }

  // We only handle EDIT here now
  if (currentEditingTaskId) {
    const idx = TASKS.findIndex((t) => t.id === currentEditingTaskId);
    if (idx === -1) return;

    TASKS[idx] = normalizeTask(
      {
        ...TASKS[idx],
        title,
        description,
        visibility,
        collaborators: currentCollaborators, // Save the list
      },
      idx,
    );

    saveTasks(TASKS);
    closeOverlay();
    renderTasks();
  }
});

/* =====================
   Delete Logic
===================== */

deleteTaskBtn.addEventListener("click", () => {
  // If creating new, delete button should be hidden, but safety check
  if (!currentEditingTaskId) return;
  deleteConfirmOverlay.classList.remove("hidden");
});

function closeDeleteModal() {
  deleteConfirmOverlay.classList.add("hidden");
}

closeDeleteConfirm.onclick = closeDeleteModal;
cancelDelete.onclick = closeDeleteModal;

confirmDelete.addEventListener("click", () => {
  if (!currentEditingTaskId) return;

  const idx = TASKS.findIndex((t) => t.id === currentEditingTaskId);
  if (idx > -1) {
    TASKS.splice(idx, 1);
    saveTasks(TASKS);
    renderTasks();
  }

  closeDeleteModal();
  closeOverlay();
});

/* =====================
   UX extras
===================== */

// ESC closes
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !overlay.classList.contains("hidden")) {
    closeOverlay();
  }
});

// Click outside card closes
overlay.addEventListener("click", (e) => {
  if (e.target === overlay) closeOverlay();
});

/* =====================
   Profile Edit Logic
===================== */

const forgotPasswordBtn = document.getElementById("forgotPasswordBtn");

// Profile Photo DOM
const profileAvatarMain = document.getElementById("profileAvatarMain");
const profileAvatarPreview = document.getElementById("profileAvatarPreview");
const profileAvatarInput = document.getElementById("profileAvatarInput");
const triggerAvatarUpload = document.getElementById("triggerAvatarUpload");
const avatarUploadContainer = document.querySelector(".avatar-upload-container");

// Load initial profile data from localStorage if available
function loadProfile() {
  const profile = JSON.parse(localStorage.getItem("user_profile") || "null");
  if (profile) {
    document.getElementById("displayName").textContent = profile.displayName || "Display Name";
    document.getElementById("username").textContent = profile.username || "@username";
    document.getElementById("bio").textContent = profile.bio || "No bio set.";

    // Load avatar
    if (profile.avatar) {
      profileAvatarMain.src = profile.avatar;
    }
  }
}

function openProfileEdit() {
  const profile = JSON.parse(localStorage.getItem("user_profile") || "null");
  const displayName = document.getElementById("displayName").textContent;
  const username = document.getElementById("username").textContent;
  const bio = document.getElementById("bio").textContent.trim();

  // Default email if not set
  const email = profile?.email || "user@example.com";
  // Default avatar
  const avatar = profile?.avatar || "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2NiZDVlMSI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgM2MxLjY2IDAgMyAxLjM0IDMgM3MtMS4zNCAzLTMgMy0zLTEuMzQtMy0zIDEuMzQtMyAzLTN6bTAgMTQuMmMtMi41IDAtNC43MS0xLjI4LTYtMy4yMi4wMy0xLjk5IDQtMy4wOCA2LTMuMDggMS45OSAwIDUuOTcgMS4wOSA2IDMuMDgtMS4yOSAxLjk0LTMuNSAzLjIyLTYgMy4yMnoiLz48L3N2Zz4=";

  document.getElementById("profileDisplayName").value = displayName;
  document.getElementById("profileUsername").value = username;
  document.getElementById("profileEmail").value = email;
  document.getElementById("profileBio").value = bio;
  profileAvatarPreview.src = avatar;

  // Reset password fields
  document.getElementById("profileOldPassword").value = "";
  document.getElementById("profileNewPassword").value = "";

  editProfileOverlay.classList.remove("hidden");
  applyModalStyles();
}

function closeProfileEdit() {
  editProfileOverlay.classList.add("hidden");
  removeModalStyles();
}

function saveProfile() {
  const displayName = document.getElementById("profileDisplayName").value.trim();
  const username = document.getElementById("profileUsername").value.trim();
  const email = document.getElementById("profileEmail").value.trim();
  const bio = document.getElementById("profileBio").value.trim();
  const avatar = profileAvatarPreview.src; // Get the current src (updated via upload or original)

  const oldPassword = document.getElementById("profileOldPassword").value;
  const newPassword = document.getElementById("profileNewPassword").value;

  // Password validation logic
  if (newPassword) {
    if (!oldPassword) {
      alert("You must enter your old password to set a new one.");
      return;
    }
    alert("Password updated successfully!");
  }

  // Update DOM
  document.getElementById("displayName").textContent = displayName;
  document.getElementById("username").textContent = username;
  document.getElementById("bio").textContent = bio;
  profileAvatarMain.src = avatar;

  // Persist
  localStorage.setItem("user_profile", JSON.stringify({ displayName, username, bio, email, avatar }));

  closeProfileEdit();
}

// Avatar Upload Logic
avatarUploadContainer.addEventListener("click", () => {
  profileAvatarInput.click();
});

profileAvatarInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      profileAvatarPreview.src = e.target.result;
    }
    reader.readAsDataURL(file);
  }
});

// Forgot Password Handler
forgotPasswordBtn.addEventListener("click", (e) => {
  e.preventDefault(); // Prevent accidental form submission if inside form
  alert("Forgot Password clicked! (UI Only)");
});

editProfileBtn.addEventListener("click", openProfileEdit);
closeEditProfile.addEventListener("click", closeProfileEdit);
cancelEditProfile.addEventListener("click", closeProfileEdit);
saveProfileBtn.addEventListener("click", saveProfile);

editProfileOverlay.addEventListener("click", (e) => {
  if (e.target === editProfileOverlay) closeProfileEdit();
});

// Load on init
loadProfile();


/* =====================
   User Lists (Mock Data)
===================== */

const MOCK_FOLLOWERS = [
  { name: "Sarah Connor", handle: "@sarah_c", initials: "SC" },
  { name: "John Doe", handle: "@jdoe_ai", initials: "JD" },
  { name: "Alice Smith", handle: "@asmith", initials: "AS" },
  { name: "Bob Wilson", handle: "@bwilson_dev", initials: "BW" },
  { name: "Eva Green", handle: "@eva_g", initials: "EG" }
];

const MOCK_FOLLOWING = [
  { name: "Tech Weekly", handle: "@tech_weekly", initials: "TW" },
  { name: "AI Research Lab", handle: "@ai_lab_official", initials: "AL" },
  { name: "Open Source Hub", handle: "@opensource", initials: "OS" }
];

// Initialize counts
document.getElementById("followersCount").textContent = MOCK_FOLLOWERS.length;
document.getElementById("followingCount").textContent = MOCK_FOLLOWING.length;


function renderUserList(users) {
  userListContainer.innerHTML = "";

  users.forEach(user => {
    const item = document.createElement("div");
    item.className = "user-list-item";
    item.innerHTML = `
      <div class="user-list-avatar">${user.initials}</div>
      <div class="user-list-info">
        <span class="user-list-name">${user.name}</span>
        <span class="user-list-handle">${user.handle}</span>
      </div>
    `;
    userListContainer.appendChild(item);
  });
}

function openUserList(type) {
  if (type === "followers") {
    userListTitle.textContent = "Followers";
    renderUserList(MOCK_FOLLOWERS);
  } else {
    userListTitle.textContent = "Following";
    renderUserList(MOCK_FOLLOWING);
  }

  userListOverlay.classList.remove("hidden");
  applyModalStyles();
}

function closeUserListModal() {
  userListOverlay.classList.add("hidden");
  removeModalStyles();
}

followersBtn.addEventListener("click", () => openUserList("followers"));
followingBtn.addEventListener("click", () => openUserList("following"));

closeUserList.addEventListener("click", closeUserListModal);
userListOverlay.addEventListener("click", (e) => {
  if (e.target === userListOverlay) closeUserListModal();
});


/* =====================
   Init
===================== */

renderTasks();

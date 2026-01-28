function getSession() {
  try {
    return JSON.parse(localStorage.getItem("session")) || null;
  } catch {
    return null;
  }
}

function saveSession(next) {
  localStorage.setItem("session", JSON.stringify(next));
}

const form = document.getElementById("adminProfileForm");
const statusEl = document.getElementById("profileStatus");
const nameEl = document.getElementById("adminName");
const usernameEl = document.getElementById("adminUsername");
const emailEl = document.getElementById("adminEmail");
const avatarEl = document.getElementById("adminAvatarUrl");
const bioEl = document.getElementById("adminBio");
const avatarImg = document.getElementById("profileAvatar");
const rolePill = document.getElementById("profileRole");

const PROFILE_KEY = "admin_profile";

function loadProfile() {
  try {
    return JSON.parse(localStorage.getItem(PROFILE_KEY)) || null;
  } catch {
    return null;
  }
}

function setStatus(msg) {
  if (!statusEl) return;
  statusEl.textContent = msg || "";
}

function hydrateForm() {
  const session = getSession();
  const stored = loadProfile() || {};
  const name = stored.name || session?.name || "System Admin";
  const username = stored.username || "admin";
  const email = stored.email || session?.email || "admin@automata.ai";
  const avatar = stored.avatar || "styles/logo/Automata_AI_Logo.webp";
  const bio = stored.bio || "";

  nameEl.value = name;
  usernameEl.value = username;
  emailEl.value = email;
  avatarEl.value = avatar;
  bioEl.value = bio;
  avatarImg.src = avatar;
  rolePill.textContent = session?.role || "Admin";
}

avatarEl?.addEventListener("input", () => {
  const next = avatarEl.value.trim() || "styles/logo/Automata_AI_Logo.webp";
  avatarImg.src = next;
});

form?.addEventListener("submit", (e) => {
  e.preventDefault();
  const name = nameEl.value.trim();
  const username = usernameEl.value.trim();
  const email = emailEl.value.trim();
  const avatar = avatarEl.value.trim() || "styles/logo/Automata_AI_Logo.webp";
  const bio = bioEl.value.trim();

  localStorage.setItem(
    PROFILE_KEY,
    JSON.stringify({ name, username, email, avatar, bio })
  );

  const session = getSession();
  if (session) {
    saveSession({ ...session, name, email });
  }

  avatarImg.src = avatar;
  setStatus("Profile updated.");
});

hydrateForm();

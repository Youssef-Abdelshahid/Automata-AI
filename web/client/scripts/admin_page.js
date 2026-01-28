const allSideMenu = document.querySelectorAll('#sidebar .side-menu.top li a');
const sectionLinks = document.querySelectorAll('#sidebar .side-menu.top li a[data-section]');
const sections = document.querySelectorAll('.admin-section');

function setActiveLink(link) {
	allSideMenu.forEach((i) => {
		i.parentElement.classList.remove('active');
	});
	link.parentElement.classList.add('active');
}

function showSection(id) {
	sections.forEach((section) => {
		section.classList.toggle('active', section.id === id);
	});
}

sectionLinks.forEach((link) => {
	link.addEventListener('click', (e) => {
		const target = link.dataset.section;
		if (!target) return;
		e.preventDefault();
		setActiveLink(link);
		showSection(target);
	});
});

showSection('admin-dashboard');

function getSession() {
	try {
		return JSON.parse(localStorage.getItem("session")) || null;
	} catch {
		return null;
	}
}

function enforceAdminAccess() {
	const session = getSession();
	const email = (session?.email || "").toLowerCase();
	const role = (session?.role || "").toLowerCase();
	if (!session || (role !== "admin" && email !== "admin@automata.ai")) {
		location.href = "index.html";
	}
}

function bindLogout() {
	const logoutBtn = document.getElementById("logoutBtn");
	if (!logoutBtn) return;
	logoutBtn.addEventListener("click", (e) => {
		e.preventDefault();
		localStorage.removeItem("session");
		location.href = "index.html";
	});
}

function bindProfile() {
	const profileLink = document.querySelector("a.profile");
	if (!profileLink) return;
	profileLink.addEventListener("click", (e) => {
		e.preventDefault();
		showSection("admin-profile");
		const profileNav = document.querySelector('[data-section="admin-profile"]');
		if (profileNav) setActiveLink(profileNav);
	});
}

enforceAdminAccess();
bindLogout();
bindProfile();

const userTableBody = document.getElementById("userTableBody");
const taskTableBody = document.getElementById("taskTableBody");

function bindSearch(inputId, tableBody) {
	const input = document.getElementById(inputId);
	if (!input || !tableBody) return;
	input.addEventListener("input", () => {
		const q = input.value.trim().toLowerCase();
		const rows = Array.from(tableBody.querySelectorAll("tr"));
		rows.forEach((row) => {
			const text = row.textContent.toLowerCase();
			row.style.display = text.includes(q) ? "" : "none";
		});
	});
}

bindSearch("userSearch", userTableBody);
bindSearch("taskSearch", taskTableBody);

const userEditor = document.getElementById("userEditor");
const userEditForm = document.getElementById("userEditForm");
const userEditTarget = document.getElementById("userEditTarget");
const userRole = document.getElementById("userRole");
const userProfile = document.getElementById("userProfile");
const userUsername = document.getElementById("userUsername");
const userName = document.getElementById("userName");
const userBio = document.getElementById("userBio");
const cancelUserEdit = document.getElementById("cancelUserEdit");
let activeUserRow = null;

function openUserEditor(row) {
	activeUserRow = row;
	userEditor.hidden = false;
	userEditTarget.textContent = row.dataset.email || "";
	userRole.value = row.dataset.role || "Developer";
	userProfile.value = row.dataset.profile || "";
	userUsername.value = row.dataset.username || "";
	userName.value = row.dataset.name || "";
	userBio.value = row.dataset.bio || "";
	userEditor.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function closeUserEditor() {
	activeUserRow = null;
	userEditor.hidden = true;
	userEditTarget.textContent = "";
	userEditForm.reset();
}

userTableBody?.addEventListener("click", (e) => {
	const btn = e.target.closest(".edit-user");
	if (!btn) return;
	const row = btn.closest("tr");
	if (row) openUserEditor(row);
});

cancelUserEdit?.addEventListener("click", closeUserEditor);

userEditForm?.addEventListener("submit", (e) => {
	e.preventDefault();
	if (!activeUserRow) return;
	const nameVal = userName.value.trim() || "User";
	const usernameVal = userUsername.value.trim() || "user";
	const roleVal = userRole.value;
	const profileVal = userProfile.value.trim() || "styles/logo/Automata_AI_Logo.webp";
	const bioVal = userBio.value.trim();

	activeUserRow.dataset.name = nameVal;
	activeUserRow.dataset.username = usernameVal;
	activeUserRow.dataset.role = roleVal;
	activeUserRow.dataset.profile = profileVal;
	activeUserRow.dataset.bio = bioVal;

	const nameEl = activeUserRow.querySelector(".user-name");
	const handleEl = activeUserRow.querySelector(".user-handle");
	const roleEl = activeUserRow.querySelector(".user-role");
	const imgEl = activeUserRow.querySelector("img");

	if (nameEl) nameEl.textContent = nameVal;
	if (handleEl) handleEl.textContent = "@" + usernameVal;
	if (roleEl) roleEl.textContent = roleVal;
	if (imgEl) imgEl.src = profileVal;

	closeUserEditor();
});

const taskEditor = document.getElementById("taskEditor");
const taskEditForm = document.getElementById("taskEditForm");
const taskEditTarget = document.getElementById("taskEditTarget");
const taskTitle = document.getElementById("taskTitle");
const taskDesc = document.getElementById("taskDesc");
const taskVisibility = document.getElementById("taskVisibility");
const taskAccess = document.getElementById("taskAccess");
const taskAccessField = document.getElementById("taskAccessField");
const cancelTaskEdit = document.getElementById("cancelTaskEdit");
let activeTaskRow = null;

function setAccessVisibility(value) {
	const isPrivate = value.toLowerCase() === "private";
	taskAccessField.style.display = isPrivate ? "block" : "none";
}

function openTaskEditor(row) {
	activeTaskRow = row;
	taskEditor.hidden = false;
	taskEditTarget.textContent = row.dataset.title || "";
	taskTitle.value = row.dataset.title || "";
	taskDesc.value = row.dataset.desc || "";
	taskVisibility.value = row.dataset.visibility || "Public";
	taskAccess.value = row.dataset.access || "";
	setAccessVisibility(taskVisibility.value);
	taskEditor.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function closeTaskEditor() {
	activeTaskRow = null;
	taskEditor.hidden = true;
	taskEditTarget.textContent = "";
	taskEditForm.reset();
	taskAccessField.style.display = "block";
}

taskTableBody?.addEventListener("click", (e) => {
	const btn = e.target.closest(".edit-task");
	if (!btn) return;
	const row = btn.closest("tr");
	if (row) openTaskEditor(row);
});

taskTableBody?.addEventListener("click", (e) => {
	const btn = e.target.closest(".delete-task");
	if (!btn) return;
	const row = btn.closest("tr");
	if (!row) return;
	if (!confirm("Delete this task?")) return;
	if (row === activeTaskRow) {
		closeTaskEditor();
	}
	row.remove();
});

taskVisibility?.addEventListener("change", (e) => {
	setAccessVisibility(e.target.value);
});

cancelTaskEdit?.addEventListener("click", closeTaskEditor);

taskEditForm?.addEventListener("submit", (e) => {
	e.preventDefault();
	if (!activeTaskRow) return;
	const titleVal = taskTitle.value.trim() || "task";
	const descVal = taskDesc.value.trim();
	const visibilityVal = taskVisibility.value;
	const accessVal = taskAccess.value.trim();

	activeTaskRow.dataset.title = titleVal;
	activeTaskRow.dataset.desc = descVal;
	activeTaskRow.dataset.visibility = visibilityVal;
	activeTaskRow.dataset.access = accessVal;

	const titleEl = activeTaskRow.querySelector(".task-title");
	const descEl = activeTaskRow.querySelector(".task-desc");
	const visibilityEl = activeTaskRow.querySelector(".task-visibility");

	if (titleEl) titleEl.textContent = titleVal;
	if (descEl) descEl.textContent = descVal;
	if (visibilityEl) {
		if (visibilityVal.toLowerCase() === "private") {
			const count = accessVal ? accessVal.split(",").filter(Boolean).length : 0;
			visibilityEl.textContent = count ? `Private (${count} users)` : "Private";
		} else {
			visibilityEl.textContent = "Public";
		}
	}

	closeTaskEditor();
});

const adminProfileForm = document.getElementById("adminProfileForm");
const profileStatus = document.getElementById("profileStatus");
const adminName = document.getElementById("adminName");
const adminUsername = document.getElementById("adminUsername");
const adminEmail = document.getElementById("adminEmail");
const adminAvatarUrl = document.getElementById("adminAvatarUrl");
const adminBio = document.getElementById("adminBio");
const profileAvatar = document.getElementById("profileAvatar");
const profileRole = document.getElementById("profileRole");

const ADMIN_PROFILE_KEY = "admin_profile";

function loadAdminProfile() {
	try {
		return JSON.parse(localStorage.getItem(ADMIN_PROFILE_KEY)) || null;
	} catch {
		return null;
	}
}

function setProfileStatus(msg) {
	if (profileStatus) profileStatus.textContent = msg || "";
}

function hydrateAdminProfile() {
	const session = getSession();
	const stored = loadAdminProfile() || {};
	const name = stored.name || session?.name || "System Admin";
	const username = stored.username || "admin";
	const email = stored.email || session?.email || "admin@automata.ai";
	const avatar = stored.avatar || "styles/logo/Automata_AI_Logo.webp";
	const bio = stored.bio || "";

	if (adminName) adminName.value = name;
	if (adminUsername) adminUsername.value = username;
	if (adminEmail) adminEmail.value = email;
	if (adminAvatarUrl) adminAvatarUrl.value = avatar;
	if (adminBio) adminBio.value = bio;
	if (profileAvatar) profileAvatar.src = avatar;
	if (profileRole) profileRole.textContent = session?.role || "Admin";
}

adminAvatarUrl?.addEventListener("input", () => {
	const next = adminAvatarUrl.value.trim() || "styles/logo/Automata_AI_Logo.webp";
	if (profileAvatar) profileAvatar.src = next;
});

adminProfileForm?.addEventListener("submit", (e) => {
	e.preventDefault();
	const name = adminName.value.trim();
	const username = adminUsername.value.trim();
	const email = adminEmail.value.trim();
	const avatar = adminAvatarUrl.value.trim() || "styles/logo/Automata_AI_Logo.webp";
	const bio = adminBio.value.trim();

	localStorage.setItem(
		ADMIN_PROFILE_KEY,
		JSON.stringify({ name, username, email, avatar, bio })
	);

	const session = getSession();
	if (session) {
		localStorage.setItem(
			"session",
			JSON.stringify({ ...session, name, email })
		);
	}

	if (profileAvatar) profileAvatar.src = avatar;
	setProfileStatus("Profile updated.");
});

hydrateAdminProfile();


const menuBar = document.querySelector('#content nav .bx.bx-menu');
const sidebar = document.getElementById('sidebar');

menuBar.addEventListener('click', function () {
	sidebar.classList.toggle('hide');
})


const searchButton = document.querySelector('#content nav form .form-input button');
const searchButtonIcon = document.querySelector('#content nav form .form-input button .bx');
const searchForm = document.querySelector('#content nav form');

searchButton.addEventListener('click', function (e) {
	if(window.innerWidth < 576) {
		e.preventDefault();
		searchForm.classList.toggle('show');
		if(searchForm.classList.contains('show')) {
			searchButtonIcon.classList.replace('bx-search', 'bx-x');
		} else {
			searchButtonIcon.classList.replace('bx-x', 'bx-search');
		}
	}
})





if(window.innerWidth < 768) {
	sidebar.classList.add('hide');
} else if(window.innerWidth > 576) {
	searchButtonIcon.classList.replace('bx-x', 'bx-search');
	searchForm.classList.remove('show');
}


window.addEventListener('resize', function () {
	if(this.innerWidth > 576) {
		searchButtonIcon.classList.replace('bx-x', 'bx-search');
		searchForm.classList.remove('show');
	}
})



const switchMode = document.getElementById('switch-mode');

switchMode.addEventListener('change', function () {
	if(this.checked) {
		document.body.classList.add('dark');
	} else {
		document.body.classList.remove('dark');
	}
})

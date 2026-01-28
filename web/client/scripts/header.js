const CACHE_KEY = 'automata_header_v1';
const NOTIF_KEY = "automata_notifications_v1";

function injectHeader(html) {
  const header = document.getElementById("header");
  if (!header) return;

  // Parse the complete HTML document
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');

  // Extract only the .topbar div
  const topbar = doc.querySelector('.topbar');

  if (topbar) {
    header.innerHTML = topbar.outerHTML;
  } else {
    // Fallback: use the full body content
    header.innerHTML = doc.body.innerHTML;
  }

  initHeader();
}

// 1. Try Cache First (Instant Render)
const cached = localStorage.getItem(CACHE_KEY);
if (cached) {
  injectHeader(cached);
}

// 2. Fetch Fresh (Background Update)
fetch("./header.html")
  .then((r) => r.text())
  .then((html) => {
    // Always update cache for next time
    if (html !== cached) {
      localStorage.setItem(CACHE_KEY, html);
    }

    // If we didn't have a cache, render now
    if (!cached) {
      injectHeader(html);
    }
    // Note: If we did have cache, we skip re-rendering to avoid 
    // replacing the header while the user is looking at it/interacting.
    // The new version will appear on next page load.
  });

function getSession() {
  try {
    const s = JSON.parse(localStorage.getItem("session"));
    return s && s.uid ? s : null;
  } catch {
    return null;
  }
}

function loadNotifications() {
  try {
    const data = JSON.parse(localStorage.getItem(NOTIF_KEY)) || [];
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}
function saveNotifications(list) {
  localStorage.setItem(NOTIF_KEY, JSON.stringify(list));
}
function seedNotificationsIfEmpty() {
  const list = loadNotifications();
  if (list.length) return list;
  const seeded = [
    {
      id: "n1",
      title: "Job started",
      detail: "Task pipeline is running for EdgeVision_v2.",
      time: "Just now",
      href: "status.html",
      read: false,
    },
    {
      id: "n2",
      title: "Job completed",
      detail: "Model export finished for SensorFusion_Alpha.",
      time: "2h ago",
      href: "results.html",
      read: false,
    },
    {
      id: "n3",
      title: "Job failed",
      detail: "Training failed for AcousticEvents_2025.",
      time: "Today",
      href: "status.html",
      read: true,
    },
    {
      id: "n4",
      title: "Report ready",
      detail: "Deployment report is ready for SmartCam_Micro.",
      time: "Yesterday",
      href: "results.html",
      read: true,
    },
    {
      id: "n5",
      title: "New public job",
      detail: "User @edge_labs published a public task.",
      time: "2 days ago",
      href: "public_tasks.html",
      read: true,
    },
  ];
  saveNotifications(seeded);
  return seeded;
}

// Expose checks for other scripts
window.getSession = getSession;
window.hasSession = () => !!getSession();

function initHeader() {
  const session = getSession();
  const isLoggedIn = !!session;

  // 1. Handle Auth Visibility
  const authLinks = document.querySelectorAll(".auth-link");
  const avatarBtn = document.getElementById("avatarBtn");
  const loginBtn = document.getElementById("loginBtn");
  const accountMenu = document.getElementById("accountMenu");
  const notifBtn = document.getElementById("notifBtn");
  const notifMenu = document.getElementById("notifMenu");
  const notifList = document.getElementById("notifList");
  const notifBadge = document.getElementById("notifBadge");
  const notifClear = document.getElementById("notifClear");

  // Toggle Nav Links
  authLinks.forEach((link) => {
    link.style.display = isLoggedIn ? "" : "none";
  });

  // Toggle Buttons
  if (isLoggedIn) {
    if (avatarBtn) avatarBtn.style.display = ""; // Revert to CSS (grid)
    if (loginBtn) loginBtn.style.display = "none";
    if (notifBtn) notifBtn.style.display = "";
    if (notifMenu) notifMenu.style.display = "";
    if (accountMenu) {
      // Ensure it is hidden initially (unless toggled)
      // styles say .menu { display: none } .menu.show { display: block }
      // So we just leave it alone (it has style="display: none" from HTML, we need to clear that to let CSS handle it?)
      // Actually HTML has style="display: none;". We should clear that so CSS defaults take over (which is display: none).
      accountMenu.style.display = "";
    }

    // Populate User Info
    const name = session.name || "Automata Demo";
    const email = session.email || "demo@automata.ai";
    const initials = (name || email)
      .split(/[\s._-]+/)
      .slice(0, 2)
      .map((w) => w?.[0] || "")
      .join("")
      .toUpperCase()
      .slice(0, 2);

    const avatarInitials = document.getElementById("avatarInitials");
    const menuName = document.getElementById("menuName");
    const menuEmail = document.getElementById("menuEmail");

    if (avatarInitials) avatarInitials.textContent = initials || "AD";
    if (menuName) menuName.textContent = name;
    if (menuEmail) menuEmail.textContent = email;

    // Attach Menu Events
    if (avatarBtn && accountMenu) {
      let open = false;
      const setOpen = (v) => {
        open = v;
        accountMenu.classList.toggle("show", open);
        avatarBtn.setAttribute("aria-expanded", String(open));
        accountMenu.setAttribute("aria-hidden", String(!open));
      };

      avatarBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (notifMenu) {
          notifMenu.classList.remove("show");
          if (notifBtn) notifBtn.setAttribute("aria-expanded", "false");
          notifMenu.setAttribute("aria-hidden", "true");
        }
        setOpen(!open);
      });

      document.addEventListener("click", (e) => {
        if (open && !accountMenu.contains(e.target)) {
          setOpen(false);
        }
      });

      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && open) setOpen(false);
      });
    }

    if (notifBtn && notifMenu && notifList && notifBadge && notifClear) {
      let openNotif = false;
      const renderNotifications = (list) => {
        notifList.innerHTML = "";
        const unread = list.filter((n) => !n.read).length;
        notifBadge.textContent = String(unread);
        notifBadge.hidden = unread === 0;

        if (!list.length) {
          const empty = document.createElement("div");
          empty.className = "notif-item";
          empty.innerHTML = '<div class="notif-title">All caught up</div><div class="notif-meta">No notifications yet.</div>';
          notifList.appendChild(empty);
          return;
        }

        list.forEach((n) => {
          const item = document.createElement("div");
          item.className = "notif-item" + (n.read ? "" : " unread");
          item.setAttribute("role", "menuitem");
          item.innerHTML = `
            <div class="notif-title">${n.title}</div>
            <div class="notif-meta">${n.detail}</div>
            <div class="notif-meta">${n.time}</div>
          `;
          item.addEventListener("click", () => {
            const next = loadNotifications().map((x) =>
              x.id === n.id ? { ...x, read: true } : x
            );
            saveNotifications(next);
            renderNotifications(next);
            if (n.href) location.href = n.href;
          });
          notifList.appendChild(item);
        });
      };

      renderNotifications(seedNotificationsIfEmpty());

      const setNotifOpen = (v) => {
        openNotif = v;
        notifMenu.classList.toggle("show", openNotif);
        notifBtn.setAttribute("aria-expanded", String(openNotif));
        notifMenu.setAttribute("aria-hidden", String(!openNotif));
      };

      notifBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (accountMenu) {
          accountMenu.classList.remove("show");
          if (avatarBtn) avatarBtn.setAttribute("aria-expanded", "false");
          accountMenu.setAttribute("aria-hidden", "true");
        }
        setNotifOpen(!openNotif);
      });

      notifClear.addEventListener("click", () => {
        const next = loadNotifications().map((n) => ({ ...n, read: true }));
        saveNotifications(next);
        renderNotifications(next);
      });

      document.addEventListener("click", (e) => {
        if (openNotif && !notifMenu.contains(e.target) && !notifBtn.contains(e.target)) {
          setNotifOpen(false);
        }
      });

      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && openNotif) setNotifOpen(false);
      });
    }

    // Mobile Menu Toggle
    const mobileMenuBtn = document.getElementById("mobileMenuBtn");
    const mainNav = document.getElementById("mainNav");

    if (mobileMenuBtn && mainNav) {
      mobileMenuBtn.addEventListener("click", () => {
        const isExpanded = mobileMenuBtn.getAttribute("aria-expanded") === "true";
        mobileMenuBtn.setAttribute("aria-expanded", !isExpanded);
        mainNav.classList.toggle("show");
      });
    }

    // Profile Navigation
    const profileBtn = document.getElementById("profileBtn");
    if (profileBtn) {
      profileBtn.addEventListener("click", () => {
        const isAdmin = (session.email || "").toLowerCase().includes("admin");
        location.href = isAdmin ? "admin_profile.html" : "user_profile.html";
      });
    }

    // Logout Logic
    const logoutBtn = document.getElementById("logoutBtn");
    if (logoutBtn) {
      logoutBtn.addEventListener("click", () => {
        localStorage.removeItem("session");
        location.href = "index.html";
      });
    }

  } else {
    // Not Logged In
    if (avatarBtn) avatarBtn.style.display = "none";
    if (loginBtn) loginBtn.style.display = ""; // Revert to CSS
    if (accountMenu) accountMenu.style.display = "none";
    if (notifBtn) notifBtn.style.display = "none";
    if (notifMenu) notifMenu.style.display = "none";
  }

  // 2. Active Link Highlight
  const currentPage = location.pathname.split("/").pop() || "index.html";
  const navLinks = document.querySelectorAll(".nav-link");

  navLinks.forEach((link) => {
    if (link.getAttribute("href") === currentPage) {
      link.classList.add("active");
    }
  });
}

// Demo helper to quickly login as admin
window.demoLoginAdmin = () => {
  const adminSession = {
    uid: "admin-demo-id",
    name: "System Admin",
    email: "admin@automata.ai",
    role: "admin"
  };
  localStorage.setItem("session", JSON.stringify(adminSession));
  location.href = "dashboard.html";
};

// Demo helper to quickly login as user
window.demoLoginUser = () => {
  const userSession = {
    uid: "demo-uid",
    name: "Automata Demo",
    email: "demo@automata.ai",
    role: "user"
  };
  localStorage.setItem("session", JSON.stringify(userSession));
  location.href = "dashboard.html";
};

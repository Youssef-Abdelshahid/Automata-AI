function getSession() {
  try {
    return JSON.parse(localStorage.getItem("session")) || null;
  } catch {
    return null;
  }
}

function setAuthLink() {
  const link = document.getElementById("authLink");
  if (!link) return;
  const session = getSession();
  if (session) {
    link.textContent = "Dashboard";
    link.href = "./dashboard.html";
  } else {
    link.textContent = "Login / Sign up";
    link.href = "./auth.html";
  }
}

function redirectToAuth(target) {
  const url = target ? `./auth.html?redirect=${encodeURIComponent(target)}` : "./auth.html";
  location.href = url;
}

function bindPlanActions() {
  const buttons = document.querySelectorAll(".plan-action");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const plan = btn.dataset.plan;
      const session = getSession();

      if (plan === "developer") {
        if (!session) {
          redirectToAuth("dashboard.html");
          return;
        }
        location.href = "./dashboard.html";
      }

      if (plan === "power") {
        if (!session) {
          redirectToAuth("billing.html");
          return;
        }
        location.href = "./billing.html";
      }
    });
  });
}

setAuthLink();
bindPlanActions();

const DEFAULT_USER = {
  email: "demo@automata.ai",
  password: "automata123",
  uid: "demo-uid",
};

function loadUsers() {
  try {
    const data = JSON.parse(localStorage.getItem("users")) || [];
    if (!data.find((u) => u.email === DEFAULT_USER.email))
      data.push(DEFAULT_USER);
    return data;
  } catch {
    return [DEFAULT_USER];
  }
}
function saveUsers(arr) {
  localStorage.setItem("users", JSON.stringify(arr));
}
function setSession(u) {
  localStorage.setItem(
    "session",
    JSON.stringify({ uid: u.uid, email: u.email })
  );
}
function uid() {
  return "u_" + Math.random().toString(36).slice(2, 10);
}

const form = document.getElementById("authForm");
const submitBtn = document.getElementById("submitBtn");
const switchBtn = document.getElementById("switchBtn");
const switchPrompt = document.getElementById("switchPrompt");
const titleEl = document.getElementById("auth-title");
const subtitleEl = document.getElementById("auth-subtitle");
const confirmWrap = document.getElementById("confirmWrap");
const cardEl = document.querySelector(".card");

const emailEl = document.getElementById("email");
const passwordEl = document.getElementById("password");
const confirmEl = document.getElementById("confirm");

const emailErr = document.getElementById("emailErr");
const passwordErr = document.getElementById("passwordErr");
const confirmErr = document.getElementById("confirmErr");

let users = loadUsers();
let mode = "signin";
const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function shakeCard() {
  cardEl.classList.remove("shake");
  void cardEl.offsetWidth;
  cardEl.classList.add("shake");
}

function setMode(next) {
  mode = next;
  const isUp = mode === "signup";
  titleEl.textContent = isUp ? "Create Account" : "Welcome Back";
  subtitleEl.textContent = isUp
    ? "Join Automata AI to build and deploy edge models."
    : "Sign in to manage your AI models.";
  submitBtn.textContent = isUp ? "Create Account" : "Sign In";
  switchPrompt.textContent = isUp
    ? "Already have an account?"
    : "New to Automata AI?";
  switchBtn.textContent = isUp ? "Sign in" : "Create account";
  confirmWrap.hidden = !isUp;

  form.reset();
  [emailErr, passwordErr, confirmErr].forEach((e) => (e.textContent = ""));
}

switchBtn.addEventListener("click", () =>
  setMode(mode === "signin" ? "signup" : "signin")
);

function validate(showErrors = false) {
  const email = emailEl.value.trim();
  const pass = passwordEl.value;
  const isUp = mode === "signup";
  let ok = true;

  if (!email) {
    if (showErrors) emailErr.textContent = "Email is required.";
    ok = false;
  } else if (!emailRe.test(email)) {
    if (showErrors) emailErr.textContent = "Enter a valid email address.";
    ok = false;
  } else {
    emailErr.textContent = "";
  }

  if (!pass) {
    if (showErrors) passwordErr.textContent = "Password is required.";
    ok = false;
  } else if (pass.length < 8) {
    if (showErrors) passwordErr.textContent = "Use at least 8 characters.";
    ok = false;
  } else {
    passwordErr.textContent = "";
  }

  if (isUp) {
    const cp = confirmEl.value;
    if (!cp) {
      if (showErrors) confirmErr.textContent = "Please confirm your password.";
      ok = false;
    } else if (cp !== pass) {
      if (showErrors) confirmErr.textContent = "Passwords do not match.";
      ok = false;
    } else {
      confirmErr.textContent = "";
    }
  } else {
    confirmErr.textContent = "";
  }

  return ok;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const ok = validate(true);
  if (!ok) {
    shakeCard();
    return;
  }

  const email = emailEl.value.trim().toLowerCase();
  const pass = passwordEl.value;

  if (mode === "signin") {
    const user = users.find(
      (u) => u.email.toLowerCase() === email && u.password === pass
    );
    if (!user) {
      passwordErr.textContent = "Incorrect email or password.";
      shakeCard();
      return;
    }
    setSession(user);
    window.location.href = "./dashboard.html";
  } else {
    if (users.some((u) => u.email.toLowerCase() === email)) {
      emailErr.textContent = "This email is already registered.";
      shakeCard();
      return;
    }
    const newUser = { email, password: pass, uid: uid() };
    users.push(newUser);
    saveUsers(users);
    setSession(newUser);
    window.location.href = "./dashboard.html";
  }
});

// init
setMode("signin");

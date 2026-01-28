const addPaymentBtn = document.getElementById("addPaymentBtn");
const cancelPaymentBtn = document.getElementById("cancelPaymentBtn");
const paymentPanel = document.getElementById("paymentPanel");
const form = document.getElementById("paymentForm");
const statusEl = document.getElementById("paymentStatus");

const cardName = document.getElementById("cardName");
const cardNumber = document.getElementById("cardNumber");
const cardExpiry = document.getElementById("cardExpiry");
const cardCvc = document.getElementById("cardCvc");
const errName = document.getElementById("cardNameErr");
const errNumber = document.getElementById("cardNumberErr");
const errExpiry = document.getElementById("cardExpiryErr");
const errCvc = document.getElementById("cardCvcErr");

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

function openPaymentForm() {
  paymentPanel?.classList.add("open");
  setTimeout(() => cardName?.focus(), 0);
}
function closePaymentForm() {
  paymentPanel?.classList.remove("open");
  form?.reset();
  clearErrors();
}

function clearErrors() {
  [errName, errNumber, errExpiry, errCvc].forEach((el) => {
    el.textContent = "";
  });
}

function luhnCheck(value) {
  let sum = 0;
  let shouldDouble = false;
  for (let i = value.length - 1; i >= 0; i -= 1) {
    let digit = Number(value[i]);
    if (Number.isNaN(digit)) return false;
    if (shouldDouble) {
      digit *= 2;
      if (digit > 9) digit -= 9;
    }
    sum += digit;
    shouldDouble = !shouldDouble;
  }
  return sum % 10 === 0;
}

function validate() {
  clearErrors();
  let ok = true;

  const nameVal = cardName.value.trim();
  if (nameVal.length < 2) {
    errName.textContent = "Enter the cardholder name.";
    ok = false;
  }

  const numRaw = cardNumber.value.replace(/\s+/g, "");
  if (!/^\d{13,19}$/.test(numRaw) || !luhnCheck(numRaw)) {
    errNumber.textContent = "Enter a valid card number.";
    ok = false;
  }

  const expRaw = cardExpiry.value.replace(/\s+/g, "");
  const match = expRaw.match(/^(\d{2})\/?(\d{2})$/);
  if (!match) {
    errExpiry.textContent = "Use MM / YY.";
    ok = false;
  } else {
    const month = Number(match[1]);
    const year = Number(match[2]) + 2000;
    const now = new Date();
    const expDate = new Date(year, month - 1, 1);
    const endOfMonth = new Date(year, month, 0);
    if (month < 1 || month > 12) {
      errExpiry.textContent = "Month must be between 01 and 12.";
      ok = false;
    } else if (endOfMonth < new Date(now.getFullYear(), now.getMonth(), 1)) {
      errExpiry.textContent = "Card is expired.";
      ok = false;
    }
  }

  const cvcVal = cardCvc.value.trim();
  if (!/^\d{3,4}$/.test(cvcVal)) {
    errCvc.textContent = "CVC must be 3 or 4 digits.";
    ok = false;
  }

  return ok;
}

function formatCardNumber(value) {
  return value.replace(/\D+/g, "").replace(/(\d{4})(?=\d)/g, "$1 ");
}

function formatExpiry(value) {
  const digits = value.replace(/\D+/g, "").slice(0, 4);
  if (digits.length <= 2) return digits;
  return `${digits.slice(0, 2)} / ${digits.slice(2)}`;
}

function setStatusFromSession() {
  const session = getSession();
  const payment = session?.payment;
  if (!payment) {
    statusEl.textContent = "No payment method added.";
    statusEl.classList.remove("success", "error");
    addPaymentBtn.textContent = "Add Payment Method";
    return;
  }
  statusEl.textContent = `Card ending ${payment.last4} • exp ${payment.expiry}`;
  statusEl.classList.add("success");
  statusEl.classList.remove("error");
  addPaymentBtn.textContent = "Edit Payment Method";
}

addPaymentBtn?.addEventListener("click", openPaymentForm);
cancelPaymentBtn?.addEventListener("click", closePaymentForm);

cardNumber?.addEventListener("input", (e) => {
  const start = e.target.selectionStart || 0;
  e.target.value = formatCardNumber(e.target.value);
  e.target.selectionStart = e.target.selectionEnd = start;
});
cardExpiry?.addEventListener("input", (e) => {
  e.target.value = formatExpiry(e.target.value);
});
cardCvc?.addEventListener("input", (e) => {
  e.target.value = e.target.value.replace(/\D+/g, "").slice(0, 4);
});

form?.addEventListener("submit", (e) => {
  e.preventDefault();
  const ok = validate();
  if (!ok) return;

  const numRaw = cardNumber.value.replace(/\s+/g, "");
  const session = getSession() || {};
  const payment = {
    name: cardName.value.trim(),
    last4: numRaw.slice(-4),
    expiry: cardExpiry.value.replace(/\s+/g, ""),
  };
  saveSession({ ...session, payment });
  setStatusFromSession();
  form.reset();
  closePaymentForm();
});

setStatusFromSession();

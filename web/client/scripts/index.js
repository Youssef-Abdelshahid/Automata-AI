function hasSession() {
  try {
    const s = JSON.parse(localStorage.getItem("session"));
    return !!(s && s.uid);
  } catch {
    return false;
  }
}
function nextUrl() {
  return hasSession() ? "./dashboard.html" : "./auth.html";
}

const continueBtn = document.getElementById("continueBtn");

const loggedIn = hasSession();
const arrowSvg = `
  <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M13 5l7 7-7 7M5 12h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
`;

if (continueBtn) {
  continueBtn.innerHTML =
    (loggedIn ? "Continue to Dashboard" : "Get Started for Free") + arrowSvg;

  continueBtn.addEventListener("click", () => {
    location.href = nextUrl();
  });
}

const reduceMotion =
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (!reduceMotion) {
  const heroTitle = document.querySelector(".hero h1");
  const heroSub = document.querySelector(".hero .sub");
  let ticking = false;

  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      const y = window.scrollY || 0;
      const s = Math.min(Math.max(y, 0), 240);
      if (heroTitle) {
        heroTitle.style.transform = `translateY(${s * 0.18}px)`;
        heroTitle.style.opacity = String(1 - s / 320);
      }
      if (heroSub) {
        heroSub.style.transform = `translateY(${s * 0.24}px)`;
        heroSub.style.opacity = String(1 - s / 280);
      }
      ticking = false;
    });
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

(function revealCardsOnScroll() {
  const cards = Array.from(document.querySelectorAll(".cards .card"));
  if (!cards.length) return;

  if (reduceMotion || !("IntersectionObserver" in window)) {
    cards.forEach((c) => c.classList.add("is-visible"));
    return;
  }

  const baseDelayMs = 220; 

  const obs = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;

        const el = entry.target;
        const i = Number(el.dataset.idx || 0);
        setTimeout(() => el.classList.add("is-visible"), i * baseDelayMs);

        obs.unobserve(el);
      });
    },
    {
      root: null,
      rootMargin: "0px 0px -5% 0px",
      threshold: 0.06,
    }
  );

  cards.forEach((el, i) => {
    el.dataset.idx = i;
    obs.observe(el);
  });
})();

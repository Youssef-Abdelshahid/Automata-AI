// Public Jobs - Like functionality
const likeBtns = document.querySelectorAll(".like-btn");
likeBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const isLiked = btn.getAttribute("data-liked") === "true";
    const likeCountEl = btn.querySelector(".like-count");
    let count = parseInt(likeCountEl.textContent);

    if (isLiked) {
      count--;
      btn.setAttribute("data-liked", "false");
    } else {
      count++;
      btn.setAttribute("data-liked", "true");
    }

    likeCountEl.textContent = count;
  });
});

// User profile links
const userLinks = document.querySelectorAll(".user-link");
userLinks.forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const username = link.getAttribute("data-user");
    location.href = `./profile.html?user=${username}`;
  });
});

// View Results
const viewResultsBtns = document.querySelectorAll(".view-results-btn");
viewResultsBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const jobId = btn.getAttribute("data-job-id");
    location.href = `./job-results.html?id=${jobId}`;
  });
});

// Auth check helper
const hasSession = () => {
  try {
    const s = JSON.parse(localStorage.getItem("session"));
    return !!(s && s.uid);
  } catch {
    return false;
  }
};

// Download Model (requires login)
const downloadBtns = document.querySelectorAll(".download-model-btn");
downloadBtns.forEach((btn) => {
  btn.addEventListener("click", (e) => {
    if (!hasSession()) {
      e.preventDefault();
      sessionStorage.setItem("redirectAfterLogin", "download");
      sessionStorage.setItem("downloadJobId", btn.getAttribute("data-job-id"));
      location.href = "./auth.html";
    } else {
      const jobId = btn.getAttribute("data-job-id");
      console.log("Downloading model for job:", jobId);
      alert("Download started for job #" + jobId);
    }
  });
});


/* =========================================
   DROPDOWN & FILTER LOGIC (Adapted from tasks.js)
   ========================================= */

const searchInput = document.getElementById("searchJobs");
const sortBtn = document.getElementById("sortBtn");
const sortList = document.getElementById("sortList");
const sortLabel = document.getElementById("sortLabel");

let currentSort = "recent";
let currentSearch = "";

// Close dropdowns helper
function closeAllDropdowns(exceptEl = null) {
  document.querySelectorAll(".dropdown.open").forEach((dd) => {
    if (dd !== exceptEl) {
      dd.classList.remove("open");
      dd.querySelector(".dropdown-btn")?.setAttribute("aria-expanded", "false");
    }
  });
}

// Setup custom dropdown
function setupDropdown(btn, list, labelEl) {
  const parent = btn.parentElement;
  
  // Toggle open
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const willOpen = !parent.classList.contains("open");
    closeAllDropdowns(parent);
    parent.classList.toggle("open", willOpen);
    btn.setAttribute("aria-expanded", String(willOpen));
  });

  // Prevent closing when clicking inside
  parent.addEventListener("click", (e) => e.stopPropagation());
  list.addEventListener("click", (e) => e.stopPropagation());

  // Handle item selection
  list.querySelectorAll("li").forEach((li) => {
    li.addEventListener("click", () => {
      // Update UI selection
      list.querySelectorAll("li").forEach((n) => n.classList.remove("selected"));
      li.classList.add("selected");
      
      const value = li.getAttribute("data-value");
      const text = li.textContent;
      
      // Update label
      labelEl.textContent = text;
      
      // Update state and re-filter
      currentSort = value;
      applyFilter();
      
      // Close dropdown
      closeAllDropdowns();
    });
  });
}

// Global click to close dropdowns
document.addEventListener("click", () => closeAllDropdowns());
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeAllDropdowns();
});

// Initialize Sort Dropdown
if (sortBtn && sortList) {
  setupDropdown(sortBtn, sortList, sortLabel);
}

// Search Input Listener
if (searchInput) {
  searchInput.addEventListener("input", (e) => {
    currentSearch = e.target.value.toLowerCase();
    applyFilter();
  });
}


/* =========================================
   PAGINATION & DISPLAY LOGIC
   ========================================= */

const jobsPerPage = 5;
let currentPage = 1;
const jobList = document.getElementById("jobList");
const pagination = document.getElementById("pagination");

// Collect all job cards on load
const allJobs = Array.from(document.querySelectorAll(".job-card"));
let filteredJobs = [...allJobs];

function applyFilter() {
  // 1. Filter by Search
  if (currentSearch) {
    filteredJobs = allJobs.filter((job) => {
      const title = job.querySelector("h3")?.textContent.toLowerCase() || "";
      const desc = job.querySelector("p")?.textContent.toLowerCase() || "";
      const user = job.querySelector(".user-link")?.textContent.toLowerCase() || "";
      return (
        title.includes(currentSearch) ||
        desc.includes(currentSearch) ||
        user.includes(currentSearch)
      );
    });
  } else {
    filteredJobs = [...allJobs];
  }

  // 2. Sort
  if (currentSort === "likes") {
    filteredJobs.sort((a, b) => {
      const likesA = parseInt(a.querySelector(".like-count").textContent) || 0;
      const likesB = parseInt(b.querySelector(".like-count").textContent) || 0;
      return likesB - likesA;
    });
  } else if (currentSort === "recent") {
    // Assuming original DOM order is "recent" (or add data-date if needed)
    // For now, we revert to original index logic if we tracked it, 
    // but simplified: just reset to original order if filtered list allows
    if (!currentSearch) {
       // If no search, we can just use allJobs order (assuming it was recent first)
       filteredJobs = [...allJobs]; 
    } 
    // If search is active, we just keep the filtered array order (stable)
  }
  // "popular" could be same as likes for this demo

  // 3. Reset to page 1 and show
  showPage(1);
}

function showPage(page) {
  currentPage = page;
  const totalPages = Math.ceil(filteredJobs.length / jobsPerPage);

  // Hide ALL jobs first (from the DOM)
  allJobs.forEach((job) => {
    job.style.display = "none";
  });

  // Calculate slice for current page
  const start = (page - 1) * jobsPerPage;
  const end = start + jobsPerPage;
  const jobsToShow = filteredJobs.slice(start, end);

  // Show only relevant jobs
  jobsToShow.forEach((job) => {
     job.style.display = "block";
     // Re-trigger visual fade-in if needed, or simple block
     job.style.opacity = "1";
     job.style.transform = "translateY(0)";
  });
  
  // Show no results message if needed?
  // (Optional: add empty state handling if filteredJobs.length === 0)

  renderPagination(totalPages);
}

function renderPagination(totalPages) {
  if (!pagination) return;
  pagination.innerHTML = "";
  
  if (totalPages <= 1) return;

  // Prev
  if (currentPage > 1) {
    const prevBtn = document.createElement("button");
    prevBtn.textContent = "«";
    prevBtn.onclick = () => showPage(currentPage - 1);
    pagination.appendChild(prevBtn);
  }

  // Numbers
  for (let i = 1; i <= totalPages; i++) {
    const btn = document.createElement("button");
    btn.textContent = i;
    if (i === currentPage) btn.classList.add("active");
    btn.onclick = () => showPage(i);
    pagination.appendChild(btn);
  }

  // Next
  if (currentPage < totalPages) {
    const nextBtn = document.createElement("button");
    nextBtn.textContent = "»";
    nextBtn.onclick = () => showPage(currentPage + 1);
    pagination.appendChild(nextBtn);
  }
}

// Initial Render
applyFilter();

// Elements
const step3Btn = document.getElementById("step3-btn");
const closeHardware = document.getElementById("closeHardware");

// Helper to open overlay
function openOverlay(overlay) {
    overlay.classList.remove("hidden");
    document.body.style.overflow = "hidden"; // Prevent scrolling
}

// Helper to close overlay
function closeOverlay(overlay) {
    overlay.classList.add("hidden");
    document.body.style.overflow = ""; // Restore scrolling

    // Collapse all details
    const cards = overlay.querySelectorAll(".info-card .extra-info");
    cards.forEach(c => c.classList.add("hidden"));
}

// Listeners
if (step3Btn) {
    step3Btn.addEventListener("click", () => openOverlay(hardwareOverlay));
}

// Close events
if (closeHardware) closeHardware.addEventListener("click", () => closeOverlay(hardwareOverlay));

// Close on outside click
if (hardwareOverlay) {
    hardwareOverlay.addEventListener("click", (e) => {
        if (e.target === hardwareOverlay) closeOverlay(hardwareOverlay);
    });
}

// Expandable Cards Logic removed - Content is static now
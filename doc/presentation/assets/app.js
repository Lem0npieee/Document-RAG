const slides = Array.from(document.querySelectorAll(".slide"));
const overview = document.getElementById("overview");
const overviewList = document.getElementById("overview-list");

let currentIndex = 0;

function getSlideMeta(slide, index) {
  const title = slide.dataset.title || `第 ${index + 1} 页`;
  const summary =
    slide.querySelector("h2, .cover-subtitle, .focus-strip p, .closing-note")?.textContent?.trim() ||
    "";
  return { title, summary };
}

function pad(number) {
  return String(number).padStart(2, "0");
}

function buildNavigation() {
  slides.forEach((slide, index) => {
    const meta = getSlideMeta(slide, index);

    const card = document.createElement("button");
    card.type = "button";
    card.className = "overview-card";
    card.innerHTML = `
      <span class="overview-card__number">${pad(index + 1)}</span>
      <h3>${meta.title}</h3>
      <p>${meta.summary}</p>
    `;
    card.addEventListener("click", () => {
      goTo(index);
      closeOverview();
    });
    overviewList.appendChild(card);
  });
}

function syncNavigation() {
  slides.forEach((slide, index) => {
    slide.classList.toggle("is-active", index === currentIndex);
  });
  window.location.hash = `slide-${currentIndex + 1}`;
}

function goTo(index) {
  if (index < 0 || index >= slides.length) return;
  currentIndex = index;
  syncNavigation();
}

function next() {
  goTo(Math.min(currentIndex + 1, slides.length - 1));
}

function prev() {
  goTo(Math.max(currentIndex - 1, 0));
}

function openOverview() {
  overview.hidden = false;
}

function closeOverview() {
  overview.hidden = true;
}

function toggleOverview() {
  overview.hidden ? openOverview() : closeOverview();
}

function initHash() {
  const match = window.location.hash.match(/slide-(\d+)/);
  if (!match) return;
  const index = Number(match[1]) - 1;
  if (!Number.isNaN(index)) goTo(index);
}

document.getElementById("close-overview").addEventListener("click", closeOverview);
document.getElementById("click-prev").addEventListener("click", prev);
document.getElementById("click-next").addEventListener("click", next);
overview.addEventListener("click", (event) => {
  if (event.target === overview) closeOverview();
});

window.addEventListener("keydown", (event) => {
  if (!overview.hidden) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeOverview();
    }
    return;
  }

  if (event.key === "ArrowRight" || event.key === "PageDown" || event.key === " ") {
    event.preventDefault();
    next();
    return;
  }

  if (event.key === "ArrowLeft" || event.key === "PageUp") {
    event.preventDefault();
    prev();
    return;
  }

  if (event.key.toLowerCase() === "m" || event.key.toLowerCase() === "o") {
    event.preventDefault();
    toggleOverview();
    return;
  }

});

buildNavigation();
initHash();
syncNavigation();

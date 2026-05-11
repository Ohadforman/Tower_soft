import { renderShell } from "./components/shell.js";
import { getPage } from "./pages/index.js";
import { getCurrentRoute, installRouter } from "./router.js";

const app = document.getElementById("app");

function attachMotion() {
  const hero = document.querySelector(".hero-panel");
  if (!hero) return;

  hero.addEventListener("mousemove", (event) => {
    const rect = hero.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) - 0.5;
    const py = ((event.clientY - rect.top) / rect.height) - 0.5;
    hero.style.setProperty("--tilt-x", `${px * 18}px`);
    hero.style.setProperty("--tilt-y", `${py * 14}px`);
    hero.style.setProperty("--glow-x", `${50 + px * 22}%`);
    hero.style.setProperty("--glow-y", `${42 + py * 18}%`);
  });

  hero.addEventListener("mouseleave", () => {
    hero.style.setProperty("--tilt-x", "0px");
    hero.style.setProperty("--tilt-y", "0px");
    hero.style.setProperty("--glow-x", "50%");
    hero.style.setProperty("--glow-y", "42%");
  });
}

async function renderRoute() {
  const route = getCurrentRoute();
  const page = getPage(route);
  app.innerHTML = renderShell(page.route);
  const pageRoot = document.getElementById("page-root");
  pageRoot.innerHTML = `<div class="loading-state">Loading ${page.label}...</div>`;

  try {
    pageRoot.innerHTML = await page.render();
    attachMotion();
  } catch (error) {
    console.error(error);
    pageRoot.innerHTML = `
      <section class="page-panel">
        <div class="section-heading">
          <span>Error</span>
          <h2>Page failed to load</h2>
          <p>${error.message}</p>
        </div>
      </section>
    `;
  }
}

installRouter(renderRoute);

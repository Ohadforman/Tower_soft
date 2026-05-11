import { pageRegistry } from "../pages/index.js";

export function renderShell(activeRoute) {
  const navItems = pageRegistry
    .map(
      (page) => `
        <a class="nav-link ${page.route === activeRoute ? "is-active" : ""}" href="#${page.route}">
          <span>${page.eyebrow}</span>
          <strong>${page.label}</strong>
        </a>
      `,
    )
    .join("");

  return `
    <div class="app-shell">
      <header class="topbar">
        <div class="brand-mark"></div>
        <div class="brand-copy">
          <span>Tower Rebuild</span>
          <strong>Command Surface</strong>
        </div>
        <div class="topbar-meta">No Streamlit. Routed pages. Live CSV data.</div>
      </header>
      <div class="layout-grid">
        <aside class="sidebar">
          <div class="sidebar-copy">
            <span class="sidebar-kicker">Page System</span>
            <h1>Build the full app like this.</h1>
            <p>
              Every page lives in the registry and renders into the same shared shell.
              This is the foundation for a full Tower replacement.
            </p>
          </div>
          <nav class="sidebar-nav">${navItems}</nav>
          <div class="sidebar-footer">
            <span>How to add a page</span>
            <p>Create a page module, register it in <code>pages/index.js</code>, then link data through a small API endpoint.</p>
          </div>
        </aside>
        <main id="page-root" class="page-root"></main>
      </div>
    </div>
  `;
}

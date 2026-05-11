function metricMarkup(metrics) {
  return metrics
    .map(
      (item) => `
        <div class="metric-pill">
          <span>${item.label}</span>
          <strong>${item.value}</strong>
        </div>
      `,
    )
    .join("");
}

function timelineMarkup(rows) {
  return rows
    .map(
      (item, index) => `
        <article class="flow-row">
          <div class="flow-id">Draw ${String(index + 1).padStart(2, "0")}</div>
          <div class="flow-main">
            <div>
              <h3>${item.preform || "Unknown preform"}</h3>
              <p>${item.project || "No project name"} · ${item.status} · ${item.priority}</p>
            </div>
            <div class="flow-value">${item.length} m</div>
          </div>
        </article>
      `,
    )
    .join("");
}

export async function renderHomePage() {
  const response = await fetch("/api/home");
  const data = await response.json();

  return `
    <section class="hero-panel hero-home">
      <div class="hero-backdrop"></div>
      <div class="hero-copy">
        <div class="eyebrow">${data.hero.subtitle}</div>
        <h2>${data.hero.title}</h2>
        <p>${data.hero.summary}</p>
        <div class="metric-row">
          ${metricMarkup(data.metrics)}
        </div>
      </div>
      <div class="hero-radar">
        <div class="radar-ring ring-a"></div>
        <div class="radar-ring ring-b"></div>
        <div class="radar-sweep"></div>
        <div class="radar-dot dot-a"></div>
        <div class="radar-dot dot-b"></div>
        <div class="radar-dot dot-c"></div>
      </div>
    </section>

    <section class="content-grid">
      <div class="content-main">
        <div class="section-heading">
          <span>Live draws</span>
          <h3>Recent order stream</h3>
        </div>
        <div class="flow-list">
          ${timelineMarkup(data.draws.recent)}
        </div>
      </div>

      <aside class="content-rail">
        <div class="rail-block">
          <span>Schedule pressure</span>
          <strong>${data.schedule.total} total events</strong>
          <p>${Object.keys(data.schedule.type_counts).join(" · ")}</p>
        </div>
        <div class="rail-block">
          <span>Parts status</span>
          <strong>${data.parts.total} tracked orders</strong>
          <p>${data.parts.maintenance_open} maintenance-related open orders need attention.</p>
        </div>
        <div class="rail-block">
          <span>Inventory signal</span>
          <strong>${data.inventory.tracked_parts} inventory entries</strong>
          <p>${data.inventory.low_stock.length} parts are currently at or below min level.</p>
        </div>
      </aside>
    </section>
  `;
}

function maintenanceEvents(items) {
  return items
    .map(
      (item) => `
        <article class="event-row compact-row">
          <div class="event-copy">
            <h3>${item.event_type}</h3>
            <p>${item.description || "No maintenance notes"}</p>
          </div>
          <div class="event-time">
            <strong>${item.start}</strong>
            <span>${item.duration_hours} h</span>
          </div>
        </article>
      `,
    )
    .join("");
}

export async function renderMaintenancePage() {
  const response = await fetch("/api/maintenance");
  const data = await response.json();

  return `
    <section class="page-panel split-panel maintenance-panel">
      <div>
        <div class="section-heading">
          <span>Maintenance</span>
          <h2>Service control layer</h2>
          <p>This is the same app shell, another route, and another endpoint. That is the core pattern we will scale.</p>
        </div>
        <div class="metric-row">
          <div class="metric-pill"><span>Upcoming maintenance</span><strong>${data.upcoming_maintenance.length}</strong></div>
          <div class="metric-pill"><span>Open maintenance orders</span><strong>${data.maintenance_open_orders}</strong></div>
          <div class="metric-pill"><span>Low stock blockers</span><strong>${data.low_stock.length}</strong></div>
        </div>
        <div class="event-list">${maintenanceEvents(data.upcoming_maintenance)}</div>
      </div>
      <div>
        <div class="section-heading minimal">
          <span>Readiness</span>
          <h3>Potential blockers</h3>
        </div>
        <div class="stack-list">
          ${data.low_stock.map((item) => `
            <article class="supply-row compact">
              <div>
                <h3>${item.part_name}</h3>
                <p>${item.component || "Unknown component"}</p>
              </div>
              <div class="supply-meta">
                <strong>${item.quantity}</strong>
                <span>Min ${item.min_level}</span>
              </div>
            </article>
          `).join("")}
        </div>
      </div>
    </section>
  `;
}

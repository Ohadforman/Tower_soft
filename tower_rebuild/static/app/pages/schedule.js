function eventMarkup(items) {
  return items
    .map(
      (item, index) => `
        <article class="event-row">
          <div class="event-index">${String(index + 1).padStart(2, "0")}</div>
          <div class="event-copy">
            <h3>${item.event_type}</h3>
            <p>${item.description || "No description supplied"}</p>
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

export async function renderSchedulePage() {
  const response = await fetch("/api/schedule");
  const data = await response.json();
  const badges = Object.entries(data.type_counts)
    .map(([label, value]) => `<div class="metric-pill"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  return `
    <section class="page-panel">
      <div class="section-heading">
        <span>Schedule</span>
        <h2>Operations timeline</h2>
        <p>The page is a dedicated module. It pulls only its own endpoint and renders inside the shared shell.</p>
      </div>
      <div class="metric-row">${badges}</div>
      <div class="event-list">${eventMarkup(data.upcoming)}</div>
    </section>
  `;
}

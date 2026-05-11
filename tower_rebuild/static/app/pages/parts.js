function orderMarkup(items) {
  return items
    .map(
      (item) => `
        <article class="supply-row">
          <div>
            <h3>${item.part_name}</h3>
            <p>${item.details || "No details"}</p>
          </div>
          <div class="supply-meta">
            <strong>${item.status}</strong>
            <span>${item.project || item.company || "General"}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

function lowStockMarkup(items) {
  return items
    .map(
      (item) => `
        <article class="supply-row compact">
          <div>
            <h3>${item.part_name}</h3>
            <p>${item.component || "Unassigned component"}</p>
          </div>
          <div class="supply-meta">
            <strong>${item.quantity}</strong>
            <span>${item.location || "Unknown location"}</span>
          </div>
        </article>
      `,
    )
    .join("");
}

export async function renderPartsPage() {
  const response = await fetch("/api/parts");
  const data = await response.json();
  const statusPills = Object.entries(data.status_counts)
    .map(([label, value]) => `<div class="metric-pill"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  return `
    <section class="page-panel split-panel">
      <div>
        <div class="section-heading">
          <span>Supply</span>
          <h2>Part orders</h2>
          <p>Open part orders and inventory pressure can live on their own page without inheriting any Streamlit layout rules.</p>
        </div>
        <div class="metric-row">${statusPills}</div>
        <div class="stack-list">${orderMarkup(data.open_orders)}</div>
      </div>
      <div>
        <div class="section-heading minimal">
          <span>Inventory</span>
          <h3>Low stock watch</h3>
        </div>
        <div class="stack-list">${lowStockMarkup(data.inventory.low_stock)}</div>
      </div>
    </section>
  `;
}

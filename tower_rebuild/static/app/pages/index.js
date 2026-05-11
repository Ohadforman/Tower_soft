import { renderHomePage } from "./home.js";
import { renderSchedulePage } from "./schedule.js";
import { renderPartsPage } from "./parts.js";
import { renderMaintenancePage } from "./maintenance.js";

export const pageRegistry = [
  {
    route: "/home",
    key: "home",
    label: "Home",
    eyebrow: "Core view",
    description: "Command deck summary of Tower operations.",
    render: renderHomePage,
  },
  {
    route: "/schedule",
    key: "schedule",
    label: "Schedule",
    eyebrow: "Operations",
    description: "Timeline and event stream for tower work.",
    render: renderSchedulePage,
  },
  {
    route: "/parts",
    key: "parts",
    label: "Parts",
    eyebrow: "Supply",
    description: "Orders, inventory, and supply pressure.",
    render: renderPartsPage,
  },
  {
    route: "/maintenance",
    key: "maintenance",
    label: "Maintenance",
    eyebrow: "Service",
    description: "Upcoming service and missing support items.",
    render: renderMaintenancePage,
  },
];

export function getPage(route) {
  return pageRegistry.find((page) => page.route === route) || pageRegistry[0];
}

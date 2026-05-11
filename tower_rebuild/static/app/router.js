export function getCurrentRoute() {
  const hash = window.location.hash.replace(/^#/, "") || "/home";
  return hash.startsWith("/") ? hash : `/${hash}`;
}

export function navigate(route) {
  window.location.hash = route;
}

export function installRouter(onRouteChange) {
  window.addEventListener("hashchange", onRouteChange);
  onRouteChange();
}

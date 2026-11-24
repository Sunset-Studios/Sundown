import { BrowserWindow } from "electron";

export function create_app_window(url, options, web_preferences) {
  const window = new BrowserWindow({
    title: "Sundown Engine",
    ...options,
    webPreferences: {
      ...web_preferences,
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  const source = new URL(url);
  function block_external_navigation(event) {
    const target = new URL(event.url);
    if (target.protocol !== source.protocol || target.host !== source.host) {
      event.preventDefault();
    }
  }
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", block_external_navigation);
  window.webContents.on("will-redirect", block_external_navigation);
  return window;
}

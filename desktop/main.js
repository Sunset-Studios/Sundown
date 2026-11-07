import { app, BrowserWindow, dialog, net, protocol } from "electron";
import path from "node:path";
import { serve_asset } from "./protocol.js";

const development = !app.isPackaged && process.argv.includes("--development");
const app_url = development ? "http://localhost:3000/" : "sundown://app/";
let main_window = null;

// Preserve browser security boundaries while supporting modules, fetch and workers.
protocol.registerSchemesAsPrivileged([
  {
    scheme: "sundown",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
      codeCache: true,
    },
  },
]);

async function create_window() {
  main_window = new BrowserWindow({
    title: "Sundown Engine",
    width: 1280,
    height: 720,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  main_window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  main_window.webContents.on("will-navigate", (event, target_url) => {
    const target = new URL(target_url);
    const source = new URL(app_url);
    if (target.protocol !== source.protocol || target.host !== source.host) {
      event.preventDefault();
    }
  });
  main_window.on("closed", () => {
    main_window = null;
  });
  await main_window.loadURL(app_url);
}

app
  .whenReady()
  .then(async () => {
    if (!development) {
      const asset_root = path.join(
        app.isPackaged ? process.resourcesPath : app.getAppPath(),
        "dist"
      );
      protocol.handle("sundown", (request) => serve_asset(request, asset_root, net.fetch));
    }

    await create_window();
    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        create_window().catch(report_startup_error);
      }
    });
  })
  .catch(report_startup_error);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

function report_startup_error(startup_error) {
  console.error("Failed to start Sundown:", startup_error);
  dialog.showErrorBox("Sundown failed to start", startup_error.message);
  app.exit(1);
}

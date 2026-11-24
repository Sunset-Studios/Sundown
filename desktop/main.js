import { app, BrowserWindow, dialog, ipcMain, net, protocol } from "electron";
import path from "node:path";
import { tray_config } from "./config.js";
import { serve_asset } from "./protocol.js";
import { SmokeSequenceSession, SmokeSession } from "./smoke.js";
import { start_tray } from "./tray.js";
import { create_app_window } from "./window.js";

const development = !app.isPackaged && process.argv.includes("--development");
const smoke = process.argv.includes("--smoke");
const tray_mode = !smoke && (tray_config.enabled || process.argv.includes("--tray"));
const smoke_options = smoke ? JSON.parse(process.env.SUNDOWN_SMOKE_OPTIONS || "{}") : null;
const app_url = smoke
  ? smoke_options.url || "sundown://app/?smoke=1"
  : development
    ? "http://localhost:3000/"
    : "sundown://app/";
let main_window = null;
let smoke_session = null;

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
  const window_preferences = smoke
    ? {
        preload: path.join(app.getAppPath(), "desktop/smoke_preload.cjs"),
        backgroundThrottling: false,
      }
    : {};
  main_window = create_app_window(app_url, { width: 1280, height: 720 }, window_preferences);
  main_window.on("closed", () => {
    main_window = null;
  });

  if (smoke) {
    const session_options = {
      url: app_url,
      output: smoke_options.output || path.resolve("diagnostics.json"),
      timeout_ms: smoke_options.timeout_ms || 60000,
      frames: smoke_options.frames || 5,
    };
    smoke_session = smoke_options.scene_switching
      ? new SmokeSequenceSession(app, main_window, ipcMain, {
          ...session_options,
          scenes: smoke_options.scenes,
        })
      : new SmokeSession(app, main_window, ipcMain, session_options);
  }
  
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

    if (tray_mode) {
      start_tray(app_url, tray_config);
      return;
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
  if (!tray_mode && process.platform !== "darwin") app.quit();
});

function report_startup_error(startup_error) {
  if (smoke) {
    smoke_session.fail("startup", startup_error.stack || startup_error.message);
    return;
  }
  console.error("Failed to start Sundown:", startup_error);
  dialog.showErrorBox("Sundown failed to start", startup_error.message);
  app.exit(1);
}

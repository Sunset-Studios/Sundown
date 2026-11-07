import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import electron from "electron";

const project_root = fileURLToPath(new URL("../", import.meta.url));
const server = spawn(process.execPath, ["tools/dev_server.js"], {
  cwd: project_root,
  env: { ...process.env, PORT: "3000" },
  stdio: ["ignore", "inherit", "inherit", "ipc"],
});
let desktop = null;
let stopping = false;

function stop(exit_code = 0) {
  if (stopping) return;
  stopping = true;
  process.exitCode = exit_code;
  desktop?.kill();
  server.kill();
}

server.once("message", (message) => {
  if (message.type !== "listening" || stopping) return;
  const environment = { ...process.env };
  delete environment.ELECTRON_RUN_AS_NODE;
  desktop = spawn(electron, [".", "--development"], {
    cwd: project_root,
    env: environment,
    stdio: "inherit",
  });
  desktop.on("error", (launch_error) => {
    console.error("Failed to launch Electron:", launch_error);
    stop(1);
  });
  desktop.on("exit", (code) => stop(code ?? 1));
});
server.on("error", (server_error) => {
  console.error("Failed to launch dev server:", server_error);
  stop(1);
});
server.on("exit", (code) => {
  if (!stopping) stop(code || 1);
});
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());

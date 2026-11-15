// ═══════════════════════════════════════════════════════════════════════════════
// Electron main/preload/IPC integration — no build or WebGPU device required.
// Protect the configuration boundary that mocked webContents tests bypass.
// ═══════════════════════════════════════════════════════════════════════════════
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import electron from "electron";

const project_root = fileURLToPath(new URL("../../", import.meta.url));
const fixture_script = `
const parameters = new URLSearchParams(location.search);
if (parameters.get("mode") === "diagnostics" || (parameters.get("mode") === "suite" && parameters.get("scene") === "BVHScene")) {
  function warn() { console.warn("fixture warning"); }
  warn();
  warn();
  window.sundown_smoke.send({ type: "diagnostic", diagnostic: {
    severity: "error", category: "fixture-validation", message: "fixture GPU failure",
    source: "fixture.wgsl", line: 17, column: 9, stack: "fixture stack"
  } });
}
if (parameters.get("mode") !== "stall") {
  window.sundown_smoke.send({ type: "complete", readiness: {
    assets_ready: true, completed_frames: Number(parameters.get("completed_frames"))
  } });
}
`;

async function start_fixture_server(context) {
  const server = createServer((request, response) => {
    response.setHeader("Content-Security-Policy", "default-src 'self'; script-src 'self'");
    if (request.url === "/fixture.js") {
      response.setHeader("Content-Type", "text/javascript");
      response.end(fixture_script);
    } else if (request.url === "/favicon.ico") {
      response.writeHead(204).end();
    } else {
      response.setHeader("Content-Type", "text/html");
      response.end(
        '<!doctype html><title>Smoke transport fixture</title><script src="/fixture.js"></script>'
      );
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(
    () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(resolve);
      })
  );
  return `http://127.0.0.1:${server.address().port}/`;
}

async function run_fixture(context, mode, completed_frames = 7, timeout_ms = 5000) {
  const directory = mkdtempSync(path.join(tmpdir(), "sundown-electron-test-"));
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  const default_output = path.join(directory, "diagnostics.json");
  const sentinel = "The suite report must not be overwritten by a scene.";
  writeFileSync(default_output, sentinel);
  const output = path.join(directory, "custom reports", "scene.json");
  const origin = await start_fixture_server(context);
  const url = `${origin}?smoke=1&mode=${mode}&completed_frames=${completed_frames}`;
  const environment = {
    ...process.env,
    SUNDOWN_SMOKE_OPTIONS: JSON.stringify({ url, output, frames: 7, timeout_ms }),
  };
  delete environment.ELECTRON_RUN_AS_NODE;
  const child = spawn(electron, [project_root, "--smoke"], {
    cwd: directory,
    env: environment,
    stdio: ["ignore", "pipe", "pipe"],
  });
  context.after(() => {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  });
  let logs = "";
  child.stdout.on("data", (chunk) => {
    logs += chunk;
  });
  child.stderr.on("data", (chunk) => {
    logs += chunk;
  });
  const result = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("close", (code, signal) => resolve({ code, signal }));
  });
  assert.ok(existsSync(output), `Missing custom report ${output}\n${logs}`);
  assert.equal(
    readFileSync(default_output, "utf8"),
    sentinel,
    "Scene overwrote the default suite path"
  );
  const report = JSON.parse(readFileSync(output, "utf8"));
  assert.equal(report.scene, url);
  return { report, result, logs };
}

test(
  "Electron main honors a custom report path and completes via the preload bridge",
  { timeout: 15000 },
  async (context) => {
    const { report, result, logs } = await run_fixture(context, "quiet");
    assert.equal(result.code, 0, logs);
    assert.equal(report.status, "passed");
    assert.deepEqual(report.readiness, { assets_ready: true, completed_frames: 7 });
    assert.deepEqual(report.diagnostics, []);
  }
);

test(
  "Electron persists real console and structured diagnostics to the requested path",
  { timeout: 15000 },
  async (context) => {
    const { report, result, logs } = await run_fixture(context, "diagnostics");
    assert.equal(result.code, 1, logs);
    const warning = report.diagnostics.find((entry) => entry.message === "fixture warning");
    assert.equal(warning.count, 2);
    assert.ok(warning.source.endsWith("/fixture.js"));
    assert.ok(warning.line > 0);
    const error = report.diagnostics.find((entry) => entry.category === "fixture-validation");
    assert.equal(error.message, "fixture GPU failure");
    assert.equal(error.source, "fixture.wgsl");
    assert.equal(error.line, 17);
    assert.equal(error.column, 9);
    assert.equal(error.stack, "fixture stack");
  }
);

test(
  "Electron enforces the configured frame count rather than its default",
  { timeout: 15000 },
  async (context) => {
    const { report, result } = await run_fixture(context, "quiet", 5);
    assert.equal(result.code, 1);
    assert.equal(report.diagnostics[0].category, "readiness");
  }
);

test(
  "Electron enforces the configured timeout rather than its default",
  { timeout: 15000 },
  async (context) => {
    const { report, result } = await run_fixture(context, "stall", 7, 120);
    assert.equal(result.code, 1);
    assert.equal(report.diagnostics[0].category, "timeout");
    assert.match(report.diagnostics[0].message, /120ms/);
  }
);

test(
  "CLI saves only one report, embeds all scene diagnostics and cleans transport files",
  { timeout: 20000 },
  async (context) => {
    const directory = mkdtempSync(path.join(tmpdir(), "sundown-single-report-test-"));
    context.after(() => rmSync(directory, { recursive: true, force: true }));
    const output_directory = path.join(directory, "reports");
    const transport_root = path.join(directory, "transport");
    mkdirSync(transport_root);
    const output = path.join(output_directory, "suite.json");
    const origin = await start_fixture_server(context);
    const child = spawn(
      process.execPath,
      [
        path.join(project_root, "tools/smoke.js"),
        "--no-build",
        "--scene",
        "SolarECSTestScene,BVHScene",
        "--frames",
        "7",
        "--timeout",
        "5000",
        "--url",
        `${origin}?mode=suite&completed_frames=7`,
        "--output",
        output,
      ],
      {
        cwd: directory,
        env: { ...process.env, TMPDIR: transport_root, TMP: transport_root, TEMP: transport_root },
        stdio: ["ignore", "pipe", "pipe"],
      }
    );
    context.after(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    });
    let logs = "";
    child.stdout.on("data", (chunk) => {
      logs += chunk;
    });
    child.stderr.on("data", (chunk) => {
      logs += chunk;
    });
    const code = await new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("close", resolve);
    });
    assert.equal(code, 1, logs);
    const report = JSON.parse(readFileSync(output, "utf8"));
    assert.equal(report.schema_version, 2);
    assert.equal(report.status, "failed");
    assert.deepEqual(
      report.scenes.map((scene) => scene.sample),
      ["SolarECSTestScene", "BVHScene"]
    );
    assert.equal(report.scenes[0].status, "passed");
    assert.equal(report.scenes[1].status, "failed");
    assert.ok(
      report.scenes[1].diagnostics.some((entry) => entry.message === "fixture GPU failure")
    );
    assert.deepEqual(readdirSync(output_directory), ["suite.json"]);
    assert.deepEqual(
      readdirSync(transport_root).filter((name) => name.startsWith("sundown-smoke-")),
      []
    );
  }
);

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import electron from "electron";
import { SmokeSceneReport } from "./smoke/scene_report.js";
import { SmokeSuiteReport } from "./smoke/suite_report.js";
import { finalize_scene_launch, finalize_sequence_scene } from "./smoke/launch_report.js";
import { load_scene_catalog, select_scenes } from "./smoke/catalog.js";

const project_root = fileURLToPath(new URL("../", import.meta.url));
let child = null;
let stopping = false;

function write_report(output, report) {
  mkdirSync(path.dirname(output), { recursive: true });
  writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
}

function run(command, args, environment = process.env, capture_output = false) {
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let stdout_truncated = false;
    let stderr_truncated = false;
    const output_limit = 32768;
    child = spawn(command, args, {
      cwd: project_root,
      env: environment,
      stdio: capture_output ? ["ignore", "pipe", "pipe"] : "inherit",
    });
    if (capture_output) {
      child.stdout.setEncoding("utf8");
      child.stderr.setEncoding("utf8");
      child.stdout.on("data", (chunk) => {
        process.stdout.write(chunk);
        stdout_truncated ||= stdout.length + chunk.length > output_limit;
        stdout = (stdout + chunk).slice(-output_limit);
      });
      child.stderr.on("data", (chunk) => {
        process.stderr.write(chunk);
        stderr_truncated ||= stderr.length + chunk.length > output_limit;
        stderr = (stderr + chunk).slice(-output_limit);
      });
    }
    child.once("error", reject);
    // 'close' follows stream draining; 'exit' can lose the final stderr messages.
    child.once("close", (code, signal) =>
      resolve({ code, signal, stdout, stderr, stdout_truncated, stderr_truncated })
    );
  });
}

async function run_sample(sample, values, frames, timeout_ms, output) {
  const url = new URL(values.url || "sundown://app/");
  url.searchParams.set("smoke", "1");
  url.searchParams.set("scene", sample.id);
  url.searchParams.set("frames", String(frames));
  url.searchParams.set("timeout", String(timeout_ms));
  const report = new SmokeSceneReport("electron", url.href);
  const environment = { ...process.env };
  delete environment.ELECTRON_RUN_AS_NODE;
  environment.SUNDOWN_SMOKE_OPTIONS = JSON.stringify({ url: url.href, frames, timeout_ms, output });
  write_report(output, report);
  console.log(`Smoke: ${sample.id}`);
  let timed_out = false;
  let process_result = null;
  let saved = null;
  let launch_error = null;
  const watchdog = setTimeout(() => {
    timed_out = true;
    child?.kill("SIGKILL");
  }, timeout_ms + 5000);
  try {
    process_result = await run(electron, [".", "--smoke"], environment, true);
    saved = JSON.parse(readFileSync(output, "utf8"));
  } catch (error) {
    launch_error = error;
  } finally {
    clearTimeout(watchdog);
    child = null;
  }
  const result = finalize_scene_launch(report, saved, process_result, {
    stopping,
    timed_out,
    error: launch_error,
    output,
    frames,
  });
  result.sample = sample.id;
  return result;
}

async function run_sequence(selected, values, frames, timeout_ms, output) {
  const url = new URL(values.url || "sundown://app/");
  url.searchParams.set("smoke", "1");
  url.searchParams.set("switch", "1");
  url.searchParams.delete("scene");
  for (const sample of selected) url.searchParams.append("scene", sample.id);
  url.searchParams.set("frames", String(frames));
  url.searchParams.set("timeout", String(timeout_ms));

  const scene_ids = selected.map((sample) => sample.id);
  const environment = { ...process.env };
  delete environment.ELECTRON_RUN_AS_NODE;
  environment.SUNDOWN_SMOKE_OPTIONS = JSON.stringify({
    url: url.href,
    frames,
    timeout_ms,
    output,
    scene_switching: true,
    scenes: scene_ids,
  });

  console.log(`Smoke sequence: ${scene_ids.join(", ")}`);
  const watchdog_ms = selected.length * (timeout_ms + 5000) + 10000;
  let timed_out = false;
  let process_result = null;
  let launch_error = null;
  const watchdog = setTimeout(() => {
    timed_out = true;
    child?.kill("SIGKILL");
  }, watchdog_ms);
  try {
    process_result = await run(electron, [".", "--smoke"], environment, true);
  } catch (error) {
    launch_error = error;
  } finally {
    clearTimeout(watchdog);
    child = null;
  }

  let checkpoint = null;
  let checkpoint_error = null;
  try {
    checkpoint = JSON.parse(readFileSync(output, "utf8"));
  } catch (error) {
    // A missing checkpoint is reported as the missing scene result it is.
    if (error.code !== "ENOENT") checkpoint_error = error;
  }

  const saved_scenes = Array.isArray(checkpoint?.scenes) ? checkpoint.scenes : [];
  const scenes = selected.map((sample, index) => {
    const result = finalize_sequence_scene(
      new SmokeSceneReport("electron", sample.id),
      saved_scenes[index],
      {
        stopping,
        timed_out,
        aborted: checkpoint?.aborted === true,
        error: launch_error ?? checkpoint_error,
        output,
        frames,
        process: process_result,
      }
    );
    result.sample = sample.id;
    return result;
  });

  const diagnostics = [];
  if (Array.isArray(checkpoint?.suite_diagnostics))
    for (const diagnostic of checkpoint.suite_diagnostics) diagnostics.push(diagnostic);
  if (
    checkpoint?.schema_version !== 1 ||
    JSON.stringify(checkpoint?.selected_scenes) !== JSON.stringify(scene_ids)
  ) {
    diagnostics.push({
      severity: "error",
      category: "launcher",
      message: "Electron did not receive the requested scene sequence",
    });
  }
  if (timed_out) {
    diagnostics.push({
      severity: "error",
      category: "timeout",
      message: `Smoke sequence exceeded its ${watchdog_ms}ms deadline`,
    });
  } else if (launch_error) {
    diagnostics.push({
      severity: "error",
      category: "launcher",
      message: launch_error.stack || launch_error.message,
    });
  }
  if (checkpoint?.aborted) {
    diagnostics.push({
      severity: "error",
      category: "aborted",
      message: "Smoke sequence was aborted before every selected scene completed",
    });
  }
  if (checkpoint && !checkpoint.finished && !checkpoint.aborted && !timed_out) {
    diagnostics.push({
      severity: "error",
      category: "launcher",
      message: "Electron exited before completing the scene sequence",
    });
  }
  if (
    !timed_out &&
    !launch_error &&
    process_result?.code !== 0 &&
    scenes.every((scene) => scene.status === "passed")
  ) {
    diagnostics.push({
      severity: "error",
      category: "launcher",
      message: `Electron exited with code ${process_result.code} after all scenes passed`,
    });
  }

  return { scenes, diagnostics };
}

async function main() {
  const { values } = parseArgs({
    options: {
      "no-build": { type: "boolean", default: false },
      "scene-switching": { type: "boolean", default: false },
      catalog: { type: "string" },
      scene: { type: "string", multiple: true },
      list: { type: "boolean", default: false },
      url: { type: "string" },
      frames: { type: "string", default: "5" },
      timeout: { type: "string", default: "60000" },
      output: { type: "string", default: "diagnostics.json" },
    },
  });
  const scenes = await load_scene_catalog(values.catalog, values.scene);
  if (values.list) {
    console.log(scenes.map((scene) => scene.id).join("\n"));
    return;
  }
  const frames = Number(values.frames);
  const timeout_ms = Number(values.timeout);
  if (
    !Number.isSafeInteger(frames) ||
    frames < 1 ||
    !Number.isSafeInteger(timeout_ms) ||
    timeout_ms < 1
  ) {
    throw new Error("--frames and --timeout must be positive integers");
  }
  const selected = select_scenes(scenes, values.scene);
  const output = path.resolve(values.output);
  let transport_directory = null;
  const suite = new SmokeSuiteReport(selected);
  function interrupt(signal) {
    stopping = true;
    child?.kill("SIGKILL");
    suite.add({
      severity: "error",
      category: "interrupted",
      message: `Smoke interrupted by ${signal}`,
    });
    suite.finish();
    write_report(output, suite);
  }
  process.on("SIGINT", interrupt);
  process.on("SIGTERM", interrupt);
  try {
    // Replace stale success before cooking. Isolated mode checkpoints after each scene.
    write_report(output, suite);
    if (!values["no-build"]) {
      const result = await run(process.platform === "win32" ? "npm.cmd" : "npm", ["run", "build"], {
        ...process.env,
        SUNDOWN_SMOKE_BUILD: "1",
      });
      if (result.code !== 0) {
        suite.add({
          severity: "error",
          category: "build",
          message: `Build failed (${result.signal || result.code})`,
        });
        stopping = true;
      }
    }
    if (!stopping) transport_directory = mkdtempSync(path.join(tmpdir(), "sundown-smoke-"));
    if (!stopping && values["scene-switching"]) {
      const sequence = await run_sequence(
        selected,
        values,
        frames,
        timeout_ms,
        path.join(transport_directory, "sequence.json")
      );
      suite.scenes.push(...sequence.scenes);
      for (const diagnostic of sequence.diagnostics) suite.add(diagnostic);
      write_report(output, suite);
    } else {
      for (const sample of selected) {
        if (stopping) break;
        suite.scenes.push(
          await run_sample(
            sample,
            values,
            frames,
            timeout_ms,
            path.join(transport_directory, "scene.json")
          )
        );
        write_report(output, suite);
      }
    }
  } catch (error) {
    suite.add({ severity: "error", category: "launcher", message: error.stack || error.message });
  } finally {
    process.removeListener("SIGINT", interrupt);
    process.removeListener("SIGTERM", interrupt);
    try {
      process.exitCode = suite.finish();
      write_report(output, suite);
      const passed = suite.scenes.filter((scene) => scene.status === "passed").length;
      console.log(
        `Smoke ${suite.status}: ${passed}/${selected.length} scenes passed. Report: ${output}`
      );
    } finally {
      // Only remove the private directory allocated by this run, never old outputs.
      if (transport_directory) rmSync(transport_directory, { recursive: true, force: true });
    }
  }
}

void main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});

// ═══════════════════════════════════════════════════════════════════════════════
// Engine diagnostics — opt-in runtime collection with launcher-neutral reports.
// Console hooks run only when enabled; GPU failures remain visible otherwise.
// ═══════════════════════════════════════════════════════════════════════════════

import { DiagnosticsReport } from "../utility/diagnostics_report.js";

const parameters = typeof location === "undefined" ? null : new URLSearchParams(location.search);
let enabled = false;
let report = null;
const pending_compilations = new Set();
const pending_work = new Set();
const subscribers = new Set();

export function begin_diagnostic_work(source) {
  if (!enabled) return null;
  const work = { source };
  pending_work.add(work);
  return work;
}

export function end_diagnostic_work(work) {
  if (work) pending_work.delete(work);
}

export function has_pending_diagnostic_work() {
  return pending_work.size > 0;
}

export function subscribe_diagnostics(listener) {
  subscribers.add(listener);
  return () => subscribers.delete(listener);
}

export function get_diagnostics_report() {
  return report;
}

export function report_diagnostic(diagnostic) {
  if (report) {
    report.add(diagnostic);
    for (const listener of subscribers) listener(diagnostic);
  } else {
    const output = diagnostic.severity === "warning" ? console.warn : console.error;
    output(`${diagnostic.category}: ${diagnostic.message} (${diagnostic.source ?? ""})`);
  }
}

export function observe_gpu_device(device) {
  device.addEventListener("uncapturederror", (event) => {
    report_diagnostic({
      severity: "error",
      category: "webgpu-validation",
      message: event.error.message,
      stack: event.error.stack,
    });
  });
  device.lost.then((info) => {
    report_diagnostic({
      severity: "error",
      category: "webgpu-device-lost",
      message: `${info.reason}: ${info.message}`,
    });
  });
}

export function observe_shader_compilation(module, source, variant) {
  if (!enabled) return;
  const compilation = module
    .getCompilationInfo()
    .then((info) => {
      for (const message of info.messages) {
        if (message.type === "info") continue;
        report_diagnostic({
          severity: message.type === "warning" ? "warning" : "error",
          category: "wgsl-compilation",
          message: `${message.message} [${variant}]`,
          source,
          // Coordinates refer to cooked WGSL, not the original include files.
          line: message.lineNum,
          column: message.linePos,
        });
      }
    })
    .catch((error) => {
      report_diagnostic({
        severity: "error",
        category: "wgsl-compilation",
        message: error.message,
        source,
      });
    })
    .finally(() => pending_compilations.delete(compilation));
  pending_compilations.add(compilation);
}

export async function wait_for_diagnostics() {
  while (pending_compilations.size) await Promise.all(pending_compilations);
}

export function install_diagnostics(options = {}) {
  if (enabled) return report;
  enabled = true;
  report = new DiagnosticsReport(
    options.launcher ?? "browser",
    options.scene ?? globalThis.location?.href ?? ""
  );
  // Browser drivers can inspect this without an Electron bridge.
  globalThis.sundown_diagnostics = report;
  for (const severity of ["warn", "error"]) {
    const original = console[severity].bind(console);
    console[severity] = (...args) => {
      const stack = new Error().stack ?? "";
      const caller = stack.split("\n")[2] ?? "";
      const source_location = /((?:https?|file|sundown):\/\/.*?):(\d+):(\d+)\)?$/.exec(caller);
      report_diagnostic({
        severity: severity === "warn" ? "warning" : "error",
        category: "console",
        message: args
          .map((value) => (value instanceof Error ? value.stack : String(value)))
          .join(" "),
        source: source_location?.[1],
        line: source_location?.[2],
        column: source_location?.[3],
        stack,
      });
      original(...args);
    };
  }
  globalThis.addEventListener(
    "error",
    (event) => {
      report_diagnostic({
        severity: "error",
        category: "uncaught-exception",
        message:
          event.message ||
          `Failed to load ${event.target?.src || event.target?.href || "resource"}`,
        source: event.filename,
        line: event.lineno,
        column: event.colno,
        stack: event.error?.stack,
      });
    },
    true
  );
  globalThis.addEventListener("unhandledrejection", (event) => {
    report_diagnostic({
      severity: "error",
      category: "unhandled-rejection",
      message: event.reason?.message ?? String(event.reason),
      stack: event.reason?.stack,
    });
  });
  return report;
}

if (parameters?.has("diagnostics")) install_diagnostics();

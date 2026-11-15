// ═══════════════════════════════════════════════════════════════════════════════
// Launcher report reconciliation — preserve renderer evidence when a report is
// incomplete or a process exits unexpectedly. Output tails are bounded upstream.
// ═══════════════════════════════════════════════════════════════════════════════
export function finalize_scene_launch(report, saved, process_result, options) {
  let failure = null;
  let category = "launcher";
  if (options.stopping) {
    failure = "Run interrupted";
  } else if (options.timed_out) {
    category = "timeout";
    failure = "Electron exceeded the smoke deadline";
  } else if (options.error) {
    failure = options.error.stack || options.error.message;
  } else if (saved?.schema_version !== 1 || !Array.isArray(saved?.diagnostics)) {
    failure = "Electron did not write a scene report";
  } else if (saved.scene !== report.scene) {
    failure = `Electron wrote a report for a different scene: ${saved.scene}`;
  } else if (!["passed", "failed"].includes(saved.status) || !saved.finished_at) {
    failure = `Electron exited before finalizing its scene report (status: ${saved.status})`;
  } else if (saved.status === "passed" && process_result?.code !== 0) {
    failure = "Electron reported success but exited unsuccessfully";
  } else if (
    saved.status === "passed" &&
    (saved.diagnostics.length ||
      saved.readiness?.assets_ready !== true ||
      !Number.isSafeInteger(saved.readiness.completed_frames) ||
      saved.readiness.completed_frames < options.frames)
  ) {
    failure = "Electron reported success without clean diagnostics and the required GPU frames";
  } else if (saved.status === "failed" && !saved.diagnostics.length) {
    failure = "Electron reported failure without diagnostics";
  }

  const process_info = {
    exit_code: process_result?.code ?? null,
    signal: process_result?.signal ?? null,
    stdout: process_result?.stdout ?? "",
    stderr: process_result?.stderr ?? "",
    stdout_truncated: process_result?.stdout_truncated ?? false,
    stderr_truncated: process_result?.stderr_truncated ?? false,
  };
  if (!failure) return { ...saved, process: process_info };

  report.add({
    severity: "error",
    category,
    message: `${failure}. Expected report: ${options.output}. Exit code: ${process_info.exit_code}; signal: ${process_info.signal ?? "none"}.`,
    source: options.output,
  });
  report.finish(saved?.readiness ?? null);
  const result = report.toJSON();
  // A partial report may already contain the actual cause, source and repeats.
  // Do not replace that evidence with the secondary launcher failure.
  if (Array.isArray(saved?.diagnostics))
    result.diagnostics = saved.diagnostics.concat(result.diagnostics);
  if (typeof saved?.started_at === "string") result.started_at = saved.started_at;
  result.process = process_info;
  return result;
}

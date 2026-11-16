// ═══════════════════════════════════════════════════════════════════════════════
// Launcher report reconciliation — preserve renderer evidence when a report is
// incomplete or a process exits unexpectedly. Output tails are bounded upstream.
// ═══════════════════════════════════════════════════════════════════════════════

export function describe_process(process_result) {
  return {
    exit_code: process_result?.code ?? null,
    signal: process_result?.signal ?? null,
    stdout: process_result?.stdout ?? "",
    stderr: process_result?.stderr ?? "",
    stdout_truncated: process_result?.stdout_truncated ?? false,
    stderr_truncated: process_result?.stderr_truncated ?? false,
  };
}

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

  const process_info = describe_process(process_result);
  if (!failure) return { ...saved, process: process_info };

  return finalize_launch_failure(
    report,
    saved,
    process_info,
    category,
    `${failure}. Expected report: ${options.output}`,
    options.output
  );
}

export function finalize_sequence_scene(report, saved, options) {
  let failure = null;
  let category = "launcher";
  const terminal =
    saved?.schema_version === 1 &&
    Array.isArray(saved?.diagnostics) &&
    ["passed", "failed"].includes(saved.status) &&
    Boolean(saved.finished_at);

  if (!terminal) {
    if (options.stopping) {
      failure = "Run interrupted";
    } else if (options.timed_out) {
      category = "timeout";
      failure = "Smoke sequence exceeded the suite deadline before this scene finished";
    } else if (options.aborted) {
      category = "aborted";
      failure = "Smoke sequence was aborted before this scene finished";
    } else if (options.error) {
      failure = options.error.stack || options.error.message;
    } else {
      failure = "Electron did not write a scene result";
    }
    if (Array.isArray(saved?.diagnostics)) failure += ` (status: ${saved.status ?? "missing"})`;
  } else if (saved.scene !== report.scene) {
    failure = `Electron wrote a result for a different scene: ${saved.scene}`;
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

  const process_info = describe_process(options.process);
  if (!failure) return { ...saved, process: process_info };

  return finalize_launch_failure(
    report,
    saved,
    process_info,
    category,
    `${failure}. Expected result: ${options.output}`,
    options.output
  );
}

function finalize_launch_failure(report, saved, process_info, category, message, output) {
  report.add({
    severity: "error",
    category,
    message: `${message}. Exit code: ${process_info.exit_code}; signal: ${process_info.signal ?? "none"}.`,
    source: output,
  });
  report.finish(saved?.readiness ?? null);
  const result = report.toJSON();
  // Preserve the original cause, source and repeats ahead of the launcher failure.
  if (Array.isArray(saved?.diagnostics))
    result.diagnostics = saved.diagnostics.concat(result.diagnostics);
  if (typeof saved?.started_at === "string") result.started_at = saved.started_at;
  result.process = process_info;
  return result;
}

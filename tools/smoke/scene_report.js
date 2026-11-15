import { DiagnosticsReport } from "../../engine/src/utility/diagnostics_report.js";

// Readiness and exit-code policy belong to the harness, not diagnostic collection.
export class SmokeSceneReport extends DiagnosticsReport {
  readiness = null;

  finish(readiness = null) {
    super.finish();
    this.readiness = readiness;
    if (
      readiness?.assets_ready !== true ||
      !Number.isSafeInteger(readiness.completed_frames) ||
      readiness.completed_frames < 1
    ) {
      this.status = "failed";
    }
    return this.status === "passed" ? 0 : 1;
  }

  toJSON() {
    return { ...super.toJSON(), readiness: this.readiness };
  }
}

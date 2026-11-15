import { DiagnosticsReport } from "../../engine/src/utility/diagnostics_report.js";

export class SmokeSuiteReport extends DiagnosticsReport {
  constructor(samples) {
    super("electron", "samples");
    this.schema_version = 2;
    this.selected_scenes = samples.map((sample) => sample.id);
    this.scenes = [];
  }

  finish() {
    super.finish();
    this.status =
      !this.diagnostics.length &&
      this.selected_scenes.length > 0 &&
      this.scenes.length === this.selected_scenes.length &&
      this.scenes.every(
        (scene, index) => scene.sample === this.selected_scenes[index] && scene.status === "passed"
      )
        ? "passed"
        : "failed";
    return this.status === "passed" ? 0 : 1;
  }

  toJSON() {
    return {
      ...super.toJSON(),
      selected_scenes: this.selected_scenes,
      scenes: this.scenes,
    };
  }
}

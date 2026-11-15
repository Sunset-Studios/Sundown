// Shared report contract: no Electron, Node, or DOM dependencies.
export class DiagnosticsReport {
  constructor(launcher, scene) {
    this.schema_version = 1;
    this.launcher = launcher;
    this.scene = scene;
    this.started_at = new Date().toISOString();
    this.finished_at = null;
    this.status = "running";
    this.diagnostics = [];
    this.entries = new Map();
  }

  add(diagnostic) {
    if (this.finished_at) this.status = "failed";
    const entry = {
      severity: diagnostic.severity === "warning" ? "warning" : "error",
      category: String(diagnostic.category ?? "runtime"),
      message: String(diagnostic.message),
      source: String(diagnostic.source ?? ""),
      line: Number(diagnostic.line) || 0,
      column: Number(diagnostic.column) || 0,
      stack: String(diagnostic.stack ?? ""),
      count: 1,
    };
    const key = JSON.stringify([
      entry.severity,
      entry.category,
      entry.message,
      entry.source,
      entry.line,
      entry.column,
    ]);
    const previous = this.entries.get(key);
    if (previous) {
      previous.count++;
    } else {
      this.entries.set(key, entry);
      this.diagnostics.push(entry);
    }
  }

  finish() {
    this.finished_at = new Date().toISOString();
    this.status = this.diagnostics.length ? "failed" : "passed";
    return this.status === "passed" ? 0 : 1;
  }

  toJSON() {
    return {
      schema_version: this.schema_version,
      launcher: this.launcher,
      scene: this.scene,
      started_at: this.started_at,
      finished_at: this.finished_at,
      status: this.status,
      diagnostics: this.diagnostics,
    };
  }
}

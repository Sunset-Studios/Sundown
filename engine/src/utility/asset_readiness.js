// ═══════════════════════════════════════════════════════════════════════════════
// Diagnostic-mode asset readiness — reusable by external startup tooling.
// GPU completion is checked separately by the renderer's post-render lifecycle.
// ═══════════════════════════════════════════════════════════════════════════════

import { has_pending_diagnostic_work } from "../core/diagnostics.js";
import { ResourceCache } from "../renderer/resource_cache.js";
import { CacheTypes } from "../renderer/renderer_types.js";
import { StreamingSystem } from "../streaming/streaming_system.js";

let meshes_ready = false;
let pipelines_ready = false;

function inspect_mesh(mesh) {
  if (
    mesh.pending_loader ||
    mesh.pending_runtime_meshlet_build ||
    mesh.pending_gltf_build_promise
  ) {
    meshes_ready = false;
  }
}

function inspect_pipeline(pipeline) {
  if (pipeline.pending) pipelines_ready = false;
}

export function are_engine_assets_ready() {
  if (has_pending_diagnostic_work() || StreamingSystem.get().requests.size > 0) return false;
  // Reuse the visitor instead of allocating iterators or temporary arrays per frame.
  meshes_ready = true;
  ResourceCache.get().fetch_all(CacheTypes.MESH).forEach(inspect_mesh);
  pipelines_ready = true;
  ResourceCache.get().fetch_all(CacheTypes.PIPELINE_STATE).forEach(inspect_pipeline);
  return meshes_ready && pipelines_ready;
}

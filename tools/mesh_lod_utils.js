import { MeshoptSimplifier } from "meshoptimizer/simplifier";

export const DEFAULT_MESH_LOD_RATIOS = Object.freeze([1.0, 0.5, 0.25, 0.125]);
export const DEFAULT_MESH_LOD_MAX_ERROR = 0.01;

function clamp_lod(lod, lod_count) {
  const numeric_lod = Number.isFinite(Number(lod)) ? Math.floor(Number(lod)) : 0;
  return Math.max(0, Math.min(numeric_lod, Math.max(0, lod_count - 1)));
}

function normalize_ratios(ratios) {
  const source = Array.isArray(ratios) && ratios.length > 0 ? ratios : DEFAULT_MESH_LOD_RATIOS;
  const normalized = [1.0];

  for (const value of source) {
    const ratio = Number(value);
    if (!Number.isFinite(ratio) || ratio <= 0.0 || ratio > 1.0 || ratio === 1.0) {
      continue;
    }
    if (!normalized.some((existing) => Math.abs(existing - ratio) <= Number.EPSILON)) {
      normalized.push(ratio);
    }
  }

  normalized.sort((a, b) => b - a);
  return normalized;
}

/**
 * Resolves project-wide glTF defaults and per-mesh overrides.
 *
 * glTF root or mesh extras may contain:
 * extras.sundown.meshLod = {
 *   ratios: [1.0, 0.5, 0.25, 0.125],
 *   maxError: 0.01,
 *   lockBorder: true,
 *   minLod: 0,
 *   sbvhLod: 0
 * }
 * LOD 0 is full resolution; increasing LOD indices select progressively coarser meshes.
 * lockBorder defaults to false because imported UV seams commonly duplicate border vertices.
 */
export function resolve_mesh_lod_settings(document_json, mesh) {
  const defaults = document_json?.extras?.sundown?.meshLod ?? {};
  const overrides = mesh?.extras?.sundown?.meshLod ?? {};
  const config = { ...defaults, ...overrides };
  const ratios = normalize_ratios(config.ratios);
  const max_error = Number(config.maxError);

  return {
    ratios,
    max_error:
      Number.isFinite(max_error) && max_error >= 0.0 ? max_error : DEFAULT_MESH_LOD_MAX_ERROR,
    lock_border: config.lockBorder === true,
    has_min_lod: config.minLod !== undefined,
    min_lod: clamp_lod(config.minLod, ratios.length),
    sbvh_lod: clamp_lod(config.sbvhLod, ratios.length),
  };
}

export async function initialize_mesh_lod_simplifier() {
  if (!MeshoptSimplifier.supported) {
    throw new Error("meshoptimizer simplifier is not supported in this Node.js runtime.");
  }
  await MeshoptSimplifier.ready;
}

export function build_mesh_lod_indices(indices, positions, settings) {
  const lods = [];
  const source_triangle_count = Math.floor(indices.length / 3);
  if (source_triangle_count <= 0) {
    return lods;
  }

  for (let lod = 0; lod < settings.ratios.length; lod++) {
    const target_ratio = settings.ratios[lod];
    if (lod === 0) {
      lods.push({
        lod,
        target_ratio,
        actual_ratio: 1.0,
        error: 0.0,
        indices,
      });
      continue;
    }

    const target_triangle_count = Math.max(
      1,
      Math.min(source_triangle_count, Math.floor(source_triangle_count * target_ratio))
    );
    const target_index_count = target_triangle_count * 3;
    const flags = settings.lock_border ? ["LockBorder"] : [];
    const [simplified_indices, error] = MeshoptSimplifier.simplify(
      indices,
      positions,
      3,
      target_index_count,
      settings.max_error,
      flags
    );
    const lod_indices = simplified_indices.length >= 3 ? simplified_indices : indices;

    lods.push({
      lod,
      target_ratio,
      actual_ratio: lod_indices.length / indices.length,
      error,
      indices: lod_indices,
    });
  }

  return lods;
}

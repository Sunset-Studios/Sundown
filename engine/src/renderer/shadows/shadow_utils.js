// -----------------------------------------------------------------------------
// Shadow Utilities
// -----------------------------------------------------------------------------
// Utilities shared by the shadow-mapping pipeline (directional light setup, etc.)
// -----------------------------------------------------------------------------

import { WORLD_FORWARD, WORLD_RIGHT } from "../../core/minimal.js";
import { TypedStack } from "../../memory/container.js";
import { quat, vec3, mat3, mat4, vec4 } from "gl-matrix";

// Default orthographic extent (±extent) for directional lights in clip-space.
// Used when constructing stable light-aligned view/projection matrices.
export const DEFAULT_LIGHT_CLIP_EXTENT = 512.0;
// virtual_dim has to match the AS-VSM instance you create (16384 by default).
export const VSM_VIRTUAL_DIM = 16384.0;
// Maximum number of clipmap levels.
export const MAX_CLIPMAP_LEVELS = 1;
// Size (world units) of one virtual-shadow-map texel in clip-map level 0.
export const VSM_WORLD_UNITS_PER_TEXEL = DEFAULT_LIGHT_CLIP_EXTENT / VSM_VIRTUAL_DIM;

/**
 * Compute a *stable* rotation quaternion that aligns the light's –Z axis with the
 * provided world-space light direction (negated position vector). The resulting
 * basis is constructed to minimise jitter when the light direction approaches
 * the world up vector (poles).
 *
 * @param {ReadonlyArray<number>} light_position – light position/direction as [x, y, z].
 * @returns {quat} A quaternion representing world→light rotation.
 */
export function compute_directional_light_rotation(light_position) {
  // Forward towards the scene (i.e. towards the origin).
  const light_forward = vec3.normalize(
    vec3.create(),
    vec3.negate(vec3.create(), light_position)
  );

  // Right (X) axis = cross(world_up, forward). Pick a stable fallback axis if they're nearly colinear.
  let x_axis = vec3.cross(vec3.create(), WORLD_FORWARD, light_forward);
  if (vec3.length(x_axis) < 1e-4) {
    x_axis = vec3.clone(WORLD_RIGHT);
  }
  vec3.normalize(x_axis, x_axis);

  // Up (Y) axis completes the orthonormal basis.
  const y_axis = vec3.cross(vec3.create(), light_forward, x_axis);
  vec3.normalize(y_axis, y_axis);

  // Build rotation matrix rows and convert to quaternion.
  const rot_rows = mat3.fromValues(
    x_axis[0],
    x_axis[1],
    x_axis[2],
    y_axis[0],
    y_axis[1],
    y_axis[2],
    light_forward[0],
    light_forward[1],
    light_forward[2]
  );
  return quat.fromMat3(quat.create(), rot_rows);
}

/**
 * Build an orthographic projection matrix centred on the origin with extents
 * ±clip_extent and the provided far plane distance. Near plane is fixed at 0.
 *
 * Layout matches WebGPU / gl-matrix column-major expectations.
 *
 * @param {number} far – Far plane distance (camera view far).
 * @param {number} [clip_extent=DEFAULT_DIRECTIONAL_LIGHT_CLIP_EXTENT] – Half-width/height of the ortho volume.
 * @returns {mat4} 4×4 projection matrix.
 */
export function build_directional_light_projection_matrix(
  far,
  aspect = 1.0,
  clip_extent = DEFAULT_LIGHT_CLIP_EXTENT
) {
  const s = 2.0 / clip_extent;
  const projection_matrix = mat4.fromValues(
    s / aspect, 0.0, 0.0, 0.0,
    0.0, s, 0.0, 0.0,
    0.0, 0.0, 1.0 / -far, 0.0,
    0.0, 0.0, 0.0, 1.0
  );
  return projection_matrix;
}

/**
 * Compute the clip-space position for a directional-light view that encloses the camera frustum.
 * It snaps the cascade origin to a virtual-page grid for stability and centers the light camera
 * over the frustum's bounding box in light-space.
 *
 * @param {vec4} world_position – The camera's world position (unused directly when computing frustum).
 * @param {quat} light_rotation – Quaternion representing world→light rotation.
 * @param {number} far – Camera far plane distance.
 * @param {mat4} inv_view_projection – Inverse of the camera's view-projection matrix.
 * @returns {vec4} The page-aligned translation for the light view matrix.
 */
const ndc = [
  [-1, -1, 0, 1], [-1, -1, 1, 1], [-1, 1, 0, 1], [-1, 1, 1, 1],
  [1, -1, 0, 1], [1, -1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1],
];
export function compute_directional_light_position_for_clip(light_rotation, world_position, inv_view_projection) {
  // Transform the eight NDC frustum corners into world-space (divide by w) and
  // then into light-space (apply light rotation). While doing so, track the AABB
  // extents in light-space so that we can determine the centre of the frustum
  // from the light's point of view.

  // Rotation matrix that transforms a world-space vector into light-space.
  const world_to_light = mat3.fromQuat(mat3.create(), light_rotation);

  const aabb_min = vec3.fromValues(Infinity, Infinity, Infinity);
  const aabb_max = vec3.fromValues(-Infinity, -Infinity, -Infinity);

  for (let i = 0; i < ndc.length; i++) {
    // World-space position of the current NDC corner.
    const corner_ws = vec4.transformMat4(vec4.create(), ndc[i], inv_view_projection);
    // Light-space position (rotation only – directional lights have no
    // translation component).
    const corner_ls = vec3.transformMat3(vec3.create(), corner_ws, world_to_light);

    // Expand light-space AABB.
    vec3.min(aabb_min, aabb_min, corner_ls);
    vec3.max(aabb_max, aabb_max, corner_ls);
  }

  // Light-space centre of the frustum AABB, **snapped** to the virtual-page grid.
  const radius = vec3.distance(aabb_min, aabb_max);
  const centre_ls_raw =
      vec3.scale(vec3.create(), vec3.add(vec3.create(), aabb_min, aabb_max), 0.5);
  const snap = VSM_WORLD_UNITS_PER_TEXEL;
  const centre_ls = vec3.fromValues(
      Math.round(centre_ls_raw[0] / snap) * snap,
      Math.round(centre_ls_raw[1] / snap) * snap,
      Math.round(centre_ls_raw[2] / snap) * snap,
  );

  // Convert the centre back to world-space so that it can be used directly as
  // the view's translation component.
  const light_to_world_rot = quat.invert(quat.create(), light_rotation);
  const centre_ws = vec3.transformQuat(vec3.create(), centre_ls, light_to_world_rot);

  const light_forward = vec3.transformQuat(vec3.create(), WORLD_FORWARD, light_rotation);
  const center_adjusted = vec3.scaleAndAdd(vec3.create(), centre_ws, light_forward, -radius);

  return vec4.fromValues(center_adjusted[0], center_adjusted[1], center_adjusted[2], 1.0);
}

/**
 * Allocator for shadow indices.
 *
 * This allocator is used to allocate and free shadow indices.
 * Shadow indices are used to map page tables and feedback bitmasks per shadow casting light.
 *
 */
export class ShadowAllocator {
  static initialized = false;
  static current_size = 1;
  static free_list = new TypedStack(1, Uint32Array);

  static init() {
    if (ShadowAllocator.initialized) return;
    for (let i = 0; i < ShadowAllocator.current_size; i++) {
      ShadowAllocator.free_list.push(i);
    }
    ShadowAllocator.initialized = true;
  }

  static allocate() {
    ShadowAllocator.init();
    if (ShadowAllocator.free_list.is_empty()) {
      return ShadowAllocator.current_size++;
    } else {
      return ShadowAllocator.free_list.pop();
    }
  }

  static free(index) {
    ShadowAllocator.init();
    ShadowAllocator.free_list.push(index);
  }

  static get_total_count() {
    return ShadowAllocator.current_size;
  }
}

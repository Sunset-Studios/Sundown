// -----------------------------------------------------------------------------
// Shadow Utilities
// -----------------------------------------------------------------------------
// Utilities shared by the shadow-mapping pipeline (directional light setup, etc.)
// -----------------------------------------------------------------------------

import { WORLD_FORWARD, WORLD_RIGHT, WORLD_UP } from "../../core/minimal.js";
import { TypedStack } from "../../memory/container.js";
import { quat, vec3, mat3, mat4, vec4 } from "gl-matrix";

// Tile size for both virtual and physical tiles.
export const TILE_SIZE = 128;
// Shadow atlas size.
export const ATLAS_SIZE = 32 * TILE_SIZE;
// virtual_dim has to match the AS-VSM instance you create (16384 by default).
export const VSM_VIRTUAL_DIM = 32 * TILE_SIZE;
// Default orthographic extent (±extent) for directional lights in clip-space 0.
// Used when constructing stable light-aligned view/projection matrices.
export const DEFAULT_LIGHT_CLIP_EXTENT = 4;
// Maximum number of clipmap levels.
export const MAX_CLIPMAP_LEVELS = 12;
// Size (world units) of one virtual-shadow-map texel in clip-map level 0.
export const VSM_WORLD_UNITS_PER_TEXEL =
  DEFAULT_LIGHT_CLIP_EXTENT * (1 << MAX_CLIPMAP_LEVELS) / ATLAS_SIZE;

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
  const light_forward = vec3.normalize(vec3.create(), vec3.negate(vec3.create(), light_position));

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
 * Computes the light's view and projection matrices for the first clipmap.
 * Ensures the ortho projection is square and centered on the frustum in light space.
 * @param {mat4} cam_inverse_view_projection - Inverse of the camera's view-projection matrix.
 * @param {vec3} light_dir - Light direction (normalized).
 * @param {number} far - Far plane for the shadow map.
 * @returns {{view: mat4, proj: mat4}}
 */
export function compute_directional_light_view_projection(
  cam_inverse_view_projection,
  light_dir,
  far
) {
  // 1. Compute light rotation (world -> light space)
  // Light "forward" is -light_dir
  const world_up = vec3.dot(light_dir, WORLD_UP) > 0.99 ? WORLD_FORWARD : WORLD_UP;
  let light_right = vec3.cross(vec3.create(), world_up, light_dir);
  if (vec3.length(light_right) < 1e-4) {
    light_right = vec3.clone(WORLD_RIGHT);
  }
  vec3.normalize(light_right, light_right);
  const light_up = vec3.cross(vec3.create(), light_dir, light_right);
  vec3.normalize(light_up, light_up);

  const rot_rows = mat3.fromValues(
    light_right[0], light_right[1], light_right[2],
    light_up[0],    light_up[1],    light_up[2],
    light_dir[0], light_dir[1], light_dir[2]
  );
  const light_rot = mat4.fromQuat(mat4.create(), quat.fromMat3(quat.create(), rot_rows));

  // 2. Center of AABB in light space
  const center_ws = vec4.transformMat4(vec4.create(), vec4.fromValues(0.0, 0.0, 0.0, 1.0), cam_inverse_view_projection);
  center_ws[0] /= center_ws[3];
  center_ws[1] /= center_ws[3];
  center_ws[2] /= center_ws[3];
  center_ws[3] = 1.0;

  const center_ls_raw = vec4.transformMat4(vec4.create(), center_ws, light_rot);

  // 3. Snap center to virtual texel grid
  const center_ls = vec3.fromValues(
    Math.round(center_ls_raw[0] / VSM_WORLD_UNITS_PER_TEXEL) * VSM_WORLD_UNITS_PER_TEXEL,
    Math.round(center_ls_raw[1] / VSM_WORLD_UNITS_PER_TEXEL) * VSM_WORLD_UNITS_PER_TEXEL,
    Math.round(center_ls_raw[2] / VSM_WORLD_UNITS_PER_TEXEL) * VSM_WORLD_UNITS_PER_TEXEL
  );

  // 4. Convert snapped center back to world space
  const inv_light_rot = mat4.invert(mat4.create(), light_rot);
  const center_ws_adjusted = vec4.transformMat4(
    vec4.create(),
    vec4.fromValues(center_ls[0], center_ls[1], center_ls[2], 1.0),
    inv_light_rot
  );
  // 5. Build light view matrix (look at snapped center from light direction)
  const eye = vec3.scaleAndAdd(vec3.create(), center_ws_adjusted, light_dir, far * 0.5);
  const light_view = mat4.lookAt(mat4.create(), eye, center_ws_adjusted, light_up);

  // 6. Fixed ortho projection
  const extent = DEFAULT_LIGHT_CLIP_EXTENT;
  const light_proj = mat4.ortho(mat4.create(),
    -extent, extent,
    -extent, extent,
    -far, far
  );
 
  return { view: light_view, proj: light_proj };
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


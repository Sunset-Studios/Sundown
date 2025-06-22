// -----------------------------------------------------------------------------
// Shadow Utilities
// -----------------------------------------------------------------------------
// Utilities shared by the shadow-mapping pipeline (directional light setup, etc.)
// -----------------------------------------------------------------------------

import { WORLD_UP, WORLD_RIGHT, WORLD_FORWARD } from "../../core/minimal.js";
import { TypedStack } from "../../memory/container.js";
import { quat, vec3, mat3, mat4, vec4 } from "gl-matrix";

// Default orthographic extent (±extent) for directional lights in clip-space.
// Used when constructing stable light-aligned view/projection matrices.
export const DEFAULT_DIRECTIONAL_LIGHT_CLIP_EXTENT = 8.0;

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
  // Forward towards the scene (-light_position).
  const light_forward = vec3.normalize(vec3.create(), vec3.negate(vec3.create(), light_position));
  // Right (X) axis = cross(forward, world_up). Pick a stable axis at the poles.
  let x_axis = vec3.cross(vec3.create(), WORLD_FORWARD, light_forward);
  if (vec3.length(x_axis) < 1e-4) {
    x_axis = vec3.clone(WORLD_FORWARD);
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
  mat3.transpose(rot_rows, rot_rows);
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
  clip_extent = DEFAULT_DIRECTIONAL_LIGHT_CLIP_EXTENT
) {
  return mat4.fromValues(
    2.0 / clip_extent,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0 / clip_extent,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0 / far,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0
  );
}

/**
 * Compute the clip-space position of a world-space position for a directional light.
 *
 * The cascade origin is snapped to a virtual-page grid to avoid sub-texel
 * jittering when the camera moves by fractional amounts.
 *
 * @param {vec4} world_position – Camera/world position.
 * @returns {vec4} The page-aligned clip-space position.
 */
export function compute_directional_light_position_for_clip(world_position, light_rotation, far) {
  // Size of one virtual page in world-space units (default AS-VSM setup).
  //   clip_extent = 8  →  full width = 16
  //   virtual tiles across (LOD0) = 128 → 16 / 128 = 0.125
  const WORLD_SPACE_PAGE_SIZE = DEFAULT_DIRECTIONAL_LIGHT_CLIP_EXTENT / 128.0;

  const light_direction = vec3.transformQuat(vec3.create(), WORLD_FORWARD, light_rotation);
  const position = vec4.scaleAndAdd(vec4.create(), world_position, light_direction, -far);

  // Snap X/Y to the nearest page, then negate (light-space translation).
  position[0] = Math.round(position[0] / WORLD_SPACE_PAGE_SIZE) * WORLD_SPACE_PAGE_SIZE;
  position[1] = -Math.round(position[1] / WORLD_SPACE_PAGE_SIZE) * WORLD_SPACE_PAGE_SIZE;

  return position;
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

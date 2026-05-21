import { vec4 } from "gl-matrix";

export const MAX_BUFFERED_FRAMES = 2;
export const INVALID_U32 = 0xffffffff;

export const WORLD_UP = vec4.fromValues(0, 1, 0, 0);
export const WORLD_FORWARD = vec4.fromValues(0, 0, 1, 0);
export const WORLD_RIGHT = vec4.fromValues(1, 0, 0, 0);

export const LightType = {
  DIRECTIONAL: 0,
  POINT: 1,
  SPOT: 2,
};

// TODO: Many of these flags are mostly communicated to the GPU, and the GPU will clear them,
// so it is unreliable to use them for CPU-side logic. We may want to add a separate flags
// entity array to Solar for CPU-only logic.
export const EntityFlags = {
  ALIVE: 1 << 0,
  DIRTY: 1 << 1,
  IGNORE_PARENT_SCALE: 1 << 2,
  IGNORE_PARENT_ROTATION: 1 << 3,
  TRANSFORM_DIRTY: 1 << 4,
  BILLBOARD: 1 << 5,
  MOVED: 1 << 6,
  HAS_MESH: 1 << 7,
  INTERACTIVE: 1 << 8,
  IGNORE_TLAS: 1 << 9,
};


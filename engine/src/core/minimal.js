import { vec4 } from "gl-matrix";

export const MAX_BUFFERED_FRAMES = 2;

export const WORLD_UP = vec4.fromValues(0, 1, 0, 0);
export const WORLD_FORWARD = vec4.fromValues(0, 0, 1, 0);
export const WORLD_RIGHT = vec4.fromValues(1, 0, 0, 0);

export const LightType = {
  DIRECTIONAL: 0,
  POINT: 1,
  SPOT: 2,
};

export const EntityFlags = {
  ALIVE: 1 << 0,
  DIRTY: 1 << 1,
  IGNORE_PARENT_SCALE: 1 << 2,
  IGNORE_PARENT_ROTATION: 1 << 3,
  TRANSFORM_DIRTY: 1 << 4,
  NO_AABB_UPDATE: 1 << 5,
  AABB_DIRTY: 1 << 6,
  BILLBOARD: 1 << 7,
};


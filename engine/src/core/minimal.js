import { vec4 } from "gl-matrix";

export const WORLD_UP = vec4.fromValues(0, 1, 0, 0);
export const WORLD_FORWARD = vec4.fromValues(0, 0, 1, 0);
export const WORLD_RIGHT = vec4.fromValues(1, 0, 0, 0);

export const LightType = {
  DIRECTIONAL: 0,
  POINT: 1,
  SPOT: 2,
};

export const EntityTransformFlags = {
  DIRTY: 1 << 0,
  IGNORE_PARENT_SCALE: 1 << 1,
  IGNORE_PARENT_ROTATION: 1 << 2,
  TRANSFORM_DIRTY: 1 << 3,
  NO_AABB_UPDATE: 1 << 4,
};

export const EntityFlags = { };


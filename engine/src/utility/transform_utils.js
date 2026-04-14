import { mat4, quat, vec3 } from "gl-matrix";
import { EntityFlags } from "../core/minimal.js";
import { EntityManager } from "../core/ecs/entity.js";
import { DEFAULT_CHUNK_CAPACITY } from "../core/ecs/solar/types.js";
import { Name } from "./names.js";

const identity_transform = mat4.create();
const temp_parent_transform = mat4.create();
const temp_local_transform = mat4.create();
const temp_translation = vec3.create();
const temp_scale = vec3.create();
const identity_rotation = quat.create();
const transform_fragment_id = Name.from("transform");

function get_transform_views(entity, instance = 0) {
  const segment_index = Math.floor(instance / DEFAULT_CHUNK_CAPACITY);
  const instance_offset = instance % DEFAULT_CHUNK_CAPACITY;
  const segment = entity?.segments?.[segment_index];
  if (!segment) {
    return null;
  }

  const views = segment.chunk.fragment_views[transform_fragment_id];
  if (!views) {
    return null;
  }

  return {
    slot: segment.slot + instance_offset,
    views,
  };
}

function get_entity_transform_flags(entity, instance = 0) {
  const transform_view = get_transform_views(entity, instance);
  if (!transform_view) {
    return EntityManager.get_entity_flags(entity);
  }

  return transform_view.views
    ? entity.segments[Math.floor(instance / DEFAULT_CHUNK_CAPACITY)].chunk.flags_meta[
        transform_view.slot
      ]
    : EntityManager.get_entity_flags(entity);
}

export function strip_parent_rotation(transform, out = mat4.create()) {
  vec3.set(temp_translation, transform[12], transform[13], transform[14]);
  mat4.getScaling(temp_scale, transform);

  return mat4.fromRotationTranslationScale(
    out,
    identity_rotation,
    temp_translation,
    temp_scale
  );
}

export function strip_parent_scale(transform, out = mat4.create()) {
  mat4.copy(out, transform);

  const scale_x = Math.hypot(out[0], out[1], out[2]);
  const scale_y = Math.hypot(out[4], out[5], out[6]);
  const scale_z = Math.hypot(out[8], out[9], out[10]);

  if (scale_x > 1e-6) {
    out[0] /= scale_x;
    out[1] /= scale_x;
    out[2] /= scale_x;
  }

  if (scale_y > 1e-6) {
    out[4] /= scale_y;
    out[5] /= scale_y;
    out[6] /= scale_y;
  }

  if (scale_z > 1e-6) {
    out[8] /= scale_z;
    out[9] /= scale_z;
    out[10] /= scale_z;
  }

  return out;
}

export function resolve_parent_transform(parent_transform, flags, out = mat4.create()) {
  if (!parent_transform) {
    return mat4.identity(out);
  }

  mat4.copy(out, parent_transform);

  if ((flags & EntityFlags.IGNORE_PARENT_ROTATION) !== 0) {
    strip_parent_rotation(out, out);
  }

  if ((flags & EntityFlags.IGNORE_PARENT_SCALE) !== 0) {
    strip_parent_scale(out, out);
  }

  return out;
}

export function compose_world_transform(
  local_position,
  local_rotation,
  local_scale,
  parent_transform = null,
  flags = 0,
  out = mat4.create()
) {
  const resolved_parent_transform = resolve_parent_transform(
    parent_transform,
    flags,
    temp_parent_transform
  );

  mat4.fromRotationTranslationScale(
    temp_local_transform,
    local_rotation,
    local_position,
    local_scale
  );

  return mat4.multiply(out, resolved_parent_transform, temp_local_transform);
}

export function get_current_world_transform(entity, instance = 0, out = mat4.create()) {
  const hierarchy = [];
  let current_entity = entity;
  let current_instance = instance;

  while (current_entity) {
    hierarchy.push({ entity: current_entity, instance: current_instance });
    current_entity = EntityManager.get_entity_parent(current_entity);
    current_instance = 0;
  }

  mat4.identity(out);

  for (let i = hierarchy.length - 1; i >= 0; i--) {
    const { entity: hierarchy_entity, instance: hierarchy_instance } = hierarchy[i];
    const transform_view = get_transform_views(hierarchy_entity, hierarchy_instance);
    if (!transform_view) {
      continue;
    }

    const position_offset = transform_view.slot * 4;
    const rotation_offset = transform_view.slot * 4;
    const scale_offset = transform_view.slot * 4;

    compose_world_transform(
      transform_view.views.position.subarray(position_offset, position_offset + 4),
      transform_view.views.rotation.subarray(rotation_offset, rotation_offset + 4),
      transform_view.views.scale.subarray(scale_offset, scale_offset + 4),
      out,
      EntityManager.get_entity_flags(hierarchy_entity),
      out
    );
  }

  return out;
}

export function get_current_world_position(entity, instance = 0, out = vec3.create()) {
  const world_transform = get_current_world_transform(entity, instance, identity_transform);
  return mat4.getTranslation(out, world_transform);
}

export function get_current_world_rotation(entity, instance = 0, out = quat.create()) {
  const world_transform = get_current_world_transform(entity, instance, identity_transform);
  return mat4.getRotation(out, world_transform);
}

export function get_current_world_scale(entity, instance = 0, out = vec3.create()) {
  const world_transform = get_current_world_transform(entity, instance, identity_transform);
  return mat4.getScaling(out, world_transform);
}

export function is_transform_hierarchy_dirty(entity, instance = 0) {
  if (get_entity_transform_flags(entity, instance) & EntityFlags.TRANSFORM_DIRTY) {
    return true;
  }

  let current_entity = EntityManager.get_entity_parent(entity);
  while (current_entity) {
    if (get_entity_transform_flags(current_entity, 0) & EntityFlags.TRANSFORM_DIRTY) {
      return true;
    }
    current_entity = EntityManager.get_entity_parent(current_entity);
  }

  return false;
}

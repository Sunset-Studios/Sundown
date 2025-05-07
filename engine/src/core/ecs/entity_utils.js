import { Mesh } from "../../renderer/mesh.js";
import { EntityManager } from "./entity.js";
import { TransformFragment } from "./fragments/transform_fragment.js";
import { StaticMeshFragment } from "./fragments/static_mesh_fragment.js";
import { VisibilityFragment } from "./fragments/visibility_fragment.js";
import { Name } from "../../utility/names.js";
import { WORLD_FORWARD, EntityFlags } from "../../core/minimal.js";
import { quat } from "gl-matrix";

/**
 * Spawns a new mesh entity with the specified position, rotation, scale, mesh, material, parent,
 * children, visibility, and transform flags.
 *
 * @param {Vec3} position - The position of the entity.
 * @param {Vec3} rotation - The rotation of the entity.
 * @param {Vec3} scale - The scale of the entity.
 * @param {Mesh} mesh - The mesh to use for the entity.
 * @param {Material} material - The material to use for the entity.
 * @param {EntityID} parent - The parent entity of the entity.
 * @param {EntityID[]} children - The children entities of the entity.
 * @param {boolean} start_visible - Whether the entity should be visible.
 * @param {EntityFlags} flags - The flags of the entity.
 */
export function spawn_mesh_entity(
  position,
  rotation,
  scale,
  mesh,
  material,
  parent = null,
  children = [],
  start_visible = true,
  flags = EntityFlags.IGNORE_PARENT_SCALE
) {
  const entity = EntityManager.create_entity([TransformFragment, StaticMeshFragment, VisibilityFragment]);

  const new_transform_view = EntityManager.get_fragment(entity, TransformFragment);
  new_transform_view.position = position;
  new_transform_view.rotation = rotation;
  new_transform_view.scale = scale;

  EntityManager.set_entity_parent(entity, parent);
  if (children.length > 0) {
    EntityManager.set_entity_children(entity, children);
  }

  let existing_flags = EntityManager.get_entity_flags(entity);
  EntityManager.set_entity_flags(entity, existing_flags | flags);

  const new_static_mesh_view = EntityManager.get_fragment(entity, StaticMeshFragment);
  new_static_mesh_view.mesh = BigInt(Name.from(mesh.name));
  new_static_mesh_view.material_slots = [BigInt(material)];

  const new_visibility_view = EntityManager.get_fragment(entity, VisibilityFragment);
  new_visibility_view.visible = start_visible;

  return entity;
}

/**
 * Spawns a new plane entity with the specified position, normal, scale, material, parent, and children.
 *
 * @param {Vec3} position - The position of the entity.
 * @param {Vec3} normal - The normal of the plane.
 * @param {Vec3} scale - The scale of the entity.
 * @param {Material} material - The material to use for the entity.
 * @param {EntityID} parent - The parent entity of the entity.
 * @param {EntityID[]} children - The children entities of the entity.
 */
export function spawn_plane_entity(
  position,
  normal,
  scale,
  material,
  parent = null,
  children = []
) {
  const entity = EntityManager.create_entity([TransformFragment, StaticMeshFragment, VisibilityFragment]);

  const new_transform_view = EntityManager.get_fragment(entity, TransformFragment);
  new_transform_view.position = position;
  new_transform_view.rotation = quat.rotationTo(WORLD_FORWARD, normal);
  new_transform_view.scale = scale;

  EntityManager.set_entity_parent(entity, parent);
  if (children.length > 0) {
    EntityManager.set_entity_children(entity, children);
  }

  const mesh = Mesh.quad();
  const new_static_mesh_view = EntityManager.get_fragment(entity, StaticMeshFragment);
  new_static_mesh_view.mesh = BigInt(Name.from(mesh.name));
  new_static_mesh_view.material_slots = [BigInt(material)];

  const new_visibility_view = EntityManager.get_fragment(entity, VisibilityFragment);
  new_visibility_view.visible = true;

  return entity;
}

/**
 * Returns the parent entity of the specified entity.
 *
 * @param {EntityID} entity - The entity to get the parent of.
 * @returns {EntityID} The parent entity of the specified entity.
 */
export function get_entity_parent(entity) {
  return EntityManager.get_entity_parent(entity);
}

/**
 * Returns the children entities of the specified entity.
 *
 * @param {EntityID} entity - The entity to get the children of.
 * @returns {EntityID[]} The children entities of the specified entity.
 */
export function get_entity_children(entity) {
  return EntityManager.get_entity_children(entity);
}

/**
 * Deletes the specified entity and optionally its children.
 *
 * @param {EntityID} entity - The entity to delete.
 * @param {boolean} delete_children - Whether to delete the children entities.
 */
export function delete_entity(entity, delete_children = false) {
  EntityManager.delete_entity(entity);
  if (delete_children) {
    const children = get_entity_children(entity);
    for (let i = 0; i < children.length; i++) {
      delete_entity(children[i], delete_children);
    }
  }
}

/**
 * Deletes all children entities of the specified entity that have the specified tag.
 *
 * @param {EntityID} entity - The entity to delete the children of.
 * @param {Tag} tag - The tag of the children entities to delete.
 */
export function delete_entity_children_with_tag(entity, tag) {
  const children = EntityManager.get_entity_children(entity);
  for (let i = 0; i < children.length; i++) {
    const child = children[i];
    if (EntityManager.has_tag(child, tag)) {
      delete_entity(child);
    }
  }
}

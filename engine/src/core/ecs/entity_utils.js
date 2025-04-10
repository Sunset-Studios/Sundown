import { Mesh } from "../../renderer/mesh.js";
import { EntityManager } from "./entity.js";
import { TransformFragment } from "./fragments/transform_fragment.js";
import { StaticMeshFragment } from "./fragments/static_mesh_fragment.js";
import { SceneGraphFragment } from "./fragments/scene_graph_fragment.js";
import { VisibilityFragment } from "./fragments/visibility_fragment.js";
import { Name } from "../../utility/names.js";
import { WORLD_FORWARD, EntityTransformFlags } from "../../core/minimal.js";
import { quat } from "gl-matrix";

export class EntityLinearDataContainer {
  constructor(container_type = Uint32Array) {
    this.container_type = container_type;
    this.linear_data = new container_type(2048);
    this.entity_indices = new Map();
    this.next_available_index = 0;
  }

  /**
   * Allocates space for a new data item and stores the data.
   * @param {number} entity - The entity ID.
   * @param {number[]} data - An array containing the new data.
   * @returns {object} The allocation details with start index and count.
   */
  allocate(entity, data) {
    const start_index = this.next_available_index;
    this.entity_indices.set(entity, { start: start_index, count: data.length });
    this.next_available_index += data.length;

    if (this.next_available_index > this.linear_data.length) {
      this.resize(Math.max(this.linear_data.length * 2, this.next_available_index));
    }

    this.linear_data.set(data, start_index);

    return this.entity_indices.get(entity);
  }

  /**
   * Updates the data for an existing entity.
   * @param {number} entity - The entity ID.
   * @param {number[]} new_data - An array containing the new data.
   */
  update(entity, new_data) {
    const entity_data = this.entity_indices.get(entity);
    if (!entity_data) {
      this.allocate(entity, new_data);
      return;
    }

    const old_count = entity_data.count;
    entity_data.count = new_data.length;
    const start_index = entity_data.start;

    const diff = entity_data.count - old_count;
    if (diff > 0) {
      this.shift_data(start_index, diff);
    } else if (diff < 0) {
      this.shift_data(start_index + old_count - 1, diff + 1);
    }

    this.linear_data.set(new_data, start_index);
  }

  /**
   * Removes an entity's data and shifts subsequent data to fill the gap.
   * @param {number} entity - The entity ID.
   */
  remove(entity) {
    const entity_data = this.entity_indices.get(entity);
    if (!entity_data) return;

    this.shift_data(entity_data.start + entity_data.count, -entity_data.count);
    this.entity_indices.delete(entity);
  }

  /**
   * Shifts data in the array to maintain contiguity.
   * @param {number} fromIndex - The index from which to start shifting.
   * @param {number} shiftAmount - The number of positions to shift (positive or negative).
   */
  shift_data(from_index, shift_amount) {
    if (shift_amount === 0) return;

    const adjusted_shift_amount = from_index + shift_amount < 0 ? -from_index : shift_amount;

    this.linear_data.copyWithin(
      from_index + adjusted_shift_amount,
      from_index
    );

    if (from_index + adjusted_shift_amount >= this.linear_data.length) {
      this.resize(Math.max(this.linear_data.length * 2, this.linear_data.length + adjusted_shift_amount));
    }

    for (const [entity, data] of this.entity_indices.entries()) {
      if (data.start > from_index) {
        data.start += adjusted_shift_amount;
      }
    }

    this.next_available_index += adjusted_shift_amount;
    this.next_available_index = Math.max(this.next_available_index, 0);
  }

  /**
   * Shifts data in the array and fills any new gaps with duplicated data.
   * @param {number} entity_index - The starting index to shift from.
   * @param {number} shift_amount - The number of positions to shift (positive or negative).
   */
  shift_data_with_fill(entity_index, shift_amount) {
    if (shift_amount === 0) return;

    const adjusted_shift_amount = entity_index + shift_amount < 0 ? -entity_index : shift_amount;

    if (adjusted_shift_amount > 0) {
      // Make space by moving data forward
      let i = Math.min(this.linear_data.length, this.linear_data.length - adjusted_shift_amount) - 1;
      for (; i >= entity_index; --i) {
        this.linear_data[i + adjusted_shift_amount] = this.linear_data[i];
      }
      // Fill the gap with duplicated data
      i += 1;
      for (; i < entity_index + adjusted_shift_amount; ++i) {
        this.linear_data[i] = this.linear_data[entity_index];
      }
    } else {
      // Compress by moving data backward
      const size = Math.max(this.linear_data.length, this.linear_data.length - adjusted_shift_amount);
      for (let i = entity_index; i < size; ++i) {
        this.linear_data[i] = this.linear_data[i - adjusted_shift_amount];
      }
    }

    // Update metadata for affected entities
    for (const [entity, data] of this.entity_indices.entries()) {
      if (data.start > entity_index) {
        data.start += adjusted_shift_amount;
      }
    }

    this.next_available_index += adjusted_shift_amount;
    this.next_available_index = Math.max(this.next_available_index, 0);

    // Resize if needed
    if (this.next_available_index >= this.linear_data.length) {
      this.resize(Math.max(this.linear_data.length * 2, this.next_available_index));
    }
  }

  /**
   * Resizes the data array to a new size.
   * @param {number} new_size - The new size of the data array.
   */
  resize(new_size) {
    const new_connections = new this.container_type(new_size);
    new_connections.set(this.linear_data);
    this.linear_data = new_connections;
  }

  /**
   * Retrieves the metadata for a specific entity.
   * @param {number} entity - The entity ID.
   * @returns {object|null} An object containing the metadata for the entity or null if not found.
   */
  get_metadata(entity) {
    const entity_data = this.entity_indices.get(entity);
    return entity_data ? entity_data : null;
  }

  /**
   * Retrieves the data for a specific entity.
   * @param {number} entity - The entity ID.
   * @returns {Uint32Array|null} An array containing the data for the entity or null if not found.
   */
  get_data_for_entity(entity) {
    const entity_data = this.get_metadata(entity);
    return entity_data ? this.linear_data.subarray(
      entity_data.start,
      entity_data.start + entity_data.count
    ) : null;
  }

  /**
   * Retrieves all data items.
   * @returns {Uint32Array} A subarray containing all active data items.
   */
  get_data() {
    return this.linear_data;
  }
}

export function spawn_mesh_entity(
  position,
  rotation,
  scale,
  mesh,
  material,
  parent = null,
  children = [],
  start_visible = true,
  transform_flags = EntityTransformFlags.IGNORE_PARENT_SCALE,
) {
  const entity = EntityManager.create_entity();

  const new_transform_view = EntityManager.add_fragment(
    entity,
    TransformFragment,
  );
  new_transform_view.position = position;
  new_transform_view.rotation = rotation;
  new_transform_view.scale = scale;

  let flags = new_transform_view.flags;
  flags |= transform_flags;
  new_transform_view.flags = flags;

  const new_scene_graph_view = EntityManager.add_fragment(
    entity,
    SceneGraphFragment,
  );
  new_scene_graph_view.parent = parent;
  if (children.length > 0) {
    new_scene_graph_view.children = children;
  }

  const new_static_mesh_view = EntityManager.add_fragment(
    entity,
    StaticMeshFragment,
  );
  new_static_mesh_view.mesh = BigInt(Name.from(mesh.name));
  new_static_mesh_view.material_slots = [material];

  const new_visibility_view = EntityManager.add_fragment(
    entity,
    VisibilityFragment,
  );
  new_visibility_view.visible = start_visible;

  return entity;
}

export function spawn_plane_entity(position, normal, scale, material, parent = null, children = []) {
  const entity = EntityManager.create_entity();

  const new_transform_view = EntityManager.add_fragment(
    entity,
    TransformFragment,
  );
  new_transform_view.position = position;
  new_transform_view.rotation = quat.rotationTo(WORLD_FORWARD, normal);
  new_transform_view.scale = scale;

  const new_scene_graph_view = EntityManager.add_fragment(
    entity,
    SceneGraphFragment,
  );
  new_scene_graph_view.parent = parent;
  if (children.length > 0) {
    new_scene_graph_view.children = children;
  }

  const mesh = Mesh.quad();
  const new_static_mesh_view = EntityManager.add_fragment(
    entity,
    StaticMeshFragment,
  );  
  new_static_mesh_view.mesh = BigInt(Name.from(mesh.name));
  new_static_mesh_view.material_slots = [material];

  const new_visibility_view = EntityManager.add_fragment(
    entity,
    VisibilityFragment,
  );
  new_visibility_view.visible = start_visible;

  return entity;
}

export function get_entity_parent(entity) {
  const scene_graph_frame = EntityManager.get_fragment(entity, SceneGraphFragment);
  return scene_graph_frame.parent;
}

export function get_entity_children(entity) {
  const scene_graph_frame = EntityManager.get_fragment(entity, SceneGraphFragment);
  return scene_graph_frame.children;
}

export function delete_entity(entity, delete_children = false) {
  EntityManager.delete_entity(entity);
  if (delete_children) {
    const children = get_entity_children(entity);
    for (let i = 0; i < children.length; i++) {
      delete_entity(children[i], delete_children);
    }
  }
}

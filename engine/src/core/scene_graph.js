import { EntityManager } from "./ecs/entity.js";
import { Buffer } from "../renderer/buffer.js";
import { Tree } from "../memory/container.js";
import { EntityID } from "./ecs/solar/types.js";

export class SceneGraph {
  static tree = new Tree();
  static scene_graph_buffer = null;
  static scene_graph_layer_counts = [];
  static scene_graph_uniforms = [];
  static dirty = false;
  static hierarchy_dirty_entities = new Set();
  static hierarchy_moved_entities = new Set();

  static set_parent(entity, parent) {
    this.tree.remove(entity);
    this.tree.add(parent, entity);
    this.dirty = true;
    this.mark_hierarchy_moved(entity);
  }

  static get_parent(entity) {
    const node = this.tree.find_node(entity);
    const parent_node = this.tree.get_parent(node);
    return parent_node ? parent_node.data : null;
  }

  static set_children(entity, children) {
    if (Array.isArray(children)) {
      const previous_children = this.get_children(entity);
      this.tree.add_multiple(entity, children, true /* replace_children */, true /* unique */);
      this.dirty = true;
      for (let i = 0; i < previous_children.length; i++) {
        this.mark_hierarchy_moved(previous_children[i]);
      }
      for (let i = 0; i < children.length; i++) {
        this.mark_hierarchy_moved(children[i]);
      }
    }
  }

  static get_children(entity) {
    const node = this.tree.find_node(entity);
    return [...this.tree.get_children(node)].map((child) => child.data);
  }

  static remove(entity) {
    this.tree.remove(entity);
    this.dirty = true;
  }

  static mark_dirty() {
    this.dirty = true;
  }

  static mark_hierarchy_dirty(entity) {
    if (entity) {
      this.hierarchy_dirty_entities.add(entity);
    }
  }

  static mark_hierarchy_moved(entity) {
    if (entity) {
      this.hierarchy_moved_entities.add(entity);
      this.mark_hierarchy_dirty(entity);
    }
  }

  static is_hierarchy_dirty(entity) {
    return this.hierarchy_dirty_entities.has(entity);
  }

  static is_hierarchy_moved(entity) {
    return this.hierarchy_moved_entities.has(entity);
  }

  static clear_hierarchy_dirty() {
    this.hierarchy_dirty_entities.clear();
    this.hierarchy_moved_entities.clear();
  }

  static flush_gpu_buffers() {
    if (!this.dirty) {
      return;
    }

    const { result, layer_counts } = this.tree.flatten(
      Int32Array,
      (out, node, size) => {
        const parent_entity = node.parent_idx !== -1
          ? this.tree.nodes[node.parent_idx].data
          : null;

        return this.#write_entity_rows(out, size, node.data, parent_entity);
      },
      (node) => EntityManager.get_entity_instance_count(node.data) * 2
    );
    // The flatten helper sizes the Int32Array in scalar elements, so each scene-graph
    // entry contributes 2 ints. Convert layer counts back to entry counts before using
    // them as vec2 scene-graph indices in the transform processor.
    this.scene_graph_layer_counts = layer_counts.map((count) => count >>> 1);

    if (!result) {
      this.dirty = false;
      return;
    }

    if (!this.scene_graph_buffer || this.scene_graph_buffer.config.size < result.byteLength) {
      this.scene_graph_buffer = Buffer.create({
        name: "scene_graph_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        raw_data: result,
        force: true,
      });
    } else {
      this.scene_graph_buffer.write_raw(result);
    }

    let offset = 0;
    this.scene_graph_uniforms = this.scene_graph_layer_counts.map((count, layer) => {
      const uni = Buffer.create({
        name: `scene_graph_uniforms_${layer}`,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        raw_data: new Uint32Array([count, offset, layer]),
        force: true,
      });
      offset += count;
      return uni;
    });

    this.dirty = false;
  }

  static #write_entity_rows(out, size, entity, parent_entity) {
    let write_offset = size;
    const parent_row_id = parent_entity ? parent_entity.id : -1;

    for (let segment_index = 0; segment_index < entity.segments.length; segment_index++) {
      const segment = entity.segments[segment_index];

      for (let row = 0; row < segment.count; row++) {
        const slot = segment.slot + row;
        const generation = segment.chunk.gen_meta[slot];
        const entity_row_id = EntityID.make(slot, segment.chunk.chunk_index, generation);

        out[write_offset++] = entity_row_id;
        out[write_offset++] = parent_row_id;
      }
    }

    return write_offset - size;
  }
}

import { EntityManager } from "./ecs/entity.js";
import { EntityID } from "./ecs/solar/types.js";
import { Buffer } from "../renderer/buffer.js";
import { Tree } from "../memory/container.js";
import { Renderer } from "../renderer/renderer.js";

export class SceneGraph {
  static tree = new Tree();
  static scene_graph_buffer = null;
  static scene_graph_layer_counts = [];
  static scene_graph_uniforms = [];
  static dirty = false;

  static set_parent(entity, parent) {
    this.tree.remove(entity);
    this.tree.add(parent, entity);
    this.dirty = true;
  }

  static get_parent(entity) {
    const node = this.tree.find_node(entity);
    const parent_node = this.tree.get_parent(node);
    return parent_node ? parent_node.data : null;
  }

  static set_children(entity, children) {
    if (Array.isArray(children)) {
      this.tree.add_multiple(entity, children, true /* replace_children */, true /* unique */);
      this.dirty = true;
    }
  }

  static get_children(entity) {
    const node = this.tree.find_node(entity);
    return [...this.tree.get_children(node)].map((child) => child.data);
  }

  static remove(entity) {
    this.tree.remove(entity);
  }

  static mark_dirty() {
    this.dirty = true;
  }

  static flush_gpu_buffers() {
    if (!this.dirty) {
      return;
    }

    const { result, layer_counts } = this.tree.flatten(
      Int32Array,
      (out, node, size) => {
        const entity_idx = node.data.id;
        const count = node.data.instance_count;
        const parent_entity_idx = this.tree.get_parent(node)
          ? this.tree.get_parent(node).data.id
          : -1;
        for (let i = 0; i < count; i++) {
          out[(size + i) * 2] = entity_idx + i;
          out[(size + i) * 2 + 1] = parent_entity_idx;
        }
        return count;
      },
      (node) => EntityManager.get_entity_instance_count(node.data) * 2
    );
    this.scene_graph_layer_counts = layer_counts;

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
    this.scene_graph_uniforms = layer_counts.map((count, layer) => {
      const uni = Buffer.create({
        name: `scene_graph_uniforms_${layer}`,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        raw_data: new Uint32Array([count, offset, layer]),
        force: true,
      });
      offset += count;
      return uni;
    });

    Renderer.get().mark_bind_groups_dirty(true);

    this.dirty = false;
  }
}

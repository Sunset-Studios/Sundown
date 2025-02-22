import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { EntityManager } from "../entity.js";
import { Tree } from "../../../memory/container.js";

const scene_graph_buffer_name = "scene_graph_buffer";
const scene_graph_cpu_buffer_name = "scene_graph_cpu_buffer";
const scene_graph_event = "scene_graph";
const scene_graph_update_event = "scene_graph_update";

class SceneGraphDataView {
  current_entity = -1n;
  absolute_entity = -1n;

  constructor() {}

  get parent() {
    return SceneGraphFragment.data.parent[this.absolute_entity];
  }

  set parent(value) {
    SceneGraphFragment.data.parent[this.absolute_entity] = value ?? -1;
    SceneGraphFragment.data.scene_graph.remove(this.current_entity);
    SceneGraphFragment.data.scene_graph.add(value ?? null, this.current_entity);
    if (SceneGraphFragment.data.dirty) {
      SceneGraphFragment.data.dirty[this.absolute_entity] = 1;
    }
    SceneGraphFragment.data.gpu_data_dirty = true;
  }

  get children() {
    const node = SceneGraphFragment.data.scene_graph.find_node(
      this.current_entity,
    );
    return [...SceneGraphFragment.data.scene_graph.get_children(node)].map(
      (child) => child.data,
    );
  }

  set children(value) {
    if (Array.isArray(value)) {
      SceneGraphFragment.data.scene_graph.add_multiple(
        this.current_entity,
        value,
        true /* replace_children */,
        true /* unique */,
      );
    }
    if (SceneGraphFragment.data.dirty) {
      SceneGraphFragment.data.dirty[this.absolute_entity] = 1;
    }
    SceneGraphFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;

    return this;
  }
}

const unmapped_state = "unmapped";

export class SceneGraphFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, SceneGraphDataView);
  static size = 0;
  static data = null;

  static initialize() {
    this.data = {
      parent: new Int32Array(1),
      scene_graph: new Tree(),
      scene_graph_flattened: [],
      scene_graph_layer_counts: [],
      scene_graph_uniforms: [],
      scene_graph_buffer: null,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers();
  }

  static resize(new_size) {
    if (new_size <= this.size) return;

    this.size = new_size;

    if (!this.data) this.initialize();

    Fragment.resize_array(this.data, "parent", new_size, Int32Array, 1);

    this.data.gpu_data_dirty = true;
  }

  static add_entity(entity) {
    const absolute_entity = EntityID.get_absolute_index(entity);
    if (absolute_entity >= this.size) {
      this.resize(absolute_entity * 2);
    }

    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const entity_instances = EntityID.get_instance_count(entity);
    for (let i = 0; i < entity_instances; i++) {
      this.data.parent[entity_offset + i] = -1;
    }
    this.data.scene_graph.remove(entity);
    this.data.gpu_data_dirty = true;
  }

  static get_entity_data(entity, instance = 0) {
    const data_view = this.data_view_allocator.allocate();
    data_view.view_entity(entity, instance);
    return data_view;
  }

  static duplicate_entity_data(entity, instance = 0) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const node = this.data.scene_graph.find_node(entity_offset);
    return {
      parent: this.data.scene_graph.get_parent(node),
      children: this.data.scene_graph
        .get_children(node)
        .map((child) => child.data),
    };
  }

  static to_gpu_data() {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        scene_graph_buffer: this.data.scene_graph_buffer,
      };
    }

    this.rebuild_buffers();

    return {
      scene_graph_buffer: this.data.scene_graph_buffer,
    };
  }

  static rebuild_buffers() {
    if (!this.data.gpu_data_dirty) return;

    {
      const { result, layer_counts } = this.data.scene_graph.flatten(
        Int32Array,
        (result, node, result_size) => {
          let instance_count = EntityID.get_instance_count(node.data);
          for (let i = 0; i < instance_count; i++) {
            result[result_size + i] =
              EntityID.get_absolute_index(node.data) + i;
          }
          return instance_count;
        },
        (node) => {
          return EntityID.get_instance_count(node.data);
        },
      );

      this.data.scene_graph_flattened = result;
      this.data.scene_graph_layer_counts = layer_counts;

      const num_elements = result?.length ?? 0;
      const gpu_data = new Int32Array(Math.max(num_elements * 2, 2));
      for (let i = 0; i < num_elements; ++i) {
        gpu_data[i * 2] = result[i];
        gpu_data[i * 2 + 1] =
          this.data.parent[result[i]] >= 0
            ? EntityID.get_absolute_index(this.data.parent[result[i]])
            : -1;
      }

      this.data.scene_graph_uniforms = new Array(layer_counts.length);
      let layer_offset = 0;
      for (let i = 0; i < layer_counts.length; ++i) {
        this.data.scene_graph_uniforms[i] = Buffer.create({
          name: "scene_graph_uniforms_" + i,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          raw_data: new Uint32Array([layer_counts[i], layer_offset, i]),
          force: true,
        });
        layer_offset += layer_counts[i];
      }

      if (
        !this.data.scene_graph_buffer ||
        this.data.scene_graph_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.scene_graph_buffer = Buffer.create({
          name: scene_graph_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(
          scene_graph_event,
          this.data.scene_graph_buffer,
        );
      } else {
        this.data.scene_graph_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(scene_graph_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers() {}

  static copy_entity_instance(to_index, from_index) {
    this.data.parent[to_index * 1 + 0] = this.data.parent[from_index * 1 + 0];

    this.data.gpu_data_dirty = true;
  }
}

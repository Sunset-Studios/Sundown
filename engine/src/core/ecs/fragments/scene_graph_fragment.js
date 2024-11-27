import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Tree } from "../../../memory/container.js";

const scene_graph_buffer_name = "scene_graph_buffer";
const scene_graph_cpu_buffer_name = "scene_graph_cpu_buffer";
const scene_graph_event = "scene_graph";
const scene_graph_update_event = "scene_graph_update";

class SceneGraphDataView {
  current_entity = -1;

  constructor() {}

  get parent() {
    return SceneGraphFragment.data.parent[this.current_entity];
  }

  set parent(value) {
    SceneGraphFragment.data.parent[this.current_entity] = value ?? -1;
    SceneGraphFragment.data.scene_graph.add(value ?? null, this.current_entity);
    if (SceneGraphFragment.data.dirty) {
      SceneGraphFragment.data.dirty[this.current_entity] = 1;
    }
    SceneGraphFragment.data.gpu_data_dirty = true;
  }

  get children() {
    return (
      SceneGraphFragment.data.scene_graph.find_node(this.current_entity)
        ?.children ?? []
    );
  }

  set children(value) {
    if (Array.isArray(value)) {
      SceneGraphFragment.data.scene_graph.add_multiple(
        this.current_entity,
        value,
        true /* replace_children */,
      );
    }
    if (SceneGraphFragment.data.dirty) {
      SceneGraphFragment.data.dirty[this.current_entity] = 1;
    }
    SceneGraphFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity) {
    this.current_entity = entity;

    return this;
  }
}

export class SceneGraphFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, SceneGraphDataView);

  static initialize() {
    this.data = {
      parent: new Int32Array(1),
      scene_graph: new Tree(),
      scene_graph_layer_counts: [],
      scene_graph_uniforms: [],
      scene_graph_buffer: null,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

    Fragment.resize_array(this.data, "parent", new_size, Int32Array, 1);

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static add_entity(entity) {
    super.add_entity(entity);
    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.data.parent[entity] = -1;
    this.data.scene_graph.remove(entity);
    this.data.gpu_data_dirty = true;
  }

  static get_entity_data(entity) {
    const data_view = this.data_view_allocator.allocate();
    data_view.fragment = this;
    data_view.view_entity(entity);
    return data_view;
  }

  static duplicate_entity_data(entity) {
    const node = this.data.scene_graph.find_node(entity);
    return {
      parent: node?.parent ?? null,
      children: node?.children ?? [],
    };
  }

  static to_gpu_data(context) {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        scene_graph_buffer: this.data.scene_graph_buffer,
      };
    }

    this.rebuild_buffers(context);

    return {
      scene_graph_buffer: this.data.scene_graph_buffer,
    };
  }

  static rebuild_buffers(context) {
    {
      const { result, layer_counts } =
        this.data.scene_graph.flatten(Int32Array);
      this.data.scene_graph_layer_counts = layer_counts;

      const num_elements = result?.length ?? 0;
      const gpu_data = new Int32Array(Math.max(num_elements * 2, 2));
      for (let i = 0; i < num_elements; ++i) {
        gpu_data[i * 2] = result[i];
        gpu_data[i * 2 + 1] = this.data.parent[result[i]];
      }

      this.data.scene_graph_uniforms = new Array(layer_counts.length);
      let layer_offset = 0;
      for (let i = 0; i < layer_counts.length; ++i) {
        this.data.scene_graph_uniforms[i] = Buffer.create(context, {
          name: "scene_graph_uniforms_" + i,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          raw_data: new Uint32Array([layer_counts[i], layer_offset]),
          force: true,
        });
        layer_offset += layer_counts[i];
      }

      if (
        !this.data.scene_graph_buffer ||
        this.data.scene_graph_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.scene_graph_buffer = Buffer.create(context, {
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
        this.data.scene_graph_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch(scene_graph_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers(context) {}
}

import { Renderer } from "../../renderer/renderer.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { ComputeTaskQueue } from "../../renderer/compute_task_queue.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { SceneGraphFragment } from "../ecs/fragments/scene_graph_fragment.js";
import { EntityTransformFlags } from "../minimal.js";
import { profile_scope } from "../../utility/performance.js";

const unmapped_state = "unmapped";
const transform_processor_update_scope_name = "TransformProcessor.update";
const transform_processing_task_name = "transform_processing";
const transform_processing_wgsl_path = "system_compute/transform_processing.wgsl";
const copy_position_rotation_scale_to_buffer_name = "copy_position_rotation_scale_to_buffer";
const reset_transform_dirty_flag_name = "reset_transform_dirty_flag";

export class TransformProcessor extends SimulationLayer {
  entity_query = null;
  transform_processing_input_lists = [];
  transform_processing_output_lists = [];

  init() {
    this.entity_query = EntityManager.create_query({
      fragment_requirements: [TransformFragment],
    });
    this.on_post_render_callback = this._on_post_render.bind(this);
    this.on_pre_render_callback = this._on_pre_render.bind(this);
    this._update_internal = this._update_internal.bind(this);
  }

  update(delta_time) {
    profile_scope(transform_processor_update_scope_name, this._update_internal);
  }

  _update_internal() {
    const transforms = EntityManager.get_fragment_array(TransformFragment);
    if (!transforms || transforms.flags.length === 0) {
      return;
    }

    const scene_graph = EntityManager.get_fragment_array(SceneGraphFragment);
    if (!scene_graph) {
      return;
    }

    for (let i = 0; i < scene_graph.scene_graph_layer_counts.length; ++i) {
      if (this.transform_processing_input_lists.length <= i) {
        this.transform_processing_input_lists.push(new Array(9));
        this.transform_processing_output_lists.push(new Array(5));
      }

      this.transform_processing_input_lists[i][0] = transforms.position_buffer;
      this.transform_processing_input_lists[i][1] = transforms.rotation_buffer;
      this.transform_processing_input_lists[i][2] = transforms.scale_buffer;
      this.transform_processing_input_lists[i][3] = transforms.flags_buffer;
      this.transform_processing_input_lists[i][4] = transforms.transforms_buffer;
      this.transform_processing_input_lists[i][5] = scene_graph.scene_graph_buffer;
      this.transform_processing_input_lists[i][6] = scene_graph.scene_graph_uniforms[i];

      this.transform_processing_output_lists[i][0] = transforms.position_buffer;
      this.transform_processing_output_lists[i][1] = transforms.rotation_buffer;
      this.transform_processing_output_lists[i][2] = transforms.scale_buffer;
      this.transform_processing_output_lists[i][3] = transforms.flags_buffer;
      this.transform_processing_output_lists[i][4] = transforms.transforms_buffer;

      ComputeTaskQueue.get().new_task(
        transform_processing_task_name + i,
        transform_processing_wgsl_path,
        this.transform_processing_input_lists[i],
        this.transform_processing_output_lists[i],
        Math.floor((scene_graph.scene_graph_layer_counts[i] + 255) / 256)
      );
    }

    Renderer.get().enqueue_pre_commands(
      reset_transform_dirty_flag_name,
      this.on_pre_render_callback
    );

    Renderer.get().enqueue_post_commands(
      copy_position_rotation_scale_to_buffer_name,
      this.on_post_render_callback
    );
  }

  _on_pre_render(graph, frame_data, encoder) {
    const transforms = EntityManager.get_fragment_array(TransformFragment);
    if (!transforms) {
      return;
    }

    for (let i = 0; i < transforms.flags.length; i++) {
      if ((transforms.flags[i] & EntityTransformFlags.TRANSFORM_DIRTY) !== 0) {
        transforms.flags[i] &= ~EntityTransformFlags.TRANSFORM_DIRTY;
        transforms.gpu_data_dirty = true;
      }
    }
  }

  _on_post_render(graph, frame_data, encoder) {
    const transforms = EntityManager.get_fragment_array(TransformFragment);
    if (!transforms) {
      return;
    }

    if (transforms.position_cpu_buffer.buffer.mapState === unmapped_state) {
      transforms.position_buffer.copy_buffer(encoder, 0, transforms.position_cpu_buffer);
    }

    if (transforms.rotation_cpu_buffer.buffer.mapState === unmapped_state) {
      transforms.rotation_buffer.copy_buffer(encoder, 0, transforms.rotation_cpu_buffer);
    }

    if (transforms.scale_cpu_buffer.buffer.mapState === unmapped_state) {
      transforms.scale_buffer.copy_buffer(encoder, 0, transforms.scale_cpu_buffer);
    }

    if (transforms.flags_cpu_buffer.buffer.mapState === unmapped_state) {
      transforms.flags_buffer.copy_buffer(encoder, 0, transforms.flags_cpu_buffer);
    }

    if (transforms.transforms_cpu_buffer.buffer.mapState === unmapped_state) {
      transforms.transforms_buffer.copy_buffer(encoder, 0, transforms.transforms_cpu_buffer);
    }
  }
}

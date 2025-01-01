import { Renderer } from "../../renderer/renderer.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { ComputeTaskQueue } from "../../renderer/compute_task_queue.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { SceneGraphFragment } from "../ecs/fragments/scene_graph_fragment.js";
import { profile_scope } from "../../utility/performance.js";

const unmapped_state = "unmapped";
const transform_processor_update_scope_name = "TransformProcessor.update";
const transform_processing_task_name = "transform_processing";
const transform_processing_wgsl_path = "system_compute/transform_processing.wgsl";
const copy_position_rotation_scale_to_buffer_name = "copy_position_rotation_scale_to_buffer";

export class TransformProcessor extends SimulationLayer {
  entity_query = null;
  transform_processing_input_lists = [];
  transform_processing_output_lists = [];

  init() {
    this.entity_query = EntityManager.get().create_query({
      fragment_requirements: [TransformFragment],
    });
    this.on_post_render_callback = this._on_post_render.bind(this);
  }

  update(delta_time) {
    profile_scope(transform_processor_update_scope_name, () => {
      const transforms = EntityManager.get().get_fragment_array(TransformFragment);
      if (!transforms || transforms.dirty.length === 0) {
        return;
      }

      const scene_graph = EntityManager.get().get_fragment_array(SceneGraphFragment);
      if (!scene_graph) {
        return;
      }

      for (let i = 0; i < scene_graph.scene_graph_layer_counts.length; ++i) {
        if (this.transform_processing_input_lists.length <= i) {
          this.transform_processing_input_lists.push(new Array(9));
          this.transform_processing_output_lists.push(new Array(4));
        }

        this.transform_processing_input_lists[i][0] = transforms.position_buffer;
        this.transform_processing_input_lists[i][1] = transforms.rotation_buffer;
        this.transform_processing_input_lists[i][2] = transforms.scale_buffer;
        this.transform_processing_input_lists[i][3] = transforms.dirty_flags_buffer;
        this.transform_processing_input_lists[i][4] = transforms.transforms_buffer;
        this.transform_processing_input_lists[i][5] = transforms.inverse_transforms_buffer;
        this.transform_processing_input_lists[i][6] = transforms.bounds_data_buffer;
        this.transform_processing_input_lists[i][7] = scene_graph.scene_graph_buffer;
        this.transform_processing_input_lists[i][8] = scene_graph.scene_graph_uniforms[i];

        this.transform_processing_output_lists[i][0] = transforms.position_buffer;
        this.transform_processing_output_lists[i][1] = transforms.rotation_buffer;
        this.transform_processing_output_lists[i][2] = transforms.scale_buffer;
        this.transform_processing_output_lists[i][3] = transforms.bounds_data_buffer;

        ComputeTaskQueue.get().new_task(
          transform_processing_task_name + i,
          transform_processing_wgsl_path,
          this.transform_processing_input_lists[i],
          this.transform_processing_output_lists[i],
          Math.floor((scene_graph.scene_graph_layer_counts[i] + 255) / 256)
        );
      }
      
      Renderer.get().enqueue_post_commands(
        copy_position_rotation_scale_to_buffer_name,
        this._on_post_render
      );
    });
  }

  _on_post_render(graph, frame_data, encoder) {
    const transforms = EntityManager.get().get_fragment_array(TransformFragment);
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
  }
}

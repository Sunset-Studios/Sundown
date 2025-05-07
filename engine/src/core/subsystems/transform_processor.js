import { Renderer } from "../../renderer/renderer.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { FragmentGpuBuffer } from "../ecs/solar/memory.js";
import { ComputeTaskQueue } from "../../renderer/compute_task_queue.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { SceneGraph } from "../scene_graph.js";
import { profile_scope } from "../../utility/performance.js";

const unmapped_state = "unmapped";

const position_buffer_name = "position";
const rotation_buffer_name = "rotation";
const scale_buffer_name = "scale";
const transforms_buffer_name = "transforms";

const transform_processor_update_scope_name = "TransformProcessor.update";
const transform_processing_task_name = "transform_processing";
const transform_processing_wgsl_path = "system_compute/transform_processing.wgsl";
const copy_position_rotation_scale_to_buffer_name = "copy_position_rotation_scale_to_buffer";

export class TransformProcessor extends SimulationLayer {
  transform_processing_input_lists = [];
  transform_processing_output_lists = [];

  init() {
    this._update_internal = this._update_internal.bind(this);
  }

  update(delta_time) {
    super.update(delta_time);
    profile_scope(transform_processor_update_scope_name, this._update_internal);
  }

  _update_internal() {
    const positions = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      position_buffer_name
    );
    const rotations = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      rotation_buffer_name
    );
    const scales = EntityManager.get_fragment_gpu_buffer(TransformFragment, scale_buffer_name);
    const transforms = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      transforms_buffer_name
    );
    const flags = FragmentGpuBuffer.entity_flags_buffer;

    for (let i = 0; i < SceneGraph.scene_graph_layer_counts.length; ++i) {
      if (this.transform_processing_input_lists.length <= i) {
        this.transform_processing_input_lists.push(new Array(9));
        this.transform_processing_output_lists.push(new Array(5));
      }

      this.transform_processing_input_lists[i][0] = positions.buffer;
      this.transform_processing_input_lists[i][1] = rotations.buffer;
      this.transform_processing_input_lists[i][2] = scales.buffer;
      this.transform_processing_input_lists[i][5] = transforms.buffer;
      this.transform_processing_input_lists[i][6] = SceneGraph.scene_graph_buffer;
      this.transform_processing_input_lists[i][7] = SceneGraph.scene_graph_uniforms[i];

      this.transform_processing_output_lists[i][0] = positions.buffer;
      this.transform_processing_output_lists[i][1] = rotations.buffer;
      this.transform_processing_output_lists[i][2] = scales.buffer;
      this.transform_processing_output_lists[i][3] = flags.buffer;

      ComputeTaskQueue.get().new_task(
        transform_processing_task_name + i,
        transform_processing_wgsl_path,
        this.transform_processing_input_lists[i],
        this.transform_processing_output_lists[i],
        Math.max(1, Math.floor((SceneGraph.scene_graph_layer_counts[i] + 255) / 256))
      );
    }
  }
}

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
const world_position_buffer_name = "world_position";
const world_rotation_buffer_name = "world_rotation";
const world_scale_buffer_name = "world_scale";

const transform_processor_update_scope_name = "TransformProcessor.update";
const transform_processing_task_name = "transform_processing";
const transform_processing_wgsl_path = "system_compute/transform_processing.wgsl";
const decompose_transform_task_name = "decompose_transform";
const decompose_transform_wgsl_path = "system_compute/decompose_transform.wgsl";

export class TransformProcessor extends SimulationLayer {
  transform_processing_input_lists = [];
  transform_processing_output_lists = [];
  decompose_transform_input_lists = [];
  decompose_transform_output_lists = [];

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

    // Get buffers for new world-space components
    const world_positions = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      world_position_buffer_name
    );
    const world_rotations = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      world_rotation_buffer_name
    );
    const world_scales = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      world_scale_buffer_name
    );

    for (let i = 0; i < SceneGraph.scene_graph_layer_counts.length; ++i) {
      if (this.transform_processing_input_lists.length <= i) {
        this.transform_processing_input_lists.push(new Array(7));
        this.transform_processing_output_lists.push(new Array(2));
        this.decompose_transform_input_lists.push(new Array(4));
        this.decompose_transform_output_lists.push(new Array(3));
      }

      this.transform_processing_input_lists[i][0] = positions.buffer;
      this.transform_processing_input_lists[i][1] = rotations.buffer;
      this.transform_processing_input_lists[i][2] = scales.buffer;
      this.transform_processing_input_lists[i][3] = transforms.buffer;
      this.transform_processing_input_lists[i][4] = flags.buffer;
      this.transform_processing_input_lists[i][5] = SceneGraph.scene_graph_buffer;
      this.transform_processing_input_lists[i][6] = SceneGraph.scene_graph_uniforms[i];

      this.transform_processing_output_lists[i][0] = transforms.buffer;
      this.transform_processing_output_lists[i][1] = flags.buffer;

      this.decompose_transform_input_lists[i][0] = transforms.buffer;
      this.decompose_transform_input_lists[i][1] = flags.buffer;
      this.decompose_transform_input_lists[i][2] = SceneGraph.scene_graph_buffer;
      this.decompose_transform_input_lists[i][3] = SceneGraph.scene_graph_uniforms[i];
      this.decompose_transform_input_lists[i][4] = world_positions.buffer;
      this.decompose_transform_input_lists[i][5] = world_rotations.buffer;
      this.decompose_transform_input_lists[i][6] = world_scales.buffer;

      this.decompose_transform_output_lists[i][0] = world_positions.buffer;
      this.decompose_transform_output_lists[i][1] = world_rotations.buffer;
      this.decompose_transform_output_lists[i][2] = world_scales.buffer;

      const dispatch_count = Math.max(1, Math.floor((SceneGraph.scene_graph_layer_counts[i] + 255) / 256));

      ComputeTaskQueue.get().new_task(
        transform_processing_task_name + i,
        transform_processing_wgsl_path,
        this.transform_processing_input_lists[i],
        this.transform_processing_output_lists[i],
        dispatch_count
      );

      ComputeTaskQueue.get().new_task(
        decompose_transform_task_name + i,
        decompose_transform_wgsl_path,
        this.decompose_transform_input_lists[i],
        this.decompose_transform_output_lists[i],
        dispatch_count
      );
    }
  }
}

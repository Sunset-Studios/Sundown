import { Renderer } from "../../renderer/renderer.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { ComputeTaskQueue } from "../../renderer/compute_task_queue.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { profile_scope } from "../../utility/performance.js";

export class TransformProcessor extends SimulationLayer {
  entity_query = null;

  init() {
    this.entity_query = EntityManager.get().create_query({
      fragment_requirements: [
        TransformFragment
      ],
    });
  }

  update(delta_time) {
    profile_scope("transform_processor_update", () => {
      const transforms =
        EntityManager.get().get_fragment_array(TransformFragment);
      if (!transforms) {
        return;
      }

      const entity_count = transforms.dirty.length;
      if (entity_count === 0) {
        return;
      }

      ComputeTaskQueue.get().new_task(
        "transform_processing",
        "system_compute/transform_processing.wgsl",
        [
          transforms.position_buffer,
          transforms.rotation_buffer,
          transforms.scale_buffer,
          transforms.dirty_flags_buffer,
          transforms.transforms_buffer,
          transforms.inverse_transforms_buffer,
          transforms.bounds_data_buffer,
        ],
        [
          transforms.dirty_flags_buffer,
          transforms.transforms_buffer,
          transforms.inverse_transforms_buffer,
          transforms.bounds_data_buffer,
        ],
        Math.floor((entity_count + 255) / 256)
      );

      Renderer.get().enqueue_post_commands(
        "copy_position_rotation_scale_to_buffer",
        (graph, frame_data, encoder) => {
          if (transforms.position_cpu_buffer.buffer.mapState === "unmapped") {
            transforms.position_buffer.copy_buffer(
              encoder,
              0,
              transforms.position_cpu_buffer
            );
          }

          if (transforms.rotation_cpu_buffer.buffer.mapState === "unmapped") {
            transforms.rotation_buffer.copy_buffer(
              encoder,
              0,
              transforms.rotation_cpu_buffer
            );
          }

          if (transforms.scale_cpu_buffer.buffer.mapState === "unmapped") {
            transforms.scale_buffer.copy_buffer(
              encoder,
              0,
              transforms.scale_cpu_buffer
            );
          }
        }
      );
    });
  }
}

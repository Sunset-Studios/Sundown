import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { EntityMasks } from "../ecs/query.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { mat4, quat, vec4 } from "gl-matrix";
import { profile_scope } from "../../utility/performance.js";

export class TransformProcessor extends SimulationLayer {
  entity_query = null;

  init() {
    this.entity_query = EntityManager.get().create_query({
      fragment_requirements: [TransformFragment],
    });
  }

  update(delta_time) {
    profile_scope("transform_processor_update", () => {
      const transforms = EntityManager.get().get_fragment_array(TransformFragment);
      if (!transforms) {
        return;
      }

      for (let i = 0; i < this.entity_query.matching_entities.length; ++i) {
        const entity = this.entity_query.matching_entities[i];
        const entity_state = this.entity_query.entity_states[i];

        transforms.gpu_data_dirty |= (entity_state & EntityMasks.Removed);

        if (!transforms.dirty[entity]) {
          continue;
        }

        transforms.prev_world_transform.set(
          transforms.world_transform.subarray(entity * 16, entity * 16 + 16),
          entity * 16
        );

        const transform = mat4.fromRotationTranslationScale(
          mat4.create(),
          quat.fromEuler(
            quat.create(),
            transforms.rotation.x[entity],
            transforms.rotation.y[entity],
            transforms.rotation.z[entity]
          ),
          vec4.fromValues(
            transforms.position.x[entity],
            transforms.position.y[entity],
            transforms.position.z[entity],
            1
          ),
          vec4.fromValues(
            transforms.scale.x[entity],
            transforms.scale.y[entity],
            transforms.scale.z[entity],
            0
          )
        );
        const inverse_transform = mat4.invert(mat4.create(), transform) ?? mat4.create();

        transforms.world_transform.set(transform, entity * 16);

        transforms.inverse_world_transform.set(
          inverse_transform,
          entity * 16
        );

        transforms.transpose_inverse_model_transform.set(
          mat4.transpose(mat4.create(), inverse_transform),
          entity * 16
        );

        transforms.dirty[entity] = 0;

        transforms.gpu_data_dirty = true;
      }
    });
  }
}

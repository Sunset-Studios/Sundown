import { SimulationLayer } from "@/core/simulation_layer.js";
import { EntityManager } from "@/core/ecs/entity.js";
import { TransformFragment } from "@/core/ecs/fragments/transform_fragment.js";
import { mat4, quat, vec4 } from "gl-matrix";

export class TransformProcessor extends SimulationLayer {
  entity_query = null;

  constructor() {
    super();
  }

  init(parent_context) {
    this.entity_query = EntityManager.get().create_query({
      fragment_requirements: [TransformFragment],
    });
  }

  update(delta_time, parent_context) {
    const transforms =
      EntityManager.get().get_fragment_array(TransformFragment);

    for (const entity of this.entity_query) {
      transforms.prev_world_transform.set(
        transforms.world_transform,
        entity * 16
      );

      if (transforms.dirty[entity] === 0) {
        continue;
      }

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

      transforms.world_transform.set(
        transform,
        entity * 16
      );

      transforms.inverse_world_transform.set(
        mat4.invert(
          mat4.create(),
          transform
        ),
        entity * 16
      );

      transforms.dirty[entity] = 0;
    }
  }
}

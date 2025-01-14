import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { profile_scope } from "../../utility/performance.js";

const entity_preprocessor_update_key = "entity_preprocessor_update";

export class EntityPreprocessor extends SimulationLayer {
  init() { }

  pre_update(delta_time) {
    super.pre_update(delta_time);
    profile_scope(entity_preprocessor_update_key, () => {
      EntityManager.refresh_entities();
      EntityManager.flush_instance_count_changes();
      EntityManager.rebuild_buffers();
      EntityManager.process_query_changes();
    });
  }
}


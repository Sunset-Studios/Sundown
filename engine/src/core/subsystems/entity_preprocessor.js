import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { profile_scope } from "../../utility/performance.js";

const entity_preprocessor_pre_update_key = "entity_preprocessor_pre_update";
const entity_preprocessor_post_update_key = "entity_preprocessor_post_update";

export class EntityPreprocessor extends SimulationLayer {
  init() {
    this._pre_update_internal = this._pre_update_internal.bind(this);
    this._post_update_internal = this._post_update_internal.bind(this);
  }

  pre_update(delta_time) {
    super.pre_update(delta_time);
    profile_scope(entity_preprocessor_pre_update_key, this._pre_update_internal);
  }

  _pre_update_internal() {
    EntityManager.flush_instance_count_changes();
    EntityManager.refresh_entities();
  }

  post_update(delta_time) {
    super.post_update(delta_time);
    profile_scope(entity_preprocessor_post_update_key, this._post_update_internal);
  }

  _post_update_internal() {
    EntityManager.process_query_changes();
    EntityManager.rebuild_buffers();
  }
}

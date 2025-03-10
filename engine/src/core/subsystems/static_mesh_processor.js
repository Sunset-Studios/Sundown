import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager, EntityID } from "../ecs/entity.js";
import { EntityMasks } from "../ecs/query.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { VisibilityFragment } from "../ecs/fragments/visibility_fragment.js";
import { MeshTaskQueue } from "../../renderer/mesh_task_queue.js";
import { Mesh } from "../../renderer/mesh.js";
import { profile_scope } from "../../utility/performance.js";

export class StaticMeshProcessor extends SimulationLayer {
  entity_query = null;

  constructor() {
    super();
  }

  init() {
    this.entity_query = EntityManager.create_query({
      fragment_requirements: [StaticMeshFragment, VisibilityFragment],
    });
    this._update_internal = this._update_internal.bind(this);
  }

  update(delta_time) {
    profile_scope("static_mesh_processor_update", this._update_internal);
  }

  _update_internal() {
    const static_meshes = EntityManager.get_fragment_array(StaticMeshFragment);
    const visibilities = EntityManager.get_fragment_array(VisibilityFragment);

    if (!static_meshes || !visibilities) {
      return;
    }

    const mesh_task_queue = MeshTaskQueue.get();

    let needs_resort = false;

    const entity_states = this.entity_query.entity_states.get_data();
    const matching_entity_data = this.entity_query.matching_entities.get_data();
    const matching_entity_offset_data = this.entity_query.matching_entity_ids.get_data();
    const matching_entity_instance_counts =
      this.entity_query.matching_entity_instance_counts.get_data();

    for (let i = 0; i < this.entity_query.matching_entities.length; ++i) {
      const entity = matching_entity_data[i];
      const entity_state = entity_states[i];
      const entity_index = matching_entity_offset_data[i];

      if (entity_state & EntityMasks.Removed) {
        mesh_task_queue.remove(entity);
        needs_resort = true;
        continue;
      }

      if (!static_meshes.dirty[entity_index] && !visibilities.dirty[entity_index]) {
        continue;
      }

      const mesh_id = Number(static_meshes.mesh[entity_index]);
      if (Mesh.loading_meshes.has(mesh_id)) {
        continue;
      }

      const material_id = Number(
        static_meshes.material_slots[entity_index * StaticMeshFragment.material_slot_stride]
      );
      const instance_count = matching_entity_instance_counts[i] || 1;

      if (mesh_id && material_id && instance_count && visibilities.visible[entity_index]) {
        mesh_task_queue.new_task(mesh_id, entity, material_id, instance_count);
      }

      static_meshes.dirty[entity_index] = 0;
      visibilities.dirty[entity_index] = 0;

      needs_resort = true;
    }

    if (needs_resort) {
      mesh_task_queue.mark_needs_sort();
    }
  }
}

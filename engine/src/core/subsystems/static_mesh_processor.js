import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { EntityMasks } from "../ecs/query.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { MeshTaskQueue } from "../../renderer/mesh_task_queue.js";
import { profile_scope } from "../../utility/performance.js";

export class StaticMeshProcessor extends SimulationLayer {
  entity_query = null;

  constructor() {
    super();
  }

  init() {
    this.entity_query = EntityManager.get().create_query({
      fragment_requirements: [StaticMeshFragment],
    });
  }

  update(delta_time) {
    profile_scope("static_mesh_processor_update", () => {
      const static_meshes =
        EntityManager.get().get_fragment_array(StaticMeshFragment);
      if (!static_meshes) {
        return;
      }

      const mesh_task_queue = MeshTaskQueue.get();

      let needs_resort = false;
      for (let i = 0; i < this.entity_query.matching_entities.length; ++i) {
        const entity = this.entity_query.matching_entities[i];
        const entity_state = this.entity_query.entity_states[i];

        if (entity_state & EntityMasks.Removed) {
          mesh_task_queue.remove(entity);
          needs_resort = true;
          continue;
        }

        if (!static_meshes.dirty[entity]) {
          continue;
        }

        needs_resort = true;

        const mesh_id = Number(static_meshes.mesh[entity]);
        const material_id = Number(
          static_meshes.material_slots[
            entity * StaticMeshFragment.material_slot_stride
          ]
        );
        const instance_count =
          Number(static_meshes.instance_count[entity]) || 1;

        if (mesh_id && material_id && instance_count) {
          mesh_task_queue.new_task(
            mesh_id,
            entity,
            material_id,
            instance_count 
          );
        }      

        static_meshes.dirty[entity] = 0;
      }

      if (needs_resort) {
        mesh_task_queue.mark_needs_sort();
      }
    });
  }
}

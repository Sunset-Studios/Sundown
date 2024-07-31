import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { MeshTaskQueue } from "../../renderer/mesh_task_queue.js";
import { ResourceCache, CacheTypes } from "../../renderer/resource_cache.js";
import { profile_scope } from "../../utility/performance.js";

export class StaticMeshProcessor extends SimulationLayer {
  entity_query = null;

  constructor() {
    super();
  }

  init(parent_context) {
    this.entity_query = EntityManager.get().create_query({
      fragment_requirements: [StaticMeshFragment],
    });
  }

  update(delta_time, parent_context) {
    profile_scope("static_mesh_processor_update", () => {
      const static_meshes =
        EntityManager.get().get_fragment_array(StaticMeshFragment);

      const resource_cache = ResourceCache.get();
      const mesh_task_queue = MeshTaskQueue.get();

      let last_mesh_id = null;
      let last_material_id = null;
      let last_entity = null;
      let mesh = null;

      let running_instance_count = 0;
      let needs_resort = false;
      for (let i = 0; i < this.entity_query.matching_entities.length; ++i) {
        const entity = this.entity_query.matching_entities[i];

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

        if (!last_entity) {
          last_entity = entity;
        }

        if (mesh_id == last_mesh_id && material_id == last_material_id) {
          running_instance_count += instance_count;
        } else {
          if (mesh_id !== last_mesh_id) {
            mesh = resource_cache.fetch(CacheTypes.MESH, mesh_id);
          }
          mesh_task_queue.new_task(
            mesh_id,
            entity,
            material_id,
            instance_count 
          );

          running_instance_count = 0;
          last_entity = entity;
        }

        last_mesh_id = mesh_id;
        last_material_id = material_id;

        static_meshes.dirty[entity] = 0;
      }

      if (running_instance_count > 0) {
        mesh_task_queue.new_task(
          last_mesh_id,
          last_entity,
          last_material_id,
          running_instance_count
        );
      }

      if (needs_resort) {
        mesh_task_queue.mark_needs_sort();
      }
    });
  }
}

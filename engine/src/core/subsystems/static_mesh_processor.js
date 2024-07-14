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
        this.entity_query = EntityManager.get().create_query({ fragment_requirements: [StaticMeshFragment] });
    }

    update(delta_time, parent_context) {
        profile_scope("static_mesh_processor_update", () => {
          const static_meshes =
            EntityManager.get().get_fragment_array(StaticMeshFragment);

          const resource_cache = ResourceCache.get();
          const mesh_task_queue = MeshTaskQueue.get();

          let last_mesh_id = null;
          let mesh = null

          mesh_task_queue.reserve(this.entity_query.get_entity_count());

          for (const entity of this.entity_query) {
            const mesh_id = Number(static_meshes.mesh[entity]);
            if (mesh_id !== last_mesh_id) {
                mesh = resource_cache.fetch(CacheTypes.MESH, mesh_id);
                last_mesh_id = mesh_id;
            }
            mesh_task_queue.new_task(mesh_id, entity);
          }
        });
    }
}
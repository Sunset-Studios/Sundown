import { SimulationLayer } from "@/core/simulation_layer.js";
import { EntityManager } from "@/core/ecs/entity.js";
import { StaticMeshFragment } from "@/core/ecs/fragments/static_mesh_fragment.js";
import { MeshTaskQueue } from "@/renderer/mesh_task_queue.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

export class StaticMeshProcessor extends SimulationLayer {
    static_meshes = [];
    entity_query = null;

    constructor() {
        super();
    }

    init() {
        this.entity_query = EntityManager.get().create_query({ fragments_requirements: [StaticMeshFragment] });
    }

    update(delta_time) {
        const static_meshes = EntityManager.get().get_fragment_array(StaticMeshFragment);

        for (const entity of this.entity_query) {
            const mesh_id = static_meshes.mesh[entity];
            const mesh = ResourceCache.get().fetch(CacheTypes.Mesh, mesh_id);

            MeshTaskQueue.get().new_task(mesh);
        }
    }
}
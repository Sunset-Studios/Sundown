import { EntityFlags } from "../minimal.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
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
    this.entity_query = EntityManager.create_query([StaticMeshFragment, VisibilityFragment]);
    this._update_internal = this._update_internal.bind(this);
  }

  update(delta_time) {
    profile_scope("static_mesh_processor_update", this._update_internal);
  }

  _update_internal() {
    const mesh_task_queue = MeshTaskQueue.get();

    this.entity_query.for_each((chunk, slot, instance_count, archetype) => {
      const entity = EntityManager.get_entity_for(chunk, slot);
      const entity_flags = EntityManager.get_entity_flags(entity);

      if ((entity_flags & EntityFlags.PENDING_DELETE) !== 0) {
        mesh_task_queue.remove(entity);
        return;
      }

      if ((entity_flags & EntityFlags.DIRTY) === 0) {
        return;
      }

      const static_meshes = chunk.get_fragment_view(StaticMeshFragment);
      const visibilities = chunk.get_fragment_view(VisibilityFragment);

      const mesh_id = Number(static_meshes.mesh[slot]);
      if (Mesh.loading_meshes.has(mesh_id)) {
        return;
      }

      const material_id = Number(
        static_meshes.material_slots[slot * StaticMeshFragment.material_slot_stride]
      );

      mesh_task_queue.remove(entity);

      const total_instance_count = EntityManager.get_entity_instance_count(entity);
      if (mesh_id && material_id && total_instance_count && visibilities.visible[slot]) {
        mesh_task_queue.new_task(mesh_id, entity, material_id, total_instance_count);
      }

      chunk.mark_dirty();
    });
  }
}

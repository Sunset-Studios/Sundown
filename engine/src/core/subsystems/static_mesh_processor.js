import { EntityFlags } from "../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../ecs/solar/types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { VisibilityFragment } from "../ecs/fragments/visibility_fragment.js";
import { MeshTaskQueue } from "../../renderer/mesh_task_queue.js";
import { profile_scope } from "../../utility/performance.js";

export class StaticMeshProcessor extends SimulationLayer {
  entity_query = null;

  constructor() {
    super();
  }

  init() {
    this.entity_query = EntityManager.create_query([StaticMeshFragment, VisibilityFragment]);
    this._update_internal = this._update_internal.bind(this);
    this._update_internal_iter_chunk = this._update_internal_iter_chunk.bind(this);
    EntityManager.on_delete(this._on_delete.bind(this));
  }

  update(delta_time) {
    profile_scope("static_mesh_processor_update", this._update_internal);
  }

  _update_internal_iter_chunk(chunk, flags, counts, archetype) {
    const dirty_flag = EntityFlags.DIRTY;
    const is_alive_flag = EntityFlags.ALIVE;
    const static_meshes = chunk.get_fragment_view(StaticMeshFragment);
    const visibilities = chunk.get_fragment_view(VisibilityFragment);
    const material_slot_stride = StaticMeshFragment.material_slot_stride;

    let should_dirty_chunk = false;
    let slot = 0;
    while (slot < DEFAULT_CHUNK_CAPACITY) {
      const entity_flags = flags[slot];

      if ((flags[slot] & dirty_flag) === 0) {
        slot += counts[slot] || 1;
        continue;
      }

      if ((entity_flags & is_alive_flag) === 0) {
        slot += counts[slot] || 1;
        continue;
      }

      const mesh_id = Number(static_meshes.mesh[slot]);
      const material_id = Number(static_meshes.material_slots[slot * material_slot_stride]);

      const entity = EntityManager.get_entity_for(chunk, slot);

      if (mesh_id && material_id && entity.instance_count && visibilities.visible[slot]) {
        MeshTaskQueue.new_task(mesh_id, entity, material_id);
      }

      slot += counts[slot] || 1;

      should_dirty_chunk = true;
    }

    if (should_dirty_chunk) {
      chunk.mark_dirty();
    }
  }

  _update_internal() {
    this.entity_query.for_each_chunk(this._update_internal_iter_chunk);
  }

  _on_delete(entity) {
    MeshTaskQueue.remove(entity);
  }
}

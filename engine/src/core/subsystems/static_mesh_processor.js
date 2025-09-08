import { EntityFlags } from "../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../ecs/solar/types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { VisibilityFragment } from "../ecs/fragments/visibility_fragment.js";
import { MeshTaskQueue } from "../../renderer/mesh_task_queue.js";
import { profile_scope } from "../../utility/performance.js";
import { ResourceCache } from "../../renderer/resource_cache.js";
import { CacheTypes } from "../../renderer/renderer_types.js";

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
    const static_meshes = chunk.get_fragment_view(StaticMeshFragment);
    const visibilities = chunk.get_fragment_view(VisibilityFragment);
    const material_slot_stride = StaticMeshFragment.material_slot_stride;

    let should_dirty_chunk = false;
    let slot = 0;
    while (slot < DEFAULT_CHUNK_CAPACITY) {
      const entity_flags = flags[slot];

      if ((entity_flags & EntityFlags.ALIVE) === 0) {
        slot += counts[slot] || 1;
        continue;
      }

      const mesh_id = Number(static_meshes.mesh[slot]);
      const entity = EntityManager.get_entity_for(chunk, slot);
      const has_mesh_tasks = MeshTaskQueue.contains(entity);

      if (mesh_id && entity.instance_count && visibilities.visible[slot] && !has_mesh_tasks) {
        const mesh = ResourceCache.get().fetch(CacheTypes.MESH, mesh_id);
        const section_count = mesh?.sections?.length || 1;
        const first_mat = Number(static_meshes.material_slots[slot * material_slot_stride]);
        for (let si = 0; si < section_count; si++) {
          const section = mesh?.sections[si] ?? null;

          let material_id = Number(static_meshes.material_slots[slot * material_slot_stride + si]);

          if (section?.material_id) {
            static_meshes.material_slots[slot * material_slot_stride + si] = BigInt(
              section.material_id
            );
            material_id = section.material_id;
          } else if (first_mat) {
            static_meshes.material_slots[slot * material_slot_stride + si] = BigInt(first_mat);
            material_id = first_mat;
          }

          if (material_id) {
            MeshTaskQueue.new_task(mesh_id, entity, material_id, si);
          }
        }

        should_dirty_chunk = true;
      }

      slot += counts[slot] || 1;
    }

    if (should_dirty_chunk) {
      chunk.mark_dirty();
    }
  }

  _update_internal() {
    if (!MeshTaskQueue.has_dirty_meshes()) {
      return;
    }

    this.entity_query.for_each_chunk(this._update_internal_iter_chunk);

    MeshTaskQueue.mark_meshes_dirty(false);
  }

  _on_delete(entity) {
    MeshTaskQueue.remove(entity);
  }
}

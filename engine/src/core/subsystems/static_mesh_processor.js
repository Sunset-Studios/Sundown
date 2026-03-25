import { EntityFlags } from "../minimal.js";
import { DEFAULT_CHUNK_CAPACITY } from "../ecs/solar/types.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { MeshTaskQueue } from "../../renderer/mesh_task_queue.js";
import { profile_scope } from "../../utility/performance.js";
import { ResourceCache } from "../../renderer/resource_cache.js";
import { CacheTypes } from "../../renderer/renderer_types.js";
import { MaterialAllocationTable } from "../../renderer/material_allocation_table.js";
import { TypedVector } from "../../memory/container.js";

export class StaticMeshProcessor extends SimulationLayer {
  entity_query = null;

  constructor() {
    super();
  }

  init() {
    this.entity_query = EntityManager.create_query([StaticMeshFragment]);
    this._update_internal = this._update_internal.bind(this);
    this._update_internal_iter_chunk = this._update_internal_iter_chunk.bind(this);
    EntityManager.on_delete(this._on_delete.bind(this));
  }

  update(delta_time) {
    profile_scope("static_mesh_processor_update", this._update_internal);
  }

  #entity_materials = new TypedVector(256, -1, BigInt64Array);
  _update_internal_iter_chunk(chunk, flags, counts, archetype) {
    const static_meshes = chunk.get_fragment_view(StaticMeshFragment);
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

      if (mesh_id && entity.instance_count) {
        MeshTaskQueue.remove(entity);

        const mesh = ResourceCache.get().fetch(CacheTypes.MESH, mesh_id);
        const section_count = mesh?.sections?.length || 1;
        const first_mat = static_meshes.material_slots[slot * material_slot_stride];

        this.#entity_materials.clear();
        for (let si = 0; si < section_count; si++) {
          const section = mesh?.sections[si] ?? null;

          let material_id = static_meshes.material_slots[slot * material_slot_stride + si];

          if (!material_id && section?.material_id) {
            static_meshes.material_slots[slot * material_slot_stride + si] = BigInt(
              section.material_id
            );
            material_id = BigInt(section.material_id);
          } else if (!material_id && first_mat) {
            static_meshes.material_slots[slot * material_slot_stride + si] = first_mat;
            material_id = first_mat;
          }

          this.#entity_materials.push(material_id);

          if (material_id) {
            MeshTaskQueue.new_task(mesh_id, entity, Number(material_id), si);
          }
        }

        const palette_offset = MaterialAllocationTable.register(entity, this.#entity_materials);
        for (let c = 0; c < counts[slot]; c++) {
          if (slot + c < DEFAULT_CHUNK_CAPACITY) {
            static_meshes.material_table_offset[slot + c] = palette_offset;
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

    MaterialAllocationTable.reset();

    this.entity_query.for_each_chunk(this._update_internal_iter_chunk);

    MeshTaskQueue.mark_meshes_dirty(false);
  }

  _on_delete(entity) {
    MeshTaskQueue.remove(entity);

    const entity_instance_count = EntityManager.get_entity_instance_count(entity);
    for (let i = 0; i < entity_instance_count; i++) {
      const static_mesh_fragment = EntityManager.get_fragment(entity, StaticMeshFragment, i);
      const material_slots = static_mesh_fragment.material_slots;
      const material_offset = static_mesh_fragment.material_table_offset;
      MaterialAllocationTable.unregister(material_offset, material_slots.length);
    }
  }
}

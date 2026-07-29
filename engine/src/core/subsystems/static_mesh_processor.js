import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { RenderTaskQueue } from "../../renderer/task_queues/render_task_queue.js";
import { profile_scope } from "../../utility/performance.js";
import { ResourceCache } from "../../renderer/resource_cache.js";
import { CacheTypes } from "../../renderer/renderer_types.js";
import { MaterialAllocationTable } from "../../renderer/material_allocation_table.js";
import { TypedVector } from "../../memory/container.js";

export class StaticMeshProcessor extends SimulationLayer {
  #entity_materials = new TypedVector(256, -1, BigInt64Array);
  #fallback_entities = new Set();

  entity_query = null;

  constructor() {
    super();
  }

  init() {
    this.entity_query = EntityManager.create_query([StaticMeshFragment]);
    this._update_internal = this._update_internal.bind(this);
    this._update_internal_iter_dirty_entity = this._update_internal_iter_dirty_entity.bind(this);
    EntityManager.on_delete(this._on_delete.bind(this));
  }

  update(delta_time) {
    profile_scope("static_mesh_processor_update", this._update_internal);
  }

  _update_internal_iter_dirty_entity(chunk, slot) {
    const entity = EntityManager.get_entity_for(chunk, slot);
    if (this.#fallback_entities.has(entity)) {
      return;
    }

    this.#fallback_entities.add(entity);
    this._process_entity(entity);
  }

  _update_internal() {
    if (!RenderTaskQueue.has_dirty_meshes()) {
      return;
    }

    const dirty_entities = RenderTaskQueue.get_dirty_mesh_entities();
    if (dirty_entities.size > 0) {
      for (const entity of dirty_entities) {
        this._process_entity(entity);
      }
    } else {
      this.#fallback_entities.clear();
      this.entity_query.for_each(this._update_internal_iter_dirty_entity);
      this.#fallback_entities.clear();
    }

    RenderTaskQueue.mark_meshes_dirty(false);
  }

  _on_delete(entity) {
    RenderTaskQueue.remove(entity);
    RenderTaskQueue.untrack_entity_mesh(entity);
    MaterialAllocationTable.unregister(entity);
  }

  _process_entity(entity) {
    if (
      !entity ||
      !EntityManager.entity_exists(entity) ||
      !EntityManager.has_fragment(entity, StaticMeshFragment)
    ) {
      return;
    }

    RenderTaskQueue.remove(entity);

    const primary_segment = entity.segments?.[0];
    if (!primary_segment) {
      RenderTaskQueue.untrack_entity_mesh(entity);
      MaterialAllocationTable.unregister(entity);
      return;
    }

    const static_meshes = primary_segment.chunk.get_fragment_view(StaticMeshFragment);
    const material_slot_stride = StaticMeshFragment.material_slot_stride;
    const slot = primary_segment.slot;
    const mesh_id = Number(static_meshes.mesh[slot]);

    if (!mesh_id || !entity.instance_count) {
      RenderTaskQueue.untrack_entity_mesh(entity);
      MaterialAllocationTable.unregister(entity);
      this._set_entity_material_table_offset(entity, 0);
      return;
    }

    const mesh = ResourceCache.get().fetch(CacheTypes.MESH, mesh_id);
    const section_count = mesh?.sections?.length || 1;
    const first_mat = static_meshes.material_slots[slot * material_slot_stride];

    this.#entity_materials.clear();
    for (let si = 0; si < section_count; si++) {
      const section = mesh?.sections?.[si] ?? null;
      let material_id = static_meshes.material_slots[slot * material_slot_stride + si];

      if (!material_id && section?.material_id) {
        static_meshes.material_slots[slot * material_slot_stride + si] = BigInt(section.material_id);
        material_id = BigInt(section.material_id);
      } else if (!material_id && first_mat) {
        static_meshes.material_slots[slot * material_slot_stride + si] = first_mat;
        material_id = first_mat;
      }

      this.#entity_materials.push(material_id);

      if (material_id) {
        RenderTaskQueue.new_task(mesh_id, entity, Number(material_id), si);
      }
    }

    const palette_offset = MaterialAllocationTable.register(entity, this.#entity_materials);
    this._set_entity_material_table_offset(entity, palette_offset);

    RenderTaskQueue.track_entity_mesh(entity, mesh_id);
  }

  _set_entity_material_table_offset(entity, palette_offset) {
    for (let i = 0; i < entity.segments.length; i++) {
      const segment = entity.segments[i];
      const static_meshes = segment.chunk.get_fragment_view(StaticMeshFragment);
      let changed = false;

      for (let j = 0; j < segment.count; j++) {
        const row = segment.slot + j;
        if (static_meshes.material_table_offset[row] !== palette_offset) {
          static_meshes.material_table_offset[row] = palette_offset;
          changed = true;
        }
      }

      if (changed) {
        segment.chunk.mark_dirty("material_table_offset");
      }
    }
  }
}

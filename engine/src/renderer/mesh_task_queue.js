import { MAX_BUFFERED_FRAMES, EntityFlags } from "../core/minimal.js";
import { EntityID, DEFAULT_CHUNK_CAPACITY } from "../core/ecs/solar/types.js";
import { ResourceCache } from "./resource_cache.js";
import { Mesh } from "./mesh.js";
import { MeshData } from "./mesh_data.js";
import { Buffer } from "./buffer.js";
import { RandomAccessAllocator, Sparse2DRandomAccessAllocator } from "../memory/allocator.js";
import { profile_scope } from "../utility/performance.js";
import { CacheTypes, MaterialFamilyType, BindGroupType } from "./renderer_types.js";
import { EntityManager } from "../core/ecs/entity.js";
import { StaticMeshFragment } from "../core/ecs/fragments/static_mesh_fragment.js";
import { MaterialAllocationTable } from "./material_allocation_table.js";

const initial_buffer_size = 1024;
const max_frame_buffer_writes = 100000;
const invalid_u32 = 0xffffffff;

class IndirectDrawBatch {
  mesh_id = 0;
  section = 0;
  material_id = 0;
  entities = [];
  instance_count = 0;
  first_index = 0;
  index_count = 0;
  base_vertex = 0;
  base_instance = 0;
}

class ObjectInstanceEntry {
  constructor(batch_index, row_field) {
    this.batch_index = batch_index;
    this.row = row_field;
    this.mesh_id = 0;
    this.section = 0;
    this.meshlet_offset = 0;
    this.meshlet_count = 0;
    this.meshlet_group_offset = 0;
    this.meshlet_group_count = 0;
  }
}

class ObjectInstanceBuffer {
  object_instance_buffer = null;
  object_instance_data = null;
  current_object_instance_write_offset = 0;
  last_object_instance_count = 0;

  init() {
    profile_scope("init_object_instance_buffer", () => {
      this.object_instance_data = new Uint32Array(initial_buffer_size * 4);
      if (!this.object_instance_buffer) {
        this.object_instance_buffer = Buffer.create({
          name: "object_instance_buffer",
          raw_data: this.object_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
    });
  }

  update_buffers(object_instances, force_update = false) {
    profile_scope("update_object_instance_buffer", () => {
      const object_instance_entries_count = object_instances.length * 2;
      if (object_instance_entries_count !== this.last_object_instance_count || force_update) {
        this.last_object_instance_count = object_instance_entries_count;
        this.current_object_instance_write_offset = 0;
      }

      // Resize object instance buffer if needed
      const required_object_instance_size = object_instances.length * 2 * 4; // 2 uint32 per instance, 4 bytes per uint32
      if (this.object_instance_buffer.config.size < required_object_instance_size) {
        const new_object_instance_data = new Uint32Array(object_instances.length * 2 * 2);
        new_object_instance_data.set(this.object_instance_data);
        this.object_instance_data = new_object_instance_data;

        this.object_instance_buffer = Buffer.create({
          name: "object_instance_buffer",
          raw_data: this.object_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          force: true,
        });
      }

      profile_scope("write_object_instance_buffer", () => {
        // only write up to the last object-instance entries once
        const total_obj_entries = this.last_object_instance_count;
        if (
          total_obj_entries > 0 &&
          this.current_object_instance_write_offset < total_obj_entries * MAX_BUFFERED_FRAMES
        ) {
          const actual_write_offset = this.current_object_instance_write_offset % total_obj_entries;
          // Update object instance buffer
          const write_count_obj = Math.min(
            total_obj_entries - actual_write_offset,
            max_frame_buffer_writes * 2
          );
          if (write_count_obj > 0) {
            for (let i = actual_write_offset; i < actual_write_offset + write_count_obj; i += 2) {
              const offset = Math.floor(i / 2);
              this.object_instance_data[i] = object_instances[offset].batch_index;
              this.object_instance_data[i + 1] = object_instances[offset].row;
            }
            this.object_instance_buffer.write_raw(
              this.object_instance_data,
              actual_write_offset * 4,
              write_count_obj,
              actual_write_offset
            );
            this.current_object_instance_write_offset += write_count_obj;
          }
        }
      });
    });
  }

  destroy() {
    this.object_instance_buffer.destroy();
    this.object_instance_buffer = null;
    this.object_instance_data = null;
  }
}

class MeshletInstanceBuffer {
  meshlet_instance_buffer = null;
  meshlet_instance_data = null;
  current_meshlet_instance_write_offset = 0;
  last_meshlet_instance_count = 0;
  static entry_stride = 2;

  init() {
    profile_scope("init_meshlet_instance_buffer", () => {
      this.meshlet_instance_data = new Uint32Array(
        initial_buffer_size * MeshletInstanceBuffer.entry_stride
      );
      this.meshlet_instance_data.fill(invalid_u32);
      if (!this.meshlet_instance_buffer) {
        this.meshlet_instance_buffer = Buffer.create({
          name: "meshlet_instance_buffer",
          raw_data: this.meshlet_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
    });
  }

  rebuild_data(object_instances, meshlet_instance_count) {
    let entry_index = 0;
    for (let object_instance_index = 0; object_instance_index < object_instances.length; ++object_instance_index) {
      const object_instance = object_instances[object_instance_index];
      const meshlet_offset = object_instance.meshlet_offset;
      const meshlet_count = object_instance.meshlet_count;

      for (let meshlet_local = 0; meshlet_local < meshlet_count; ++meshlet_local) {
        const data_index = entry_index * MeshletInstanceBuffer.entry_stride;
        this.meshlet_instance_data[data_index + 0] = object_instance_index;
        this.meshlet_instance_data[data_index + 1] = meshlet_offset + meshlet_local;
        entry_index++;
      }
    }

    const total_words = meshlet_instance_count * MeshletInstanceBuffer.entry_stride;
    if (entry_index * MeshletInstanceBuffer.entry_stride < total_words) {
      this.meshlet_instance_data.fill(
        invalid_u32,
        entry_index * MeshletInstanceBuffer.entry_stride,
        total_words
      );
    }
  }

  update_buffers(object_instances, meshlet_instance_count, force_update = false) {
    profile_scope("update_meshlet_instance_buffer", () => {
      const entry_count = meshlet_instance_count;
      let needs_rebuild = force_update;
      if (entry_count !== this.last_meshlet_instance_count || force_update) {
        this.last_meshlet_instance_count = entry_count;
        needs_rebuild = true;
      }

      const required_size = entry_count * MeshletInstanceBuffer.entry_stride * 4;
      if (this.meshlet_instance_buffer.config.size < required_size) {
        const new_meshlet_instance_data = new Uint32Array(
          Math.max(entry_count, 1) * MeshletInstanceBuffer.entry_stride * 2
        );
        new_meshlet_instance_data.fill(invalid_u32);
        new_meshlet_instance_data.set(this.meshlet_instance_data);
        this.meshlet_instance_data = new_meshlet_instance_data;
        needs_rebuild = true;

        this.meshlet_instance_buffer = Buffer.create({
          name: "meshlet_instance_buffer",
          raw_data: this.meshlet_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          force: true,
        });
      }

      if (needs_rebuild) {
        this.rebuild_data(object_instances, meshlet_instance_count);
      }

      profile_scope("write_meshlet_instance_buffer", () => {
        const total_entries = this.last_meshlet_instance_count;
        if (
          total_entries > 0 &&
          this.current_meshlet_instance_write_offset < total_entries * MAX_BUFFERED_FRAMES
        ) {
          const actual_write_offset =
            this.current_meshlet_instance_write_offset % total_entries;
          const write_entry_count = Math.min(
            total_entries - actual_write_offset,
            max_frame_buffer_writes
          );
          if (write_entry_count > 0) {
            const data_offset = actual_write_offset * MeshletInstanceBuffer.entry_stride;
            const write_word_count = write_entry_count * MeshletInstanceBuffer.entry_stride;
            this.meshlet_instance_buffer.write_raw(
              this.meshlet_instance_data,
              data_offset * 4,
              write_word_count,
              data_offset
            );
            this.current_meshlet_instance_write_offset += write_entry_count;
          }
        }
      });
    });
  }

  destroy() {
    this.meshlet_instance_buffer.destroy();
    this.meshlet_instance_buffer = null;
    this.meshlet_instance_data = null;
  }
}

class IndirectDrawObject {
  view_index = 0;
  clipmap_index = 0;
  indirect_draw_buffer = null;
  indirect_draw_data = null;
  current_indirect_draw_write_offset = 0;
  last_indirect_draw_count = 0;

  init() {
    profile_scope("init_indirect_draw_object", () => {
      this.indirect_draw_data = new Uint32Array(initial_buffer_size * 5);

      const suffix = `_view_${this.view_index}_clipmap_${this.clipmap_index}`;
      if (!this.indirect_draw_buffer) {
        this.indirect_draw_buffer = Buffer.create({
          name: `indirect_draw_buffer${suffix}`,
          raw_data: this.indirect_draw_data,
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        });
      }
    });
  }

  // We assume this gets called once per frame
  update_buffers(batches, force_update = false) {
    profile_scope("update_indirect_buffers", () => {
      const suffix = `_view_${this.view_index}_clipmap_${this.clipmap_index}`;
      const indirect_draw_entries_count = batches.length * 5;
      if (indirect_draw_entries_count !== this.last_indirect_draw_count || force_update) {
        this.last_indirect_draw_count = indirect_draw_entries_count;
        this.current_indirect_draw_write_offset = 0;
      }

      // Resize indirect draw buffer if needed
      const required_indirect_draw_size = batches.length * 5 * 4; // 5 uint32 per batch, 4 bytes per uint32
      if (this.indirect_draw_buffer.config.size < required_indirect_draw_size) {
        const new_indirect_draw_data = new Uint32Array(batches.length * 5 * 2);
        new_indirect_draw_data.set(this.indirect_draw_data);
        this.indirect_draw_data = new_indirect_draw_data;

        this.indirect_draw_buffer = Buffer.create({
          name: `indirect_draw_buffer${suffix}`,
          raw_data: this.indirect_draw_data,
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          force: true,
        });
      }

      profile_scope("write_indirect_draw_buffer", () => {
        const total_indirect_entries = this.last_indirect_draw_count;
        if (
          total_indirect_entries > 0 &&
          this.current_indirect_draw_write_offset < total_indirect_entries * MAX_BUFFERED_FRAMES
        ) {
          const actual_write_offset =
            this.current_indirect_draw_write_offset % total_indirect_entries;
          // Update indirect draw buffer
          const write_count = Math.min(
            total_indirect_entries - actual_write_offset,
            max_frame_buffer_writes * 5
          );
          if (write_count > 0) {
            for (let i = actual_write_offset; i < actual_write_offset + write_count; i += 5) {
              const offset = Math.floor(i / 5);
              this.indirect_draw_data[i + 0] = batches[offset].index_count;
              this.indirect_draw_data[i + 1] = 0; // Regular draw uses 0 instance count until updated
              this.indirect_draw_data[i + 2] = 0; // Index buffer is global, so we don't need to offset it
              this.indirect_draw_data[i + 3] = batches[offset].base_vertex;
              this.indirect_draw_data[i + 4] = batches[offset].base_instance;
            }
            this.indirect_draw_buffer.write_raw(
              this.indirect_draw_data,
              actual_write_offset * 4,
              write_count,
              actual_write_offset
            );

            this.current_indirect_draw_write_offset += write_count;
            if (
              this.current_indirect_draw_write_offset >=
              total_indirect_entries * MAX_BUFFERED_FRAMES
            ) {
              this.current_indirect_draw_write_offset = 0;
            }
          }
        }
      });
    });
  }

  reset_instance_counts() {
    for (let i = 0; i < this.last_indirect_draw_count; i += 5) {
      this.indirect_draw_data[i + 1] = 0; // Reset instance count to 0
    }
    this.indirect_draw_buffer.write_raw(this.indirect_draw_data);
  }

  destroy() {
    this.indirect_draw_buffer.destroy();
    this.indirect_draw_buffer = null;
    this.indirect_draw_data = null;
  }
}

class MeshTask {
  mesh_id = null;
  entity = null;
  material_id = null;
  section = 0;

  static init(task, mesh_id, entity, material_id = null, section = 0) {
    task.mesh_id = mesh_id;
    task.entity = entity;
    task.material_id = material_id;
    task.section = section;
  }
}

export class MeshTaskQueue {
  static tasks = [];
  static batches = [];
  static object_instances = [];
  static material_buckets = [];
  static object_instance_buffer = new ObjectInstanceBuffer();
  static meshlet_instance_buffer = new MeshletInstanceBuffer();
  static indirect_draw_objects = new Sparse2DRandomAccessAllocator(16, 4, IndirectDrawObject); // Per-view indirect draw objects
  static tasks_allocator = new RandomAccessAllocator(256, MeshTask); // TODO: This can potentially use a TypedVector depending on how it's structured. May need to split the fields out.
  static object_instance_allocator = new RandomAccessAllocator(256, ObjectInstanceEntry); // TODO: This can potentially use a TypedVector depending on how it's structured. May need to split the fields out.
  static needs_sort = false;
  static initialized = false;
  static entity_task_map = new Map(); // new: Map<Entity, Map<"meshId:materialId", Task>>
  static static_mesh_query = null;
  static meshes_dirty = false;
  static total_meshlet_instances = 0;

  static mark_needs_sort() {
    this.needs_sort = true;
  }

  static reserve(num_tasks) {
    this.tasks.length = num_tasks;
  }

  static reset() {
    this.tasks.length = 0;
    this.tasks_allocator.reset();
  }

  static _get_task_key(mesh_id, section, material_id) {
    const a = BigInt(mesh_id);
    const b = BigInt(section);
    const c = BigInt(material_id ?? 0);
    const ab = a >= b ? a * a + a + b : b * b + a;
    return ab >= c ? ab * ab + ab + c : c * c + ab;
  }

  static new_task(mesh_id, entity, material_id = null, section = 0, resort = true) {
    const key = this._get_task_key(mesh_id, section, material_id);
    let tasks_for_entity = this.entity_task_map.get(entity);
    if (tasks_for_entity?.has(key)) {
      return tasks_for_entity.get(key);
    }

    // 2) otherwise allocate & enqueue a brand-new task
    const task = this.tasks_allocator.allocate();
    MeshTask.init(task, mesh_id, entity, material_id, section);
    this.tasks.push(task);

    // 3) record it in our per-entity map
    if (!tasks_for_entity) {
      tasks_for_entity = new Map();
      this.entity_task_map.set(entity, tasks_for_entity);
    }
    tasks_for_entity.set(key, task);

    if (resort) {
      this.needs_sort = true;
    }
    return task;
  }

  static add_material_bucket(material_id) {
    if (!this.material_buckets.includes(material_id)) {
      this.material_buckets.push(material_id);
    }
  }

  static sort_and_batch() {
    if (!this.initialized) {
      this.initialized = true;
      this.object_instance_buffer.init();
      this.meshlet_instance_buffer.init();
    }

    profile_scope("sort_and_batch", () => {
      if (this.needs_sort) {
        this.batches.length = 0;
        this.object_instances.length = 0;
        this.material_buckets.length = 0;
        this.total_meshlet_instances = 0;
        this.object_instance_allocator.reset();

        this.tasks.sort((a, b) => {
          let diff = a.material_id - b.material_id;
          if (diff !== 0) return diff;
          diff = a.mesh_id - b.mesh_id;
          if (diff !== 0) return diff;
          diff = a.section - b.section;
          return diff;
        });

        let last_batch = null;
        for (let i = 0; i < this.tasks.length; i++) {
          const task = this.tasks[i];

          const last_batch_matches =
            last_batch &&
            last_batch.mesh_id === task.mesh_id &&
            last_batch.section === task.section &&
            last_batch.material_id === task.material_id;

          if (!last_batch_matches) {
            const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);
            if (!mesh) {
              continue;
            }

            const batch = new IndirectDrawBatch();
            batch.mesh_id = task.mesh_id;
            batch.section = task.section;
            batch.material_id = task.material_id;

            batch.base_instance = last_batch
              ? last_batch.base_instance + last_batch.instance_count
              : 0;
            batch.instance_count = task.entity.instance_count;

            const section = mesh.sections?.[task.section] || {
              first_index: 0,
              index_count: mesh.index_count,
            };
            batch.first_index = section.first_index;
            batch.index_count = section.index_count;
            batch.base_vertex = mesh.vertex_buffer_offset;

            batch.entities.length = batch.instance_count;
            batch.entities.fill(task.entity);

            this.add_material_bucket(batch.material_id);

            last_batch = batch;

            this.batches.push(batch);
          } else {
            last_batch.instance_count += task.entity.instance_count;
            const start_index = last_batch.entities.length;
            const new_length = last_batch.entities.length + task.entity.instance_count;
            last_batch.entities.length = new_length;
            last_batch.entities.fill(task.entity, start_index, new_length);
          }
        }

        // Sort batches by material id
        this.batches.sort((a, b) => a.material_id - b.material_id);

        // Add object instances to the object instance buffer
        for (let i = 0; i < this.batches.length; i++) {
          const batch = this.batches[i];
          // skip scanning the same entity over & over
          const visited_entities = new Set();
          for (let j = 0; j < batch.entities.length; j++) {
            const entity = batch.entities[j];
            if (visited_entities.has(entity)) continue;
            visited_entities.add(entity);

            for (let k = 0, segs = entity.segments, n = segs.length; k < n; k++) {
              const seg = segs[k];
              const cidx = seg.chunk.chunk_index;
              const start = seg.slot;
              const mesh = ResourceCache.get().fetch(CacheTypes.MESH, batch.mesh_id);
              const meshlet_section = mesh?.meshlet_sections?.[batch.section] ?? {
                meshlet_offset: 0,
                meshlet_count: 0,
                meshlet_group_offset: 0,
                meshlet_group_count: 0,
              };
              for (let l = 0, cnt = seg.count; l < cnt; l++) {
                const entry = this.object_instance_allocator.allocate();
                entry.batch_index = i;
                entry.row = EntityID.make_row_field(start + l, cidx);
                entry.mesh_id = batch.mesh_id;
                entry.section = batch.section;
                entry.meshlet_offset = meshlet_section.meshlet_offset;
                entry.meshlet_count = meshlet_section.meshlet_count;
                entry.meshlet_group_offset = meshlet_section.meshlet_group_offset;
                entry.meshlet_group_count = meshlet_section.meshlet_group_count;
                this.object_instances.push(entry);
                this.total_meshlet_instances += meshlet_section.meshlet_count;
              }
            }
          }
        }

        // This ensures that we render transparent materials after opaque materials
        this.material_buckets.sort((a, b) => {
          const a_material = ResourceCache.get().fetch(CacheTypes.MATERIAL, a);
          const b_material = ResourceCache.get().fetch(CacheTypes.MATERIAL, b);
          return a_material.family - b_material.family;
        });

        // Clear out the indirect draw buffers if there are no batches or object instances
        if (this.object_instances.length <= 0 && this.object_instance_buffer.object_instance_data) {
          this.object_instance_buffer.object_instance_data.fill(0);
        }
        if (this.object_instances.length <= 0 && this.meshlet_instance_buffer.meshlet_instance_data) {
          this.meshlet_instance_buffer.meshlet_instance_data.fill(0);
        }
        for (let i = 0; i < this.indirect_draw_objects.x_capacity; i++) {
          for (let j = 0; j < this.indirect_draw_objects.y_capacity; j++) {
            const obj = this.indirect_draw_objects.get(i, j);
            if (obj && obj.indirect_draw_data) {
              obj.indirect_draw_data.fill(0);
            }
          }
        }
      }

      this.object_instance_buffer.update_buffers(this.object_instances, this.needs_sort);
      this.meshlet_instance_buffer.update_buffers(
        this.object_instances,
        this.total_meshlet_instances,
        this.needs_sort
      );

      for (let i = 0; i < this.indirect_draw_objects.x_capacity; i++) {
        for (let j = 0; j < this.indirect_draw_objects.y_capacity; j++) {
          const obj = this.indirect_draw_objects.get(i, j);
          if (obj && obj.indirect_draw_data) {
            obj.update_buffers(this.batches, this.needs_sort);
          }
        }
      }

      MaterialAllocationTable.upload_buffers();

      this.needs_sort = false;
    });
  }

  static contains(entity) {
    return this.entity_task_map.has(entity) && this.entity_task_map.get(entity).size > 0;
  }

  static remove(entity, resort = true) {
    const tasks_for_entity = this.entity_task_map.get(entity);
    if (!tasks_for_entity) return;

    // keep only those tasks whose key is NOT in tasks_for_entity
    this.tasks = this.tasks.filter((task) => {
      if (task.entity !== entity) return true;
      const key = this._get_task_key(task.mesh_id, task.section, task.material_id);
      return !tasks_for_entity.has(key);
    });

    this.entity_task_map.delete(entity);
    if (this.tasks.length === 0) {
      this.batches.length = 0;
      this.object_instances.length = 0;
      this.material_buckets.length = 0;
    }
    this.needs_sort |= resort;
  }

  /**
   * Mark .
   */
  static mark_meshes_dirty(dirty = true) {
    this.meshes_dirty = dirty;
  }

  /**
   * Check if the meshes are dirty.
   */
  static has_dirty_meshes() {
    return this.meshes_dirty;
  }

  /**
   * Invalidate a mesh by removing all tasks associated with it and letting them requeue in the static mesh processor
   */
  static invalidate_mesh(mesh_id) {
    if (!this.static_mesh_query) {
      this.static_mesh_query = EntityManager.create_query([StaticMeshFragment]);
    }
    this.static_mesh_query.for_each_chunk((chunk, flags, counts, archetype) => {
      const static_meshes = chunk.get_fragment_view(StaticMeshFragment);
      let slot = 0;
      while (slot < DEFAULT_CHUNK_CAPACITY) {
        const entity_flags = flags[slot];
        if ((entity_flags & EntityFlags.ALIVE) === 0) {
          slot += counts[slot] || 1;
          continue;
        }
        if (Number(static_meshes.mesh[slot]) === mesh_id) {
          MeshTaskQueue.mark_meshes_dirty();
        }
        slot += counts[slot] || 1;
      }
    });
    this.needs_sort = true;
  }

  /**
   * Get the object instance buffer.
   */
  static get_object_instance_buffer() {
    return this.object_instance_buffer.object_instance_buffer;
  }

  static get_meshlet_instance_buffer() {
    return this.meshlet_instance_buffer.meshlet_instance_buffer;
  }

  /**
   * Get the indirect draw buffer for a specific view.
   */
  static get_indirect_draw_buffer(view_index = 0, clipmap_index = 0) {
    return this.get_indirect_draw_object(view_index, clipmap_index).indirect_draw_buffer;
  }

  /**
   * Get the material buckets.
   */
  static get_material_buckets() {
    return this.material_buckets;
  }

  /**
   * Get the total number of draw calls.
   */
  static get_total_draw_count() {
    return this.object_instances.length;
  }

  /**
   * Get the maximum number of meshlet instances addressable this frame.
   */
  static get_total_meshlet_count() {
    return this.total_meshlet_instances;
  }

  /**
   * Get or create the IndirectDrawObject for a given view and clipmap index.
   */
  static get_indirect_draw_object(view_index = 0, clipmap_index = 0) {
    let obj = this.indirect_draw_objects.get(view_index, clipmap_index);
    if (!obj) {
      obj = this.allocate_view_data(view_index, clipmap_index);
    }
    return obj;
  }

  /**
   * Allocate view data for a given view index and clipmap index.
   */
  static allocate_view_data(view_index = 0, clipmap_index = 0) {
    const obj = this.indirect_draw_objects.allocate_at(view_index, clipmap_index);
    obj.view_index = view_index;
    obj.clipmap_index = clipmap_index;
    obj.init();
    this.needs_sort = true;
    return obj;
  }

  /**
   * Deallocate view data for a given view index and clipmap index.
   */
  static deallocate_view_data(view_index, clipmap_index = 0) {
    const obj = this.indirect_draw_objects.get(view_index, clipmap_index);
    if (obj) {
      obj.destroy();
      this.indirect_draw_objects.deallocate_at(view_index, clipmap_index);
    }
  }

  static submit_draws(render_pass, rg_frame_data, should_reset = false) {
    let last_material = null;
    this.tasks.forEach((task) => {
      const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);

      const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, task.material_id);
      if (material && material !== last_material) {
        // Material binds will rebind a pipeline state, so we need to rebind the bind groups here
        material.bind(render_pass, render_pass.frame_bind_groups, render_pass.frame_attachments);
        if (render_pass.frame_bind_groups[BindGroupType.Global]) {
          render_pass.frame_bind_groups[BindGroupType.Global].bind(render_pass);
        }
        if (render_pass.frame_bind_groups[BindGroupType.Pass]) {
          render_pass.frame_bind_groups[BindGroupType.Pass].bind(render_pass);
        }
        last_material = material;
      }

      render_pass.pass.draw(
        mesh.vertex_count,
        task.entity.instance_count,
        mesh.vertex_buffer_offset
      );
    });
    if (should_reset) {
      this.reset();
    }
  }

  static submit_indexed_indirect_draws(
    render_pass,
    view_index = 0,
    clipmap_index = 0,
    skip_material_bind = true,
    opaque_only = false,
    depth_only = false,
    should_reset = false,
    indirect_draw_buffer = null
  ) {
    const index_buffer = MeshData.index_buffer;
    const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

    let last_material = null;
    let last_depth_only = false;

    const indirect_draw_object = this.get_indirect_draw_object(view_index, clipmap_index);
    const indirect_buffer = indirect_draw_buffer ?? indirect_draw_object.indirect_draw_buffer;

    for (let i = 0; i < this.batches.length; ++i) {
      const batch = this.batches[i];
      const mesh = ResourceCache.get().fetch(CacheTypes.MESH, batch.mesh_id);
      if (mesh.index_buffer_offset === -1) {
        continue;
      }

      const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, batch.material_id);
      if (opaque_only && material.family !== MaterialFamilyType.Opaque) {
        continue;
      }

      if (!skip_material_bind) {
        if (material && (material !== last_material || last_depth_only !== depth_only)) {
          // Material binds will rebind a pipeline state; choose depth or normal
          material.bind(
            render_pass,
            render_pass.frame_bind_groups,
            render_pass.frame_attachments,
            depth_only
          );
          if (render_pass.frame_bind_groups[BindGroupType.Global]) {
            render_pass.frame_bind_groups[BindGroupType.Global].bind(render_pass);
          }
          if (render_pass.frame_bind_groups[BindGroupType.Pass]) {
            render_pass.frame_bind_groups[BindGroupType.Pass].bind(render_pass);
          }
          last_material = material;
          last_depth_only = depth_only;
        }
      }

      render_pass.pass.setIndexBuffer(
        index_buffer.buffer,
        index_buffer.config.element_type,
        (mesh.index_buffer_offset + batch.first_index) * index_buffer_multiplier,
        batch.index_count * index_buffer_multiplier
      );
      render_pass.pass.drawIndexedIndirect(
        indirect_buffer.buffer,
        i * 20 // 5 * 4 bytes per draw call
      );
    }
    if (should_reset) {
      this.reset();
    }
  }

  static submit_material_indexed_indirect_draws(
    render_pass,
    material_id,
    view_index = 0,
    clipmap_index = 0,
    depth_only = false,
    indirect_draw_buffer = null,
    should_reset = false
  ) {
    const index_buffer = MeshData.index_buffer;
    const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

    const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, material_id);
    if (material) {
      // Material binds will rebind a pipeline state, so we need to rebind the bind groups here
      material.bind(
        render_pass,
        render_pass.frame_bind_groups,
        render_pass.frame_attachments,
        depth_only
      );
      if (render_pass.frame_bind_groups[BindGroupType.Global]) {
        render_pass.frame_bind_groups[BindGroupType.Global].bind(render_pass);
      }
      if (render_pass.frame_bind_groups[BindGroupType.Pass]) {
        render_pass.frame_bind_groups[BindGroupType.Pass].bind(render_pass);
      }
    }

    const indirect_draw_object = this.get_indirect_draw_object(view_index, clipmap_index);
    const indirect_buffer = indirect_draw_buffer ?? indirect_draw_object.indirect_draw_buffer;

    for (let i = 0; i < this.batches.length; ++i) {
      if (this.batches[i].material_id !== material_id) {
        continue;
      }
      const batch = this.batches[i];
      const mesh = ResourceCache.get().fetch(CacheTypes.MESH, batch.mesh_id);
      if (mesh.index_buffer_offset === -1) {
        continue;
      }

      render_pass.pass.setIndexBuffer(
        index_buffer.buffer,
        index_buffer.config.element_type,
        (mesh.index_buffer_offset + batch.first_index) * index_buffer_multiplier,
        batch.index_count * index_buffer_multiplier
      );
      render_pass.pass.drawIndexedIndirect(
        indirect_buffer.buffer,
        i * 20 // 5 * 4 bytes per draw call
      );
    }
    if (should_reset) {
      this.reset();
    }
  }

  static draw_quad(render_pass, instance_count = 1) {
    const index_buffer = MeshData.index_buffer;
    const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

    const mesh = Mesh.quad();
    render_pass.pass.setIndexBuffer(
      index_buffer.buffer,
      index_buffer.config.element_type,
      mesh.index_buffer_offset * index_buffer_multiplier,
      mesh.index_count * index_buffer_multiplier
    );
    render_pass.pass.drawIndexed(mesh.index_count, instance_count, 0, mesh.vertex_buffer_offset);
  }

  static draw_cube(render_pass, instance_count = 1) {
    const index_buffer = MeshData.index_buffer;
    const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

    const mesh = Mesh.cube();
    render_pass.pass.setIndexBuffer(
      index_buffer.buffer,
      index_buffer.config.element_type,
      mesh.index_buffer_offset * index_buffer_multiplier,
      mesh.index_count * index_buffer_multiplier
    );
    render_pass.pass.drawIndexed(mesh.index_count, instance_count, 0, mesh.vertex_buffer_offset);
  }

  static draw_sphere(render_pass, instance_count = 1) {
    const index_buffer = MeshData.index_buffer;
    const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

    const mesh = Mesh.sphere();
    render_pass.pass.setIndexBuffer(
      index_buffer.buffer,
      index_buffer.config.element_type,
      mesh.index_buffer_offset * index_buffer_multiplier,
      mesh.index_count * index_buffer_multiplier
    );
    render_pass.pass.drawIndexed(mesh.index_count, instance_count, 0, mesh.vertex_buffer_offset);
  }
}

import { MAX_BUFFERED_FRAMES } from "../core/minimal.js";
import { EntityID } from "../core/ecs/solar/types.js";
import { Buffer } from "./buffer.js";
import { RandomAccessAllocator, Sparse2DRandomAccessAllocator } from "../memory/allocator.js";
import { ResourceCache } from "./resource_cache.js";
import { MeshData } from "./mesh_data.js";
import { MaterialAllocationTable } from "./material_allocation_table.js";
import { profile_scope } from "../utility/performance.js";
import { BindGroupType, CacheTypes, MaterialFamilyType, MaterialPassType } from "./renderer_types.js";
import { Name } from "../utility/names.js";
import { draw_quad } from "./draw_helpers.js";

export const invalid_u32 = 0xffffffff;

const initial_buffer_size = 1024;
const max_frame_buffer_writes = 100000;

class IndirectDrawBatch {
  mesh_id = 0;
  section = 0;
  material_id = 0;
  visibility_bucket_id = invalid_u32;
  entities = [];
  instance_count = 0;
  first_index = 0;
  index_count = 0;
  base_vertex = 0;
  base_instance = 0;
}

class VisibilityShaderBucket {
  key = invalid_u32;
  shader = null;
  depth_shader = null;
  resolve_shader = null;
  template_name = "";
  representative_material_id = 0;
  family = MaterialFamilyType.Opaque;
  queue = null;
  queue_name = "";
}

class ObjectInstanceEntry {
  constructor(batch_index = 0, row_field = invalid_u32) {
    this.batch_index = batch_index;
    this.row = row_field;
    this.visibility_bucket_id = invalid_u32;
    this.mesh_id = 0;
    this.section = 0;
    this.meshlet_offset = 0;
    this.meshlet_count = 0;
    this.meshlet_group_offset = 0;
    this.meshlet_group_count = 0;
  }
}

/**
 * Describes a single renderable mesh submission before batching.
 *
 * Render task queues keep these records small and allocator-friendly so systems can
 * submit work freely during simulation while the renderer later coalesces compatible
 * tasks into GPU-friendly batches.
 */
export class MeshTask {
  mesh_id = null;
  entity = null;
  material_id = null;
  section = 0;
  visibility_bucket_id = invalid_u32;

  /**
   * Reinitializes an allocator-owned task without allocating a new object.
   *
   * Queues call this from their task allocators to keep hot submission paths stable
   * and to let subclasses swap in specialized task shapes with the same lifecycle.
   *
   * @param {MeshTask} task Allocator-owned task object to populate.
   * @param {number} mesh_id Mesh resource identifier.
   * @param {object} entity Entity/chunk view that owns the submitted instances.
   * @param {?number} material_id Material resource identifier for the section.
   * @param {number} section Mesh section to draw.
   * @param {number} visibility_bucket_id Bucket key used by visibility passes.
   */
  static init(task, mesh_id, entity, material_id = null, section = 0, visibility_bucket_id = invalid_u32) {
    task.mesh_id = mesh_id;
    task.entity = entity;
    task.material_id = material_id;
    task.section = section;
    task.visibility_bucket_id = visibility_bucket_id;
  }
}

/**
 * GPU storage for per-object instance metadata consumed by culling and visibility.
 *
 * The buffer is deliberately queue-local: each RenderTaskQueue owns its own object
 * instance stream, allowing the renderer to process many independent sources of
 * work without relying on a global "primary" mesh queue.
 */
export class ObjectInstanceBuffer {
  static entry_stride = 4;

  object_instance_buffer = null;
  object_instance_data = null;
  current_object_instance_write_offset = 0;
  last_object_instance_count = 0;

  /**
   * @param {string} name Stable GPU resource name used by the render graph.
   */
  constructor(name = "object_instance_buffer") {
    this.name = name;
  }

  /**
   * Allocates the initial GPU-side storage.
   *
   * This is separated from construction so queues can define static buffer members
   * cheaply and only materialize GPU resources once the queue is used.
   */
  init() {
    profile_scope("init_object_instance_buffer", () => {
      this.object_instance_data = new Uint32Array(
        initial_buffer_size * ObjectInstanceBuffer.entry_stride
      );
      if (!this.object_instance_buffer) {
        this.object_instance_buffer = Buffer.create({
          name: this.name,
          raw_data: this.object_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
    });
  }

  /**
   * Replaces the backing data wholesale when a custom queue has already packed it.
   *
   * This gives specialized queues a direct upload path while preserving the same
   * render graph contract as the default object-instance layout.
   *
   * @param {Uint32Array} data Packed object instance words.
   * @param {number} element_count Number of words to upload.
   */
  update_raw_data(data, element_count = data?.length ?? 0) {
    this.object_instance_data = data;
    this.object_instance_buffer = Buffer.create({
      name: this.name,
      raw_data: data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.object_instance_buffer.write_raw(data, 0, element_count, 0);
  }

  /**
   * Uploads queue object instances incrementally to avoid large per-frame stalls.
   *
   * The default queue rebuilds object instance entries on the CPU, then streams
   * bounded slices to the GPU across buffered frames.
   *
   * @param {Array<object>} object_instances CPU-side object instance entries.
   * @param {boolean} force_update Forces write cursors to restart after rebuilds.
   */
  update_buffers(object_instances, force_update = false) {
    profile_scope("update_object_instance_buffer", () => {
      const object_instance_entries_count =
        object_instances.length * ObjectInstanceBuffer.entry_stride;
      if (object_instance_entries_count !== this.last_object_instance_count || force_update) {
        this.last_object_instance_count = object_instance_entries_count;
        this.current_object_instance_write_offset = 0;
      }

      const required_object_instance_size =
        object_instances.length * ObjectInstanceBuffer.entry_stride * 4;
      if (this.object_instance_buffer.config.size < required_object_instance_size) {
        const new_object_instance_data = new Uint32Array(
          object_instances.length * ObjectInstanceBuffer.entry_stride * 2
        );
        new_object_instance_data.set(this.object_instance_data);
        this.object_instance_data = new_object_instance_data;

        this.object_instance_buffer = Buffer.create({
          name: this.name,
          raw_data: this.object_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          force: true,
        });
      }

      profile_scope("write_object_instance_buffer", () => {
        const total_obj_entries = this.last_object_instance_count;
        if (
          total_obj_entries > 0 &&
          this.current_object_instance_write_offset < total_obj_entries * MAX_BUFFERED_FRAMES
        ) {
          const actual_write_offset = this.current_object_instance_write_offset % total_obj_entries;
          const write_count = Math.min(
            total_obj_entries - actual_write_offset,
            max_frame_buffer_writes * ObjectInstanceBuffer.entry_stride
          );
          if (write_count > 0) {
            for (
              let i = actual_write_offset;
              i < actual_write_offset + write_count;
              i += ObjectInstanceBuffer.entry_stride
            ) {
              const offset = Math.floor(i / ObjectInstanceBuffer.entry_stride);
              this.object_instance_data[i] = object_instances[offset].batch_index;
              this.object_instance_data[i + 1] = object_instances[offset].row;
              this.object_instance_data[i + 2] = object_instances[offset].visibility_bucket_id;
              this.object_instance_data[i + 3] = 0;
            }
            this.object_instance_buffer.write_raw(
              this.object_instance_data,
              actual_write_offset * 4,
              write_count,
              actual_write_offset
            );
            this.current_object_instance_write_offset += write_count;
            if (
              this.current_object_instance_write_offset >=
              total_obj_entries * MAX_BUFFERED_FRAMES
            ) {
              this.current_object_instance_write_offset = 0;
            }
          }
        }
      });
    });
  }

  /**
   * Releases GPU storage when a queue or view lifecycle is torn down.
   */
  destroy() {
    this.object_instance_buffer.destroy();
    this.object_instance_buffer = null;
    this.object_instance_data = null;
  }
}

/**
 * GPU storage mapping visible meshlet instances back to object instances.
 *
 * Culling and visibility operate over meshlets, but material evaluation often needs
 * the originating object/entity data. This buffer is the compact bridge between
 * those two domains for a single render task queue.
 */
export class MeshletInstanceBuffer {
  static entry_stride = 2;

  meshlet_instance_buffer = null;
  meshlet_instance_data = null;
  current_meshlet_instance_write_offset = 0;
  last_meshlet_instance_count = 0;

  /**
   * @param {string} name Stable GPU resource name used by the render graph.
   */
  constructor(name = "meshlet_instance_buffer") {
    this.name = name;
  }

  /**
   * Allocates the initial meshlet-instance storage.
   */
  init() {
    profile_scope("init_meshlet_instance_buffer", () => {
      this.meshlet_instance_data = new Uint32Array(
        initial_buffer_size * MeshletInstanceBuffer.entry_stride
      );
      this.meshlet_instance_data.fill(invalid_u32);
      if (!this.meshlet_instance_buffer) {
        this.meshlet_instance_buffer = Buffer.create({
          name: this.name,
          raw_data: this.meshlet_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
    });
  }

  /**
   * Replaces the backing data for queues that produce custom meshlet streams.
   *
   * @param {Uint32Array} data Packed meshlet instance words.
   * @param {number} element_count Number of words to upload.
   */
  update_raw_data(data, element_count = data?.length ?? 0) {
    this.meshlet_instance_data = data;
    this.meshlet_instance_buffer = Buffer.create({
      name: this.name,
      raw_data: data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.meshlet_instance_buffer.write_raw(data, 0, element_count, 0);
  }

  /**
   * Rebuilds the default object-to-meshlet lookup table.
   *
   * Queues with ordinary meshlet-backed geometry can use this layout directly;
   * queues with procedural or instanced representations may override the upload path.
   *
   * @param {Array<object>} object_instances Queue object instances.
   * @param {number} meshlet_instance_count Total meshlet instances in the queue.
   */
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

  /**
   * Uploads the meshlet-instance stream used by culling and bucket compaction.
   *
   * @param {Array<object>} object_instances Queue object instances.
   * @param {number} meshlet_instance_count Total meshlet instances in the queue.
   * @param {boolean} force_update Forces write cursors to restart after rebuilds.
   */
  update_buffers(object_instances, meshlet_instance_count, force_update = false) {
    profile_scope("update_meshlet_instance_buffer", () => {
      const entry_count = meshlet_instance_count * MeshletInstanceBuffer.entry_stride;
      if (entry_count !== this.last_meshlet_instance_count || force_update) {
        this.last_meshlet_instance_count = entry_count;
        this.current_meshlet_instance_write_offset = 0;
      }

      const required_size = entry_count * MeshletInstanceBuffer.entry_stride * 4;
      if (this.meshlet_instance_buffer.config.size < required_size) {
        const new_meshlet_instance_data = new Uint32Array(
          Math.max(entry_count, 1) * MeshletInstanceBuffer.entry_stride * 2
        );
        new_meshlet_instance_data.fill(invalid_u32);
        new_meshlet_instance_data.set(this.meshlet_instance_data);
        this.meshlet_instance_data = new_meshlet_instance_data;

        this.meshlet_instance_buffer = Buffer.create({
          name: this.name,
          raw_data: this.meshlet_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          force: true,
        });
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
            max_frame_buffer_writes * MeshletInstanceBuffer.entry_stride
          );
          if (write_entry_count > 0) {
            this.rebuild_data(object_instances, meshlet_instance_count);

            this.meshlet_instance_buffer.write_raw(
              this.meshlet_instance_data,
              actual_write_offset * 4,
              write_entry_count,
              actual_write_offset
            );
            this.current_meshlet_instance_write_offset += write_entry_count;
            if (
              this.current_meshlet_instance_write_offset >=
              total_entries * MAX_BUFFERED_FRAMES
            ) {
              this.current_meshlet_instance_write_offset = 0;
            }
          }
        }
      });
    });
  }

  /**
   * Releases GPU storage when a queue or view lifecycle is torn down.
   */
  destroy() {
    this.meshlet_instance_buffer.destroy();
    this.meshlet_instance_buffer = null;
    this.meshlet_instance_data = null;
  }
}

/**
 * Per-view indirect draw storage for a render task queue.
 *
 * Visibility and culling write draw counts per view/clipmap, while render passes
 * consume those counts later. Keeping this object view-scoped prevents queues from
 * trampling each other's indirect arguments.
 */
export class IndirectDrawObject {
  view_index = 0;
  clipmap_index = 0;
  indirect_draw_buffer = null;
  indirect_draw_data = null;
  current_indirect_draw_write_offset = 0;
  last_indirect_draw_count = 0;

  /**
   * Creates the indirect draw buffer for this view/clipmap slot.
   *
   * @param {string} name_prefix Resource name prefix supplied by the owning queue.
   */
  init(name_prefix = "indirect_draw_buffer") {
    profile_scope("init_indirect_draw_object", () => {
      this.name_prefix = name_prefix;
      this.indirect_draw_data = new Uint32Array(initial_buffer_size * 5);

      const suffix = `_view_${this.view_index}_clipmap_${this.clipmap_index}`;
      if (!this.indirect_draw_buffer) {
        this.indirect_draw_buffer = Buffer.create({
          name: `${this.name_prefix}${suffix}`,
          raw_data: this.indirect_draw_data,
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        });
      }
    });
  }

  /**
   * Uploads indexed indirect draw arguments for the queue's current batches.
   *
   * The instance count is initialized to zero because GPU culling/compaction owns
   * the final visible instance count for each batch.
   *
   * @param {Array<object>} batches CPU-side draw batches.
   * @param {boolean} force_update Forces write cursors to restart after rebuilds.
   */
  update_buffers(batches, force_update = false) {
    profile_scope("update_indirect_buffers", () => {
      const suffix = `_view_${this.view_index}_clipmap_${this.clipmap_index}`;
      const indirect_draw_entries_count = batches.length * 5;
      if (indirect_draw_entries_count !== this.last_indirect_draw_count || force_update) {
        this.last_indirect_draw_count = indirect_draw_entries_count;
        this.current_indirect_draw_write_offset = 0;
      }

      const required_indirect_draw_size = batches.length * 5 * 4;
      if (this.indirect_draw_buffer.config.size < required_indirect_draw_size) {
        const new_indirect_draw_data = new Uint32Array(batches.length * 5 * 2);
        new_indirect_draw_data.set(this.indirect_draw_data);
        this.indirect_draw_data = new_indirect_draw_data;

        this.indirect_draw_buffer = Buffer.create({
          name: `${this.name_prefix}${suffix}`,
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
          const write_count = Math.min(
            total_indirect_entries - actual_write_offset,
            max_frame_buffer_writes * 5
          );
          if (write_count > 0) {
            for (let i = actual_write_offset; i < actual_write_offset + write_count; i += 5) {
              const offset = Math.floor(i / 5);
              this.indirect_draw_data[i + 0] = batches[offset].index_count;
              this.indirect_draw_data[i + 1] = 0;
              this.indirect_draw_data[i + 2] = 0;
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

  /**
   * Releases the view-local indirect draw buffer.
   */
  destroy() {
    this.indirect_draw_buffer.destroy();
    this.indirect_draw_buffer = null;
    this.indirect_draw_data = null;
  }
}

/**
 * Base class for renderer work queues that submit mesh-like tasks into visibility.
 *
 * A RenderTaskQueue is the shared contract between scene systems and renderer
 * strategies: systems submit tasks, queues sort/batch/upload their own GPU buffers,
 * and strategies iterate registered queues generically for culling, visibility,
 * resolve, shadows, and debug passes. Subclasses can override batching or draw
 * submission while still presenting the same queue-shaped interface.
 */
export class RenderTaskQueue {
  static queue_name = "render_task_queue";
  static tasks = [];
  static batches = [];
  static object_instances = [];
  static visibility_shader_buckets = [];
  static dirty_mesh_entities = new Set();
  static entity_mesh_map = new Map();
  static mesh_entity_map = new Map();
  static entity_task_map = new Map();

  static object_instance_buffer = new ObjectInstanceBuffer();
  static meshlet_instance_buffer = new MeshletInstanceBuffer();

  static indirect_draw_objects = new Sparse2DRandomAccessAllocator(16, 4, IndirectDrawObject);
  static tasks_allocator = new RandomAccessAllocator(256, MeshTask);
  static object_instance_allocator = new RandomAccessAllocator(256, ObjectInstanceEntry);
  static visibility_bucket_allocator = new RandomAccessAllocator(256, VisibilityShaderBucket);

  static task_type = MeshTask;
  static batch_type = IndirectDrawBatch;

  static needs_sort = false;
  static initialized = false;
  static meshes_dirty = false;
  static total_meshlet_instances = 0;

  /**
   * Marks CPU-side task state as needing a batch rebuild.
   *
   * Call this whenever task ordering, materials, mesh sections, or instance layout
   * changed in a way that invalidates the queue's current GPU buffers.
   */
  static mark_needs_sort() {
    this.needs_sort = true;
  }

  /**
   * Reserves task array capacity for callers that know their submission count.
   *
   * This is a small optimization hook for bulk loaders or procedural generators
   * that want to avoid repeated array growth during task submission.
   *
   * @param {number} num_tasks Expected number of submitted tasks.
   */
  static reserve(num_tasks) {
    this.tasks.length = num_tasks;
  }

  /**
   * Clears transient task submissions while keeping queue-owned buffers alive.
   *
   * This is useful for queues whose tasks are rebuilt every frame, while still
   * preserving allocator and GPU resource ownership.
   */
  static reset() {
    this.tasks.length = 0;
    this.tasks_allocator?.reset();
  }

  /**
   * Submits a renderable mesh section to this queue.
   *
   * Duplicate submissions for the same entity/mesh/material/section collapse to the
   * existing task, keeping processors idempotent across repeated component updates.
   *
   * @param {number} mesh_id Mesh resource identifier.
   * @param {object} entity Entity/chunk view that owns the submitted instances.
   * @param {?number} material_id Material resource identifier for this section.
   * @param {number} section Mesh section to draw.
   * @param {boolean} resort Whether this submission should invalidate batching.
   * @returns {MeshTask} Existing or newly allocated task.
   */
  static new_task(mesh_id, entity, material_id = null, section = 0, resort = true) {
    const visibility_bucket = this._get_visibility_bucket_config(material_id);
    const visibility_bucket_id = visibility_bucket?.id ?? invalid_u32;
    const key = this._get_task_key(mesh_id, section, material_id, visibility_bucket_id);
    let tasks_for_entity = this.entity_task_map.get(entity);
    if (tasks_for_entity?.has(key)) {
      return tasks_for_entity.get(key);
    }

    const task = this.tasks_allocator.allocate();
    this.task_type.init(task, mesh_id, entity, material_id, section, visibility_bucket_id);
    this.tasks.push(task);

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

  /**
   * Checks whether an entity currently contributes work to this queue.
   *
   * Processors use this to decide whether component changes need queue mutation.
   *
   * @param {object} entity Entity/chunk view to query.
   * @returns {boolean} True when the entity has one or more queued tasks.
   */
  static contains(entity) {
    return this.entity_task_map.has(entity) && this.entity_task_map.get(entity).size > 0;
  }

  /**
   * Removes all queued tasks owned by an entity.
   *
   * This keeps queue state aligned with ECS destruction or component removal and
   * optionally triggers a batch rebuild so stale instances disappear from GPU draws.
   *
   * @param {object} entity Entity/chunk view to remove.
   * @param {boolean} resort Whether removal should invalidate batching.
   */
  static remove(entity, resort = true) {
    const tasks_for_entity = this.entity_task_map.get(entity);
    if (!tasks_for_entity) return;

    // keep only those tasks whose key is NOT in tasks_for_entity
    this.tasks = this.tasks.filter((task) => {
      if (task.entity !== entity) return true;
      const key = this._get_task_key(
        task.mesh_id,
        task.section,
        task.material_id,
        task.visibility_bucket_id
      );
      return !tasks_for_entity.has(key);
    });

    this.entity_task_map.delete(entity);
    if (this.tasks.length === 0) {
      this.batches.length = 0;
      this.object_instances.length = 0;
      this.visibility_shader_buckets.length = 0;
    }
    this.needs_sort |= resort;
  }

  /**
   * Records that mesh-dependent queue data needs to be refreshed.
   *
   * Asset systems call this when mesh data changes so processors can resubmit or
   * rebuild only affected entities instead of flushing every queue blindly.
   *
   * @param {boolean} dirty Whether to mark or clear dirty state.
   * @param {?object} entity Optional entity to mark when known.
   */
  static mark_meshes_dirty(dirty = true, entity = null) {
    if (!dirty) {
      this.dirty_mesh_entities.clear();
      this.meshes_dirty = false;
      return;
    }

    this.meshes_dirty = true;
    if (entity) {
      this.dirty_mesh_entities.add(entity);
    } else {
      this.dirty_mesh_entities.clear();
    }
  }

  /**
   * Reports whether any tracked mesh/entity relationship is dirty.
   *
   * @returns {boolean} True when queue processors should inspect dirty mesh users.
   */
  static has_dirty_meshes() {
    return this.meshes_dirty;
  }

  /**
   * Returns the entities affected by dirty mesh resources.
   *
   * @returns {Set<object>} Entity set awaiting mesh-driven refresh.
   */
  static get_dirty_mesh_entities() {
    return this.dirty_mesh_entities;
  }

  /**
   * Tracks which mesh resource an entity depends on.
   *
   * This reverse lookup lets asset invalidation find exactly which queued entities
   * need refresh when a mesh is reprocessed or reloaded.
   *
   * @param {object} entity Entity/chunk view to track.
   * @param {number} mesh_id Mesh resource identifier.
   */
  static track_entity_mesh(entity, mesh_id) {
    const previous_mesh_id = this.entity_mesh_map.get(entity);
    if (previous_mesh_id === mesh_id) {
      return;
    }

    if (previous_mesh_id) {
      const previous_entities = this.mesh_entity_map.get(previous_mesh_id);
      previous_entities?.delete(entity);
      if (previous_entities?.size === 0) {
        this.mesh_entity_map.delete(previous_mesh_id);
      }
    }

    if (mesh_id) {
      let entities = this.mesh_entity_map.get(mesh_id);
      if (!entities) {
        entities = new Set();
        this.mesh_entity_map.set(mesh_id, entities);
      }
      entities.add(entity);
      this.entity_mesh_map.set(entity, mesh_id);
    } else {
      this.entity_mesh_map.delete(entity);
    }
  }

  /**
   * Removes mesh dependency tracking for an entity.
   *
   * @param {object} entity Entity/chunk view to untrack.
   */
  static untrack_entity_mesh(entity) {
    this.track_entity_mesh(entity, 0);
  }

  /**
   * Marks every queued entity using a mesh as dirty.
   *
   * This is the queue-local half of mesh hot-reload support: the registry fans the
   * invalidation out to all queues, and each queue maps the mesh to its users.
   *
   * @param {number} mesh_id Mesh resource identifier that changed.
   */
  static invalidate_mesh(mesh_id) {
    const registered_entities = this.mesh_entity_map.get(mesh_id);
    if (registered_entities?.size > 0) {
      for (const entity of registered_entities) {
        this.mark_meshes_dirty(true, entity);
      }
      this.needs_sort = true;
    }
  }

  /**
   * Ensures the queue has one visibility bucket for a task's material pipeline.
   *
   * Visibility passes are submitted per bucket so materials can bind their own
   * shaders and resources while sharing the queue's culling and compaction outputs.
   *
   * @param {MeshTask} task Task whose material determines the bucket.
   * @param {Set<number>} visibility_bucket_residency_set Deduplication set.
   */
  static add_visibility_bucket(task, visibility_bucket_residency_set) {
    if (task.visibility_bucket_id === invalid_u32 || visibility_bucket_residency_set.has(task.visibility_bucket_id)) {
      return;
    }

    const visibility_config = this._get_visibility_bucket_config(task.material_id);
    const visibility_bucket = this.visibility_bucket_allocator.allocate();
    visibility_bucket.key = task.visibility_bucket_id;
    visibility_bucket.shader = visibility_config?.shader ?? null;
    visibility_bucket.depth_shader = visibility_config?.depth_shader ?? null;
    visibility_bucket.resolve_shader = visibility_config?.resolve_shader ?? null;
    visibility_bucket.template_name = visibility_config?.template_name ?? "";
    visibility_bucket.representative_material_id =
      visibility_config?.representative_material_id ?? task.material_id;
    visibility_bucket.family =
      visibility_config?.family ?? MaterialFamilyType.Opaque;
    this.visibility_shader_buckets.push(visibility_bucket);

    visibility_bucket_residency_set.add(task.visibility_bucket_id);
  }

  /**
   * Converts submitted tasks into GPU-ready queue buffers.
   *
   * Renderer strategies call this once per frame for every registered queue before
   * render graph construction, guaranteeing culling and visibility see coherent
   * object, meshlet, material bucket, and indirect draw data.
   */
  static sort_and_batch() {
    this._initialize_queue_buffers();

    profile_scope(`RenderTaskQueue.sort_and_batch`, () => {
      if (this.needs_sort) {
        this._reset_rebuild_state();
        this._sort_tasks();
        this._rebuild_batches();
        this._rebuild_object_instances();
        this._sort_visibility_buckets();
        this.clear_queue_buffers();
      }

      this.upload_queue_buffers();

      MaterialAllocationTable.upload_buffers();

      this.needs_sort = false;
    });
  }

  /**
   * Returns the queue's object instance GPU buffer.
   *
   * Strategies require every queue to provide this buffer so culling can remain
   * generic and avoid any special "main mesh queue" path.
   *
   * @returns {import("./buffer.js").Buffer} Object instance storage buffer.
   */
  static get_object_instance_buffer() {
    return this.object_instance_buffer.object_instance_buffer;
  }

  /**
   * Returns the queue's meshlet instance GPU buffer.
   *
   * @returns {import("./buffer.js").Buffer} Meshlet instance storage buffer.
   */
  static get_meshlet_instance_buffer() {
    return this.meshlet_instance_buffer.meshlet_instance_buffer;
  }

  /**
   * Returns the indirect draw buffer for a view/clipmap.
   *
   * Visibility and shadow passes use separate view slots, so this accessor keeps
   * those draw argument streams isolated while exposing a simple queue API.
   *
   * @param {number} view_index View slot index.
   * @param {number} clipmap_index Optional clipmap slot for shadow cascades.
   * @returns {import("./buffer.js").Buffer} Indirect draw argument buffer.
   */
  static get_indirect_draw_buffer(view_index = 0, clipmap_index = 0) {
    return this.get_indirect_draw_object(view_index, clipmap_index).indirect_draw_buffer;
  }

  /**
   * Returns material buckets that need visibility/depth/resolve submissions.
   *
   * Strategies iterate these buckets to submit one material-compatible pass per
   * queue bucket rather than branching on concrete queue types.
   *
   * @returns {Array<object>} Visibility shader buckets.
   */
  static get_visibility_shader_buckets() {
    return this.visibility_shader_buckets;
  }

  /**
   * Returns the number of object instances currently addressable by culling.
   *
   * @returns {number} Object instance count.
   */
  static get_total_draw_count() {
    return this.object_instances?.length ?? 0;
  }

  /**
   * Returns the number of meshlet instances currently addressable by culling.
   *
   * @returns {number} Meshlet instance count.
   */
  static get_total_meshlet_count() {
    return this.total_meshlet_instances ?? 0;
  }

  /**
   * Returns or creates the view-local indirect draw object.
   *
   * This lazy path makes new views cheap: renderers can ask queues for view data
   * during graph setup without having to preallocate every possible slot.
   *
   * @param {number} view_index View slot index.
   * @param {number} clipmap_index Optional clipmap slot for shadow cascades.
   * @returns {IndirectDrawObject} View-local indirect draw state.
   */
  static get_indirect_draw_object(view_index = 0, clipmap_index = 0) {
    let obj = this.indirect_draw_objects.get(view_index, clipmap_index);
    if (!obj) {
      obj = this.allocate_view_data(view_index, clipmap_index);
    }
    return obj;
  }

  /**
   * Allocates per-view queue resources.
   *
   * Registries call this when a renderer creates a new view so every queue can
   * participate in culling, shadows, and visibility without renderer-specific hooks.
   *
   * @param {number} view_index View slot index.
   * @param {number} clipmap_index Optional clipmap slot for shadow cascades.
   * @returns {IndirectDrawObject} Allocated view-local indirect draw state.
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
   * Releases per-view queue resources.
   *
   * @param {number} view_index View slot index.
   * @param {number} clipmap_index Optional clipmap slot for shadow cascades.
   */
  static deallocate_view_data(view_index, clipmap_index = 0) {
    const obj = this.indirect_draw_objects.get(view_index, clipmap_index);
    if (obj) {
      obj.destroy();
      this.indirect_draw_objects.deallocate_at(view_index, clipmap_index);
    }
  }

  /**
   * Uploads all queue-owned GPU buffers for the current batch state.
   *
   * Subclasses can override this when they produce procedural instance data or
   * need a different packing strategy while preserving the renderer-facing API.
   */
  static upload_queue_buffers() {
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
  }

  /**
   * Clears stale CPU backing data when a queue becomes empty.
   *
   * This prevents previous frame data from lingering in reused buffers after all
   * tasks are removed, which keeps zero-work queues harmless to later passes.
   */
  static clear_queue_buffers() {
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

  /**
   * Submits the default indexed indirect draw path for mesh-backed queues.
   *
   * This is the compatibility path for passes that draw conventional mesh sections.
   * More specialized queues can override bucket submission methods instead, while
   * still sharing sorting, culling, and buffer ownership.
   *
   * @param {object} render_pass Active render pass wrapper.
   * @param {number} view_index View slot index.
   * @param {number} clipmap_index Optional clipmap slot for shadow cascades.
   * @param {boolean} skip_material_bind Whether materials are already bound.
   * @param {boolean} opaque_only Whether to skip non-opaque material batches.
   * @param {number} pass_type Material pass type being submitted.
   * @param {boolean} should_reset Whether to clear transient tasks after drawing.
   * @param {?object} indirect_draw_buffer Optional culling-produced indirect buffer.
   */
  static submit_indexed_indirect_draws(
    render_pass,
    view_index = 0,
    clipmap_index = 0,
    skip_material_bind = true,
    opaque_only = false,
    pass_type = MaterialPassType.Raster,
    should_reset = false,
    indirect_draw_buffer = null
  ) {
    const index_buffer = MeshData.index_buffer;
    const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

    let last_material = null;
    let last_pass_type = -1;

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
        if (material && (material !== last_material || last_pass_type !== pass_type)) {
          if (!material.bind(
            render_pass,
            render_pass.frame_bind_groups,
            render_pass.frame_attachments,
            pass_type
          )) {
            continue;
          };
          if (render_pass.frame_bind_groups[BindGroupType.Global]) {
            render_pass.frame_bind_groups[BindGroupType.Global].bind(render_pass);
          }
          if (render_pass.frame_bind_groups[BindGroupType.Pass]) {
            render_pass.frame_bind_groups[BindGroupType.Pass].bind(render_pass);
          }
          last_material = material;
          last_pass_type = pass_type;
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
        i * 20
      );
    }
    if (should_reset) {
      this.reset();
    }
  }

  /**
   * Binds the representative material for a visibility bucket.
   *
   * Bucket rendering intentionally binds by material family/template instead of by
   * individual task, so a queue can draw many coalesced instances with one material
   * setup per depth, raster, or resolve submission.
   *
   * @param {object} render_pass Active render pass wrapper.
   * @param {object} bucket Visibility shader bucket to bind.
   * @param {number} pass_type Material pass type being submitted.
   * @returns {?boolean} True when bound, false when unavailable, null for empty buckets.
   */
  static bind_visibility_bucket_material(render_pass, bucket, pass_type = MaterialPassType.Raster) {
    if (!bucket?.representative_material_id || !bucket?.shader) {
      return null;
    }

    const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, bucket.representative_material_id);
    if (!material) {
      return false;
    }

    if (!material.bind(
      render_pass,
      render_pass.frame_bind_groups,
      render_pass.frame_attachments,
      pass_type
    )) {
      return false;
    };

    if (render_pass.frame_bind_groups[BindGroupType.Global]) {
      render_pass.frame_bind_groups[BindGroupType.Global].bind(render_pass);
    }
    if (render_pass.frame_bind_groups[BindGroupType.Pass]) {
      render_pass.frame_bind_groups[BindGroupType.Pass].bind(render_pass);
    }
    return true;
  }

  /**
   * Submits one visibility bucket using a culling-produced indirect draw.
   *
   * Renderer strategies call this through the bucket's owning queue, giving custom
   * queues a single override point for procedural, instanced, or non-indexed draws.
   *
   * @param {object} render_pass Active render pass wrapper.
   * @param {object} bucket Visibility shader bucket to draw.
   * @param {object} indirect_buffer GPU indirect draw arguments for this bucket.
   * @param {number} pass_type Material pass type being submitted.
   */
  static submit_visibility_bucket_indirect_draw(render_pass, bucket, indirect_buffer, pass_type = MaterialPassType.Raster) {
    if (!this.bind_visibility_bucket_material(render_pass, bucket, pass_type)) {
      return;
    }
    render_pass.pass.drawIndirect(indirect_buffer.buffer, 0);
  }

  /**
   * Submits a fullscreen resolve for a visibility bucket.
   *
   * Resolve runs after visibility rasterization and lets each material decode the
   * visibility buffer into G-buffer targets with its own material bindings.
   *
   * @param {object} render_pass Active render pass wrapper.
   * @param {object} bucket Visibility shader bucket to resolve.
   * @param {number} instance_count Number of resolve instances to draw.
   */
  static submit_visibility_bucket_resolve(render_pass, bucket, instance_count = 1) {
    if (!this.bind_visibility_bucket_material(render_pass, bucket, MaterialPassType.Resolve)) {
      return;
    }
    draw_quad(render_pass, instance_count);
  }

  static _get_task_key(mesh_id, section, material_id, visibility_bucket_id = invalid_u32) {
    const a = BigInt(mesh_id);
    const b = BigInt(section);
    const c = BigInt(material_id ?? 0);
    const ab = a >= b ? a * a + a + b : b * b + a;
    const abc = ab >= c ? ab * ab + ab + c : c * c + ab;
    const d = BigInt(visibility_bucket_id === invalid_u32 ? 0 : visibility_bucket_id);
    return abc >= d ? abc * abc + abc + d : d * d + abc;
  }

  static _get_visibility_bucket_config(material_id) {
    const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, material_id);
    return {
      id: Name.from(
        `${material.template.name}|${material?.template?.shader?.file_path ?? ""}`
      ),
      shader: material?.template?.shader ?? null,
      depth_shader: material?.template?.depth_shader ?? null,
      resolve_shader: material?.template?.resolve_shader ?? null,
      template_name: material.template.name,
      representative_material_id: material_id,
      family: material.family,
    };
  }

  static _sort_tasks() {
    this.tasks.sort((a, b) => {
      const a_bucket =
        a.visibility_bucket_id === invalid_u32 ? Number.MAX_SAFE_INTEGER : a.visibility_bucket_id;
      const b_bucket =
        b.visibility_bucket_id === invalid_u32 ? Number.MAX_SAFE_INTEGER : b.visibility_bucket_id;
      let diff = a_bucket - b_bucket;
      if (diff !== 0) return diff;
      diff = a.material_id - b.material_id;
      if (diff !== 0) return diff;
      diff = a.mesh_id - b.mesh_id;
      if (diff !== 0) return diff;
      diff = a.section - b.section;
      return diff;
    });
  }

  static _task_matches_batch(task, batch) {
    return Boolean(
      batch &&
      batch.visibility_bucket_id === task.visibility_bucket_id &&
      batch.mesh_id === task.mesh_id &&
      batch.section === task.section &&
      batch.material_id === task.material_id
    );
  }

  static _create_batch_for_task(task, last_batch) {
    const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);
    if (!mesh) {
      return null;
    }

    const batch = new this.batch_type();
    batch.mesh_id = task.mesh_id;
    batch.section = task.section;
    batch.material_id = task.material_id;
    batch.visibility_bucket_id = task.visibility_bucket_id;
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
    return batch;
  }

  static _append_task_to_batch(task, batch) {
    batch.instance_count += task.entity.instance_count;
    const start_index = batch.entities.length;
    const new_length = batch.entities.length + task.entity.instance_count;
    batch.entities.length = new_length;
    batch.entities.fill(task.entity, start_index, new_length);
  }

  static _rebuild_batches() {
    let last_batch = null;
    const visibility_bucket_residency_set = new Set();
    for (let i = 0; i < this.tasks.length; i++) {
      const task = this.tasks[i];

      if (!this._task_matches_batch(task, last_batch)) {
        const batch = this._create_batch_for_task(task, last_batch);
        if (batch) {
          this.add_visibility_bucket(task, visibility_bucket_residency_set);
          this.batches.push(batch);
          last_batch = batch;
        }
      } else {
        this._append_task_to_batch(task, last_batch);
      }
    }
  }

  static _rebuild_object_instances() {
    for (let i = 0; i < this.batches.length; i++) {
      const batch = this.batches[i];
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
            entry.visibility_bucket_id = batch.visibility_bucket_id;
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
  }

  static _sort_visibility_buckets() {
    this.visibility_shader_buckets.sort((a, b) => {
      let diff = a.family - b.family;
      if (diff !== 0) return diff;
      return a.key - b.key;
    });
  }

  static _reset_rebuild_state() {
    this.batches.length = 0;
    this.object_instances.length = 0;
    this.visibility_shader_buckets.length = 0;
    this.total_meshlet_instances = 0;
    this.object_instance_allocator.reset();
    this.visibility_bucket_allocator.reset();
  }

  static _initialize_queue_buffers() {
    if (!this.initialized) {
      this.initialized = true;
      this.object_instance_buffer.init();
      this.meshlet_instance_buffer.init();
    }
  }
}

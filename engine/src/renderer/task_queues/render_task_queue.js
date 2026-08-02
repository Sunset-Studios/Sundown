import { MAX_BUFFERED_FRAMES, INVALID_U32 } from "../../core/minimal.js";
import { EntityID } from "../../core/ecs/solar/types.js";
import { Buffer } from "../buffer.js";
import {
  FreeListAllocator,
  RandomAccessAllocator,
  Sparse2DRandomAccessAllocator,
} from "../../memory/allocator.js";
import { ResourceCache } from "../resource_cache.js";
import { MeshData } from "../mesh_data.js";
import { MaterialAllocationTable } from "../material_allocation_table.js";
import { profile_scope } from "../../utility/performance.js";
import { BindGroupType, CacheTypes, MaterialFamilyType, MaterialPassType } from "../renderer_types.js";
import { Name } from "../../utility/names.js";
import { draw_quad } from "../draw_helpers.js";

export const RenderWorkKind = {
  Mesh: "mesh",
  UI3DMesh: "ui_3d_mesh",
  SkinnedMesh: "skinned_mesh",
  ProceduralMesh: "procedural_mesh",
  DebugMesh: "debug_mesh",
  Default: "mesh",
};

const initial_buffer_size = 1024;
const max_frame_buffer_writes = 100000;
// Mirrored by OIF_DOUBLE_SIDED in common_types.wgsl. Packing this into the existing
// fourth word keeps meshlet cone culling allocation- and bandwidth-neutral.
const object_instance_flag_double_sided = 1 << 0;

class IndirectDrawBatch {
  mesh_id = 0;
  section = 0;
  material_id = 0;
  visibility_bucket_id = INVALID_U32;
  entities = [];
  instance_count = 0;
  first_index = 0;
  index_count = 0;
  base_vertex = 0;
  base_instance = 0;
  object_instance_flags = 0;
}

class VisibilityShaderBucket {
  key = INVALID_U32;
  shader = null;
  depth_shader = null;
  resolve_shader = null;
  forward_shader = null;
  template_name = "";
  representative_material_id = 0;
  family = MaterialFamilyType.Opaque;
  queue = null;
  queue_name = "";
}

class ObjectInstanceEntry {
  constructor(batch_index = 0, row_field = INVALID_U32) {
    this.batch_index = batch_index;
    this.row = row_field;
    this.visibility_bucket_id = INVALID_U32;
    this.mesh_id = 0;
    this.section = 0;
    this.meshlet_offset = 0;
    this.meshlet_count = 0;
    this.meshlet_group_offset = 0;
    this.meshlet_group_count = 0;
    this.flags = 0;
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
  lane_id = RenderWorkKind.Mesh;
  mesh_id = null;
  entity = null;
  material_id = null;
  section = 0;
  visibility_bucket_id = INVALID_U32;
  key = "";
  queue_index = -1;

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
   * @param {string} key Per-entity dedupe key for this task.
   * @param {number} queue_index Current index inside the queue task array.
   */
  static init(
    task,
    mesh_id,
    entity,
    material_id = null,
    section = 0,
    visibility_bucket_id = INVALID_U32,
    key = "",
    queue_index = -1,
    lane_id = RenderWorkKind.Mesh
  ) {
    task.lane_id = lane_id;
    task.mesh_id = mesh_id;
    task.entity = entity;
    task.material_id = material_id;
    task.section = section;
    task.visibility_bucket_id = visibility_bucket_id;
    task.key = key;
    task.queue_index = queue_index;
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
              this.object_instance_data[i + 3] = object_instances[offset].flags;
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
      this.meshlet_instance_data.fill(INVALID_U32);
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
        INVALID_U32,
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
        new_meshlet_instance_data.fill(INVALID_U32);
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
 * Owns submitted work records, dedupe maps, and task allocation.
 *
 * The store is intentionally not mesh-specific: lanes decide how descriptors become
 * keys and visibility data, while the store provides a single centralized place for
 * lane refreshes, and stable queue indices.
 */
export class RenderTaskStore {
  constructor(task_type = MeshTask) {
    this.task_type = task_type;
    this.tasks_allocator = new FreeListAllocator(256, task_type);
    this.tasks = [];
    this.entity_task_map = new Map();
  }

  reset() {
    this.tasks.length = 0;
    this.tasks_allocator.reset();
    this.entity_task_map.clear();
  }

  add(lane, descriptor) {
    const key = lane.get_task_key(descriptor);

    let tasks_for_entity = this.entity_task_map.get(descriptor.entity);
    if (tasks_for_entity?.has(key)) {
      const existing_task = tasks_for_entity.get(key);
      existing_task.lane_id = lane.id;
      return existing_task;
    }

    const task = this._allocate_task();
    const queue_index = this.tasks.length;
    this.task_type.init(
      task,
      descriptor.mesh_id,
      descriptor.entity,
      descriptor.material_id,
      descriptor.section,
      descriptor.visibility_bucket_id,
      key,
      queue_index,
      lane.id
    );
    this.tasks.push(task);

    if (!tasks_for_entity) {
      tasks_for_entity = new Map();
      this.entity_task_map.set(descriptor.entity, tasks_for_entity);
    }
    tasks_for_entity.set(key, task);
    return task;
  }

  contains(entity) {
    return this.entity_task_map.has(entity) && this.entity_task_map.get(entity).size > 0;
  }

  remove(entity) {
    const tasks_for_entity = this.entity_task_map.get(entity);
    if (!tasks_for_entity) {
      return false;
    }

    const tasks = Array.from(tasks_for_entity.values());
    for (const task of tasks) {
      this.remove_at(task.queue_index);
    }
    return tasks.length > 0;
  }

  remove_matching(lane_id = null, predicate = null) {
    let removed = false;
    for (let i = this.tasks.length - 1; i >= 0; i--) {
      const task = this.tasks[i];
      const matches_lane = lane_id === null || task.lane_id === lane_id;
      const matches_predicate = predicate ? predicate(task) : true;
      if (matches_lane && matches_predicate) {
        this.remove_at(i);
        removed = true;
      }
    }
    return removed;
  }

  sort(compare) {
    this.tasks.sort(compare);
    for (let i = 0; i < this.tasks.length; i++) {
      this.tasks[i].queue_index = i;
    }
  }

  tasks_for_lane(lane_id) {
    const tasks = [];
    for (let i = 0; i < this.tasks.length; i++) {
      const task = this.tasks[i];
      if (task.lane_id === lane_id) {
        tasks.push(task);
      }
    }
    return tasks;
  }

  remove_at(index) {
    if (index < 0 || index >= this.tasks.length) {
      return;
    }

    const last_index = this.tasks.length - 1;
    const removed_task = this.tasks[index];
    const swapped_task = this.tasks[last_index];
    this._untrack_task(removed_task);
    this.tasks.pop();

    removed_task.queue_index = -1;
    this.tasks_allocator.deallocate(removed_task);

    if (index === last_index) {
      return;
    }

    this.tasks[index] = swapped_task;
    swapped_task.queue_index = index;
  }

  _allocate_task() {
    return this.tasks_allocator.allocate();
  }

  _untrack_task(task) {
    const tasks_for_entity = this.entity_task_map.get(task.entity);
    if (!tasks_for_entity) {
      return;
    }
    if (tasks_for_entity.get(task.key) === task) {
      tasks_for_entity.delete(task.key);
    }
    if (tasks_for_entity.size === 0) {
      this.entity_task_map.delete(task.entity);
    }
  }
}

/**
 * Centralized GPU resource owner for all mesh-like render work.
 *
 * Lanes feed packed batches and instances into this object, and renderer
 * strategies continue to consume one object buffer, one meshlet buffer, and one
 * indirect draw stream per view/clipmap.
 */
export class RenderQueueGpuResources {
  constructor({
    object_instance_buffer = new ObjectInstanceBuffer(),
    meshlet_instance_buffer = new MeshletInstanceBuffer(),
    indirect_draw_objects = new Sparse2DRandomAccessAllocator(16, 4, IndirectDrawObject),
    indirect_draw_name_prefix = "indirect_draw_buffer",
  } = {}) {
    this.object_instance_buffer = object_instance_buffer;
    this.meshlet_instance_buffer = meshlet_instance_buffer;
    this.indirect_draw_objects = indirect_draw_objects;
    this.indirect_draw_name_prefix = indirect_draw_name_prefix;
    this.initialized = false;
  }

  init() {
    if (this.initialized) {
      return;
    }
    this.initialized = true;
    this.object_instance_buffer.init();
    this.meshlet_instance_buffer.init();
  }

  get_indirect_draw_object(view_index = 0, clipmap_index = 0) {
    let obj = this.indirect_draw_objects.get(view_index, clipmap_index);
    if (!obj) {
      obj = this.allocate_view_data(view_index, clipmap_index);
    }
    return obj;
  }

  allocate_view_data(view_index = 0, clipmap_index = 0) {
    const obj = this.indirect_draw_objects.allocate_at(view_index, clipmap_index);
    obj.view_index = view_index;
    obj.clipmap_index = clipmap_index;
    obj.init(this.indirect_draw_name_prefix);
    return obj;
  }

  deallocate_view_data(view_index, clipmap_index = 0) {
    const obj = this.indirect_draw_objects.get(view_index, clipmap_index);
    if (obj) {
      obj.destroy();
      this.indirect_draw_objects.deallocate_at(view_index, clipmap_index);
    }
  }

  upload({ batches, object_instances, total_meshlet_instances, force_update = false }) {
    this.object_instance_buffer.update_buffers(object_instances, force_update);
    this.meshlet_instance_buffer.update_buffers(
      object_instances,
      total_meshlet_instances,
      force_update
    );

    for (let i = 0; i < this.indirect_draw_objects.x_capacity; i++) {
      for (let j = 0; j < this.indirect_draw_objects.y_capacity; j++) {
        const obj = this.indirect_draw_objects.get(i, j);
        if (obj && obj.indirect_draw_data) {
          obj.update_buffers(batches, force_update);
        }
      }
    }
  }

  clear_if_empty(object_instances) {
    if (object_instances.length > 0) {
      return;
    }

    if (this.object_instance_buffer.object_instance_data) {
      this.object_instance_buffer.object_instance_data.fill(0);
    }
    if (this.meshlet_instance_buffer.meshlet_instance_data) {
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
}

/**
 * Lane policy for conventional indexed, meshlet-backed mesh sections.
 *
 * Other mesh work can be siphoned through the same RenderWorkQueue by registering
 * lanes with different descriptor normalization, batch keys, packing, or draw
 * backends while preserving the renderer-facing central queue contract.
 */
export class IndexedMeshQueueLane {
  constructor({
    id = RenderWorkKind.Mesh,
    name = id,
    task_type = MeshTask,
    batch_type = IndirectDrawBatch,
  } = {}) {
    this.id = id;
    this.task_type = task_type;
    this.batch_type = batch_type;
  }

  normalize_task_descriptor(descriptor) {
    const material_id = descriptor.material_id ?? null;
    const visibility_bucket = this.get_visibility_bucket_config(material_id);
    return {
      mesh_id: descriptor.mesh_id,
      entity: descriptor.entity,
      material_id,
      section: descriptor.section ?? 0,
      visibility_bucket_id: visibility_bucket?.id ?? INVALID_U32,
    };
  }

  get_task_key(descriptor) {
    return `${this.id}|${descriptor.mesh_id}|${descriptor.section}|${descriptor.material_id ?? 0}|${descriptor.visibility_bucket_id}`;
  }

  compare_tasks(a, b) {
    const a_bucket =
      a.visibility_bucket_id === INVALID_U32 ? Number.MAX_SAFE_INTEGER : a.visibility_bucket_id;
    const b_bucket =
      b.visibility_bucket_id === INVALID_U32 ? Number.MAX_SAFE_INTEGER : b.visibility_bucket_id;
    let diff = a_bucket - b_bucket;
    if (diff !== 0) return diff;
    diff = (a.material_id ?? 0) - (b.material_id ?? 0);
    if (diff !== 0) return diff;
    diff = (a.mesh_id ?? 0) - (b.mesh_id ?? 0);
    if (diff !== 0) return diff;
    return (a.section ?? 0) - (b.section ?? 0);
  }

  prepare(tasks, context) {
    const first_batch_index = context.batches.length;
    this.build_batches(tasks, context);
    this.pack_object_instances(first_batch_index, context);
  }

  build_batches(tasks, context) {
    let last_lane_batch = null;
    for (let i = 0; i < tasks.length; i++) {
      const task = tasks[i];
      if (!this.task_matches_batch(task, last_lane_batch)) {
        const batch = this.create_batch_for_task(task, context.last_batch());
        if (batch) {
          this.add_visibility_bucket(task, context);
          context.batches.push(batch);
          last_lane_batch = batch;
        }
      } else {
        this.append_task_to_batch(task, last_lane_batch);
      }
    }
  }

  task_matches_batch(task, batch) {
    return Boolean(
      batch &&
      batch.lane_id === this.id &&
      batch.visibility_bucket_id === task.visibility_bucket_id &&
      batch.mesh_id === task.mesh_id &&
      batch.section === task.section &&
      batch.material_id === task.material_id
    );
  }

  create_batch_for_task(task, previous_global_batch) {
    const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);
    if (!mesh) {
      return null;
    }

    const batch = new this.batch_type();
    batch.lane_id = this.id;
    batch.mesh_id = task.mesh_id;
    batch.section = task.section;
    batch.material_id = task.material_id;
    batch.visibility_bucket_id = task.visibility_bucket_id;
    batch.base_instance = previous_global_batch
      ? previous_global_batch.base_instance + previous_global_batch.instance_count
      : 0;
    batch.instance_count = task.entity.instance_count;

    const section = mesh.sections?.[task.section] || {
      first_index: 0,
      index_count: mesh.index_count,
    };
    batch.first_index = section.first_index;
    batch.index_count = section.index_count;
    batch.base_vertex = mesh.vertex_buffer_offset;
    const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, task.material_id);
    batch.object_instance_flags =
      material?.template?.pipeline_state_config?.rasterizer_state?.cull_mode === "none"
        ? object_instance_flag_double_sided
        : 0;

    batch.entities.length = 1;
    batch.entities[0] = task.entity;
    return batch;
  }

  append_task_to_batch(task, batch) {
    batch.instance_count += task.entity.instance_count;
    batch.entities.push(task.entity);
  }

  submit_visibility_bucket_indirect_draw(queue, render_pass, bucket, indirect_buffer, pass_type = MaterialPassType.Raster) {
    if (!queue.bind_visibility_bucket_material(render_pass, bucket, pass_type)) {
      return;
    }
    render_pass.pass.drawIndirect(indirect_buffer.buffer, 0);
  }

  submit_visibility_bucket_resolve(queue, render_pass, bucket, instance_count = 1) {
    if (!queue.bind_visibility_bucket_material(render_pass, bucket, MaterialPassType.Resolve)) {
      return;
    }
    draw_quad(render_pass, instance_count);
  }

  add_visibility_bucket(task, context) {
    if (
      task.visibility_bucket_id === INVALID_U32 ||
      context.visibility_bucket_residency_set.has(task.visibility_bucket_id)
    ) {
      return;
    }

    const visibility_config = this.get_visibility_bucket_config(task.material_id);
    const visibility_bucket = context.visibility_bucket_allocator.allocate();
    visibility_bucket.key = task.visibility_bucket_id;
    visibility_bucket.shader = visibility_config?.shader ?? null;
    visibility_bucket.depth_shader = visibility_config?.depth_shader ?? null;
    visibility_bucket.resolve_shader = visibility_config?.resolve_shader ?? null;
    visibility_bucket.forward_shader = visibility_config?.forward_shader ?? null;
    visibility_bucket.template_name = visibility_config?.template_name ?? "";
    visibility_bucket.representative_material_id =
      visibility_config?.representative_material_id ?? task.material_id;
    visibility_bucket.family =
      visibility_config?.family ?? MaterialFamilyType.Opaque;
    visibility_bucket.queue = context.queue;
    visibility_bucket.queue_name = context.queue.queue_name;
    visibility_bucket.lane = this;
    visibility_bucket.lane_id = this.id;
    const target_buckets = visibility_bucket.family === MaterialFamilyType.Transparent
      ? context.visibility_forward_buckets
      : context.visibility_shader_buckets;
    context.visibility_all_buckets.push(visibility_bucket);
    target_buckets.push(visibility_bucket);
    context.visibility_bucket_residency_set.add(task.visibility_bucket_id);
  }

  pack_object_instances(first_batch_index, context) {
    for (let i = first_batch_index; i < context.batches.length; i++) {
      const batch = context.batches[i];
      if (batch.lane_id !== this.id) {
        continue;
      }

      const mesh = ResourceCache.get().fetch(CacheTypes.MESH, batch.mesh_id);
      const meshlet_section = mesh?.meshlet_sections?.[batch.section] ?? {
        meshlet_offset: 0,
        meshlet_count: 0,
        meshlet_group_offset: 0,
        meshlet_group_count: 0,
      };

      for (let j = 0; j < batch.entities.length; j++) {
        const entity = batch.entities[j];

        for (let k = 0, segs = entity.segments, n = segs.length; k < n; k++) {
          const seg = segs[k];
          const cidx = seg.chunk.chunk_index;
          const start = seg.slot;
          for (let l = 0, cnt = seg.count; l < cnt; l++) {
            const entry = context.object_instance_allocator.allocate();
            entry.batch_index = i;
            entry.row = EntityID.make_row_field(start + l, cidx);
            entry.visibility_bucket_id = batch.visibility_bucket_id;
            entry.mesh_id = batch.mesh_id;
            entry.section = batch.section;
            entry.meshlet_offset = meshlet_section.meshlet_offset;
            entry.meshlet_count = meshlet_section.meshlet_count;
            entry.meshlet_group_offset = meshlet_section.meshlet_group_offset;
            entry.meshlet_group_count = meshlet_section.meshlet_group_count;
            entry.flags = batch.object_instance_flags;
            context.object_instances.push(entry);
            context.add_meshlet_instances(meshlet_section.meshlet_count);
          }
        }
      }
    }
  }

  get_visibility_bucket_config(material_id) {
    const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, material_id);
    if (!material?.template) {
      return null;
    }

    return {
      id: Name.from(
        `${material.template.name}|${material.family}|${material.template.shader?.file_path ?? ""}`
      ),
      shader: material.template.shader ?? null,
      depth_shader: material.template.depth_shader ?? null,
      resolve_shader: material.template.resolve_shader ?? null,
      forward_shader: material.template.forward_shader ?? null,
      template_name: material.template.name,
      representative_material_id: material_id,
      family: material.family,
    };
  }
}

/**
 * Single renderer work intake for mesh-like submissions.
 *
 * Producers submit descriptors into this central queue. Internally, registered
 * lanes decide how each work kind is sorted, batched, packed, and eventually drawn.
 * The compatibility export `RenderTaskQueue` points at this class so existing
 * systems can migrate gradually.
 */
export class RenderWorkQueue {
  static queue_name = "render_work_queue";
  static task_store = new RenderTaskStore();
  static resources = new RenderQueueGpuResources();
  static lanes = new Map();
  static lane_order = [];

  static batches = [];
  static object_instances = [];
  static visibility_all_buckets = [];
  static visibility_shader_buckets = [];
  static visibility_forward_buckets = [];

  static dirty_mesh_entities = new Set();
  static entity_mesh_map = new Map();
  static mesh_entity_map = new Map();

  static object_instance_allocator = new RandomAccessAllocator(256, ObjectInstanceEntry);
  static visibility_bucket_allocator = new RandomAccessAllocator(256, VisibilityShaderBucket);

  static needs_sort = false;
  static meshes_dirty = false;
  static total_meshlet_instances = 0;

  static get tasks() {
    return this.task_store.tasks;
  }

  static get entity_task_map() {
    return this.task_store.entity_task_map;
  }

  static get tasks_allocator() {
    return this.task_store.tasks_allocator;
  }

  static get object_instance_buffer() {
    return this.resources.object_instance_buffer;
  }

  static get meshlet_instance_buffer() {
    return this.resources.meshlet_instance_buffer;
  }

  static get indirect_draw_objects() {
    return this.resources.indirect_draw_objects;
  }

  static register_lane(lane) {
    this.lanes.set(lane.id, lane);
    if (!this.lane_order.includes(lane.id)) {
      this.lane_order.push(lane.id);
    }
    this.needs_sort = true;
    return lane;
  }

  static resolve_lane(lane_id = RenderWorkKind.Default) {
    return this.lanes.get(lane_id) ?? this.lanes.get(RenderWorkKind.Default);
  }

  static submit(descriptor, resort = true) {
    const lane = this.resolve_lane(descriptor.lane_id);
    const normalized = lane.normalize_task_descriptor(descriptor);
    const task = this.task_store.add(lane, normalized);
    if (resort) {
      this.needs_sort = true;
    }
    return task;
  }

  static new_task(mesh_id, entity, material_id = null, section = 0, resort = true) {
    return this.submit(
      {
        mesh_id,
        entity,
        material_id,
        section,
      },
      resort
    );
  }

  static mark_needs_sort() {
    this.needs_sort = true;
  }

  static reset() {
    this.task_store.reset();
    this.needs_sort = true;
  }

  static contains(entity) {
    return this.task_store.contains(entity);
  }

  static remove(entity, resort = true) {
    const removed = this.task_store.remove(entity);
    if (removed && this.tasks.length === 0) {
      this._clear_rebuilt_cpu_state();
    }
    this.needs_sort ||= removed && resort;
  }

  static remove_tasks(lane_id = null, predicate = null, resort = true) {
    const removed = this.task_store.remove_matching(lane_id, predicate);
    if (removed && this.tasks.length === 0) {
      this._clear_rebuilt_cpu_state();
    }
    this.needs_sort ||= removed && resort;
    return removed;
  }

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

  static has_dirty_meshes() {
    return this.meshes_dirty;
  }

  static get_dirty_mesh_entities() {
    return this.dirty_mesh_entities;
  }

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

  static untrack_entity_mesh(entity) {
    this.track_entity_mesh(entity, 0);
  }

  static invalidate_mesh(mesh_id) {
    const registered_entities = this.mesh_entity_map.get(mesh_id);
    if (registered_entities?.size > 0) {
      for (const entity of registered_entities) {
        this.mark_meshes_dirty(true, entity);
      }
      this.needs_sort = true;
    }
  }

  static sort_and_batch() {
    this.prepare();
  }

  static prepare() {
    this.resources.init();

    profile_scope("RenderWorkQueue.prepare", () => {
      if (this.needs_sort) {
        this._reset_rebuild_state();
        this._sort_tasks();
        this._prepare_lanes();
        this._sort_visibility_buckets();
        this.clear_queue_buffers();
      }

      this.upload_queue_buffers();
      MaterialAllocationTable.upload_buffers();
      this.needs_sort = false;
    });
  }

  static get_object_instance_buffer() {
    return this.resources.object_instance_buffer.object_instance_buffer;
  }

  static get_meshlet_instance_buffer() {
    return this.resources.meshlet_instance_buffer.meshlet_instance_buffer;
  }

  static get_indirect_draw_buffer(view_index = 0, clipmap_index = 0) {
    return this.get_indirect_draw_object(view_index, clipmap_index).indirect_draw_buffer;
  }

  static get_visibility_shader_buckets() {
    return this.visibility_shader_buckets;
  }

  static get_visibility_forward_buckets() {
    return this.visibility_forward_buckets;
  }

  static get_visibility_all_buckets() {
    return this.visibility_all_buckets;
  }

  static get_total_draw_count() {
    return this.object_instances?.length ?? 0;
  }

  static get_total_meshlet_count() {
    return this.total_meshlet_instances ?? 0;
  }

  static get_indirect_draw_object(view_index = 0, clipmap_index = 0) {
    return this.resources.get_indirect_draw_object(view_index, clipmap_index);
  }

  static allocate_view_data(view_index = 0, clipmap_index = 0) {
    const obj = this.resources.allocate_view_data(view_index, clipmap_index);
    this.needs_sort = true;
    return obj;
  }

  static deallocate_view_data(view_index, clipmap_index = 0) {
    this.resources.deallocate_view_data(view_index, clipmap_index);
  }

  static upload_queue_buffers() {
    this.resources.upload({
      batches: this.batches,
      object_instances: this.object_instances,
      total_meshlet_instances: this.total_meshlet_instances,
      force_update: this.needs_sort,
    });
  }

  static clear_queue_buffers() {
    this.resources.clear_if_empty(this.object_instances);
  }

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
      if (!mesh || mesh.index_buffer_offset === -1) {
        continue;
      }

      const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, batch.material_id);
      if (opaque_only && material?.family !== MaterialFamilyType.Opaque) {
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
          }
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

  static bind_visibility_bucket_material(render_pass, bucket, pass_type = MaterialPassType.Raster) {
    const shader = pass_type === MaterialPassType.Forward
      ? bucket?.forward_shader
      : bucket?.shader;
    if (!bucket?.representative_material_id || !shader) {
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
    }

    if (render_pass.frame_bind_groups[BindGroupType.Global]) {
      render_pass.frame_bind_groups[BindGroupType.Global].bind(render_pass);
    }
    if (render_pass.frame_bind_groups[BindGroupType.Pass]) {
      render_pass.frame_bind_groups[BindGroupType.Pass].bind(render_pass);
    }
    return true;
  }

  static submit_visibility_bucket_indirect_draw(render_pass, bucket, indirect_buffer, pass_type = MaterialPassType.Raster) {
    if (bucket?.lane?.submit_visibility_bucket_indirect_draw) {
      bucket.lane.submit_visibility_bucket_indirect_draw(
        this,
        render_pass,
        bucket,
        indirect_buffer,
        pass_type
      );
      return;
    }

    if (!this.bind_visibility_bucket_material(render_pass, bucket, pass_type)) {
      return;
    }
    render_pass.pass.drawIndirect(indirect_buffer.buffer, 0);
  }

  static submit_visibility_bucket_resolve(render_pass, bucket, instance_count = 1) {
    if (bucket?.lane?.submit_visibility_bucket_resolve) {
      bucket.lane.submit_visibility_bucket_resolve(
        this,
        render_pass,
        bucket,
        instance_count
      );
      return;
    }

    if (!this.bind_visibility_bucket_material(render_pass, bucket, MaterialPassType.Resolve)) {
      return;
    }
    draw_quad(render_pass, instance_count);
  }

  static _sort_tasks() {
    const lane_indices = new Map();
    for (let i = 0; i < this.lane_order.length; i++) {
      lane_indices.set(this.lane_order[i], i);
    }

    this.task_store.sort((a, b) => {
      const lane_diff =
        (lane_indices.get(a.lane_id) ?? Number.MAX_SAFE_INTEGER) -
        (lane_indices.get(b.lane_id) ?? Number.MAX_SAFE_INTEGER);
      if (lane_diff !== 0) {
        return lane_diff;
      }

      const lane = this.lanes.get(a.lane_id) ?? this.resolve_lane();
      return lane.compare_tasks(a, b);
    });
  }

  static _prepare_lanes() {
    const context = {
      queue: this,
      batches: this.batches,
      object_instances: this.object_instances,
      visibility_all_buckets: this.visibility_all_buckets,
      visibility_shader_buckets: this.visibility_shader_buckets,
      visibility_forward_buckets: this.visibility_forward_buckets,
      object_instance_allocator: this.object_instance_allocator,
      visibility_bucket_allocator: this.visibility_bucket_allocator,
      visibility_bucket_residency_set: new Set(),
      last_batch: () => this.batches[this.batches.length - 1] ?? null,
      add_meshlet_instances: (count) => {
        this.total_meshlet_instances += count;
      },
    };

    for (let i = 0; i < this.lane_order.length; i++) {
      const lane_id = this.lane_order[i];
      const lane = this.lanes.get(lane_id);
      if (!lane) {
        continue;
      }
      lane.prepare(this.task_store.tasks_for_lane(lane_id), context);
    }
  }

  static _sort_visibility_buckets() {
    const compare_buckets = (a, b) => {
      let diff = a.family - b.family;
      if (diff !== 0) return diff;
      return a.key - b.key;
    };
    this.visibility_all_buckets.sort(compare_buckets);
    this.visibility_shader_buckets.sort(compare_buckets);
    this.visibility_forward_buckets.sort(compare_buckets);
  }

  static _reset_rebuild_state() {
    this._clear_rebuilt_cpu_state();
    this.object_instance_allocator.reset();
    this.visibility_bucket_allocator.reset();
  }

  static _clear_rebuilt_cpu_state() {
    this.batches.length = 0;
    this.object_instances.length = 0;
    this.visibility_all_buckets.length = 0;
    this.visibility_shader_buckets.length = 0;
    this.visibility_forward_buckets.length = 0;
    this.total_meshlet_instances = 0;
  }
}

RenderWorkQueue.register_lane(new IndexedMeshQueueLane({
  id: RenderWorkKind.Mesh,
}));
RenderWorkQueue.register_lane(new IndexedMeshQueueLane({
  id: RenderWorkKind.UI3DMesh,
}));

export { RenderWorkQueue as RenderTaskQueue };

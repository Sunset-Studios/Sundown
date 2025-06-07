import { MAX_BUFFERED_FRAMES } from "../core/minimal.js";
import { EntityID } from "../core/ecs/solar/types.js";
import { Renderer } from "./renderer.js";
import { ResourceCache } from "./resource_cache.js";
import { Mesh } from "./mesh.js";
import { Buffer } from "./buffer.js";
import { Name } from "../utility/names.js";
import { RandomAccessAllocator, SparseRandomAccessAllocator } from "../memory/allocator.js";
import { profile_scope } from "../utility/performance.js";
import { CacheTypes, MaterialFamilyType, BindGroupType } from "./renderer_types.js";

const initial_buffer_size = 1024;
const max_frame_buffer_writes = 100000;

class IndirectDrawBatch {
  mesh_id = 0;
  material_id = 0;
  entities = [];
  index_buffer_id = null;
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

        Renderer.get().mark_bind_groups_dirty(true);
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

class IndirectDrawObject {
  view_index = 0;
  indirect_draw_buffer = null;
  visible_instance_buffer_no_occlusion = null;
  visible_instance_buffer = null;
  indirect_draw_data = null;
  current_indirect_draw_write_offset = 0;
  last_indirect_draw_count = 0;

  init() {
    profile_scope("init_indirect_draw_object", () => {
      this.indirect_draw_data = new Uint32Array(initial_buffer_size * 5);

      const suffix = this.view_index === 0 ? "" : `_view_${this.view_index}`;
      if (!this.indirect_draw_buffer) {
        this.indirect_draw_buffer = Buffer.create({
          name: `indirect_draw_buffer${suffix}`,
          raw_data: this.indirect_draw_data,
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        });
      }

      if (!this.visible_instance_buffer_no_occlusion) {
        this.visible_instance_buffer_no_occlusion = Buffer.create({
          name: `visible_instance_buffer_no_occlusion${suffix}`,
          raw_data: new Int32Array(initial_buffer_size),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }

      if (!this.visible_instance_buffer) {
        this.visible_instance_buffer = Buffer.create({
          name: `visible_instance_buffer${suffix}`,
          raw_data: new Int32Array(initial_buffer_size),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
    });
  }

  // We assume this gets called once per frame
  update_buffers(batches, object_instances, force_update = false) {
    profile_scope("update_indirect_buffers", () => {
      const suffix = this.view_index === 0 ? "" : `_view_${this.view_index}`;
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

        Renderer.get().mark_bind_groups_dirty(true);
      }

      // Resize object instance buffer if needed
      const required_object_instance_size = object_instances.length * 4; // 4 bytes per instance
      if (this.visible_instance_buffer_no_occlusion.config.size < required_object_instance_size) {
        this.visible_instance_buffer_no_occlusion = Buffer.create({
          name: `visible_instance_buffer_no_occlusion${suffix}`,
          raw_data: new Int32Array(object_instances.length * 2),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
      }
      if (this.visible_instance_buffer.config.size < required_object_instance_size) {
        this.visible_instance_buffer = Buffer.create({
          name: `visible_instance_buffer${suffix}`,
          raw_data: new Int32Array(object_instances.length * 2),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
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
              this.indirect_draw_data[i + 2] = batches[offset].first_index;
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
    this.indirect_draw_buffer.write_raw(
      this.indirect_draw_data,
    );
  }

  destroy() {
    this.indirect_draw_buffer.destroy();
    this.visible_instance_buffer_no_occlusion.destroy();
    this.visible_instance_buffer.destroy();
    this.indirect_draw_buffer = null;
    this.visible_instance_buffer_no_occlusion = null;
    this.visible_instance_buffer = null;
    this.indirect_draw_data = null;
  }
}

class MeshTask {
  mesh_id = null;
  entity = null;
  material_id = null;

  static init(task, mesh_id, entity, material_id = null) {
    task.mesh_id = mesh_id;
    task.entity = entity;
    task.material_id = material_id;
  }
}

export class MeshTaskQueue {
  static tasks = [];
  static batches = [];
  static object_instances = [];
  static material_buckets = [];
  static object_instance_buffer = new ObjectInstanceBuffer();
  static indirect_draw_objects = new SparseRandomAccessAllocator(256, IndirectDrawObject); // Per-view indirect draw objects
  static tasks_allocator = new RandomAccessAllocator(256, MeshTask); // TODO: This can potentially use a TypedVector depending on how it's structured. May need to split the fields out.
  static object_instance_allocator = new RandomAccessAllocator(256, ObjectInstanceEntry); // TODO: This can potentially use a TypedVector depending on how it's structured. May need to split the fields out.
  static needs_sort = false;
  static initialized = false;
  static entity_task_map = new Map(); // new: Map<Entity, Map<"meshId:materialId", Task>>

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

  static _get_task_key(mesh_id, material_id) {
    const a = mesh_id; // assume already BigInt
    const b = material_id ?? 0n; // also BigInt
    // Szudzik's pairing:
    if (a >= b) {
      return a * a + a + b;
    } else {
      return b * b + a;
    }
  }

  static new_task(mesh_id, entity, material_id = null, resort = true) {
    const key = this._get_task_key(mesh_id, material_id);
    let tasks_for_entity = this.entity_task_map.get(entity);
    if (tasks_for_entity?.has(key)) {
      return tasks_for_entity.get(key);
    }

    // 2) otherwise allocate & enqueue a brand-new task
    const task = this.tasks_allocator.allocate();
    MeshTask.init(task, mesh_id, entity, material_id);
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
    }

    profile_scope("sort_and_batch", () => {
      if (this.needs_sort) {
        this.batches.length = 0;
        this.object_instances.length = 0;
        this.material_buckets.length = 0;
        this.object_instance_allocator.reset();

        this.tasks.sort((a, b) => a.mesh_id - b.mesh_id - (a.material_id - b.material_id));

        let last_batch = null;
        for (let i = 0; i < this.tasks.length; i++) {
          const task = this.tasks[i];

          const last_batch_matches =
            last_batch &&
            last_batch.mesh_id === task.mesh_id &&
            last_batch.material_id === task.material_id;

          if (!last_batch_matches) {
            const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);
            if (!mesh || !mesh.index_buffer) {
              continue;
            }

            const batch = new IndirectDrawBatch();
            batch.mesh_id = task.mesh_id;
            batch.material_id = task.material_id;
            batch.index_buffer_id = Name.from(mesh.index_buffer.config.name);

            batch.base_instance = last_batch
              ? last_batch.base_instance + last_batch.instance_count
              : 0;
            batch.instance_count = task.entity.instance_count;

            batch.first_index = 0;
            batch.index_count = mesh.index_count;
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
              for (let l = 0, cnt = seg.count; l < cnt; l++) {
                const entry = this.object_instance_allocator.allocate();
                entry.batch_index = i;
                entry.row = EntityID.make_row_field(start + l, cidx);
                this.object_instances.push(entry);
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
        for (let i = 0; i < this.indirect_draw_objects.length; i++) {
          const obj = this.indirect_draw_objects.get(i);
          if (obj && obj.indirect_draw_data) {
            obj.indirect_draw_data.fill(0);
          }
        }
      }

      this.object_instance_buffer.update_buffers(this.object_instances, this.needs_sort);

      for (let i = 0; i < this.indirect_draw_objects.length; i++) {
        const obj = this.indirect_draw_objects.get(i);
        if (obj && obj.indirect_draw_data) {
          obj.update_buffers(this.batches, this.object_instances, this.needs_sort);
        }
      }

      this.needs_sort = false;
    });
  }

  static remove(entity, resort = true) {
    const tasks_for_entity = this.entity_task_map.get(entity);
    if (!tasks_for_entity) return;

    // keep only those tasks whose key is NOT in tasks_for_entity
    this.tasks = this.tasks.filter((task) => {
      if (task.entity !== entity) return true;
      const key = this._get_task_key(task.mesh_id, task.material_id);
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
   * Get the object instance buffer.
   */
  static get_object_instance_buffer() {
    return this.object_instance_buffer.object_instance_buffer;
  }

  /**
   * Get the visible object instance buffer without occlusion.
   */
  static get_visible_instance_buffer_no_occlusion(view_index = 0) {
    return this.get_indirect_draw_object(view_index).visible_instance_buffer_no_occlusion;
  }

  /**
   * Get the visible object instance buffer.
   */
  static get_visible_instance_buffer(view_index = 0) {
    return this.get_indirect_draw_object(view_index).visible_instance_buffer;
  }

  /**
   * Get the indirect draw buffer for a specific view.
   */
  static get_indirect_draw_buffer(view_index = 0) {
    return this.get_indirect_draw_object(view_index).indirect_draw_buffer;
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
   * Get or create the IndirectDrawObject for a given view.
   */
  static get_indirect_draw_object(view_index = 0) {
    let obj = this.indirect_draw_objects.get(view_index);
    if (!obj) {
      obj = this.allocate_view_data(view_index);
    }
    return obj;
  }

  /**
   * Allocate view data for a given view index.
   */
  static allocate_view_data(view_index = 0) {
    const obj = this.indirect_draw_objects.allocate_at(view_index);
    obj.view_index = view_index;
    obj.init();
    this.needs_sort = true;
    return obj;
  }

  /**
   * Deallocate view data for a given view index.
   */
  static deallocate_view_data(view_index) {
    const obj = this.indirect_draw_objects.get(view_index);
    if (obj) {
      obj.destroy();
      this.indirect_draw_objects.deallocate_at(view_index);
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

  static submit_indexed_draws(render_pass, rg_frame_data, should_reset = false) {
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

      render_pass.pass.setIndexBuffer(
        mesh.index_buffer.buffer,
        mesh.index_buffer.config.element_type
      );
      render_pass.pass.drawIndexed(
        mesh.index_count,
        task.entity.instance_count,
        0,
        mesh.vertex_buffer_offset
      );
    });
    if (should_reset) {
      this.reset();
    }
  }

  static submit_indexed_indirect_draws(
    render_pass,
    rg_frame_data,
    should_reset = false,
    skip_material_bind = true,
    opaque_only = false,
    indirect_draw_buffer = null,
    view_index = 0
  ) {
    let last_material = null;

    const indirect_draw_object = this.get_indirect_draw_object(view_index);
    const indirect_buffer = indirect_draw_buffer ?? indirect_draw_object.indirect_draw_buffer;

    for (let i = 0; i < this.batches.length; ++i) {
      const batch = this.batches[i];
      const index_buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, batch.index_buffer_id);

      const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, batch.material_id);
      if (opaque_only && material.family !== MaterialFamilyType.Opaque) {
        continue;
      }

      if (!skip_material_bind) {
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
      }

      render_pass.pass.setIndexBuffer(index_buffer.buffer, index_buffer.config.element_type);
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
    rg_frame_data,
    material_id,
    should_reset = false,
    indirect_draw_buffer = null,
    view_index = 0
  ) {
    const material = ResourceCache.get().fetch(CacheTypes.MATERIAL, material_id);
    if (material) {
      // Material binds will rebind a pipeline state, so we need to rebind the bind groups here
      material.bind(render_pass, render_pass.frame_bind_groups, render_pass.frame_attachments);
      if (render_pass.frame_bind_groups[BindGroupType.Global]) {
        render_pass.frame_bind_groups[BindGroupType.Global].bind(render_pass);
      }
      if (render_pass.frame_bind_groups[BindGroupType.Pass]) {
        render_pass.frame_bind_groups[BindGroupType.Pass].bind(render_pass);
      }
    }

    const indirect_draw_object = this.get_indirect_draw_object(view_index);
    const indirect_buffer = indirect_draw_buffer ?? indirect_draw_object.indirect_draw_buffer;

    for (let i = 0; i < this.batches.length; ++i) {
      if (this.batches[i].material_id !== material_id) {
        continue;
      }
      const batch = this.batches[i];
      const index_buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, batch.index_buffer_id);

      render_pass.pass.setIndexBuffer(index_buffer.buffer, index_buffer.config.element_type);
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
    const mesh = Mesh.quad();
    render_pass.pass.setIndexBuffer(
      mesh.index_buffer.buffer,
      mesh.index_buffer.config.element_type
    );
    render_pass.pass.drawIndexed(mesh.index_count, instance_count, 0, mesh.vertex_buffer_offset);
  }

  static draw_cube(render_pass, instance_count = 1) {
    const mesh = Mesh.cube();
    render_pass.pass.setIndexBuffer(
      mesh.index_buffer.buffer,
      mesh.index_buffer.config.element_type
    );
    render_pass.pass.drawIndexed(mesh.index_count, instance_count, 0, mesh.vertex_buffer_offset);
  }
}

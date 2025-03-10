import { ResourceCache } from "./resource_cache.js";
import { Mesh } from "./mesh.js";
import { Buffer } from "./buffer.js";
import { Name } from "../utility/names.js";
import { FrameAllocator } from "../memory/allocator.js";
import { profile_scope } from "../utility/performance.js";
import { CacheTypes, MaterialFamilyType, BindGroupType } from "./renderer_types.js";

const max_objects = 5000000;
const max_frame_buffer_writes = 1000;

class IndirectDrawBatch {
  mesh_id = 0;
  material_id = 0;
  entity_ids = [];
  index_buffer_id = null;
  instance_count = 0;
  first_index = 0;
  index_count = 0;
  base_vertex = 0;
  base_instance = 0;
}

class ObjectInstanceEntry {
  constructor(batch_index, entity_index, entity_instance_index) {
    this.batch_index = batch_index;
    this.entity_index = entity_index;
    this.entity_instance_index = entity_instance_index;
  }
}

class IndirectDrawObject {
  indirect_draw_buffer = null;
  object_instance_buffer = null;
  compacted_object_instance_buffer = null;
  indirect_draw_data = new Uint32Array(max_objects * 5);
  object_instance_data = new Uint32Array(max_objects * 4);
  current_indirect_draw_write_offset = 0;
  current_object_instance_write_offset = 0;

  init() {
    profile_scope("init_indirect_draw_object", () => {
      if (!this.indirect_draw_buffer) {
        this.indirect_draw_buffer = Buffer.create({
          name: "indirect_draw_buffer",
          raw_data: this.indirect_draw_data,
          usage:
            GPUBufferUsage.INDIRECT |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.STORAGE,
        });
      }
      if (!this.object_instance_buffer) {
        this.object_instance_buffer = Buffer.create({
          name: "object_instance_buffer",
          raw_data: this.object_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
      if (!this.compacted_object_instance_buffer) {
        this.compacted_object_instance_buffer = Buffer.create({
          name: "compacted_object_instance_buffer",
          raw_data: new Uint32Array(max_objects * 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }
    });
  }

  // We assume this gets called once per frame
  update_buffers(batches, object_instances) {
    profile_scope("update_indirect_buffers", () => {
      // Resize indirect draw buffer if needed
      const required_indirect_draw_size = batches.length * 5 * 4; // 5 uint32 per batch, 4 bytes per uint32
      if (this.indirect_draw_buffer.size < required_indirect_draw_size) {
        this.indirect_draw_buffer.destroy();
        this.indirect_draw_buffer = Buffer.create({
          name: "indirect_draw_buffer",
          data: batches.map((batch) => [
            batch.index_count,
            0,
            batch.first_index,
            batch.base_vertex,
            batch.base_instance,
          ]),
          usage:
            GPUBufferUsage.INDIRECT |
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST,
        });
      }

      // Resize object instance buffer if needed
      const required_object_instance_size = object_instances.length * 4 * 4; // 4 uint32 per instance, 4 bytes per uint32
      if (this.object_instance_buffer.size < required_object_instance_size) {
        this.object_instance_buffer.destroy();
        this.object_instance_buffer = Buffer.create({
          name: "object_instance_buffer",
          data: object_instances.map((instance) => [
            instance.batch_index,
            instance.entity_index,
            instance.entity_instance_index,
            0, // padding
          ]),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }

      profile_scope("write_indirect_draw_buffer", () => {
        // Update indirect draw buffer
        let write_length = 0;
        let max_length = batches.length * 5;

        if (max_length > 0) {
          for (let i = 0; i < batches.length; i++) {
            const offset = i * 5;
            this.indirect_draw_data[offset] = batches[i].index_count;
            this.indirect_draw_data[offset + 1] = 0;
            this.indirect_draw_data[offset + 2] = batches[i].first_index;
            this.indirect_draw_data[offset + 3] = batches[i].base_vertex;
            this.indirect_draw_data[offset + 4] = batches[i].base_instance;
          }
          write_length = Math.min(max_length, max_frame_buffer_writes * 5);
          write_length = Math.min(
            write_length,
            max_length - this.current_indirect_draw_write_offset
          );
        }

        this.indirect_draw_buffer.write_raw(
          this.indirect_draw_data,
          this.current_indirect_draw_write_offset * 4,
          write_length,
          this.current_indirect_draw_write_offset
        );

        this.current_indirect_draw_write_offset += write_length;
        if (this.current_indirect_draw_write_offset >= max_length) {
          this.current_indirect_draw_write_offset = 0;
        }
      });

      profile_scope("write_object_instance_buffer", () => {
        // Update object instance buffer
        let write_length = 0;
        let max_length = object_instances.length * 4;

        if (max_length > 0) {
          for (let i = 0; i < object_instances.length; i++) {
            const offset = i * 3;
            this.object_instance_data[offset] = object_instances[i].batch_index;
            this.object_instance_data[offset + 1] =
              object_instances[i].entity_index;
            this.object_instance_data[offset + 2] =
              object_instances[i].entity_instance_index;
            this.object_instance_data[offset + 3] = 0; // padding
          }
          write_length = Math.min(
            write_length,
            max_length - this.current_object_instance_write_offset
          );
          write_length = Math.min(max_length, max_frame_buffer_writes * 4);
        }

        this.object_instance_buffer.write_raw(
          this.object_instance_data,
          this.current_object_instance_write_offset * 4,
          write_length,
          this.current_object_instance_write_offset
        );

        this.current_object_instance_write_offset += write_length;
        if (this.current_object_instance_write_offset >= max_length) {
          this.current_object_instance_write_offset = 0;
        }
      });
    });
  }
}

class MeshTask {
  mesh_id = null;
  entity = null;
  material_id = null;
  instance_count = 1;

  static init(task, mesh_id, entity, material_id = null, instance_count = 1) {
    task.mesh_id = mesh_id;
    task.entity = entity;
    task.material_id = material_id;
    task.instance_count = instance_count;
  }
}

export class MeshTaskQueue {
  static instance = null;
  constructor() {
    if (MeshTaskQueue.instance) {
      return MeshTaskQueue.instance;
    }
    this.tasks = [];
    this.batches = [];
    this.object_instances = [];
    this.material_buckets = [];
    this.indirect_draw_object = new IndirectDrawObject();
    this.tasks_allocator = new FrameAllocator(max_objects, MeshTask);
    this.object_instance_allocator = new FrameAllocator(
      max_objects,
      ObjectInstanceEntry
    );
    this.has_transparency = false;
    this.needs_sort = false;
    this.initialized = false;
  }

  static get() {
    if (!MeshTaskQueue.instance) {
      MeshTaskQueue.instance = new MeshTaskQueue();
    }
    return MeshTaskQueue.instance;
  }

  mark_needs_sort() {
    this.needs_sort = true;
  }

  reserve(num_tasks) {
    this.tasks.length = num_tasks;
  }

  reset() {
    this.tasks.length = 0;
    this.tasks_allocator.reset();
  }

  new_task(
    mesh_id,
    entity,
    material_id = null,
    instance_count = 1,
    resort = false
  ) {
    const task = this.tasks_allocator.allocate();

    MeshTask.init(task, mesh_id, entity, material_id, instance_count);

    this.tasks.push(task);

    if (resort) {
      this.needs_sort = true;
    }

    return task;
  }

  add_material_bucket(material_id) {
    if (!this.material_buckets.includes(material_id)) {
      this.material_buckets.push(material_id);
    }
  }

  sort_and_batch() {
    if (!this.initialized) {
      this.initialized = true;
      this.indirect_draw_object.init();
    }

    if (this.needs_sort) {
      this.tasks.sort(
        (a, b) => a.mesh_id - b.mesh_id - (a.material_id - b.material_id)
      );
    }

    profile_scope("sort_and_batch", () => {
      if (this.needs_sort) {
        this.batches.length = 0;
        this.object_instances.length = 0;
        this.material_buckets.length = 0;
        this.object_instance_allocator.reset();

        let last_batch = null;
        for (let i = 0; i < this.tasks.length; i++) {
          const task = this.tasks[i];
          if (
            last_batch &&
            last_batch.mesh_id === task.mesh_id &&
            last_batch.material_id === task.material_id
          ) {
            last_batch.instance_count += task.instance_count;
            last_batch.entity_ids.length += task.instance_count;
            const start_index =
              last_batch.entity_ids.length - task.instance_count;
            for (let j = 0; j < task.instance_count; j++) {
              last_batch.entity_ids[start_index + j] = task.entity;
            }
          } else {
            const mesh = ResourceCache.get().fetch(
              CacheTypes.MESH,
              task.mesh_id
            );

            const batch = new IndirectDrawBatch();
            batch.mesh_id = task.mesh_id;
            batch.material_id = task.material_id;
            batch.index_buffer_id = Name.from(mesh.index_buffer.config.name);
            batch.instance_count = task.instance_count;
            batch.first_index = 0;
            batch.index_count = mesh.index_count;
            batch.base_vertex = mesh.vertex_buffer_offset;
            batch.base_instance = last_batch
              ? last_batch.base_instance + last_batch.instance_count
              : 0;

            batch.entity_ids.length = task.instance_count;
            batch.entity_ids.fill(task.entity);

            last_batch = batch;

            this.batches.push(batch);
          }
        }

        // Sort batches by material id
        this.batches.sort((a, b) => a.material_id - b.material_id);

        for (let j = 0; j < this.batches.length; j++) {
          this.add_material_bucket(this.batches[j].material_id);

          let last_batch_entity_id = -1;
          let entity_instance_index = 0;
          for (let k = 0; k < this.batches[j].instance_count; k++) {
            if (this.batches[j].entity_ids[k] !== last_batch_entity_id) {
              entity_instance_index = 0;
            }
            last_batch_entity_id = this.batches[j].entity_ids[k];

            const object_instance = this.object_instance_allocator.allocate();
            object_instance.entity_instance_index = entity_instance_index++;
            object_instance.batch_index = j;
            object_instance.entity_index = last_batch_entity_id;
            this.object_instances.push(object_instance);
          }
        }

        // This ensures that we render transparent materials after opaque materials
        this.material_buckets.sort((a, b) => {
          const a_material = ResourceCache.get().fetch(CacheTypes.MATERIAL, a);
          const b_material = ResourceCache.get().fetch(CacheTypes.MATERIAL, b);
          return a_material.family - b_material.family;
        });

        this.has_transparency = this.material_buckets.find((bucket) => {
          const material = ResourceCache.get().fetch(
            CacheTypes.MATERIAL,
            bucket
          );
          return material.family === MaterialFamilyType.Transparent;
        });

        if (
          this.batches.length <= 0 &&
          this.indirect_draw_object.indirect_draw_data
        ) {
          this.indirect_draw_object.indirect_draw_data.fill(0);
        }
        if (
          this.object_instances.length <= 0 &&
          this.indirect_draw_object.object_instance_data
        ) {
          this.indirect_draw_object.object_instance_data.fill(0);
        }
      }

      this.needs_sort = false;

      this.indirect_draw_object.update_buffers(
        this.batches,
        this.object_instances
      );
    });
  }

  remove(entity) {
    // Use a single-pass, in-place removal algorithm
    let write_index = 0;
    for (let read_index = 0; read_index < this.tasks.length; read_index++) {
      if (this.tasks[read_index].entity !== entity) {
        if (write_index !== read_index) {
          this.tasks[write_index] = this.tasks[read_index];
        }
        write_index++;
      }
    }

    // Trim the array to the new size
    if (write_index < this.tasks.length) {
      this.tasks.length = write_index;
    }

    // Reset the batches and object instances if there are no tasks left
    if (this.tasks.length == 0) {
      this.batches.length = 0;
      this.object_instances.length = 0;
      this.material_buckets.length = 0;
    }
  }

  get_object_instance_buffer() {
    return this.indirect_draw_object.object_instance_buffer;
  }

  get_compacted_object_instance_buffer() {
    return this.indirect_draw_object.compacted_object_instance_buffer;
  }

  get_indirect_draw_buffer() {
    return this.indirect_draw_object.indirect_draw_buffer;
  }

  get_material_buckets() {
    return this.material_buckets;
  }

  get_total_draw_count() {
    return this.object_instances.length;
  }

  submit_draws(render_pass, rg_frame_data, should_reset = false) {
    let last_material = null;
    this.tasks.forEach((task) => {
      const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);

      const material = ResourceCache.get().fetch(
        CacheTypes.MATERIAL,
        task.material_id
      );
      if (material && material !== last_material) {
        // Material binds will rebind a pipeline state, so we need to rebind the bind groups here
        material.bind(
          render_pass,
          rg_frame_data.pass_bind_groups,
          rg_frame_data.pass_attachments
        );
        if (rg_frame_data.pass_bind_groups[BindGroupType.Global]) {
          rg_frame_data.pass_bind_groups[BindGroupType.Global].bind(
            render_pass
          );
        }
        if (rg_frame_data.pass_bind_groups[BindGroupType.Pass]) {
          rg_frame_data.pass_bind_groups[BindGroupType.Pass].bind(render_pass);
        }
        last_material = material;
      }

      render_pass.pass.draw(
        mesh.vertex_count,
        task.instance_count,
        mesh.vertex_buffer_offset
      );
    });
    if (should_reset) {
      this.reset();
    }
  }

  submit_indexed_draws(render_pass, rg_frame_data, should_reset = false) {
    let last_material = null;
    this.tasks.forEach((task) => {
      const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);

      const material = ResourceCache.get().fetch(
        CacheTypes.MATERIAL,
        task.material_id
      );
      if (material && material !== last_material) {
        // Material binds will rebind a pipeline state, so we need to rebind the bind groups here
        material.bind(
          render_pass,
          rg_frame_data.pass_bind_groups,
          rg_frame_data.pass_attachments
        );
        if (rg_frame_data.pass_bind_groups[BindGroupType.Global]) {
          rg_frame_data.pass_bind_groups[BindGroupType.Global].bind(
            render_pass
          );
        }
        if (rg_frame_data.pass_bind_groups[BindGroupType.Pass]) {
          rg_frame_data.pass_bind_groups[BindGroupType.Pass].bind(render_pass);
        }
        last_material = material;
      }

      render_pass.pass.setIndexBuffer(
        mesh.index_buffer.buffer,
        mesh.index_buffer.config.element_type
      );
      render_pass.pass.drawIndexed(
        mesh.index_count,
        task.instance_count,
        0,
        mesh.vertex_buffer_offset
      );
    });
    if (should_reset) {
      this.reset();
    }
  }

  submit_indexed_indirect_draws(
    render_pass,
    rg_frame_data,
    should_reset = false,
    skip_material_bind = true
  ) {
    let last_material = null;
    for (let i = 0; i < this.batches.length; ++i) {
      const batch = this.batches[i];
      const index_buffer = ResourceCache.get().fetch(
        CacheTypes.BUFFER,
        batch.index_buffer_id
      );

      const material = ResourceCache.get().fetch(
        CacheTypes.MATERIAL,
        batch.material_id
      );
      if (!skip_material_bind && material && material !== last_material) {
        // Material binds will rebind a pipeline state, so we need to rebind the bind groups here
        material.bind(
          render_pass,
          rg_frame_data.pass_bind_groups,
          rg_frame_data.pass_attachments
        );
        if (rg_frame_data.pass_bind_groups[BindGroupType.Global]) {
          rg_frame_data.pass_bind_groups[BindGroupType.Global].bind(
            render_pass
          );
        }
        if (rg_frame_data.pass_bind_groups[BindGroupType.Pass]) {
          rg_frame_data.pass_bind_groups[BindGroupType.Pass].bind(render_pass);
        }
        last_material = material;
      }

      render_pass.pass.setIndexBuffer(
        index_buffer.buffer,
        index_buffer.config.element_type
      );
      render_pass.pass.drawIndexedIndirect(
        this.indirect_draw_object.indirect_draw_buffer.buffer,
        i * 20 // 5 * 4 bytes per draw call
      );
    }
    if (should_reset) {
      this.reset();
    }
  }

  submit_material_indexed_indirect_draws(
    render_pass,
    rg_frame_data,
    material_id,
    should_reset = false
  ) {
    const material = ResourceCache.get().fetch(
      CacheTypes.MATERIAL,
      material_id
    );
    if (material) {
      // Material binds will rebind a pipeline state, so we need to rebind the bind groups here
      material.bind(
        render_pass,
        rg_frame_data.pass_bind_groups,
        rg_frame_data.pass_attachments
      );
      if (rg_frame_data.pass_bind_groups[BindGroupType.Global]) {
        rg_frame_data.pass_bind_groups[BindGroupType.Global].bind(render_pass);
      }
      if (rg_frame_data.pass_bind_groups[BindGroupType.Pass]) {
        rg_frame_data.pass_bind_groups[BindGroupType.Pass].bind(render_pass);
      }
    }

    for (let i = 0; i < this.batches.length; ++i) {
      if (this.batches[i].material_id !== material_id) {
        continue;
      }
      const batch = this.batches[i];
      const index_buffer = ResourceCache.get().fetch(
        CacheTypes.BUFFER,
        batch.index_buffer_id
      );

      render_pass.pass.setIndexBuffer(
        index_buffer.buffer,
        index_buffer.config.element_type
      );
      render_pass.pass.drawIndexedIndirect(
        this.indirect_draw_object.indirect_draw_buffer.buffer,
        i * 20 // 5 * 4 bytes per draw call
      );
    }
    if (should_reset) {
      this.reset();
    }
  }

  draw_quad(render_pass, instance_count = 1) {
    const mesh = Mesh.quad();
    render_pass.pass.setIndexBuffer(
      mesh.index_buffer.buffer,
      mesh.index_buffer.config.element_type
    );
    render_pass.pass.drawIndexed(
      mesh.index_count,
      instance_count,
      0,
      mesh.vertex_buffer_offset
    );
  }

  draw_cube(render_pass, instance_count = 1) {
    const mesh = Mesh.cube();
    render_pass.pass.setIndexBuffer(
      mesh.index_buffer.buffer,
      mesh.index_buffer.config.element_type
    );
    render_pass.pass.drawIndexed(
      mesh.index_count,
      instance_count,
      0,
      mesh.vertex_buffer_offset
    );
  }
}

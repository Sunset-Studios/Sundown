import { ResourceCache, CacheTypes } from "./resource_cache.js";
import { Mesh } from "./mesh.js";
import { Buffer } from "./buffer.js";
import { Name } from "../utility/names.js";
import { FrameAllocator } from "../memory/allocator.js";
import { profile_scope } from "../utility/performance.js";
import { BindGroupType } from "./bind_group.js";

const max_objects = 1000000;

class IndirectDrawBatch {
  mesh_id = 0;
  material_id = 0;
  entity_id = 0;
  index_buffer_id = null;
  instance_count = 0;
  first_index = 0;
  index_count = 0;
  base_vertex = 0;
  base_instance = 0;
}

class ObjectInstanceEntry {
  constructor(batch_index, entity_index, material_index) {
    this.batch_index = batch_index;
    this.entity_index = entity_index;
  }
}

class IndirectDrawObject {
  indirect_draw_buffer = null;
  object_instance_buffer = null;
  indirect_draw_data = new Uint32Array(max_objects * 5);
  object_instance_data = new Uint32Array(max_objects * 2);

  // We assume this gets called once per frame
  update_buffers(context, batches, object_instances) {
    profile_scope("update_indirect_buffers", () => {
      if (!this.indirect_draw_buffer) {
        for (let i = 0; i < batches.length; i++) {
          const offset = i * 5;
          this.indirect_draw_data[offset] = batches[i].index_count;
          this.indirect_draw_data[offset + 1] = 0;
          this.indirect_draw_data[offset + 2] = batches[i].first_index;
          this.indirect_draw_data[offset + 3] = batches[i].base_vertex;
          this.indirect_draw_data[offset + 4] = batches[i].base_instance;
        }

        this.indirect_draw_buffer = Buffer.create(context, {
          name: "indirect_draw_buffer",
          raw_data: this.indirect_draw_data,
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        });
      }
      if (!this.object_instance_buffer) {
        for (let i = 0; i < object_instances.length; i++) {
          const offset = i * 2;
          this.object_instance_data[offset] = object_instances[i].batch_index;
          this.object_instance_data[offset + 1] = object_instances[i].entity_index;
        }

        this.object_instance_buffer = Buffer.create(context, {
          name: "object_instance_buffer",
          raw_data: this.object_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }

      // Resize indirect draw buffer if needed
      const required_indirect_draw_size = batches.length * 5 * 4; // 5 uint32 per batch, 4 bytes per uint32
      if (this.indirect_draw_buffer.size < required_indirect_draw_size) {
        ResourceCache.get().remove(
          CacheTypes.BUFFER,
          this.indirect_draw_buffer.physical_id
        );
        this.indirect_draw_buffer = Buffer.create(context, {
          name: "indirect_draw_buffer",
          data: batches.map((batch) => [
            batch.index_count,
            0,
            batch.first_index,
            batch.base_vertex,
            batch.base_instance,
          ]),
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }

      // Resize object instance buffer if needed
      const required_object_instance_size = object_instances.length * 2 * 4; // 3 uint32 per instance, 4 bytes per uint32
      if (this.object_instance_buffer.size < required_object_instance_size) {
        ResourceCache.get().remove(
          CacheTypes.BUFFER,
          this.object_instance_buffer.physical_id
        );
        this.object_instance_buffer = Buffer.create(context, {
          name: "object_instance_buffer",
          data: object_instances.map((instance) => [
            instance.batch_index,
            instance.entity_index,
          ]),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }

      profile_scope("write_indirect_draw_buffer", () => {
        // Update indirect draw buffer
        for (let i = 0; i < batches.length; i++) {
          const offset = i * 5;
          this.indirect_draw_data[offset] = batches[i].index_count;
          this.indirect_draw_data[offset + 1] = 0;
          this.indirect_draw_data[offset + 2] = batches[i].first_index;
          this.indirect_draw_data[offset + 3] = batches[i].base_vertex;
          this.indirect_draw_data[offset + 4] = batches[i].base_instance;
        }
        this.indirect_draw_buffer.write_raw(context, this.indirect_draw_data, 0, batches.length * 5);
      });

      profile_scope("write_object_instance_buffer", () => {
        // Update object instance buffer
        for (let i = 0; i < object_instances.length; i++) {
          const offset = i * 2;
          this.object_instance_data[offset] = object_instances[i].batch_index;
          this.object_instance_data[offset + 1] = object_instances[i].entity_index;
        }
        this.object_instance_buffer.write_raw(context, this.object_instance_data, 0, object_instances.length * 2);
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
  constructor() {
    if (MeshTaskQueue.instance) {
      return MeshTaskQueue.instance;
    }
    this.tasks = [];
    this.batches = [];
    this.object_instances = [];
    this.material_buckets = new Set();
    this.current_task_offset = 0;
    this.needs_sort = false;
    this.indirect_draw_object = new IndirectDrawObject();
    this.tasks_allocator = new FrameAllocator(max_objects, new MeshTask());
    this.object_instance_allocator = new FrameAllocator(max_objects, new ObjectInstanceEntry());
    MeshTaskQueue.instance = this;
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
    this.current_task_offset = 0;
    this.tasks_allocator.reset();
  }

  new_task(mesh_id, entity, material_id = null, instance_count = 1, resort = false) {
    const task = this.tasks_allocator.allocate();

    MeshTask.init(task, mesh_id, entity, material_id, instance_count);

    if (this.current_task_offset >= this.tasks.length) {
      this.tasks.push(task);
    } else {
      this.tasks[this.current_task_offset++] = task;
    }

    if (resort) {
      this.needs_sort = true;
    }

    return task;
  }

  sort_and_batch(context) {
    if (!this.needs_sort) {
      return;
    }

    profile_scope("sort_and_batch", () => {
      this.tasks.sort(
        (a, b) => a.mesh_id - b.mesh_id - (a.material_id - b.material_id)
      );

      this.batches = [];
      this.object_instances = [];
      this.object_instance_allocator.reset();

      this.tasks.forEach((task, task_index) => {
        const last_batch = this.batches[this.batches.length - 1];
        if (
          last_batch &&
          last_batch.mesh_id === task.mesh_id &&
          last_batch.material_id === task.material_id
        ) {
          last_batch.instance_count += 1;
        } else {
          const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);

          const batch = new IndirectDrawBatch();
          batch.mesh_id = task.mesh_id;
          batch.material_id = task.material_id;
          batch.index_buffer_id = Name.from(mesh.index_buffer.config.name);
          batch.instance_count = task.instance_count;
          batch.first_index = 0;
          batch.index_count = mesh.indices.length;
          batch.base_vertex = mesh.vertex_buffer_offset;
          batch.base_instance = task_index;
          batch.entity_id = task.entity;

          this.batches.push(batch);
        }
      });

      // Sort batches by material id
      this.batches.sort((a, b) => a.material_id - b.material_id);

      this.material_buckets.clear();
      this.batches.forEach((batch, index) => {
        this.material_buckets.add(batch.material_id);
        for (let j = 0; j < batch.instance_count; j++) {
          const object_instance = this.object_instance_allocator.allocate();
          object_instance.batch_index = index;
          object_instance.entity_index = batch.entity_id + j;
          this.object_instances.push(object_instance);
        }
      });

      this.indirect_draw_object.update_buffers(
        context,
        this.batches,
        this.object_instances
      );
    });
  }

  remove(entity) {
    // Filter out tasks associated with the given entity
    this.tasks = this.tasks.filter(task => task.entity !== entity);
    
    // Update the current_task_offset
    this.current_task_offset = this.tasks.length;

    // Reset the batches and object instances if there are no tasks left
    if (this.tasks.length == 0) {
      this.batches = [];
      this.object_instances = [];
      this.material_buckets.clear();
    }
  }

  get_object_instance_buffer() {
    return this.indirect_draw_object.object_instance_buffer;
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
        mesh.vertices.length,
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
        mesh.indices.length,
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

    const material_batches = this.batches.filter(
      (batch) => batch.material_id === material_id
    );

    for (let i = 0; i < material_batches.length; ++i) {
      const batch = material_batches[i];
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

  draw_quad(context, render_pass) {
    const mesh = Mesh.quad(context);
    render_pass.pass.setIndexBuffer(
      mesh.index_buffer.buffer,
      mesh.index_buffer.config.element_type
    );
    render_pass.pass.drawIndexed(
      mesh.indices.length,
      1,
      0,
      mesh.vertex_buffer_offset
    );
  }

  draw_cube(context, render_pass) {
    const mesh = Mesh.cube(context);
    render_pass.pass.setIndexBuffer(
      mesh.index_buffer.buffer,
      mesh.index_buffer.config.element_type
    );
    render_pass.pass.drawIndexed(
      mesh.indices.length,
      1,
      0,
      mesh.vertex_buffer_offset
    );
  }
}

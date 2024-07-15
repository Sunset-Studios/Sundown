import { ResourceCache, CacheTypes } from "./resource_cache.js";
import { Mesh } from "./mesh.js";
import { Buffer } from "./buffer.js";
import { Name } from "../utility/names.js";
import { FrameAllocator } from "../memory/allocator.js";
import { profile_scope } from "../utility/performance.js";

class IndirectDrawBatch {
  mesh_id = 0;
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
    this.material_index = material_index;
  }
}

class IndirectDrawObject {
  batches = [];
  object_instances = [];
  indirect_draw_buffer = null;
  object_instance_buffer = null;

  // We assume this gets called once per frame
  update_buffers(context) {
    profile_scope("update_indirect_buffers", () => {
      if (!this.indirect_draw_buffer) {
        const indirect_draw_data = new Uint32Array(
          Math.max(this.batches.length * 5, 5)
        );
        this.batches.forEach((batch, index) => {
          const offset = index * 5;
          indirect_draw_data[offset] = batch.index_count;
          indirect_draw_data[offset + 1] = batch.instance_count;
          indirect_draw_data[offset + 2] = batch.first_index;
          indirect_draw_data[offset + 3] = batch.base_vertex;
          indirect_draw_data[offset + 4] = batch.base_instance;
        });

        this.indirect_draw_buffer = Buffer.create(context, {
          name: "indirect_draw_buffer",
          raw_data: indirect_draw_data,
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
      }
      if (!this.object_instance_buffer) {
        const object_instance_data = new Uint32Array(
          Math.max(this.object_instances.length * 3, 3)
        );
        this.object_instances.forEach((instance, index) => {
          const offset = index * 3;
          object_instance_data[offset] = instance.batch_index;
          object_instance_data[offset + 1] = instance.entity_index;
          object_instance_data[offset + 2] = instance.material_index;
        });

        this.object_instance_buffer = Buffer.create(context, {
          name: "object_instance_buffer",
          raw_data: object_instance_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }

      // Resize indirect draw buffer if needed
      const required_indirect_draw_size = this.batches.length * 5 * 4; // 5 uint32 per batch, 4 bytes per uint32
      if (this.indirect_draw_buffer.size < required_indirect_draw_size) {
        ResourceCache.get().remove(
          CacheTypes.BUFFER,
          this.indirect_draw_buffer.physical_id
        );
        this.indirect_draw_buffer = Buffer.create(context, {
          name: "indirect_draw_buffer",
          data: this.batches.map((batch) => [
            batch.index_count,
            batch.instance_count,
            batch.first_index,
            batch.base_vertex,
            batch.base_instance,
          ]),
          usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
      }

      // Resize object instance buffer if needed
      const required_object_instance_size =
        this.object_instances.length * 3 * 4; // 3 uint32 per instance, 4 bytes per uint32
      if (this.object_instance_buffer.size < required_object_instance_size) {
        ResourceCache.get().remove(
          CacheTypes.BUFFER,
          this.object_instance_buffer.physical_id
        );
        this.object_instance_buffer = Buffer.create(context, {
          name: "object_instance_buffer",
          data: this.object_instances.map((instance) => [
            instance.batch_index,
            instance.entity_index,
            instance.material_index,
          ]),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
      }

      // Update indirect draw buffer
      const indirect_draw_data = new Uint32Array(this.batches.length * 5);
      this.batches.forEach((batch, index) => {
        const offset = index * 5;
        indirect_draw_data[offset] = batch.index_count;
        indirect_draw_data[offset + 1] = batch.instance_count;
        indirect_draw_data[offset + 2] = batch.first_index;
        indirect_draw_data[offset + 3] = batch.base_vertex;
        indirect_draw_data[offset + 4] = batch.base_instance;
      });
      this.indirect_draw_buffer.write(context, indirect_draw_data);

      // Update object instance buffer
      const object_instance_data = new Uint32Array(
        this.object_instances.length * 3
      );
      this.object_instances.forEach((instance, index) => {
        const offset = index * 3;
        object_instance_data[offset] = instance.batch_index;
        object_instance_data[offset + 1] = instance.entity_index;
        object_instance_data[offset + 2] = instance.material_index;
      });
      this.object_instance_buffer.write(context, object_instance_data);
    });
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
  constructor() {
    if (MeshTaskQueue.instance) {
      return MeshTaskQueue.instance;
    }
    const max_objects = 1000000;
    this.tasks = [];
    this.current_task_offset = 0;
    this.indirect_draw_object = new IndirectDrawObject();
    this.tasks_allocator = new FrameAllocator(max_objects, new MeshTask());
    MeshTaskQueue.instance = this;
  }

  static get() {
    if (!MeshTaskQueue.instance) {
      MeshTaskQueue.instance = new MeshTaskQueue();
    }
    return MeshTaskQueue.instance;
  }

  reserve(num_tasks) {
    this.tasks.length = num_tasks;
  }

  reset() {
    this.tasks.length = 0;
    this.current_task_offset = 0;
    this.tasks_allocator.reset();
  }

  new_task(mesh_id, entity) {
    const task = this.tasks_allocator.allocate();

    MeshTask.init(task, mesh_id, entity);

    if (this.current_task_offset >= this.tasks.length) {
      this.tasks.push(task);
    } else {
      this.tasks[this.current_task_offset++] = task;
    }

    return task;
  }

  sort_and_batch(context) {
    this.tasks.sort((a, b) => a.mesh_id < b.mesh_id);

    this.indirect_draw_object.batches = [];
    this.indirect_draw_object.object_instances = [];

    this.tasks.forEach((task, task_index) => {
      const last_batch =
        this.indirect_draw_object.batches[
          this.indirect_draw_object.batches.length - 1
        ];
      if (last_batch && last_batch.mesh_id === task.mesh_id) {
        last_batch.instance_count += 1;
      } else {
        const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);

        const batch = new IndirectDrawBatch();
        batch.mesh_id = task.mesh_id;
        batch.index_buffer_id = Name.from(mesh.index_buffer.config.name);
        batch.instance_count = 1;
        batch.first_index = 0;
        batch.index_count = mesh.indices.length;
        batch.base_vertex = mesh.vertex_buffer_offset;
        batch.base_instance = task_index;

        this.indirect_draw_object.batches.push(batch);
      }
    });

    for (let i = 0; i < this.indirect_draw_object.batches.length; i++) {
      const batch = this.indirect_draw_object.batches[i];
      for (let j = 0; j < batch.instance_count; j++) {
        const object_instance = new ObjectInstanceEntry(
          i,
          this.tasks[batch.base_instance + j].entity,
          this.tasks[batch.base_instance + j].material_id
        );
        this.indirect_draw_object.object_instances.push(object_instance);
      }
    }

    this.indirect_draw_object.update_buffers(context);
  }

  get_object_instance_buffer() {
    return this.indirect_draw_object.object_instance_buffer;
  }

  submit_draws(render_pass, should_reset = false) {
    this.tasks.forEach((task) => {
      const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);
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

  submit_indexed_draws(render_pass, should_reset = false) {
    this.tasks.forEach((task) => {
      const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);
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

  submit_indexed_indirect_draws(render_pass, should_reset = false) {
    for (let i = 0; i < this.indirect_draw_object.batches.length; ++i) {
      const batch = this.indirect_draw_object.batches[i];
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

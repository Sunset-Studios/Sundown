import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";

export class TransformFragment extends Fragment {
  static initialize() {
    this.data = {
      position: new Float32Array(4),
      rotation: new Float32Array(4),
      scale: new Float32Array(4),
      dirty: new Uint32Array(1),
      position_buffer: null,
      position_cpu_buffer: null,
      rotation_buffer: null,
      rotation_cpu_buffer: null,
      scale_buffer: null,
      scale_cpu_buffer: null,
      dirty_flags_buffer: null,
      transforms_buffer: null,
      inverse_transforms_buffer: null,
      bounds_data_buffer: null,
      gpu_data_dirty: true,
    };
    Renderer.get().on_post_render(this.on_post_render.bind(this));
    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

    Fragment.resize_array(this.data, "position", new_size, Float32Array, 4);
    Fragment.resize_array(this.data, "rotation", new_size, Float32Array, 4);
    Fragment.resize_array(this.data, "scale", new_size, Float32Array, 4);
    Fragment.resize_array(this.data, "dirty", new_size, Uint32Array, 1);

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static add_entity(entity, data) {
    super.add_entity(entity, data);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.update_entity_data(entity, {
      position: { x: 0, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0, w: 1 },
      scale: { x: 1, y: 1, z: 1 },
    });
  }

  static get_entity_data(entity) {
    return {
      position: {
        x: this.data.position[entity * 4],
        y: this.data.position[entity * 4 + 1],
        z: this.data.position[entity * 4 + 2],
      },
      rotation: {
        x: this.data.rotation[entity * 4],
        y: this.data.rotation[entity * 4 + 1],
        z: this.data.rotation[entity * 4 + 2],
        w: this.data.rotation[entity * 4 + 3],
      },
      scale: {
        x: this.data.scale[entity * 4],
        y: this.data.scale[entity * 4 + 1],
        z: this.data.scale[entity * 4 + 2],
      },
    };
  }

  static duplicate_entity_data(entity) {
    const data = this.get_entity_data(entity);
    return {
      position: { x: data.position.x, y: data.position.y, z: data.position.z },
      rotation: {
        x: data.rotation.x,
        y: data.rotation.y,
        z: data.rotation.z,
        w: data.rotation.w,
      },
      scale: { x: data.scale.x, y: data.scale.y, z: data.scale.z },
    };
  }

  static update_entity_data(entity, data) {
    if (!this.data) {
      this.initialize();
    }

    const context = Renderer.get().graphics_context;

    if (data.position) {
      this.data.position[entity * 4 + 0] = data.position.x;
      this.data.position[entity * 4 + 1] = data.position.y;
      this.data.position[entity * 4 + 2] = data.position.z;
      this.data.position[entity * 4 + 3] = 1.0;
      this.data.position_buffer.write_raw(
        context,
        this.data.position.subarray(entity * 4, entity * 4 + 4),
        entity * 4 * Float32Array.BYTES_PER_ELEMENT,
      );
    }
    if (data.rotation) {
      this.data.rotation[entity * 4 + 0] = data.rotation.x;
      this.data.rotation[entity * 4 + 1] = data.rotation.y;
      this.data.rotation[entity * 4 + 2] = data.rotation.z;
      this.data.rotation[entity * 4 + 3] = data.rotation.w;
      this.data.rotation_buffer.write_raw(
        context,
        this.data.rotation.subarray(entity * 4, entity * 4 + 4),
        entity * 4 * Float32Array.BYTES_PER_ELEMENT,
      );
    }
    if (data.scale) {
      this.data.scale[entity * 4 + 0] = data.scale.x;
      this.data.scale[entity * 4 + 1] = data.scale.y;
      this.data.scale[entity * 4 + 2] = data.scale.z;
      this.data.scale[entity * 4 + 3] = 0.0;
      this.data.scale_buffer.write_raw(
        context,
        this.data.scale.subarray(entity * 4, entity * 4 + 4),
        entity * 4 * Float32Array.BYTES_PER_ELEMENT,
      );
    }

    this.data.dirty[entity] = 1;
    this.data.dirty_flags_buffer.write_raw(
      context,
      this.data.dirty.subarray(entity, entity + 1),
      entity * Uint32Array.BYTES_PER_ELEMENT,
    );
  }

  static to_gpu_data(context) {
    if (!this.data) {
      this.initialize();
    }

    return {
      transforms_buffer: this.data.transforms_buffer,
      inverse_transforms_buffer: this.data.inverse_transforms_buffer,
      bounds_data_buffer: this.data.bounds_data_buffer,
      position_buffer: this.data.position_buffer,
      rotation_buffer: this.data.rotation_buffer,
      scale_buffer: this.data.scale_buffer,
      dirty_flags_buffer: this.data.dirty_flags_buffer,
    };
  }

  static rebuild_buffers(context) {
    {
      const gpu_data = this.data.position
        ? this.data.position
        : new Float32Array(this.size * 4);
      if (
        !this.data.position_buffer ||
        this.data.position_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.position_buffer = Buffer.create(context, {
          name: "position_buffer",
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        this.data.position_cpu_buffer = Buffer.create(context, {
          name: "position_cpu_buffer",
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      } else {
        this.data.position_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch("position", this.data.position_buffer);
    }

    {
      const gpu_data = this.data.rotation
        ? this.data.rotation
        : new Float32Array(this.size * 4);
      if (
        !this.data.rotation_buffer ||
        this.data.rotation_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.rotation_buffer = Buffer.create(context, {
          name: "rotation_buffer",
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        this.data.rotation_cpu_buffer = Buffer.create(context, {
          name: "rotation_cpu_buffer",
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      } else {
        this.data.rotation_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch("rotation", this.data.rotation_buffer);
    }

    {
      const gpu_data = this.data.scale
        ? this.data.scale
        : new Float32Array(this.size * 4);
      if (
        !this.data.scale_buffer ||
        this.data.scale_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.scale_buffer = Buffer.create(context, {
          name: "scale_buffer",
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        this.data.scale_cpu_buffer = Buffer.create(context, {
          name: "scale_cpu_buffer",
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      } else {
        this.data.scale_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch("scale", this.data.scale_buffer);
    }

    {
      const gpu_data = this.data.dirty;

      if (
        !this.data.dirty_flags_buffer ||
        this.data.dirty_flags_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.dirty_flags_buffer = Buffer.create(context, {
          name: "dirty_flags_buffer",
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
      } else {
        this.data.dirty_flags_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch("dirty_flags", this.data.dirty_flags_buffer);
    }

    {
      const gpu_data = this.data.transforms
        ? this.data.transforms
        : new Float32Array(this.size * 32);
      if (
        !this.data.transforms_buffer ||
        this.data.transforms_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.transforms_buffer = Buffer.create(context, {
          name: "transforms_buffer",
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
      } else {
        this.data.transforms_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch("transforms", this.data.transforms_buffer);
    }

    {
      const gpu_data = this.data.inverse_transforms
        ? this.data.inverse_transforms
        : new Float32Array(this.size * 32);
      if (
        !this.data.inverse_transforms_buffer ||
        this.data.inverse_transforms_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.inverse_transforms_buffer = Buffer.create(context, {
          name: "inverse_transforms_buffer",
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
      } else {
        this.data.inverse_transforms_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch(
        "inverse_transforms",
        this.data.inverse_transforms_buffer,
      );
    }

    {
      const gpu_data = this.data.bounds_data
        ? this.data.bounds_data
        : new Float32Array(this.size * 8);
      if (
        !this.data.bounds_data_buffer ||
        this.data.bounds_data_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.bounds_data_buffer = Buffer.create(context, {
          name: "bounds_data_buffer",
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
      } else {
        this.data.bounds_data_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch("bounds_data", this.data.bounds_data_buffer);
    }

    this.data.gpu_data_dirty = false;

    const dirty_flags_buffer_size = this.data.dirty.byteLength;
    if (
      !this.data.dirty_flags_buffer ||
      this.data.dirty_flags_buffer.config.size < dirty_flags_buffer_size
    ) {
      this.data.dirty_flags_buffer = Buffer.create(context, {
        name: "transform_fragment_dirty_flags_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        raw_data: this.data.dirty,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    if (
      !this.data.transforms_buffer ||
      this.data.transforms_buffer.config.size <
        this.size * 32 * Float32Array.BYTES_PER_ELEMENT
    ) {
      this.data.transforms_buffer = Buffer.create(context, {
        name: "transform_fragment_transforms_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        raw_data: new Float32Array(this.size * 32),
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    if (
      !this.data.inverse_transforms_buffer ||
      this.data.inverse_transforms_buffer.config.size <
        this.size * 32 * Float32Array.BYTES_PER_ELEMENT
    ) {
      this.data.inverse_transforms_buffer = Buffer.create(context, {
        name: "transform_fragment_inverse_transforms_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        raw_data: new Float32Array(this.size * 32),
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    if (
      !this.data.bounds_data_buffer ||
      this.data.bounds_data_buffer.config.size <
        this.size * 8 * Float32Array.BYTES_PER_ELEMENT
    ) {
      this.data.bounds_data_buffer = Buffer.create(context, {
        name: "transform_fragment_bounds_data_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        raw_data: new Float32Array(this.size * 8),
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }
  }

  static async sync_buffers(context) {
    if (this.data.position_cpu_buffer?.buffer.mapState === "unmapped") {
      await this.data.position_cpu_buffer.read(
        context,
        this.data.position,
        this.data.position.byteLength,
        0,
        0,
        Float32Array,
      );
    }

    if (this.data.rotation_cpu_buffer?.buffer.mapState === "unmapped") {
      await this.data.rotation_cpu_buffer.read(
        context,
        this.data.rotation,
        this.data.rotation.byteLength,
        0,
        0,
        Float32Array,
      );
    }

    if (this.data.scale_cpu_buffer?.buffer.mapState === "unmapped") {
      await this.data.scale_cpu_buffer.read(
        context,
        this.data.scale,
        this.data.scale.byteLength,
        0,
        0,
        Float32Array,
      );
    }
  }

  static async on_post_render() {
    if (!this.data) {
      return;
    }

    await this.sync_buffers(Renderer.get().graphics_context);
  }
}

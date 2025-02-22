import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { EntityManager } from "../entity.js";
import { EntityTransformFlags } from "../../minimal.js";

const position_buffer_name = "position_buffer";
const position_cpu_buffer_name = "position_cpu_buffer";
const position_event = "position";
const position_update_event = "position_update";

const rotation_buffer_name = "rotation_buffer";
const rotation_cpu_buffer_name = "rotation_cpu_buffer";
const rotation_event = "rotation";
const rotation_update_event = "rotation_update";

const scale_buffer_name = "scale_buffer";
const scale_cpu_buffer_name = "scale_cpu_buffer";
const scale_event = "scale";
const scale_update_event = "scale_update";

const flags_buffer_name = "flags_buffer";
const flags_cpu_buffer_name = "flags_cpu_buffer";
const flags_event = "flags";
const flags_update_event = "flags_update";

const transforms_buffer_name = "transforms_buffer";
const transforms_cpu_buffer_name = "transforms_cpu_buffer";
const transforms_event = "transforms";
const transforms_update_event = "transforms_update";

const bounds_data_buffer_name = "bounds_data_buffer";
const bounds_data_cpu_buffer_name = "bounds_data_cpu_buffer";
const bounds_data_event = "bounds_data";
const bounds_data_update_event = "bounds_data_update";

class TransformDataView {
  current_entity = -1n;
  absolute_entity = -1n;

  constructor() {}

  get position() {
    return [
      TransformFragment.data.position[this.absolute_entity * 4],
      TransformFragment.data.position[this.absolute_entity * 4 + 1],
      TransformFragment.data.position[this.absolute_entity * 4 + 2],
    ];
  }

  set position(value) {
    TransformFragment.data.position[this.absolute_entity * 4] = value[0];
    TransformFragment.data.position[this.absolute_entity * 4 + 1] = value[1];
    TransformFragment.data.position[this.absolute_entity * 4 + 2] = value[2];
    TransformFragment.data.position[this.absolute_entity * 4 + 3] = 1.0;
    TransformFragment.data.flags[this.absolute_entity] |=
      EntityTransformFlags.DIRTY;
    TransformFragment.data.gpu_data_dirty = true;
  }

  get rotation() {
    return [
      TransformFragment.data.rotation[this.absolute_entity * 4],
      TransformFragment.data.rotation[this.absolute_entity * 4 + 1],
      TransformFragment.data.rotation[this.absolute_entity * 4 + 2],
      TransformFragment.data.rotation[this.absolute_entity * 4 + 3],
    ];
  }

  set rotation(value) {
    TransformFragment.data.rotation[this.absolute_entity * 4] = value[0];
    TransformFragment.data.rotation[this.absolute_entity * 4 + 1] = value[1];
    TransformFragment.data.rotation[this.absolute_entity * 4 + 2] = value[2];
    TransformFragment.data.rotation[this.absolute_entity * 4 + 3] = value[3];
    TransformFragment.data.flags[this.absolute_entity] |=
      EntityTransformFlags.DIRTY;
    TransformFragment.data.gpu_data_dirty = true;
  }

  get scale() {
    return [
      TransformFragment.data.scale[this.absolute_entity * 4],
      TransformFragment.data.scale[this.absolute_entity * 4 + 1],
      TransformFragment.data.scale[this.absolute_entity * 4 + 2],
    ];
  }

  set scale(value) {
    TransformFragment.data.scale[this.absolute_entity * 4] = value[0];
    TransformFragment.data.scale[this.absolute_entity * 4 + 1] = value[1];
    TransformFragment.data.scale[this.absolute_entity * 4 + 2] = value[2];
    TransformFragment.data.scale[this.absolute_entity * 4 + 3] = 0.0;
    TransformFragment.data.flags[this.absolute_entity] |=
      EntityTransformFlags.DIRTY;
    TransformFragment.data.gpu_data_dirty = true;
  }

  get flags() {
    return TransformFragment.data.flags[this.absolute_entity];
  }

  set flags(value) {
    TransformFragment.data.flags[this.absolute_entity] =
      TransformFragment.data.flags instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (TransformFragment.data.dirty) {
      TransformFragment.data.dirty[this.absolute_entity] = 1;
    }
    TransformFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;

    return this;
  }
}

const unmapped_state = "unmapped";

export class TransformFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, TransformDataView);
  static size = 0;
  static data = null;

  static initialize() {
    this.data = {
      position: new Float32Array(4),
      rotation: new Float32Array(4),
      scale: new Float32Array(4),
      flags: new Int32Array(1),
      position_buffer: null,
      position_cpu_buffer: null,
      rotation_buffer: null,
      rotation_cpu_buffer: null,
      scale_buffer: null,
      scale_cpu_buffer: null,
      flags_buffer: null,
      transforms_buffer: null,
      bounds_data_buffer: null,
      gpu_data_dirty: true,
    };
    Renderer.get().on_post_render(this.on_post_render.bind(this));

    this.rebuild_buffers();
  }

  static resize(new_size) {
    if (new_size <= this.size) return;

    this.size = new_size;

    if (!this.data) this.initialize();

    Fragment.resize_array(this.data, "position", new_size, Float32Array, 4);
    Fragment.resize_array(this.data, "rotation", new_size, Float32Array, 4);
    Fragment.resize_array(this.data, "scale", new_size, Float32Array, 4);
    Fragment.resize_array(this.data, "flags", new_size, Int32Array, 1);

    this.data.gpu_data_dirty = true;
  }

  static add_entity(entity) {
    const absolute_entity = EntityID.get_absolute_index(entity);
    if (absolute_entity >= this.size) {
      this.resize(absolute_entity * 2);
    }

    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const entity_instances = EntityID.get_instance_count(entity);
    for (let i = 0; i < entity_instances; i++) {
      this.data.position[(entity_offset + i) * 4] = 0;
      this.data.position[(entity_offset + i) * 4 + 1] = 0;
      this.data.position[(entity_offset + i) * 4 + 2] = 0;
      this.data.position[(entity_offset + i) * 4 + 3] = 1;
      this.data.rotation[(entity_offset + i) * 4] = 0;
      this.data.rotation[(entity_offset + i) * 4 + 1] = 0;
      this.data.rotation[(entity_offset + i) * 4 + 2] = 0;
      this.data.rotation[(entity_offset + i) * 4 + 3] = 0;
      this.data.scale[(entity_offset + i) * 4] = 1;
      this.data.scale[(entity_offset + i) * 4 + 1] = 1;
      this.data.scale[(entity_offset + i) * 4 + 2] = 1;
      this.data.scale[(entity_offset + i) * 4 + 3] = 0;
      this.data.flags[entity_offset + i] = 0;
    }
  }

  static get_entity_data(entity, instance = 0) {
    const data_view = this.data_view_allocator.allocate();
    data_view.view_entity(entity, instance);
    return data_view;
  }

  static duplicate_entity_data(entity, instance = 0) {
    const entity_offset = EntityID.get_absolute_index(entity);
    return {
      position: [
        this.data.position[entity_offset * 4],
        this.data.position[entity_offset * 4 + 1],
        this.data.position[entity_offset * 4 + 2],
      ],
      rotation: [
        this.data.rotation[entity_offset * 4],
        this.data.rotation[entity_offset * 4 + 1],
        this.data.rotation[entity_offset * 4 + 2],
        this.data.rotation[entity_offset * 4 + 3],
      ],
      scale: [
        this.data.scale[entity_offset * 4],
        this.data.scale[entity_offset * 4 + 1],
        this.data.scale[entity_offset * 4 + 2],
      ],
      flags: this.data.flags[entity_offset],
    };
  }

  static to_gpu_data() {
    if (!this.data) {
      this.initialize();
    }

    return {
      transforms_buffer: this.data.transforms_buffer,
      bounds_data_buffer: this.data.bounds_data_buffer,
      position_buffer: this.data.position_buffer,
      rotation_buffer: this.data.rotation_buffer,
      scale_buffer: this.data.scale_buffer,
      flags_buffer: this.data.flags_buffer,
    };
  }

  static rebuild_buffers() {
    if (!this.data.gpu_data_dirty) return;

    {
      const gpu_data = this.data.position
        ? this.data.position
        : new Float32Array(this.size * 4 + 4);
      if (
        !this.data.position_buffer ||
        this.data.position_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.position_buffer = Buffer.create({
          name: position_buffer_name,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        this.data.position_cpu_buffer = Buffer.create({
          name: position_cpu_buffer_name,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(position_event, this.data.position_buffer);
      } else {
        this.data.position_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(position_update_event);
    }

    {
      const gpu_data = this.data.rotation
        ? this.data.rotation
        : new Float32Array(this.size * 4 + 4);
      if (
        !this.data.rotation_buffer ||
        this.data.rotation_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.rotation_buffer = Buffer.create({
          name: rotation_buffer_name,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        this.data.rotation_cpu_buffer = Buffer.create({
          name: rotation_cpu_buffer_name,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(rotation_event, this.data.rotation_buffer);
      } else {
        this.data.rotation_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(rotation_update_event);
    }

    {
      const gpu_data = this.data.scale
        ? this.data.scale
        : new Float32Array(this.size * 4 + 4);
      if (
        !this.data.scale_buffer ||
        this.data.scale_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.scale_buffer = Buffer.create({
          name: scale_buffer_name,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        this.data.scale_cpu_buffer = Buffer.create({
          name: scale_cpu_buffer_name,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(scale_event, this.data.scale_buffer);
      } else {
        this.data.scale_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(scale_update_event);
    }

    {
      const gpu_data = this.data.flags
        ? this.data.flags
        : new Int32Array(this.size * 1 + 1);
      if (
        !this.data.flags_buffer ||
        this.data.flags_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.flags_buffer = Buffer.create({
          name: flags_buffer_name,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(flags_event, this.data.flags_buffer);
      } else {
        this.data.flags_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(flags_update_event);
    }

    {
      const gpu_data = this.data.transforms
        ? this.data.transforms
        : new Float32Array(this.size * 32 + 32);
      if (
        !this.data.transforms_buffer ||
        this.data.transforms_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.transforms_buffer = Buffer.create({
          name: transforms_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(
          transforms_event,
          this.data.transforms_buffer,
        );
      } else {
        this.data.transforms_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(transforms_update_event);
    }

    {
      const gpu_data = this.data.bounds_data
        ? this.data.bounds_data
        : new Float32Array(this.size * 8 + 8);
      if (
        !this.data.bounds_data_buffer ||
        this.data.bounds_data_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.bounds_data_buffer = Buffer.create({
          name: bounds_data_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(
          bounds_data_event,
          this.data.bounds_data_buffer,
        );
      } else {
        this.data.bounds_data_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(bounds_data_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers() {
    if (this.data.position_cpu_buffer?.buffer.mapState === unmapped_state) {
      await this.data.position_cpu_buffer.read(
        this.data.position,
        this.data.position.byteLength,
        0,
        0,
        Float32Array,
      );
    }

    if (this.data.rotation_cpu_buffer?.buffer.mapState === unmapped_state) {
      await this.data.rotation_cpu_buffer.read(
        this.data.rotation,
        this.data.rotation.byteLength,
        0,
        0,
        Float32Array,
      );
    }

    if (this.data.scale_cpu_buffer?.buffer.mapState === unmapped_state) {
      await this.data.scale_cpu_buffer.read(
        this.data.scale,
        this.data.scale.byteLength,
        0,
        0,
        Float32Array,
      );
    }
  }

  static copy_entity_instance(to_index, from_index) {
    this.data.position[to_index * 4 + 0] =
      this.data.position[from_index * 4 + 0];
    this.data.position[to_index * 4 + 1] =
      this.data.position[from_index * 4 + 1];
    this.data.position[to_index * 4 + 2] =
      this.data.position[from_index * 4 + 2];
    this.data.position[to_index * 4 + 3] =
      this.data.position[from_index * 4 + 3];

    this.data.rotation[to_index * 4 + 0] =
      this.data.rotation[from_index * 4 + 0];
    this.data.rotation[to_index * 4 + 1] =
      this.data.rotation[from_index * 4 + 1];
    this.data.rotation[to_index * 4 + 2] =
      this.data.rotation[from_index * 4 + 2];
    this.data.rotation[to_index * 4 + 3] =
      this.data.rotation[from_index * 4 + 3];

    this.data.scale[to_index * 4 + 0] = this.data.scale[from_index * 4 + 0];
    this.data.scale[to_index * 4 + 1] = this.data.scale[from_index * 4 + 1];
    this.data.scale[to_index * 4 + 2] = this.data.scale[from_index * 4 + 2];
    this.data.scale[to_index * 4 + 3] = this.data.scale[from_index * 4 + 3];

    this.data.flags[to_index * 1 + 0] = this.data.flags[from_index * 1 + 0];

    this.data.gpu_data_dirty = true;
  }

  static async on_post_render() {
    if (!this.data) {
      return;
    }

    await this.sync_buffers();
  }
}

import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { EntityManager } from "../entity.js";
import { MAX_BUFFERED_FRAMES } from "../../../core/minimal.js";
import { EntityTransformFlags } from "../../minimal.js";
import { AABB } from "../../../acceleration/aabb.js";
import { BufferSync } from "../../../renderer/buffer.js";

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

const aabb_node_index_buffer_name = "aabb_node_index_buffer";
const aabb_node_index_cpu_buffer_name = "aabb_node_index_cpu_buffer";
const aabb_node_index_event = "aabb_node_index";
const aabb_node_index_update_event = "aabb_node_index_update";

const flags_buffer_name = "flags_buffer";
const flags_cpu_buffer_name = "flags_cpu_buffer";
const flags_event = "flags";
const flags_update_event = "flags_update";

const dirty_buffer_name = "dirty_buffer";
const dirty_cpu_buffer_name = "dirty_cpu_buffer";
const dirty_event = "dirty";
const dirty_update_event = "dirty_update";

const transforms_buffer_name = "transforms_buffer";
const transforms_cpu_buffer_name = "transforms_cpu_buffer";
const transforms_event = "transforms";
const transforms_update_event = "transforms_update";

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
    TransformFragment.data.dirty[this.absolute_entity] = 1;
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
    TransformFragment.data.dirty[this.absolute_entity] = 1;
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
    TransformFragment.data.dirty[this.absolute_entity] = 1;
    TransformFragment.data.gpu_data_dirty = true;
  }

  get aabb_node_index() {
    return TransformFragment.data.aabb_node_index[this.absolute_entity];
  }

  set aabb_node_index(value) {
    TransformFragment.data.aabb_node_index[this.absolute_entity] =
      TransformFragment.data.aabb_node_index instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (TransformFragment.data.dirty) {
      TransformFragment.data.dirty[this.absolute_entity] = 1;
    }
    TransformFragment.data.gpu_data_dirty = true;
  }

  get transforms() {
    return [
      TransformFragment.data.transforms[this.absolute_entity * 32],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 1],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 2],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 3],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 4],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 5],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 6],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 7],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 8],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 9],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 10],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 11],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 12],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 13],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 14],
      TransformFragment.data.transforms[this.absolute_entity * 32 + 15],
    ];
  }

  set transforms(value) {
    TransformFragment.data.transforms[this.absolute_entity * 32] = value[0];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 1] = value[1];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 2] = value[2];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 3] = value[3];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 4] = value[4];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 5] = value[5];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 6] = value[6];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 7] = value[7];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 8] = value[8];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 9] = value[9];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 10] =
      value[10];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 11] =
      value[11];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 12] =
      value[12];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 13] =
      value[13];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 14] =
      value[14];
    TransformFragment.data.transforms[this.absolute_entity * 32 + 15] =
      value[15];
    TransformFragment.data.flags[this.absolute_entity] |=
      EntityTransformFlags.TRANSFORM_DIRTY;
    TransformFragment.data.gpu_data_dirty = true;
  }

  get flags() {
    return TransformFragment.data.flags[this.absolute_entity];
  }

  set flags(value) {
    TransformFragment.data.flags[this.absolute_entity] = value;
    TransformFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return TransformFragment.data.dirty[this.absolute_entity];
  }

  set dirty(value) {
    TransformFragment.data.dirty[this.absolute_entity] = value;
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

  static MAX_DIRTY_FLAG_RETAIN_FRAMES = 12;

  static initialize() {
    this.data = {
      position: new Float32Array(4),
      rotation: new Float32Array(4),
      scale: new Float32Array(4),
      aabb_node_index: new Uint32Array(1),
      transforms: new Float32Array(32),
      flags: new Int32Array(1),
      dirty: new Uint32Array(1),
      dirty_flag_retain_frames: 0,
      position_buffer: null,
      rotation_buffer: null,
      scale_buffer: null,
      aabb_node_index_buffer: null,
      flags_buffer: null,
      flags_cpu_buffer: Array(MAX_BUFFERED_FRAMES).fill(null),
      dirty_buffer: null,
      transforms_buffer: null,
      transforms_cpu_buffer: Array(MAX_BUFFERED_FRAMES).fill(null),
      gpu_data_dirty: true,
    };
    Renderer.get().on_post_render(this.on_post_render.bind(this));

    this.rebuild_buffers();
  }

  static resize(new_size) {
    if (new_size <= this.size) return;

    new_size *= 2;
    this.size = new_size;

    if (!this.data) this.initialize();

    Fragment.resize_array(this.data, "position", new_size, Float32Array, 4);
    Fragment.resize_array(this.data, "rotation", new_size, Float32Array, 4);
    Fragment.resize_array(this.data, "scale", new_size, Float32Array, 4);
    Fragment.resize_array(
      this.data,
      "aabb_node_index",
      new_size,
      Uint32Array,
      1,
    );
    Fragment.resize_array(this.data, "transforms", new_size, Float32Array, 32);
    Fragment.resize_array(this.data, "flags", new_size, Int32Array, 1);
    Fragment.resize_array(this.data, "dirty", new_size, Uint32Array, 1);

    this.data.gpu_data_dirty = true;
  }

  static add_entity(entity) {
    const absolute_entity = EntityID.get_absolute_index(entity);
    if (absolute_entity >= this.size) {
      this.resize(absolute_entity * 2);
    }

    this.data.flags[absolute_entity] |= EntityTransformFlags.VALID;

    const aabb_node_index = AABB.allocate_node(entity);
    this.data.aabb_node_index[absolute_entity] = aabb_node_index;

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
      this.data.aabb_node_index[entity_offset + i] = 0;
      this.data.flags[entity_offset + i] = 0;
      this.data.dirty[entity_offset + i] = 0;
    }
    this.data.gpu_data_dirty = true;
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
      aabb_node_index: this.data.aabb_node_index[entity_offset],
      flags: this.data.flags[entity_offset],
      dirty: this.data.dirty[entity_offset],
    };
  }

  static to_gpu_data() {
    if (!this.data) {
      this.initialize();
    }

    return {
      transforms_buffer: this.data.transforms_buffer,
      position_buffer: this.data.position_buffer,
      rotation_buffer: this.data.rotation_buffer,
      scale_buffer: this.data.scale_buffer,
      flags_buffer: this.data.flags_buffer,
      aabb_node_index_buffer: this.data.aabb_node_index_buffer,
      dirty_buffer: this.data.dirty_buffer,
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

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(scale_event, this.data.scale_buffer);
      } else {
        this.data.scale_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(scale_update_event);
    }

    {
      const gpu_data = this.data.aabb_node_index
        ? this.data.aabb_node_index
        : new Uint32Array(this.size * 1 + 1);
      if (
        !this.data.aabb_node_index_buffer ||
        this.data.aabb_node_index_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.aabb_node_index_buffer = Buffer.create({
          name: aabb_node_index_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(
          aabb_node_index_event,
          this.data.aabb_node_index_buffer,
        );
      } else {
        this.data.aabb_node_index_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(aabb_node_index_update_event);
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

        for (let i = 0; i < MAX_BUFFERED_FRAMES; i++) {
          this.data.flags_cpu_buffer[i] = Buffer.create({
            name: `flags_cpu_buffer_${i}`,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            raw_data: gpu_data,
            force: true,
          });
        }
        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(flags_event, this.data.flags_buffer);
      } else {
        this.data.flags_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(flags_update_event);
    }

    {
      const gpu_data = this.data.dirty
        ? this.data.dirty
        : new Uint32Array(this.size * 1 + 1);
      if (
        !this.data.dirty_buffer ||
        this.data.dirty_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.dirty_buffer = Buffer.create({
          name: dirty_buffer_name,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(dirty_event, this.data.dirty_buffer);
      } else {
        this.data.dirty_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(dirty_update_event);
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
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
          raw_data: gpu_data,
          force: true,
        });

        for (let i = 0; i < MAX_BUFFERED_FRAMES; i++) {
          this.data.transforms_cpu_buffer[i] = Buffer.create({
            name: `transforms_cpu_buffer_${i}`,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            raw_data: gpu_data,
            force: true,
          });
        }
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

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers() {
    const buffered_frame = Renderer.get().get_buffered_frame_number();

    if (
      this.data.flags_cpu_buffer[buffered_frame]?.buffer.mapState ===
      unmapped_state
    ) {
      await this.data.flags_cpu_buffer[buffered_frame].read(
        this.data.flags,
        this.data.flags.byteLength,
        0,
        0,
        Int32Array,
      );
    }

    if (
      this.data.transforms_cpu_buffer[buffered_frame]?.buffer.mapState ===
      unmapped_state
    ) {
      await this.data.transforms_cpu_buffer[buffered_frame].read(
        this.data.transforms,
        this.data.transforms.byteLength,
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

    this.data.aabb_node_index[to_index * 1 + 0] =
      this.data.aabb_node_index[from_index * 1 + 0];

    this.data.transforms[to_index * 32 + 0] =
      this.data.transforms[from_index * 32 + 0];
    this.data.transforms[to_index * 32 + 1] =
      this.data.transforms[from_index * 32 + 1];
    this.data.transforms[to_index * 32 + 2] =
      this.data.transforms[from_index * 32 + 2];
    this.data.transforms[to_index * 32 + 3] =
      this.data.transforms[from_index * 32 + 3];
    this.data.transforms[to_index * 32 + 4] =
      this.data.transforms[from_index * 32 + 4];
    this.data.transforms[to_index * 32 + 5] =
      this.data.transforms[from_index * 32 + 5];
    this.data.transforms[to_index * 32 + 6] =
      this.data.transforms[from_index * 32 + 6];
    this.data.transforms[to_index * 32 + 7] =
      this.data.transforms[from_index * 32 + 7];
    this.data.transforms[to_index * 32 + 8] =
      this.data.transforms[from_index * 32 + 8];
    this.data.transforms[to_index * 32 + 9] =
      this.data.transforms[from_index * 32 + 9];
    this.data.transforms[to_index * 32 + 10] =
      this.data.transforms[from_index * 32 + 10];
    this.data.transforms[to_index * 32 + 11] =
      this.data.transforms[from_index * 32 + 11];
    this.data.transforms[to_index * 32 + 12] =
      this.data.transforms[from_index * 32 + 12];
    this.data.transforms[to_index * 32 + 13] =
      this.data.transforms[from_index * 32 + 13];
    this.data.transforms[to_index * 32 + 14] =
      this.data.transforms[from_index * 32 + 14];
    this.data.transforms[to_index * 32 + 15] =
      this.data.transforms[from_index * 32 + 15];
    this.data.transforms[to_index * 32 + 16] =
      this.data.transforms[from_index * 32 + 16];
    this.data.transforms[to_index * 32 + 17] =
      this.data.transforms[from_index * 32 + 17];
    this.data.transforms[to_index * 32 + 18] =
      this.data.transforms[from_index * 32 + 18];
    this.data.transforms[to_index * 32 + 19] =
      this.data.transforms[from_index * 32 + 19];
    this.data.transforms[to_index * 32 + 20] =
      this.data.transforms[from_index * 32 + 20];
    this.data.transforms[to_index * 32 + 21] =
      this.data.transforms[from_index * 32 + 21];
    this.data.transforms[to_index * 32 + 22] =
      this.data.transforms[from_index * 32 + 22];
    this.data.transforms[to_index * 32 + 23] =
      this.data.transforms[from_index * 32 + 23];
    this.data.transforms[to_index * 32 + 24] =
      this.data.transforms[from_index * 32 + 24];
    this.data.transforms[to_index * 32 + 25] =
      this.data.transforms[from_index * 32 + 25];
    this.data.transforms[to_index * 32 + 26] =
      this.data.transforms[from_index * 32 + 26];
    this.data.transforms[to_index * 32 + 27] =
      this.data.transforms[from_index * 32 + 27];
    this.data.transforms[to_index * 32 + 28] =
      this.data.transforms[from_index * 32 + 28];
    this.data.transforms[to_index * 32 + 29] =
      this.data.transforms[from_index * 32 + 29];
    this.data.transforms[to_index * 32 + 30] =
      this.data.transforms[from_index * 32 + 30];
    this.data.transforms[to_index * 32 + 31] =
      this.data.transforms[from_index * 32 + 31];

    this.data.flags[to_index * 1 + 0] = this.data.flags[from_index * 1 + 0];

    this.data.dirty[to_index * 1 + 0] = this.data.dirty[from_index * 1 + 0];

    this.data.gpu_data_dirty = true;

    if (
      to_index > from_index &&
      this.data.flags[to_index] & EntityTransformFlags.VALID
    ) {
      this.data.aabb_node_index[to_index] = AABB.allocate_node(from_index);
    } else if (to_index < from_index) {
      if (this.data.aabb_node_index[to_index] !== 0) {
        AABB.free_node(this.data.aabb_node_index[to_index]);
      }
      this.data.aabb_node_index[to_index] =
        this.data.aabb_node_index[from_index];
    }
  }

  static get_world_position(entity, instance = 0) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const entity_index = entity_offset + instance;

    const translation_x = this.data.transforms[entity_index * 32 + 12];
    const translation_y = this.data.transforms[entity_index * 32 + 13];
    const translation_z = this.data.transforms[entity_index * 32 + 14];

    return [translation_x, translation_y, translation_z];
  }

  static get_world_rotation(entity, instance = 0) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const entity_index = entity_offset + instance;

    const m00 = this.data.transforms[entity_index * 32 + 0];
    const m01 = this.data.transforms[entity_index * 32 + 1];
    const m02 = this.data.transforms[entity_index * 32 + 2];
    const m10 = this.data.transforms[entity_index * 32 + 4];
    const m11 = this.data.transforms[entity_index * 32 + 5];
    const m12 = this.data.transforms[entity_index * 32 + 6];
    const m20 = this.data.transforms[entity_index * 32 + 8];
    const m21 = this.data.transforms[entity_index * 32 + 9];
    const m22 = this.data.transforms[entity_index * 32 + 10];

    const trace = m00 + m11 + m22;
    let qx, qy, qz, qw;

    if (trace > 0) {
      const s = 0.5 / Math.sqrt(trace + 1.0);
      qw = 0.25 / s;
      qx = (m21 - m12) * s;
      qy = (m02 - m20) * s;
      qz = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
      const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
      qw = (m21 - m12) / s;
      qx = 0.25 * s;
      qy = (m01 + m10) / s;
      qz = (m02 + m20) / s;
    } else if (m11 > m22) {
      const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
      qw = (m02 - m20) / s;
      qx = (m01 + m10) / s;
      qy = 0.25 * s;
      qz = (m12 + m21) / s;
    } else {
      const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
      qw = (m10 - m01) / s;
      qx = (m02 + m20) / s;
      qy = (m12 + m21) / s;
      qz = 0.25 * s;
    }

    return [qx, qy, qz, qw];
  }

  static get_world_scale(entity, instance = 0) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const entity_index = entity_offset + instance;

    const scale_x = Math.sqrt(
      this.data.transforms[entity_index * 32 + 0] *
        this.data.transforms[entity_index * 32 + 0] +
        this.data.transforms[entity_index * 32 + 1] *
          this.data.transforms[entity_index * 32 + 1] +
        this.data.transforms[entity_index * 32 + 2] *
          this.data.transforms[entity_index * 32 + 2],
    );

    const scale_y = Math.sqrt(
      this.data.transforms[entity_index * 32 + 4] *
        this.data.transforms[entity_index * 32 + 4] +
        this.data.transforms[entity_index * 32 + 5] *
          this.data.transforms[entity_index * 32 + 5] +
        this.data.transforms[entity_index * 32 + 6] *
          this.data.transforms[entity_index * 32 + 6],
    );

    const scale_z = Math.sqrt(
      this.data.transforms[entity_index * 32 + 8] *
        this.data.transforms[entity_index * 32 + 8] +
        this.data.transforms[entity_index * 32 + 9] *
          this.data.transforms[entity_index * 32 + 9] +
        this.data.transforms[entity_index * 32 + 10] *
          this.data.transforms[entity_index * 32 + 10],
    );

    return [scale_x, scale_y, scale_z];
  }

  static add_world_offset(entity, offset, instance = 0) {
    const entity_offset = EntityID.get_absolute_index(entity);
    const entity_index = entity_offset + instance;

    this.data.transforms[entity_index * 32 + 12] += offset[0];
    this.data.transforms[entity_index * 32 + 13] += offset[1];
    this.data.transforms[entity_index * 32 + 14] += offset[2];
  }

  static attempt_clear_all_dirty_flags() {
    ++this.data.dirty_flag_retain_frames;
    if (this.dirty_flag_retain_frames >= this.MAX_DIRTY_FLAG_RETAIN_FRAMES) {
      for (let i = 0; i < this.size; i++) {
        this.data.dirty[i] = 0;
      }
      this.data.gpu_data_dirty = true;
      this.data.dirty_flag_retain_frames = 0;
    }
  }

  static async on_post_render() {
    if (!this.data) {
      return;
    }
    BufferSync.request_sync(this);
  }
}

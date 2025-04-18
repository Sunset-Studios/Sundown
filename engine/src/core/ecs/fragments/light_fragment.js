import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";
import { EntityManager } from "../entity.js";
import { MAX_BUFFERED_FRAMES } from "../../../core/minimal.js";

const light_fragment_buffer_name = "light_fragment_buffer";
const light_fragment_cpu_buffer_name = "light_fragment_cpu_buffer";
const light_fragment_event = "light_fragment";
const light_fragment_update_event = "light_fragment_update";

class PositionDataView {
  constructor() {
    this.current_entity = -1n;
    this.absolute_entity = -1n;
  }

  get x() {
    return LightFragment.data.position.x[this.absolute_entity];
  }

  set x(value) {
    LightFragment.data.position.x[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get y() {
    return LightFragment.data.position.y[this.absolute_entity];
  }

  set y(value) {
    LightFragment.data.position.y[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get z() {
    return LightFragment.data.position.z[this.absolute_entity];
  }

  set z(value) {
    LightFragment.data.position.z[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;
    return this;
  }
}

class DirectionDataView {
  constructor() {
    this.current_entity = -1n;
    this.absolute_entity = -1n;
  }

  get x() {
    return LightFragment.data.direction.x[this.absolute_entity];
  }

  set x(value) {
    LightFragment.data.direction.x[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get y() {
    return LightFragment.data.direction.y[this.absolute_entity];
  }

  set y(value) {
    LightFragment.data.direction.y[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get z() {
    return LightFragment.data.direction.z[this.absolute_entity];
  }

  set z(value) {
    LightFragment.data.direction.z[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;
    return this;
  }
}

class ColorDataView {
  constructor() {
    this.current_entity = -1n;
    this.absolute_entity = -1n;
  }

  get r() {
    return LightFragment.data.color.r[this.absolute_entity];
  }

  set r(value) {
    LightFragment.data.color.r[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get g() {
    return LightFragment.data.color.g[this.absolute_entity];
  }

  set g(value) {
    LightFragment.data.color.g[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get b() {
    return LightFragment.data.color.b[this.absolute_entity];
  }

  set b(value) {
    LightFragment.data.color.b[this.absolute_entity] = value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;
    return this;
  }
}

class LightDataView {
  current_entity = -1n;
  absolute_entity = -1n;

  constructor() {
    this.position = new PositionDataView(this);
    this.direction = new DirectionDataView(this);
    this.color = new ColorDataView(this);
  }

  get type() {
    return LightFragment.data.type[this.absolute_entity];
  }

  set type(value) {
    LightFragment.data.type[this.absolute_entity] =
      LightFragment.data.type instanceof BigInt64Array ? BigInt(value) : value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get intensity() {
    return LightFragment.data.intensity[this.absolute_entity];
  }

  set intensity(value) {
    LightFragment.data.intensity[this.absolute_entity] =
      LightFragment.data.intensity instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get radius() {
    return LightFragment.data.radius[this.absolute_entity];
  }

  set radius(value) {
    LightFragment.data.radius[this.absolute_entity] =
      LightFragment.data.radius instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get attenuation() {
    return LightFragment.data.attenuation[this.absolute_entity];
  }

  set attenuation(value) {
    LightFragment.data.attenuation[this.absolute_entity] =
      LightFragment.data.attenuation instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get outer_angle() {
    return LightFragment.data.outer_angle[this.absolute_entity];
  }

  set outer_angle(value) {
    LightFragment.data.outer_angle[this.absolute_entity] =
      LightFragment.data.outer_angle instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get active() {
    return LightFragment.data.active[this.absolute_entity];
  }

  set active(value) {
    LightFragment.data.active[this.absolute_entity] =
      LightFragment.data.active instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return LightFragment.data.dirty[this.absolute_entity];
  }

  set dirty(value) {
    LightFragment.data.dirty[this.absolute_entity] =
      LightFragment.data.dirty instanceof BigInt64Array ? BigInt(value) : value;
    if (LightFragment.data.dirty) {
      LightFragment.data.dirty[this.absolute_entity] = 1;
    }
    LightFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity, instance = 0) {
    this.current_entity = entity;
    this.absolute_entity = EntityID.get_absolute_index(entity) + instance;

    this.position.view_entity(entity, instance);
    this.direction.view_entity(entity, instance);
    this.color.view_entity(entity, instance);

    return this;
  }
}

const unmapped_state = "unmapped";

export class LightFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(256, LightDataView);
  static size = 0;
  static data = null;

  static initialize() {
    this.data = {
      position: {
        x: new Float32Array(1),
        y: new Float32Array(1),
        z: new Float32Array(1),
      },
      direction: {
        x: new Float32Array(1),
        y: new Float32Array(1),
        z: new Float32Array(1),
      },
      color: {
        r: new Float32Array(1),
        g: new Float32Array(1),
        b: new Float32Array(1),
      },
      type: new Uint8Array(1),
      intensity: new Float32Array(1),
      radius: new Float32Array(1),
      attenuation: new Float32Array(1),
      outer_angle: new Float32Array(1),
      active: new Uint8Array(1),
      dirty: new Uint8Array(1),
      light_fragment_buffer: null,
      valid_prev: new Int32Array(1),
      valid_next: new Int32Array(1),
      first_valid_index: -1,
      last_valid_index: -1,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers();
  }

  static resize(new_size) {
    if (new_size <= this.size) return;

    new_size *= 2;
    this.size = new_size;

    if (!this.data) this.initialize();

    Object.keys(this.data.position).forEach((axis) => {
      Fragment.resize_array(this.data.position, axis, new_size, Float32Array);
    });
    Object.keys(this.data.direction).forEach((axis) => {
      Fragment.resize_array(this.data.direction, axis, new_size, Float32Array);
    });
    Object.keys(this.data.color).forEach((axis) => {
      Fragment.resize_array(this.data.color, axis, new_size, Float32Array);
    });
    Fragment.resize_array(this.data, "type", new_size, Uint8Array, 1);
    Fragment.resize_array(this.data, "intensity", new_size, Float32Array, 1);
    Fragment.resize_array(this.data, "radius", new_size, Float32Array, 1);
    Fragment.resize_array(this.data, "attenuation", new_size, Float32Array, 1);
    Fragment.resize_array(this.data, "outer_angle", new_size, Float32Array, 1);
    Fragment.resize_array(this.data, "active", new_size, Uint8Array, 1);
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array, 1);

    Fragment.resize_array(this.data, "valid_prev", new_size, Int32Array, 1);
    Fragment.resize_array(this.data, "valid_next", new_size, Int32Array, 1);

    this.data.gpu_data_dirty = true;
  }

  static add_entity(entity) {
    const absolute_entity = EntityID.get_absolute_index(entity);
    if (absolute_entity >= this.size) {
      this.resize(absolute_entity * 2);
    }

    const idx = Number(absolute_entity);
    const tail = this.data.last_valid_index;
    if (tail >= 0) {
      this.data.valid_next[tail] = idx;
      this.data.valid_prev[idx] = tail;
    } else {
      this.data.first_valid_index = idx;
      this.data.valid_prev[idx] = -1;
    }
    this.data.valid_next[idx] = -1;
    this.data.last_valid_index = idx;

    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    const instance_count = EntityID.get_instance_count(entity);
    const entity_offset = EntityID.get_absolute_index(entity);

    for (let i = 0; i < instance_count; ++i) {
      const entity_index = entity_offset + i;
      this.data.position.x[entity_index] =
        this.data.position instanceof BigInt64Array ? 0n : 0;
      this.data.position.y[entity_index] =
        this.data.position instanceof BigInt64Array ? 0n : 0;
      this.data.position.z[entity_index] =
        this.data.position instanceof BigInt64Array ? 0n : 0;

      this.data.direction.x[entity_index] =
        this.data.direction instanceof BigInt64Array ? 0n : 0;
      this.data.direction.y[entity_index] =
        this.data.direction instanceof BigInt64Array ? 0n : 0;
      this.data.direction.z[entity_index] =
        this.data.direction instanceof BigInt64Array ? 0n : 0;

      this.data.color.r[entity_index] =
        this.data.color instanceof BigInt64Array ? 0n : 0;
      this.data.color.g[entity_index] =
        this.data.color instanceof BigInt64Array ? 0n : 0;
      this.data.color.b[entity_index] =
        this.data.color instanceof BigInt64Array ? 0n : 0;

      this.data.type[entity_index] =
        this.data.type instanceof BigInt64Array ? 0n : 0;
      this.data.intensity[entity_index] =
        this.data.intensity instanceof BigInt64Array ? 0n : 0;
      this.data.radius[entity_index] =
        this.data.radius instanceof BigInt64Array ? 0n : 0;
      this.data.attenuation[entity_index] =
        this.data.attenuation instanceof BigInt64Array ? 0n : 0;
      this.data.outer_angle[entity_index] =
        this.data.outer_angle instanceof BigInt64Array ? 0n : 0;
      this.data.active[entity_index] =
        this.data.active instanceof BigInt64Array ? 0n : 0;
    }

    this.data.gpu_data_dirty = true;

    // unlink from the live‐list in O(1)
    const idx = Number(EntityID.get_absolute_index(entity));
    const p = this.data.valid_prev[idx];
    const n = this.data.valid_next[idx];

    if (p >= 0) {
      this.data.valid_next[p] = n;
    } else {
      this.data.first_valid_index = n;
    }

    if (n >= 0) {
      this.data.valid_prev[n] = p;
    } else {
      this.data.last_valid_index = p;
    }

    // clear pointers for safety
    this.data.valid_prev[idx] = -1;
    this.data.valid_next[idx] = -1;
  }

  static get_entity_data(entity, instance = 0) {
    const data_view = this.data_view_allocator.allocate();
    data_view.view_entity(entity, instance);
    return data_view;
  }

  static duplicate_entity_data(entity, instance = 0) {
    const data = {};
    const entity_index = EntityID.get_absolute_index(entity);
    data.position = {
      x: this.data.position.x[entity_index],
      y: this.data.position.y[entity_index],
      z: this.data.position.z[entity_index],
    };
    data.direction = {
      x: this.data.direction.x[entity_index],
      y: this.data.direction.y[entity_index],
      z: this.data.direction.z[entity_index],
    };
    data.color = {
      r: this.data.color.r[entity_index],
      g: this.data.color.g[entity_index],
      b: this.data.color.b[entity_index],
    };
    data.type = this.data.type[entity_index];
    data.intensity = this.data.intensity[entity_index];
    data.radius = this.data.radius[entity_index];
    data.attenuation = this.data.attenuation[entity_index];
    data.outer_angle = this.data.outer_angle[entity_index];
    data.active = this.data.active[entity_index];
    data.dirty = this.data.dirty[entity_index];
    return data;
  }

  static to_gpu_data() {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        light_fragment_buffer: this.data.light_fragment_buffer,
      };
    }

    this.rebuild_buffers();

    return {
      light_fragment_buffer: this.data.light_fragment_buffer,
    };
  }

  static rebuild_buffers() {
    if (!this.data.gpu_data_dirty) return;

    {
      let total_active = 0;
      for (let i = 0; i < this.size; i++) {
        if (this.data.active[i]) {
          total_active++;
        }
      }

      const gpu_data = new Float32Array(Math.max(total_active * 16, 16));
      let offset = 0;
      for (let i = 0; i < this.size; i++) {
        if (!this.data.active[i]) {
          continue;
        }

        gpu_data[offset] = this.data.position.x[i];
        gpu_data[offset + 1] = this.data.position.y[i];
        gpu_data[offset + 2] = this.data.position.z[i];
        gpu_data[offset + 3] = 0; // padding
        gpu_data[offset + 4] = this.data.direction.x[i];
        gpu_data[offset + 5] = this.data.direction.y[i];
        gpu_data[offset + 6] = this.data.direction.z[i];
        gpu_data[offset + 7] = 0; // padding
        gpu_data[offset + 8] = this.data.color.r[i];
        gpu_data[offset + 9] = this.data.color.g[i];
        gpu_data[offset + 10] = this.data.color.b[i];
        gpu_data[offset + 11] = this.data.type[i];
        gpu_data[offset + 12] = this.data.intensity[i];
        gpu_data[offset + 13] = this.data.radius[i];
        gpu_data[offset + 14] = this.data.attenuation[i];
        gpu_data[offset + 15] = this.data.outer_angle[i];

        offset += 16;
      }

      if (
        !this.data.light_fragment_buffer ||
        this.data.light_fragment_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.light_fragment_buffer = Buffer.create({
          name: light_fragment_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(
          light_fragment_event,
          this.data.light_fragment_buffer,
        );
      } else {
        this.data.light_fragment_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(light_fragment_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers() {
    const buffered_frame = Renderer.get().get_buffered_frame_number();
  }

  static copy_entity_instance(to_index, from_index) {
    this.data.position.x[to_index * 1 + 0] =
      this.data.position.x[from_index * 1 + 0];

    this.data.position.y[to_index * 1 + 0] =
      this.data.position.y[from_index * 1 + 0];

    this.data.position.z[to_index * 1 + 0] =
      this.data.position.z[from_index * 1 + 0];

    this.data.direction.x[to_index * 1 + 0] =
      this.data.direction.x[from_index * 1 + 0];

    this.data.direction.y[to_index * 1 + 0] =
      this.data.direction.y[from_index * 1 + 0];

    this.data.direction.z[to_index * 1 + 0] =
      this.data.direction.z[from_index * 1 + 0];

    this.data.color.r[to_index * 1 + 0] = this.data.color.r[from_index * 1 + 0];

    this.data.color.g[to_index * 1 + 0] = this.data.color.g[from_index * 1 + 0];

    this.data.color.b[to_index * 1 + 0] = this.data.color.b[from_index * 1 + 0];

    this.data.type[to_index * 1 + 0] = this.data.type[from_index * 1 + 0];

    this.data.intensity[to_index * 1 + 0] =
      this.data.intensity[from_index * 1 + 0];

    this.data.radius[to_index * 1 + 0] = this.data.radius[from_index * 1 + 0];

    this.data.attenuation[to_index * 1 + 0] =
      this.data.attenuation[from_index * 1 + 0];

    this.data.outer_angle[to_index * 1 + 0] =
      this.data.outer_angle[from_index * 1 + 0];

    this.data.active[to_index * 1 + 0] = this.data.active[from_index * 1 + 0];

    this.data.dirty[to_index * 1 + 0] = this.data.dirty[from_index * 1 + 0];

    this.data.gpu_data_dirty = true;
  }

  static get highest_entity() {
    return this.data.last_valid_index;
  }
}

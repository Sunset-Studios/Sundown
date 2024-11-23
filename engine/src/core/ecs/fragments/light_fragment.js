import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";

const light_fragment_buffer_name = "light_fragment_buffer";
const light_fragment_cpu_buffer_name = "light_fragment_cpu_buffer";
const light_fragment_event = "light_fragment";
const light_fragment_update_event = "light_fragment_update";

export class LightFragment extends Fragment {
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
      gpu_data_dirty: true,
    };

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

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

    this.rebuild_buffers(Renderer.get().graphics_context);
  }

  static add_entity(entity, data) {
    super.add_entity(entity, data);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.update_entity_data(entity, {
      position: {
        x: 0,
        y: 0,
        z: 0,
      },
      direction: {
        x: 0,
        y: 0,
        z: 0,
      },
      color: {
        r: 0,
        g: 0,
        b: 0,
      },
      type: 0,
      intensity: 0,
      radius: 0,
      attenuation: 0,
      outer_angle: 0,
      active: 0,
      dirty: 0,
    });
  }

  static get_entity_data(entity) {
    return super.get_entity_data(entity);
  }

  static duplicate_entity_data(entity) {
    const data = {};
    data.position = {
      x: this.data.position.x[entity],
      y: this.data.position.y[entity],
      z: this.data.position.z[entity],
    };
    data.direction = {
      x: this.data.direction.x[entity],
      y: this.data.direction.y[entity],
      z: this.data.direction.z[entity],
    };
    data.color = {
      r: this.data.color.r[entity],
      g: this.data.color.g[entity],
      b: this.data.color.b[entity],
    };
    data.type = this.data.type[entity];
    data.intensity = this.data.intensity[entity];
    data.radius = this.data.radius[entity];
    data.attenuation = this.data.attenuation[entity];
    data.outer_angle = this.data.outer_angle[entity];
    data.active = this.data.active[entity];
    data.dirty = this.data.dirty[entity];
    return data;
  }

  static update_entity_data(entity, data) {
    if (!this.data) {
      this.initialize();
    }

    super.update_entity_data(entity, data);

    if (this.data.dirty) {
      this.data.dirty[entity] = 1;
    }
    this.data.gpu_data_dirty = true;

    this.data.active[entity] = 1;
  }

  static to_gpu_data(context) {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        light_fragment_buffer: this.data.light_fragment_buffer,
      };
    }

    this.rebuild_buffers(context);

    return {
      light_fragment_buffer: this.data.light_fragment_buffer,
    };
  }

  static rebuild_buffers(context) {
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
        this.data.light_fragment_buffer = Buffer.create(context, {
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
        this.data.light_fragment_buffer.write(context, gpu_data);
      }

      global_dispatcher.dispatch(light_fragment_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers(context) {}
}

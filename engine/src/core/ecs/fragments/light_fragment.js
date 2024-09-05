import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { Fragment } from "../fragment.js";

export const LightType = {
  DIRECTIONAL: 0,
  POINT: 1,
  SPOT: 2,
  AREA: 3,
};

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
      dirty: new Uint8Array(1),
      active: new Uint8Array(1),
      gpu_buffer: null,
      gpu_data_dirty: true,
    };
  }

  static resize(new_size) {
    if (!this.data) {
      this.initialize();
    }

    super.resize(new_size);

    ["position", "direction"].forEach((prop) => {
      ["x", "y", "z"].forEach((axis) => {
        Fragment.resize_array(this.data[prop], axis, new_size);
      });
    });

    ["color"].forEach((prop) => {
      ["r", "g", "b"].forEach((axis) => {
        Fragment.resize_array(this.data[prop], axis, new_size);
      });
    });

    Fragment.resize_array(this.data, "type", new_size, Uint8Array);
    Fragment.resize_array(this.data, "intensity", new_size, Float32Array);
    Fragment.resize_array(this.data, "radius", new_size, Float32Array);
    Fragment.resize_array(this.data, "attenuation", new_size, Float32Array);
    Fragment.resize_array(this.data, "outer_angle", new_size, Float32Array);
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array);
    Fragment.resize_array(this.data, "active", new_size, Uint8Array);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.update_entity_data(entity, {
      position: {
        x: 0.0,
        y: 0.0,
        z: 0.0,
      },
      direction: {
        x: 0.0,
        y: 0.0,
        z: 0.0,
      },
      color: {
        r: 0.0,
        g: 0.0,
        b: 0.0,
      },
      type: 0,
      intensity: 0.0,
      radius: 0.0,
      attenuation: 0.0,
      outer_angle: 0.0,
      dirty: 0,
      active: 0,
    });
  }

  static update_entity_data(entity, data) {
    if (!this.data) {
      this.initialize();
    }

    super.update_entity_data(entity, data);
    this.data.dirty[entity] = 1;
    this.data.active[entity] = 1;
    this.data.gpu_data_dirty = true;
  }

  static to_gpu_data(context) {
    if (!this.data) {
      this.initialize();
    }

    if (!this.data.gpu_data_dirty) {
      return { gpu_buffer: this.data.gpu_buffer };
    }

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

    // Resize the buffer if necessary
    if (
      this.data.gpu_buffer &&
      this.data.gpu_buffer.config.size < gpu_data.byteLength
    ) {
      this.data.gpu_buffer.destroy(context);
      this.data.gpu_buffer = null;
    }

    if (!this.data.gpu_buffer) {
      this.data.gpu_buffer = Buffer.create(context, {
        name: "light_fragment_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        raw_data: gpu_data,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    } else {
      this.data.gpu_buffer.write(context, gpu_data);
    }

    this.data.gpu_data_dirty = false;

    return { gpu_buffer: this.data.gpu_buffer };
  }
}

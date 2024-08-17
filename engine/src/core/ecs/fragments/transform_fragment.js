import { Buffer } from "../../../renderer/buffer.js";
import { Fragment } from "../fragment.js";

export class TransformFragment extends Fragment {
  static initialize() {
    this.data = {
      position: {
        x: new Float32Array(1),
        y: new Float32Array(1),
        z: new Float32Array(1),
      },
      rotation: {
        x: new Float32Array(1),
        y: new Float32Array(1),
        z: new Float32Array(1),
      },
      scale: {
        x: new Float32Array(1),
        y: new Float32Array(1),
        z: new Float32Array(1),
      },
      prev_world_transform: new Float32Array(16),
      world_transform: new Float32Array(16),
      inverse_world_transform: new Float32Array(16),
      transpose_inverse_model_transform: new Float32Array(16),
      dirty: new Uint8Array(1),
      gpu_buffer: null,
      gpu_data_dirty: true,
    };
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.update_entity_data(entity, {
      position: { x: 0, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0 },
      scale: { x: 0, y: 0, z: 0 },
    });
  }

  static resize(new_size) {
    if (!this.data) {
      this.initialize();
    }

    super.resize(new_size);

    ["position", "rotation", "scale"].forEach((prop) => {
      ["x", "y", "z"].forEach((axis) => {
        Fragment.resize_array(this.data[prop], axis, new_size);
      });
    });

    Fragment.resize_array(this.data, "prev_world_transform", new_size, Float32Array, 16);
    Fragment.resize_array(this.data, "world_transform", new_size, Float32Array, 16);
    Fragment.resize_array(this.data, "inverse_world_transform", new_size, Float32Array, 16);
    Fragment.resize_array(this.data, "transpose_inverse_model_transform", new_size, Float32Array, 16);
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array);
  }

  static update_entity_data(entity, data) {
    if (!this.data) {
      this.initialize();
    }

    super.update_entity_data(entity, data);

    this.data.dirty[entity] = 1;
  }

  static to_gpu_data(context) {
    if (!this.data) {
      this.initialize();
    }

    if (!this.data.gpu_data_dirty) {
      return { gpu_buffer: this.data.gpu_buffer };
    }

    const gpu_data = new Float32Array(Math.max(this.size * 56, 56));
    for (let i = 0; i < this.size; i++) {
      const gpu_data_offset = i * 56;
      const transform_data_offset = i * 16;
      const vector_data_offset = i;
      for (let j = 0; j < 16; j++) {
        gpu_data[gpu_data_offset + j] =
          this.data.world_transform[transform_data_offset + j];
      }
      for (let j = 0; j < 16; j++) {
        gpu_data[gpu_data_offset + 16 + j] =
          this.data.inverse_world_transform[transform_data_offset + j];
      }
      for (let j = 0; j < 16; j++) {
        gpu_data[gpu_data_offset + 32 + j] =
          this.data.transpose_inverse_model_transform[transform_data_offset + j];
      }

      const scale = Math.sqrt(
        Math.pow(this.data.scale.x[vector_data_offset], 2.0) +
          Math.pow(this.data.scale.y[vector_data_offset], 2.0) +
          Math.pow(this.data.scale.z[vector_data_offset], 2.0)
      );

      gpu_data[gpu_data_offset + 48] = this.data.position.x[vector_data_offset];
      gpu_data[gpu_data_offset + 49] = this.data.position.y[vector_data_offset];
      gpu_data[gpu_data_offset + 50] = this.data.position.z[vector_data_offset];
      gpu_data[gpu_data_offset + 51] = scale;

      gpu_data[gpu_data_offset + 52] = 1.0; // TODO: bounds extent
      gpu_data[gpu_data_offset + 53] = 1.0; // TODO: bounds extent
      gpu_data[gpu_data_offset + 54] = 1.0; // TODO: bounds extent
      gpu_data[gpu_data_offset + 55] = 1.0; // TODO: bounds custom scale
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
        name: "transform_fragment_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        raw_data: gpu_data,
      });
    } else {
      this.data.gpu_buffer.write(context, gpu_data);
    }

    this.data.gpu_data_dirty = false;

    return { gpu_buffer: this.data.gpu_buffer };
  }
}

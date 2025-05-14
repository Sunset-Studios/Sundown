import { Fragment } from "../fragment.js";
import { SolarFragmentView } from "../solar/view.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Name } from "../../../utility/names.js";

/**
 * The Transform fragment class.
 * Use `EntityManager.get_fragment(entity, Transform)` to get a fragment instance for an entity.
 */
export class TransformFragment extends Fragment {
  static id = Name.from("transform");
  static field_key_map = new Map();
  static fields = {
    position: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: true,
      buffer_name: "position",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    rotation: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: true,
      buffer_name: "rotation",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    scale: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: true,
      buffer_name: "scale",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    aabb_node_index: {
      ctor: Uint32Array,
      elements: 1,
      default: 0,
      gpu_buffer: true,
      buffer_name: "aabb_node_index",
      is_container: false,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      cpu_readback: false,
    },
    transforms: {
      ctor: Float32Array,
      elements: 32,
      default: 0,
      gpu_buffer: true,
      buffer_name: "transforms",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
    },
  };
  static buffer_data = new Map(); // key → { buffer: FragmentGpuBuffer, stride: number }

  static get view_allocator() {
    if (!this._view_allocator) {
      this._view_allocator = new RingBufferAllocator(
        256,
        new SolarFragmentView(this),
      );
    }
    return this._view_allocator;
  }

  static is_valid() {
    return this.id && this.fields && this.view_allocator;
  }

  static get_buffer_name(field_name) {
    return this.field_key_map.get(field_name);
  }

  static get_world_position(entity, instance = 0) {
    const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    const transform = transform_fragment.transform;

    return [transform[12], transform[13], transform[14]];
  }

  static get_world_rotation(entity, instance = 0) {
    const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    const transform = transform_fragment.transform;

    const m00 = transform[0];
    const m01 = transform[1];
    const m02 = transform[2];
    const m10 = transform[4];
    const m11 = transform[5];
    const m12 = transform[6];
    const m20 = transform[8];
    const m21 = transform[9];
    const m22 = transform[10];

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
    const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    const transform = transform_fragment.transform;

    const scale_x = Math.sqrt(
      transform[0] * transform[0] +
        transform[1] * transform[1] +
        transform[2] * transform[2],
    );

    const scale_y = Math.sqrt(
      transform[4] * transform[4] +
        transform[5] * transform[5] +
        transform[6] * transform[6],
    );

    const scale_z = Math.sqrt(
      transform[8] * transform[8] +
        transform[9] * transform[9] +
        transform[10] * transform[10],
    );

    return [scale_x, scale_y, scale_z];
  }

  static add_world_offset(entity, offset, instance = 0) {
    const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    const transform = transform_fragment.transform;

    transform[12] += offset[0];
    transform[13] += offset[1];
    transform[14] += offset[2];

    transform_fragment.transform = transform;
  }
}

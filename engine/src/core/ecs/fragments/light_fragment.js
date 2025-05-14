import { Fragment } from "../fragment.js";
import { SolarFragmentView } from "../solar/view.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Name } from "../../../utility/names.js";

/**
 * The Light fragment class.
 * Use `EntityManager.get_fragment(entity, Light)` to get a fragment instance for an entity.
 */
export class LightFragment extends Fragment {
  static id = Name.from("light");
  static field_key_map = new Map();
  static fields = {
    position: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "position",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    direction: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "direction",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    color: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "color",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    type: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "type",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    intensity: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "intensity",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    radius: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "radius",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    attenuation: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "attenuation",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    outer_angle: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "outer_angle",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    active: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "active",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    padding1: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "padding1",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    padding2: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "padding2",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
  };
  static buffer_data = new Map(); // key → { buffer: FragmentGpuBuffer, stride: number }

  static gpu_buffers = {
    light_fragment: {
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      stride: 80,
      buffer_name: "light_fragment",
      cpu_readback: false,
      fields: [
        "position",
        "direction",
        "color",
        "type",
        "intensity",
        "radius",
        "attenuation",
        "outer_angle",
        "active",
        "padding1",
        "padding2",
      ],
    },
  };

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
}

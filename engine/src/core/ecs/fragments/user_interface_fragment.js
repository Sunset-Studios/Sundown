import { Fragment } from "../fragment.js";
import { SolarFragmentView } from "../solar/view.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Name } from "../../../utility/names.js";

/**
 * The UserInterface fragment class.
 * Use `EntityManager.get_fragment(entity, UserInterface)` to get a fragment instance for an entity.
 */
export class UserInterfaceFragment extends Fragment {
  static id = Name.from("user_interface");
  static field_key_map = new Map();
  static fields = {
    allows_cursor_events: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "allows_cursor_events",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    auto_size: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "auto_size",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    was_cursor_inside: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "was_cursor_inside",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    is_cursor_inside: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "is_cursor_inside",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    was_clicked: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "was_clicked",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    is_clicked: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "is_clicked",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    is_pressed: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "is_pressed",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    was_pressed: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "was_pressed",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    consume_events: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "consume_events",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    color: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: true,
      buffer_name: "color",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    emissive: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: true,
      buffer_name: "emissive",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    },
    rounding: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: true,
      buffer_name: "rounding",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
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
}

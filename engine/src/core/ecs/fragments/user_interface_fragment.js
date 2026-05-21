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
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    auto_size: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "auto_size",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    was_cursor_inside: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "was_cursor_inside",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    is_cursor_inside: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "is_cursor_inside",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    was_clicked: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "was_clicked",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    is_clicked: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "is_clicked",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    is_pressed: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "is_pressed",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    was_pressed: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "was_pressed",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    consume_events: {
      ctor: Uint8Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "consume_events",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_emissive: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_emissive",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_rounding: {
      ctor: Float32Array,
      elements: 1,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_rounding",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_origin: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_origin",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_x_axis: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_x_axis",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_y_axis: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_y_axis",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_uv_rect: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_uv_rect",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_color: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_color",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_border_color: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_border_color",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_params: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: false,
      buffer_name: "ui_params",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    ui_text_glyph: {
      ctor: Uint32Array,
      elements: 1,
      default: 0,
      gpu_buffer: true,
      buffer_name: "ui_text_glyph",
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
  };
  static buffer_data = new Map(); // key → { buffer: FragmentGpuBuffer, stride: number }

  static gpu_buffers = {
    ui_data: {
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      stride: 112,
      buffer_name: "ui_data",
      cpu_readback: false,
      fields: [
        "ui_origin",
        "ui_x_axis",
        "ui_y_axis",
        "ui_uv_rect",
        "ui_color",
        "ui_border_color",
        "ui_params",
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

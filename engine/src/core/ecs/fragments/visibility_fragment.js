import { Fragment } from "../fragment.js";
import { SolarFragmentView } from "../solar/view.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Name } from "../../../utility/names.js";

/**
 * The Visibility fragment class.
 * Use `EntityManager.get_fragment(entity, Visibility)` to get a fragment instance for an entity.
 */
export class VisibilityFragment extends Fragment {
  static id = Name.from("visibility");
  static field_key_map = new Map();
  static fields = {
    visible: {
      ctor: Uint32Array,
      elements: 1,
      default: 0,
      gpu_buffer: true,
      buffer_name: "visible",
      is_container: false,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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

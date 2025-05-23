import { Fragment } from "../fragment.js";
import { SolarFragmentView } from "../solar/view.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Name } from "../../../utility/names.js";

/**
 * The StaticMesh fragment class.
 * Use `EntityManager.get_fragment(entity, StaticMesh)` to get a fragment instance for an entity.
 */
export class StaticMeshFragment extends Fragment {
  static id = Name.from("static_mesh");
  static field_key_map = new Map();
  static fields = {
    mesh: {
      ctor: BigInt64Array,
      elements: 1,
      default: 0n,
      gpu_buffer: false,
      buffer_name: "mesh",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
    },
    material_slots: {
      ctor: BigInt64Array,
      elements: 16,
      default: 0n,
      gpu_buffer: false,
      buffer_name: "material_slots",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
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

  static material_slot_stride = 16;
}

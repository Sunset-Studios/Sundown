import { Fragment } from "../fragment.js";
import { SolarFragmentView } from "../solar/view.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { Name } from "../../../utility/names.js";
import { EntityFlags } from "../../minimal.js";
import { EntityManager } from "../entity.js";
import { DEFAULT_CHUNK_CAPACITY } from "../solar/types.js";
import { BVH } from "../../../acceleration/bvh.js";

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
      setter(value, typed_array, element_offset) {
        if (
          typed_array
            .subarray(element_offset, element_offset + 4)
            .every((v, i) => v === value[i])
        ) {
          return;
        }

        this.chunk.flags_meta[this.slot + this.instance] |= EntityFlags.MOVED;

        typed_array.set(value, element_offset);
      },
      cpu_readback: false,
      buffer_multiplier: 1,
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
      setter(value, typed_array, element_offset) {
        if (
          typed_array
            .subarray(element_offset, element_offset + 4)
            .every((v, i) => v === value[i])
        ) {
          return;
        }

        this.chunk.flags_meta[this.slot + this.instance] |= EntityFlags.MOVED;

        typed_array.set(value, element_offset);
      },
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    scale: {
      ctor: Float32Array,
      elements: 4,
      default: 1,
      gpu_buffer: true,
      buffer_name: "scale",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      setter(value, typed_array, element_offset) {
        if (
          typed_array
            .subarray(element_offset, element_offset + 4)
            .every((v, i) => v === value[i])
        ) {
          return;
        }

        this.chunk.flags_meta[this.slot + this.instance] |= EntityFlags.MOVED;

        typed_array.set(value, element_offset);
      },
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    bounds: {
      ctor: Float32Array,
      elements: 8,
      default: 0,
      gpu_buffer: true,
      buffer_name: "bounds",
      is_container: false,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      cpu_readback: false,
      buffer_multiplier: 2.1,
    },
    transforms: {
      ctor: Float32Array,
      elements: 48,
      default: 0,
      gpu_buffer: true,
      buffer_name: "transforms",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: false,
      buffer_multiplier: 1,
    },
    world_position: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: true,
      buffer_name: "world_position",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      buffer_multiplier: 1,
    },
    world_rotation: {
      ctor: Float32Array,
      elements: 4,
      default: 0,
      gpu_buffer: true,
      buffer_name: "world_rotation",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      buffer_multiplier: 1,
    },
    world_scale: {
      ctor: Float32Array,
      elements: 4,
      default: 1,
      gpu_buffer: true,
      buffer_name: "world_scale",
      is_container: false,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      buffer_multiplier: 1,
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
    return transform_fragment.world_position.slice(0, 3);
  }

  static get_world_rotation(entity, instance = 0) {
    const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    return transform_fragment.world_rotation.slice();
  }

  static get_world_scale(entity, instance = 0) {
    const transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );
    return transform_fragment.world_scale.slice(0, 3);
  }

  static add_world_offset(entity, offset, instance = 0) {
    const local_transform_fragment = EntityManager.get_fragment(
      entity,
      TransformFragment,
      instance,
    );

    let position = local_transform_fragment.position;
    position[0] += offset[0];
    position[1] += offset[1];
    position[2] += offset[2];
    local_transform_fragment.position = position;
  }
}

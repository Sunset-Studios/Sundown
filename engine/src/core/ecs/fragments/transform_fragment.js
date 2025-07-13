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
      default: 1,
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
      cpu_readback: false,
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

    local_transform_fragment.position[0] += offset[0];
    local_transform_fragment.position[1] += offset[1];
    local_transform_fragment.position[2] += offset[2];

    EntityManager.set_entity_dirty(entity, true);
  }
}

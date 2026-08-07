import { mat4, quat, vec3 } from "gl-matrix";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { FragmentGpuBuffer } from "../ecs/solar/memory.js";
import { ComputeTaskQueue } from "../../renderer/task_queues/compute_task_queue.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { SceneGraph } from "../scene_graph.js";
import { EntityFlags } from "../minimal.js";
import { compose_world_transform } from "../../utility/transform_utils.js";
import { profile_scope } from "../../utility/performance.js";

const transform_processor_update_scope_name = "TransformProcessor.update";
const position_buffer_name = "position";
const rotation_buffer_name = "rotation";
const scale_buffer_name = "scale";
const transform_buffer_name = "transforms";
const world_position_buffer_name = "world_position";
const world_rotation_buffer_name = "world_rotation";
const world_scale_buffer_name = "world_scale";
const transform_processing_task_name = "transform_processing";
const transform_processing_wgsl_path = "system_compute/transform_processing.wgsl";
const compact_transforms_buffer_name = "compact_transforms";
const compact_transform_processing_task_name = "compact_transform_processing";
const compact_transform_processing_wgsl_path =
  "system_compute/compact_transform_processing.wgsl";
const compact_transform_float_stride = 32;
const compact_transform_workgroup_size = 128;
const transform_relevant_flag_mask =
  EntityFlags.IGNORE_PARENT_SCALE |
  EntityFlags.IGNORE_PARENT_ROTATION |
  EntityFlags.INTERACTIVE;

export class TransformProcessor extends SimulationLayer {
  static compact_transforms_gpu_buffer = null;

  transform_processing_input_lists = [];
  transform_processing_output_lists = [];
  compact_transform_processing_inputs = new Array(2);
  compact_transform_processing_outputs = new Array(1);

  #dirty_root_entities = new Set();
  #entity_state_cache = new Map();
  #dirty_transform_chunks = new Set();

  #world_transform = mat4.create();
  #inverse_transform = mat4.create();
  #transpose_inverse_transform = mat4.create();
  #previous_transform = mat4.create();
  #world_position = vec3.create();
  #world_scale = vec3.create();
  #world_rotation = quat.create();
  _on_flags_changed = null;
  _on_delete = null;

  init() {
    if (!TransformProcessor.compact_transforms_gpu_buffer) {
      TransformProcessor.compact_transforms_gpu_buffer = new FragmentGpuBuffer(
        compact_transforms_buffer_name,
        FragmentGpuBuffer.initial_max_rows,
        compact_transform_float_stride * Float32Array.BYTES_PER_ELEMENT,
        false,
        true,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );
    }

    this._update_internal = this._update_internal.bind(this);
    this._on_flags_changed = this._on_flags_changed_internal.bind(this);
    this._on_delete = this._on_delete_internal.bind(this);
    EntityManager.on_flags_changed(this._on_flags_changed);
    EntityManager.on_delete(this._on_delete);
  }

  cleanup() {
    if (this._on_flags_changed) {
      EntityManager.off_flags_changed(this._on_flags_changed);
      this._on_flags_changed = null;
    }
  }

  post_update(delta_time) {
    super.post_update(delta_time);
    profile_scope(transform_processor_update_scope_name, this._update_internal);
  }

  static get_compact_transforms_buffer() {
    const compact_transforms = TransformProcessor.compact_transforms_gpu_buffer?.buffer;
    if (!compact_transforms) {
      throw new Error(
        "Compact transforms are unavailable before TransformProcessor initialization."
      );
    }
    return compact_transforms;
  }

  _update_internal() {
    this.#update_interactive_transforms();
    this.#queue_gpu_transform_tasks();
  }

  #update_interactive_transforms() {
    this.#dirty_transform_chunks.clear();
    this.#entity_state_cache.clear();
    this.#seed_dirty_roots();
    this.#process_dirty_interactive_subtrees();

    for (const chunk of this.#dirty_transform_chunks) {
      chunk.mark_dirty(transform_buffer_name);
      chunk.mark_dirty(world_position_buffer_name);
      chunk.mark_dirty(world_rotation_buffer_name);
      chunk.mark_dirty(world_scale_buffer_name);
    }

    SceneGraph.clear_hierarchy_dirty();
  }

  #seed_dirty_roots() {
    const transform_dirty_entities = TransformFragment.consume_dirty_entities();
    for (let i = 0; i < transform_dirty_entities.length; i++) {
      this.#dirty_root_entities.add(transform_dirty_entities[i]);
    }

    for (const entity of SceneGraph.hierarchy_dirty_entities) {
      this.#dirty_root_entities.add(entity);
    }
  }

  #process_dirty_interactive_subtrees() {
    if (this.#dirty_root_entities.size === 0) {
      return;
    }

    const visited_entities = new Set();
    const node_stack = Array.from(this.#dirty_root_entities);
    this.#dirty_root_entities.clear();

    while (node_stack.length > 0) {
      const entity = node_stack.pop();
      if (!entity || visited_entities.has(entity)) {
        continue;
      }
      visited_entities.add(entity);

      if (
        EntityManager.has_fragment(entity, TransformFragment) &&
        (EntityManager.get_entity_flags(entity) & EntityFlags.INTERACTIVE) !== 0
      ) {
        this.#get_entity_transform_state(entity);
      }

      const children = EntityManager.get_entity_children(entity);
      for (let i = 0; i < children.length; i++) {
        node_stack.push(children[i]);
      }
    }
  }

  #queue_gpu_transform_tasks() {
    const entity_index_map_buffer = FragmentGpuBuffer.entity_index_map_buffer;
    const positions = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      position_buffer_name
    );
    const rotations = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      rotation_buffer_name
    );
    const scales = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      scale_buffer_name
    );
    const transforms = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      transform_buffer_name
    );
    const compact_transforms = TransformProcessor.get_compact_transforms_buffer();
    const flags = FragmentGpuBuffer.entity_flags_buffer;

    if (
      !entity_index_map_buffer ||
      !positions ||
      !rotations ||
      !scales ||
      !transforms ||
      !flags ||
      !SceneGraph.scene_graph_buffer
    ) {
      return;
    }

    for (let i = 0; i < SceneGraph.scene_graph_layer_counts.length; ++i) {
      if (!SceneGraph.scene_graph_uniforms[i]) {
        continue;
      }

      if (this.transform_processing_input_lists.length <= i) {
        this.transform_processing_input_lists.push(new Array(8));
        this.transform_processing_output_lists.push(new Array(2));
      }

      this.transform_processing_input_lists[i][0] = positions.buffer;
      this.transform_processing_input_lists[i][1] = rotations.buffer;
      this.transform_processing_input_lists[i][2] = scales.buffer;
      this.transform_processing_input_lists[i][3] = transforms.buffer;
      this.transform_processing_input_lists[i][4] = flags.buffer;
      this.transform_processing_input_lists[i][5] = SceneGraph.scene_graph_buffer;
      this.transform_processing_input_lists[i][6] = SceneGraph.scene_graph_uniforms[i];
      this.transform_processing_input_lists[i][7] = entity_index_map_buffer.buffer;

      this.transform_processing_output_lists[i][0] = transforms.buffer;
      this.transform_processing_output_lists[i][1] = flags.buffer;

      const transform_dispatch_count = Math.max(
        1,
        Math.ceil(SceneGraph.scene_graph_layer_counts[i] / 256)
      );

      ComputeTaskQueue.new_task(
        transform_processing_task_name + i,
        transform_processing_wgsl_path,
        this.transform_processing_input_lists[i],
        this.transform_processing_output_lists[i],
        transform_dispatch_count
      );
    }

    this.compact_transform_processing_inputs[0] = transforms.buffer;
    this.compact_transform_processing_inputs[1] = compact_transforms;
    this.compact_transform_processing_outputs[0] = compact_transforms;

    ComputeTaskQueue.new_task(
      compact_transform_processing_task_name,
      compact_transform_processing_wgsl_path,
      this.compact_transform_processing_inputs,
      this.compact_transform_processing_outputs,
      Math.max(
        1,
        Math.ceil(EntityManager.get_max_rows() / compact_transform_workgroup_size)
      )
    );
  }

  #get_entity_transform_state(entity) {
    if (this.#entity_state_cache.has(entity)) {
      return this.#entity_state_cache.get(entity);
    }

    const parent_entity = EntityManager.get_entity_parent(entity);
    let parent_transform = null;
    let parent_dirty = false;

    if (parent_entity && EntityManager.has_fragment(parent_entity, TransformFragment)) {
      const parent_state = this.#get_entity_transform_state(parent_entity);
      parent_transform = parent_state.transform;
      parent_dirty = parent_state.dirty;
    }

    const hierarchy_dirty = parent_dirty || SceneGraph.is_hierarchy_dirty(entity);
    const is_interactive = (EntityManager.get_entity_flags(entity) & EntityFlags.INTERACTIVE) !== 0;
    const state = is_interactive
      ? this.#process_entity_transform(entity, parent_transform, hierarchy_dirty)
      : this.#resolve_entity_transform_state(entity, parent_transform, hierarchy_dirty);

    this.#entity_state_cache.set(entity, state);
    return state;
  }

  #resolve_entity_transform_state(entity, parent_transform, parent_dirty) {
    for (let segment_index = 0; segment_index < entity.segments.length; segment_index++) {
      const segment = entity.segments[segment_index];
      const transform_views = segment.chunk.get_fragment_view(TransformFragment);
      if (!transform_views || segment.count === 0) {
        continue;
      }

      const slot = segment.slot;
      const transform_offset = slot * 48;
      const flag = segment.chunk.flags_meta[slot];
      const local_transform_dirty = (flag & EntityFlags.TRANSFORM_DIRTY) !== 0;
      const has_cached_transform = transform_views.transforms[transform_offset + 15] === 1.0;
      const world_transform_dirty = parent_dirty || local_transform_dirty;

      if (!world_transform_dirty && has_cached_transform) {
        return {
          transform: transform_views.transforms.subarray(
            transform_offset,
            transform_offset + 16
          ),
          dirty: false,
        };
      }

      compose_world_transform(
        transform_views.position.subarray(slot * 4, slot * 4 + 4),
        transform_views.rotation.subarray(slot * 4, slot * 4 + 4),
        transform_views.scale.subarray(slot * 4, slot * 4 + 4),
        parent_transform,
        flag,
        this.#world_transform
      );

      return {
        transform: mat4.clone(this.#world_transform),
        dirty: world_transform_dirty,
      };
    }

    return {
      transform: parent_transform,
      dirty: parent_dirty,
    };
  }

  #process_entity_transform(entity, parent_transform, parent_dirty) {
    let first_instance_world_transform = parent_transform;
    let first_instance_dirty = parent_dirty;
    let found_first_instance = false;
    const hierarchy_moved = SceneGraph.is_hierarchy_moved(entity);

    for (let segment_index = 0; segment_index < entity.segments.length; segment_index++) {
      const segment = entity.segments[segment_index];
      const transform_views = segment.chunk.get_fragment_view(TransformFragment);
      if (!transform_views) {
        continue;
      }

      const positions = transform_views.position;
      const rotations = transform_views.rotation;
      const scales = transform_views.scale;
      const transforms = transform_views.transforms;
      const world_positions = transform_views.world_position;
      const world_rotations = transform_views.world_rotation;
      const world_scales = transform_views.world_scale;
      const flags = segment.chunk.flags_meta;
      let chunk_was_updated = false;

      for (let row = 0; row < segment.count; row++) {
        const slot = segment.slot + row;
        const position_offset = slot * 4;
        const rotation_offset = slot * 4;
        const scale_offset = slot * 4;
        const transform_offset = slot * 48;
        const flag = flags[slot];
        const interactive = (flag & EntityFlags.INTERACTIVE) !== 0;
        const has_cached_transform = transforms[transform_offset + 15] === 1.0;
        const local_transform_dirty = (flag & EntityFlags.TRANSFORM_DIRTY) !== 0;
        const world_transform_dirty =
          parent_dirty ||
          local_transform_dirty ||
          (interactive && !has_cached_transform);
        const needs_cpu_write = interactive && world_transform_dirty;

        if (!interactive || world_transform_dirty) {
          compose_world_transform(
            positions.subarray(position_offset, position_offset + 4),
            rotations.subarray(rotation_offset, rotation_offset + 4),
            scales.subarray(scale_offset, scale_offset + 4),
            parent_transform,
            flag,
            this.#world_transform
          );
        }

        if (needs_cpu_write) {
          if (has_cached_transform) {
            this.#previous_transform.set(
              transforms.subarray(transform_offset, transform_offset + 16)
            );
          } else {
            this.#previous_transform.set(this.#world_transform);
          }
          transforms.set(this.#world_transform, transform_offset);
          transforms.set(this.#previous_transform, transform_offset + 32);

          if (mat4.invert(this.#inverse_transform, this.#world_transform)) {
            mat4.transpose(this.#transpose_inverse_transform, this.#inverse_transform);
          } else {
            mat4.identity(this.#transpose_inverse_transform);
          }
          transforms.set(this.#transpose_inverse_transform, transform_offset + 16);

          mat4.getTranslation(this.#world_position, this.#world_transform);
          world_positions[position_offset + 0] = this.#world_position[0];
          world_positions[position_offset + 1] = this.#world_position[1];
          world_positions[position_offset + 2] = this.#world_position[2];
          world_positions[position_offset + 3] = 1.0;

          mat4.getScaling(this.#world_scale, this.#world_transform);
          world_scales[scale_offset + 0] = this.#world_scale[0];
          world_scales[scale_offset + 1] = this.#world_scale[1];
          world_scales[scale_offset + 2] = this.#world_scale[2];
          world_scales[scale_offset + 3] = 1.0;

          quat.identity(this.#world_rotation);
          mat4.getRotation(this.#world_rotation, this.#world_transform);
          world_rotations[rotation_offset + 0] = this.#world_rotation[0];
          world_rotations[rotation_offset + 1] = this.#world_rotation[1];
          world_rotations[rotation_offset + 2] = this.#world_rotation[2];
          world_rotations[rotation_offset + 3] = this.#world_rotation[3];

          flags[slot] =
            (flags[slot] |
              EntityFlags.DIRTY |
              (hierarchy_moved ? EntityFlags.MOVED : 0)) &
            ~EntityFlags.TRANSFORM_DIRTY;
          chunk_was_updated = true;
        }

        if (!found_first_instance) {
          if (interactive && !needs_cpu_write && has_cached_transform) {
            first_instance_world_transform = transforms.subarray(
              transform_offset,
              transform_offset + 16
            );
          } else if (interactive && has_cached_transform) {
            first_instance_world_transform = transforms.subarray(
              transform_offset,
              transform_offset + 16
            );
          } else {
            first_instance_world_transform = mat4.clone(this.#world_transform);
          }
          first_instance_dirty = world_transform_dirty;
          found_first_instance = true;
        }
      }

      if (chunk_was_updated) {
        this.#dirty_transform_chunks.add(segment.chunk);
      }
    }

    return {
      transform: first_instance_world_transform,
      dirty: first_instance_dirty,
    };
  }

  _on_flags_changed_internal(entity, previous_flags, next_flags) {
    const changed_mask = (previous_flags ^ next_flags) & transform_relevant_flag_mask;
    if (changed_mask === 0 || !EntityManager.has_fragment(entity, TransformFragment)) {
      return;
    }

    this.#dirty_root_entities.add(entity);
    this.#mark_entity_transform_dirty(
      entity,
      (changed_mask &
        (EntityFlags.IGNORE_PARENT_SCALE | EntityFlags.IGNORE_PARENT_ROTATION)) !==
      0
    );
  }

  #mark_entity_transform_dirty(entity, mark_moved = false) {
    const dirty_mask =
      EntityFlags.DIRTY |
      EntityFlags.TRANSFORM_DIRTY |
      (mark_moved ? EntityFlags.MOVED : 0);

    for (let i = 0; i < entity.segments.length; i++) {
      const segment = entity.segments[i];
      for (let j = 0; j < segment.count; j++) {
        segment.chunk.flags_meta[segment.slot + j] |= dirty_mask;
      }
      segment.chunk.mark_dirty();
    }
  }

  _on_delete_internal(entity) {
    this.#dirty_root_entities.delete(entity);
    this.#entity_state_cache.delete(entity);
  }
}

import { Renderer } from "../renderer/renderer.js";
import { Buffer } from "../renderer/buffer.js";
import { BufferFlags } from "../renderer/renderer_types.js";
import { Texture } from "../renderer/texture.js";
import { RingBufferAllocator } from "../memory/allocator.js";
import { ResizableBitArray, TypedStack } from "../memory/container.js";
import { MeshTaskQueue } from "../renderer/mesh_task_queue.js";
import { mat4, vec4, vec3, vec2 } from "gl-matrix";
import { WORLD_FORWARD, WORLD_UP } from "./minimal.js";
import { radians } from "../utility/math.js";

const view_buffer_name = "view_buffer";
const frame_info_buffer_name = "frame_info_buffer";
const temporal_jitter_sample_count = 2;
const temporal_jitter_sample_offsets = [
  [-0.25, 0.25],
  [0.25, -0.25],
];

function apply_projection_jitter(projection_matrix, jitter_ndc) {
  for (let col = 0; col < 4; col++) {
    const row_base = col * 4;
    projection_matrix[row_base + 0] += jitter_ndc[0] * projection_matrix[row_base + 3];
    projection_matrix[row_base + 1] += jitter_ndc[1] * projection_matrix[row_base + 3];
  }
}

export class SharedViewBuffer {
  // --- Field Offsets (in floats) ---
  static offsets = {
    view_matrix: 0,
    prev_view_matrix: 16,
    projection_matrix: 32,
    prev_projection_matrix: 48,
    view_projection_matrix: 64,
    inverse_view_projection_matrix: 80,
    prev_inverse_view_projection_matrix: 96,
    view_direction: 112,
    near: 116,
    far: 117,
    culling_enabled: 118,
    occlusion_enabled: 119,
    frustum: 120,
    view_position: 144,
    view_rotation: 148,
    view_right: 152,
    fov: 156,
    aspect_ratio: 157,
    distance_check_enabled: 158,
    velocity: 159,
    zoom: 163,
    clipmap_count: 164,
  };

  // --- Per-view Access Wrapper ---
  /**
   * JS wrapper to read/write parts of raw_data
   */
  static View = class {
    idx = 0;

    get base() {
      return this.idx * SharedViewBuffer.floats_per_view;
    }
    get view_matrix() {
      const f = SharedViewBuffer.offsets.view_matrix;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 16);
    }
    set view_matrix(m) {
      const f = SharedViewBuffer.offsets.view_matrix;
      SharedViewBuffer.raw_data.set(m, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get prev_view_matrix() {
      const f = SharedViewBuffer.offsets.prev_view_matrix;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 16);
    }
    set prev_view_matrix(m) {
      const f = SharedViewBuffer.offsets.prev_view_matrix;
      SharedViewBuffer.raw_data.set(m, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get projection_matrix() {
      const f = SharedViewBuffer.offsets.projection_matrix;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 16);
    }
    set projection_matrix(m) {
      const f = SharedViewBuffer.offsets.projection_matrix;
      SharedViewBuffer.raw_data.set(m, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get prev_projection_matrix() {
      const f = SharedViewBuffer.offsets.prev_projection_matrix;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 16);
    }
    set prev_projection_matrix(m) {
      const f = SharedViewBuffer.offsets.prev_projection_matrix;
      SharedViewBuffer.raw_data.set(m, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get view_projection_matrix() {
      const f = SharedViewBuffer.offsets.view_projection_matrix;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 16);
    }
    set view_projection_matrix(m) {
      const f = SharedViewBuffer.offsets.view_projection_matrix;
      SharedViewBuffer.raw_data.set(m, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get inverse_view_projection_matrix() {
      const f = SharedViewBuffer.offsets.inverse_view_projection_matrix;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 16);
    }
    set inverse_view_projection_matrix(m) {
      const f = SharedViewBuffer.offsets.inverse_view_projection_matrix;
      SharedViewBuffer.raw_data.set(m, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get prev_inverse_view_projection_matrix() {
      const f = SharedViewBuffer.offsets.prev_inverse_view_projection_matrix;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 16);
    }
    set prev_inverse_view_projection_matrix(m) {
      const f = SharedViewBuffer.offsets.prev_inverse_view_projection_matrix;
      SharedViewBuffer.raw_data.set(m, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get view_position() {
      const f = SharedViewBuffer.offsets.view_position;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 4);
    }
    set view_position(v) {
      const f = SharedViewBuffer.offsets.view_position;
      const pos = SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 4);
      if (v && !vec4.equals(pos, v)) {
        SharedViewBuffer.raw_data.set(v, this.base + f);
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get view_rotation() {
      const f = SharedViewBuffer.offsets.view_rotation;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 4);
    }
    set view_rotation(v) {
      const f = SharedViewBuffer.offsets.view_rotation;
      const rot = SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 4);
      if (v && !vec4.equals(rot, v)) {
        SharedViewBuffer.raw_data.set(v, this.base + f);
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get forward() {
      const f = SharedViewBuffer.offsets.view_direction;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 4);
    }
    set forward(v) {
      const f = SharedViewBuffer.offsets.view_direction;
      SharedViewBuffer.raw_data.set(v, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get right() {
      const f = SharedViewBuffer.offsets.view_right;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 4);
    }
    set right(v) {
      const f = SharedViewBuffer.offsets.view_right;
      SharedViewBuffer.raw_data.set(v, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get frustum() {
      const f = SharedViewBuffer.offsets.frustum;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 24);
    }
    set frustum(fr) {
      const f = SharedViewBuffer.offsets.frustum;
      SharedViewBuffer.raw_data.set(fr, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get fov() {
      const f = SharedViewBuffer.offsets.fov;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set fov(fo) {
      const f = SharedViewBuffer.offsets.fov;
      if (fo !== undefined && fo !== null && SharedViewBuffer.raw_data[this.base + f] !== fo) {
        SharedViewBuffer.raw_data[this.base + f] = fo;
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get aspect_ratio() {
      const f = SharedViewBuffer.offsets.aspect_ratio;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set aspect_ratio(a) {
      const f = SharedViewBuffer.offsets.aspect_ratio;
      if (a && SharedViewBuffer.raw_data[this.base + f] !== a) {
        SharedViewBuffer.raw_data[this.base + f] = a;
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get near() {
      const f = SharedViewBuffer.offsets.near;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set near(n) {
      const f = SharedViewBuffer.offsets.near;
      if (n && SharedViewBuffer.raw_data[this.base + f] !== n) {
        SharedViewBuffer.raw_data[this.base + f] = n;
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get far() {
      const f = SharedViewBuffer.offsets.far;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set far(fa) {
      const f = SharedViewBuffer.offsets.far;
      if (fa && SharedViewBuffer.raw_data[this.base + f] !== fa) {
        SharedViewBuffer.raw_data[this.base + f] = fa;
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get culling_enabled() {
      const f = SharedViewBuffer.offsets.culling_enabled;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set culling_enabled(enabled) {
      const f = SharedViewBuffer.offsets.culling_enabled;
      if (SharedViewBuffer.raw_data[this.base + f] !== enabled) {
        SharedViewBuffer.raw_data[this.base + f] = enabled;
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get occlusion_enabled() {
      const f = SharedViewBuffer.offsets.occlusion_enabled;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set occlusion_enabled(enabled) {
      const f = SharedViewBuffer.offsets.occlusion_enabled;
      if (SharedViewBuffer.raw_data[this.base + f] !== enabled) {
        SharedViewBuffer.raw_data[this.base + f] = enabled;
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get distance_check_enabled() {
      const f = SharedViewBuffer.offsets.distance_check_enabled;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set distance_check_enabled(enabled) {
      const f = SharedViewBuffer.offsets.distance_check_enabled;
      if (SharedViewBuffer.raw_data[this.base + f] !== enabled) {
        SharedViewBuffer.raw_data[this.base + f] = enabled;
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get velocity() {
      const f = SharedViewBuffer.offsets.velocity;
      return SharedViewBuffer.raw_data.subarray(this.base + f, this.base + f + 4);
    }
    set velocity(v) {
      const f = SharedViewBuffer.offsets.velocity;
      SharedViewBuffer.raw_data.set(v, this.base + f);
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get zoom() {
      const f = SharedViewBuffer.offsets.zoom;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set zoom(z) {
      const f = SharedViewBuffer.offsets.zoom;
      SharedViewBuffer.raw_data[this.base + f] = z;
      SharedViewBuffer.dirty_states.set(this.idx, 1);
    }
    get custom_projection_enabled() {
      return SharedViewBuffer.custom_projection_matrix_enabled.get(this.idx);
    }
    set custom_projection_enabled(enabled) {
      if (SharedViewBuffer.custom_projection_matrix_enabled.get(this.idx) !== enabled) {
        SharedViewBuffer.custom_projection_matrix_enabled.set(this.idx, enabled);
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get custom_view_matrix_enabled() {
      return SharedViewBuffer.custom_view_matrix_enabled.get(this.idx);
    }
    set custom_view_matrix_enabled(enabled) {
      if (SharedViewBuffer.custom_view_matrix_enabled.get(this.idx) !== enabled) {
        SharedViewBuffer.custom_view_matrix_enabled.set(this.idx, enabled);
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    get renderable_state() {
      return SharedViewBuffer.is_render_active(this.idx);
    }
    set renderable_state(state) {
      SharedViewBuffer.set_render_active(this.idx, state);
    }
    get clipmap_count() {
      const f = SharedViewBuffer.offsets.clipmap_count;
      return SharedViewBuffer.raw_data[this.base + f];
    }
    set clipmap_count(count) {
      const f = SharedViewBuffer.offsets.clipmap_count;
      if (SharedViewBuffer.raw_data[this.base + f] !== count) {
        SharedViewBuffer.raw_data[this.base + f] = count;
        SharedViewBuffer.dirty_states.set(this.idx, 1);
      }
    }
    set_index(index) {
      this.idx = index;
      return this;
    }
    get_index() {
      return this.idx;
    }
    destroy() {
      SharedViewBuffer.remove_view_data(this.idx);
    }
  };

  // --- Layout Configuration ---
  // floats per view: 7x16 matrix + 4x4 vector + 6x4 frustum + 4 floats = 156
  static floats_per_view = 168;
  static type_size_bytes = SharedViewBuffer.floats_per_view * 4;

  // --- Raw Data & Dirty Flags ---
  static raw_data = new Float32Array(0);
  static dirty_states = new ResizableBitArray(256);
  static renderable_states = new ResizableBitArray(256);
  static moved_states = new ResizableBitArray(256);
  static custom_projection_matrix_enabled = new ResizableBitArray(256);
  static custom_view_matrix_enabled = new ResizableBitArray(256);
  static free_list = new TypedStack(16, Uint32Array);
  static temporal_jitter_enabled = false;

  // --- View Pool & GPU Resources ---
  static buffer = null;
  static buffer_size = 0;
  static views = new RingBufferAllocator(256, SharedViewBuffer.View);

  // --- Data Management Methods ---
  /** Allocate a new view record (identity/default) and return its wrapper */
  static add_view_data() {
    let idx;

    // reuse a freed slot if available
    if (!SharedViewBuffer.free_list.is_empty()) {
      idx = SharedViewBuffer.free_list.pop();
    } else {
      // append new slot
      const old = SharedViewBuffer.raw_data;
      idx = old.length / SharedViewBuffer.floats_per_view;
      const new_data = new Float32Array(old.length + SharedViewBuffer.floats_per_view);
      new_data.set(old, 0);
      SharedViewBuffer.raw_data = new_data;
    }

    const base = idx * SharedViewBuffer.floats_per_view;

    SharedViewBuffer.raw_data.set(mat4.create(), base + SharedViewBuffer.offsets.view_matrix);
    SharedViewBuffer.raw_data.set(mat4.create(), base + SharedViewBuffer.offsets.prev_view_matrix);
    SharedViewBuffer.raw_data.set(mat4.create(), base + SharedViewBuffer.offsets.projection_matrix);
    SharedViewBuffer.raw_data.set(
      mat4.create(),
      base + SharedViewBuffer.offsets.prev_projection_matrix
    );
    SharedViewBuffer.raw_data.set(
      mat4.create(),
      base + SharedViewBuffer.offsets.view_projection_matrix
    );
    SharedViewBuffer.raw_data.set(
      mat4.create(),
      base + SharedViewBuffer.offsets.inverse_view_projection_matrix
    );
    SharedViewBuffer.raw_data.set(
      mat4.create(),
      base + SharedViewBuffer.offsets.prev_inverse_view_projection_matrix
    );
    SharedViewBuffer.raw_data.set(
      vec4.fromValues(0, 0, 0, 1),
      base + SharedViewBuffer.offsets.view_position
    );
    SharedViewBuffer.raw_data.set(
      vec4.fromValues(0, 0, 0, 0),
      base + SharedViewBuffer.offsets.view_rotation
    );
    SharedViewBuffer.raw_data.set(WORLD_FORWARD, base + SharedViewBuffer.offsets.view_direction);
    SharedViewBuffer.raw_data.set(
      vec4.fromValues(1, 0, 0, 0),
      base + SharedViewBuffer.offsets.view_right
    );
    SharedViewBuffer.raw_data.set(Array(24).fill(0), base + SharedViewBuffer.offsets.frustum);
    SharedViewBuffer.raw_data.set([radians(90.0)], base + SharedViewBuffer.offsets.fov);
    SharedViewBuffer.raw_data.set([1.0], base + SharedViewBuffer.offsets.aspect_ratio);
    SharedViewBuffer.raw_data.set([0.1], base + SharedViewBuffer.offsets.near);
    SharedViewBuffer.raw_data.set([10000.0], base + SharedViewBuffer.offsets.far);
    SharedViewBuffer.raw_data.set([1], base + SharedViewBuffer.offsets.culling_enabled);
    SharedViewBuffer.raw_data.set([1], base + SharedViewBuffer.offsets.occlusion_enabled);
    SharedViewBuffer.raw_data.set([1], base + SharedViewBuffer.offsets.distance_check_enabled);
    SharedViewBuffer.raw_data.set([0, 0, 0, 0], base + SharedViewBuffer.offsets.velocity);
    SharedViewBuffer.raw_data.set([1.0], base + SharedViewBuffer.offsets.zoom);
    SharedViewBuffer.raw_data.set([1], base + SharedViewBuffer.offsets.clipmap_count);

    SharedViewBuffer.custom_projection_matrix_enabled.set(idx, 0);
    SharedViewBuffer.custom_view_matrix_enabled.set(idx, 0);
    SharedViewBuffer.dirty_states.set(idx, 1);
    SharedViewBuffer.renderable_states.set(idx, 0);
    SharedViewBuffer.moved_states.set(idx, 0);

    if (
      !SharedViewBuffer.buffer ||
      SharedViewBuffer.raw_data.byteLength > SharedViewBuffer.buffer_size
    ) {
      SharedViewBuffer.build();
    }

    const view = SharedViewBuffer.views.allocate();

    return view.set_index(idx);
  }

  /** Remove a view, compact raw_data, and rebuild GPU buffer */
  static remove_view_data(idx, compact = false) {
    const length = SharedViewBuffer.raw_data.length / SharedViewBuffer.floats_per_view;

    if (compact) {
      const last = length - 1;
      const last_dirty_state = SharedViewBuffer.dirty_states.get(last);
      const last_renderable_state = SharedViewBuffer.is_render_active(last);
      const last_moved_state = SharedViewBuffer.moved_states.get(last);

      SharedViewBuffer.set_render_active(idx, false);

      // Swap with last view if not the last view
      if (idx !== last) {
        SharedViewBuffer.set_render_active(last, false);

        const src = last * SharedViewBuffer.floats_per_view;
        const dst = idx * SharedViewBuffer.floats_per_view;
        SharedViewBuffer.raw_data.copyWithin(dst, src, src + SharedViewBuffer.floats_per_view);

        SharedViewBuffer.dirty_states.set(idx, last_dirty_state);
        SharedViewBuffer.set_render_active(idx, last_renderable_state);
        SharedViewBuffer.moved_states.set(idx, last_moved_state);
      }

      SharedViewBuffer.raw_data = SharedViewBuffer.raw_data.subarray(
        0,
        last * SharedViewBuffer.floats_per_view
      );
    } else {
      SharedViewBuffer.dirty_states.set(idx, 0);
      SharedViewBuffer.moved_states.set(idx, 0);
      if (SharedViewBuffer.is_render_active(idx)) {
        const clipmap_count = SharedViewBuffer.raw_data[
          idx * SharedViewBuffer.floats_per_view + SharedViewBuffer.offsets.clipmap_count
        ];
        for (let i = 0; i < clipmap_count; i++) {
          MeshTaskQueue.deallocate_view_data(idx, i);
        }
      }
      SharedViewBuffer.set_render_active(idx, false);
      SharedViewBuffer.fill(
        0,
        idx * SharedViewBuffer.floats_per_view,
        SharedViewBuffer.floats_per_view
      );
      // recycle this index for future allocations
      SharedViewBuffer.free_list.push(idx);
    }
  }

  /** Get a view by index */
  static get_view_data(i) {
    const view = SharedViewBuffer.views.allocate();
    return view.set_index(i);
  }

  /** Get the number of views */
  static get_view_data_count() {
    return SharedViewBuffer.raw_data.length / SharedViewBuffer.floats_per_view;
  }

  /** Check if a specific view is active. */
  static is_render_active(view_index) {
    return !!SharedViewBuffer.renderable_states.get(view_index);
  }

  /** Check if a specific view was moved. */
  static was_moved(view_index) {
    return !!SharedViewBuffer.moved_states.get(view_index);
  }

  static set_temporal_jitter_enabled(enabled) {
    const new_enabled = !!enabled;
    if (SharedViewBuffer.temporal_jitter_enabled === new_enabled) {
      return;
    }

    SharedViewBuffer.temporal_jitter_enabled = new_enabled;
    const count = SharedViewBuffer.get_view_data_count();
    for (let i = 0; i < count; i++) {
      SharedViewBuffer.dirty_states.set(i, 1);
    }
  }

  static get_temporal_jitter(frame_index, resolution) {
    if (!SharedViewBuffer.temporal_jitter_enabled) {
      return vec2.fromValues(0.0, 0.0);
    }

    const sample_index =
      ((Math.floor(frame_index) % temporal_jitter_sample_count) +
        temporal_jitter_sample_count) %
      temporal_jitter_sample_count;
    const sample_offset = temporal_jitter_sample_offsets[sample_index];
    const jitter_x = sample_offset[0] * 2.0 / Math.max(1.0, resolution[0]);
    const jitter_y = sample_offset[1] * 2.0 / Math.max(1.0, resolution[1]);
    return vec2.fromValues(jitter_x, jitter_y);
  }

  /** Request cull update for a specific view index. */
  static set_render_active(view_index, active = true) {
    const old_state = SharedViewBuffer.renderable_states.get(view_index);
    const new_state = active ? 1 : 0;
    if (old_state === new_state) return;

    const clipmap_count = SharedViewBuffer.raw_data[
      view_index * SharedViewBuffer.floats_per_view + SharedViewBuffer.offsets.clipmap_count
    ];

    for (let i = 0; i < clipmap_count; i++) {
      if (active) {
        MeshTaskQueue.allocate_view_data(view_index, i);
    } else {
        MeshTaskQueue.deallocate_view_data(view_index, i);
      }
    }

    SharedViewBuffer.renderable_states.set(view_index, new_state);
  }

  /** Recompute view_projection/inverse/frustum for all or selected views */
  static update_transforms(indices = null) {
    const count = SharedViewBuffer.raw_data.length / SharedViewBuffer.floats_per_view;
    const list = indices ?? Array.from({ length: count }, (_, i) => i);
    const active_view_index = SharedFrameInfoBuffer.get_view_index();
    const frame_index = SharedFrameInfoBuffer.get_frame_index();
    const resolution = SharedFrameInfoBuffer.frame_info.resolution;

    for (let i = 0; i < list.length; ++i) {
      const idx = list[i];
      const base = idx * SharedViewBuffer.floats_per_view;
      const temporal_jitter_update =
        SharedViewBuffer.temporal_jitter_enabled &&
        idx === active_view_index &&
        !SharedViewBuffer.custom_projection_matrix_enabled.get(idx);

      if (
        !SharedViewBuffer.dirty_states.get(idx) &&
        !SharedViewBuffer.moved_states.get(idx) &&
        !temporal_jitter_update
      ) {
        continue;
      }
      
      if (!SharedViewBuffer.dirty_states.get(idx)) {
        SharedViewBuffer.moved_states.set(idx, 0);
      }

      // Copy previous view and projection matrices
      SharedViewBuffer.raw_data.copyWithin(
        base + SharedViewBuffer.offsets.prev_view_matrix,
        base + SharedViewBuffer.offsets.view_matrix,
        base + SharedViewBuffer.offsets.view_matrix + 16
      );
      SharedViewBuffer.raw_data.copyWithin(
        base + SharedViewBuffer.offsets.prev_projection_matrix,
        base + SharedViewBuffer.offsets.projection_matrix,
        base + SharedViewBuffer.offsets.projection_matrix + 16
      );

      let projection_matrix;

      if (SharedViewBuffer.custom_projection_matrix_enabled.get(idx)) {
        // Use the projection matrix supplied by the user
        projection_matrix = SharedViewBuffer.raw_data.subarray(
          base + SharedViewBuffer.offsets.projection_matrix,
          base + SharedViewBuffer.offsets.projection_matrix + 16
        );
      } else {
        // Compute projection matrix (perspective or orthographic)
        projection_matrix = mat4.create();

        if (SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.fov] > 0.0) {
          mat4.perspective(
            projection_matrix,
            SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.fov],
            SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.aspect_ratio],
            SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.near],
            SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.far]
          );
        } else {
          const far = SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.far];
          const zoom = SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.zoom];
          const aspect_ratio = SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.aspect_ratio];
          const height = far * zoom;
          const width = height * aspect_ratio;
          mat4.ortho(
            projection_matrix,
            -width,
            width,
            -height,
            height,
            -far,
            far
          );
        }

        // Store computed projection back to raw_data
        SharedViewBuffer.raw_data.set(
          projection_matrix,
          base + SharedViewBuffer.offsets.projection_matrix
        );
      }

      if (temporal_jitter_update) {
        apply_projection_jitter(
          projection_matrix,
          SharedViewBuffer.get_temporal_jitter(frame_index, resolution)
        );
        SharedViewBuffer.raw_data.set(
          projection_matrix,
          base + SharedViewBuffer.offsets.projection_matrix
        );
      }

      // Compute view direction vector
      const view_rotation = SharedViewBuffer.raw_data.subarray(
        base + SharedViewBuffer.offsets.view_rotation,
        base + SharedViewBuffer.offsets.view_rotation + 4
      );
      let view_direction = vec4.create();
      vec4.transformQuat(view_direction, WORLD_FORWARD, view_rotation);
      vec4.normalize(view_direction, view_direction);
      SharedViewBuffer.raw_data.set(view_direction, base + SharedViewBuffer.offsets.view_direction);

      // Compute view right vector
      let right = vec3.cross(vec3.create(), view_direction, WORLD_UP);
      right = vec4.fromValues(right[0], right[1], right[2], 0);
      vec4.normalize(right, right);
      SharedViewBuffer.raw_data.set(right, base + SharedViewBuffer.offsets.view_right);

      // Compute view target
      const view_position = SharedViewBuffer.raw_data.subarray(
        base + SharedViewBuffer.offsets.view_position,
        base + SharedViewBuffer.offsets.view_position + 4
      );
      const view_target = vec4.create();
      vec4.scaleAndAdd(view_target, view_position, view_direction, 1.0);

      let view_matrix;
      if (SharedViewBuffer.custom_view_matrix_enabled.get(idx)) {
        // Use the view matrix supplied by the user
        view_matrix = SharedViewBuffer.raw_data.subarray( 
          base + SharedViewBuffer.offsets.view_matrix,
          base + SharedViewBuffer.offsets.view_matrix + 16
        );
      } else {
        // Compute view matrix
        view_matrix = mat4.create();
        mat4.lookAt(view_matrix, view_position, view_target, WORLD_UP);
        SharedViewBuffer.raw_data.set(view_matrix, base + SharedViewBuffer.offsets.view_matrix);
      }

      // Compute view projection matrix and inverse
      const view_projection_matrix = mat4.create();
      mat4.mul(view_projection_matrix, projection_matrix, view_matrix);
      SharedViewBuffer.raw_data.set(
        view_projection_matrix,
        base + SharedViewBuffer.offsets.view_projection_matrix
      );
      const inverse_view_projection_matrix = mat4.create();
      mat4.invert(inverse_view_projection_matrix, view_projection_matrix);
      SharedViewBuffer.raw_data.set(
        inverse_view_projection_matrix,
        base + SharedViewBuffer.offsets.inverse_view_projection_matrix
      );

      const prev_view_matrix = SharedViewBuffer.raw_data.subarray(
        base + SharedViewBuffer.offsets.prev_view_matrix,
        base + SharedViewBuffer.offsets.prev_view_matrix + 16
      );
      const prev_projection_matrix = SharedViewBuffer.raw_data.subarray(
        base + SharedViewBuffer.offsets.prev_projection_matrix,
        base + SharedViewBuffer.offsets.prev_projection_matrix + 16
      );
      const prev_view_projection_matrix = mat4.create();
      mat4.mul(prev_view_projection_matrix, prev_projection_matrix, prev_view_matrix);
      const prev_inverse_view_projection_matrix = mat4.create();
      mat4.invert(prev_inverse_view_projection_matrix, prev_view_projection_matrix);
      SharedViewBuffer.raw_data.set(
        prev_inverse_view_projection_matrix,
        base + SharedViewBuffer.offsets.prev_inverse_view_projection_matrix
      );

      // Frustum planes
      const fr = Array(24).fill(0);
      const tmpv = vec3.create();
      // Left plane
      vec3.set(
        tmpv,
        view_projection_matrix[3] + view_projection_matrix[0],
        view_projection_matrix[7] + view_projection_matrix[4],
        view_projection_matrix[11] + view_projection_matrix[8]
      );
      let l = vec3.length(tmpv);
      fr[0] = tmpv[0] / l;
      fr[1] = tmpv[1] / l;
      fr[2] = tmpv[2] / l;
      fr[3] = (view_projection_matrix[15] + view_projection_matrix[12]) / l;
      // Right
      vec3.set(
        tmpv,
        view_projection_matrix[3] - view_projection_matrix[0],
        view_projection_matrix[7] - view_projection_matrix[4],
        view_projection_matrix[11] - view_projection_matrix[8]
      );
      l = vec3.length(tmpv);
      fr[4] = tmpv[0] / l;
      fr[5] = tmpv[1] / l;
      fr[6] = tmpv[2] / l;
      fr[7] = (view_projection_matrix[15] - view_projection_matrix[12]) / l;
      // Top
      vec3.set(
        tmpv,
        view_projection_matrix[3] - view_projection_matrix[1],
        view_projection_matrix[7] - view_projection_matrix[5],
        view_projection_matrix[11] - view_projection_matrix[9]
      );
      l = vec3.length(tmpv);
      fr[8] = tmpv[0] / l;
      fr[9] = tmpv[1] / l;
      fr[10] = tmpv[2] / l;
      fr[11] = (view_projection_matrix[15] - view_projection_matrix[13]) / l;
      // Bottom
      vec3.set(
        tmpv,
        view_projection_matrix[3] + view_projection_matrix[1],
        view_projection_matrix[7] + view_projection_matrix[5],
        view_projection_matrix[11] + view_projection_matrix[9]
      );
      l = vec3.length(tmpv);
      fr[12] = tmpv[0] / l;
      fr[13] = tmpv[1] / l;
      fr[14] = tmpv[2] / l;
      fr[15] = (view_projection_matrix[15] + view_projection_matrix[13]) / l;
      // Near
      vec3.set(
        tmpv,
        view_projection_matrix[3] + view_projection_matrix[2],
        view_projection_matrix[7] + view_projection_matrix[6],
        view_projection_matrix[11] + view_projection_matrix[10]
      );
      l = vec3.length(tmpv);
      fr[16] = tmpv[0] / l;
      fr[17] = tmpv[1] / l;
      fr[18] = tmpv[2] / l;
      fr[19] = (view_projection_matrix[15] + view_projection_matrix[14]) / l;
      // Far
      vec3.set(
        tmpv,
        view_projection_matrix[3] - view_projection_matrix[2],
        view_projection_matrix[7] - view_projection_matrix[6],
        view_projection_matrix[11] - view_projection_matrix[10]
      );
      l = vec3.length(tmpv);
      fr[20] = tmpv[0] / l;
      fr[21] = tmpv[1] / l;
      fr[22] = tmpv[2] / l;
      fr[23] = (view_projection_matrix[15] - view_projection_matrix[14]) / l;
      SharedViewBuffer.raw_data.set(fr, base + SharedViewBuffer.offsets.frustum);

      // Compute view velocity as the difference between camera positions
      // extracted from the previous and current view matrices.
      // Invert both matrices to get the camera transform (whose translation is the camera position)
      const inv_prev_view = mat4.invert(mat4.create(), prev_view_matrix);
      const inv_new_view = mat4.invert(mat4.create(), view_matrix);

      // Extract camera positions (translation components are at indices 12, 13, 14)
      const prev_camera_position = vec3.fromValues(
          inv_prev_view[12], inv_prev_view[13], inv_prev_view[14]
      );
      const new_camera_position = vec3.fromValues(
          inv_new_view[12], inv_new_view[13], inv_new_view[14]
      );

      // Compute velocity as difference (optionally divide by delta time if available)
      const velocity = vec3.subtract(vec3.create(), new_camera_position, prev_camera_position);
      SharedViewBuffer.raw_data.set(velocity, base + SharedViewBuffer.offsets.velocity);

      // upload full view block using element-count write
      const view_slice = SharedViewBuffer.raw_data.subarray(
        base,
        base + SharedViewBuffer.floats_per_view
      );
      SharedViewBuffer.buffer.write(view_slice, base * 4);

      if (SharedViewBuffer.dirty_states.get(idx)) {
        SharedViewBuffer.dirty_states.set(idx, 0);
        SharedViewBuffer.moved_states.set(idx, 1);
      }
    }
  }

  /** Rebuild the GPU buffer from raw_data */
  static build() {
    if (
      !SharedViewBuffer.buffer ||
      SharedViewBuffer.raw_data.byteLength > SharedViewBuffer.buffer_size
    ) {
      const buffer_length = SharedViewBuffer.raw_data.byteLength * 2;
      SharedViewBuffer.buffer = Buffer.create({
        name: view_buffer_name,
        size: buffer_length,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: true,
      });
      SharedViewBuffer.buffer_size = buffer_length;

      Renderer.get().refresh_global_shader_bindings();
    }

    // rewrite entire buffer using element-count write
    SharedViewBuffer.buffer.write(SharedViewBuffer.raw_data, 0);
  }
}

export class SharedEnvironmentData {
  static skybox = null;
  static skydome_data = null;

  static skydome_data_buffer = new Float32Array([
    1, 1, 1, 1, // color (for regular skybox)
    2.99, // sunlight_intensity
    0.9997966769, // sunlight_angular_radius
    0.9, // atmospheric_rayleigh
    2.542, // atmospheric_turbidity
    0.002, // mie_coefficient
    0.8, // mie_directional_g
    0, // view index 
    1, // sky_type (0 = skybox, 1 = skydome)
  ]);

  static set_skybox(name, texture_paths) {
    this.skybox = Texture.load({
      name: name,
      paths: texture_paths,
      format: "rgba8unorm",
      dimension: "cube",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Update skydome buffer to indicate skybox mode
    this.skydome_data_buffer[11] = 0; // sky_type = 0 (skybox)
    if (!this.skydome_data) {
      this.skydome_data = Buffer.create({
        name: name + "_skydome_data",
        raw_data: this.skydome_data_buffer,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        force: true,
      });
    } else {
      this.skydome_data.write(this.skydome_data_buffer);
    }
    
    return this.skybox;
  }

  static set_skydome(name) {
    this.skybox = Texture.default_cube();

    // Skydome is enabled - update sky_type flag
    this.skydome_data_buffer[11] = 1; // sky_type = 1 (skydome)
    if (!this.skydome_data) {
      this.skydome_data = Buffer.create({
        name: name + "_skydome_data",
        raw_data: this.skydome_data_buffer,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        force: true,
      });
    } else {
      this.skydome_data.write(this.skydome_data_buffer);
    }
  }

  static set_skybox_color(color) {
    this.skydome_data_buffer[0] = color[0];
    this.skydome_data_buffer[1] = color[1];
    this.skydome_data_buffer[2] = color[2];
    this.skydome_data_buffer[3] = color[3];
    this.skydome_data.write(this.skydome_data_buffer);
  }

  static set_skydome_view(view_index) {
    this.skydome_data_buffer[10] = view_index;
    this.skydome_data.write(this.skydome_data_buffer);
  }

  static get_skybox() {
    return this.skybox;
  }

  static get_skydome_view() {
    return this.skydome_data_buffer[10];
  }

  static get_skydome_data() {
    return this.skydome_data;
  }
}

export class SharedFrameInfoBuffer {
  static frame_info = {
    view_index: 0,
    time: 0,
    frame_index: 0,
    resolution: vec2.create(),
    cursor_world_position: vec4.create(),
    padding0: 0,
  };
  static buffer = null;
  static size = 0;

  static get_view_index() {
    return this.frame_info.view_index;
  }

  static get_time() {
    return this.frame_info.time;
  }

  static set_view_index(index) {
    this.frame_info.view_index = index;
    if (!this.buffer) {
      this.build();
    } else {
      this.buffer.write(this._get_gpu_type_layout(this.frame_info));
    }
  }

  static set_time(time) {
    this.frame_info.time = time;
    if (!this.buffer) {
      this.build();
    } else {
      this.buffer.write(this._get_gpu_type_layout(this.frame_info));
    }
  }

  static set_frame_index(frame_index) {
    this.frame_info.frame_index = frame_index;
    if (!this.buffer) {
      this.build();
    } else {
      this.buffer.write(this._get_gpu_type_layout(this.frame_info));
    }
  }

  static get_frame_index() {
    return this.frame_info.frame_index;
  }

  static set_cursor_world_position(cursor_world_position) {
    this.frame_info.cursor_world_position = cursor_world_position;
    if (!this.buffer) {
      this.build();
    } else {
      this.buffer.write(this._get_gpu_type_layout(this.frame_info));
    }
  }

  static set_resolution(resolution) {
    this.frame_info.resolution = resolution;
    if (!this.buffer) {
      this.build();
    } else {
      this.buffer.write(this._get_gpu_type_layout(this.frame_info));
    }
  }

  static build() {
    if (this.buffer) {
      this.buffer.destroy();
    }

    const gpu_layout = this._get_gpu_type_layout(this.frame_info);

    this.size = gpu_layout.length * 4;

    this.buffer = Buffer.create({
      name: frame_info_buffer_name,
      data: gpu_layout,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  static _get_gpu_type_layout(item) {
    return Array.of(
      item.view_index,
      item.time,
      item.frame_index,
      item.padding0,
      ...item.resolution,
      item.padding0,
      item.padding0,
      ...item.cursor_world_position,
    );
  }
}

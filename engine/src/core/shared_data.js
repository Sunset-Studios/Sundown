import { Buffer } from "../renderer/buffer.js";
import { Texture } from "../renderer/texture.js";
import { Renderer } from "../renderer/renderer.js";
import { RingBufferAllocator } from "../memory/allocator.js";
import { ResizableBitArray, TypedStack } from "../memory/container.js";
import { MeshTaskQueue } from "../renderer/mesh_task_queue.js";
import { mat4, vec4, vec3, vec2 } from "gl-matrix";
import { WORLD_FORWARD, WORLD_UP } from "./minimal.js";
import { radians } from "../utility/math.js";

const vertex_buffer_name = "vertex_buffer";
const view_buffer_name = "view_buffer";
const frame_info_buffer_name = "frame_info_buffer";

export class SharedVertexBuffer {
  static buffer = null;
  static vertex_data = [];
  static size = 0;

  static add_vertex_data(data) {
    const offset = this.vertex_data.length;
    this.vertex_data.push(...data);
    this.size = this._get_byte_size();
    this.build();
    return offset;
  }

  static _get_byte_size() {
    return (
      this.vertex_data.map((v) => v.position.concat(v.normal, v.tangent, v.bitangent, v.uv)).flat()
        .length * 4
    );
  }

  static build() {
    this.buffer = Buffer.create({
      name: vertex_buffer_name,
      data: this.vertex_data
        .map((v) => v.position.concat(v.normal, v.tangent, v.bitangent, v.uv))
        .flat(),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: true,
    });
    Renderer.get().refresh_global_shader_bindings();
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
    view_direction: 96,
    near: 100,
    far: 101,
    culling_enabled: 102,
    occlusion_enabled: 103,
    frustum: 104,
    view_position: 128,
    view_rotation: 132,
    view_right: 136,
    fov: 140,
    aspect_ratio: 141,
    distance_check_enabled: 142,
    velocity: 143,
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
      if (fo && SharedViewBuffer.raw_data[this.base + f] !== fo) {
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
    get renderable_state() {
      return SharedViewBuffer.is_render_active(this.idx);
    }
    set renderable_state(state) {
      SharedViewBuffer.set_render_active(this.idx, state);
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
  // floats per view: 6×16 matrix + 4×4 vector + 6×4 frustum + 4 floats = 140
  static floats_per_view = 148;
  static type_size_bytes = SharedViewBuffer.floats_per_view * 4;

  // --- Raw Data & Dirty Flags ---
  static raw_data = new Float32Array(0);
  static dirty_states = new ResizableBitArray(256);
  static renderable_states = new ResizableBitArray(256);
  static free_list = new TypedStack(16, Uint32Array);

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
    SharedViewBuffer.raw_data.set([0.001], base + SharedViewBuffer.offsets.near);
    SharedViewBuffer.raw_data.set([1000.0], base + SharedViewBuffer.offsets.far);
    SharedViewBuffer.raw_data.set([1], base + SharedViewBuffer.offsets.culling_enabled);
    SharedViewBuffer.raw_data.set([1], base + SharedViewBuffer.offsets.occlusion_enabled);
    SharedViewBuffer.raw_data.set([1], base + SharedViewBuffer.offsets.distance_check_enabled);
    SharedViewBuffer.raw_data.set([0, 0, 0, 0], base + SharedViewBuffer.offsets.velocity);
    SharedViewBuffer.dirty_states.set(idx, 1);
    SharedViewBuffer.renderable_states.set(idx, 0);

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

      SharedViewBuffer.set_render_active(idx, false);

      // Swap with last view if not the last view
      if (idx !== last) {
        SharedViewBuffer.set_render_active(last, false);

        const src = last * SharedViewBuffer.floats_per_view;
        const dst = idx * SharedViewBuffer.floats_per_view;
        SharedViewBuffer.raw_data.copyWithin(dst, src, src + SharedViewBuffer.floats_per_view);

        SharedViewBuffer.dirty_states.set(idx, last_dirty_state);
        SharedViewBuffer.set_render_active(idx, last_renderable_state);
      }

      SharedViewBuffer.raw_data = SharedViewBuffer.raw_data.subarray(
        0,
        last * SharedViewBuffer.floats_per_view
      );
    } else {
      SharedViewBuffer.dirty_states.set(idx, 0);
      if (SharedViewBuffer.is_render_active(idx)) {
        MeshTaskQueue.deallocate_view_data(idx);
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

  /** Request cull update for a specific view index. */
  static set_render_active(view_index, active = true) {
    const old_state = SharedViewBuffer.renderable_states.get(view_index);
    const new_state = active ? 1 : 0;
    if (old_state === new_state) return;

    if (active) {
      MeshTaskQueue.allocate_view_data(view_index);
    } else {
      MeshTaskQueue.deallocate_view_data(view_index);
    }

    SharedViewBuffer.renderable_states.set(view_index, new_state);
  }

  /** Recompute view_projection/inverse/frustum for all or selected views */
  static update_transforms(indices = null) {
    const count = SharedViewBuffer.raw_data.length / SharedViewBuffer.floats_per_view;
    const list = indices ?? Array.from({ length: count }, (_, i) => i);

    for (let i = 0; i < list.length; ++i) {
      const idx = list[i];

      if (!SharedViewBuffer.dirty_states.get(idx)) {
        continue;
      }

      const base = idx * SharedViewBuffer.floats_per_view;

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

      // Compute projection matrix
      const projection_matrix = mat4.create();
      mat4.perspective(
        projection_matrix,
        SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.fov],
        SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.aspect_ratio],
        SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.near],
        SharedViewBuffer.raw_data[base + SharedViewBuffer.offsets.far]
      );
      SharedViewBuffer.raw_data.set(
        projection_matrix,
        base + SharedViewBuffer.offsets.projection_matrix
      );

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

      // Compute view matrix
      const view_matrix = mat4.create();
      mat4.lookAt(view_matrix, view_position, view_target, WORLD_UP);
      SharedViewBuffer.raw_data.set(view_matrix, base + SharedViewBuffer.offsets.view_matrix);

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
      const prev_view_matrix = SharedViewBuffer.raw_data.subarray(
          base + SharedViewBuffer.offsets.prev_view_matrix,
          base + SharedViewBuffer.offsets.prev_view_matrix + 16
      );

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

      SharedViewBuffer.dirty_states.set(idx, 0);
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

export class SharedEnvironmentMapData {
  static skybox = null;
  static skybox_data = null;
  static skybox_data_buffer = new Float32Array([1, 1, 1, 1]);

  static set_skybox(name, texture_paths) {
    const skybox = Texture.load(texture_paths, {
      name: name,
      format: "rgba8unorm",
      dimension: "cube",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      force: true,
    });

    const skybox_data = Buffer.create({
      name: name + "_data",
      raw_data: this.skybox_data_buffer,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: true,
    });

    this.skybox = skybox;
    this.skybox_data = skybox_data;

    return skybox;
  }

  static set_skybox_color(color) {
    this.skybox_data_buffer.set(color);
    this.skybox_data.write(this.skybox_data_buffer);
  }

  static get_skybox() {
    return this.skybox;
  }

  static get_skybox_data() {
    return this.skybox_data;
  }
}

export class SharedFrameInfoBuffer {
  static frame_info = {
    view_index: 0,
    time: 0,
    resolution: vec2.create(),
    cursor_world_position: vec4.create(),
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

    Renderer.get().refresh_global_shader_bindings();
  }

  static _get_gpu_type_layout(item) {
    return Array.of(item.view_index, item.time, ...item.resolution, ...item.cursor_world_position);
  }
}

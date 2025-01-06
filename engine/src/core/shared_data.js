import { Buffer } from "../renderer/buffer.js";
import { Texture } from "../renderer/texture.js";
import { Renderer } from "../renderer/renderer.js";
import { mat4, vec4, quat, vec3, vec2 } from "gl-matrix";
import { WORLD_UP, WORLD_FORWARD } from "./minimal.js";
import { Vector } from "../memory/container.js";
import _ from "lodash";

const vertex_buffer_name = "vertex_buffer";
const view_buffer_name = "view_buffer";
const frame_info_buffer_name = "frame_info_buffer";
const entity_metadata_buffer_name = "entity_metadata_buffer";

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
      this.vertex_data
        .map((v) =>
          v.position.concat(v.normal, v.color, v.uv, v.tangent, v.bitangent)
        )
        .flat().length * 4
    );
  }

  static build() {
    if (this.buffer) {
      this.buffer.destroy();
    }

    this.buffer = Buffer.create({
      name: vertex_buffer_name,
      data: this.vertex_data.map((v) =>
        v.position.concat(v.normal, v.color, v.uv, v.tangent, v.bitangent)
      ),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    Renderer.get().refresh_global_shader_bindings();
  }
}

export class SharedViewBuffer {
  static view_data = [];
  static type_size_bytes = 0;
  static raw_data = null;
  static size = 0;

  // Adds a view data to the buffer and returns the index. Should be called during setup, before the simulation begins.
  static add_view_data(view_data = {}) {
    this.view_data.push({
      position: view_data.position ?? vec4.create(),
      rotation: view_data.rotation ?? quat.create(),
      fov: view_data.fov ?? 75,
      aspect_ratio: view_data.aspect_ratio ?? 1,
      near: view_data.near ?? 0.001,
      far: view_data.far ?? 1000,
      view_forward: view_data.view_forward ?? vec4.create(),
      view_right: view_data.view_right ?? vec4.create(),
      view_matrix: view_data.view_matrix ?? mat4.create(),
      projection_matrix: view_data.projection_matrix ?? mat4.create(),
      frustum: view_data.frustum ?? Array(24).fill(0),
      prev_view_matrix: view_data.prev_view_matrix ?? mat4.create(),
      prev_projection_matrix: view_data.prev_projection_matrix ?? mat4.create(),
      view_projection_matrix: view_data.view_projection_matrix ?? mat4.create(),
      inverse_view_projection_matrix:
        view_data.inverse_view_projection_matrix_direction_only ??
        mat4.create(),
    });

    if (this.type_size_bytes === 0) {
      this.type_size_bytes =
        this._get_gpu_type_layout(this.view_data[0]).flat().length * 4;
    }

    const flattened_data = this.view_data
      .map((x) => this._get_gpu_type_layout(x))
      .flat();
    this.raw_data = new Float32Array(flattened_data.length);
    this.raw_data.set(flattened_data);
    this.size = this.type_size_bytes * this.view_data.length;

    this.build();

    return this.view_data.length - 1;
  }

  static get_view_data(index) {
    console.assert(
      index < this.view_data.length && index >= 0,
      "View data index out of bounds"
    );
    return this.view_data[index];
  }

  // Updates the view data at the given index. Can be called during setup or at runtime.
  static set_view_data(index, view_data) {
    let dirty = false;
    if (
      view_data.position &&
      !vec4.equals(this.view_data[index].position, view_data.position)
    ) {
      this.view_data[index].position = vec4.clone(view_data.position);
      dirty = true;
    }
    if (
      view_data.rotation &&
      !quat.equals(this.view_data[index].rotation, view_data.rotation)
    ) {
      this.view_data[index].rotation = quat.clone(view_data.rotation);
      dirty = true;
    }
    if (view_data.fov && this.view_data[index].fov !== view_data.fov) {
      this.view_data[index].fov = view_data.fov;
      dirty = true;
    }
    if (
      view_data.aspect_ratio &&
      this.view_data[index].aspect_ratio !== view_data.aspect_ratio
    ) {
      this.view_data[index].aspect_ratio = view_data.aspect_ratio;
      dirty = true;
    }
    if (view_data.near && this.view_data[index].near !== view_data.near) {
      this.view_data[index].near = view_data.near;
      dirty = true;
    }
    if (view_data.far && this.view_data[index].far !== view_data.far) {
      this.view_data[index].far = view_data.far;
      dirty = true;
    }
    this.view_data[index].dirty |= dirty;
  }

  // Updates the view transforms at the given index, given the view data previously set. Can be called during setup or at runtime.
  static update_transforms(index) {
    if (this.view_data[index].dirty) {
      if (this.view_data[index].projection_matrix) {
        this.view_data[index].prev_projection_matrix = mat4.clone(
          this.view_data[index].projection_matrix
        );
      }
      if (this.view_data[index].view_matrix) {
        this.view_data[index].prev_view_matrix = mat4.clone(
          this.view_data[index].view_matrix
        );
      }

      {
        mat4.perspective(
          this.view_data[index].projection_matrix,
          this.view_data[index].fov,
          this.view_data[index].aspect_ratio,
          this.view_data[index].near,
          this.view_data[index].far
        );
      }

      {
        this.view_data[index].view_forward = vec4.transformQuat(
          vec4.create(),
          WORLD_FORWARD,
          this.view_data[index].rotation
        );
        this.view_data[index].view_forward = vec4.normalize(
          vec4.create(),
          this.view_data[index].view_forward
        );

        const right = vec3.cross(
          vec3.create(),
          this.view_data[index].view_forward,
          WORLD_UP
        );
        this.view_data[index].view_right = vec4.fromValues(
          right[0],
          right[1],
          right[2],
          0
        );
        this.view_data[index].view_right = vec3.normalize(
          vec3.create(),
          this.view_data[index].view_right
        );

        const view_target = vec4.create();
        vec4.scaleAndAdd(
          view_target,
          this.view_data[index].position,
          this.view_data[index].view_forward,
          1.0
        );

        mat4.lookAt(
          this.view_data[index].view_matrix,
          this.view_data[index].position,
          view_target,
          WORLD_UP
        );
      }

      {
        mat4.mul(
          this.view_data[index].view_projection_matrix,
          this.view_data[index].projection_matrix,
          this.view_data[index].view_matrix
        );
        mat4.invert(
          this.view_data[index].inverse_view_projection_matrix,
          this.view_data[index].view_projection_matrix
        );
      }

      {
        const vp = this.view_data[index].view_projection_matrix;

        // Left clipping plane
        const frustum = Array(24).fill(0);
        const tmp = vec3.create();
        vec3.set(tmp, vp[3] + vp[0], vp[7] + vp[4], vp[11] + vp[8]);
        let l = vec3.length(tmp);
        frustum[0] = tmp[0] / l;
        frustum[1] = tmp[1] / l;
        frustum[2] = tmp[2] / l;
        frustum[3] = (vp[15] + vp[12]) / l;
        // Right clipping plane
        vec3.set(tmp, vp[3] - vp[0], vp[7] - vp[4], vp[11] - vp[8]);
        l = vec3.length(tmp);
        frustum[4] = tmp[0] / l;
        frustum[5] = tmp[1] / l;
        frustum[6] = tmp[2] / l;
        frustum[7] = (vp[15] - vp[12]) / l;
        // Top clipping plane
        vec3.set(tmp, vp[3] - vp[1], vp[7] - vp[5], vp[11] - vp[9]);
        l = vec3.length(tmp);
        frustum[8] = tmp[0] / l;
        frustum[9] = tmp[1] / l;
        frustum[10] = tmp[2] / l;
        frustum[11] = (vp[15] - vp[13]) / l;
        // Bottom clipping plane
        vec3.set(tmp, vp[3] + vp[1], vp[7] + vp[5], vp[11] + vp[9]);
        l = vec3.length(tmp);
        frustum[12] = tmp[0] / l;
        frustum[13] = tmp[1] / l;
        frustum[14] = tmp[2] / l;
        frustum[15] = (vp[15] + vp[13]) / l;
        // Near clipping plane
        vec3.set(tmp, vp[2], vp[6], vp[10]);
        l = vec3.length(tmp);
        frustum[16] = tmp[0] / l;
        frustum[17] = tmp[1] / l;
        frustum[18] = tmp[2] / l;
        frustum[19] = vp[14] / l;
        // Far clipping plane
        vec3.set(tmp, vp[3] - vp[2], vp[7] - vp[6], vp[11] - vp[10]);
        l = vec3.length(tmp);
        frustum[20] = tmp[0] / l;
        frustum[21] = tmp[1] / l;
        frustum[22] = tmp[2] / l;
        frustum[23] = (vp[15] - vp[14]) / l;

        this.view_data[index].frustum = frustum;
      }

      if (this.buffer) {
        this.buffer.write(this.view_data.map(this._get_gpu_type_layout));
      } else {
        this.build();
      }

      this.view_data[index].dirty = false;
    }
  }

  // Builds the GPU resident view buffer. Should be called during setup, before the simulation begins.
  static build() {
    if (this.buffer) {
      this.buffer.destroy();
    }

    this.buffer = Buffer.create({
      name: view_buffer_name,
      data: this.view_data.map(this._get_gpu_type_layout),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    Renderer.get().refresh_global_shader_bindings();
  }

  static _get_gpu_type_layout(item) {
    return Array.of(
      ...item.view_matrix,
      ...item.prev_view_matrix,
      ...item.projection_matrix,
      ...item.prev_projection_matrix,
      ...item.view_projection_matrix,
      ...item.inverse_view_projection_matrix,
      ...item.position,
      ...item.view_forward,
      ...item.frustum
    );
  }
}

export class SharedEnvironmentMapData {
  static skybox = null;

  static async add_skybox(name, texture_paths) {
    const skybox = await Texture.load(texture_paths, {
      name: name,
      format: "rgba8unorm",
      dimension: "cube",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.skybox = skybox;

    return skybox;
  }

  static remove_skybox() {
    this.skybox = null;
  }

  static get_skybox() {
    return this.skybox;
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
    return Array.of(
      item.view_index,
      item.time,
      ...item.resolution,
      ...item.cursor_world_position
    );
  }
}

export class SharedEntityMetadataBuffer {
  static entity_metadata = new Vector(256, Uint32Array);
  static num_entities = 0;
  static buffer = null;
  static buffer_size = 0;

  static get_entity_offset(entity) {
    return this.entity_metadata.get(entity * 2);
  }

  static get_entity_count(entity) {
    return this.entity_metadata.get(entity * 2 + 1);
  }

  static get_absolute_entity_count() {
    const adjusted_entity_count = this.num_entities - 1;
    return this.entity_metadata.get(adjusted_entity_count * 2) + this.entity_metadata.get(adjusted_entity_count * 2 + 1);
  }

  static set_entity_instance_count(entity, count) {
    const old_count = this.entity_metadata.get(entity * 2 + 1);
    this.entity_metadata.set(entity * 2 + 1, count);
    
    // Shift subsequent entity offsets by the change in count
    const count_delta = count - old_count;
    if (count_delta !== 0) {
      for (let i = entity + 1; i < this.num_entities; i++) {
        const current_offset = this.entity_metadata.get(i * 2);
        this.entity_metadata.set(i * 2, current_offset + count_delta);
      }
    }

    if (!this.buffer) {
      this.build();
    } else {
      this.buffer.write(this._get_gpu_type_layout());
    }
  }

  static add_entity(entity) {
    const adjusted_entity = entity + 1;

    this.resize(adjusted_entity);

    if (this.num_entities < adjusted_entity) {
      // Calculate the correct offset based on sum of previous entity counts
      let offset = 0;
      if (entity > 0) {
        const prev_entity = entity - 1;
        offset = this.entity_metadata.get(prev_entity * 2) + // Previous offset
                this.entity_metadata.get(prev_entity * 2 + 1); // Plus previous count
      }
      
      this.entity_metadata.set(entity * 2, offset);
      this.entity_metadata.set(entity * 2 + 1, 1);
      this.num_entities = adjusted_entity;
    }
  }

  static resize(new_size) {
    if (this.entity_metadata.length <= new_size * 2) {
      this.entity_metadata.resize(Math.max(4, new_size * 4)); // Double the size of the buffer
      this.build();
    }
  }

  static build() {
    if (this.buffer) {
      this.buffer.destroy();
    }

    const gpu_layout = this._get_gpu_type_layout();

    this.buffer_size = gpu_layout.byteLength;

    this.buffer = Buffer.create({
      name: entity_metadata_buffer_name,
      raw_data: gpu_layout,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: true,
    });

    Renderer.get().refresh_global_shader_bindings();
  }

  static _get_gpu_type_layout() {
    return this.entity_metadata.get_data();
  }

  static write() {
    this.buffer.write(this._get_gpu_type_layout());
  }
}
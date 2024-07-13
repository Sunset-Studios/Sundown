import { Buffer } from "../renderer/buffer.js";
import { Texture } from "../renderer/texture.js";
import { Renderer } from "../renderer/renderer.js";
import { mat4, vec4, quat, vec3 } from "gl-matrix";
import { WORLD_UP, WORLD_FORWARD } from "./minimal.js";
import _ from "lodash";

export class SharedVertexBuffer {
  buffer = null;
  vertex_data = [];
  size = 0;

  constructor() {
    if (SharedVertexBuffer.instance) {
      return SharedVertexBuffer.instance;
    }
    SharedVertexBuffer.instance = this;
  }

  static get() {
    if (!SharedVertexBuffer.instance) {
      SharedVertexBuffer.instance = new SharedVertexBuffer();
    }
    return SharedVertexBuffer.instance;
  }

  add_vertex_data(context, data) {
    const offset = this.vertex_data.length;
    this.vertex_data.push(...data);
    this.size = this._get_byte_size();
    this.build(context);
    return offset;
  }

  _get_byte_size() {
    return (
      this.vertex_data
        .map((v) => v.position.concat(v.normal, v.color, v.uv, v.tangent, v.bitangent))
        .flat().length * 4
    );
  }

  build(context) {
    if (this.buffer) {
      this.buffer.destroy();
    }

    this.buffer = Buffer.create(context, {
      name: "vertex_buffer",
      data: this.vertex_data.map((v) =>
        v.position.concat(v.normal, v.color, v.uv, v.tangent, v.bitangent)
      ),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    Renderer.get().refresh_global_shader_bindings();
  }
}

export class SharedViewBuffer {
  view_data = [];
  type_size_bytes = 0;
  raw_data = null;
  size = 0;

  constructor() {
    if (SharedViewBuffer.instance) {
      return SharedViewBuffer.instance;
    }
    SharedViewBuffer.instance = this;
  }

  static get() {
    if (!SharedViewBuffer.instance) {
      return new SharedViewBuffer();
    }
    return SharedViewBuffer.instance;
  }

  // Adds a view data to the buffer and returns the index. Should be called during setup, before the simulation begins.
  add_view_data(context, view_data = {}) {
    this.view_data.push({
      position: view_data.position ?? vec4.create(), 
      rotation: view_data.rotation ?? quat.create(),
      fov: view_data.fov ?? 75,
      aspect_ratio: view_data.aspect_ratio ?? 1,
      near: view_data.near ?? 0.01,
      far: view_data.far ?? 1000,
      view_forward: view_data.view_forward ?? vec4.create(),
      view_right: view_data.view_right ?? vec4.create(),
      view_matrix: view_data.view_matrix ?? mat4.create(),
      projection_matrix: view_data.projection_matrix ?? mat4.create(),
      prev_view_matrix: view_data.prev_view_matrix ?? mat4.create(),
      prev_projection_matrix: view_data.prev_projection_matrix ?? mat4.create(),
      view_projection_matrix: view_data.view_projection_matrix ?? mat4.create(),
      inverse_view_projection_matrix: view_data.inverse_view_projection_matrix_direction_only ?? mat4.create(),
    });

    if (this.type_size_bytes === 0) {
      this.type_size_bytes =
        this._get_gpu_type_layout(this.view_data[0]).length * 4;
    }

    const flattened_data = this.view_data
      .map((x) => this._get_gpu_type_layout(x))
      .flat();
    this.raw_data = new Float32Array(flattened_data.length);
    this.raw_data.set(flattened_data);
    this.size = this.type_size_bytes * this.view_data.length;

    this.build(context);

    return this.view_data.length - 1;
  }

  get_view_data(index) {
    console.assert(index < this.view_data.length && index >= 0, "View data index out of bounds");
    return this.view_data[index];
  }

  // Updates the view data at the given index. Can be called during setup or at runtime.
  set_view_data(context, index, view_data) {
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
    this.view_data[index].dirty = dirty;
  }

  // Updates the view transforms at the given index, given the view data previously set. Can be called during setup or at runtime.
  update_transforms(context, index) {
    if (this.view_data[index].dirty) {
      if (this.view_data[index].projection_matrix) {
        this.view_data[index].prev_projection_matrix =
          mat4.clone(this.view_data[index].projection_matrix);
      }
      if (this.view_data[index].view_matrix) {
        this.view_data[index].prev_view_matrix =
          mat4.clone(this.view_data[index].view_matrix);
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
          this.view_data[index].view_forward = vec4.transformQuat(vec4.create(), WORLD_FORWARD, this.view_data[index].rotation);
          this.view_data[index].view_forward = vec4.normalize(vec4.create(), this.view_data[index].view_forward);

          const right = vec3.cross(vec3.create(), this.view_data[index].view_forward, WORLD_UP);
          this.view_data[index].view_right = vec4.fromValues(right[0], right[1], right[2], 0);
          this.view_data[index].view_right = vec3.normalize(vec3.create(), this.view_data[index].view_right);

          const view_target = vec4.create();
          vec4.scaleAndAdd(view_target, this.view_data[index].position, this.view_data[index].view_forward, 1.0);
    
          mat4.lookAt(
            this.view_data[index].view_matrix,
            this.view_data[index].position,
            view_target,
            WORLD_UP
          );
      }

      {
        mat4.mul(this.view_data[index].view_projection_matrix, this.view_data[index].projection_matrix, this.view_data[index].view_matrix);
        mat4.invert(this.view_data[index].inverse_view_projection_matrix, this.view_data[index].view_projection_matrix);
      }

      if (this.buffer) {
        const offset = index * this.type_size_bytes;
        this.buffer.write(
          context,
          this.view_data.map(this._get_gpu_type_layout)
        );
      } else {
        this.build(context);
      }

      this.view_data[index].dirty = false;
    }
  }

  // Builds the GPU resident view buffer. Should be called during setup, before the simulation begins.
  build(context) {
    if (this.buffer) {
      this.buffer.destroy();
    }

    this.buffer = Buffer.create(context, {
      name: "view_buffer",
      data: this.view_data.map(this._get_gpu_type_layout),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    Renderer.get().refresh_global_shader_bindings();
  }

  _get_gpu_type_layout(item) {
    return Array.of(
      ...item.view_matrix,
      ...item.prev_view_matrix,
      ...item.projection_matrix,
      ...item.prev_projection_matrix,
      ...item.view_projection_matrix,
      ...item.inverse_view_projection_matrix,
    );
  }
}

export class SharedEnvironmentMapData {
  skybox = null;

  constructor() {
    if (SharedEnvironmentMapData.instance) {
      return SharedEnvironmentMapData.instance;
    }
    SharedEnvironmentMapData.instance = this;
  }

  static get() {
    if (!SharedEnvironmentMapData.instance) {
      return new SharedEnvironmentMapData();
    }
    return SharedEnvironmentMapData.instance;
  }

  async add_skybox(context, name, texture_paths) {
    const skybox = await Texture.load(context, texture_paths, {
      name: name,
      format: "rgba8unorm",
      dimension: "cube",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.skybox = skybox;

    return skybox;
  }

  remove_skybox() {
    this.skybox = null;
  }

  get_skybox() {
    return this.skybox;
  }
}

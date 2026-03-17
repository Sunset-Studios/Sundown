import { Renderer } from "./renderer.js";
import { Buffer } from "./buffer.js";
import { Name } from "../utility/names.js";
import { MeshBLAS } from "../acceleration/mesh_blas.js";
import { npot } from "../utility/math.js";

const vertex_buffer_name = "vertex_buffer";
const index_buffer_name = "index_buffer";
const mesh_bounds_buffer_name = "mesh_bounds_buffer";
const meshlet_buffer_name = "meshlet_buffer";
const meshlet_vertex_buffer_name = "meshlet_vertex_buffer";
const meshlet_triangle_buffer_name = "meshlet_triangle_buffer";
const meshlet_group_buffer_name = "meshlet_group_buffer";

const initial_max_meshes = 256;
const initial_vertex_buffer_size = 1024;
const initial_index_buffer_size = 1024 * 3;
const initial_meshlet_capacity = 256;
const initial_meshlet_vertex_capacity = 1024;
const initial_meshlet_triangle_capacity = 1024;
const initial_meshlet_group_capacity = 256;

const mesh_bounds_size = 8;
const meshlet_stride = 80;
const meshlet_group_stride = 64;
const storage_usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

export class MeshData {
  static is_initialized = false;
  static name_to_index = new Map();

  static bounds = null;
  static mesh_bounds_buffer = null;

  static mesh_count = 0;
  static vertex_buffer_head = 0;
  static index_buffer_head = 0;
  static meshlet_buffer_head = 0;
  static meshlet_vertex_buffer_head = 0;
  static meshlet_triangle_buffer_head = 0;
  static meshlet_group_buffer_head = 0;

  static vertex_data = null;
  static vertex_buffer = null;
  static index_data = null;
  static index_buffer = null;
  static meshlet_data = null;
  static meshlet_buffer = null;
  static meshlet_vertex_data = null;
  static meshlet_vertex_buffer = null;
  static meshlet_triangle_data = null;
  static meshlet_triangle_buffer = null;
  static meshlet_group_data = null;
  static meshlet_group_buffer = null;

  static initialize() {
    if (this.is_initialized) return;

    this.bounds = new Float32Array(initial_max_meshes * mesh_bounds_size);
    this.mesh_bounds_buffer = Buffer.create({
      name: mesh_bounds_buffer_name,
      usage: storage_usage,
      size: initial_max_meshes * mesh_bounds_size,
      force: true,
    });

    this.vertex_data = new Float32Array(initial_vertex_buffer_size);
    this.vertex_buffer = Buffer.create({
      name: vertex_buffer_name,
      usage: storage_usage,
      size: initial_vertex_buffer_size,
      force: true,
    });

    this.index_data = new Uint32Array(initial_index_buffer_size);
    this.index_buffer = Buffer.create({
      name: index_buffer_name,
      usage: GPUBufferUsage.INDEX | storage_usage,
      size: initial_index_buffer_size,
      element_type: "uint32",
      force: true,
    });

    this.meshlet_data = new Uint8Array(initial_meshlet_capacity * meshlet_stride);
    this.meshlet_buffer = Buffer.create({
      name: meshlet_buffer_name,
      raw_data: this.meshlet_data,
      usage: storage_usage,
      force: true,
    });

    this.meshlet_vertex_data = new Uint32Array(initial_meshlet_vertex_capacity);
    this.meshlet_vertex_buffer = Buffer.create({
      name: meshlet_vertex_buffer_name,
      raw_data: this.meshlet_vertex_data,
      usage: storage_usage,
      force: true,
    });

    this.meshlet_triangle_data = new Uint8Array(initial_meshlet_triangle_capacity);
    this.meshlet_triangle_buffer = Buffer.create({
      name: meshlet_triangle_buffer_name,
      raw_data: this.meshlet_triangle_data,
      usage: storage_usage,
      force: true,
    });

    this.meshlet_group_data = new Uint8Array(initial_meshlet_group_capacity * meshlet_group_stride);
    this.meshlet_group_buffer = Buffer.create({
      name: meshlet_group_buffer_name,
      raw_data: this.meshlet_group_data,
      usage: storage_usage,
      force: true,
    });

    this.is_initialized = true;
  }

  static register(mesh) {
    if (!this.is_initialized) {
      this.initialize();
    }

    const key = Name.from(mesh.name);
    if (this.name_to_index.has(key)) {
      return this.name_to_index.get(key);
    }

    const index = this.mesh_count++;
    this.name_to_index.set(key, index);

    mesh.mesh_data_index = index;
  }

  static update(mesh) {
    if (!this.is_initialized) {
      this.initialize();
    }

    if (mesh.vertices && mesh.vertex_buffer_offset === -1) {
      mesh.vertex_buffer_offset = this._add_vertex_data(mesh);
      mesh.index_buffer_offset = this._add_index_data(mesh);
    }

    if (mesh.meshlet_data && mesh.meshlet_buffer_offset === -1) {
      this._add_meshlet_data(mesh);
    }

    if (mesh.bounds_min_and_max) {
      this._set_bounds(mesh.mesh_data_index, mesh.bounds_min_and_max);
      MeshBLAS.build_from_mesh(mesh);
    }
  }

  static unregister(mesh) {
    if (!this.is_initialized) {
      this.initialize();
    }

    const key = Name.from(mesh.name);
    const index = this.name_to_index.get(key);
    if (index === undefined) return;

    this._set_bounds(index, [0, 0, 0, 0, 0, 0]);

    MeshBLAS.release(index);

    this.name_to_index.delete(key);
  }

  static update_mesh_bounds(mesh) {
    if (!this.is_initialized) {
      this.initialize();
    }

    const key = Name.from(mesh.name);
    const index = this.name_to_index.get(key);
    if (index === undefined) return;

    const b = mesh.bounds_min_and_max;
    if (!b) return;

    this._set_bounds(index, b);
    MeshBLAS.build_from_mesh(mesh);
  }

  static _set_bounds(index, b) {
    const base = index * mesh_bounds_size;
    if (base >= this.bounds.length) {
      this._resize_bounds(base * 2);
    }

    this.bounds[base + 0] = b[0] || 0.0;
    this.bounds[base + 1] = b[1] || 0.0;
    this.bounds[base + 2] = b[2] || 0.0;
    this.bounds[base + 3] = 0.0;
    this.bounds[base + 4] = b[3] || 0.0;
    this.bounds[base + 5] = b[4] || 0.0;
    this.bounds[base + 6] = b[5] || 0.0;
    this.bounds[base + 7] = 0.0;

    this._upload_bounds(index);
  }

  static _upload_bounds(index) {
    const base = index * mesh_bounds_size;
    this.mesh_bounds_buffer.write_raw(
      this.bounds.subarray(base, base + mesh_bounds_size),
      base * 4,
      mesh_bounds_size
    );
  }

  static _resize_bounds(new_max) {
    if (new_max <= this.bounds.length) return;

    const next_bounds = new Float32Array(new_max * mesh_bounds_size);
    next_bounds.set(this.bounds);
    this.bounds = next_bounds;

    this.mesh_bounds_buffer = Buffer.create({
      name: mesh_bounds_buffer_name,
      usage: storage_usage,
      size: this.bounds.length,
      force: true,
    });
  }

  static _add_vertex_data(mesh) {
    const packed = mesh.vertices.flatMap((v) =>
      v.position.concat(v.normal, v.tangent, v.bitangent, v.uv, v.extra_data)
    );

    const floats_per_vertex = packed.length / mesh.vertices.length;
    const write_offset = this.vertex_buffer_head;
    const required = write_offset + packed.length;
    if (required > this.vertex_data.length) {
      this._resize_vertex_data(required * 2);
    }

    this.vertex_data.set(packed, write_offset);
    this._upload_vertex_data();

    const old_vertex_offset = Math.floor(write_offset / floats_per_vertex);
    this.vertex_buffer_head = write_offset + packed.length;
    return old_vertex_offset;
  }

  static _upload_vertex_data() {
    this.vertex_buffer.write_raw(this.vertex_data);
  }

  static _resize_vertex_data(new_size) {
    if (new_size <= this.vertex_data.length) return;

    const next_vertex_data = new Float32Array(new_size);
    next_vertex_data.set(this.vertex_data);
    this.vertex_data = next_vertex_data;

    this.vertex_buffer = Buffer.create({
      name: vertex_buffer_name,
      usage: storage_usage,
      size: this.vertex_data.length,
      force: true,
    });

    Renderer.get().refresh_global_shader_bindings();
  }

  static _add_index_data(mesh) {
    const packed = mesh.indices;
    const write_offset = this.index_buffer_head;
    const required = write_offset + packed.length;
    if (required > this.index_data.length) {
      this._resize_index_data(required * 2);
    }

    this.index_data.set(packed, write_offset);
    this._upload_index_data();

    const old_index_offset = write_offset;
    this.index_buffer_head = write_offset + packed.length;
    return old_index_offset;
  }

  static _upload_index_data() {
    this.index_buffer.write_raw(this.index_data);
  }

  static _resize_index_data(new_size) {
    if (new_size <= this.index_data.length) return;

    const next_index_data = new Uint32Array(new_size);
    next_index_data.set(this.index_data);
    this.index_data = next_index_data;

    this.index_buffer = Buffer.create({
      name: index_buffer_name,
      usage: GPUBufferUsage.INDEX | storage_usage,
      size: this.index_data.length,
      element_type: "uint32",
      force: true,
    });
  }

  static _add_meshlet_data(mesh) {
    const upload = mesh.meshlet_data;
    if (!upload || upload.meshlets.length === 0) {
      mesh.meshlet_data = null;
      return;
    }

    const meshlet_offset = this.meshlet_buffer_head;
    const meshlet_vertex_offset = this.meshlet_vertex_buffer_head;
    const meshlet_triangle_offset = this.meshlet_triangle_buffer_head;
    const meshlet_group_offset = this.meshlet_group_buffer_head;

    this._resize_meshlet_data((meshlet_offset + upload.meshlets.length) * meshlet_stride);
    this._resize_meshlet_vertex_data(meshlet_vertex_offset + upload.meshlet_vertices.length);
    this._resize_meshlet_triangle_data(meshlet_triangle_offset + upload.meshlet_triangles.length);
    this._resize_meshlet_group_data(
      (meshlet_group_offset + upload.meshlet_groups.length) * meshlet_group_stride
    );

    for (let i = 0; i < upload.meshlet_vertices.length; i++) {
      this.meshlet_vertex_data[meshlet_vertex_offset + i] =
        upload.meshlet_vertices[i] + mesh.vertex_buffer_offset;
    }
    this.meshlet_vertex_buffer.write_raw(this.meshlet_vertex_data);

    this.meshlet_triangle_data.set(upload.meshlet_triangles, meshlet_triangle_offset);
    this.meshlet_triangle_buffer.write_raw(this.meshlet_triangle_data);

    const meshlet_view = new DataView(this.meshlet_data.buffer);
    for (let i = 0; i < upload.meshlets.length; i++) {
      const meshlet = upload.meshlets[i];
      const base = (meshlet_offset + i) * meshlet_stride;

      meshlet_view.setUint32(base + 0, meshlet_vertex_offset + meshlet.vertex_offset, true);
      meshlet_view.setUint32(base + 4, meshlet.vertex_count, true);
      meshlet_view.setUint32(base + 8, meshlet_triangle_offset + meshlet.triangle_offset, true);
      meshlet_view.setUint32(base + 12, meshlet.triangle_count, true);

      meshlet_view.setFloat32(base + 16, meshlet.center[0], true);
      meshlet_view.setFloat32(base + 20, meshlet.center[1], true);
      meshlet_view.setFloat32(base + 24, meshlet.center[2], true);
      meshlet_view.setFloat32(base + 28, meshlet.radius, true);

      meshlet_view.setFloat32(base + 32, meshlet.bounds_min[0], true);
      meshlet_view.setFloat32(base + 36, meshlet.bounds_min[1], true);
      meshlet_view.setFloat32(base + 40, meshlet.bounds_min[2], true);
      meshlet_view.setFloat32(base + 44, 0.0, true);

      meshlet_view.setFloat32(base + 48, meshlet.bounds_max[0], true);
      meshlet_view.setFloat32(base + 52, meshlet.bounds_max[1], true);
      meshlet_view.setFloat32(base + 56, meshlet.bounds_max[2], true);
      meshlet_view.setFloat32(base + 60, 0.0, true);

      meshlet_view.setFloat32(base + 64, meshlet.normal_cone_axis[0], true);
      meshlet_view.setFloat32(base + 68, meshlet.normal_cone_axis[1], true);
      meshlet_view.setFloat32(base + 72, meshlet.normal_cone_axis[2], true);
      meshlet_view.setFloat32(base + 76, meshlet.normal_cone_cutoff, true);
    }
    this.meshlet_buffer.write_raw(this.meshlet_data);

    const meshlet_group_view = new DataView(this.meshlet_group_data.buffer);
    for (let i = 0; i < upload.meshlet_groups.length; i++) {
      const group = upload.meshlet_groups[i];
      const base = (meshlet_group_offset + i) * meshlet_group_stride;

      meshlet_group_view.setUint32(base + 0, meshlet_offset + group.meshlet_offset, true);
      meshlet_group_view.setUint32(base + 4, group.meshlet_count, true);
      meshlet_group_view.setUint32(base + 8, 0, true);
      meshlet_group_view.setUint32(base + 12, 0, true);

      meshlet_group_view.setFloat32(base + 16, group.center[0], true);
      meshlet_group_view.setFloat32(base + 20, group.center[1], true);
      meshlet_group_view.setFloat32(base + 24, group.center[2], true);
      meshlet_group_view.setFloat32(base + 28, group.radius, true);

      meshlet_group_view.setFloat32(base + 32, group.bounds_min[0], true);
      meshlet_group_view.setFloat32(base + 36, group.bounds_min[1], true);
      meshlet_group_view.setFloat32(base + 40, group.bounds_min[2], true);
      meshlet_group_view.setFloat32(base + 44, 0.0, true);

      meshlet_group_view.setFloat32(base + 48, group.bounds_max[0], true);
      meshlet_group_view.setFloat32(base + 52, group.bounds_max[1], true);
      meshlet_group_view.setFloat32(base + 56, group.bounds_max[2], true);
      meshlet_group_view.setFloat32(base + 60, 0.0, true);
    }
    this.meshlet_group_buffer.write_raw(this.meshlet_group_data);

    mesh.meshlet_buffer_offset = meshlet_offset;
    mesh.meshlet_vertex_buffer_offset = meshlet_vertex_offset;
    mesh.meshlet_triangle_buffer_offset = meshlet_triangle_offset;
    mesh.meshlet_group_buffer_offset = meshlet_group_offset;
    mesh.meshlet_count = upload.meshlets.length;
    mesh.meshlet_group_count = upload.meshlet_groups.length;
    mesh.meshlet_sections = upload.sections.map((section) => ({
      meshlet_offset: meshlet_offset + section.meshlet_offset,
      meshlet_count: section.meshlet_count,
      meshlet_group_offset: meshlet_group_offset + section.meshlet_group_offset,
      meshlet_group_count: section.meshlet_group_count,
    }));

    this.meshlet_buffer_head = meshlet_offset + upload.meshlets.length;
    this.meshlet_vertex_buffer_head = meshlet_vertex_offset + upload.meshlet_vertices.length;
    this.meshlet_triangle_buffer_head = meshlet_triangle_offset + upload.meshlet_triangles.length;
    this.meshlet_group_buffer_head = meshlet_group_offset + upload.meshlet_groups.length;

    mesh.meshlet_data = null;
  }

  static _resize_meshlet_data(required_size) {
    if (required_size <= this.meshlet_data.length) return;
    required_size = npot(required_size);

    const next_meshlet_data = new Uint8Array(required_size);
    next_meshlet_data.set(this.meshlet_data);
    this.meshlet_data = next_meshlet_data;

    this.meshlet_buffer = Buffer.create({
      name: meshlet_buffer_name,
      raw_data: this.meshlet_data,
      usage: storage_usage,
      force: true,
    });
  }

  static _resize_meshlet_vertex_data(required_size) {
    if (required_size <= this.meshlet_vertex_data.length) return;
    required_size = npot(required_size);

    const next_meshlet_vertex_data = new Uint32Array(required_size);
    next_meshlet_vertex_data.set(this.meshlet_vertex_data);
    this.meshlet_vertex_data = next_meshlet_vertex_data;

    this.meshlet_vertex_buffer = Buffer.create({
      name: meshlet_vertex_buffer_name,
      raw_data: this.meshlet_vertex_data,
      usage: storage_usage,
      force: true,
    });
  }

  static _resize_meshlet_triangle_data(required_size) {
    if (required_size <= this.meshlet_triangle_data.length) return;
    required_size = npot(required_size);

    const next_meshlet_triangle_data = new Uint8Array(required_size);
    next_meshlet_triangle_data.set(this.meshlet_triangle_data);
    this.meshlet_triangle_data = next_meshlet_triangle_data;

    this.meshlet_triangle_buffer = Buffer.create({
      name: meshlet_triangle_buffer_name,
      raw_data: this.meshlet_triangle_data,
      usage: storage_usage,
      force: true,
    });
  }

  static _resize_meshlet_group_data(required_size) {
    if (required_size <= this.meshlet_group_data.length) return;
    required_size = npot(required_size);

    const next_meshlet_group_data = new Uint8Array(required_size);
    next_meshlet_group_data.set(this.meshlet_group_data);
    this.meshlet_group_data = next_meshlet_group_data;

    this.meshlet_group_buffer = Buffer.create({
      name: meshlet_group_buffer_name,
      raw_data: this.meshlet_group_data,
      usage: storage_usage,
      force: true,
    });
  }

  static get_index_by_name_hash(name_hash) {
    const index = this.name_to_index.get(name_hash) ?? 0xffffffff;
    return index;
  }

  static #gpu_data = {
    mesh_bounds_buffer: null,
    vertex_buffer: null,
    index_buffer: null,
    meshlet_buffer: null,
    meshlet_vertex_buffer: null,
    meshlet_triangle_buffer: null,
    meshlet_group_buffer: null,
  };

  static to_gpu_data() {
    if (!this.is_initialized) this.initialize();
    this.#gpu_data.mesh_bounds_buffer = this.mesh_bounds_buffer;
    this.#gpu_data.vertex_buffer = this.vertex_buffer;
    this.#gpu_data.index_buffer = this.index_buffer;
    this.#gpu_data.meshlet_buffer = this.meshlet_buffer;
    this.#gpu_data.meshlet_vertex_buffer = this.meshlet_vertex_buffer;
    this.#gpu_data.meshlet_triangle_buffer = this.meshlet_triangle_buffer;
    this.#gpu_data.meshlet_group_buffer = this.meshlet_group_buffer;
    return this.#gpu_data;
  }
}

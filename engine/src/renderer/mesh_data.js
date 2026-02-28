import { BufferFlags } from "./renderer_types.js";
import { Buffer } from "./buffer.js";
import { Name } from "../utility/names.js";
import { MeshBLAS } from "../acceleration/mesh_blas.js";

const vertex_buffer_name = "vertex_buffer";
const index_buffer_name = "index_buffer";
const mesh_bounds_buffer_name = "mesh_bounds_buffer";

const initial_max_meshes = 256;
const initial_vertex_buffer_size = 1024;
const initial_index_buffer_size = 1024 * 3;

const mesh_bounds_size = 8;

export class MeshData {
  static is_initialized = false;
  static name_to_index = new Map();

  static bounds = null;
  static mesh_bounds_buffer = null;

  static mesh_count = 0;
  static vertex_buffer_head = 0;
  static index_buffer_head = 0;
  static vertex_data = null;
  static vertex_buffer = null;
  static index_data = null;
  static index_buffer = null;

  static initialize() {
    if (this.is_initialized) return;

    this.bounds = new Float32Array(initial_max_meshes * mesh_bounds_size);
    this.mesh_bounds_buffer = Buffer.create({
      name: mesh_bounds_buffer_name,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: initial_max_meshes * mesh_bounds_size,
      force: true,
    });

    this.vertex_data = new Float32Array(initial_vertex_buffer_size);
    this.vertex_buffer = Buffer.create({
      name: vertex_buffer_name,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      flags: BufferFlags.GlobalBinding,
      size: initial_vertex_buffer_size,
      force: true,
    });

    this.index_data = new Uint32Array(initial_index_buffer_size);
    this.index_buffer = Buffer.create({
      name: index_buffer_name,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: initial_index_buffer_size,
      element_type: "uint32",
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

    // Ensure vertex data is uploaded first so BLAS leaf builder knows vertex offsets
    if (mesh.vertices && mesh.vertex_buffer_offset === -1) {
      mesh.vertex_buffer_offset = this._add_vertex_data(mesh);
      mesh.index_buffer_offset = this._add_index_data(mesh);
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

    // TODO: Remove and recycle vertex data, potentially using a fixed chunk free list

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
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: this.mesh_count * mesh_bounds_size,
      force: true,
    });
  }

  static _add_vertex_data(mesh) {
    const packed = mesh.vertices.flatMap((v) =>
      v.position.concat(v.normal, v.tangent, v.bitangent, v.uv, v.extra_data)
    );

    const floats_per_vertex = packed.length / mesh.vertices.length;

    const write_offset = this.vertex_buffer_head; // float index
    const required = write_offset + packed.length;
    if (required > this.vertex_data.length) {
      this._resize_vertex_data(required * 2);
    }

    this.vertex_data.set(packed, write_offset);

    this._upload_vertex_data();

    const old_vertex_offset = Math.floor(write_offset / floats_per_vertex); // vertex index
    this.vertex_buffer_head = write_offset + packed.length;
    return old_vertex_offset;
  }

  static _upload_vertex_data() {
    // Write whole vertex buffer for now
    this.vertex_buffer.write_raw(this.vertex_data);
  }

  static _resize_vertex_data(new_size) {
    if (new_size <= this.vertex_data.length) return;

    const next_vertex_data = new Float32Array(new_size);
    next_vertex_data.set(this.vertex_data);
    this.vertex_data = next_vertex_data;

    this.vertex_buffer = Buffer.create({
      name: vertex_buffer_name,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      flags: BufferFlags.GlobalBinding,
      size: this.vertex_data.length,
      force: true,
    });
  }

  static _add_index_data(mesh) {
    const packed = mesh.indices;

    const write_offset = this.index_buffer_head; // float index
    const required = write_offset + packed.length;
    if (required > this.index_data.length) {
      this._resize_index_data(required * 2);
    }

    this.index_data.set(packed, write_offset);

    this._upload_index_data();

    const old_index_offset = write_offset; // index index
    this.index_buffer_head = write_offset + packed.length;
    return old_index_offset;
  }

  static _upload_index_data() {
    // Write whole index buffer for now
    this.index_buffer.write_raw(this.index_data);
  }

  static _resize_index_data(new_size) {
    if (new_size <= this.index_data.length) return;

    const next_index_data = new Uint32Array(new_size);
    next_index_data.set(this.index_data);
    this.index_data = next_index_data;

    this.index_buffer = Buffer.create({
      name: index_buffer_name,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: this.index_data.length,
      element_type: "uint32",
      force: true,
    });
  }

  static get_index_by_name_hash(name_hash) {
    const index = this.name_to_index.get(name_hash) ?? 0xffffffff;
    return index;
  }

  static #gpu_data = { mesh_bounds_buffer: null, vertex_buffer: null, index_buffer: null };
  static to_gpu_data() {
    if (!this.is_initialized) this.initialize();
    this.#gpu_data.mesh_bounds_buffer = this.mesh_bounds_buffer;
    this.#gpu_data.vertex_buffer = this.vertex_buffer;
    this.#gpu_data.index_buffer = this.index_buffer;
    return this.#gpu_data;
  }
}

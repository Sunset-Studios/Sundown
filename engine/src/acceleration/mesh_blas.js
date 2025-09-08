import { Buffer } from "../renderer/buffer.js";
import { Renderer } from "../renderer/renderer.js";

const BVH4_NODE_BYTE_SIZE = 48;
const INITIAL_MAX_PAGES = 256;
const PAGE_SIZE = 64;

/**
 * Global buffer of mesh BLAS structures with simple page-based allocation.
 * Each mesh occupies one or more pages of BVH4 nodes inside a single GPU buffer.
 */
export class MeshBLAS {
  static initialized = false;
  static max_pages = INITIAL_MAX_PAGES;
  static blas_size = 0;
  static directory = null; // Uint32Array of entries: base, capacity, leaf_count, prim_base, node_base
  static free_pages = [];
  static allocations = new Map(); // mesh_id -> { start, pages, capacity }
  static dirty_meshes = new Set();
  static mesh_meta = new Map(); // mesh_id -> { first_vertex, leaf_count, index_buffer }
  static nodes_buffer = null;
  static directory_buffer = null;
  static leaf_bounds_uniforms = null; // Uint32Array
  static leaf_bounds_uniforms_buffer = null;
  static dummy_index_buffer = null;

  static initialize() {
    if (this.initialized) return;

    this.max_pages = INITIAL_MAX_PAGES;
    this.blas_size = INITIAL_MAX_PAGES * PAGE_SIZE;

    this.nodes_buffer = Buffer.create({
      name: "blas_nodes_buffer",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: this.blas_size * BVH4_NODE_BYTE_SIZE,
      force: true,
    });

    // Directory holds per-mesh BLAS metadata: base, capacity, leaf_count, prim_base, node_base
    this.directory = new Uint32Array(INITIAL_MAX_PAGES * 5);
    this.directory_buffer = Buffer.create({
      name: "blas_directory_buffer",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: this.directory.length * 4, // 5 * 4 bytes = 20 bytes
      force: true,
    });

    // Uniforms: mesh_id, base_node, first_vertex, triangle_count
    this.leaf_bounds_uniforms = new Uint32Array(6);
    this.leaf_bounds_uniforms_buffer = Buffer.create({
      name: "blas_leaf_bounds_uniforms",
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      size: 20, // 5 * 4 bytes = 20 bytes
      force: true,
    });

    // Placeholder index buffer binding
    this.dummy_index_buffer = Buffer.create({
      name: "mesh_dummy_indices",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: 4,
      force: true,
    });

    for (let i = 0; i < INITIAL_MAX_PAGES; i++) {
      this.free_pages.push(i);
    }
    this.initialized = true;
  }

  /** Allocate contiguous pages for a mesh. */
  static allocate(mesh_id, node_count) {
    this.initialize();

    const needed = Math.ceil(node_count / PAGE_SIZE);
    // Ensure we will have a contiguous run of free pages; may grow the buffer
    this._ensure_contiguous_free_pages(needed);

    let start = -1;
    const i = this._find_contiguous_free_run(needed);
    if (i >= 0) {
      start = this.free_pages[i];
      this.free_pages.splice(i, needed);
    }
    if (start < 0) return -1;
    this.allocations.set(mesh_id, { start, pages: needed, capacity: needed * PAGE_SIZE });
    return start * PAGE_SIZE;
  }

  /** Release pages associated with a mesh. */
  static release(mesh_id) {
    const alloc = this.allocations.get(mesh_id);
    if (!alloc) return;
    for (let i = 0; i < alloc.pages; i++) {
      this.free_pages.push(alloc.start + i);
    }
    this.free_pages.sort((a, b) => a - b);
    this.allocations.delete(mesh_id);
  }

  /** Upload node data for a mesh. */
  static upload(mesh_id, data) {
    const alloc = this.allocations.get(mesh_id);
    if (!alloc) return;
    const byteOffset = alloc.start * PAGE_SIZE * BVH4_NODE_BYTE_SIZE;
    this.nodes_buffer.write_raw(data, byteOffset, data.length);
  }

  /** Prepare BLAS allocation and mark mesh as dirty for processor-driven build. */
  static build_from_mesh(mesh, index_buffer) {
    if (!mesh) return;

    const mesh_id = mesh.mesh_data_index;
    if (mesh_id === undefined) return;

    const triangle_count =
      mesh.indices && mesh.indices.length ? Math.floor(mesh.indices.length / 3) : 0;
    if (triangle_count <= 0) return;

    // BVH4 internal nodes count ~ ceil((L - 1) / 2). Allocate leaves + internal nodes
    const leaf_count = triangle_count >>> 0;
    const internal_count = Math.ceil(Math.max(leaf_count - 1, 0) / 2);
    const total_required = Math.max(leaf_count + internal_count, 1);

    // Ensure directory can hold this mesh's entry
    this._ensure_directory_capacity_for_mesh(mesh_id);

    let base_node = 0;
    const existing = this.allocations.get(mesh_id);
    if (!existing || existing.capacity < total_required) {
      const offset = this.allocate(mesh_id, total_required);
      if (offset < 0) return;
      base_node = offset;
    } else {
      base_node = existing.start * PAGE_SIZE;
    }

    // Update directory entry: base, capacity, leaf_count, prim_base, node_base
    const alloc = this.allocations.get(mesh_id);
    const dir_base = mesh_id * 4;
    this.directory[dir_base + 0] = base_node >>> 0;
    this.directory[dir_base + 1] = alloc.capacity >>> 0;
    this.directory[dir_base + 2] = leaf_count >>> 0;
    this.directory[dir_base + 3] = 0; // prim_base
    this.directory[dir_base + 4] = 0; // node_base

    this.directory_buffer.write_raw(
      this.directory.subarray(dir_base, dir_base + 5),
      dir_base * 4,
      5
    );

    // Record mesh metadata and mark dirty for the MeshBLASProcessor to build this BLAS.
    this.mesh_meta.set(mesh_id, {
      first_vertex: mesh.vertex_buffer_offset || 0,
      leaf_count,
      base_node,
      index_buffer: index_buffer || this.dummy_index_buffer,
    });
    this.dirty_meshes.add(mesh_id);
  }

  // Ensure there is a contiguous run of free pages; may trigger growth
  static _ensure_contiguous_free_pages(required_pages) {
    // Fast path: try to find a contiguous run in current free list
    const has_run = this._find_contiguous_free_run(required_pages) >= 0;
    if (has_run) return;

    // Grow pages so that at least `required_pages` fresh pages are appended contiguously
    const additional_pages = Math.max(required_pages, this.max_pages);
    this._grow_nodes_pages(additional_pages);
  }

  static _find_contiguous_free_run(required_pages) {
    // Assumes free_pages is sorted
    for (let i = 0; i <= this.free_pages.length - required_pages; i++) {
      let contiguous = true;
      for (let j = 1; j < required_pages; j++) {
        if (this.free_pages[i + j] !== this.free_pages[i] + j) {
          contiguous = false;
          break;
        }
      }
      if (contiguous) {
        return i;
      }
    }
    return -1;
  }

  static _grow_nodes_pages(additional_pages) {
    const old_max_pages = this.max_pages;
    const new_max_pages = old_max_pages + Math.max(1, additional_pages);
    this.max_pages = new_max_pages;
    this.blas_size = new_max_pages * PAGE_SIZE;

    // Recreate the nodes buffer at the new size; contents are rebuilt
    this.nodes_buffer = Buffer.create({
      name: "blas_nodes_buffer",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: this.blas_size * BVH4_NODE_BYTE_SIZE,
      force: true,
    });

    // Append newly available pages as free and keep list sorted
    for (let i = old_max_pages; i < new_max_pages; i++) {
      this.free_pages.push(i);
    }
    this.free_pages.sort((a, b) => a - b);

    // Existing BLAS content must be rebuilt since buffer changed
    for (const mesh_id of this.allocations.keys()) {
      this.dirty_meshes.add(mesh_id);
    }

    Renderer.get().mark_bind_groups_dirty(true);
  }

  static _ensure_directory_capacity_for_mesh(mesh_id) {
    const required_entries = (mesh_id >>> 0) + 1;
    const current_entries = this.directory ? this.directory.length / 5 : 0;
    if (required_entries <= current_entries) return;

    const new_entries = Math.max(required_entries, Math.max(256, current_entries * 2));
    const next_directory = new Uint32Array(new_entries * 5);
    if (this.directory) next_directory.set(this.directory);
    this.directory = next_directory;

    this.directory_buffer = Buffer.create({
      name: "blas_directory_buffer",
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      size: this.directory.length * 4, // 5 * 4 bytes = 20 bytes
      force: true,
    });

    // Optional: upload all current directory entries (mostly zeros initially)
    this.directory_buffer.write_raw(this.directory);

    Renderer.get().mark_bind_groups_dirty(true);
  }

  static #return_gpu_data = {
    nodes_buffer: null,
    info_buffer: null,
    directory_buffer: null,
  };
  static to_gpu_data() {
    this.initialize();
    this.#return_gpu_data.nodes_buffer = this.nodes_buffer;
    this.#return_gpu_data.info_buffer = this.leaf_bounds_uniforms_buffer;
    this.#return_gpu_data.directory_buffer = this.directory_buffer;
    return this.#return_gpu_data;
  }
}

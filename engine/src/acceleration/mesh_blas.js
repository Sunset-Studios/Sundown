import { Buffer } from "../renderer/buffer.js";
import { MeshData } from "../renderer/mesh_data.js";
import { build_mesh_sbvh, patch_sbvh_child_indices } from "./sbvh_builder.js";

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// ██╗     ██╗███████╗███████╗██╗  ██╗██████╗ ██╗      █████╗ ███████╗    ███╗   ███╗ ██████╗ ██████╗
// ████╗ ████║██╔════╝██╔════╝██║  ██║██╔══██╗██║     ██╔══██╗██╔════╝    ████╗ ████║██╔════╝ ██╔══██╗
// ██╔████╔██║█████╗  ███████╗███████║██████╔╝██║     ███████║███████╗    ██╔████╔██║██║  ███╗██████╔╝
// ██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║██╔══██╗██║     ██╔══██║╚════██║    ██║╚██╔╝██║██║   ██║██╔══██╗
// ██║ ╚═╝ ██║███████╗███████║██║  ██║██████╔╝███████╗██║  ██║███████║    ██║ ╚═╝ ██║╚██████╔╝██║  ██║
// ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// MeshBLAS - Advanced Bottom-Level Acceleration Structure Manager
//
// This system implements a BVH2 architecture for high-performance raytracing:
//
// 🏗️  ARCHITECTURE OVERVIEW:
//     • BVH2 Storage: Binary tree nodes for traversal
//     • Page-Based Allocation: Efficient memory management with 64-node pages
//     • Dynamic Scratch Buffers: Auto-resizing build workspace
//     • Unified Buffer Pool: Reduces GPU bind group overhead
//
// 🧠 ALGORITHMIC FOUNDATION:
//     • BVH2 Build: Classic recursive binary hierarchy (2N-1 nodes for N triangles)
//     • Morton Code Sorting: Z-order curve for spatial coherence
//     • OneSweep Radix Sort: High-performance GPU sorting algorithm
//
// 💡 PERFORMANCE DESIGN:
//     • Memory Locality: Contiguous page allocation for cache efficiency
//     • Buffer Reuse: Single allocation per mesh type across all instances
//     • Build Batching: Amortized allocation overhead across multiple meshes
//     • GPU-Optimized: All heavy computation done on compute shaders
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                           📊 CORE BLAS CONFIGURATION CONSTANTS                               │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const BVH2_NODE_DATA_SIZE = 8; // BVH2 node: AABB (6 floats) + metadata (2 u32)
const INITIAL_MAX_PAGES = 256; // Conservative initial allocation (16K nodes)
const PAGE_SIZE = 64; // Nodes per page (optimal for GPU workgroup size)
const UINT32_BYTES = 4; // Standard 32-bit integer size
const DIRECTORY_ENTRY_SIZE = 5; // Per-mesh metadata: [bvh2_base, leaf_count, primitive_count, first_vertex, first_index]

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                         🔧 COMPUTE SHADER BUILD CONFIGURATION                                │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const SCRATCH_WORKGROUP_SIZE = 256; // GPU workgroup size (hardware optimal)
const SCRATCH_TILE_SIZE = SCRATCH_WORKGROUP_SIZE * 16; // 4K primitives per processing tile
const SCRATCH_RADIX = 256; // 8-bit radix for OneSweep sort (2^8)
const SCRATCH_RADIX_PASSES = 4; // 4 passes for 32-bit Morton codes
const INITIAL_SCRATCH_CAPACITY = 4096; // Start with 4K triangle capacity
const SCRATCH_GROWTH_FACTOR = 2.0; // Exponential growth to prevent frequent reallocations

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                              🖥️  GPU BUFFER CONFIGURATION                                    │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const STORAGE_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
 * ║                                    🏛️  MeshBLAS CLASS                                        ║
 * ║                      Advanced Bottom-Level Acceleration Structure Manager                   ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
 *
 * A sophisticated GPU memory management system for mesh-level raytracing acceleration structures.
 * Implements a BVH2 architecture with intelligent page-based allocation and dynamic scratch
 * buffer management for optimal performance across varying mesh complexities.
 *
 * ┌────────────────────────── 🎯 PRIMARY RESPONSIBILITIES ──────────────────────────┐
 * │                                                                                   │
 * │  • BVH2 Storage: Maintains binary BVH representation                           │
 * │  • Intelligent Page Allocation: 64-node pages for optimal memory utilization    │
 * │  • Dynamic Scratch Management: Auto-resizing build workspace buffers            │
 * │  • Unified Buffer Architecture: Single shared buffers reduce GPU overhead       │
 * │  • Build Pipeline Coordination: Seamless integration with compute shaders       │
 * │                                                                                   │
 * └───────────────────────────────────────────────────────────────────────────────────┘
 *
 * ┌────────────────────────── 🧮 MEMORY LAYOUT STRATEGY ────────────────────────────┐
 * │                                                                                   │
 * │  Directory Entry Format (per mesh):                                              │
 * │  ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬──────┐ │
 * │  │ bvh2_base   │ bvh2_cap    │ leaf_count  │ v_off       │ i_off       │ pad  │ │
 * │  │ (u32)       │ (u32)       │ (u32)       │ (u32)       │ (u32)       │(u32) │ │
 * │  └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴──────┘ │
 * │                                                                                   │
 * │  BVH2 Allocation: 2N-1 nodes for N triangles (binary tree structure)            │
 * │  BVH2 Allocation: 2N-1 nodes for N triangles (binary tree structure)           │
 * │                                                                                   │
 * └───────────────────────────────────────────────────────────────────────────────────┘
 */
export class MeshBLAS {
  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                                  🏗️  CORE SYSTEM STATE                                     ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝
  static #initialized = false; // Main system initialization flag
  static #scratch_initialized = false; // Scratch buffer system initialization flag

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                             📦 PAGED BLAS STORAGE SYSTEM                                 ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝
  static #bvh2_nodes_buffer = null; // Global BVH2 storage: binary tree nodes
  static #directory_buffer = null; // Per-mesh allocation metadata directory (GPU accessible)
  static #directory = null; // CPU-side directory mirror for fast updates
  static #dummy_index_buffer = null; // Fallback buffer for meshes without index data

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                           🔧 BVH2 ALLOCATION MANAGEMENT SYSTEM                            ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝
  static #bvh2_max_pages = INITIAL_MAX_PAGES; // Current maximum page capacity
  static #bvh2_blas_size = INITIAL_MAX_PAGES * PAGE_SIZE; // Total nodes available
  static #bvh2_free_pages = []; // Available page indices (sorted)
  static #bvh2_allocations = new Map(); // mesh_id -> AllocationInfo mapping
  static #bvh2_allocated_node_count = 0; // Actual number of BVH2 nodes allocated (not capacity)

  static #directory_entry_count = 0; // Actual number of directory entries used

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                              📊 MESH LIFECYCLE TRACKING                                   ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝
  static #mesh_metadata = new Map(); // mesh_id -> comprehensive build metadata
  static #dirty_meshes = new Set(); // mesh_ids requiring BVH rebuild

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                        🚀 DYNAMIC SINGLE-MESH SCRATCH SYSTEM                              ║
  // ║                                                                                            ║
  // ║  These buffers auto-resize to accommodate meshes of varying complexity. The scratch       ║
  // ║  system uses an exponential growth strategy to minimize allocation overhead while          ║
  // ║  supporting everything from small debug meshes to massive architectural models.           ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝
  static #current_scratch_capacity = 0; // Current triangle capacity

  // Morton Code Generation & Sorting Buffers
  static #morton_codes_buffer = null; // Primary Morton codes (32-bit Z-order)
  static #temp_morton_codes_buffer = null; // Temporary storage for radix sort
  static #sorted_indices_buffer = null; // Triangle indices sorted by Morton code
  static #temp_sorted_indices_buffer = null; // Temporary storage for sort passes

  // OneSweep Radix Sort Infrastructure
  static #onesweep_global_hist_buffer = null; // Global histogram across all passes
  static #onesweep_pass_hist_buffer = null; // Per-pass histogram data
  static #onesweep_tile_indices_buffer = null; // Tile boundary indices for work distribution

  // BVH Build State Management
  static #parent_indices_buffer = null; // Parent node indices for hierarchy
  static #bvh_index_pairs_buffer = null; // Child-parent index pairs for BVH construction

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                                 🚀 PUBLIC API METHODS                                      ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                           🏗️  SYSTEM INITIALIZATION                                     │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Initialize the paged MeshBLAS system. Sets up the BVH2 buffer pool,
   * creates the allocation directory, and prepares the scratch buffer system.
   *
   * This method is safe to call multiple times and will only initialize once.
   *
   * 🧠 INITIALIZATION STRATEGY:
   *    • Conservative Initial Allocation: Starts with 256 pages (16K nodes each)
   *    • Lazy Scratch Initialization: Scratch buffers created on first build request
   *    • Directory Pre-allocation: Metadata directory sized for initial capacity
   *    • Dummy Buffer Creation: Fallback for meshes without explicit index data
   */
  static initialize() {
    if (this.#initialized) return;

    // Create BVH2 nodes storage buffer (leaves + internal BVH2 nodes)
    this.#bvh2_nodes_buffer = Buffer.create({
      name: "mesh_blas_bvh2_nodes",
      usage: STORAGE_USAGE,
      size: this.#bvh2_blas_size * BVH2_NODE_DATA_SIZE,
      force: true,
    });

    // Create directory for per-mesh allocation metadata
    this.#directory = new Uint32Array(INITIAL_MAX_PAGES * DIRECTORY_ENTRY_SIZE);
    this.#directory_buffer = Buffer.create({
      name: "mesh_blas_directory",
      usage: STORAGE_USAGE,
      size: this.#directory.length,
      force: true,
    });

    // Create dummy index buffer for meshes without indices
    this.#dummy_index_buffer = Buffer.create({
      name: "mesh_blas_dummy_indices",
      usage: STORAGE_USAGE,
      size: 1,
      force: true,
    });

    // Initialize free page list
    for (let i = 0; i < INITIAL_MAX_PAGES; i++) {
      this.#bvh2_free_pages.push(i);
    }

    this.#initialized = true;

    this.#initialize_scratch();
  }

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                            🗑️  MESH RESOURCE CLEANUP                                   │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Release all resources associated with a mesh, including BVH2 page allocations,
   * metadata tracking, and rebuild flags. This performs a complete cleanup of the mesh
   * from the acceleration structure system.
   *
   * 🔄 CLEANUP PROCESS:
   *    • BVH2 Pages: Returns allocated pages to the free pool
   *    • Free List Maintenance: Keeps page lists sorted for optimal allocation
   *    • Metadata Cleanup: Removes all tracking data for the mesh
   *    • Build State Reset: Clears any pending rebuild flags
   *
   * @param {number} mesh_id - Unique mesh identifier to release
   */
  static release(mesh_id) {
    // Release BVH2 allocation
    const bvh2_allocation = this.#bvh2_allocations.get(mesh_id);
    if (bvh2_allocation) {
      for (let i = 0; i < bvh2_allocation.page_count; i++) {
        this.#bvh2_free_pages.push(bvh2_allocation.start_page + i);
      }
      this.#bvh2_free_pages.sort((a, b) => a - b);
      this.#bvh2_allocations.delete(mesh_id);
    }

    // Recalculate actual allocation counts after release
    this.#recalculate_allocation_counts();

    // Clean up all related data
    this.#mesh_metadata.delete(mesh_id);
    this.#dirty_meshes.delete(mesh_id);
  }
  
  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                        📊 RECALCULATE ALLOCATION COUNTS                                  │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Recalculates the actual allocation counts after mesh releases.
   *
   * @private
   */
  static #recalculate_allocation_counts() {
    // Recalculate BVH2 high water mark
    this.#bvh2_allocated_node_count = 0;
    for (const allocation of this.#bvh2_allocations.values()) {
      this.#bvh2_allocated_node_count = Math.max(
        this.#bvh2_allocated_node_count,
        allocation.base_node_index + allocation.actual_node_count
      );
    }
    
    // Recalculate directory entry count (highest mesh_id + 1)
    this.#directory_entry_count = 0;
    for (const mesh_id of this.#mesh_metadata.keys()) {
      this.#directory_entry_count = Math.max(this.#directory_entry_count, mesh_id + 1);
    }
  }

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                       📐 MESH BLAS ALLOCATION & BUILD SETUP                             │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Analyzes a mesh and prepares BVH2 allocations based on triangle count.
   * Calculates optimal page requirements and updates the allocation directory. Marks the
   * mesh as dirty to trigger a rebuild during the next build pass.
   *
   * 🔍 ALLOCATION ANALYSIS:
   *    • Triangle Count Extraction: From mesh.indices.length / 3
   *    • BVH2 Requirements: 2N-1 nodes for N triangles (binary tree formula)
   *    • Page Calculation: Rounds up to 64-node page boundaries
   *    • Directory Update: Records base indices and capacities for GPU access
   *
   * @param {object} mesh - Mesh object containing indices and GPU buffer references
   */
  static build_from_mesh(mesh) {
    this.initialize();

    if (!mesh || mesh.mesh_data_index === undefined) return;

    const mesh_id = mesh.mesh_data_index;
    const first_vertex = mesh.vertex_buffer_offset || 0;
    const first_index = mesh.index_buffer_offset || 0;
    const triangle_count = Math.floor(mesh.index_count / 3);
    if (triangle_count <= 0) return;

    let sbvh_build = null;
    if (mesh.cooked_sbvh?.node_data) {
      sbvh_build = {
        primitive_count: mesh.cooked_sbvh.primitive_count,
        reference_count: mesh.cooked_sbvh.reference_count,
        node_count: mesh.cooked_sbvh.node_count,
        node_data: mesh.cooked_sbvh.node_data,
      };
    } else if (mesh.vertices && mesh.indices) {
      sbvh_build = build_mesh_sbvh(mesh);
    } else {
      const existing_meta = this.#mesh_metadata.get(mesh_id);
      if (existing_meta?.local_node_data && existing_meta.primitive_count > 0) {
        sbvh_build = {
          primitive_count: existing_meta.primitive_count,
          reference_count: existing_meta.leaf_count,
          node_count: existing_meta.bvh2_node_count,
          node_data: existing_meta.local_node_data,
        };
      }
    }

    if (!sbvh_build) return;

    // Ensure system capacity and update allocations for both buffer types
    this.#ensure_directory_capacity_for_mesh(mesh_id);
    this.#update_mesh_allocation(
      mesh_id,
      sbvh_build.primitive_count,
      sbvh_build.reference_count,
      first_vertex,
      first_index,
      sbvh_build.node_data
    );

    this.#dirty_meshes.add(mesh_id);
  }

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                      🛠️  DYNAMIC SCRATCH BUFFER PREPARATION                            │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Ensures scratch buffers have sufficient capacity for building the specified mesh size.
   * Uses intelligent exponential growth strategy to minimize reallocations while supporting
   * everything from tiny debug meshes to massive architectural models.
   *
   * 📊 CAPACITY MANAGEMENT:
   *    • Automatic Resize: Grows buffers if primitive_count exceeds current capacity
   *    • Exponential Growth: 2x growth factor prevents frequent reallocations
   *    • Thread Block Calculation: Determines GPU workgroup requirements
   *    • Memory Efficiency: Balances capacity vs. waste for optimal performance
   *
   * 🚀 PERFORMANCE IMPACT:
   *    Buffer resizing triggers bind group invalidation, causing a brief GPU pipeline stall.
   *    The exponential growth strategy minimizes this overhead by growing aggressively.
   *
   * @param {number} mesh_id - Mesh identifier (used for debug logging)
   * @param {number} primitive_count - Number of triangles requiring build capacity
   * @returns {object} Build configuration: {mesh_id, primitive_count, thread_blocks, scratch_capacity}
   */
  static prepare_build(mesh_id, primitive_count) {
    this.initialize();

    // Ensure scratch buffers can handle this mesh size
    this.#ensure_scratch_capacity(primitive_count);

    const thread_blocks = Math.max(1, Math.ceil(primitive_count / SCRATCH_TILE_SIZE));

    return {
      mesh_id,
      primitive_count,
      thread_blocks,
      scratch_capacity: this.#current_scratch_capacity,
    };
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                              🔧 PRIVATE IMPLEMENTATION METHODS                            ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                        🚀 SCRATCH BUFFER SYSTEM INITIALIZATION                          │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Initialize the dynamic single-mesh scratch system with conservative initial capacity.
   * This creates all the temporary buffers needed for BVH construction including Morton
   * code generation, radix sorting, and BVH2 construction state management.
   *
   * 🛠️ BUFFER CREATION STRATEGY:
   *    • Conservative Start: 4K triangle capacity to handle most common cases
   *    • Complete Buffer Set: All buffers needed for full build pipeline
   *    • Lazy Initialization: Only called when first build is requested
   *
   * @private
   */
  static #initialize_scratch() {
    if (this.#scratch_initialized) return;

    this.#current_scratch_capacity = INITIAL_SCRATCH_CAPACITY;
    this.#create_scratch_buffers(this.#current_scratch_capacity);
    this.#scratch_initialized = true;
  }

  /**
   * Ensure scratch buffers have sufficient capacity for the given primitive count.
   * Resizes buffers if necessary using exponential growth strategy.
   * @param {number} required_primitives - Number of primitives that need to be processed
   * @private
   */
  static #ensure_scratch_capacity(required_primitives) {
    this.#initialize_scratch();

    // Check if current capacity is sufficient
    if (required_primitives <= this.#current_scratch_capacity) {
      return;
    }

    // Calculate new capacity using exponential growth
    let new_capacity = this.#current_scratch_capacity;
    while (new_capacity < required_primitives) {
      new_capacity = Math.ceil(new_capacity * SCRATCH_GROWTH_FACTOR);
    }

    // Recreate buffers with new capacity
    this.#current_scratch_capacity = new_capacity;
    this.#create_scratch_buffers(new_capacity);
  }

  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                              💾 BVH ALLOCATION SYSTEM                                       │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                        📊 MESH ALLOCATION COORDINATOR                                   │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Coordinates the allocation of BVH2 node space for a mesh, updates the
   * GPU-accessible directory with allocation metadata, and maintains CPU-side tracking
   * information for the build pipeline.
   *
   * 🔄 ALLOCATION PROCESS:
   *    • BVH2 Allocation: Binary tree structure requiring 2N-1 nodes
   *    • Directory Update: GPU metadata with base indices and capacities
   *    • Metadata Storage: CPU build pipeline tracking information
   *
   * @param {number} mesh_id - Unique mesh identifier
   * @param {number} primitive_count - Number of source triangles in the mesh
   * @param {number} reference_count - Number of BLAS leaf references after spatial splitting
   * @param {number} first_vertex - Vertex buffer offset for this mesh
   * @param {object} first_index - Index buffer offset for this mesh
   * @param {Float32Array} node_data - Flattened BLAS node payload for GPU upload
   * @private
   */
  static #update_mesh_allocation(
    mesh_id,
    primitive_count,
    reference_count,
    first_vertex,
    first_index,
    node_data
  ) {
    const index_buffer = MeshData.index_buffer;

    // Allocate BVH2 nodes (2N - 1) using the split-reference leaf count.
    let bvh2_base_index = 0;
    const existing_bvh2 = this.#bvh2_allocations.get(mesh_id);
    const bvh2_nodes_required = Math.max(1, 2 * reference_count - 1);

    if (!existing_bvh2 || existing_bvh2.node_capacity < bvh2_nodes_required) {
      bvh2_base_index = this.#allocate_bvh2(mesh_id, reference_count);
      if (bvh2_base_index < 0) return;
    } else {
      bvh2_base_index = existing_bvh2.base_node_index;
      existing_bvh2.actual_node_count = bvh2_nodes_required;
      this.#bvh2_allocated_node_count = Math.max(
        this.#bvh2_allocated_node_count,
        existing_bvh2.base_node_index + existing_bvh2.actual_node_count
      );
    }

    const directory_offset = mesh_id * DIRECTORY_ENTRY_SIZE;

    this.#directory[directory_offset + 0] = bvh2_base_index >>> 0;
    this.#directory[directory_offset + 1] = reference_count >>> 0;
    this.#directory[directory_offset + 2] = primitive_count >>> 0;
    this.#directory[directory_offset + 3] = first_vertex >>> 0;
    this.#directory[directory_offset + 4] = first_index >>> 0;

    // Write directory entry to GPU
    this.#directory_buffer.write_raw(
      this.#directory.subarray(directory_offset, directory_offset + DIRECTORY_ENTRY_SIZE),
      directory_offset * UINT32_BYTES,
      DIRECTORY_ENTRY_SIZE
    );

    // Track actual directory entry count
    this.#directory_entry_count = Math.max(this.#directory_entry_count, mesh_id + 1);

    const gpu_node_data = patch_sbvh_child_indices(node_data, bvh2_base_index, BVH2_NODE_DATA_SIZE);

    // Store mesh metadata for build pipeline
    this.#mesh_metadata.set(mesh_id, {
      first_vertex: first_vertex,
      first_index: first_index,
      primitive_count: primitive_count,
      leaf_count: reference_count,
      bvh2_base_node_index: bvh2_base_index,
      bvh2_node_count: bvh2_nodes_required,
      local_node_data: node_data,
      node_data: gpu_node_data,
      index_buffer: index_buffer || this.#dummy_index_buffer,
    });
  }

  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                              🌳 BVH2 PAGE ALLOCATION SYSTEM                                  │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                           🔢 BVH2 CONTIGUOUS PAGE ALLOCATOR                             │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Allocates contiguous page runs for BVH2 node storage using the classic binary tree
   * formula: 2N-1 nodes for N triangles. Ensures spatial locality for optimal GPU
   * cache performance during traversal operations.
   *
   * 🧮 ALLOCATION MATHEMATICS:
   *    • Binary Tree Formula: 2N - 1 total nodes (N leaves + N-1 internal)
   *    • Page Requirements: ⌈nodes / 64⌉ contiguous pages needed
   *    • Memory Layout: Contiguous allocation ensures cache-friendly access patterns
   *    • Growth Strategy: Auto-expands buffer if insufficient contiguous space
   *
   * @param {number} mesh_id - Unique mesh identifier for tracking
   * @param {number} primitive_count - Number of triangles requiring BVH2 storage
   * @returns {number} Base node index in global BVH2 buffer, or -1 if allocation fails
   * @private
   */
  static #allocate_bvh2(mesh_id, primitive_count) {
    this.initialize();

    const bvh2_node_count = Math.max(1, 2 * primitive_count - 1); // 2N - 1
    const pages_needed = Math.ceil(bvh2_node_count / PAGE_SIZE);

    // Ensure contiguous free pages are available
    this.#ensure_contiguous_free_bvh2_pages(pages_needed);

    // Find and allocate contiguous page run
    const page_index = this.#find_contiguous_free_bvh2_run(pages_needed);
    if (page_index < 0) return -1;

    const start_page = this.#bvh2_free_pages[page_index];
    this.#bvh2_free_pages.splice(page_index, pages_needed);

    // Create BVH2 allocation record
    const allocation = {
      start_page: start_page,
      page_count: pages_needed,
      node_capacity: pages_needed * PAGE_SIZE,
      base_node_index: start_page * PAGE_SIZE,
      actual_node_count: bvh2_node_count,
    };

    this.#bvh2_allocations.set(mesh_id, allocation);
    
    // Update actual allocated node count
    this.#bvh2_allocated_node_count = Math.max(
      this.#bvh2_allocated_node_count,
      allocation.base_node_index + allocation.actual_node_count
    );

    return allocation.base_node_index;
  }


  /**
   * Ensure contiguous free BVH2 pages are available, growing buffer if needed.
   * @private
   */
  static #ensure_contiguous_free_bvh2_pages(required_pages) {
    if (this.#find_contiguous_free_bvh2_run(required_pages) >= 0) return;
    const additional_pages = Math.max(required_pages, this.#bvh2_max_pages);
    this.#grow_bvh2_buffer(additional_pages);
  }


  /**
   * Find index of contiguous free BVH2 page run.
   * @private
   */
  static #find_contiguous_free_bvh2_run(required_pages) {
    const free_count = this.#bvh2_free_pages.length;
    if (required_pages > free_count) return -1;

    for (let i = 0; i <= free_count - required_pages; i++) {
      let is_contiguous = true;
      const start_page = this.#bvh2_free_pages[i];

      for (let j = 1; j < required_pages; j++) {
        if (this.#bvh2_free_pages[i + j] !== start_page + j) {
          is_contiguous = false;
          break;
        }
      }

      if (is_contiguous) return i;
    }

    return -1;
  }


  /**
   * Grow the BVH2 nodes buffer by adding additional pages.
   * @private
   */
  static #grow_bvh2_buffer(additional_pages) {
    const old_max_pages = this.#bvh2_max_pages;
    const new_max_pages = old_max_pages + Math.max(1, additional_pages);

    this.#bvh2_max_pages = new_max_pages;
    this.#bvh2_blas_size = new_max_pages * PAGE_SIZE;

    // Recreate BVH2 nodes buffer with new capacity
    this.#bvh2_nodes_buffer = Buffer.create({
      name: "mesh_blas_bvh2_nodes",
      usage: STORAGE_USAGE,
      size: this.#bvh2_blas_size * BVH2_NODE_DATA_SIZE,
      force: true,
    });

    // Add new pages to free list (maintain sorted order)
    for (let i = old_max_pages; i < new_max_pages; i++) {
      this.#bvh2_free_pages.push(i);
    }
    this.#bvh2_free_pages.sort((a, b) => a - b);

    // Mark all meshes dirty since buffer was recreated
    for (const mesh_id of this.#bvh2_allocations.keys()) {
      this.#dirty_meshes.add(mesh_id);
    }
  }


  /**
   * Ensure directory has capacity for the given mesh ID.
   * @private
   */
  static #ensure_directory_capacity_for_mesh(mesh_id) {
    const required_entries = (mesh_id >>> 0) + 1;
    const current_entries = this.#directory ? this.#directory.length / DIRECTORY_ENTRY_SIZE : 0;

    if (required_entries <= current_entries) return;

    // Grow directory with reasonable expansion policy
    const new_entries = Math.max(
      required_entries,
      Math.max(INITIAL_MAX_PAGES, current_entries * 2)
    );

    const new_directory = new Uint32Array(new_entries * DIRECTORY_ENTRY_SIZE);
    new_directory.set(this.#directory);
    this.#directory = new_directory;

    // Recreate directory buffer
    this.#directory_buffer = Buffer.create({
      name: "mesh_blas_directory",
      usage: STORAGE_USAGE,
      raw_data: this.#directory,
      force: true,
    });
  }

  // ---------------------------------------------------------------------------
  // Dynamic Scratch Buffer Management System
  // ---------------------------------------------------------------------------

  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                           🚧 SCRATCH BUFFER INFRASTRUCTURE                                   │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                        🔨 DYNAMIC SCRATCH BUFFER CREATION                               │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Creates a complete set of scratch buffers optimized for the specified primitive capacity.
   * These temporary buffers support the full BVH build pipeline from Morton code generation
   * through BVH2 construction.
   *
   * 📦 BUFFER ARCHITECTURE:
   *    • Morton Codes: Primary + temporary for radix sort ping-pong
   *    • Sorted Indices: Triangle indices organized by spatial Morton code
   *    • OneSweep Histograms: Global + per-pass statistics for efficient sorting
   *    • BVH Build State: Parent indices and index pair workspace
   *
   * @param {number} primitive_capacity - Maximum number of triangles to support
   * @private
   */
  static #create_scratch_buffers(primitive_capacity) {
    const max_thread_blocks = Math.max(1, Math.ceil(primitive_capacity / SCRATCH_TILE_SIZE));
    const pass_histogram_size =
      max_thread_blocks * SCRATCH_RADIX * SCRATCH_RADIX_PASSES;

    this.#morton_codes_buffer = Buffer.create({
      name: "mesh_blas_morton_codes",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: primitive_capacity,
      force: true,
    });

    this.#temp_morton_codes_buffer = Buffer.create({
      name: "mesh_blas_temp_morton_codes",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: primitive_capacity,
      force: true,
    });

    this.#sorted_indices_buffer = Buffer.create({
      name: "mesh_blas_sorted_indices",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: primitive_capacity,
      force: true,
    });

    this.#temp_sorted_indices_buffer = Buffer.create({
      name: "mesh_blas_temp_sorted_indices",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: primitive_capacity,
      force: true,
    });

    this.#onesweep_global_hist_buffer = Buffer.create({
      name: "mesh_blas_global_hist",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: SCRATCH_RADIX * SCRATCH_RADIX_PASSES,
      force: true,
    });

    this.#onesweep_pass_hist_buffer = Buffer.create({
      name: "mesh_blas_pass_hist",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: pass_histogram_size,
      force: true,
    });

    this.#onesweep_tile_indices_buffer = Buffer.create({
      name: "mesh_blas_tile_indices",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: SCRATCH_RADIX_PASSES,
      force: true,
    });

    this.#parent_indices_buffer = Buffer.create({
      name: "mesh_blas_parent_indices",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: primitive_capacity,
      force: true,
    });

    this.#bvh_index_pairs_buffer = Buffer.create({
      name: "mesh_blas_index_pairs",
      usage: STORAGE_USAGE | GPUBufferUsage.COPY_SRC,
      size: primitive_capacity * 2, // u64 per primitive
      force: true,
    });
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                               📡 PUBLIC ACCESSOR METHODS                                  ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                              🏷️  SYSTEM STATE ACCESSORS                                    │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /** @returns {Set<number>} Set of mesh IDs requiring BVH rebuild */
  static get dirty_meshes() {
    return this.#dirty_meshes;
  }

  /** @returns {boolean} True if both main and scratch systems are initialized */
  static get is_ready() {
    return this.#initialized && this.#scratch_initialized;
  }

  /** @returns {number} Current scratch buffer capacity in triangles */
  static get scratch_capacity() {
    return this.#current_scratch_capacity;
  }

  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                        🧮 MORTON CODE & SORTING BUFFER ACCESSORS                             │
  // │                                                                                               │
  // │  These buffers support the Morton code generation and OneSweep radix sorting pipeline.      │
  // │  All buffers auto-resize based on mesh complexity via the prepare_build() method.           │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /** @returns {Buffer} Primary Morton codes buffer (32-bit Z-order values) */
  static get morton_codes_buffer() {
    return this.#morton_codes_buffer;
  }

  /** @returns {Buffer} Temporary Morton codes for radix sort ping-pong */
  static get temp_morton_codes_buffer() {
    return this.#temp_morton_codes_buffer;
  }

  /** @returns {Buffer} Triangle indices sorted by Morton code */
  static get sorted_indices_buffer() {
    return this.#sorted_indices_buffer;
  }

  /** @returns {Buffer} Temporary indices for sort passes */
  static get temp_sorted_indices_buffer() {
    return this.#temp_sorted_indices_buffer;
  }

  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                          📊 ONESWEEP RADIX SORT INFRASTRUCTURE                               │
  // │                                                                                               │
  // │  OneSweep is a high-performance GPU sorting algorithm that uses global and per-pass          │
  // │  histograms to achieve optimal work distribution across GPU compute units.                   │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /** @returns {Buffer} Global histogram across all sort passes */
  static get onesweep_global_hist_buffer() {
    return this.#onesweep_global_hist_buffer;
  }

  /** @returns {Buffer} Per-pass histogram data for work distribution */
  static get onesweep_pass_hist_buffer() {
    return this.#onesweep_pass_hist_buffer;
  }

  /** @returns {Buffer} Tile boundary indices for workgroup coordination */
  static get onesweep_tile_indices_buffer() {
    return this.#onesweep_tile_indices_buffer;
  }

  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                             🌳 BVH BUILD STATE MANAGEMENT                                    │
  // │                                                                                               │
  // │  These buffers coordinate the multi-step BVH construction process.                           │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /** @returns {Buffer} Parent node indices for hierarchy construction */
  static get parent_idx_buffer() {
    return this.#parent_indices_buffer;
  }

  /** @returns {Buffer} Child-parent index pairs for BVH construction */
  static get bvh_index_pairs_buffer() {
    return this.#bvh_index_pairs_buffer;
  }

  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                             🔍 BLAS SIZE ACCESSORS                                          │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /** @returns {Buffer} BLAS bounds size */
  static get bounds_size() {
    return this.#bvh2_blas_size;
  }


  // ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  // │                              🔍 MESH METADATA & GPU DATA                                     │
  // └─────────────────────────────────────────────────────────────────────────────────────────────┘

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                            📋 MESH METADATA RETRIEVAL                                   │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Retrieve comprehensive build metadata for a specific mesh, including allocation information,
   * node counts, and buffer references needed by the build pipeline.
   *
   * @param {number} mesh_id - Unique mesh identifier
   * @returns {object|undefined} Complete mesh metadata or undefined if mesh not found
   */
  static get_mesh_meta(mesh_id) {
    return this.#mesh_metadata.get(mesh_id);
  }

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                           🖥️  GPU SHADER BINDING DATA                                   │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Provides a cached object containing all GPU buffers needed for shader binding. This includes
   * the main BVH2 storage buffers and the allocation directory for GPU-side mesh lookups.
   *
   * 🔄 CACHING STRATEGY:
   *    • Static Cache Object: Reused to minimize allocation overhead
   *    • Automatic Initialization: Ensures system is ready before data access
   *    • Buffer Reference Updates: Refreshes cache with current buffer instances
   *
   * @returns {object} GPU binding data: {bvh2_nodes_buffer, directory_buffer}
   */
  static #gpu_data_cache = {
    bvh2_nodes_buffer: null,
    directory_buffer: null,
    morton_codes_buffer: null,
    sorted_indices_buffer: null,
    temp_morton_codes_buffer: null,
    temp_sorted_indices_buffer: null,
    onesweep_global_hist_buffer: null,
    onesweep_pass_hist_buffer: null,
    onesweep_tile_indices_buffer: null,
    parent_idx_buffer: null,
    bvh_index_pairs_buffer: null,
  };
  static to_gpu_data() {
    this.initialize();
    this.#gpu_data_cache.bvh2_nodes_buffer = this.#bvh2_nodes_buffer;
    this.#gpu_data_cache.directory_buffer = this.#directory_buffer;
    this.#gpu_data_cache.morton_codes_buffer = this.#morton_codes_buffer;
    this.#gpu_data_cache.sorted_indices_buffer = this.#sorted_indices_buffer;
    this.#gpu_data_cache.temp_morton_codes_buffer = this.#temp_morton_codes_buffer;
    this.#gpu_data_cache.temp_sorted_indices_buffer = this.#temp_sorted_indices_buffer;
    this.#gpu_data_cache.onesweep_global_hist_buffer = this.#onesweep_global_hist_buffer;
    this.#gpu_data_cache.onesweep_pass_hist_buffer = this.#onesweep_pass_hist_buffer;
    this.#gpu_data_cache.onesweep_tile_indices_buffer = this.#onesweep_tile_indices_buffer;
    this.#gpu_data_cache.parent_idx_buffer = this.#parent_indices_buffer;
    this.#gpu_data_cache.bvh_index_pairs_buffer = this.#bvh_index_pairs_buffer;
    return this.#gpu_data_cache;
  }
}

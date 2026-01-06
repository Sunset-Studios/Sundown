// ═══════════════════════════════════════════════════════════════════════════════════════════════
//  ████████╗██╗      █████╗ ███████╗    ██████╗ ██╗   ██╗██╗  ██╗
//  ╚══██╔══╝██║     ██╔══██╗██╔════╝    ██╔══██╗██║   ██║██║  ██║
//     ██║   ██║     ███████║███████╗    ██████╔╝██║   ██║███████║
//     ██║   ██║     ██╔══██║╚════██║    ██╔══██╗╚██╗ ██╔╝██╔══██║
//     ██║   ███████╗██║  ██║███████║    ██████╔╝ ╚████╔╝ ██║  ██║
//     ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝    ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
//  High-Performance GPU-Accelerated Top-Level Acceleration Structure (TLAS)
//  
//  This module implements a state-of-the-art BVH (Bounding Volume Hierarchy) for real-time
//  raytracing and spatial queries. Features include:
//  
//  • GPU-native construction using radix-sorted Morton codes and H-PLOC algorithm
//  • BVH8 nodes for optimal SIMD traversal performance and less depth traversal cost
//  • Decoupled from entity system for maximum flexibility
//  • Dynamic resizing with lazy buffer allocation for memory efficiency
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════

import { Renderer } from "../renderer/renderer.js";
import { Buffer } from "../renderer/buffer.js";

// ╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                    BUFFER IDENTIFIERS                                        ║
// ╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  These constants define the unique names for all GPU buffers used in the BVH construction    ║
// ║  and traversal pipeline. Each buffer serves a specific purpose in the acceleration structure. ║
// ╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

// ─── Core BVH Data Structures ───────────────────────────────────────────────────────────────────
const SCENE_BVH_BUFFER_NAME = "scene_bvh_buffer";                    // Scene bounding box (min/max)
const BVH8_NODES_BUFFER_NAME = "bvh8_nodes_buffer";                  // BVH8 internal/leaf nodes  
const PARENT_IDX_BUFFER_NAME = "bvh_parent_idx";                     // Parent indices for traversal
const BVH8_BUILD_STATE_BUFFER_NAME = "bvh8_build_state";             // Build algorithm state counters

// ─── Morton Code Generation & Sorting ───────────────────────────────────────────────────────────
const MORTON_CODES_BUFFER_NAME = "morton_codes_buffer";              // 3D Morton codes for primitives
const TEMP_MORTON_CODES_BUFFER_NAME = "temp_morton_codes_buffer";     // Temporary sorting workspace
const SORTED_INDICES_BUFFER_NAME = "sorted_indices_buffer";           // Primitive indices post-sort
const TEMP_SORTED_INDICES_BUFFER_NAME = "temp_sorted_indices_buffer"; // Temporary index workspace

// ─── OneSweep Radix Sort Infrastructure ─────────────────────────────────────────────────────────
const ONESWEEP_GLOBAL_HIST_BUFFER_NAME = "onesweep_global_histogram"; // Global histogram for radix sort
const ONESWEEP_PASS_HIST_BUFFER_NAME = "onesweep_pass_histogram";     // Per-pass histograms
const ONESWEEP_TILE_INDICES_BUFFER_NAME = "onesweep_tile_indices";    // Tile indexing for workgroups
const ONESWEEP_ERROR_COUNT_BUFFER_NAME = "onesweep_error_count";      // Error tracking for debugging

// ─── BVH8 Construction Workspace ────────────────────────────────────────────────────────────────
const BVH8_INDEX_PAIRS_BUFFER_NAME = "bvh8_index_pairs";             // Work queue index pairs
const BVH8_PRIM_INDICES_BUFFER_NAME = "bvh8_prim_indices";           // Final primitive index mapping
const BVH8_DEBUG_WATCHDOG_BUFFER_NAME = "bvh8_debug_watchdog";       // Debug counters and validation

// ╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                 ALGORITHM CONFIGURATION                                       ║
// ╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  OneSweep radix sort parameters optimized for modern GPU architectures.                      ║
// ║  These values balance memory bandwidth, compute utilization, and cache efficiency.           ║
// ╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

// ─── OneSweep Radix Sort Configuration ──────────────────────────────────────────────────────────
const RADIX_BITS = 8;                                    // 8 bits per radix pass (256 buckets)
export const RADIX = 1 << RADIX_BITS;                   // 256 - number of buckets per pass
export const RADIX_PASSES = 32 / RADIX_BITS;            // 4 - total passes for 32-bit keys
export const WORKGROUP_SIZE = 256;                      // GPU workgroup size (matches radix)
export const ITEMS_PER_TILE = 16;                       // Items processed per thread
export const TILE_SIZE = WORKGROUP_SIZE * ITEMS_PER_TILE; // Total items per workgroup tile

// ╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                   MEMORY LAYOUT CONSTANTS                                    ║
// ╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  These constants define the precise memory layout for GPU data structures.                   ║
// ║  Sizes are carefully chosen for optimal GPU memory alignment and cache performance.          ║
// ╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

// ─── Data Structure Sizes ───────────────────────────────────────────────────────────────────────
const SCENE_BVH_BYTE_SIZE = 32;                         // Scene bounds: float4 min + float4 max
const BVH8_NODE_BYTE_SIZE = 64;                         // BVH8 node: 4x float4 for bounds + metadata
const DEFAULT_BVH_SIZE = 1024;                          // Initial capacity for BVH nodes

// ─── GPU Buffer Usage Patterns ──────────────────────────────────────────────────────────────────
const storage_usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

// ╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                      BVH CLASS                                               ║
// ╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  High-performance Top-Level Acceleration Structure (TLAS) optimized for GPU raytracing.     ║
// ║                                                                                               ║
// ║  Architecture Features:                                                                       ║
// ║  • Cache-friendly memory layout with structure-of-arrays design                              ║
// ║  • GPU-native construction using Morton codes and OneSweep radix sort                        ║
// ║  • BVH8 nodes for 8-way SIMD traversal on modern GPUs                                        ║
// ║  • Completely decoupled from entity system for maximum flexibility                           ║
// ║  • Dynamic resizing with lazy buffer allocation for memory efficiency                        ║
// ╚═══════════════════════════════════════════════════════════════════════════════════════════════╝

export class BVH {
  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  //                                    INITIALIZATION STATE
  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  
  static is_initialized = false;                        // Tracks whether BVH system is ready
  static bvh_size = DEFAULT_BVH_SIZE;                   // Current maximum number of BVH nodes
  static modified = true;                               // Flag for lazy buffer reconstruction

  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  //                                       SCENE DATA
  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  
  // Scene bounding box stored as [min.xyz, padding, max.xyz, padding]
  // Initialized to inverted bounds for proper expansion during construction
  static scene_bounds = new Float32Array([
    Number.POSITIVE_INFINITY,    // min.x - Will shrink to actual minimum
    Number.POSITIVE_INFINITY,    // min.y - Will shrink to actual minimum  
    Number.POSITIVE_INFINITY,    // min.z - Will shrink to actual minimum
    0,                           // padding for GPU alignment
    Number.NEGATIVE_INFINITY,    // max.x - Will expand to actual maximum
    Number.NEGATIVE_INFINITY,    // max.y - Will expand to actual maximum
    Number.NEGATIVE_INFINITY,    // max.z - Will expand to actual maximum
    0,                           // padding for GPU alignment
  ]);

  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  //                                     GPU BUFFER HANDLES
  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  
  // ─── Core BVH Structure Buffers ─────────────────────────────────────────────────────────────────
  static scene_bounds_buffer = null;                    // Scene AABB for Morton code normalization
  static bvh8_nodes_buffer = null;                      // BVH8 internal and leaf node data
  static bvh_info_buffer = null;                        // Build statistics and metadata
  static parent_idx_buffer = null;                      // Parent node indices for traversal
  
  // ─── Morton Code & Sorting Infrastructure ───────────────────────────────────────────────────────
  static morton_codes_buffer = null;                    // 3D Morton codes for spatial sorting
  static temp_morton_codes_buffer = null;               // Temporary workspace for radix sort
  static sorted_indices_buffer = null;                  // Primitive indices after sorting
  static temp_sorted_indices_buffer = null;             // Temporary workspace for index sorting
  
  // ─── OneSweep Radix Sort Workspace ──────────────────────────────────────────────────────────────
  static onesweep_global_hist_buffer = null;            // Global histogram for prefix sums
  static onesweep_pass_hist_buffer = null;              // Per-workgroup histogram data
  static onesweep_tile_indices_buffer = null;           // Tile management for workgroups
  static onesweep_error_count_buffer = null;            // Error tracking and validation
  
  // ─── BVH8 Construction Workspace ────────────────────────────────────────────────────────────────
  static bvh8_build_state_buffer = null;                // Build algorithm state and counters
  static bvh8_index_pairs_buffer = null;                // Work queue for parallel construction
  static bvh8_prim_indices_buffer = null;               // Final primitive index remapping
  static bvh8_debug_watchdog_buffer = null;             // Debug counters with CPU readback

  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  //                                   INITIALIZATION METHODS
  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  
  /**
   * Initialize the BVH system and create essential GPU buffers.
   * 
   * This method sets up the core infrastructure required for BVH construction and traversal.
   * It creates the scene bounds buffer which serves as the root of our spatial hierarchy.
   * 
   * The initialization is designed to be idempotent - multiple calls are safe and will
   * not recreate already existing resources.
   * 
   * @returns {void}
   */
  static initialize() {
    // Early exit if already initialized - avoid redundant work
    if (this.is_initialized) return;

    // Create the scene bounds buffer which stores the world-space AABB
    // This buffer is used for Morton code normalization during BVH construction
    this.scene_bounds_buffer = Buffer.create({
      name: SCENE_BVH_BUFFER_NAME,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      size: SCENE_BVH_BYTE_SIZE,        // float4 min + float4 max = 32 bytes
      force: true,                      // Force creation even if buffer exists
    });

    // Initialize all other buffers with default sizing
    // This creates the full GPU memory infrastructure needed for BVH operations
    this.rebuild_buffers();

    // Mark as initialized to prevent redundant setup
    this.is_initialized = true;
  }

  /**
   * Dynamically resize the BVH to accommodate more primitives.
   * 
   * This method implements a growth-only resizing strategy for optimal memory utilization.
   * When the current capacity is insufficient, all GPU buffers are marked for lazy
   * reconstruction on the next rebuild_buffers() call.
   * 
   * The resize operation is designed to be conservative - we only grow when absolutely
   * necessary to avoid frequent memory reallocations which can cause GPU stalls.
   * 
   * @param {number} new_size - The new minimum capacity required for BVH nodes
   * @returns {void}
   */
  static resize(new_size) {
    // Only resize if we actually need more capacity
    if (new_size <= this.bvh_size) return;
    
    // Update the target size and mark buffers as needing reconstruction
    this.bvh_size = new_size;
    this.modified = true;               // Trigger lazy buffer reallocation
  }

  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  //                                   BUFFER MANAGEMENT
  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  
  /**
   * Rebuild all GPU buffers to match current size requirements.
   * 
   * This method implements lazy buffer allocation - buffers are only recreated when:
   * • The BVH has been marked as modified (size change, etc.)
   * • A buffer doesn't exist yet
   * • A buffer is too small for current requirements
   * 
   * The buffer creation process is optimized to minimize GPU memory fragmentation
   * by allocating all related buffers in sequence. Each buffer serves a specific
   * purpose in the BVH construction and traversal pipeline.
   * 
   * Memory Layout Strategy:
   * • All buffers use storage usage for compute shader access
   * • Size calculations account for GPU alignment requirements
   * • Temporary buffers match primary buffer sizes for ping-pong operations
   * 
   * @returns {void}
   */
  static rebuild_buffers() {
    // Early exit if no changes have occurred - avoid unnecessary work
    if (!this.modified) return;
    
    // Calculate memory requirements based on current BVH capacity
    // Each primitive needs 4 bytes (u32) for indices and Morton codes
    const required_primitive_size = this.bvh_size * 4;

    // ─── Morton Code Buffers ────────────────────────────────────────────────────────────────────
    // Morton codes provide a space-filling curve mapping 3D positions to 1D keys
    // This enables efficient spatial sorting while preserving locality
    
    if (
      !this.morton_codes_buffer ||
      this.morton_codes_buffer.config.size < required_primitive_size
    ) {
      this.morton_codes_buffer = Buffer.create({
        name: MORTON_CODES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,     // One u32 Morton code per primitive
        force: true,
      });

      // Notify renderer that bind groups need updating due to buffer changes
      Renderer.get().mark_bind_groups_dirty(true);
    }

    // Temporary workspace for Morton code radix sort ping-pong operations
    if (
      !this.temp_morton_codes_buffer ||
      this.temp_morton_codes_buffer.config.size < required_primitive_size
    ) {
      this.temp_morton_codes_buffer = Buffer.create({
        name: TEMP_MORTON_CODES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,     // Must match primary buffer for swapping
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    // ─── Primitive Index Buffers ───────────────────────────────────────────────────────────────
    // These buffers track the mapping between sorted Morton codes and original primitive indices
    
    if (
      !this.sorted_indices_buffer ||
      this.sorted_indices_buffer.config.size < required_primitive_size
    ) {
      this.sorted_indices_buffer = Buffer.create({
        name: SORTED_INDICES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,     // One u32 index per primitive
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    // Temporary workspace for index sorting - pairs with Morton code sorting
    if (
      !this.temp_sorted_indices_buffer ||
      this.temp_sorted_indices_buffer.config.size < required_primitive_size
    ) {
      this.temp_sorted_indices_buffer = Buffer.create({
        name: TEMP_SORTED_INDICES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,     // Must match primary buffer for swapping
        force: true,
      });
    }

    // ─── OneSweep Radix Sort Infrastructure ────────────────────────────────────────────────────
    // OneSweep is a high-performance GPU radix sort that processes data in a single pass
    // These buffers provide the workspace needed for histogram computation and prefix sums
    
    // Calculate workgroup requirements for current data size
    const max_thread_blocks = Math.max(1, Math.ceil(this.bvh_size / TILE_SIZE));
    
    // Pass histogram: Each workgroup needs RADIX buckets for each radix pass
    const pass_hist_count = max_thread_blocks * RADIX * RADIX_PASSES;
    const pass_hist_size_bytes = pass_hist_count * 4;  // u32 counters
    
    if (
      !this.onesweep_pass_hist_buffer ||
      this.onesweep_pass_hist_buffer.config.size < pass_hist_size_bytes
    ) {
      this.onesweep_pass_hist_buffer = Buffer.create({
        name: ONESWEEP_PASS_HIST_BUFFER_NAME,
        usage: storage_usage,
        size: pass_hist_size_bytes,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    // Global histogram: Aggregated bucket counts across all workgroups
    const global_hist_count = RADIX * RADIX_PASSES;    // 256 buckets × 4 passes = 1024 entries
    const global_hist_size_bytes = global_hist_count * 4;  // u32 counters
    
    if (
      !this.onesweep_global_hist_buffer ||
      this.onesweep_global_hist_buffer.config.size < global_hist_size_bytes
    ) {
      this.onesweep_global_hist_buffer = Buffer.create({
        name: ONESWEEP_GLOBAL_HIST_BUFFER_NAME,
        usage: storage_usage,
        size: global_hist_size_bytes,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    // Tile indices: Track which tiles are active for each radix pass
    const tile_indices_size_bytes = RADIX_PASSES * 4;  // One u32 per radix pass
    
    if (
      !this.onesweep_tile_indices_buffer ||
      this.onesweep_tile_indices_buffer.config.size < tile_indices_size_bytes
    ) {
      this.onesweep_tile_indices_buffer = Buffer.create({
        name: ONESWEEP_TILE_INDICES_BUFFER_NAME,
        usage: storage_usage,
        size: tile_indices_size_bytes,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    // Error counting: Debug validation for sort algorithm correctness
    const error_count_size_bytes = 4;  // Single u32 error counter
    
    if (
      !this.onesweep_error_count_buffer ||
      this.onesweep_error_count_buffer.config.size < error_count_size_bytes
    ) {
      this.onesweep_error_count_buffer = Buffer.create({
        name: ONESWEEP_ERROR_COUNT_BUFFER_NAME,
        usage: storage_usage,
        size: error_count_size_bytes,
        force: true,
      });
    }

    // ─── BVH8 Node Structure ────────────────────────────────────────────────────────────────────
    // BVH8 uses 8-way branching for optimal SIMD traversal on modern GPUs
    // Each node stores bounding boxes for up to 8 children plus metadata
    
    const required_bvh8_size = this.bvh_size * BVH8_NODE_BYTE_SIZE;  // 64 bytes per node

    if (!this.bvh8_nodes_buffer || this.bvh8_nodes_buffer.config.size < required_bvh8_size) {
      this.bvh8_nodes_buffer = Buffer.create({
        name: BVH8_NODES_BUFFER_NAME,
        usage: storage_usage,
        size: required_bvh8_size,
        force: true,
      });

      Renderer.get().mark_bind_groups_dirty(true);
    }

    // ─── BVH Metadata and Statistics ───────────────────────────────────────────────────────────
    // This buffer stores build statistics and parameters needed by traversal shaders
    
    // BVH info buffer (u32 values: leaf_count, node_count, prim_count, padding)
    if (!this.bvh_info_buffer) {
      this.bvh_info_buffer = Buffer.create({
        name: "bvh_info",
        usage: storage_usage | GPUBufferUsage.UNIFORM,
        size: 16,                         // 4 u32 values with GPU alignment
        force: true,
      });
    }

    // Parent index mapping for hierarchical traversal and updates
    if (!this.parent_idx_buffer || this.parent_idx_buffer.config.size < required_primitive_size) {
      this.parent_idx_buffer = Buffer.create({
        name: PARENT_IDX_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,     // One u32 parent index per node
        force: true,
      });
    }

    // ─── BVH8 Construction State Management ─────────────────────────────────────────────────────
    // These buffers manage the parallel BVH8 construction algorithm state
    
    // Build state: Atomic counters for work distribution and progress tracking
    // Layout: [work_counter, node_counter, leaf_counter, work_alloc_counter, prim_count]
    if (!this.bvh8_build_state_buffer) {
      this.bvh8_build_state_buffer = Buffer.create({
        name: BVH8_BUILD_STATE_BUFFER_NAME,
        usage: storage_usage,
        size: 20,                         // 5 u32 atomic counters
        force: true,
      });
    }

    // Index pairs: Work queue entries for parallel BVH construction
    // Each entry contains start/end indices for a work unit
    const index_pairs_size = this.bvh_size * 8;    // u64 pairs (start, end)
    
    if (!this.bvh8_index_pairs_buffer || this.bvh8_index_pairs_buffer.config.size < index_pairs_size) {
      this.bvh8_index_pairs_buffer = Buffer.create({
        name: BVH8_INDEX_PAIRS_BUFFER_NAME,
        usage: storage_usage,
        size: index_pairs_size,
        force: true,
      });
    }

    // Primitive indices: Final mapping from BVH leaves to original primitives
    if (!this.bvh8_prim_indices_buffer || this.bvh8_prim_indices_buffer.config.size < required_primitive_size) {
      this.bvh8_prim_indices_buffer = Buffer.create({
        name: BVH8_PRIM_INDICES_BUFFER_NAME,
        usage: storage_usage,
        size: required_primitive_size,     // One u32 index per primitive
        force: true,
      });
    }

    // ─── Debug and Profiling Infrastructure ────────────────────────────────────────────────────
    // Debug watchdog: Performance counters and validation with CPU readback capability
    // Layout: [iteration_count, max_depth_reached, error_flags, padding]
    if (!this.bvh8_debug_watchdog_buffer) {
      this.bvh8_debug_watchdog_buffer = Buffer.create({
        name: BVH8_DEBUG_WATCHDOG_BUFFER_NAME,
        usage: storage_usage,
        size: 16,                         // 4 u32 counters with alignment
        force: true,
        cpu_readback: true,               // Enable CPU access for debugging
        raw_data: new Uint32Array(4),     // Pre-allocated CPU staging
      });
    }

    // Mark buffer reconstruction as complete
    this.modified = false;
  }

  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  //                                    UTILITY METHODS
  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  
  /**
   * Reset scene bounds to inverted infinity for fresh AABB computation.
   * 
   * This method prepares the scene bounds for a new BVH construction cycle.
   * By setting min bounds to positive infinity and max bounds to negative infinity,
   * we ensure that the first primitive will correctly initialize the bounds and
   * subsequent primitives will properly expand them.
   * 
   * The bounds are immediately uploaded to GPU memory to ensure consistency
   * between CPU and GPU representations during construction.
   * 
   * @returns {void}
   */
  static clear_scene_bounds() {
    // Reset minimum bounds to positive infinity (will shrink to actual minimum)
    this.scene_bounds[0] = Number.POSITIVE_INFINITY;  // min.x
    this.scene_bounds[1] = Number.POSITIVE_INFINITY;  // min.y
    this.scene_bounds[2] = Number.POSITIVE_INFINITY;  // min.z
    this.scene_bounds[3] = 0;                         // padding
    
    // Reset maximum bounds to negative infinity (will expand to actual maximum)
    this.scene_bounds[4] = Number.NEGATIVE_INFINITY;  // max.x
    this.scene_bounds[5] = Number.NEGATIVE_INFINITY;  // max.y
    this.scene_bounds[6] = Number.NEGATIVE_INFINITY;  // max.z
    this.scene_bounds[7] = 0;                         // padding
    
    // Upload the reset bounds to GPU memory immediately
    this.scene_bounds_buffer.write_raw(this.scene_bounds);
  }

  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  //                                  GPU INTERFACE METHODS
  // ═══════════════════════════════════════════════════════════════════════════════════════════════
  
  // Private buffer collection for GPU binding - populated dynamically by to_gpu_data()
  // This object serves as a stable interface for the renderer's bind group system
  static #data_buffers = {
    // Core BVH structure buffers
    scene_bounds_buffer: null,              // Scene AABB for Morton normalization
    bvh8_nodes_buffer: null,                // BVH8 internal and leaf nodes
    bvh_info_buffer: null,                  // Build statistics and metadata
    parent_idx_buffer: null,                // Parent node indices
    
    // Morton code and sorting infrastructure
    morton_codes_buffer: null,              // 3D Morton codes for primitives
    temp_morton_codes_buffer: null,         // Temporary sorting workspace
    sorted_indices_buffer: null,            // Primitive indices after sorting
    temp_sorted_indices_buffer: null,       // Temporary index workspace
    
    // OneSweep radix sort workspace
    onesweep_global_hist_buffer: null,      // Global histogram for prefix sums
    onesweep_pass_hist_buffer: null,        // Per-workgroup histogram data
    onesweep_tile_indices_buffer: null,     // Tile management
    onesweep_error_count_buffer: null,      // Error tracking
  };

  /**
   * Prepare and return GPU buffer collection for shader binding.
   * 
   * This method serves as the primary interface between the BVH system and the
   * GPU renderer. It ensures all buffers are up-to-date and properly sized before
   * returning a stable object reference for bind group creation.
   * 
   * The returned object contains all buffers needed for:
   * • BVH construction (Morton codes, sorting workspace, build state)
   * • BVH traversal (nodes, parent indices, primitive mappings)
   * • Debug and profiling (statistics, error tracking)
   * 
   * Buffer Management:
   * • Automatically triggers buffer rebuilding if size requirements changed
   * • Maintains stable object references for efficient bind group caching
   * • Includes all temporary workspace buffers for multi-pass algorithms
   * 
   * @returns {Object} Complete collection of GPU buffers ready for binding
   */
  static to_gpu_data() {
    // Ensure all buffers are properly sized and allocated
    this.rebuild_buffers();

    // ─── Populate Core BVH Structure Buffers ───────────────────────────────────────────────────
    this.#data_buffers.scene_bounds_buffer = this.scene_bounds_buffer;
    this.#data_buffers.bvh8_nodes_buffer = this.bvh8_nodes_buffer;
    this.#data_buffers.bvh_info_buffer = this.bvh_info_buffer;
    this.#data_buffers.parent_idx_buffer = this.parent_idx_buffer;
    
    // ─── Populate Morton Code and Sorting Infrastructure ───────────────────────────────────────
    this.#data_buffers.morton_codes_buffer = this.morton_codes_buffer;
    this.#data_buffers.temp_morton_codes_buffer = this.temp_morton_codes_buffer;
    this.#data_buffers.sorted_indices_buffer = this.sorted_indices_buffer;
    this.#data_buffers.temp_sorted_indices_buffer = this.temp_sorted_indices_buffer;
    
    // ─── Populate OneSweep Radix Sort Workspace ────────────────────────────────────────────────
    this.#data_buffers.onesweep_global_hist_buffer = this.onesweep_global_hist_buffer;
    this.#data_buffers.onesweep_pass_hist_buffer = this.onesweep_pass_hist_buffer;
    this.#data_buffers.onesweep_tile_indices_buffer = this.onesweep_tile_indices_buffer;
    this.#data_buffers.onesweep_error_count_buffer = this.onesweep_error_count_buffer;
    
    // ─── Populate BVH8 Construction Workspace ──────────────────────────────────────────────────
    this.#data_buffers.bvh8_build_state_buffer = this.bvh8_build_state_buffer;
    this.#data_buffers.bvh8_index_pairs_buffer = this.bvh8_index_pairs_buffer;
    this.#data_buffers.bvh8_prim_indices_buffer = this.bvh8_prim_indices_buffer;
    this.#data_buffers.bvh8_debug_watchdog_buffer = this.bvh8_debug_watchdog_buffer;

    return this.#data_buffers;
  }
}

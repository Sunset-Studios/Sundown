import { SimulationLayer } from "../core/simulation_layer.js";
import { ComputeTaskQueue } from "../renderer/compute_task_queue.js";
import { Buffer } from "../renderer/buffer.js";
import { BVH } from "./bvh.js";
import { MeshBLAS } from "./mesh_blas.js";

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// ███╗   ███╗███████╗███████╗██╗  ██╗██████╗ ██╗      █████╗ ███████╗    ██████╗ ██████╗  ██████╗
// ████╗ ████║██╔════╝██╔════╝██║  ██║██╔══██╗██║     ██╔══██╗██╔════╝    ██╔══██╗██╔══██╗██╔═══██╗
// ██╔████╔██║█████╗  ███████╗███████║██████╔╝██║     ███████║███████╗    ██████╔╝██████╔╝██║   ██║
// ██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║██╔══██╗██║     ██╔══██║╚════██║    ██╔═══╝ ██╔══██╗██║   ██║
// ██║ ╚═╝ ██║███████╗███████║██║  ██║██████╔╝███████╗██║  ██║███████║    ██║     ██║  ██║╚██████╔╝
// ╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// MeshBLASProcessor - GPU-Accelerated BVH Construction Pipeline
//
// This system orchestrates the complete GPU compute pipeline for building per-mesh Bottom-Level
// Acceleration Structures (BLAS), implementing state-of-the-art algorithms for high-performance
// raytracing acceleration structure construction.
//
// 🔬 ALGORITHMIC FOUNDATION:
//     • Morton Code Computation: Z-order curve spatial coherence for efficient tree construction
//     • OneSweep Radix Sort: Ultra-high-performance GPU sorting for Morton codes
//     • H-PLOC (Hierarchical Parallel Locally-Ordered Clustering): Wave-optimised Modern GPU bottom-up BVH builder
//     • BVH4 Conversion: Wave-optimised quaternary tree generation from binary trees
//
// 🚀 COMPUTE PIPELINE ARCHITECTURE:
//     Phase 1: Leaf Bounds Generation → Compute triangle bounding boxes
//     Phase 2: Morton Code Generation → Z-order spatial indexing
//     Phase 3: OneSweep Radix Sort → Ultra-fast primitive ordering
//     Phase 4: H-PLOC BVH2 Build → Wave-optimised parallel binary tree construction
//     Phase 5: BVH4 Conversion → Wave-optimised quaternary tree generation
//
// 💡 PERFORMANCE OPTIMIZATIONS:
//     • Minimal CPU-GPU Synchronization: Fully GPU-driven compute pipeline
//     • Buffer Reuse Strategy: Shared scratch space across all mesh builds
//     • Work Distribution: Optimal GPU workgroup sizing for varying mesh complexities
//     • Memory Bandwidth Optimization: Coalesced access patterns throughout pipeline
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                            🏷️  COMPUTE TASK IDENTIFIERS                                    │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const hploc_compute_morton_codes_task_name = "hploc_compute_morton_codes";
const hploc_init_leaf_clusters_task_name = "hploc_init_leaf_clusters";
const hploc_build_bvh2_task_name = "hploc_build_bvh2";
const hploc_convert_parallel_single_pass_task_name = "hploc_convert_parallel_single_pass";

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                              📄 COMPUTE SHADER RESOURCES                                     │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const bvh_sorting_wgsl_path = "acceleration/bvh_sorting.wgsl"; // OneSweep radix sort implementation
const bvh_morton_wgsl_path = "acceleration/bvh_morton.wgsl"; // Z-order Morton code generation
const bvh_as_init_wgsl_path = "acceleration/bvh_as_init.wgsl"; // BVH initialization and leaf setup
const bvh_processing_wgsl_path = "acceleration/bvh_processing.wgsl"; // H-PLOC BVH2 construction
const bvh4_processing_wgsl_path = "acceleration/bvh4_processing.wgsl"; // BVH4 conversion and optimization

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                            🔧 COMPUTE SHADER ENTRY POINTS                                    │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const compute_morton_codes_cs_entry_point = "compute_morton_codes"; // Morton code generation
const onesweep_init_cs_entry_point = "onesweep_init"; // OneSweep initialization
const onesweep_histogram_cs_entry_point = "onesweep_global_histogram"; // Global histogram phase
const onesweep_scan_cs_entry_point = "onesweep_scan"; // Prefix scan phase
const onesweep_digit_binning_cs_entry_point = "onesweep_digit_binning"; // Digit binning phase
const initialize_leaf_clusters_cs_entry_point = "initialize_leaf_clusters"; // Leaf cluster initialization
const build_bvh2_hploc_cs_entry_point = "build_bvh2_hploc"; // H-PLOC BVH2 construction
const convert_bvh2_to_bvh4_cs_entry_point = "convert_bvh2_to_bvh4"; // BVH4 conversion

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                          ⚙️  PIPELINE CONFIGURATION CONSTANTS                              │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const RADIX_PASSES = 4; // OneSweep passes: 4 × 8-bit = 32-bit Morton codes
const WORKGROUP_SIZE = 256; // GPU workgroup size (hardware optimal)
const TILE_SIZE = WORKGROUP_SIZE * 16; // 4K primitives per tile (OneSweep optimization)

/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
 * ║                                🏭 MeshBLASProcessor CLASS                                    ║
 * ║                     GPU Compute Pipeline Orchestrator for BVH Construction                  ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
 *
 * A sophisticated compute pipeline controller that coordinates the complete GPU-based construction
 * of per-mesh acceleration structures. Implements modern algorithms including H-PLOC hierarchical
 * cluster construction and OneSweep radix sorting for optimal GPU utilization.
 *
 * ┌────────────────────────── 🎯 PRIMARY RESPONSIBILITIES ──────────────────────────┐
 * │                                                                                 │
 * │  • Pipeline Orchestration: Coordinates 5-phase GPU compute pipeline             │
 * │  • Resource Management: Manages uniform buffers and binding arrays              │
 * │  • Work Distribution: Optimizes GPU workgroup dispatch for varying complexities │
 * │  • Quality Assurance: Integrates debug infrastructure and error detection       │
 * │                                                                                 │
 * └──────────────────────────────────────────────────────────────────────────────────┘
 *
 * ┌────────────────────────── 🔄 COMPUTE PIPELINE OVERVIEW ─────────────────────────┐
 * │                                                                                   │
 * │  Phase 1: Leaf Bounds    → Triangle AABB computation                             │
 * │  Phase 2: Morton Codes   → Z-order spatial indexing (3D → 1D mapping)           │
 * │  Phase 3: OneSweep Sort  → Ultra-fast GPU radix sort (4 × 8-bit passes)         │
 * │  Phase 4: H-PLOC BVH2    → Parallel binary tree construction                     │
 * │  Phase 5: BVH4 Convert   → Hardware-optimized quaternary tree generation        │
 * │                                                                                   │
 * └───────────────────────────────────────────────────────────────────────────────────┘
 */
export class MeshBLASProcessor extends SimulationLayer {
  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                              🏗️  SYSTEM INITIALIZATION                                     ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                         🚀 PIPELINE SYSTEM INITIALIZATION                               │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Initializes the complete GPU compute pipeline infrastructure, including uniform buffer
   * management, CPU-side data structures, and pre-allocated binding arrays for optimal
   * performance during the build process.
   *
   * 🛠️ INITIALIZATION COMPONENTS:
   *    • MeshBLAS System: Underlying memory management and buffer allocation
   *    • Radix Sort Uniforms: Per-pass parameter buffers for OneSweep algorithm
   *    • CPU Data Structures: Pre-allocated arrays for parameter updates
   *    • Binding Array Pool: Pre-sized arrays to eliminate allocation overhead
   */
  constructor() {
    super();

    // Initialize the underlying memory management system
    MeshBLAS.initialize();

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                       📊 ONESWEEP RADIX SORT INFRASTRUCTURE                             │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    // Create dedicated uniform buffers for each radix sort pass
    // OneSweep requires 4 passes for 32-bit Morton codes (4 × 8-bit digits)
    this.sort_uniforms_buffers = [];
    for (let i = 0; i < RADIX_PASSES; i++) {
      this.sort_uniforms_buffers[i] = Buffer.create({
        name: `mesh_blas_radix_uniforms_pass_${i}`,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        size: 16, // RadixSortParams: [primitive_count, bit_shift, thread_blocks, padding]
      });
    }

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                          💾 CPU-SIDE PARAMETER MANAGEMENT                               │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    this.radix_uniforms = new Uint32Array(4); // OneSweep parameters: [count, shift, blocks, pad]
    this.bvh_build_data = new Uint32Array(6); // BVH build state: [leaf_count, bvh2_count, prim_count, bvh2_base, bvh4_base, is_blas]

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                     🔗 PRE-ALLOCATED BINDING ARRAY INFRASTRUCTURE                       │
    // │                                                                                          │
    // │  These arrays are pre-sized to eliminate allocation overhead during the critical build  │
    // │  path. Each pipeline phase has dedicated input/output binding arrays for optimal        │
    // │  performance and clear separation of concerns.                                           │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    // Morton Code Generation Phase Bindings
    this.morton_code_inputs = new Array(5); // [bounds, codes, indices, scene_aabb, build_info]
    this.morton_code_outputs = new Array(2); // [codes, indices]

    // OneSweep Radix Sort Phase Bindings
    this.radix_sort_inputs = new Array(8); // [src_codes, dst_codes, src_indices, dst_indices, global_hist, pass_hist, tile_indices, uniforms]
    this.radix_sort_outputs = new Array(3); // [global_hist, pass_hist, tile_indices]

    // H-PLOC BVH2 Construction Phase Bindings
    this.bvh2_inputs = new Array(6); // [bounds, indices, build_info, codes, parent_idx, index_pairs]
    this.bvh2_outputs = new Array(2); // [bounds, bounds] (reused for efficiency)

    // BVH4 Conversion Phase Bindings
    this.bvh4_inputs = new Array(7); // [bvh2_bounds, bvh4_nodes, build_state, index_pairs, prim_indices, build_info, debug_watchdog]
    this.bvh4_outputs = new Array(5); // [bvh4_nodes, build_state, index_pairs, prim_indices, debug_watchdog]

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                          📋 BUILD QUEUE MANAGEMENT SYSTEM                               │
    // │                                                                                          │
    // │  Queue system to process only one mesh BVH build per frame, preventing frame rate      │
    // │  drops from multiple simultaneous builds while maintaining steady progress through      │
    // │  all dirty meshes.                                                                      │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘
    this.build_queue = []; // Queue of pending mesh builds: [{mesh_id, leaf_count, mesh_info_buffer, mesh_selector_buffer}]
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                                🔄 PUBLIC PIPELINE METHODS                                 ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                            🕰️ SIMULATION LAYER UPDATE                                   │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Called each frame to manage mesh BVH construction queue. This method maintains optimal
   * frame rates by queuing dirty meshes and processing only one mesh build per frame,
   * ensuring consistent performance while making steady progress through all builds.
   *
   * @param {number} delta_time - Frame delta time (inherited from SimulationLayer)
   */
  update(delta_time) {
    super.update(delta_time);
    
    // Queue any newly dirty meshes for processing
    this.queue_dirty_meshes();
    
    // Process one queued build per frame to maintain performance
    this.process_queued_build();
  }

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                         🏭 CORE COMPUTE PIPELINE ORCHESTRATION                          │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Orchestrates the complete 5-phase GPU compute pipeline for building a mesh's acceleration
   * structure. This method sequences Morton code generation, OneSweep radix sorting, H-PLOC
   * BVH2 construction, and final BVH4 conversion into an efficient asynchronous pipeline.
   *
   * 🔬 PIPELINE PHASES:
   *    Phase 1: Morton Code Generation   → Spatial indexing via Z-order curve mapping
   *    Phase 2: OneSweep Radix Sort     → 4-pass ultra-fast GPU sorting (8-bit digits)
   *    Phase 3: H-PLOC BVH2 Build       → Parallel binary tree construction
   *    Phase 4: BVH4 Conversion         → Wave-optimised quaternary tree generation
   *
   * @param {number} mesh_id - Unique mesh identifier for resource allocation
   * @param {number} primitive_count - Number of triangles in the mesh
   * @param {Buffer} mesh_info_buffer - Per-mesh build parameters and state
   */
  build(mesh_id, primitive_count, mesh_info_buffer) {
    if (!primitive_count) return;

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                           ⚙️ PIPELINE INITIALIZATION                                   │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    const morton_workgroups = Math.ceil(primitive_count / WORKGROUP_SIZE);
    const mesh_meta = MeshBLAS.get_mesh_meta(mesh_id);
    if (!mesh_meta) return;

    // Initialize per-mesh build state for GPU compute shaders
    // BVHData structure: [leaf_count, bvh2_count, primitive_count, bvh2_base_index, bvh4_base_index, is_blas]
    this.bvh_build_data[0] = 0; // leaf_count (updated by shaders)
    this.bvh_build_data[1] = 0; // bvh2_count (updated by shaders)
    this.bvh_build_data[2] = primitive_count; // primitive_count (input parameter)
    this.bvh_build_data[3] = mesh_meta.bvh2_base_node_index >>> 0; // bvh2_base_index (allocation offset)
    this.bvh_build_data[4] = mesh_meta.bvh4_base_node_index >>> 0; // bvh4_base_index (allocation offset)
    this.bvh_build_data[5] = 1; // is_blas - 1 for BLAS (store triangle IDs directly)
    mesh_info_buffer.write(this.bvh_build_data);

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                       📍 PHASE 1: MORTON CODE GENERATION                                │
    // │                                                                                          │
    // │  Computes 32-bit Morton codes for each triangle using Z-order curve mapping. This       │
    // │  converts 3D spatial coordinates into a 1D index that preserves spatial locality,      │
    // │  enabling efficient clustering during BVH construction.                                 │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    
    // Setup Morton code generation compute dispatch
    const tlas_gpu_data = BVH.to_gpu_data();
    const blas_gpu_data = MeshBLAS.to_gpu_data();

    // Acquire shared scratch buffers from the MeshBLAS system
    const morton_codes_buffer = blas_gpu_data.morton_codes_buffer;
    const sorted_indices_buffer = blas_gpu_data.sorted_indices_buffer;
    const bvh2_nodes_buffer = blas_gpu_data.bvh2_nodes_buffer;

    // Input Bindings: [bvh2_bounds, morton_codes, sorted_indices, scene_aabb, mesh_build_info]
    this.morton_code_inputs[0] = bvh2_nodes_buffer; // Triangle bounding boxes (input)
    this.morton_code_inputs[1] = morton_codes_buffer; // Morton codes output buffer
    this.morton_code_inputs[2] = sorted_indices_buffer; // Triangle indices output buffer
    this.morton_code_inputs[3] = tlas_gpu_data.scene_bounds_buffer; // Global scene AABB for normalization
    this.morton_code_inputs[4] = mesh_info_buffer; // Per-mesh build parameters

    // Output Bindings: [morton_codes, sorted_indices]
    this.morton_code_outputs[0] = morton_codes_buffer;
    this.morton_code_outputs[1] = sorted_indices_buffer;

    // Dispatch Morton code generation compute shader
    ComputeTaskQueue.new_task(
      `${hploc_compute_morton_codes_task_name}_${mesh_id}`,
      bvh_morton_wgsl_path,
      this.morton_code_inputs,
      this.morton_code_outputs,
      morton_workgroups, // One thread per triangle for Morton code computation
      1,
      1,
      compute_morton_codes_cs_entry_point
    );

    // ┌────────────────────────────────────────────────────────────────────────────────────────┐
    // │                        📊 PHASE 2: ONESWEEP RADIX SORT SETUP                          │
    // │                                                                                        │
    // │  OneSweep is a state-of-the-art GPU sorting algorithm that achieves optimal work       │
    // │  distribution and throughput through usual global histograms and prefix scans while    |
    // │  requiring minimal GPU memory overhead. It sorts 32-bit Morton codes in fewer passes   |
    // │  than traditional radix sorts while requiring less overall global memory operations.   │
    // └────────────────────────────────────────────────────────────────────────────────────────┘

    // Calculate optimal work distribution for OneSweep algorithm
    const thread_blocks = Math.max(1, Math.ceil(primitive_count / TILE_SIZE));

    // Initialize radix sort parameters: [primitive_count, bit_shift, thread_blocks, padding]
    this.radix_uniforms[0] = primitive_count; // Total number of elements to sort
    this.radix_uniforms[1] = 0; // Bit shift (updated per pass: 0, 8, 16, 24)
    this.radix_uniforms[2] = thread_blocks; // Number of thread blocks for work distribution
    this.radix_uniforms[3] = 0; // Padding for alignment
    this.sort_uniforms_buffers[0].write(this.radix_uniforms);

    // Acquire OneSweep algorithm infrastructure buffers
    const temp_morton_codes_buffer = blas_gpu_data.temp_morton_codes_buffer; // Ping-pong buffer for codes
    const temp_sorted_indices_buffer = blas_gpu_data.temp_sorted_indices_buffer; // Ping-pong buffer for indices
    const onesweep_global_hist_buffer = blas_gpu_data.onesweep_global_hist_buffer; // Global histogram (256×4 entries)
    const onesweep_pass_hist_buffer = blas_gpu_data.onesweep_pass_hist_buffer; // Per-pass histogram data
    const onesweep_tile_indices_buffer = blas_gpu_data.onesweep_tile_indices_buffer; // Tile boundary management

    // Setup comprehensive input bindings for OneSweep phases
    this.radix_sort_inputs[0] = morton_codes_buffer; // Source Morton codes
    this.radix_sort_inputs[1] = temp_morton_codes_buffer; // Destination Morton codes
    this.radix_sort_inputs[2] = sorted_indices_buffer; // Source triangle indices
    this.radix_sort_inputs[3] = temp_sorted_indices_buffer; // Destination triangle indices
    this.radix_sort_inputs[4] = onesweep_global_hist_buffer; // Global digit histogram
    this.radix_sort_inputs[5] = onesweep_pass_hist_buffer; // Per-pass working histogram
    this.radix_sort_inputs[6] = onesweep_tile_indices_buffer; // Tile management data
    this.radix_sort_inputs[7] = this.sort_uniforms_buffers[0]; // Radix sort parameters

    // Setup output bindings for histogram and infrastructure updates
    this.radix_sort_outputs[0] = onesweep_global_hist_buffer;
    this.radix_sort_outputs[1] = onesweep_pass_hist_buffer;
    this.radix_sort_outputs[2] = onesweep_tile_indices_buffer;

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                         🏁 ONESWEEP INITIALIZATION PHASE                                │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘
    ComputeTaskQueue.new_task(
      `mesh_${mesh_id}_onesweep_init`,
      bvh_sorting_wgsl_path,
      this.radix_sort_inputs,
      this.radix_sort_outputs,
      256, // Fixed workgroup size for initialization
      1,
      1,
      onesweep_init_cs_entry_point
    );

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                         📈 GLOBAL HISTOGRAM COMPUTATION                                 │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘
    ComputeTaskQueue.new_task(
      `mesh_${mesh_id}_onesweep_histogram`,
      bvh_sorting_wgsl_path,
      this.radix_sort_inputs,
      [onesweep_global_hist_buffer],
      thread_blocks, // Parallel histogram across all thread blocks
      1,
      1,
      onesweep_histogram_cs_entry_point
    );

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                           🔍 PREFIX SCAN COMPUTATION                                     │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘
    ComputeTaskQueue.new_task(
      `mesh_${mesh_id}_onesweep_scan`,
      bvh_sorting_wgsl_path,
      this.radix_sort_inputs,
      [onesweep_pass_hist_buffer],
      RADIX_PASSES, // One workgroup per radix pass (4 total)
      1,
      1,
      onesweep_scan_cs_entry_point
    );

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                        🔀 ONESWEEP DIGIT BINNING PASSES                                │
    // │                                                                                          │
    // │  Execute 4 digit binning passes to sort 32-bit Morton codes by 8-bit radix digits.     │
    // │  Uses ping-pong buffers to alternate between source and destination on each pass.      │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    const bit_shifts = [0, 8, 16, 24]; // LSB to MSB: process 8-bit digits in sequence

    for (let pass = 0; pass < bit_shifts.length; pass++) {
      const bit_shift = bit_shifts[pass];
      const src_is_morton = pass % 2 === 0; // Ping-pong between buffers

      // Determine source and destination buffers for this pass
      const src_morton_buffer = src_is_morton ? morton_codes_buffer : temp_morton_codes_buffer;
      const dst_morton_buffer = src_is_morton ? temp_morton_codes_buffer : morton_codes_buffer;
      const src_indices_buffer = src_is_morton ? sorted_indices_buffer : temp_sorted_indices_buffer;
      const dst_indices_buffer = src_is_morton ? temp_sorted_indices_buffer : sorted_indices_buffer;

      // Update radix sort parameters for current digit position
      this.radix_uniforms[1] = bit_shift; // Set bit shift for current 8-bit digit
      this.sort_uniforms_buffers[pass].write(this.radix_uniforms);

      // Configure input bindings for digit binning pass
      const pass_inputs = [
        src_morton_buffer, // Source Morton codes
        dst_morton_buffer, // Destination Morton codes
        src_indices_buffer, // Source triangle indices
        dst_indices_buffer, // Destination triangle indices
        onesweep_global_hist_buffer, // Global histogram data
        onesweep_pass_hist_buffer, // Pass-specific histogram
        onesweep_tile_indices_buffer, // Tile boundary indices
        this.sort_uniforms_buffers[pass], // Per-pass uniform parameters
      ];

      // Configure output bindings (sorted data + updated histogram)
      const pass_outputs = [dst_morton_buffer, dst_indices_buffer, onesweep_pass_hist_buffer];

      // Dispatch digit binning compute shader for this pass
      ComputeTaskQueue.new_task(
        `mesh_${mesh_id}_radix_sort_pass_${bit_shift}`,
        bvh_sorting_wgsl_path,
        pass_inputs,
        pass_outputs,
        thread_blocks, // Optimal work distribution across GPU
        1,
        1,
        onesweep_digit_binning_cs_entry_point
      );
    }

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                       🌳 PHASE 3: H-PLOC BVH2 CONSTRUCTION                             │
    // │                                                                                         │
    // │  Hierarchical Parallel Locally-Ordered Clustering builds a binary BVH using the         │
    // │  sorted Morton codes in a bottom-up fashion. This modern algorithm provides excellent   |
    // |  GPU parallelism while maintaining high-quality hierarchy construction for optimal ray  │
    // │  traversal .                                                                            │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    // Calculate workgroup distribution for BVH2 construction
    const bvh2_workgroups = Math.max(1, Math.ceil(primitive_count / 128)); // 128 threads per workgroup for BVH ops

    // Acquire additional scratch buffers for hierarchy construction
    const parent_idx_buffer = blas_gpu_data.parent_idx_buffer; // Parent node index tracking
    const bvh4_index_pairs_buffer = blas_gpu_data.bvh4_index_pairs_buffer; // Child-parent relationships

    // Configure BVH2 construction input bindings
    this.bvh2_inputs[0] = bvh2_nodes_buffer; // BVH2 node storage (input/output)
    this.bvh2_inputs[1] = sorted_indices_buffer; // Morton-sorted triangle indices
    this.bvh2_inputs[2] = mesh_info_buffer; // Per-mesh build parameters
    this.bvh2_inputs[3] = morton_codes_buffer; // Sorted Morton codes for clustering
    this.bvh2_inputs[4] = parent_idx_buffer; // Parent index tracking buffer
    this.bvh2_inputs[5] = bvh4_index_pairs_buffer; // Index pairs for BVH4 conversion prep

    // Configure BVH2 construction output bindings
    this.bvh2_outputs[0] = bvh2_nodes_buffer; // Updated BVH2 nodes with computed bounds
    this.bvh2_outputs[1] = bvh2_nodes_buffer; // Reused binding for efficiency

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                           🌱 LEAF CLUSTER INITIALIZATION                                │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘
    ComputeTaskQueue.new_task(
      `${hploc_init_leaf_clusters_task_name}_${mesh_id}`,
      bvh_as_init_wgsl_path,
      this.bvh2_inputs,
      this.bvh2_outputs,
      bvh2_workgroups, // Parallel leaf processing
      1,
      1,
      initialize_leaf_clusters_cs_entry_point
    );

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                         🏗️ HIERARCHICAL BINARY TREE CONSTRUCTION                      │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘
    ComputeTaskQueue.new_task(
      `${hploc_build_bvh2_task_name}_${mesh_id}`,
      bvh_processing_wgsl_path,
      this.bvh2_inputs,
      this.bvh2_outputs,
      bvh2_workgroups, // Parallel internal node construction
      1,
      1,
      build_bvh2_hploc_cs_entry_point
    );

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                     ⚡ PHASE 4: BVH4 CONVERSION & OPTIMIZATION                         │
    // │                                                                                          │
    // │  Convert the binary BVH2 tree into a hardware-optimized quaternary BVH4 structure.     │
    // │  BVH4 reduces traversal depth and matches modern GPU SIMD widths for superior          │
    // │  ray tracing performance compared to binary structures.                                 │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    // Initialize BVH4 construction state machine
    // BuildState: [work_counter, node_counter, leaf_counter, work_alloc_counter, primitive_count]
    const build_state = new Uint32Array(5);
    build_state[0] = 0; // work_counter: Active work items in queue
    build_state[1] = 1; // node_counter: Next available node index (start at root)
    build_state[2] = 0; // leaf_counter: Number of leaf nodes created
    build_state[3] = 1; // work_alloc_counter: Work allocation tracking
    build_state[4] = primitive_count; // primitive_count: Total triangles for validation

    // Acquire BVH4 conversion infrastructure buffers
    const bvh4_build_state_buffer = blas_gpu_data.bvh4_build_state_buffer; // Build state machine
    const bvh4_nodes_buffer = blas_gpu_data.bvh4_nodes_buffer; // Final BVH4 nodes
    const bvh4_prim_indices_buffer = blas_gpu_data.bvh4_prim_indices_buffer; // Primitive assignments
    const bvh4_debug_watchdog_buffer = blas_gpu_data.bvh4_debug_watchdog_buffer; // Debug/safety monitoring

    // Initialize build state and reset debug counters
    bvh4_build_state_buffer.write(build_state);
    bvh4_debug_watchdog_buffer.write_raw(new Uint32Array(4)); // Clear debug state

    // Configure BVH4 conversion input bindings
    this.bvh4_inputs[0] = bvh2_nodes_buffer; // Source BVH2 tree for conversion
    this.bvh4_inputs[1] = bvh4_nodes_buffer; // Destination BVH4 node buffer
    this.bvh4_inputs[2] = bvh4_build_state_buffer; // Conversion state machine
    this.bvh4_inputs[3] = bvh4_index_pairs_buffer; // Child-parent relationship tracking
    this.bvh4_inputs[4] = bvh4_prim_indices_buffer; // Triangle index assignments
    this.bvh4_inputs[5] = mesh_info_buffer; // Per-mesh build parameters
    this.bvh4_inputs[6] = bvh4_debug_watchdog_buffer; // Debug/safety infrastructure

    // Configure BVH4 conversion output bindings
    this.bvh4_outputs[0] = bvh4_nodes_buffer; // Final optimized BVH4 tree
    this.bvh4_outputs[1] = bvh4_build_state_buffer; // Updated build state
    this.bvh4_outputs[2] = bvh4_index_pairs_buffer; // Updated relationship data
    this.bvh4_outputs[3] = bvh4_prim_indices_buffer; // Final primitive assignments
    this.bvh4_outputs[4] = bvh4_debug_watchdog_buffer; // Updated debug counters

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                        🔀 PARALLEL BVH4 CONVERSION DISPATCH                             │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    const bvh4_workgroups = Math.max(1, Math.ceil(primitive_count / 32)); // 32 threads per workgroup
    ComputeTaskQueue.new_task(
      `${hploc_convert_parallel_single_pass_task_name}_${mesh_id}`,
      bvh4_processing_wgsl_path,
      this.bvh4_inputs,
      this.bvh4_outputs,
      bvh4_workgroups, // Parallel conversion of BVH2 to BVH4
      1,
      1,
      convert_bvh2_to_bvh4_cs_entry_point
    );

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                        🔀 ATLAS BUILD DISPATCH                                         │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘
    
    MeshBLAS.build_atlas();
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                           🔧 PRIVATE PIPELINE ORCHESTRATION METHODS                       ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                        🏭 DIRTY MESH QUEUE ORCHESTRATOR                                 │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Queues all meshes marked as dirty and requiring BVH reconstruction for processing.
   * This method prepares the necessary resources and adds builds to the processing queue
   * instead of executing them immediately, enabling frame-rate-friendly single-build-per-frame
   * processing.
   *
   * 🔄 QUEUING WORKFLOW:
   *    Step 1: Dirty Mesh Detection    → Query MeshBLAS system for pending builds
   *    Step 2: Resource Preparation    → Auto-resize scratch buffers for largest mesh
   *    Step 3: Buffer Creation         → Create per-mesh parameter buffers
   *    Step 4: Queue Addition          → Add prepared build to processing queue
   *    Step 5: State Management        → Clear dirty flags after queuing
   *
   * 🚀 PERFORMANCE OPTIMIZATIONS:
   *    • Deferred Processing: Builds queued for later single-per-frame execution
   *    • Resource Pre-allocation: Buffers created during queuing phase
   *    • Queue Management: FIFO processing maintains build order consistency
   */
  queue_dirty_meshes() {
    const dirty_meshes = MeshBLAS.dirty_meshes;
    if (!dirty_meshes || dirty_meshes.size === 0) return;

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                           📋 BATCH PROCESSING SETUP                                     │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    // Process each dirty mesh and add to build queue
    for (const mesh_id of dirty_meshes) {
      const mesh_meta = MeshBLAS.get_mesh_meta(mesh_id);
      if (!mesh_meta) continue; // Skip meshes without valid metadata

      const leaf_count = mesh_meta.leaf_count >>> 0;
      if (!leaf_count) continue; // Skip empty meshes

      // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
      // │                         📊 PER-MESH RESOURCE PREPARATION                                │
      // └─────────────────────────────────────────────────────────────────────────────────────────┘

      // Ensure scratch buffer capacity matches mesh complexity (auto-resizes if needed)
      MeshBLAS.prepare_build(mesh_id, leaf_count);

      // Create temporary per-mesh parameter buffers
      const mesh_info_buffer = Buffer.create({
        name: `mesh_${mesh_id}_build_info`,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        size: 6 * 4, // BVHData structure: [leaf_count, bvh2_count, prim_count, bvh2_base, bvh4_base, is_blas]
      });

      const mesh_selector_buffer = Buffer.create({
        name: `mesh_${mesh_id}_selector`,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        size: 4, // Single u32: mesh_id for directory lookup
      });
      mesh_selector_buffer.write_raw(new Uint32Array([mesh_id >>> 0]));

      // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
      // │                              📋 ADD BUILD TO QUEUE                                      │
      // └─────────────────────────────────────────────────────────────────────────────────────────┘

      // Add prepared build to processing queue for frame-rate-friendly execution
      this.build_queue.push({
        mesh_id: mesh_id,
        leaf_count: leaf_count,
        mesh_info_buffer: mesh_info_buffer,
        mesh_selector_buffer: mesh_selector_buffer,
        mesh_meta: mesh_meta
      });
    }

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                          🏁 QUEUING COMPLETION & STATE CLEANUP                          │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    // Clear dirty mesh flags after successful queuing (builds will be processed later)
    dirty_meshes.clear();
  }

  /**
   * ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   * │                           🔄 SINGLE BUILD PROCESSOR                                      │
   * └─────────────────────────────────────────────────────────────────────────────────────────┘
   *
   * Processes one queued mesh BVH build per frame to maintain optimal performance.
   * This method executes the complete 5-phase pipeline for a single mesh, including
   * leaf bounds generation, Morton code computation, OneSweep sorting, H-PLOC BVH2
   * construction, and BVH4 conversion.
   *
   * 🎯 FRAME-RATE OPTIMIZATION:
   *    • Single Build Processing: Only one mesh build executed per frame
   *    • FIFO Queue Management: Maintains consistent build order
   *    • Resource Cleanup: Automatic buffer cleanup after build completion
   *    • Progressive Processing: Steady advancement through all queued builds
   */
  process_queued_build() {
    // Return early if no builds are queued
    if (this.build_queue.length === 0) return;

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                           📋 DEQUEUE NEXT BUILD                                         │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    // Get the next build from queue (FIFO order)
    const build_item = this.build_queue.shift();
    const { mesh_id, leaf_count, mesh_info_buffer, mesh_selector_buffer, mesh_meta } = build_item;

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                        🔢 PHASE 0: LEAF BOUNDS GENERATION                               │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    const blas_data = MeshBLAS.to_gpu_data();

    // Generate triangle bounding boxes from mesh geometry
    const leaf_workgroups = Math.max(1, Math.ceil(leaf_count / 64)); // 64 triangles per workgroup
    ComputeTaskQueue.new_task(
      `mesh_${mesh_id}_generate_leaf_bounds`,
      "acceleration/blas_leaf_bounds.wgsl",
      [
        blas_data.bvh2_nodes_buffer, // Destination: BVH2 leaf node storage
        blas_data.directory_buffer, // Mesh allocation directory
        mesh_selector_buffer, // Current mesh identifier
        mesh_meta.index_buffer, // Triangle index data
      ],
      [blas_data.bvh2_nodes_buffer], // Updated leaf bounds
      leaf_workgroups,
      1,
      1,
      "write_leaf_bounds"
    );

    // ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    // │                      🚀 COMPLETE 5-PHASE PIPELINE EXECUTION                             │
    // └─────────────────────────────────────────────────────────────────────────────────────────┘

    // Execute the complete compute pipeline: Morton → Sort → H-PLOC → BVH4
    this.build(
      mesh_id,
      leaf_count, // Triangle count
      mesh_info_buffer // Build parameters
    );
  }
}

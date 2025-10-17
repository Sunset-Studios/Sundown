// ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
//  ██████╗ ██╗   ██╗██╗  ██╗    ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗ ██████╗ ██████╗ 
//  ██╔══██╗██║   ██║██║  ██║    ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝██╔═══██╗██╔══██╗
//  ██████╔╝██║   ██║███████║    ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗██║   ██║██████╔╝
//  ██╔══██╗╚██╗ ██╔╝██╔══██║    ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║██║   ██║██╔══██╗
//  ██████╔╝ ╚████╔╝ ██║  ██║    ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║╚██████╔╝██║  ██║
//  ╚═════╝   ╚═══╝  ╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
// ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
//
//  GPU-Accelerated H-PLOC BVH Construction Pipeline
//  
//  This module implements a state-of-the-art GPU-driven H-PLOC (Hierarchical Parallel Locally-Ordered Clustering) 
//  algorithm for building high-performance BVH2/BVH4 acceleration structures.
//  
//  Pipeline Architecture:
//  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//  │  Bounds Update  │───▶│  Morton Codes   │─── │  OneSweep Sort  │───▶│  BVH2 → BVH4    │
//  │  & Validation   │    │  Generation     │    │  (Radix Sort)   │    │  Conversion     │
//  └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
//          │                        │                        │                        │
//          ▼                        ▼                        ▼                        ▼
//    • Entity AABB           • 3D Morton Z-order      • 4-pass radix sort     • Parallel BVH4
//    • Scene bounds          • Spatial hashing        • OneSweep algorithm    • SIMD traversal
//                            • Locality preservation  • Warp-optimized        • Cache efficiency
//
// ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════

import { ComputeTaskQueue } from "../renderer/compute_task_queue.js";
import { BVH, WORKGROUP_SIZE, TILE_SIZE, RADIX_PASSES } from "./bvh.js";
import { EntityManager } from "../core/ecs/entity.js";
import { TransformFragment } from "../core/ecs/fragments/transform_fragment.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { Buffer } from "../renderer/buffer.js";
import { StaticMeshFragment } from "../core/ecs/fragments/static_mesh_fragment.js";
import { MeshData } from "../renderer/mesh_data.js";

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                          COMPUTE TASK IDENTIFIERS                                             ║
// ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  These constants define the task names used by the ComputeTaskQueue system for GPU compute dispatch.          ║
// ║  Each task represents a distinct phase in the BVH construction pipeline.                                       ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ─── Primary BVH Construction Phases ──────────────────────────────────────────────────────────────────────────────
const bounds_processing_task_name = "bounds_processing";                      // Entity bounds computation & culling
const hploc_compute_morton_codes_task_name = "hploc_compute_morton_codes";     // 3D Morton code generation
const hploc_init_leaf_clusters_task_name = "hploc_init_leaf_clusters";        // BVH2 leaf initialization
const hploc_build_bvh2_task_name = "hploc_build_bvh2";                        // H-PLOC binary BVH construction
const hploc_convert_parallel_single_pass_task_name = "hploc_convert_parallel_single_pass"; // BVH2 → BVH4 conversion

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                            SHADER RESOURCE PATHS                                               ║
// ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  WGSL shader file paths for each compute kernel in the BVH construction pipeline.                              ║
// ║  Each shader is specialized for a specific algorithm phase.                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ─── Core Algorithm Shaders ───────────────────────────────────────────────────────────────────────────────────────
const bounds_processing_wgsl_path = "system_compute/bounds_processing.wgsl";   // Entity bounds & scene AABB
const bvh_morton_wgsl_path = "acceleration/bvh_morton.wgsl";                  // Morton code generation
const bvh_sorting_wgsl_path = "acceleration/bvh_sorting.wgsl";                // OneSweep radix sort
const bvh_as_init_wgsl_path = "acceleration/bvh_as_init.wgsl";                // BVH2 initialization
const bvh_processing_wgsl_path = "acceleration/bvh_processing.wgsl";          // H-PLOC BVH2 construction
const bvh4_processing_wgsl_path = "acceleration/bvh4_processing.wgsl";        // BVH4 conversion & optimization

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                          COMPUTE SHADER ENTRY POINTS                                          ║
// ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  Function names for compute shader entry points. Each entry point implements a specific algorithm.             ║
// ║  The naming follows the structure: algorithm_operation_cs                                                       ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ─── Morton Code Generation ───────────────────────────────────────────────────────────────────────────────────────
const compute_morton_codes_cs_entry_point = "compute_morton_codes";           // 3D position → Morton Z-order curve

// ─── OneSweep Radix Sort Pipeline ─────────────────────────────────────────────────────────────────────────────────
const onesweep_init_cs_entry_point = "onesweep_init";                        // Initialize sort workspace
const onesweep_histogram_cs_entry_point = "onesweep_global_histogram";       // Build digit histograms
const onesweep_scan_cs_entry_point = "onesweep_scan";                        // Prefix sum computation
const onesweep_digit_binning_cs_entry_point = "onesweep_digit_binning";      // Scatter sorted elements

// ─── H-PLOC BVH Construction ──────────────────────────────────────────────────────────────────────────────────────
const initialize_leaf_clusters_cs_entry_point = "initialize_leaf_clusters";  // Setup BVH2 leaf nodes
const build_bvh2_hploc_cs_entry_point = "build_bvh2_hploc";                  // Parallel hierarchical clustering
const convert_bvh2_to_bvh4_cs_entry_point = "convert_bvh2_to_bvh4";          // Wide BVH4 node construction

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                           ECS FRAGMENT BUFFER NAMES                                            ║
// ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  String identifiers for accessing ECS fragment GPU buffers. These correspond to data layouts                   ║
// ║  defined in the fragment systems and provide type-safe buffer access.                                          ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ─── Entity Component Data Streams ────────────────────────────────────────────────────────────────────────────────
const transforms_name = "transforms";                                        // Transform matrices (position/rotation/scale)
const bounds_name = "bounds";                                                // Entity bounding boxes (AABB)
const mesh_asset_id_name = "mesh_asset_id";                                  // Mesh resource identifiers

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║                                            BVH PROCESSOR CLASS                                                 ║
// ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
// ║  High-performance GPU-driven BVH construction orchestrator implementing the H-PLOC algorithm.                  ║
// ║                                                                                                                  ║
// ║  Architecture Features:                                                                                          ║
// ║  • Frame-coherent BVH reconstruction with dynamic primitive count handling                                      ║
// ║  • GPU-native OneSweep radix sort for optimal Morton code ordering                                              ║
// ║  • Parallel hierarchical clustering using H-PLOC for balanced tree construction                                ║
// ║  • Automatic BVH2 → BVH4 conversion for SIMD-optimized traversal performance                                   ║
// ║  • Memory-efficient ping-pong buffer management for multi-pass algorithms                                       ║
// ║  • Integrated with ECS for seamless entity data streaming to GPU                                                ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

export class BVHProcessor {
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  //                                           PROCESSOR STATE
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  
  is_initialised = false;                        // Tracks processor initialization status
  max_primitives = 256;                          // Current maximum primitive capacity
  
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  //                                         ALGORITHM WORKSPACE BUFFERS
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  
  // ─── OneSweep Radix Sort Uniforms ─────────────────────────────────────────────────────────────────────────────────
  sort_uniforms_buffers = [];                    // Per-pass uniform buffers (4 passes for 32-bit keys)
  radix_uniforms_data = new Uint32Array(4);      // [element_count, radix_shift, thread_blocks, padding]
  
  // ─── BVH Construction Parameters ──────────────────────────────────────────────────────────────────────────────────
  bvh2_uniforms = new Uint32Array(2);            // BVH2 algorithm parameters
  bvh2_data = new Uint32Array(6);                // [leaf_count, node_count, prim_count, prim_base, node_base, is_blas]
  
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  //                                        COMPUTE KERNEL I/O BUFFERS
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  // 
  //  These arrays hold GPU buffer references for each compute kernel's input/output binding.
  //  The arrays are pre-allocated and reused each frame to avoid allocation overhead.
  //
  
  // ─── Bounds Processing Phase ──────────────────────────────────────────────────────────────────────────────────────
  bounds_processing_inputs = [null, null, null, null, null, null];   // Entity data, scene bounds input
  bounds_processing_outputs = [null, null, null, null, null];         // Updated bounds, culling flags
  
  // ─── Morton Code Generation Phase ─────────────────────────────────────────────────────────────────────────────────
  morton_code_inputs = [null, null, null, null, null, null, null, null];  // Bounds, scene AABB, metadata
  morton_code_outputs = [null, null];                                     // Morton codes, initial indices
  
  // ─── OneSweep Radix Sort Phase ────────────────────────────────────────────────────────────────────────────────────
  radix_sort_inputs = [null, null, null, null, null, null, null, null];   // Keys, values, sort workspace
  radix_sort_outputs = [null, null];                                      // Sorted keys, sorted indices
  
  // ─── BVH2 Construction Phase ──────────────────────────────────────────────────────────────────────────────────────
  bvh2_inputs = [null, null, null, null, null, null, null, null, null, null]; // Sorted data, workspace
  bvh2_outputs = [null, null, null];                                          // Binary BVH nodes, metadata
  
  // ─── BVH4 Conversion Phase ────────────────────────────────────────────────────────────────────────────────────────
  bvh4_inputs = [null, null, null, null, null, null, null];              // BVH2 data, build state, debug
  bvh4_outputs = [null, null, null, null, null];                         // BVH4 nodes, final indices, debug

  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  //                                           INITIALIZATION
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  
  /**
   * Initialize the BVH processor and set up GPU resources.
   * 
   * This constructor performs the essential setup required for GPU-driven BVH construction:
   * 
   * 1. **BVH Infrastructure Setup**: Initializes the underlying BVH system with default capacity
   * 2. **OneSweep Sort Workspace**: Creates uniform buffers for each radix sort pass
   * 3. **Memory Layout Preparation**: Pre-allocates workspace for optimal GPU performance
   * 
   * The OneSweep radix sort requires separate uniform buffers for each of the 4 passes
   * (8 bits per pass × 4 passes = 32-bit key coverage). Each buffer stores:
   * - Element count for the current sort operation
   * - Radix shift value (0, 8, 16, 24 bits)
   * - Thread block count for optimal workgroup distribution
   * - Padding for GPU memory alignment
   * 
   * @returns {void}
   */
  constructor() {
    // Initialize the underlying BVH system with default primitive capacity
    BVH.initialize(this.max_primitives);
    
    // Create uniform buffers for OneSweep radix sort algorithm
    // Each pass needs its own uniform buffer for radix shift configuration
    for (let i = 0; i < RADIX_PASSES; i++) {
      this.sort_uniforms_buffers[i] = Buffer.create({
        name: `sort_uniforms_${i}`,           // Unique identifier for each pass
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        size: 16,                            // 4 × u32 values with alignment
      });
    }
  }

  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  //                                       MAIN ORCHESTRATION METHOD
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  
  /**
   * Execute the complete BVH construction pipeline for the current frame.
   * 
   * This method orchestrates the entire H-PLOC BVH construction process through a carefully
   * ordered sequence of GPU compute dispatches. Each phase depends on the output of previous
   * phases, creating a pipeline that transforms entity data into a high-performance BVH4.
   * 
   * Pipeline Execution Order:
   * ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   * │ 1. Update       │───▶│ 2. Morton       │───▶│ 3. OneSweep     │
   * │    Bounds       │    │    Codes        │    │    Sort         │
   * └─────────────────┘    └─────────────────┘    └─────────────────┘
   *          │                        │                        │
   *          ▼                        ▼                        ▼
   * • Entity AABBs            • Z-order curve           • Sorted by locality
   * • Scene bounds            • Spatial hashing         • Cache-friendly access
   * • Visibility culling      • 3D → 1D mapping         • Ready for clustering
   * 
   * ┌─────────────────┐    ┌─────────────────┐
   * │ 4. Build        │───▶│ 5. Convert      │
   * │    BVH2         │    │    BVH2→BVH4    │
   * └─────────────────┘    └─────────────────┘
   *          │                        │
   *          ▼                        ▼
   * • Binary hierarchy        • 4-way branching
   * • H-PLOC clustering       • SIMD traversal
   * • Balanced partitioning   • Cache optimization
   * 
   * @returns {void} - Computation is entirely GPU-side
   */
  build() {
    // Early exit if no primitives to process - avoid unnecessary GPU work
    const primitive_count = EntityManager.get_max_rows();
    if (primitive_count === 0) return;

    // Dynamic capacity management - grow when needed with 2x expansion
    // This amortizes reallocation cost while handling dynamic scenes
    if (this.max_primitives < primitive_count) {
      this.resize(primitive_count * 2);         // Double capacity for future growth
    }

    // ═══ Execute H-PLOC Pipeline Phases ═══════════════════════════════════════════════════════════════════════════
    this.update_bounds();                       // Phase 1: Entity AABB computation & scene bounds
    this.compute_morton_codes();                // Phase 2: 3D positions → Morton Z-order codes
    this.clear_onesweep();                      // Phase 3a: Initialize OneSweep sort workspace
    this.radix_sort();                          // Phase 3b: Sort primitives by Morton codes
    this.build_bvh2();                          // Phase 4: Construct binary BVH using H-PLOC
    this.convert_bvh2_to_bvh4();                // Phase 5: Convert to wide BVH4 for performance
  }

  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  //                                         BVH CONSTRUCTION PHASES
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  
  /**
   * Phase 1: Update entity bounding boxes and compute global scene bounds.
   * 
   * This phase performs several critical preprocessing steps:
   * 
   * **Entity AABB Computation**:
   * • Transform mesh bounds by entity transform matrices
   * • Handle dynamic entities with time-varying transforms
   * • Apply frustum culling flags for visibility optimization
   * 
   * **Scene Bounds Accumulation**:
   * • Compute tight world-space AABB encompassing all visible entities
   * • Essential for Morton code normalization in next phase
   * • Updated atomically on GPU for thread safety
   * 
   * **Memory Access Pattern**:
   * • Coalesced reads from ECS fragment buffers
   * • Atomic updates to global scene bounds
   * • Write-through to entity bounds for subsequent phases
   * 
   * The bounds processing uses 256-thread workgroups for optimal occupancy,
   * with each thread processing one entity's transform and bounds.
   * 
   * @returns {void}
   */
  update_bounds() {
    // Reset scene bounds to inverted infinity for proper min/max accumulation
    BVH.clear_scene_bounds();

    // Gather entity count and GPU resource handles
    const total_rows = EntityManager.get_max_rows();
    const tlas_buffers = BVH.to_gpu_data();
    
    // ─── ECS Fragment Buffer Access ───────────────────────────────────────────────────────────────────────────────
    const transforms_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      transforms_name                           // Transform matrices (position/rotation/scale)
    );
    const bounds_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name                               // Entity bounding boxes (will be updated)
    );
    const static_mesh_ids_buffer = EntityManager.get_fragment_gpu_buffer(
      StaticMeshFragment,
      mesh_asset_id_name                        // Mesh asset identifiers for bounds lookup
    );
    const entity_flags_buffer = FragmentGpuBuffer.entity_flags_buffer; // Visibility/culling flags

    // Mesh resource data for bounds transformation
    const mesh_data = MeshData.to_gpu_data();

    // ─── Configure Compute Kernel Input Bindings ─────────────────────────────────────────────────────────────────
    this.bounds_processing_inputs[0] = transforms_buffer.buffer;        // Entity transform matrices
    this.bounds_processing_inputs[1] = entity_flags_buffer.buffer;      // Visibility and culling flags
    this.bounds_processing_inputs[2] = bounds_buffer.buffer;            // Current entity bounds (input)
    this.bounds_processing_inputs[3] = tlas_buffers.scene_bounds_buffer; // Global scene AABB (accumulator)
    this.bounds_processing_inputs[4] = static_mesh_ids_buffer.buffer;   // Mesh asset ID mappings
    this.bounds_processing_inputs[5] = mesh_data.mesh_bounds_buffer;    // Mesh-space bounding boxes

    // ─── Configure Compute Kernel Output Bindings ────────────────────────────────────────────────────────────────
    this.bounds_processing_outputs[0] = bounds_buffer.buffer;           // Updated entity bounds
    this.bounds_processing_outputs[1] = entity_flags_buffer.buffer;     // Updated culling flags
    this.bounds_processing_outputs[2] = tlas_buffers.scene_bounds_buffer; // Accumulated scene bounds

    // ─── Dispatch Bounds Processing Compute Kernel ───────────────────────────────────────────────────────────────
    ComputeTaskQueue.new_task(
      bounds_processing_task_name,
      bounds_processing_wgsl_path,
      this.bounds_processing_inputs,
      this.bounds_processing_outputs,
      Math.max(1, Math.floor((total_rows + 255) / 256)) // 256 threads per workgroup
    );
  }

  /**
   * Phase 2: Generate 3D Morton codes for spatial locality preservation.
   * 
   * Morton codes (also known as Z-order curves) map 3D spatial coordinates to 1D values
   * while preserving spatial locality. This is crucial for efficient BVH construction as
   * nearby primitives in 3D space remain close in the sorted Morton order.
   * 
   * **Morton Code Algorithm**:
   * 1. **Normalize Coordinates**: Map entity centers to [0,1]³ using scene bounds
   * 2. **Discretize**: Convert to fixed-point integers (typically 10 bits per axis)
   * 3. **Interleave Bits**: Weave x,y,z bits → zyxzyxzyx... pattern (30-bit codes)
   * 4. **Initialize Indices**: Set up primitive index array for subsequent sorting
   * 
   * **Spatial Properties**:
   * • Primitives close in 3D space have similar Morton codes
   * • Enables cache-friendly memory access during BVH traversal
   * • Supports efficient range queries and spatial partitioning
   * 
   * **GPU Implementation Details**:
   * • Each thread processes one primitive (entity)
   * • Workgroup size matches WORKGROUP_SIZE for optimal occupancy
   * • Uses bit manipulation for efficient Morton code computation
   * 
   * @returns {void}
   */
  compute_morton_codes() {
    // Get accurate primitive count for algorithm initialization
    const true_primitive_count = EntityManager.get_total_subscribed(TransformFragment);
    const conservative_primitive_count = EntityManager.get_max_rows();
    const bvh = BVH.to_gpu_data();
    const workgroups = Math.ceil(conservative_primitive_count / WORKGROUP_SIZE);

    // Access entity bounds computed in previous phase
    const bounds_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name                               // Updated entity AABBs from bounds processing
    );

    // ─── Initialize BVH Construction Metadata ─────────────────────────────────────────────────────────────────────
    // Reset algorithm counters and parameters for this frame's BVH construction
    this.bvh2_data[0] = 0;                      // leaf_count - Will be filled during BVH2 construction
    this.bvh2_data[1] = 0;                      // node_count - Internal node counter
    this.bvh2_data[2] = true_primitive_count;   // prim_count - Actual number of primitives to process
    this.bvh2_data[3] = 0;                      // prim_base - Base index for primitive storage
    this.bvh2_data[4] = 0;                      // node_base - Base index for node storage
    this.bvh2_data[5] = 0;                      // is_blas - 0 for TLAS (store AABB indices)
    bvh.bvh_info_buffer.write(this.bvh2_data);

    // ─── Configure Morton Code Generation Inputs ─────────────────────────────────────────────────────────────────
    this.morton_code_inputs[0] = bounds_buffer.buffer;         // Entity bounding boxes for center computation
    this.morton_code_inputs[1] = bvh.morton_codes_buffer;      // Output: Generated Morton codes
    this.morton_code_inputs[2] = bvh.sorted_indices_buffer;    // Output: Initial primitive indices [0,1,2...]
    this.morton_code_inputs[3] = bvh.scene_bounds_buffer;      // Scene AABB for coordinate normalization
    this.morton_code_inputs[4] = bvh.bvh_info_buffer;         // Algorithm metadata and counters

    // ─── Configure Morton Code Generation Outputs ────────────────────────────────────────────────────────────────
    this.morton_code_outputs[0] = bvh.morton_codes_buffer;     // 30-bit Morton Z-order codes
    this.morton_code_outputs[1] = bvh.sorted_indices_buffer;   // Primitive indices for sorting

    // ─── Dispatch Morton Code Generation Kernel ──────────────────────────────────────────────────────────────────
    ComputeTaskQueue.new_task(
      hploc_compute_morton_codes_task_name,
      bvh_morton_wgsl_path,
      this.morton_code_inputs,
      this.morton_code_outputs,
      workgroups,                               // One workgroup per WORKGROUP_SIZE primitives
      1,
      1,
      compute_morton_codes_cs_entry_point
    );
  }

  /**
   * Phase 3a: Initialize OneSweep radix sort workspace and counters.
   * 
   * OneSweep is a high-performance GPU radix sort algorithm that sorts data in a single
   * pass through memory, minimizing bandwidth usage and maximizing cache efficiency.
   * This initialization phase prepares the workspace for the subsequent sort operations.
   * 
   * **OneSweep Algorithm Overview**:
   * • **Single-Pass Design**: Avoids multiple data traversals common in other GPU sorts
   * • **Histogram-Based**: Uses digit frequency analysis for efficient partitioning  
   * • **Workgroup-Local Processing**: Minimizes global memory synchronization
   * • **Cache-Optimized**: Designed for modern GPU memory hierarchies
   * 
   * **Initialization Tasks**:
   * 1. **Clear Histograms**: Zero out digit frequency counters for all radix passes
   * 2. **Reset Tile Indices**: Initialize workgroup tile tracking arrays
   * 3. **Setup Parameters**: Configure element counts and workgroup distribution
   * 4. **Prepare Workspace**: Clear temporary buffers for ping-pong operations
   * 
   * The algorithm processes 8 bits (1 radix) per pass, requiring 4 passes total
   * for 32-bit Morton codes (4 × 8 = 32 bits).
   * 
   * @returns {void}
   */
  clear_onesweep() {
    const element_count = EntityManager.get_max_rows();
    const bvh = BVH.to_gpu_data();
    const thread_blocks = Math.max(1, Math.ceil(element_count / TILE_SIZE));

    // ─── Configure OneSweep Algorithm Parameters ──────────────────────────────────────────────────────────────────
    this.radix_uniforms_data[0] = element_count;    // Total number of Morton codes to sort
    this.radix_uniforms_data[1] = 0;                // radix_shift - Will be updated per pass (0,8,16,24)
    this.radix_uniforms_data[2] = thread_blocks;    // Number of workgroups for optimal GPU utilization
    this.radix_uniforms_data[3] = 0;                // Padding for GPU memory alignment
    this.sort_uniforms_buffers[0].write(this.radix_uniforms_data);

    // ─── Initialize OneSweep Workspace Buffers ───────────────────────────────────────────────────────────────────
    // Clear all histograms, counters, and tile tracking data structures
    // This kernel zeroes workspace memory and prepares for histogram computation
    ComputeTaskQueue.new_task(
      "onesweep_init",
      bvh_sorting_wgsl_path,
      [
        bvh.morton_codes_buffer,              // Primary keys buffer (layout compatibility)
        bvh.temp_morton_codes_buffer,         // Temporary keys workspace (ping-pong)
        bvh.sorted_indices_buffer,            // Primary values buffer (primitive indices)
        bvh.temp_sorted_indices_buffer,       // Temporary values workspace (ping-pong)
        bvh.onesweep_global_hist_buffer,      // Global histogram across all workgroups
        bvh.onesweep_pass_hist_buffer,        // Per-workgroup histogram data
        bvh.onesweep_tile_indices_buffer,     // Tile management for load balancing
        this.sort_uniforms_buffers[0],        // Algorithm parameters and counters
      ],
      [
        bvh.onesweep_global_hist_buffer,      // Cleared global histogram
        bvh.onesweep_pass_hist_buffer,        // Cleared pass histograms
        bvh.onesweep_tile_indices_buffer,     // Initialized tile indices
      ],
      256,                                   // Single workgroup for initialization
      1,
      1,
      onesweep_init_cs_entry_point
    );
  }

  /**
   * Phase 3b: Execute OneSweep radix sort to order primitives by Morton codes.
   * 
   * This implements the core OneSweep algorithm through a carefully orchestrated sequence
   * of GPU compute dispatches. The algorithm sorts 32-bit Morton codes using a 3-phase
   * approach that minimizes memory bandwidth while maximizing GPU parallelism.
   * 
   * **OneSweep Algorithm Phases**:
   * 
   * **Phase 1 - Histogram Construction**:
   * • Each workgroup builds local histograms for its data tile
   * • Histograms count digit frequencies for all 4 radix passes
   * • Results accumulated into global histogram for prefix sum computation
   * 
   * **Phase 2 - Prefix Sum Scan**:
   * • Compute exclusive prefix sums over global histogram
   * • Determines target positions for each digit value
   * • Enables conflict-free parallel scatter in Phase 3
   * 
   * **Phase 3 - Digit Binning (4 Radix Passes)**:
   * • Pass 0: Sort bits [7:0]   (least significant digit)
   * • Pass 1: Sort bits [15:8]  
   * • Pass 2: Sort bits [23:16]
   * • Pass 3: Sort bits [31:24] (most significant digit)
   * • Each pass uses ping-pong buffers to avoid read-after-write hazards
   * 
   * **Memory Access Pattern**:
   * • Coalesced reads from source buffers (optimal memory bandwidth)
   * • Scattered writes to destination buffers (sorted order)
   * • Ping-pong between primary and temporary buffers each pass
   * 
   * @returns {void}
   */
  radix_sort() {
    const element_count = EntityManager.get_max_rows();
    const bvh = BVH.to_gpu_data();
    const thread_blocks = Math.max(1, Math.ceil(element_count / TILE_SIZE));

    // ═══ Phase 1: Global Histogram Construction ═══════════════════════════════════════════════════════════════════
    // Build frequency histograms for all digit positions across all workgroups
    // Each workgroup processes a tile of data and contributes to the global histogram
    ComputeTaskQueue.new_task(
      `onesweep_histogram`,
      bvh_sorting_wgsl_path,
      [
        bvh.morton_codes_buffer,              // Input: Morton codes to analyze
        bvh.temp_morton_codes_buffer,         // (Unused in histogram phase)
        bvh.sorted_indices_buffer,            // Input: Primitive indices 
        bvh.temp_sorted_indices_buffer,       // (Unused in histogram phase)
        bvh.onesweep_global_hist_buffer,      // Output: Global digit frequency counts
        bvh.onesweep_pass_hist_buffer,        // Workspace: Per-workgroup histograms
        bvh.onesweep_tile_indices_buffer,     // Workspace: Tile management
        this.sort_uniforms_buffers[0],        // Parameters: Element count, etc.
      ],
      [bvh.onesweep_global_hist_buffer],      // Updated global histogram
      thread_blocks,                          // One workgroup per data tile
      1,
      1,
      onesweep_histogram_cs_entry_point
    );

    // ═══ Phase 2: Exclusive Prefix Sum Scan ══════════════════════════════════════════════════════════════════════
    // Compute prefix sums over global histogram to determine target scatter positions
    // This enables conflict-free parallel writes in the digit binning phase
    ComputeTaskQueue.new_task(
      `onesweep_scan`,
      bvh_sorting_wgsl_path,
      [
        bvh.morton_codes_buffer,              // (Layout compatibility)
        bvh.temp_morton_codes_buffer,         // (Layout compatibility)
        bvh.sorted_indices_buffer,            // (Layout compatibility)
        bvh.temp_sorted_indices_buffer,       // (Layout compatibility)
        bvh.onesweep_global_hist_buffer,      // Input: Global histogram from Phase 1
        bvh.onesweep_pass_hist_buffer,        // Output: Prefix sums for scatter
        bvh.onesweep_tile_indices_buffer,     // Workspace: Per-pass tile tracking
        this.sort_uniforms_buffers[0],        // Parameters
      ],
      [bvh.onesweep_pass_hist_buffer],        // Updated prefix sum data
      RADIX_PASSES,                           // One workgroup per radix pass
      1,
      1,
      onesweep_scan_cs_entry_point
    );

    // ═══ Phase 3: Digit Binning with Ping-Pong Buffers ══════════════════════════════════════════════════════════
    // Execute 4 radix passes, each sorting 8 bits of the 32-bit Morton codes
    // Ping-pong between buffers to avoid read-after-write dependencies
    const radix_shifts = [0, 8, 16, 24];     // Bit positions for each 8-bit radix
    
    for (let pass_index = 0; pass_index < radix_shifts.length; pass_index++) {
      const radix_shift = radix_shifts[pass_index];
      
      // ─── Ping-Pong Buffer Selection ─────────────────────────────────────────────────────────────────────────────
      // Alternate between primary and temporary buffers to avoid conflicts
      // Pass 0,2: primary → temp,  Pass 1,3: temp → primary
      const src_is_primary = (pass_index % 2) === 0;
      
      const src_keys = src_is_primary ? bvh.morton_codes_buffer : bvh.temp_morton_codes_buffer;
      const dst_keys = src_is_primary ? bvh.temp_morton_codes_buffer : bvh.morton_codes_buffer;
      const src_vals = src_is_primary ? bvh.sorted_indices_buffer : bvh.temp_sorted_indices_buffer;
      const dst_vals = src_is_primary ? bvh.temp_sorted_indices_buffer : bvh.sorted_indices_buffer;

      // ─── Update Pass-Specific Parameters ──────────────────────────────────────────────────────────────────────
      this.radix_uniforms_data[1] = radix_shift;     // Which 8-bit digit to sort on
      this.sort_uniforms_buffers[pass_index].write(this.radix_uniforms_data);

      // ─── Dispatch Digit Binning Kernel ────────────────────────────────────────────────────────────────────────
      ComputeTaskQueue.new_task(
        `onesweep_digit_binning_${radix_shift}`,
        bvh_sorting_wgsl_path,
        [
          src_keys,                           // Input: Keys to sort (Morton codes)
          dst_keys,                           // Output: Sorted keys
          src_vals,                           // Input: Values to sort (primitive indices)
          dst_vals,                           // Output: Sorted values
          bvh.onesweep_global_hist_buffer,    // Input: Global histogram data
          bvh.onesweep_pass_hist_buffer,      // Input: Prefix sums for scatter positions
          bvh.onesweep_tile_indices_buffer,   // Workspace: Tile coordination
          this.sort_uniforms_buffers[pass_index], // Parameters: radix_shift, etc.
        ],
        [dst_keys, dst_vals, bvh.onesweep_pass_hist_buffer], // Outputs: Sorted data
        thread_blocks,                        // Parallel processing across all tiles
        1,
        1,
        onesweep_digit_binning_cs_entry_point
      );
    }
  }

  /**
   * Phase 4: Construct binary BVH using the H-PLOC algorithm.
   * 
   * H-PLOC (Hierarchical Parallel Locally-Ordered Clustering) is a state-of-the-art
   * algorithm for building high-quality BVHs on the GPU. It combines the spatial locality
   * of Morton codes with parallel hierarchical clustering for optimal tree quality.
   * 
   * **H-PLOC Algorithm Overview**:
   * 
   * **Leaf Cluster Initialization**:
   * • Create leaf nodes for each primitive using sorted Morton order
   * • Initialize bounding boxes from entity AABBs
   * • Set up parent-child relationships for bottom-up construction
   * • Prepare work queues for parallel internal node processing
   * 
   * **Hierarchical Clustering**:
   * • Process primitives in Morton-sorted order for spatial coherence
   * • Use parallel reduction to merge leaf clusters into internal nodes
   * • Apply Surface Area Heuristic (SAH) for optimal split selection
   * • Build balanced binary tree through recursive partitioning
   * 
   * **Quality Characteristics**:
   * • Produces well-balanced trees with good traversal performance
   * • Minimizes surface area and expected ray-traversal cost
   * • Maintains spatial locality throughout the hierarchy
   * • Enables efficient parallel construction on GPU architectures using warp intrinsics
   * 
   * The algorithm operates in two phases: leaf initialization followed by internal
   * node construction through parallel bottom-up merging.
   * 
   * @returns {void}
   */
  build_bvh2() {
    const primitive_count = EntityManager.get_total_subscribed(TransformFragment);
    const bvh = BVH.to_gpu_data();
    const bvh2_workgroups = Math.max(1, Math.ceil(primitive_count / 128));

    // Access entity bounds for BVH node AABB computation
    const bounds_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name                               // World-space entity bounding boxes
    );

    // ─── Configure BVH2 Construction Input Bindings ──────────────────────────────────────────────────────────────
    this.bvh2_inputs[0] = bounds_buffer.buffer;         // Entity bounding boxes for leaf nodes
    this.bvh2_inputs[1] = bvh.sorted_indices_buffer;    // Morton-sorted primitive indices
    this.bvh2_inputs[2] = bvh.bvh_info_buffer;          // Algorithm metadata and counters
    this.bvh2_inputs[3] = bvh.morton_codes_buffer;      // Sorted Morton codes for clustering
    this.bvh2_inputs[4] = bvh.parent_idx_buffer;        // Parent node indices (will be filled)
    this.bvh2_inputs[5] = bvh.bvh4_index_pairs_buffer;  // Work queue for parallel construction

    // ─── Configure BVH2 Construction Output Bindings ─────────────────────────────────────────────────────────────
    this.bvh2_outputs[0] = bounds_buffer.buffer;        // Updated with internal node bounds
    this.bvh2_outputs[1] = bvh.bvh_info_buffer;         // Updated algorithm statistics

    // ═══ Phase 4a: Initialize Leaf Clusters ══════════════════════════════════════════════════════════════════════
    // Create leaf nodes from primitives and initialize the bottom level of the BVH
    // Each leaf corresponds to one primitive with its world-space bounding box
    ComputeTaskQueue.new_task(
      hploc_init_leaf_clusters_task_name,
      bvh_as_init_wgsl_path,
      this.bvh2_inputs,
      this.bvh2_outputs,
      bvh2_workgroups,                          // 128 threads per workgroup for leaf processing
      1,
      1,
      initialize_leaf_clusters_cs_entry_point
    );

    // ═══ Phase 4b: H-PLOC Binary BVH Construction ════════════════════════════════════════════════════════════════
    // Execute the core H-PLOC algorithm to build internal nodes through hierarchical clustering
    // This creates a binary tree structure optimized for traversal performance
    ComputeTaskQueue.new_task(
      hploc_build_bvh2_task_name,
      bvh_processing_wgsl_path,
      this.bvh2_inputs,
      this.bvh2_outputs,
      bvh2_workgroups,                          // Parallel processing of internal node construction
      1,
      1,
      build_bvh2_hploc_cs_entry_point
    );
  }

  /**
   * Phase 5: Convert binary BVH2 to wide BVH4 for optimal traversal performance.
   * 
   * This final phase transforms the binary BVH produced by H-PLOC into a 4-way branching
   * BVH4 structure. BVH4 nodes provide significant performance benefits on modern GPUs
   * through SIMD (Single Instruction, Multiple Data) parallel processing and less depth traversal cost overall.
   * 
   * **BVH4 Performance Advantages**:
   * 
   * **SIMD Traversal Efficiency**:
   * • Process 4 child nodes simultaneously using vector instructions
   * • Reduce branch divergence in GPU wavefronts/warps
   * • Minimize traversal depth through wider branching factor
   * • Enable efficient ray-AABB intersection testing for 4 boxes at once
   * 
   * **Cache Optimization**:
   * • Fewer memory accesses due to reduced tree depth
   * • Better spatial locality with grouped child nodes
   * • Optimal memory layout for GPU cache lines
   * • Reduced bandwidth requirements during traversal
   * 
   * **Conversion Algorithm**:
   * • Traverse BVH2 structure and identify collapsible subtrees
   * • Merge compatible binary nodes into 4-way nodes
   * • Preserve spatial coherence and bounding volume hierarchy
   * • Maintain primitive index mappings for correct intersection results
   * 
   * The algorithm uses a parallel single-pass approach with work queues to efficiently
   * convert the entire BVH2 structure while maintaining thread safety on the GPU.
   * 
   * @returns {void}
   */
  convert_bvh2_to_bvh4() {
    const primitive_count = EntityManager.get_total_subscribed(TransformFragment);
    const bvh = BVH.to_gpu_data();

    // Access final entity bounds data from BVH2 construction
    const bounds_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name                               // Entity bounds updated by BVH2 construction
    );
    
    // ─── Initialize BVH4 Construction State ──────────────────────────────────────────────────────────────────────
    // Set up atomic counters and algorithm parameters for parallel BVH4 construction
    // This follows CUDA-style parallel algorithms with work distribution
    const build_state_data = new Uint32Array(5);
    build_state_data[0] = 0;                    // work_counter - Atomic counter for work items
    build_state_data[1] = 1;                    // node_counter - Next available BVH4 node index
    build_state_data[2] = 0;                    // leaf_counter - Number of leaf nodes created
    build_state_data[3] = 1;                    // work_alloc_counter - Work queue allocation
    build_state_data[4] = primitive_count;      // prim_count - Total primitives for validation
    bvh.bvh4_build_state_buffer.write(build_state_data);

    // ─── Configure BVH4 Conversion Input Bindings ────────────────────────────────────────────────────────────────
    this.bvh4_inputs[0] = bounds_buffer.buffer;             // Entity bounding boxes
    this.bvh4_inputs[1] = bvh.bvh4_nodes_buffer;            // Output: BVH4 node storage
    this.bvh4_inputs[2] = bvh.bvh4_build_state_buffer;      // Algorithm state counters
    this.bvh4_inputs[3] = bvh.bvh4_index_pairs_buffer;      // Work queue for parallel processing
    this.bvh4_inputs[4] = bvh.bvh4_prim_indices_buffer;     // Primitive index remapping
    this.bvh4_inputs[5] = bvh.bvh_info_buffer;              // BVH metadata from previous phases
    this.bvh4_inputs[6] = bvh.bvh4_debug_watchdog_buffer;   // Debug counters and validation

    // ─── Configure BVH4 Conversion Output Bindings ───────────────────────────────────────────────────────────────
    this.bvh4_outputs[0] = bvh.bvh4_nodes_buffer;           // Generated BVH4 nodes
    this.bvh4_outputs[1] = bvh.bvh4_build_state_buffer;     // Updated algorithm state
    this.bvh4_outputs[2] = bvh.bvh4_index_pairs_buffer;     // Updated work queue
    this.bvh4_outputs[3] = bvh.bvh4_prim_indices_buffer;    // Final primitive mappings
    this.bvh4_outputs[4] = bvh.bvh4_debug_watchdog_buffer;  // Debug statistics

    // ─── Initialize Debug Watchdog Counters ──────────────────────────────────────────────────────────────────────
    // Clear debug counters before algorithm execution for clean profiling data
    bvh.bvh4_debug_watchdog_buffer.write_raw(new Uint32Array(4));
    
    // ═══ Execute Parallel BVH4 Conversion ════════════════════════════════════════════════════════════════════════
    // Launch parallel conversion kernel with optimal workgroup distribution
    // 32 threads per workgroup provides good occupancy for BVH processing
    const workgroups = Math.max(1, Math.ceil(primitive_count / 32));
    ComputeTaskQueue.new_task(
      hploc_convert_parallel_single_pass_task_name,
      bvh4_processing_wgsl_path,
      this.bvh4_inputs,
      this.bvh4_outputs,
      workgroups,                               // Parallel processing across all primitives
      1,
      1,
      convert_bvh2_to_bvh4_cs_entry_point
    );
  }

  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  //                                           UTILITY METHODS
  // ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  
  /**
   * Dynamically resize the processor to handle larger primitive counts.
   * 
   * This method updates the processor's capacity to handle scenes with varying numbers
   * of primitives. The resize operation triggers lazy buffer reallocation in the underlying
   * BVH system, ensuring all GPU memory resources scale appropriately.
   * 
   * **Scaling Strategy**:
   * • Growth-only resizing to avoid frequent reallocations
   * • Triggers BVH buffer resize for consistent memory layout
   * • Maintains existing uniform buffers (size-independent)
   * • Preserves processor state and configuration
   * 
   * **Memory Management**:
   * • BVH buffers are resized lazily on next rebuild_buffers() call
   * • No immediate GPU memory allocation to avoid frame stalls
   * • Old buffers remain valid until replacement buffers are created
   * 
   * @param {number} new_max_primitives - New minimum primitive capacity required
   * @returns {void}
   */
  resize(new_max_primitives) {
    this.max_primitives = new_max_primitives;
    BVH.resize(new_max_primitives);           // Trigger BVH buffer resize
  }

  /**
   * Clean up all GPU resources and destroy the processor.
   * 
   * This method performs complete cleanup of all GPU buffers and resources allocated
   * by the BVH processor. It should be called when the processor is no longer needed
   * to prevent GPU memory leaks.
   * 
   * **Cleanup Operations**:
   * • Destroy underlying BVH system and all associated buffers
   * • Release OneSweep radix sort uniform buffers
   * • Clear all buffer references to prevent dangling pointers
   * • Mark processor as uninitialized
   * 
   * **Resource Management**:
   * • Gracefully handles partial initialization states
   * • Safely destroys only allocated resources
   * • Prevents double-destruction through null checks
   * 
   * @returns {void}
   */
  destroy() {
    // Destroy the underlying BVH system and all associated GPU buffers
    BVH.destroy();
    
    // Clean up OneSweep radix sort uniform buffers
    for (let i = 0; i < RADIX_PASSES; i++) {
      if (this.sort_uniforms_buffers[i]) {
        this.sort_uniforms_buffers[i].destroy();
      }
    }
  }
}

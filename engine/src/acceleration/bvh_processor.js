/*
 * GPU-driven H-PLOC acceleration structure builder.
 * This class orchestrates the construction of a BVH4 from a set of primitives
 * each frame by dispatching a series of compute kernels.
 *
 * The process follows the H-PLOC algorithm:
 * 1. Compute Morton codes for all primitives.
 * 2. Sort primitive indices based on Morton codes (radix sort).
 * 3. Build a binary BVH (BVH2) using parallel hierarchical clustering.
 * 4. Convert the BVH2 into a wider BVH4 for improved traversal performance.
 */

import { ComputeTaskQueue } from "../renderer/compute_task_queue.js";
import { BVH, WORKGROUP_SIZE, TILE_SIZE, RADIX_PASSES } from "./bvh.js";
import { EntityManager } from "../core/ecs/entity.js";
import { TransformFragment } from "../core/ecs/fragments/transform_fragment.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { Buffer } from "../renderer/buffer.js";
import { StaticMeshFragment } from "../core/ecs/fragments/static_mesh_fragment.js";
import { MeshData } from "../renderer/mesh_data.js";

const bounds_processing_task_name = "bounds_processing";
const bounds_processing_wgsl_path = "system_compute/bounds_processing.wgsl";
const hploc_compute_morton_codes_task_name = "hploc_compute_morton_codes";
const hploc_init_leaf_clusters_task_name = "hploc_init_leaf_clusters";
const hploc_build_bvh2_task_name = "hploc_build_bvh2";
const hploc_convert_parallel_single_pass_task_name = "hploc_convert_parallel_single_pass";

const bvh_sorting_wgsl_path = "acceleration/bvh_sorting.wgsl";
const bvh_morton_wgsl_path = "acceleration/bvh_morton.wgsl";
const bvh_as_init_wgsl_path = "acceleration/bvh_as_init.wgsl";
const bvh_processing_wgsl_path = "acceleration/bvh_processing.wgsl";
const bvh4_processing_wgsl_path = "acceleration/bvh4_processing.wgsl";

const compute_morton_codes_cs_entry_point = "compute_morton_codes";
const onesweep_init_cs_entry_point = "onesweep_init";
const onesweep_histogram_cs_entry_point = "onesweep_global_histogram";
const onesweep_scan_cs_entry_point = "onesweep_scan";
const onesweep_digit_binning_cs_entry_point = "onesweep_digit_binning";
const initialize_leaf_clusters_cs_entry_point = "initialize_leaf_clusters";
const build_bvh2_hploc_cs_entry_point = "build_bvh2_hploc";
const convert_bvh2_to_bvh4_cs_entry_point = "convert_bvh2_to_bvh4";

const transforms_name = "transforms";
const bounds_name = "bounds";
const mesh_asset_id_name = "mesh_asset_id";

export class BVHProcessor {
  is_initialised = false;
  max_primitives = 256;
  sort_uniforms_buffers = [];
  radix_uniforms_data = new Uint32Array(4);
  bvh2_uniforms = new Uint32Array(2);
  bvh2_data = new Uint32Array(6);
  bounds_processing_inputs = [null, null, null, null, null, null];
  bounds_processing_outputs = [null, null, null, null, null];
  morton_code_inputs = [null, null, null, null, null, null, null, null];
  morton_code_outputs = [null, null];
  radix_sort_inputs = [null, null, null, null, null, null, null, null];
  radix_sort_outputs = [null, null];
  bvh2_inputs = [null, null, null, null, null, null, null, null, null, null];
  bvh2_outputs = [null, null, null];
  bvh4_inputs = [null, null, null, null, null];
  bvh4_outputs = [null, null, null, null];

  constructor() {
    BVH.initialize(this.max_primitives);
    for (let i = 0; i < RADIX_PASSES; i++) {
      this.sort_uniforms_buffers[i] = Buffer.create({
        name: `sort_uniforms_${i}`,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        size: 16,
      });
    }
  }

  build() {
    const primitive_count = EntityManager.get_max_rows();
    if (primitive_count === 0) return;

    if (this.max_primitives < primitive_count) {
      this.resize(primitive_count * 2);
    }

    this.update_bounds();
    this.compute_morton_codes();
    this.clear_onesweep();
    this.radix_sort();
    this.build_bvh2();
    this.convert_bvh2_to_bvh4();
  }

  update_bounds() {
    BVH.clear_scene_bounds();

    const total_rows = EntityManager.get_max_rows();
    const tlas_buffers = BVH.to_gpu_data();
    const transforms_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      transforms_name
    );
    const bounds_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name
    );
    const static_mesh_ids_buffer = EntityManager.get_fragment_gpu_buffer(
      StaticMeshFragment,
      mesh_asset_id_name
    );
    const entity_flags_buffer = FragmentGpuBuffer.entity_flags_buffer;

    const mesh_data = MeshData.to_gpu_data();

    this.bounds_processing_inputs[0] = transforms_buffer.buffer;
    this.bounds_processing_inputs[1] = entity_flags_buffer.buffer;
    this.bounds_processing_inputs[2] = bounds_buffer.buffer;
    this.bounds_processing_inputs[3] = tlas_buffers.scene_bounds_buffer;
    this.bounds_processing_inputs[4] = static_mesh_ids_buffer.buffer;
    this.bounds_processing_inputs[5] = mesh_data.mesh_bounds_buffer;

    this.bounds_processing_outputs[0] = bounds_buffer.buffer;
    this.bounds_processing_outputs[1] = entity_flags_buffer.buffer;
    this.bounds_processing_outputs[2] = tlas_buffers.scene_bounds_buffer;

    ComputeTaskQueue.new_task(
      bounds_processing_task_name,
      bounds_processing_wgsl_path,
      this.bounds_processing_inputs,
      this.bounds_processing_outputs,
      Math.max(1, Math.floor((total_rows + 255) / 256))
    );
  }

  compute_morton_codes() {
    const true_primitive_count = EntityManager.get_total_subscribed(TransformFragment);
    const conservative_primitive_count = EntityManager.get_max_rows();
    const bvh = BVH.to_gpu_data();
    const workgroups = Math.ceil(conservative_primitive_count / WORKGROUP_SIZE);

    const bounds_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name
    );

    // Reset counters for this frame
    this.bvh2_data[0] = 0; // leaf_count
    this.bvh2_data[1] = 0; // bvh2_count
    this.bvh2_data[2] = true_primitive_count; // prim_count
    this.bvh2_data[3] = 0; // prim_base
    this.bvh2_data[4] = 0; // node_base
    bvh.bvh_info_buffer.write(this.bvh2_data);

    this.morton_code_inputs[0] = bounds_buffer.buffer;
    this.morton_code_inputs[1] = bvh.morton_codes_buffer;
    this.morton_code_inputs[2] = bvh.sorted_indices_buffer;
    this.morton_code_inputs[3] = bvh.scene_bounds_buffer;
    this.morton_code_inputs[4] = bvh.bvh_info_buffer;

    this.morton_code_outputs[0] = bvh.morton_codes_buffer;
    this.morton_code_outputs[1] = bvh.sorted_indices_buffer;

    ComputeTaskQueue.new_task(
      hploc_compute_morton_codes_task_name,
      bvh_morton_wgsl_path,
      this.morton_code_inputs,
      this.morton_code_outputs,
      workgroups,
      1,
      1,
      compute_morton_codes_cs_entry_point
    );
  }

  clear_onesweep() {
    const element_count = EntityManager.get_max_rows();
    const bvh = BVH.to_gpu_data();
    const thread_blocks = Math.max(1, Math.ceil(element_count / TILE_SIZE));

    this.radix_uniforms_data[0] = element_count;
    this.radix_uniforms_data[1] = 0; // radix_shift
    this.radix_uniforms_data[2] = thread_blocks;
    this.radix_uniforms_data[3] = 0; // _padding
    this.sort_uniforms_buffers[0].write(this.radix_uniforms_data);

    // Initialize pass/global histograms and tile indices
    ComputeTaskQueue.new_task(
      "onesweep_init",
      bvh_sorting_wgsl_path,
      [
        bvh.morton_codes_buffer, // keys_buffer (layout compatibility)
        bvh.temp_morton_codes_buffer, // scatter_out (layout compatibility)
        bvh.sorted_indices_buffer, // values_buffer
        bvh.temp_sorted_indices_buffer, // values_scatter_out
        bvh.onesweep_global_hist_buffer, // global_historgram
        bvh.onesweep_pass_hist_buffer, // pass_histogram
        bvh.onesweep_tile_indices_buffer, // tile_indices
        this.sort_uniforms_buffers[0], // params
      ],
      [
        bvh.onesweep_global_hist_buffer,
        bvh.onesweep_pass_hist_buffer,
        bvh.onesweep_tile_indices_buffer,
      ],
      256,
      1,
      1,
      onesweep_init_cs_entry_point
    );
  }

  radix_sort() {
    const element_count = EntityManager.get_max_rows();
    const bvh = BVH.to_gpu_data();
    const thread_blocks = Math.max(1, Math.ceil(element_count / TILE_SIZE));

    // Phase 1: build global histograms (per pass, but kernel accumulates per-digit halves)
    ComputeTaskQueue.new_task(
      `onesweep_histogram`,
      bvh_sorting_wgsl_path,
      [
        bvh.morton_codes_buffer, // keys_buffer
        bvh.temp_morton_codes_buffer, // scatter_out (unused in this kernel)
        bvh.sorted_indices_buffer, // values_buffer
        bvh.temp_sorted_indices_buffer, // values_scatter_out
        bvh.onesweep_global_hist_buffer, // global_historgram
        bvh.onesweep_pass_hist_buffer, // pass_histogram
        bvh.onesweep_tile_indices_buffer, // tile_indices
        this.sort_uniforms_buffers[0], // params
      ],
      [bvh.onesweep_global_hist_buffer],
      thread_blocks,
      1,
      1,
      onesweep_histogram_cs_entry_point
    );

    // Phase 2: scan global histogram per radix pass
    ComputeTaskQueue.new_task(
      `onesweep_scan`,
      bvh_sorting_wgsl_path,
      [
        bvh.morton_codes_buffer,
        bvh.temp_morton_codes_buffer,
        bvh.sorted_indices_buffer,
        bvh.temp_sorted_indices_buffer,
        bvh.onesweep_global_hist_buffer,
        bvh.onesweep_pass_hist_buffer,
        bvh.onesweep_tile_indices_buffer,
        this.sort_uniforms_buffers[0],
      ],
      [bvh.onesweep_pass_hist_buffer],
      RADIX_PASSES,
      1,
      1,
      onesweep_scan_cs_entry_point
    );

    // Phase 3: digit binning (4 passes with ping-pong buffers)
    const passes = [0, 8, 16, 24];
    for (let i = 0; i < passes.length; i++) {
      const shift = passes[i];
      const src_is_morton = i % 2 === 0; // 0->codes, 8->alt, 16->codes, 24->alt
      const src_buffer = src_is_morton ? bvh.morton_codes_buffer : bvh.temp_morton_codes_buffer;
      const dst_buffer = src_is_morton ? bvh.temp_morton_codes_buffer : bvh.morton_codes_buffer;
      const src_vals = src_is_morton ? bvh.sorted_indices_buffer : bvh.temp_sorted_indices_buffer;
      const dst_vals = src_is_morton ? bvh.temp_sorted_indices_buffer : bvh.sorted_indices_buffer;

      this.radix_uniforms_data[1] = shift;
      this.sort_uniforms_buffers[i].write(this.radix_uniforms_data);

      ComputeTaskQueue.new_task(
        `onesweep_digit_binning_${shift}`,
        bvh_sorting_wgsl_path,
        [
          src_buffer, // keys_buffer
          dst_buffer, // scatter_out
          src_vals, // values_buffer
          dst_vals, // values_scatter_out
          bvh.onesweep_global_hist_buffer, // global_historgram
          bvh.onesweep_pass_hist_buffer, // pass_histogram
          bvh.onesweep_tile_indices_buffer, // tile_indices
          this.sort_uniforms_buffers[i], // params
        ],
        [dst_buffer, dst_vals, bvh.onesweep_pass_hist_buffer],
        thread_blocks,
        1,
        1,
        onesweep_digit_binning_cs_entry_point
      );
    }
  }

  build_bvh2() {
    const primitive_count = EntityManager.get_total_subscribed(TransformFragment);
    const bvh = BVH.to_gpu_data();
    const bvh2_workgroups = Math.ceil(primitive_count / 128);

    const bounds_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name
    );

    this.bvh2_inputs[0] = bounds_buffer.buffer;
    this.bvh2_inputs[1] = bvh.sorted_indices_buffer;
    this.bvh2_inputs[2] = bvh.bvh_info_buffer;
    this.bvh2_inputs[3] = bvh.morton_codes_buffer;
    this.bvh2_inputs[4] = bvh.parent_idx_buffer;
    this.bvh2_inputs[5] = bvh.bvh4_index_pairs_buffer;

    this.bvh2_outputs[0] = bounds_buffer.buffer;
    this.bvh2_outputs[1] = bvh.bvh_info_buffer;

    ComputeTaskQueue.new_task(
      hploc_init_leaf_clusters_task_name,
      bvh_as_init_wgsl_path,
      this.bvh2_inputs,
      this.bvh2_outputs,
      bvh2_workgroups,
      1,
      1,
      initialize_leaf_clusters_cs_entry_point
    );

    // H-PLOC build kernel (direct port)
    ComputeTaskQueue.new_task(
      hploc_build_bvh2_task_name,
      bvh_processing_wgsl_path,
      this.bvh2_inputs,
      this.bvh2_outputs,
      bvh2_workgroups,
      1,
      1,
      build_bvh2_hploc_cs_entry_point
    );
  }

  convert_bvh2_to_bvh4() {
    const primitive_count = EntityManager.get_total_subscribed(TransformFragment);
    const bvh = BVH.to_gpu_data();

    const bounds_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      bounds_name
    );
    
    // Initialize build state for CUDA-style algorithm
    const build_state_data = new Uint32Array(5);
    build_state_data[0] = 0; // work_counter
    build_state_data[1] = 1; // node_counter (start at 1, root allocated separately)
    build_state_data[2] = 0; // leaf_counter  
    build_state_data[3] = 1; // work_alloc_counter
    build_state_data[4] = primitive_count; // prim_count
    bvh.bvh4_build_state_buffer.write(build_state_data);

    this.bvh4_inputs[0] = bounds_buffer.buffer;
    this.bvh4_inputs[1] = bvh.bvh4_nodes_buffer;
    this.bvh4_inputs[2] = bvh.bvh4_build_state_buffer;
    this.bvh4_inputs[3] = bvh.bvh4_index_pairs_buffer;
    this.bvh4_inputs[4] = bvh.bvh4_prim_indices_buffer;
    this.bvh4_inputs[5] = bvh.bvh_info_buffer;
    this.bvh4_inputs[6] = bvh.bvh4_debug_watchdog_buffer;

    this.bvh4_outputs[0] = bvh.bvh4_nodes_buffer;
    this.bvh4_outputs[1] = bvh.bvh4_build_state_buffer;
    this.bvh4_outputs[2] = bvh.bvh4_index_pairs_buffer;
    this.bvh4_outputs[3] = bvh.bvh4_prim_indices_buffer;
    this.bvh4_outputs[4] = bvh.bvh4_debug_watchdog_buffer;

    // Zero watchdog before dispatch
    bvh.bvh4_debug_watchdog_buffer.write_raw(new Uint32Array(4));
    
    const workgroups = Math.max(1, Math.ceil(primitive_count / 32));
    ComputeTaskQueue.new_task(
      hploc_convert_parallel_single_pass_task_name,
      bvh4_processing_wgsl_path,
      this.bvh4_inputs,
      this.bvh4_outputs,
      workgroups,
      1,
      1,
      convert_bvh2_to_bvh4_cs_entry_point
    );
  }

  resize(new_max_primitives) {
    this.max_primitives = new_max_primitives;
    BVH.resize(new_max_primitives);
  }

  destroy() {
    BVH.destroy();
    for (let i = 0; i < RADIX_PASSES; i++) {
      if (this.sort_uniforms_buffers[i]) {
        this.sort_uniforms_buffers[i].destroy();
      }
    }
  }
}

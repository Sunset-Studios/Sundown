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
import { BVH } from "./bvh.js";
import { EntityManager } from "../core/ecs/entity.js";
import { TransformFragment } from "../core/ecs/fragments/transform_fragment.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { Buffer } from "../renderer/buffer.js";

const bounds_processing_task_name = "bounds_processing";
const bounds_processing_wgsl_path = "system_compute/bounds_processing.wgsl";
const transforms_name = "transforms";
const aabb_node_index_name = "aabb_node_index";

const WORKGROUP_SIZE = 256;
const RADIX_BITS = 4;
const MORTON_CODE_BITS = 30;

export class BVHProcessor {
  is_initialised = false;
  max_primitives = 256;
  sort_uniforms_buffer = null;
  bounds_processing_inputs = [null, null, null, null, null, null];
  bounds_processing_outputs = [null, null, null, null, null];

  constructor() {
    BVH.initialize(this.max_primitives);
    this.sort_uniforms_buffer = Buffer.create({
      name: "sort_uniforms",
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      size: 8,
    });
  }

  build() {
    const primitive_count = EntityManager.get_entity_count();
    if (primitive_count === 0) return;

    if (this.max_primitives < primitive_count) {
      this.resize(primitive_count * 2);
    }

    this.update_bounds();
    // this.compute_morton_codes(primitive_count);
    // this.radix_sort(primitive_count);
    // this.build_bvh2(primitive_count);
    // this.convert_bvh2_to_bvh4(primitive_count);
  }

  update_bounds() {
    BVH.clear_scene_bounds();

    const total_rows = EntityManager.get_max_rows();
    const tlas_buffers = BVH.to_gpu_data();
    const transforms_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      transforms_name
    );
    const aabb_node_index_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      aabb_node_index_name
    );
    const entity_flags_buffer = FragmentGpuBuffer.entity_flags_buffer;


    this.bounds_processing_inputs[0] = transforms_buffer.buffer;
    this.bounds_processing_inputs[1] = entity_flags_buffer.buffer;
    this.bounds_processing_inputs[2] = tlas_buffers.bounds_buffer;
    this.bounds_processing_inputs[3] = tlas_buffers.user_data_buffer;
    this.bounds_processing_inputs[4] = aabb_node_index_buffer.buffer;
    this.bounds_processing_inputs[5] = tlas_buffers.scene_bounds_buffer;

    this.bounds_processing_outputs[0] = tlas_buffers.bounds_buffer;
    this.bounds_processing_outputs[1] = entity_flags_buffer.buffer;
    this.bounds_processing_outputs[2] = tlas_buffers.user_data_buffer;
    this.bounds_processing_outputs[3] = tlas_buffers.scene_bounds_buffer;

    ComputeTaskQueue.new_task(
      bounds_processing_task_name,
      bounds_processing_wgsl_path,
      this.bounds_processing_inputs,
      this.bounds_processing_outputs,
      Math.max(1, Math.floor((total_rows + 255) / 256))
    );
  }

  compute_morton_codes(primitive_count) {
    const bvh = BVH.to_gpu_data();
    const workgroups = Math.ceil(primitive_count / WORKGROUP_SIZE);

    // Write element count into uniforms (bit_start = 0)
    const uniforms_data = new Uint32Array(2);
    uniforms_data[0] = 0;
    uniforms_data[1] = primitive_count;
    this.sort_uniforms_buffer.write(uniforms_data);

    // Bindings order must match acceleration/bvh_sorting.wgsl group(1):
    // 0 bounds, 1 in_morton, 2 in_sorted, 3 out_morton, 4 out_sorted, 5 histogram, 6 scene_aabb, 7 sort_uniforms
    const bindings = [
      bvh.bounds_buffer,
      bvh.morton_codes_buffer,
      bvh.sorted_indices_buffer,
      bvh.temp_morton_codes_buffer,
      bvh.temp_sorted_indices_buffer,
      bvh.histogram_buffer,
      bvh.scene_bounds_buffer,
      this.sort_uniforms_buffer,
    ];

    ComputeTaskQueue.new_task(
      `hploc_compute_morton_codes`,
      "acceleration/bvh_sorting.wgsl",
      bindings,
      [bvh.morton_codes_buffer, bvh.sorted_indices_buffer],
      workgroups,
      1,
      1,
      "compute_morton_codes"
    );
  }

  radix_sort(element_count) {
    const bvh = BVH.to_gpu_data();
    const radix_sort_workgroups = Math.ceil(element_count / WORKGROUP_SIZE);

    let input_keys = bvh.morton_codes_buffer;
    let input_values = bvh.sorted_indices_buffer;
    let output_keys = bvh.temp_morton_codes_buffer;
    let output_values = bvh.temp_sorted_indices_buffer;

    const num_passes = Math.ceil(MORTON_CODE_BITS / RADIX_BITS);

    const uniforms_data = new Uint32Array(2);
    uniforms_data[1] = element_count;

    for (let pass = 0; pass < num_passes; pass++) {
      uniforms_data[0] = pass * RADIX_BITS;
      this.sort_uniforms_buffer.write(uniforms_data);

      // Prepare uniforms
      ComputeTaskQueue.new_task(
        `hploc_radix_clear_hist_${pass}`,
        "acceleration/bvh_sorting.wgsl",
        [
          bvh.bounds_buffer,
          input_keys,
          input_values,
          output_keys,
          output_values,
          bvh.histogram_buffer,
          bvh.scene_bounds_buffer,
          this.sort_uniforms_buffer,
        ],
        [bvh.histogram_buffer],
        1,
        1,
        1,
        "clear_histogram"
      );

      // Compute histogram
      ComputeTaskQueue.new_task(
        `hploc_radix_histogram_${pass}`,
        "acceleration/bvh_sorting.wgsl",
        [
          bvh.bounds_buffer,
          input_keys,
          input_values,
          output_keys,
          output_values,
          bvh.histogram_buffer,
          bvh.scene_bounds_buffer,
          this.sort_uniforms_buffer,
        ],
        [bvh.histogram_buffer],
        radix_sort_workgroups,
        1,
        1,
        "compute_histogram"
      );

      // Prefix sum
      ComputeTaskQueue.new_task(
        `hploc_radix_prefix_${pass}`,
        "acceleration/bvh_sorting.wgsl",
        [
          bvh.bounds_buffer,
          input_keys,
          input_values,
          output_keys,
          output_values,
          bvh.histogram_buffer,
          bvh.scene_bounds_buffer,
          this.sort_uniforms_buffer,
        ],
        [bvh.histogram_buffer],
        1,
        1,
        1,
        "prefix_sum"
      );

      // Scatter
      ComputeTaskQueue.new_task(
        `hploc_radix_scatter_${pass}`,
        "acceleration/bvh_sorting.wgsl",
        [
          bvh.bounds_buffer,
          input_keys,
          input_values,
          output_keys,
          output_values,
          bvh.histogram_buffer,
          bvh.scene_bounds_buffer,
          this.sort_uniforms_buffer,
        ],
        [output_keys, output_values],
        radix_sort_workgroups,
        1,
        1,
        "scatter"
      );

      // Swap
      let swap = input_keys;
      input_keys = output_keys;
      output_keys = swap;
      swap = input_values;
      input_values = output_values;
      output_values = swap;
    }
  }

  build_bvh2(primitive_count) {
    const bvh = BVH.to_gpu_data();
    const bvh2_workgroups = Math.ceil(primitive_count / 64);

    // Initialize leaf clusters
    const hploc_uniforms = new Uint32Array(2);
    hploc_uniforms[0] = primitive_count; // primitive_count
    hploc_uniforms[1] = 0; // pass_num
    this.sort_uniforms_buffer.write(hploc_uniforms);

    // Reset combined node counters: [bvh2_count, bvh4_count]
    const zeros = new Uint32Array(2);
    zeros[0] = 0;
    zeros[1] = 0;
    bvh.node_counters_buffer.write(zeros);

    // Bind order must match acceleration/bvh_processing.wgsl group(1)
    const base_bindings = [
      bvh.bounds_buffer,
      bvh.sorted_indices_buffer,
      bvh.bvh2_nodes_buffer,
      bvh.bvh4_nodes_buffer,
      bvh.node_counters_buffer,
      this.sort_uniforms_buffer, // reuse buffer for HPLOCUniforms
      bvh.clusters_in_buffer,
      bvh.clusters_out_buffer,
      bvh.scene_bounds_buffer,
    ];

    ComputeTaskQueue.new_task(
      `hploc_init_leaf_clusters`,
      "acceleration/bvh_processing.wgsl",
      base_bindings,
      [bvh.bvh2_nodes_buffer, bvh.clusters_in_buffer, bvh.node_counters_buffer],
      Math.ceil(primitive_count / 1),
      1,
      1,
      "initialize_leaf_clusters"
    );

    // For now, skip the complex wave reduction kernel and depend on later conversion
    ComputeTaskQueue.new_task(
      `hploc_build_bvh2`,
      "acceleration/bvh_processing.wgsl",
      base_bindings,
      [bvh.bvh2_nodes_buffer],
      bvh2_workgroups,
      1,
      1,
      "build_bvh2_hploc"
    );
  }

  convert_bvh2_to_bvh4(primitive_count) {
    const bvh = BVH.to_gpu_data();
    // Parallel conversion: each thread converts a top-level BVH2 root using a local stack

    const total_nodes_estimate = primitive_count * 2 - 1;
    const workgroups = Math.max(1, Math.ceil(total_nodes_estimate / WORKGROUP_SIZE));
    ComputeTaskQueue.new_task(
      `hploc_convert_parallel_single_pass`,
      "acceleration/bvh_processing.wgsl",
      [
        bvh.bounds_buffer,
        bvh.sorted_indices_buffer,
        bvh.bvh2_nodes_buffer,
        bvh.bvh4_nodes_buffer,
        bvh.node_counters_buffer,
        this.sort_uniforms_buffer,
        bvh.clusters_in_buffer,
        bvh.clusters_out_buffer,
        bvh.scene_bounds_buffer,
      ],
      [bvh.bvh4_nodes_buffer, bvh.node_counters_buffer],
      workgroups,
      1,
      1,
      "convert_bvh2_to_bvh4"
    );
  }

  resize(new_max_primitives) {
    this.max_primitives = new_max_primitives;
    BVH.resize(new_max_primitives);
  }

  destroy() {
    BVH.destroy();
    if (this.sort_uniforms_buffer) {
      this.sort_uniforms_buffer.destroy();
    }
  }
}

import { SimulationLayer } from "../core/simulation_layer.js";
import { ComputeTaskQueue } from "../renderer/compute_task_queue.js";
import { Buffer } from "../renderer/buffer.js";
import { MeshBLAS } from "./mesh_blas.js";

// GPU H-PLOC phases for building per-mesh BLAS structures.
// Mirrors the TLAS builder but operates on mesh-local buffers.

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

const RADIX_PASSES = 4;
const RADIX = 256;
const WORKGROUP_SIZE = 256;
const TILE_SIZE = WORKGROUP_SIZE * 16;

const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export class MeshBLASProcessor extends SimulationLayer {
  constructor(max_primitives = 256) {
    super();

    this.max_primitives = max_primitives;

    this.sort_uniforms_buffers = [];
    for (let i = 0; i < RADIX_PASSES; i++) {
      this.sort_uniforms_buffers[i] = Buffer.create({
        name: `mesh_blas_sort_uniforms_${i}`,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        size: 16,
      });
    }

    const primitive_bytes = max_primitives * 4;

    this.morton_codes_buffer = Buffer.create({
      name: "mesh_blas_morton_codes",
      usage: storage,
      size: primitive_bytes,
    });
    this.temp_morton_codes_buffer = Buffer.create({
      name: "mesh_blas_temp_morton_codes",
      usage: storage,
      size: primitive_bytes,
    });
    this.sorted_indices_buffer = Buffer.create({
      name: "mesh_blas_sorted_indices",
      usage: storage,
      size: primitive_bytes,
    });
    this.temp_sorted_indices_buffer = Buffer.create({
      name: "mesh_blas_temp_sorted_indices",
      usage: storage,
      size: primitive_bytes,
    });

    const thread_blocks = Math.max(1, Math.ceil(max_primitives / TILE_SIZE));
    const pass_hist_count = thread_blocks * RADIX * RADIX_PASSES;

    this.onesweep_global_hist_buffer = Buffer.create({
      name: "mesh_blas_global_hist",
      usage: storage,
      size: RADIX * RADIX_PASSES * 4,
    });
    this.onesweep_pass_hist_buffer = Buffer.create({
      name: "mesh_blas_pass_hist",
      usage: storage,
      size: pass_hist_count * 4,
    });
    this.onesweep_tile_indices_buffer = Buffer.create({
      name: "mesh_blas_tile_indices",
      usage: storage,
      size: RADIX_PASSES * 4,
    });

    this.parent_idx_buffer = Buffer.create({
      name: "mesh_blas_parent_idx",
      usage: storage,
      size: primitive_bytes,
    });
    this.bvh4_nodes_buffer = Buffer.create({
      name: "mesh_blas_nodes",
      usage: storage,
      size: max_primitives * 48,
    });
    this.bvh_info_buffer = Buffer.create({
      name: "mesh_blas_info",
      usage: storage,
      size: 16,
    });
    this.bvh4_build_state_buffer = Buffer.create({
      name: "mesh_blas_build_state",
      usage: storage,
      size: 20,
    });
    this.bvh4_index_pairs_buffer = Buffer.create({
      name: "mesh_blas_index_pairs",
      usage: storage,
      size: primitive_bytes * 2,
    });
    this.bvh4_prim_indices_buffer = Buffer.create({
      name: "mesh_blas_prim_indices",
      usage: storage,
      size: primitive_bytes,
    });
    this.bvh4_debug_watchdog_buffer = Buffer.create({
      name: "mesh_blas_debug_watchdog",
      usage: storage,
      size: 16,
    });

    this.radix_uniforms = new Uint32Array(4);
    this.bvh2_data = new Uint32Array(4);

    this.morton_code_inputs = new Array(4);
    this.morton_code_outputs = new Array(2);
    this.radix_sort_inputs = new Array(8);
    this.radix_sort_outputs = new Array(3);
    this.bvh2_inputs = new Array(6);
    this.bvh2_outputs = new Array(2);
    this.bvh4_inputs = new Array(7);
    this.bvh4_outputs = new Array(5);
  }

  update(delta_time) {
    super.update(delta_time);
    this.build_dirty_meshes();
  }

  build(bounds_buffer, primitive_count) {
    if (!primitive_count) return;
    if (primitive_count > this.max_primitives) return;

    const morton_workgroups = Math.ceil(primitive_count / WORKGROUP_SIZE);

    this.morton_code_inputs[0] = bounds_buffer;
    this.morton_code_inputs[1] = this.morton_codes_buffer;
    this.morton_code_inputs[2] = this.sorted_indices_buffer;
    this.morton_code_inputs[3] = this.bvh_info_buffer;

    this.morton_code_outputs[0] = this.morton_codes_buffer;
    this.morton_code_outputs[1] = this.sorted_indices_buffer;

    ComputeTaskQueue.new_task(
      hploc_compute_morton_codes_task_name,
      bvh_morton_wgsl_path,
      this.morton_code_inputs,
      this.morton_code_outputs,
      morton_workgroups,
      1,
      1,
      compute_morton_codes_cs_entry_point
    );

    const thread_blocks = Math.max(1, Math.ceil(primitive_count / TILE_SIZE));
    this.radix_uniforms[0] = primitive_count;
    this.radix_uniforms[1] = 0;
    this.radix_uniforms[2] = thread_blocks;
    this.radix_uniforms[3] = 0;
    this.sort_uniforms_buffers[0].write(this.radix_uniforms);

    this.radix_sort_inputs[0] = this.morton_codes_buffer;
    this.radix_sort_inputs[1] = this.temp_morton_codes_buffer;
    this.radix_sort_inputs[2] = this.sorted_indices_buffer;
    this.radix_sort_inputs[3] = this.temp_sorted_indices_buffer;
    this.radix_sort_inputs[4] = this.onesweep_global_hist_buffer;
    this.radix_sort_inputs[5] = this.onesweep_pass_hist_buffer;
    this.radix_sort_inputs[6] = this.onesweep_tile_indices_buffer;
    this.radix_sort_inputs[7] = this.sort_uniforms_buffers[0];

    this.radix_sort_outputs[0] = this.onesweep_global_hist_buffer;
    this.radix_sort_outputs[1] = this.onesweep_pass_hist_buffer;
    this.radix_sort_outputs[2] = this.onesweep_tile_indices_buffer;

    ComputeTaskQueue.new_task(
      "mesh_onesweep_init",
      bvh_sorting_wgsl_path,
      this.radix_sort_inputs,
      this.radix_sort_outputs,
      256,
      1,
      1,
      onesweep_init_cs_entry_point
    );

    ComputeTaskQueue.new_task(
      "mesh_onesweep_histogram",
      bvh_sorting_wgsl_path,
      this.radix_sort_inputs,
      [this.onesweep_global_hist_buffer],
      thread_blocks,
      1,
      1,
      onesweep_histogram_cs_entry_point
    );

    ComputeTaskQueue.new_task(
      "mesh_onesweep_scan",
      bvh_sorting_wgsl_path,
      this.radix_sort_inputs,
      [this.onesweep_pass_hist_buffer],
      RADIX_PASSES,
      1,
      1,
      onesweep_scan_cs_entry_point
    );

    const passes = [0, 8, 16, 24];
    for (let i = 0; i < passes.length; i++) {
      const shift = passes[i];
      const src_is_morton = i % 2 === 0;
      const src_buffer = src_is_morton ? this.morton_codes_buffer : this.temp_morton_codes_buffer;
      const dst_buffer = src_is_morton ? this.temp_morton_codes_buffer : this.morton_codes_buffer;
      const src_vals = src_is_morton ? this.sorted_indices_buffer : this.temp_sorted_indices_buffer;
      const dst_vals = src_is_morton ? this.temp_sorted_indices_buffer : this.sorted_indices_buffer;

      this.radix_uniforms[1] = shift;
      this.sort_uniforms_buffers[i].write(this.radix_uniforms);

      const inputs = [
        src_buffer,
        dst_buffer,
        src_vals,
        dst_vals,
        this.onesweep_global_hist_buffer,
        this.onesweep_pass_hist_buffer,
        this.onesweep_tile_indices_buffer,
        this.sort_uniforms_buffers[i],
      ];

      const outputs = [dst_buffer, dst_vals, this.onesweep_pass_hist_buffer];

      ComputeTaskQueue.new_task(
        `mesh_onesweep_digit_binning_${shift}`,
        bvh_sorting_wgsl_path,
        inputs,
        outputs,
        thread_blocks,
        1,
        1,
        onesweep_digit_binning_cs_entry_point
      );
    }

    const bvh2_workgroups = Math.ceil(primitive_count / 128);

    this.bvh2_data[0] = 0;
    this.bvh2_data[1] = 0;
    this.bvh2_data[2] = 0xffffffff;
    this.bvh2_data[3] = primitive_count;
    this.bvh_info_buffer.write(this.bvh2_data);

    this.bvh2_inputs[0] = bounds_buffer;
    this.bvh2_inputs[1] = this.sorted_indices_buffer;
    this.bvh2_inputs[2] = this.bvh_info_buffer;
    this.bvh2_inputs[3] = this.morton_codes_buffer;
    this.bvh2_inputs[4] = this.parent_idx_buffer;
    this.bvh2_inputs[5] = this.bvh4_index_pairs_buffer;

    this.bvh2_outputs[0] = bounds_buffer;
    this.bvh2_outputs[1] = this.bvh_info_buffer;

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

    const build_state = new Uint32Array(5);
    build_state[0] = 0;
    build_state[1] = 1;
    build_state[2] = 0;
    build_state[3] = 1;
    build_state[4] = primitive_count;
    this.bvh4_build_state_buffer.write(build_state);
    this.bvh4_debug_watchdog_buffer.write_raw(new Uint32Array(4));

    this.bvh4_inputs[0] = bounds_buffer;
    this.bvh4_inputs[1] = this.bvh4_nodes_buffer;
    this.bvh4_inputs[2] = this.bvh4_build_state_buffer;
    this.bvh4_inputs[3] = this.bvh4_index_pairs_buffer;
    this.bvh4_inputs[4] = this.bvh4_prim_indices_buffer;
    this.bvh4_inputs[5] = this.bvh_info_buffer;
    this.bvh4_inputs[6] = this.bvh4_debug_watchdog_buffer;

    this.bvh4_outputs[0] = this.bvh4_nodes_buffer;
    this.bvh4_outputs[1] = this.bvh4_build_state_buffer;
    this.bvh4_outputs[2] = this.bvh4_index_pairs_buffer;
    this.bvh4_outputs[3] = this.bvh4_prim_indices_buffer;
    this.bvh4_outputs[4] = this.bvh4_debug_watchdog_buffer;

    const bvh4_workgroups = Math.max(1, Math.ceil(primitive_count / 32));
    ComputeTaskQueue.new_task(
      hploc_convert_parallel_single_pass_task_name,
      bvh4_processing_wgsl_path,
      this.bvh4_inputs,
      this.bvh4_outputs,
      bvh4_workgroups,
      1,
      1,
      convert_bvh2_to_bvh4_cs_entry_point
    );
  }

  build_dirty_meshes() {
    if (!MeshBLAS.dirty_meshes || MeshBLAS.dirty_meshes.size === 0) return;

    const blas_data = MeshBLAS.to_gpu_data();

    for (const mesh_id of MeshBLAS.dirty_meshes) {
      const meta = MeshBLAS.mesh_meta.get(mesh_id);
      if (!meta) continue;

      const leaf_count = meta.leaf_count >>> 0;
      if (!leaf_count || leaf_count > this.max_primitives) continue;

      MeshBLAS.leaf_bounds_uniforms[0] = mesh_id >>> 0;
      MeshBLAS.leaf_bounds_uniforms[1] = meta.base_node >>> 0;
      MeshBLAS.leaf_bounds_uniforms[2] = meta.first_vertex >>> 0;
      MeshBLAS.leaf_bounds_uniforms[3] = leaf_count >>> 0;
      blas_data.info_buffer.write(MeshBLAS.leaf_bounds_uniforms);

      const leaf_workgroups = Math.max(1, Math.ceil(leaf_count / 64));
      ComputeTaskQueue.new_task(
        `mesh_${mesh_id}_leaf_bounds`,
        "acceleration/blas_leaf_bounds.wgsl",
        [
          blas_data.nodes_buffer,
          blas_data.directory_buffer,
          blas_data.info_buffer,
          meta.index_buffer,
        ],
        [blas_data.nodes_buffer],
        leaf_workgroups,
        1,
        1,
        "write_leaf_bounds"
      );

      this.build(blas_data.nodes_buffer, leaf_count);
    }

    MeshBLAS.dirty_meshes.clear();
  }
}

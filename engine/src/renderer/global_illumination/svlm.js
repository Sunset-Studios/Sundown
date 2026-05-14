import { Buffer } from "../buffer.js";
import { RenderPassFlags } from "../renderer_types.js";
import { draw_quad } from "../draw_helpers.js";
import { npot, clamp, ceil_div } from "../../utility/math.js";

const PROBES_PER_BRICK = 64;
const NODE_U32_STRIDE = 16;
const LEAF_U32_STRIDE = 16;
const LINE_FLOAT_STRIDE = 20;
const LINES_PER_BOX = 12;
const PARAM_WORD_COUNT = 28;
const COUNTER_U32_COUNT = 56;
const THREADS_PER_GROUP = 128;
const PROBE_DEBUG_WORKGROUP_X = 8;
const PROBE_DEBUG_WORKGROUP_Y = 8;

// These offsets mirror the field order of SVLMParams in svlm_common.wgsl.
// SVLMParams is stored as f32 slots; shaders cast fields back to integer types
// at their use sites when a field represents a count, index, or enum.
const PARAM_WORLD_MIN_X = 0;
const PARAM_WORLD_MIN_Y = 1;
const PARAM_WORLD_MIN_Z = 2;
const PARAM_ROOT_SIZE = 3;
const PARAM_ROOT_DIM_X = 4;
const PARAM_ROOT_DIM_Y = 5;
const PARAM_ROOT_DIM_Z = 6;
const PARAM_MAX_LEVEL = 7;
const PARAM_MAX_NODES = 8;
const PARAM_MAX_LEAF_BRICKS = 9;
const PARAM_MIN_LEVEL = 10;
const PARAM_NEAR_FACTOR = 11;
const PARAM_OCC_MIN = 12;
const PARAM_OCC_MAX = 13;
const PARAM_REQUESTED_ROOT_SIZE = 14;
const PARAM_BAKE_PADDING = 15;
const PARAM_BAKE_SERIAL = 16;
const PARAM_ROOT_COUNT = 17;
const PARAM_SCENE_MIN_X = 18;
const PARAM_SCENE_MIN_Y = 19;
const PARAM_SCENE_MIN_Z = 20;
const PARAM_SCENE_MAX_X = 21;
const PARAM_SCENE_MAX_Y = 22;
const PARAM_SCENE_MAX_Z = 23;
const PARAM_DEBUG_LEVEL = 24;
const PARAM_DEBUG_LEAF_PAGE_GROUPS_Y = 25;
const PARAM_DEBUG_GATHER_PAGE_GROUPS_X = 26;
const PARAM_DEBUG_GATHER_PAGE_GROUPS_Y = 27;

const COUNTER_NODE_COUNT = 0;
const COUNTER_CURR_COUNT = 1;
const COUNTER_NEXT_COUNT = 2;
const COUNTER_LEAF_COUNT = 3;
const COUNTER_PROBE_COUNT = 4;
const COUNTER_STATUS = 5;
const COUNTER_MAX_LEVEL_REACHED = 6;
const COUNTER_SPLIT_BASE = 8;
const COUNTER_LEVEL_BASE = 24;

const STATUS_NODE_OVERFLOW = 1 << 0;
const STATUS_LEAF_OVERFLOW = 1 << 1;
const STATUS_ROOT_OVERFLOW = 1 << 2;

const MAX_COMPUTE_WORKGROUPS = 65535;
const MAX_STORAGE_BINDING_SIZE = 128 * 1024 * 1024;

// The SVLM pipeline is split into small compute passes.
// The general pipeline is: derive volume, seed roots, classify one frontier, then publish the next frontier.
const svlm_begin_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_begin.wgsl" },
  },
};

const svlm_seed_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_seed_roots.wgsl" },
  },
};

const svlm_classify_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_classify_level.wgsl" },
  },
};

const svlm_advance_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_advance_level.wgsl" },
  },
};

const svlm_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_debug_lines.wgsl" },
  },
};

const svlm_probe_debug_clear_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_probe_debug_clear.wgsl" },
  },
};

const svlm_probe_debug_gather_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_probe_debug_gather.wgsl" },
  },
};

const svlm_probe_debug_splat_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_probe_debug_splat.wgsl" },
  },
};

const svlm_probe_debug_resolve_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_probe_debug_resolve.wgsl" },
  },
};

/**
 * GPU-driven sparse volumetric lightmapper structure builder.
 *
 * This class does not bake irradiance yet. It owns the brick hierarchy and
 * probe allocation surface:
 * - derive a root brick grid from TLAS bounds
 * - classify candidate bricks against TLAS/BLAS data on the GPU
 * - emit leaf bricks with implicit 4x4x4 probe lattices
 * - provide debug brick/probe views and readback stats for the dev console
 */
export class SparseVolumetricLightmapper {
  config = {
    root_brick_size: 64.0,
    max_level: 5,
    min_level: 1,
    bake_padding: 2.0,
    near_geometry_factor: 0.75,
    occupancy_split_min: 0.01,
    occupancy_split_max: 0.65,
    max_nodes: 131072,
    max_leaf_bricks: 32768,
    auto_resize_growth: 2.0,
  };

  debug_config = {
    debug_level: -1,
  };

  stats = {
    baked: false,
    bake_serial: 0,
    root_dims: [0, 0, 0],
    root_brick_size: 0,
    min_level: 0,
    max_level: 0,
    max_level_reached: 0,
    node_count: 0,
    leaf_count: 0,
    probe_count: 0,
    probes_per_leaf: PROBES_PER_BRICK,
    per_level_counts: [],
    split_counts: [],
    world_min: [0, 0, 0],
    world_max: [0, 0, 0],
    scene_min: [0, 0, 0],
    scene_max: [0, 0, 0],
    node_bytes: 0,
    leaf_bytes: 0,
    total_bytes: 0,
    debug_leaf_count: 0,
    debug_level: -1,
    truncated_by_node_limit: false,
    truncated_by_leaf_limit: false,
  };

  bake_serial = 0;
  bake_requested = false;
  bake_in_flight = false;

  params_data = new Float32Array(PARAM_WORD_COUNT);
  params_buffer = null;

  counters_data = new Uint32Array(COUNTER_U32_COUNT);
  counter_buffer = null;

  node_buffer = null;
  curr_node_buffer = null;
  next_node_buffer = null;
  leaf_brick_buffer = null;
  debug_line_buffer = null;
  debug_texture = null;

  debug_lines_dirty = false;

  get_stats() {
    return this.stats;
  }

  bake(options = {}) {
    this.config = this._clamp_budget_config({ ...this.config, ...this._sanitize_options(options) });
    this.debug_config = this._sanitize_debug_options(options);
    this.bake_serial += 1;
    this.bake_requested = true;
    this.bake_in_flight = false;

    this.counters_data.fill(0);

    this._write_param_data();
    this._write_param_buffer();

    this._reset_stats();
    this._release_debug_line_buffer();

    return this.stats;
  }

  clear() {
    this.bake_requested = false;
    this.bake_in_flight = false;
    this.debug_texture = null;

    this.params_data.fill(0);
    this._write_param_data();
    this._write_param_buffer();

    this.counters_data.fill(0);
    this._write_counter_buffer();

    this._reset_stats();
    this._release_debug_line_buffer();
  }

  add_bake_passes(
    render_graph,
    {
      tlas_bvh_info,
      tlas_bvh2_nodes,
      entity_transforms,
      entity_index_lookup,
      blas_directory,
      blas_bvh2_nodes,
      force_recreate = false,
    }
  ) {
    this._refresh_stats_from_readback();

    if (!this.bake_requested) return;

    const node_words = Math.max(1, this.config.max_nodes * NODE_U32_STRIDE);
    const queue_words = Math.max(1, this.config.max_nodes);
    const leaf_words = Math.max(1, this.config.max_leaf_bricks * LEAF_U32_STRIDE);
    const max_dispatch_groups = ceil_div(this.config.max_nodes, THREADS_PER_GROUP);

    // Persistent build buffers. Sizes are budget-driven and may grow after a
    // readback reports overflow, but the bake itself remains fully GPU-driven.
    this.params_buffer = Buffer.create({
      name: "svlm_params",
      raw_data: this.params_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
      force: force_recreate || !this.params_buffer,
    });

    this.counter_buffer = Buffer.create({
      name: "svlm_counters",
      raw_data: this.counters_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
      force: force_recreate,
    });

    this.node_buffer = Buffer.create({
      name: "svlm_node_pool",
      size: node_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    this.curr_node_buffer = Buffer.create({
      name: "svlm_curr_nodes",
      size: queue_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    this.next_node_buffer = Buffer.create({
      name: "svlm_next_nodes",
      size: queue_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    this.leaf_brick_buffer = Buffer.create({
      name: "svlm_leaf_bricks",
      size: leaf_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const params = render_graph.register_buffer(this.params_buffer.config.name);
    const counters = render_graph.register_buffer(this.counter_buffer.config.name);
    const nodes = render_graph.register_buffer(this.node_buffer.config.name);
    const curr_nodes_a = render_graph.register_buffer(this.curr_node_buffer.config.name);
    const curr_nodes_b = render_graph.register_buffer(this.next_node_buffer.config.name);
    const leaves = render_graph.register_buffer(this.leaf_brick_buffer.config.name);

    this._write_param_buffer();

    // Pass 0: reset counters and derive the actual root grid from the TLAS root.
    render_graph.add_pass(
      `svlm_begin_${this.bake_serial}`,
      RenderPassFlags.Compute,
      {
        inputs: [params, counters, tlas_bvh_info, tlas_bvh2_nodes],
        outputs: [params, counters],
        shader_setup: svlm_begin_shader_setup,
      },
      (graph, frame_data) => {
        graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
      }
    );

    // Pass 1: write one node per root brick and initialize the current frontier.
    render_graph.add_pass(
      `svlm_seed_roots_${this.bake_serial}`,
      RenderPassFlags.Compute,
      {
        inputs: [params, counters, nodes, curr_nodes_a],
        outputs: [nodes, curr_nodes_a, counters],
        shader_setup: svlm_seed_shader_setup,
      },
      (graph, frame_data) => {
        graph.get_physical_pass(frame_data.current_pass).dispatch(max_dispatch_groups, 1, 1);
      }
    );

    let curr_nodes = curr_nodes_a;
    let next_nodes = curr_nodes_b;
    const levels = Math.max(0, Math.floor(this.config.max_level));
    for (let level = 0; level <= levels; level += 1) {
      // Pass 2N: classify this frontier. Each active node becomes invalid, a
      // leaf brick, or eight appended children in next_nodes.
      render_graph.add_pass(
        `svlm_classify_l${level}_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            params,
            counters,
            tlas_bvh_info,
            tlas_bvh2_nodes,
            entity_transforms,
            entity_index_lookup,
            blas_directory,
            blas_bvh2_nodes,
            nodes,
            curr_nodes,
            next_nodes,
            leaves,
          ],
          outputs: [counters, nodes, next_nodes, leaves],
          shader_setup: svlm_classify_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(max_dispatch_groups, 1, 1);
        }
      );

      // Pass 2N+1: publish next_count as curr_count. The CPU-side handle swap
      // below makes the next render-graph pass read from the freshly filled
      // queue without copying any GPU data.
      render_graph.add_pass(
        `svlm_advance_l${level}_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [counters],
          outputs: [counters],
          shader_setup: svlm_advance_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
        }
      );

      const tmp = curr_nodes;
      curr_nodes = next_nodes;
      next_nodes = tmp;
    }

    this.bake_requested = false;
    this.bake_in_flight = true;
    this.debug_lines_dirty = true;
    this.stats.baked = true;
  }

  add_debug_geometry_passes(
    render_graph,
    {
      main_albedo_image,
      main_smra_image,
      main_normal_image,
      main_motion_emissive_image,
      main_depth_image,
    }
  ) {
    if (!this.params_buffer || !this.counter_buffer || !this.leaf_brick_buffer) {
      return;
    }


    const debug_floats = Math.max(20, this.config.max_leaf_bricks * LINES_PER_BOX * LINE_FLOAT_STRIDE);
    const debug_line_count = this.config.max_leaf_bricks * LINES_PER_BOX;
    const max_debug_groups = ceil_div(this.config.max_leaf_bricks, THREADS_PER_GROUP);
    const needs_debug_line_update =
      this.debug_lines_dirty ||
      !this.debug_line_buffer ||
      this.debug_line_buffer.config.size < debug_floats * Float32Array.BYTES_PER_ELEMENT;

    const line_data = render_graph.register_buffer(this.debug_line_buffer.config.name);
    const params = render_graph.register_buffer(this.params_buffer.config.name);
    const counters = render_graph.register_buffer(this.counter_buffer.config.name);
    const leaves = render_graph.register_buffer(this.leaf_brick_buffer.config.name);

    this._write_param_buffer();

    // Line data is only allocated when the brick debug view is drawn. The bake
    // data itself is independent of this transient visualization buffer.
    this.debug_line_buffer = Buffer.create({
      name: "svlm_debug_line_data",
      size: debug_floats,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    if (needs_debug_line_update) {
      // Rebuild line records only when the hierarchy or debug level changes.
      render_graph.add_pass(
        `svlm_debug_lines_update_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters, leaves, line_data],
          outputs: [line_data, counters],
          shader_setup: svlm_debug_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(max_debug_groups, 1, 1);
        }
      );

      this.debug_lines_dirty = false;
    }

    render_graph.add_pass(
      "svlm_debug_brick_lines",
      RenderPassFlags.Graphics,
      {
        inputs: [line_data],
        outputs: [
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
        ],
        shader_setup: {
          pipeline_shaders: {
            vertex: { path: "line.wgsl" },
            fragment: { path: "line.wgsl" },
          },
          rasterizer_state: {
            cull_mode: "none",
          },
        },
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        draw_quad(pass, debug_line_count);
      }
    );
  }

  add_probe_debug_passes(
    render_graph,
    width,
    height,
    depth_texture,
    scene_color,
    force_recreate = false
  ) {
    this.debug_texture = null;

    if (!this.params_buffer || !this.leaf_brick_buffer || !this.counter_buffer) {
      return;
    }

    const max_dispatch_leaf_count = MAX_COMPUTE_WORKGROUPS * MAX_COMPUTE_WORKGROUPS * PROBE_DEBUG_WORKGROUP_Y;
    const source_leaf_count = Math.min(this.config.max_leaf_bricks, max_dispatch_leaf_count);
    if (source_leaf_count <= 0) {
      return;
    }

    const leaf_group_count = Math.ceil(source_leaf_count / PROBE_DEBUG_WORKGROUP_Y);
    const leaf_page_groups_y = Math.max(1, Math.min(leaf_group_count, MAX_COMPUTE_WORKGROUPS));
    const leaf_page_count_z = Math.ceil(leaf_group_count / leaf_page_groups_y);
    const gather_group_count = ceil_div(source_leaf_count, THREADS_PER_GROUP);
    const gather_page_groups_x = Math.max(1, Math.min(gather_group_count, MAX_COMPUTE_WORKGROUPS));
    const gather_remaining_groups = Math.ceil(gather_group_count / gather_page_groups_x);
    const gather_page_groups_y = Math.max(1, Math.min(gather_remaining_groups, MAX_COMPUTE_WORKGROUPS));
    const gather_page_count_z = Math.ceil(gather_remaining_groups / gather_page_groups_y);

    // Probe debug scans and splats in paged 3D dispatches so large leaf budgets
    // never exceed maxComputeWorkgroupsPerDimension on a single axis.
    this.params_data[PARAM_DEBUG_LEAF_PAGE_GROUPS_Y] = leaf_page_groups_y;
    this.params_data[PARAM_DEBUG_GATHER_PAGE_GROUPS_X] = gather_page_groups_x;
    this.params_data[PARAM_DEBUG_GATHER_PAGE_GROUPS_Y] = gather_page_groups_y;
    this._write_param_buffer();

    const params = render_graph.register_buffer(this.params_buffer.config.name);
    const counters = render_graph.register_buffer(this.counter_buffer.config.name);
    const leaves = render_graph.register_buffer(this.leaf_brick_buffer.config.name);

    // One packed u32 per pixel stores closest probe depth plus compact shade and
    // level-color data before resolve composites probe spheres over scene color.
    const debug_depth = render_graph.create_buffer({
      name: "svlm_probe_debug_depth",
      size: Math.max(1, width * height),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const debug_leaf_indices = render_graph.create_buffer({
      name: "svlm_probe_debug_leaf_indices",
      size: Math.max(2, source_leaf_count + 1),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.debug_texture = render_graph.create_image({
      name: "svlm_probe_debug",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    render_graph.add_pass(
      "svlm_probe_debug_clear",
      RenderPassFlags.Compute,
      {
        inputs: [debug_depth, depth_texture, debug_leaf_indices],
        outputs: [debug_depth, debug_leaf_indices],
        shader_setup: svlm_probe_debug_clear_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      "svlm_probe_debug_gather",
      RenderPassFlags.Compute,
      {
        inputs: [params, counters, leaves, debug_leaf_indices],
        outputs: [debug_leaf_indices],
        shader_setup: svlm_probe_debug_gather_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(gather_page_groups_x, gather_page_groups_y, gather_page_count_z);
      }
    );

    render_graph.add_pass(
      "svlm_probe_debug_splat",
      RenderPassFlags.Compute,
      {
        inputs: [params, counters, leaves, depth_texture, debug_depth, debug_leaf_indices],
        outputs: [debug_depth],
        shader_setup: svlm_probe_debug_splat_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(PROBES_PER_BRICK / PROBE_DEBUG_WORKGROUP_X),
          leaf_page_groups_y,
          leaf_page_count_z
        );
      }
    );

    render_graph.add_pass(
      "svlm_probe_debug_resolve",
      RenderPassFlags.Compute,
      {
        inputs: [scene_color, debug_depth, this.debug_texture],
        outputs: [this.debug_texture],
        shader_setup: svlm_probe_debug_resolve_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }

  _sanitize_options(options) {
    const out = {};
    if (options.root_brick_size !== undefined) {
      out.root_brick_size = Math.max(0.0, Number(options.root_brick_size) || 0.0);
    }
    if (options.max_level !== undefined) {
      out.max_level = clamp(Math.floor(Number(options.max_level)), 0, 8);
    }
    if (options.min_level !== undefined) {
      out.min_level = clamp(Math.floor(Number(options.min_level)), 0, out.max_level ?? this.config.max_level);
    }
    if (options.max_leaf_bricks !== undefined) {
      out.max_leaf_bricks = Math.max(1, Math.floor(Number(options.max_leaf_bricks)));
    }
    if (options.max_nodes !== undefined) {
      out.max_nodes = Math.max(9, Math.floor(Number(options.max_nodes)));
    }
    return out;
  }

  _clamp_budget_config(config) {
    const limits = this._get_budget_limits();
    const max_leaf_bricks = clamp(
      Math.floor(Number(config.max_leaf_bricks) || this.config.max_leaf_bricks),
      1,
      limits.max_leaf_bricks
    );
    const max_nodes = clamp(
      Math.floor(Number(config.max_nodes) || this.config.max_nodes),
      9,
      limits.max_nodes
    );
    return {
      ...config,
      max_nodes,
      max_leaf_bricks,
    };
  }

  _get_budget_limits() {
    const max_buffer_bytes = Math.max(4, MAX_STORAGE_BINDING_SIZE);
    const max_nodes_by_storage = Math.floor(max_buffer_bytes / (NODE_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT));
    const max_nodes_by_dispatch = MAX_COMPUTE_WORKGROUPS * THREADS_PER_GROUP;
    const max_leaf_by_leaf_buffer = Math.floor(max_buffer_bytes / (LEAF_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT));

    return {
      max_nodes: Math.max(9, Math.min(max_nodes_by_storage, max_nodes_by_dispatch)),
      max_leaf_bricks: Math.max(1, max_leaf_by_leaf_buffer),
    };
  }

  _sanitize_debug_options(options) {
    const out = {};
    if (options.debug_level !== undefined) {
      const level = Number(options.debug_level);
      out.debug_level = Number.isFinite(level) ? clamp(Math.floor(level), -1, this.config.max_level) : -1;
    }
    return out;
  }

  _write_param_data() {
    this.params_data.fill(0);
    // Fields written by the GPU begin pass, such as world_min/root_dims, are
    // cleared here and then populated after TLAS bounds are known on-GPU.
    this.params_data[PARAM_MAX_LEVEL] = this.config.max_level;
    this.params_data[PARAM_MAX_NODES] = this.config.max_nodes;
    this.params_data[PARAM_MAX_LEAF_BRICKS] = this.config.max_leaf_bricks;
    this.params_data[PARAM_MIN_LEVEL] = this.config.min_level;
    this.params_data[PARAM_NEAR_FACTOR] = this.config.near_geometry_factor;
    this.params_data[PARAM_OCC_MIN] = this.config.occupancy_split_min;
    this.params_data[PARAM_OCC_MAX] = this.config.occupancy_split_max;
    this.params_data[PARAM_REQUESTED_ROOT_SIZE] = this.config.root_brick_size;
    this.params_data[PARAM_BAKE_PADDING] = this.config.bake_padding;
    this.params_data[PARAM_BAKE_SERIAL] = this.bake_serial;
    this.params_data[PARAM_DEBUG_LEVEL] = this.debug_config.debug_level;
    this.params_data[PARAM_DEBUG_LEAF_PAGE_GROUPS_Y] = 0;
    this.params_data[PARAM_DEBUG_GATHER_PAGE_GROUPS_X] = 0;
    this.params_data[PARAM_DEBUG_GATHER_PAGE_GROUPS_Y] = 0;
  }

  _write_param_buffer() {
    if (this.params_buffer) {
      this.params_buffer.write_raw(this.params_data);
    }
  }

  _write_counter_buffer() {
    if (this.counter_buffer) {
      this.counter_buffer.write_raw(this.counters_data);
    }
  }

  _refresh_stats_from_readback() {
    const node_count = this.counters_data[COUNTER_NODE_COUNT] || 0;
    const leaf_count = this.counters_data[COUNTER_LEAF_COUNT] || 0;
    const probe_count = Math.min(this.counters_data[COUNTER_PROBE_COUNT] || 0, this.config.max_leaf_bricks * PROBES_PER_BRICK);
    const status = this.counters_data[COUNTER_STATUS] || 0;
    const root_dims = [
      Math.floor(this.params_data[PARAM_ROOT_DIM_X] || 0),
      Math.floor(this.params_data[PARAM_ROOT_DIM_Y] || 0),
      Math.floor(this.params_data[PARAM_ROOT_DIM_Z] || 0),
    ];
    const root_brick_size = this.params_data[PARAM_ROOT_SIZE] || 0;
    const world_min = [
      this.params_data[PARAM_WORLD_MIN_X] || 0,
      this.params_data[PARAM_WORLD_MIN_Y] || 0,
      this.params_data[PARAM_WORLD_MIN_Z] || 0,
    ];
    const world_max = [
      world_min[0] + root_dims[0] * root_brick_size,
      world_min[1] + root_dims[1] * root_brick_size,
      world_min[2] + root_dims[2] * root_brick_size,
    ];
    const scene_min = [
      this.params_data[PARAM_SCENE_MIN_X] || 0,
      this.params_data[PARAM_SCENE_MIN_Y] || 0,
      this.params_data[PARAM_SCENE_MIN_Z] || 0,
    ];
    const scene_max = [
      this.params_data[PARAM_SCENE_MAX_X] || 0,
      this.params_data[PARAM_SCENE_MAX_Y] || 0,
      this.params_data[PARAM_SCENE_MAX_Z] || 0,
    ];
    const per_level_counts = [];
    const split_counts = [];
    // Level histograms are counters written during classification. They make it
    // easy to see whether refinement is clustering where geometry actually is.
    for (let i = 0; i <= this.config.max_level; i += 1) {
      per_level_counts.push(this.counters_data[COUNTER_LEVEL_BASE + i] || 0);
      split_counts.push(this.counters_data[COUNTER_SPLIT_BASE + i] || 0);
    }

    const node_bytes = node_count * NODE_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT;
    const leaf_bytes = leaf_count * LEAF_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT;

    this.stats.baked = this.bake_in_flight || leaf_count > 0;
    this.stats.bake_serial = this.bake_serial;
    this.stats.root_dims = root_dims;
    this.stats.root_brick_size = root_brick_size;
    this.stats.min_level = this.config.min_level;
    this.stats.max_level = this.config.max_level;
    this.stats.max_level_reached = this.counters_data[COUNTER_MAX_LEVEL_REACHED] || 0;
    this.stats.node_count = node_count;
    this.stats.leaf_count = leaf_count;
    this.stats.probe_count = probe_count;
    this.stats.probes_per_leaf = PROBES_PER_BRICK;
    this.stats.per_level_counts = per_level_counts;
    this.stats.split_counts = split_counts;
    this.stats.world_min = world_min;
    this.stats.world_max = world_max;
    this.stats.scene_min = scene_min;
    this.stats.scene_max = scene_max;
    this.stats.node_bytes = node_bytes;
    this.stats.leaf_bytes = leaf_bytes;
    this.stats.total_bytes = node_bytes + leaf_bytes;
    this.stats.debug_leaf_count = Math.min(leaf_count, this.config.max_leaf_bricks);
    this.stats.debug_level = this.debug_config.debug_level;
    this.stats.truncated_by_node_limit = (status & (STATUS_NODE_OVERFLOW | STATUS_ROOT_OVERFLOW)) !== 0;
    this.stats.truncated_by_leaf_limit = (status & STATUS_LEAF_OVERFLOW) !== 0;
    this.stats.config = { ...this.config };

    this._maybe_queue_auto_resize_rebake({
      node_count,
      leaf_count,
      status,
    });
  }

  _maybe_queue_auto_resize_rebake({ node_count, leaf_count, status }) {
    if (this.bake_requested || !this.bake_in_flight) {
      return false;
    }

    const node_overflow = (status & (STATUS_NODE_OVERFLOW | STATUS_ROOT_OVERFLOW)) !== 0;
    const leaf_overflow = (status & STATUS_LEAF_OVERFLOW) !== 0;
    if (!node_overflow && !leaf_overflow) {
      return false;
    }

    // Auto-resize is the default fallback: if the GPU build proves a budget was
    // too small, grow within device limits and queue another GPU bake instead of
    // leaving the user to guess the required node/leaf counts.
    const limits = this._get_budget_limits();
    const growth = Math.max(1.25, Number(this.config.auto_resize_growth) || 2.0);
    let next_nodes = this.config.max_nodes;
    let next_leaf_bricks = this.config.max_leaf_bricks;

    if (node_overflow) {
      const required_nodes = Math.max(node_count + 8, Math.ceil(this.config.max_nodes * growth));
      next_nodes = Math.min(limits.max_nodes, npot(required_nodes));
    }

    if (leaf_overflow) {
      const required_leaves = Math.max(leaf_count + 1, Math.ceil(this.config.max_leaf_bricks * growth));
      next_leaf_bricks = Math.min(limits.max_leaf_bricks, npot(required_leaves));
    }

    const can_grow =
      next_nodes > this.config.max_nodes ||
      next_leaf_bricks > this.config.max_leaf_bricks;

    if (!can_grow) {
      return false;
    }

    this.bake({
      max_nodes: next_nodes,
      max_leaf_bricks: next_leaf_bricks,
      _auto_resize: true,
    });
    return true;
  }

  _release_debug_line_buffer() {
    if (!this.debug_line_buffer) {
      return;
    }
    this.debug_line_buffer.destroy();
    this.debug_line_buffer = null;
  }

  _reset_stats() {
    this.stats.baked = false;
    this.stats.bake_serial = this.bake_serial;
    this.stats.root_dims = [0, 0, 0];
    this.stats.root_brick_size = 0;
    this.stats.min_level = this.config.min_level;
    this.stats.max_level = this.config.max_level;
    this.stats.max_level_reached = 0;
    this.stats.node_count = 0;
    this.stats.leaf_count = 0;
    this.stats.probe_count = 0;
    this.stats.probes_per_leaf = PROBES_PER_BRICK;
    this.stats.per_level_counts = [];
    this.stats.split_counts = [];
    this.stats.world_min = [0, 0, 0];
    this.stats.world_max = [0, 0, 0];
    this.stats.scene_min = [0, 0, 0];
    this.stats.scene_max = [0, 0, 0];
    this.stats.node_bytes = 0;
    this.stats.leaf_bytes = 0;
    this.stats.total_bytes = 0;
    this.stats.debug_leaf_count = 0;
    this.stats.debug_level = this.debug_config.debug_level;
    this.stats.truncated_by_node_limit = false;
    this.stats.truncated_by_leaf_limit = false;
    this.stats.config = { ...this.config };
  }
}

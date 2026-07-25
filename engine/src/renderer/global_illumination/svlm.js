import { Buffer } from "../buffer.js";
import { RenderPassFlags } from "../renderer_types.js";
import { draw_quad } from "../draw_helpers.js";
import { npot, clamp, ceil_div } from "../../utility/math.js";
import {
  register_material_buffers,
  register_scene_lighting_data,
  register_texture_pools,
} from "../render_graph_utils.js";

const PROBES_PER_BRICK = 64;
const SH_WORDS_PER_PROBE = 6;
const PROBE_RAY_U32_STRIDE = 24;
const NODE_U32_STRIDE = 7;
const LEAF_U32_STRIDE = 6;
const LINE_FLOAT_STRIDE = 20;
const LINES_PER_BOX = 12;
const PARAM_WORD_COUNT = 34;
const COUNTER_U32_COUNT = 44;
const THREADS_PER_GROUP = 128;
const PROBE_DEBUG_WORKGROUP_X = 8;
const PROBE_DEBUG_WORKGROUP_Y = 8;
const BAKE_FORMAT_VERSION = 4;

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
const PARAM_LEAF_CAPACITY = 9;
const PARAM_MIN_LEVEL = 10;
const PARAM_NEAR_FACTOR = 11;
const PARAM_OCC_MIN = 12;
const PARAM_OCC_MAX = 13;
const PARAM_REQUESTED_ROOT_SIZE = 14;
const PARAM_BAKE_PADDING = 15;
const PARAM_BAKE_SERIAL = 16;
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
const PARAM_IRRADIANCE_RAYS_PER_PROBE = 28;
const PARAM_IRRADIANCE_PROBES_PER_BATCH = 29;
const PARAM_IRRADIANCE_SAMPLE_COUNT = 30;
const PARAM_IRRADIANCE_MAX_RAY_DISTANCE = 31;
const PARAM_IRRADIANCE_FORMAT_VERSION = 32;
const PARAM_IRRADIANCE_SH_WORDS_PER_PROBE = 33;

const COUNTER_NODE_COUNT = 0;
const COUNTER_LEAF_COUNT = 3;
const COUNTER_PROBE_COUNT = 4;
const COUNTER_STATUS = 5;
const COUNTER_MAX_LEVEL_REACHED = 6;
const COUNTER_SPLIT_BASE = 8;
const COUNTER_LEVEL_BASE = 24;
const COUNTER_IRRADIANCE_SAMPLE_INDEX = 41;
const COUNTER_IRRADIANCE_COMPLETED_PROBE_SAMPLES = 42;
const COUNTER_IRRADIANCE_STATUS = 43;

const STATUS_NODE_OVERFLOW = 1 << 0;
const STATUS_LEAF_OVERFLOW = 1 << 1;
const STATUS_ROOT_OVERFLOW = 1 << 2;
const IRRADIANCE_STATUS_ALLOCATION_COMPLETE = 1 << 1;

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

const svlm_finish_allocation_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_finish_allocation.wgsl" },
  },
};

const svlm_irradiance_trace_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_trace_init.wgsl" },
  },
};

const svlm_irradiance_trace_hit_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_trace_hit.wgsl" },
  },
};

const svlm_irradiance_trace_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_trace_shade.wgsl" },
  },
};

const svlm_irradiance_accumulate_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_accumulate.wgsl" },
  },
};

const svlm_irradiance_advance_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_advance.wgsl" },
  },
};

const compact_emissive_lights_shader_setup = {
  pipeline_shaders: {
    compute: { path: "system_compute/compact_emissive_lights.wgsl" },
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
 * This class owns the brick hierarchy and persistent baked probe payload:
 * - derive a root brick grid from TLAS bounds
 * - classify candidate bricks against TLAS/BLAS data on the GPU
 * - emit leaf bricks with implicit 4x4x4 probe lattices
 * - progressively trace and accumulate packed L1 RGB SH at every probe
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
    auto_resize_growth: 2.0,
    irradiance_rays_per_probe: 1024,
    irradiance_probes_per_batch: 2048,
    irradiance_sample_count: 1,
    irradiance_max_ray_distance: 100000.0,
    max_emissive_lights: 32768,
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
    leaf_capacity: 0,
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
    irradiance_bytes: 0,
    irradiance_allocated_bytes: 0,
    total_bytes: 0,
    irradiance_usable: false,
    irradiance_ready: false,
    irradiance_in_progress: false,
    irradiance_allocation_pending: false,
    irradiance_capacity_exceeded: false,
    irradiance_progress: 0,
    irradiance_sample_index: 0,
    irradiance_sample_count: 0,
    irradiance_completed_probe_samples: 0,
    irradiance_required_probe_samples: 0,
    irradiance_rays_per_probe: 0,
    debug_leaf_count: 0,
    debug_level: -1,
    truncated_by_node_limit: false,
    truncated_by_leaf_limit: false,
  };

  bake_serial = 0;
  bake_requested = false;
  bake_in_flight = false;
  irradiance_bake_in_flight = false;
  irradiance_allocation_pending = false;
  irradiance_probe_capacity = 0;

  params_data = new Float32Array(PARAM_WORD_COUNT);
  params_buffer = null;

  counters_data = new Uint32Array(COUNTER_U32_COUNT);
  counter_buffer = null;

  node_buffer = null;
  curr_node_buffer = null;
  next_node_buffer = null;
  leaf_brick_buffer = null;
  irradiance_buffer = null;
  irradiance_ray_buffer = null;
  emissive_light_buffer = null;
  debug_line_buffer = null;
  debug_texture = null;

  debug_lines_dirty = false;

  get_stats() {
    return this.stats;
  }

  /**
   * Returns the GPU-resident bake payload and stable layout metadata without
   * performing readback or disk I/O. All durable buffers include COPY_SRC and
   * COPY_DST so a future serializer/loader can stage this exact payload.
   */
  get_bake_artifact() {
    return {
      format: "sundown-svlm",
      version: BAKE_FORMAT_VERSION,
      usable: this.stats.irradiance_usable,
      ready: this.stats.irradiance_ready,
      layout: {
        params_word_count: PARAM_WORD_COUNT,
        node_words_per_record: NODE_U32_STRIDE,
        leaf_words_per_record: LEAF_U32_STRIDE,
        probes_per_leaf: PROBES_PER_BRICK,
        irradiance_encoding: "sh-l1-rgb-f16",
        irradiance_words_per_probe: SH_WORDS_PER_PROBE,
      },
      metadata: {
        bake_serial: this.bake_serial,
        node_count: this.stats.node_count,
        leaf_count: this.stats.leaf_count,
        probe_count: this.stats.probe_count,
        completed_probe_samples: this.stats.irradiance_completed_probe_samples,
        irradiance_probe_capacity: this.irradiance_probe_capacity,
        irradiance_sample_count: this.config.irradiance_sample_count,
        world_min: [...this.stats.world_min],
        world_max: [...this.stats.world_max],
      },
      buffers: {
        params: this.params_buffer,
        nodes: this.node_buffer,
        leaf_bricks: this.leaf_brick_buffer,
        irradiance: this.irradiance_buffer,
      },
    };
  }

  bake(options = {}) {
    this.config = this._clamp_budget_config({ ...this.config, ...this._sanitize_options(options) });
    this.debug_config = this._sanitize_debug_options(options);
    this.bake_serial += 1;
    this.bake_requested = true;
    this.bake_in_flight = false;
    this.irradiance_bake_in_flight = false;
    this.irradiance_allocation_pending = false;
    this.irradiance_probe_capacity = 0;

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
    this.irradiance_bake_in_flight = false;
    this.irradiance_allocation_pending = false;
    this.irradiance_probe_capacity = 0;
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
      index_buffer,
      dense_lights,
      force_recreate = false,
    }
  ) {
    this._refresh_stats_from_readback();

    const build_hierarchy = this.bake_requested;
    const allocation_complete =
      (this.counters_data[COUNTER_IRRADIANCE_STATUS] & IRRADIANCE_STATUS_ALLOCATION_COMPLETE) !== 0;
    let start_irradiance = false;

    if (!build_hierarchy && this.irradiance_allocation_pending && allocation_complete) {
      const realized_probe_count = Math.min(
        this.counters_data[COUNTER_PROBE_COUNT] || 0,
        this.config.max_nodes * PROBES_PER_BRICK
      );
      const max_irradiance_probes = Math.floor(
        MAX_STORAGE_BINDING_SIZE / (SH_WORDS_PER_PROBE * Uint32Array.BYTES_PER_ELEMENT)
      );

      this.irradiance_allocation_pending = false;
      if (realized_probe_count <= max_irradiance_probes) {
        this.irradiance_probe_capacity = realized_probe_count;
        this.irradiance_bake_in_flight = true;
        start_irradiance = true;
      } else {
        this.irradiance_probe_capacity = 0;
        this.stats.irradiance_capacity_exceeded = true;
      }
    }

    const gather_irradiance = this.irradiance_bake_in_flight;
    if (!build_hierarchy && !gather_irradiance) return;

    const node_words = Math.max(1, this.config.max_nodes * NODE_U32_STRIDE);
    const queue_words = Math.max(1, this.config.max_nodes);
    const leaf_words = Math.max(1, this.config.max_nodes * LEAF_U32_STRIDE);
    const rays_per_probe = this.config.irradiance_rays_per_probe;
    const probes_per_batch = this.config.irradiance_probes_per_batch;
    const ray_words = Math.max(4, 4 + probes_per_batch * rays_per_probe * PROBE_RAY_U32_STRIDE);
    const emissive_words = Math.max(4, 4 + this.config.max_emissive_lights * 12);
    const max_dispatch_groups = ceil_div(this.config.max_nodes, THREADS_PER_GROUP);

    // Persistent hierarchy and irradiance buffers are the in-memory bake
    // artifact. COPY_SRC/COPY_DST are intentional future save/load seams.
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
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      force: force_recreate,
    });

    const params = render_graph.register_buffer(this.params_buffer.config.name);
    const counters = render_graph.register_buffer(this.counter_buffer.config.name);
    const nodes = render_graph.register_buffer(this.node_buffer.config.name);
    const curr_nodes_a = render_graph.register_buffer(this.curr_node_buffer.config.name);
    const curr_nodes_b = render_graph.register_buffer(this.next_node_buffer.config.name);
    const leaves = render_graph.register_buffer(this.leaf_brick_buffer.config.name);
    let irradiance = null;
    let ray_data = null;
    let emissive_lights = null;
    let materials = null;
    let textures = null;
    let lighting = null;

    if (gather_irradiance) {
      const irradiance_words = Math.max(
        SH_WORDS_PER_PROBE,
        this.irradiance_probe_capacity * SH_WORDS_PER_PROBE
      );
      this.irradiance_buffer = Buffer.create({
        name: "svlm_probe_irradiance",
        size: irradiance_words,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        // Recreate once at the hierarchy handoff so smaller bakes release the
        // previous allocation instead of retaining a high-water-mark buffer.
        force: force_recreate || start_irradiance,
      });

      this.irradiance_ray_buffer = Buffer.create({
        name: "svlm_irradiance_ray_data",
        size: ray_words,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: force_recreate,
      });

      this.emissive_light_buffer = Buffer.create({
        name: "svlm_emissive_lights",
        size: emissive_words,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: force_recreate,
      });

      irradiance = render_graph.register_buffer(this.irradiance_buffer.config.name);
      ray_data = render_graph.register_buffer(this.irradiance_ray_buffer.config.name);
      emissive_lights = render_graph.register_buffer(this.emissive_light_buffer.config.name);
      materials = register_material_buffers(render_graph);
      textures = register_texture_pools(render_graph);
      lighting = register_scene_lighting_data(render_graph);
    }

    if (build_hierarchy) {
      this._write_param_buffer();
    }

    if (build_hierarchy) {
      // Reset counters and derive the actual root grid from the TLAS root.
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

      // Write one node per root brick and initialize the current frontier.
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
        // Each active node becomes a leaf brick or eight appended children.
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

      render_graph.add_pass(
        `svlm_finish_allocation_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [counters],
          outputs: [counters],
          shader_setup: svlm_finish_allocation_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
        }
      );
    }

    if (start_irradiance) {
      // Build the persistent light list only after the realized probe count has
      // been read back and the exact-size irradiance buffer exists.
      render_graph.add_pass(
        `svlm_compact_emissive_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            tlas_bvh2_nodes,
            tlas_bvh_info,
            blas_directory,
            index_buffer,
            entity_transforms,
            materials.params_gpu_buffer,
            materials.material_offsets_buffer,
            materials.material_palette_buffer,
            entity_index_lookup,
            emissive_lights,
            textures.albedo,
            textures.emission,
          ],
          outputs: [emissive_lights],
          shader_setup: compact_emissive_lights_shader_setup,
        },
        (graph, frame_data) => {
          const bounds_buffer = graph.get_physical_buffer(tlas_bvh2_nodes);
          graph.get_physical_buffer(emissive_lights).write_raw(new Uint32Array([0, 0, 0, 0]), 0);
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(
              Math.ceil(Math.floor(bounds_buffer.config.size / 32) / THREADS_PER_GROUP),
              1,
              1
            );
        }
      );
    }

    if (gather_irradiance) {
      // Irradiance is intentionally progressive: one bounded ray batch is
      // reused each frame while packed SH coefficients remain persistent. The
      // GPU owns the cursor and sample index after this one readback handoff.
      render_graph.add_pass(
        `svlm_irradiance_trace_init_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters, leaves, ray_data],
          outputs: [ray_data],
          shader_setup: svlm_irradiance_trace_init_shader_setup,
        },
        (graph, frame_data) => {
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(
              ceil_div(probes_per_batch * rays_per_probe, THREADS_PER_GROUP),
              1,
              1
            );
        }
      );

      render_graph.add_pass(
        `svlm_irradiance_trace_hit_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            ray_data,
            tlas_bvh2_nodes,
            tlas_bvh_info,
            blas_bvh2_nodes,
            blas_directory,
            entity_transforms,
            index_buffer,
            dense_lights,
            emissive_lights,
            entity_index_lookup,
          ],
          outputs: [ray_data],
          shader_setup: svlm_irradiance_trace_hit_shader_setup,
        },
        (graph, frame_data) => {
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(ceil_div(probes_per_batch * rays_per_probe, THREADS_PER_GROUP), 1, 1);
        }
      );

      render_graph.add_pass(
        `svlm_irradiance_trace_shade_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            lighting.scene_lighting_buffer,
            ray_data,
            materials.params_gpu_buffer,
            materials.material_offsets_buffer,
            materials.material_palette_buffer,
            entity_index_lookup,
            textures.albedo,
            textures.emission,
            lighting.skybox_image,
          ],
          outputs: [ray_data],
          shader_setup: svlm_irradiance_trace_shade_shader_setup,
        },
        (graph, frame_data) => {
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(ceil_div(probes_per_batch * rays_per_probe, THREADS_PER_GROUP), 1, 1);
        }
      );

      render_graph.add_pass(
        `svlm_irradiance_accumulate_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters, ray_data, irradiance],
          outputs: [counters, irradiance],
          shader_setup: svlm_irradiance_accumulate_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(probes_per_batch, 1, 1);
        }
      );

      render_graph.add_pass(
        `svlm_irradiance_advance_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters],
          outputs: [counters],
          shader_setup: svlm_irradiance_advance_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
        }
      );
    }

    if (build_hierarchy) {
      this.bake_requested = false;
      this.bake_in_flight = true;
      this.irradiance_allocation_pending = true;
      this.irradiance_bake_in_flight = false;
      this.debug_lines_dirty = true;
      this.stats.baked = true;
      this.stats.irradiance_in_progress = false;
    }
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

    const debug_leaf_count = Math.max(0, this.stats.leaf_count);
    if (debug_leaf_count === 0) {
      return;
    }
    const debug_floats = Math.max(
      20,
      debug_leaf_count * LINES_PER_BOX * LINE_FLOAT_STRIDE
    );
    const debug_line_count = debug_leaf_count * LINES_PER_BOX;
    const max_debug_groups = ceil_div(debug_leaf_count, THREADS_PER_GROUP);
    const needs_debug_line_update =
      this.debug_lines_dirty ||
      !this.debug_line_buffer ||
      this.debug_line_buffer.config.size < debug_floats * Float32Array.BYTES_PER_ELEMENT;

    // Line data is only allocated when the brick debug view is drawn. The bake
    // data itself is independent of this transient visualization buffer.
    this.debug_line_buffer = Buffer.create({
      name: "svlm_debug_line_data",
      size: debug_floats,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const line_data = render_graph.register_buffer(this.debug_line_buffer.config.name);
    const params = render_graph.register_buffer(this.params_buffer.config.name);
    const counters = render_graph.register_buffer(this.counter_buffer.config.name);
    const leaves = render_graph.register_buffer(this.leaf_brick_buffer.config.name);

    this._write_param_buffer();

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

    if (
      !this.params_buffer ||
      !this.leaf_brick_buffer ||
      !this.counter_buffer ||
      !this.irradiance_buffer ||
      this.irradiance_probe_capacity <= 0
    ) {
      return;
    }

    const max_dispatch_leaf_count =
      MAX_COMPUTE_WORKGROUPS * MAX_COMPUTE_WORKGROUPS * PROBE_DEBUG_WORKGROUP_Y;
    const source_leaf_count = Math.min(this.stats.leaf_count, max_dispatch_leaf_count);
    if (source_leaf_count <= 0) {
      return;
    }

    const leaf_group_count = Math.ceil(source_leaf_count / PROBE_DEBUG_WORKGROUP_Y);
    const leaf_page_groups_y = Math.max(1, Math.min(leaf_group_count, MAX_COMPUTE_WORKGROUPS));
    const leaf_page_count_z = Math.ceil(leaf_group_count / leaf_page_groups_y);
    const gather_group_count = ceil_div(source_leaf_count, THREADS_PER_GROUP);
    const gather_page_groups_x = Math.max(1, Math.min(gather_group_count, MAX_COMPUTE_WORKGROUPS));
    const gather_remaining_groups = Math.ceil(gather_group_count / gather_page_groups_x);
    const gather_page_groups_y = Math.max(
      1,
      Math.min(gather_remaining_groups, MAX_COMPUTE_WORKGROUPS)
    );
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
    const irradiance = render_graph.register_buffer(this.irradiance_buffer.config.name);

    // One packed u32 per pixel stores closest probe depth plus compact baked
    // irradiance color before resolve composites probe spheres over scene color.
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
        inputs: [
          params,
          counters,
          leaves,
          depth_texture,
          debug_depth,
          debug_leaf_indices,
          irradiance,
        ],
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
      out.min_level = clamp(
        Math.floor(Number(options.min_level)),
        0,
        out.max_level ?? this.config.max_level
      );
    }
    if (options.max_nodes !== undefined) {
      out.max_nodes = Math.max(9, Math.floor(Number(options.max_nodes)));
    }
    if (options.irradiance_rays_per_probe !== undefined) {
      out.irradiance_rays_per_probe = Math.max(
        1,
        Math.floor(Number(options.irradiance_rays_per_probe))
      );
    }
    if (options.irradiance_probes_per_batch !== undefined) {
      out.irradiance_probes_per_batch = Math.max(
        1,
        Math.floor(Number(options.irradiance_probes_per_batch))
      );
    }
    if (options.irradiance_sample_count !== undefined) {
      out.irradiance_sample_count = Math.max(
        1,
        Math.floor(Number(options.irradiance_sample_count))
      );
    }
    if (options.irradiance_max_ray_distance !== undefined) {
      out.irradiance_max_ray_distance = Math.max(
        1.0,
        Number(options.irradiance_max_ray_distance) || 1.0
      );
    }
    if (options.max_emissive_lights !== undefined) {
      out.max_emissive_lights = Math.max(1, Math.floor(Number(options.max_emissive_lights)));
    }
    return out;
  }

  _clamp_budget_config(config) {
    const limits = this._get_budget_limits();
    const max_nodes = clamp(
      Math.floor(Number(config.max_nodes) || this.config.max_nodes),
      9,
      limits.max_nodes
    );
    const max_rays_per_probe = Math.max(
      1,
      Math.floor(
        (MAX_STORAGE_BINDING_SIZE - 4 * Uint32Array.BYTES_PER_ELEMENT) /
          (PROBE_RAY_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
      )
    );
    const irradiance_rays_per_probe = clamp(
      Math.floor(Number(config.irradiance_rays_per_probe) || this.config.irradiance_rays_per_probe),
      1,
      max_rays_per_probe
    );
    const max_batch_by_ray_buffer = Math.max(
      1,
      Math.floor(
        (MAX_STORAGE_BINDING_SIZE - 4 * Uint32Array.BYTES_PER_ELEMENT) /
          (irradiance_rays_per_probe * PROBE_RAY_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
      )
    );
    const irradiance_probes_per_batch = clamp(
      Math.floor(
        Number(config.irradiance_probes_per_batch) || this.config.irradiance_probes_per_batch
      ),
      1,
      Math.min(MAX_COMPUTE_WORKGROUPS, max_batch_by_ray_buffer)
    );
    const max_emissive_lights = clamp(
      Math.floor(Number(config.max_emissive_lights) || this.config.max_emissive_lights),
      1,
      Math.floor(
        (MAX_STORAGE_BINDING_SIZE - 4 * Uint32Array.BYTES_PER_ELEMENT) /
          (12 * Uint32Array.BYTES_PER_ELEMENT)
      )
    );
    return {
      ...config,
      max_nodes,
      irradiance_rays_per_probe,
      irradiance_probes_per_batch,
      irradiance_sample_count: clamp(
        Math.floor(Number(config.irradiance_sample_count) || this.config.irradiance_sample_count),
        1,
        64
      ),
      irradiance_max_ray_distance: Math.max(
        1.0,
        Number(config.irradiance_max_ray_distance) || this.config.irradiance_max_ray_distance
      ),
      max_emissive_lights,
    };
  }

  _get_budget_limits() {
    const max_buffer_bytes = Math.max(4, MAX_STORAGE_BINDING_SIZE);
    const max_nodes_by_storage = Math.floor(
      max_buffer_bytes / (NODE_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
    );
    const max_nodes_by_dispatch = MAX_COMPUTE_WORKGROUPS * THREADS_PER_GROUP;
    const max_leaf_by_leaf_buffer = Math.floor(
      max_buffer_bytes / (LEAF_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
    );

    return {
      // Every surviving leaf originates from one node, so sizing both pools
      // from the node budget removes the independent leaf truncation point.
      max_nodes: Math.max(
        9,
        Math.min(max_nodes_by_storage, max_nodes_by_dispatch, max_leaf_by_leaf_buffer)
      ),
    };
  }

  _sanitize_debug_options(options) {
    const out = {};
    if (options.debug_level !== undefined) {
      const level = Number(options.debug_level);
      out.debug_level = Number.isFinite(level)
        ? clamp(Math.floor(level), -1, this.config.max_level)
        : -1;
    }
    return out;
  }

  _write_param_data() {
    this.params_data.fill(0);
    // Fields written by the GPU begin pass, such as world_min/root_dims, are
    // cleared here and then populated after TLAS bounds are known on-GPU.
    this.params_data[PARAM_MAX_LEVEL] = this.config.max_level;
    this.params_data[PARAM_MAX_NODES] = this.config.max_nodes;
    this.params_data[PARAM_LEAF_CAPACITY] = this.config.max_nodes;
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
    this.params_data[PARAM_IRRADIANCE_RAYS_PER_PROBE] = this.config.irradiance_rays_per_probe;
    this.params_data[PARAM_IRRADIANCE_PROBES_PER_BATCH] = this.config.irradiance_probes_per_batch;
    this.params_data[PARAM_IRRADIANCE_SAMPLE_COUNT] = this.config.irradiance_sample_count;
    this.params_data[PARAM_IRRADIANCE_MAX_RAY_DISTANCE] = this.config.irradiance_max_ray_distance;
    this.params_data[PARAM_IRRADIANCE_FORMAT_VERSION] = BAKE_FORMAT_VERSION;
    this.params_data[PARAM_IRRADIANCE_SH_WORDS_PER_PROBE] = SH_WORDS_PER_PROBE;
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
    const required_leaf_count = this.counters_data[COUNTER_LEAF_COUNT] || 0;
    const leaf_count = Math.min(required_leaf_count, this.config.max_nodes);
    const probe_count = Math.min(
      this.counters_data[COUNTER_PROBE_COUNT] || 0,
      this.config.max_nodes * PROBES_PER_BRICK
    );
    const status = this.counters_data[COUNTER_STATUS] || 0;
    const irradiance_sample_index = this.counters_data[COUNTER_IRRADIANCE_SAMPLE_INDEX] || 0;
    const irradiance_completed_probe_samples =
      this.counters_data[COUNTER_IRRADIANCE_COMPLETED_PROBE_SAMPLES] || 0;
    const max_irradiance_probes = Math.floor(
      MAX_STORAGE_BINDING_SIZE / (SH_WORDS_PER_PROBE * Uint32Array.BYTES_PER_ELEMENT)
    );
    const total_probe_samples = probe_count * this.config.irradiance_sample_count;
    // Probe batches are written in ascending probe order. Once one complete
    // sweep exists, every allocated probe has a coherent SH payload that can be
    // previewed even while later sample sets continue refining it.
    const irradiance_usable =
      probe_count > 0 &&
      this.irradiance_probe_capacity >= probe_count &&
      irradiance_completed_probe_samples >= probe_count;
    const irradiance_payload_complete =
      irradiance_usable &&
      irradiance_completed_probe_samples >= total_probe_samples;
    // The completed-write count is the authoritative completion proof. Do not
    // hide a valid payload if the redundant sticky status bit reaches CPU
    // readback a frame later than the counter values.
    const irradiance_ready = irradiance_payload_complete;
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
    const irradiance_bytes = probe_count * SH_WORDS_PER_PROBE * Uint32Array.BYTES_PER_ELEMENT;

    if (this.irradiance_bake_in_flight && irradiance_ready) {
      this.irradiance_bake_in_flight = false;
    }

    this.stats.baked = this.bake_in_flight || leaf_count > 0;
    this.stats.bake_serial = this.bake_serial;
    this.stats.root_dims = root_dims;
    this.stats.root_brick_size = root_brick_size;
    this.stats.min_level = this.config.min_level;
    this.stats.max_level = this.config.max_level;
    this.stats.max_level_reached = this.counters_data[COUNTER_MAX_LEVEL_REACHED] || 0;
    this.stats.node_count = node_count;
    this.stats.leaf_count = leaf_count;
    this.stats.leaf_capacity = this.config.max_nodes;
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
    this.stats.irradiance_bytes = irradiance_bytes;
    this.stats.irradiance_allocated_bytes =
      this.irradiance_probe_capacity * SH_WORDS_PER_PROBE * Uint32Array.BYTES_PER_ELEMENT;
    this.stats.total_bytes = node_bytes + leaf_bytes + irradiance_bytes;
    this.stats.irradiance_usable = irradiance_usable;
    this.stats.irradiance_ready = irradiance_ready;
    this.stats.irradiance_in_progress =
      this.irradiance_bake_in_flight || this.irradiance_allocation_pending;
    this.stats.irradiance_allocation_pending = this.irradiance_allocation_pending;
    this.stats.irradiance_capacity_exceeded = probe_count > max_irradiance_probes;
    this.stats.irradiance_progress = irradiance_ready
      ? 1
      : total_probe_samples > 0
        ? clamp(irradiance_completed_probe_samples / total_probe_samples, 0, 1)
        : 0;
    this.stats.irradiance_sample_index = irradiance_sample_index;
    this.stats.irradiance_sample_count = this.config.irradiance_sample_count;
    this.stats.irradiance_completed_probe_samples = irradiance_completed_probe_samples;
    this.stats.irradiance_required_probe_samples = total_probe_samples;
    this.stats.irradiance_rays_per_probe = this.config.irradiance_rays_per_probe;
    this.stats.debug_leaf_count = leaf_count;
    this.stats.debug_level = this.debug_config.debug_level;
    this.stats.truncated_by_node_limit =
      (status & (STATUS_NODE_OVERFLOW | STATUS_ROOT_OVERFLOW)) !== 0;
    this.stats.truncated_by_leaf_limit = (status & STATUS_LEAF_OVERFLOW) !== 0;
    this.stats.config = { ...this.config };

    this._maybe_queue_auto_resize_rebake({
      node_count,
      leaf_count: required_leaf_count,
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
    if (node_overflow || leaf_overflow) {
      const required_nodes = Math.max(
        node_count + 8,
        leaf_count + 1,
        Math.ceil(this.config.max_nodes * growth)
      );
      next_nodes = Math.min(limits.max_nodes, npot(required_nodes));
    }

    const can_grow = next_nodes > this.config.max_nodes;

    if (!can_grow) {
      return false;
    }

    this.bake({
      max_nodes: next_nodes,
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
    this.stats.leaf_capacity = 0;
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
    this.stats.irradiance_bytes = 0;
    this.stats.irradiance_allocated_bytes = 0;
    this.stats.total_bytes = 0;
    this.stats.irradiance_usable = false;
    this.stats.irradiance_ready = false;
    this.stats.irradiance_in_progress = false;
    this.stats.irradiance_allocation_pending = false;
    this.stats.irradiance_capacity_exceeded = false;
    this.stats.irradiance_progress = 0;
    this.stats.irradiance_sample_index = 0;
    this.stats.irradiance_sample_count = this.config.irradiance_sample_count;
    this.stats.irradiance_completed_probe_samples = 0;
    this.stats.irradiance_required_probe_samples = 0;
    this.stats.irradiance_rays_per_probe = this.config.irradiance_rays_per_probe;
    this.stats.debug_leaf_count = 0;
    this.stats.debug_level = this.debug_config.debug_level;
    this.stats.truncated_by_node_limit = false;
    this.stats.truncated_by_leaf_limit = false;
    this.stats.config = { ...this.config };
  }
}

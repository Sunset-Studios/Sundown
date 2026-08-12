import { GIModule, GIPipelineStage } from "./gi_pipeline.js";
import { SharedFrameInfoBuffer, SharedViewBuffer } from "../../core/shared_data.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";
import { Texture } from "../texture.js";
import { Buffer } from "../buffer.js";
import { DebugDrawType } from "../renderer_types.js";
import { floor_to_multiple, clamp } from "../../utility/math.js";
import {
  register_material_buffers,
  register_scene_lighting_data,
  register_texture_pools,
} from "../render_graph_utils.js";

const COMPUTE_WORKGROUP_SIZE = 128;
const PROBE_SCHEDULER_PRIORITY_COUNT = 2;
const MAX_PROBE_CASCADES = 6;
// Must match the scalar-aligned DDGIProbeRayData layout in ddgi_common.wgsl.
const PROBE_RAY_DATA_WORD_COUNT = 13;
// Must match DDGIMSMEProbeStats in ddgi_common.wgsl.
const PROBE_MSME_STATS_WORD_COUNT = 8;
const PROBE_COUNTERS_NAME = "probe_volume_gi_counters";
const SURFACE_CACHE_COUNTERS_NAME = "surface_cache_counters";
const SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT = 9;
const SURFACE_CACHE_TRACE_DISPATCH_OFFSET = 0;
const SURFACE_CACHE_UPDATE_DISPATCH_OFFSET = 3 * Uint32Array.BYTES_PER_ELEMENT;
const SURFACE_CACHE_BOOTSTRAP_UPDATE_DISPATCH_OFFSET = 6 * Uint32Array.BYTES_PER_ELEMENT;
// Must match SurfaceCacheHitInfo and SurfaceCacheRadianceInfo in surface_cache_common.wgsl.
const SURFACE_CACHE_HIT_WORD_COUNT = 12;
const SURFACE_CACHE_RADIANCE_WORD_COUNT = 16;
const EMPTY_EMISSIVE_LIGHT_HEADER = new Uint32Array(4);

const compute_shader = (path) => ({ pipeline_shaders: { compute: { path } } });

function hash_u32(value) {
  let x = value >>> 0;
  x = ((x >>> 16) ^ x) * 0x45d9f3b;
  x = ((x >>> 16) ^ x) * 0x45d9f3b;
  return ((x >>> 16) ^ x) >>> 0;
}

function gcd_u32(a, b) {
  while (b !== 0) [a, b] = [b, a % b];
  return a >>> 0;
}

function coprime_stride(seed, modulus) {
  if (modulus <= 1) return 1;
  const range = modulus - 1;
  let stride = ((hash_u32(seed) % range) + 1) | 1;
  stride = ((stride - 1) % range) + 1;
  for (let i = 0; i < 32 && gcd_u32(stride, modulus) !== 1; i++) {
    stride = (stride + 2) % modulus || 1;
  }
  return gcd_u32(stride, modulus) === 1 ? stride : 1;
}

function permutation_params(probe_count) {
  const seed = hash_u32(probe_count ^ 0xa3c59ac3);
  return {
    stride: coprime_stride(seed, probe_count),
    base_offset: hash_u32(seed ^ 0x85ebca6b) % probe_count,
    frame_stride: coprime_stride(seed ^ 0xc2b2ae35, probe_count),
  };
}

/**
 * Scrolling probe-volume tracing, hit shading, and SH accumulation.
 * Disable shading and accumulation to record only the tracing portion.
 */
export class ProbeVolumeRadianceCache extends GIModule {
  constructor({ stages = {} } = {}) {
    super({
      name: "probe-volume-radiance-cache",
      representation: "probe-volume-radiance-cache",
      stages,
      shader_setups: {
        reset: compute_shader("gi/ddgi_reset.wgsl"),
        scroll_reset: compute_shader("gi/ddgi_probe_scroll_reset.wgsl"),
        feedback_clear: compute_shader("gi/ddgi_probe_surface_feedback_clear.wgsl"),
        feedback: compute_shader("gi/ddgi_probe_surface_feedback.wgsl"),
        active_mark: compute_shader("gi/ddgi_probe_active_mark.wgsl"),
        depth_slots_init: compute_shader("gi/ddgi_depth_slots_init.wgsl"),
        depth_slots_reclaim: compute_shader("gi/ddgi_depth_slots_reclaim.wgsl"),
        depth_slots_allocate: compute_shader("gi/ddgi_depth_slots_allocate.wgsl"),
        active_prefix_sum: compute_shader("gi/ddgi_probe_active_prefix_sum.wgsl"),
        active_block_scan: compute_shader("gi/ddgi_probe_active_block_prefix_scan.wgsl"),
        compact: compute_shader("gi/ddgi_probe_indices_init.wgsl"),
        trace_init: compute_shader("gi/ddgi_probe_trace_init.wgsl"),
        trace_hit: compute_shader("gi/ddgi_probe_trace_hit.wgsl"),
        classify: compute_shader("gi/ddgi_probe_state_classify.wgsl"),
        compact_emissive: compute_shader("system_compute/compact_emissive_lights.wgsl"),
        shade: compute_shader("gi/ddgi_probe_trace_shade.wgsl"),
        accumulate: compute_shader("gi/ddgi_sh_probe_accumulate.wgsl"),
        depth_update: compute_shader("gi/ddgi_depth_update.wgsl"),
        sample: compute_shader("gi/ddgi_sh_probe_sample.wgsl"),
        resolve: compute_shader("gi/ddgi_diffuse_resolve.wgsl"),
        atrous: compute_shader("gi/ddgi_atrous_diffuse.wgsl"),
        debug: compute_shader("gi/ddgi_sh_probe_debug.wgsl"),
      },
    });
    this.params_data = new Float32Array(32 + 16 * MAX_PROBE_CASCADES);
    this.snapped_origins = null;
    this.scroll_offsets = null;
    this.initialized = null;
    this.counters_buffer = null;
    this.counters_data = null;
    this.depth_slot_signature = null;
  }

  reset_runtime_state() {
    this.snapped_origins = null;
    this.scroll_offsets = null;
    this.initialized = null;
    this.depth_slot_signature = null;
  }

  _setup_trace_resources(render_graph, context) {
    const config = context.config;

    const view = SharedViewBuffer.get_view_data(SharedFrameInfoBuffer.get_view_index());

    const grid_dims = config.probe_grid_dimensions ?? [32, 32, 32];
    const cascade_count = Math.min(Math.floor(config.cascade_count || 1), MAX_PROBE_CASCADES);
    const probes_per_cascade = grid_dims[0] * grid_dims[1] * grid_dims[2];
    const probe_count = probes_per_cascade * cascade_count;
    const probes_per_frame =
      config.probes_per_frame === 0 ? probe_count : Math.min(probe_count, config.probes_per_frame);
    const max_rays_per_probe = Math.max(1, Math.floor(config.max_rays_per_probe));
    const max_ray_length = Math.max(0.001, Number(config.max_ray_length) || 128.0);
    const probe_total_ray_count = probes_per_frame * max_rays_per_probe;
    const block_count = Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE);

    const depth_resolutions = Array.from({ length: cascade_count }, (_, index) => {
      return config.probe_depth_resolutions?.[index] ?? 16;
    });

    if (!this.snapped_origins || this.snapped_origins.length !== cascade_count) {
      this.snapped_origins = Array.from({ length: cascade_count }, () => new Float32Array(3));
      this.scroll_offsets = Array.from({ length: cascade_count }, () => new Int32Array(3));
      this.initialized = Array.from({ length: cascade_count }, () => false);
    }

    const cascade_data = new Float32Array(cascade_count * 16);
    let total_depth_texel_count = 0;
    let max_depth_texel_count_per_probe = 1;
    let cascade_has_scroll = false;
    for (let cascade = 0; cascade < cascade_count; cascade++) {
      const spacing =
        config.probe_spacing * Math.pow(Math.max(1, config.cascade_spacing_multiplier), cascade);
      const half = grid_dims.map((dimension) => (dimension - 1) * 0.5 * spacing);
      const next = view.view_position
        .slice(0, 3)
        .map((position, axis) => floor_to_multiple(position, spacing) - half[axis]);
      const origin = this.snapped_origins[cascade];
      const scroll = this.scroll_offsets[cascade];
      const delta = this.initialized[cascade]
        ? next.map((position, axis) => Math.round((position - origin[axis]) / spacing))
        : [0, 0, 0];
      origin.set(next);
      this.initialized[cascade] = true;
      for (let axis = 0; axis < 3; axis++) {
        if (delta[axis] !== 0) {
          scroll[axis] =
            (((scroll[axis] + delta[axis]) % grid_dims[axis]) + grid_dims[axis]) % grid_dims[axis];
          cascade_has_scroll = true;
        }
      }
      const resolution = depth_resolutions[cascade];
      const texels = resolution * resolution;
      max_depth_texel_count_per_probe = Math.max(max_depth_texel_count_per_probe, texels);
      const base = cascade * 16;
      cascade_data.set([origin[0], origin[1], origin[2], spacing], base);
      cascade_data.set([scroll[0], scroll[1], scroll[2], 0], base + 4);
      cascade_data.set([delta[0], delta[1], delta[2], delta.some(Boolean) ? 1 : 0], base + 8);
      cascade_data.set([resolution, texels, total_depth_texel_count, 0], base + 12);
      total_depth_texel_count += probes_per_cascade * texels;
    }

    const depth_words_per_slot = Math.ceil(max_depth_texel_count_per_probe / 2);
    const depth_slot_count = Math.min(
      probe_count,
      Math.max(1, Math.floor(config.probe_depth_slot_count ?? 65536))
    );
    const depth_slot_signature = [
      probe_count,
      depth_slot_count,
      depth_words_per_slot,
      config.probe_spacing,
      max_ray_length,
      depth_resolutions.join(","),
    ].join(":");
    const depth_slots_need_reset =
      context.force_recreate || this.depth_slot_signature !== depth_slot_signature;
    this.depth_slot_signature = depth_slot_signature;

    Object.assign(context, {
      grid_dims,
      cascade_count,
      probes_per_cascade,
      probe_count,
      probes_per_frame,
      max_rays_per_probe,
      max_ray_length,
      probe_total_ray_count,
      total_depth_texel_count,
      max_depth_texel_count_per_probe,
      max_depth_word_count_per_probe: depth_words_per_slot,
      depth_words_per_slot,
      depth_slot_count,
      depth_slots_need_reset,
      cascade_data,
      cascade_has_scroll,
      grid_log2: grid_dims.map((value) => Math.round(Math.log2(value))),
      grid_mask: grid_dims.map((value) => value - 1),
      frame_index: SharedFrameInfoBuffer.get_frame_index(),
    });
    context.ping_pong_frame = context.frame_index % 2;

    this.create_buffer(render_graph, "params", {
      name: "probe_volume_params",
      size: this.params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.create_buffer(render_graph, "update_indices", {
      name: "probe_volume_update_indices",
      size: probes_per_frame,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.create_buffer(render_graph, "surface_flags", {
      name: "probe_volume_surface_flags",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.create_buffer(render_graph, "active_flags", {
      name: "probe_volume_active_flags",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.create_buffer(render_graph, "active_prefix_sum", {
      name: "probe_volume_active_prefix_sum",
      size: probe_count * PROBE_SCHEDULER_PRIORITY_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.create_buffer(render_graph, "active_block_sums", {
      name: "probe_volume_active_block_sums",
      size: block_count * PROBE_SCHEDULER_PRIORITY_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.create_buffer(render_graph, "active_block_prefixes", {
      name: "probe_volume_active_block_prefixes",
      size: block_count * PROBE_SCHEDULER_PRIORITY_COUNT + PROBE_SCHEDULER_PRIORITY_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    const sparse_depth_force = context.force_recreate || depth_slots_need_reset;
    this.create_buffer(render_graph, "depth_slot_indices", {
      name: "probe_volume_depth_slot_indices",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: sparse_depth_force,
    });
    this.create_buffer(render_graph, "depth_slot_owners", {
      name: "probe_volume_depth_slot_owners",
      size: depth_slot_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: sparse_depth_force,
    });
    this.create_buffer(render_graph, "depth_slot_last_used", {
      name: "probe_volume_depth_slot_last_used",
      size: depth_slot_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: sparse_depth_force,
    });
    this.create_buffer(render_graph, "depth_slot_free_list", {
      name: "probe_volume_depth_slot_free_list",
      size: depth_slot_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: sparse_depth_force,
    });
    this.create_buffer(render_graph, "depth_slot_allocator_state", {
      name: "probe_volume_depth_slot_allocator_state",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: sparse_depth_force,
    });
    this.create_buffer(render_graph, "ray_hits", {
      name: "probe_volume_ray_hits",
      size: 4 + probe_total_ray_count * PROBE_RAY_DATA_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.create_buffer(render_graph, "probe_states", {
      name: "probe_volume_states",
      size: probe_count * 2,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.create_buffer(render_graph, "emissive_lights", {
      name: "probe_volume_emissive_lights",
      size: 4 + Math.max(1, Math.floor(config.max_emissive_lights)) * 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });

    this.import_resource(
      "entity_index_lookup",
      render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name)
    );

    this._ensure_counters(context.force_recreate);

    this.import_resource("counters", render_graph.register_buffer(PROBE_COUNTERS_NAME));
  }

  _setup_shading_resources(render_graph) {
    this.material_buffers = register_material_buffers(render_graph);
    this.texture_pools = register_texture_pools(render_graph);
    this.scene_lighting_data = register_scene_lighting_data(render_graph);
  }

  _setup_accumulation_resources(render_graph, context) {
    const {
      width,
      height,
      probe_count,
      depth_slot_count,
      depth_words_per_slot,
      depth_slots_need_reset,
      force_recreate,
    } = context;
    this.create_buffer(render_graph, "history_valid", {
      name: "probe_sh_history_valid",
      size: probe_count,
      usage: GPUBufferUsage.STORAGE,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "depth_moments", {
      name: "probe_sh_depth_moments",
      size: depth_slot_count * depth_words_per_slot,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate || depth_slots_need_reset,
    });
    this.create_buffer(render_graph, "sh_probes", {
      name: "probe_sh_coefficients",
      size: probe_count * 6,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "msme_stats", {
      name: "probe_sh_msme_stats",
      size: probe_count * PROBE_MSME_STATS_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    this.create_image(render_graph, "direct_output", {
      name: "probe_sh_direct_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "diffuse_output", {
      name: "probe_sh_diffuse_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "specular_output", {
      name: "probe_sh_specular_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    const upscale = Math.max(1, Math.floor(context.config.diffuse_sample_upscale_factor || 1));
    context.diffuse_sample_upscale_factor = upscale;
    context.diffuse_sample_width = Math.max(1, Math.ceil(width / upscale));
    context.diffuse_sample_height = Math.max(1, Math.ceil(height / upscale));

    if (upscale > 1) {
      this.create_image(render_graph, "diffuse_sample_output", {
        name: "probe_sh_diffuse_sample_intermediate",
        format: "rgba16float",
        width: context.diffuse_sample_width,
        height: context.diffuse_sample_height,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        force: force_recreate,
      });
    } else {
      this.import_resource("diffuse_sample_output", this.get_resource("diffuse_output"));
    }

    this.create_image(render_graph, "atrous_ping", {
      name: "probe_sh_diffuse_atrous_ping",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "atrous_pong", {
      name: "probe_sh_diffuse_atrous_pong",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    this.atrous_params_data = new Float32Array([1, 0.04, 64, 1]);
    this.create_buffer(render_graph, "atrous_params", {
      name: "probe_sh_diffuse_atrous_params",
      size: this.atrous_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
  }

  setup(render_graph, context, branch) {
    this._setup_trace_resources(render_graph, context, branch);
    this._setup_shading_resources(render_graph, context, branch);
    this._setup_accumulation_resources(render_graph, context, branch);
  }

  _ensure_counters(force_recreate) {
    if (!force_recreate && this.counters_buffer) {
      return;
    }

    if (this.counters_buffer) {
      this.counters_buffer.destroy();
    }

    this.counters_data = new Uint32Array(6);
    this.counters_buffer = Buffer.create({
      name: PROBE_COUNTERS_NAME,
      raw_data: this.counters_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
    });
  }

  _record_trace_passes(render_graph, context, branch) {
    const { config, inputs } = context;
    const accumulator = this;
    const shade = this;

    const params = this.get_resource("params");
    const counters = this.get_resource("counters");
    const ray_hits = this.get_resource("ray_hits");
    const states = this.get_resource("probe_states");
    const update_indices = this.get_resource("update_indices");
    const surface_flags = this.get_resource("surface_flags");
    const active_flags = this.get_resource("active_flags");
    const active_prefix_sum = this.get_resource("active_prefix_sum");
    const active_block_sums = this.get_resource("active_block_sums");
    const active_block_prefixes = this.get_resource("active_block_prefixes");
    const depth_slot_indices = this.get_resource("depth_slot_indices");
    const depth_slot_owners = this.get_resource("depth_slot_owners");
    const depth_slot_last_used = this.get_resource("depth_slot_last_used");
    const depth_slot_free_list = this.get_resource("depth_slot_free_list");
    const depth_slot_allocator_state = this.get_resource("depth_slot_allocator_state");
    const emissive_lights = this.get_resource("emissive_lights");
    const entity_index_lookup = this.get_resource("entity_index_lookup");
    const sh_probes = accumulator.get_resource("sh_probes");
    const depth_moments = accumulator.get_resource("depth_moments");
    const msme_stats = accumulator.get_resource("msme_stats");
    const materials = shade.material_buffers;
    const textures = shade.texture_pools;

    const permutation = permutation_params(context.probe_count);

    this.add_graph_local_pass(render_graph, "probe_volume_upload_params", (graph) => {
      const primary_origin = this.snapped_origins[0];

      this.params_data[0] = context.probe_count;
      this.params_data[1] = context.max_rays_per_probe;
      this.params_data[2] = context.probes_per_frame;
      this.params_data[3] = config.probe_spacing;
      this.params_data[4] = context.grid_dims[0];
      this.params_data[5] = context.grid_dims[1];
      this.params_data[6] = context.grid_dims[2];
      this.params_data[7] = config.probe_radius;
      this.params_data[8] = primary_origin[0];
      this.params_data[9] = primary_origin[1];
      this.params_data[10] = primary_origin[2];
      this.params_data[11] = 0;
      this.params_data[12] = context.grid_log2[0];
      this.params_data[13] = context.grid_log2[1];
      this.params_data[14] = context.grid_log2[2];
      this.params_data[15] = 0;
      this.params_data[16] = context.grid_mask[0];
      this.params_data[17] = context.grid_mask[1];
      this.params_data[18] = context.grid_mask[2];
      this.params_data[19] = 0;
      this.params_data[20] = context.depth_slot_count;
      this.params_data[21] = context.depth_words_per_slot;
      this.params_data[22] = context.max_depth_texel_count_per_probe;
      this.params_data[23] = Math.max(
        1,
        Math.floor(config.probe_depth_slot_retention_frames ?? 120)
      );
      this.params_data[24] = context.frame_index;
      this.params_data[25] = config.indirect_boost;
      this.params_data[26] = context.cascade_count;
      this.params_data[27] = context.max_ray_length;
      this.params_data[28] = permutation.stride;
      this.params_data[29] = permutation.base_offset;
      this.params_data[30] = permutation.frame_stride;
      this.params_data[31] = 0;

      this.params_data.set(context.cascade_data, 32);

      graph.get_physical_buffer(params).write_raw(this.params_data);
    });

    this.add_compute_pass(
      render_graph,
      "compact_emissive",
      "compact_emissive_lights",
      {
        inputs: [
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_directory,
          inputs.index_buffer,
          inputs.entity_transforms,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          entity_index_lookup,
          emissive_lights,
          textures.albedo,
          textures.emission,
        ],
        outputs: [emissive_lights],
      },
      (graph, frame_data) => {
        const bounds_buffer = graph.get_physical_buffer(inputs.tlas_bvh2_bounds);
        const emissive_lights_buffer = graph.get_physical_buffer(emissive_lights);
        emissive_lights_buffer.write_raw(new Uint32Array([0, 0, 0, 0]), 0);

        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(Math.floor(bounds_buffer.config.size / 32) / COMPUTE_WORKGROUP_SIZE),
            1,
            1
          );
      }
    );

    this.add_compute_pass(
      render_graph,
      "reset",
      "probe_volume_reset",
      {
        inputs: [counters, inputs.dense_lights, ray_hits],
        outputs: [counters, ray_hits],
      },
      (graph, frame_data) => graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1)
    );
    if (context.depth_slots_need_reset) {
      this.add_compute_pass(
        render_graph,
        "depth_slots_init",
        "probe_volume_depth_slots_init",
        {
          inputs: [
            depth_slot_indices,
            depth_slot_owners,
            depth_slot_last_used,
            depth_slot_free_list,
            depth_slot_allocator_state,
          ],
          outputs: [
            depth_slot_indices,
            depth_slot_owners,
            depth_slot_last_used,
            depth_slot_free_list,
            depth_slot_allocator_state,
          ],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(
              Math.ceil(
                Math.max(context.probe_count, context.depth_slot_count) / COMPUTE_WORKGROUP_SIZE
              ),
              1,
              1
            )
      );
    }
    if (context.cascade_has_scroll) {
      this.add_compute_pass(
        render_graph,
        "scroll_reset",
        "probe_volume_scroll_reset",
        {
          inputs: [
            params,
            sh_probes,
            states,
            msme_stats,
            depth_slot_indices,
            depth_slot_owners,
            depth_slot_free_list,
            depth_slot_allocator_state,
          ],
          outputs: [
            sh_probes,
            states,
            msme_stats,
            depth_slot_indices,
            depth_slot_owners,
            depth_slot_free_list,
            depth_slot_allocator_state,
          ],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
      );
    }
    this.add_compute_pass(
      render_graph,
      "feedback_clear",
      "probe_volume_feedback_clear",
      {
        inputs: [params, surface_flags],
        outputs: [surface_flags],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "feedback",
      "probe_volume_feedback",
      {
        inputs: [params, inputs.hzb_texture, inputs.gbuffer_normal, surface_flags],
        outputs: [surface_flags],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );
    this.add_compute_pass(
      render_graph,
      "active_mark",
      "probe_volume_active_mark",
      {
        inputs: [
          params,
          states,
          surface_flags,
          active_flags,
          depth_slot_indices,
          depth_slot_last_used,
        ],
        outputs: [active_flags, states, depth_slot_last_used],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "depth_slots_reclaim",
      "probe_volume_depth_slots_reclaim",
      {
        inputs: [
          params,
          depth_slot_indices,
          depth_slot_owners,
          depth_slot_last_used,
          depth_slot_free_list,
          depth_slot_allocator_state,
        ],
        outputs: [
          depth_slot_indices,
          depth_slot_owners,
          depth_slot_free_list,
          depth_slot_allocator_state,
        ],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.depth_slot_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "active_prefix_sum",
      "probe_volume_active_prefix_sum",
      {
        inputs: [active_flags, active_prefix_sum, active_block_sums],
        outputs: [active_prefix_sum, active_block_sums],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "active_block_scan",
      "probe_volume_active_block_prefix_scan",
      {
        inputs: [active_block_sums, active_block_prefixes, counters, params],
        outputs: [active_block_prefixes, counters],
      },
      (graph, frame_data) => graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "compact",
      "probe_volume_compact_active_probes",
      {
        inputs: [
          params,
          update_indices,
          active_flags,
          active_prefix_sum,
          active_block_prefixes,
          counters,
        ],
        outputs: [update_indices],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "depth_slots_allocate",
      "probe_volume_depth_slots_allocate",
      {
        inputs: [
          params,
          update_indices,
          counters,
          depth_slot_indices,
          depth_slot_owners,
          depth_slot_last_used,
          depth_slot_free_list,
          depth_slot_allocator_state,
          depth_moments,
        ],
        outputs: [
          depth_slot_indices,
          depth_slot_owners,
          depth_slot_last_used,
          depth_slot_allocator_state,
          depth_moments,
        ],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probes_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "trace_init",
      "probe_volume_trace_init",
      {
        inputs: [params, ray_hits, counters],
        outputs: [ray_hits],
      },
      (graph, frame_data) => graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "trace_hit",
      "probe_volume_trace_hits",
      {
        inputs: [
          params,
          ray_hits,
          update_indices,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          inputs.compact_transforms,
          inputs.index_buffer,
          inputs.dense_lights,
          emissive_lights,
          entity_index_lookup,
        ],
        outputs: [ray_hits],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probe_total_ray_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
  }

  _record_shading_passes(render_graph, context, branch) {
    const trace = this;
    const accumulator = this;
    const materials = this.material_buffers;
    const textures = this.texture_pools;
    const lighting = this.scene_lighting_data;
    const ray_hits = trace.get_resource("ray_hits");
    this.add_compute_pass(
      render_graph,
      "shade",
      "probe_sh_shade_hits",
      {
        inputs: [
          trace.get_resource("params"),
          lighting.scene_lighting_buffer,
          ray_hits,
          trace.get_resource("update_indices"),
          trace.get_resource("probe_states"),
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          accumulator.get_resource("sh_probes"),
          accumulator.get_resource("depth_moments"),
          trace.get_resource("entity_index_lookup"),
          context.inputs.entity_transforms,
          textures.albedo,
          textures.normal,
          textures.roughness,
          textures.metallic,
          textures.ao,
          textures.height,
          textures.specular,
          textures.emission,
          lighting.skybox_image,
          trace.get_resource("depth_slot_indices"),
        ],
        outputs: [ray_hits],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probe_total_ray_count / 128), 1, 1)
    );
  }

  _record_accumulation_passes(render_graph, context, branch) {
    const trace = this;
    const params = trace.get_resource("params");
    const states = trace.get_resource("probe_states");
    const ray_hits = trace.get_resource("ray_hits");
    const history_valid = this.get_resource("history_valid");
    const sh_probes = this.get_resource("sh_probes");
    const msme_stats = this.get_resource("msme_stats");
    this.add_compute_pass(
      render_graph,
      "accumulate",
      "probe_sh_accumulate",
      {
        inputs: [
          params,
          trace.get_resource("update_indices"),
          ray_hits,
          history_valid,
          sh_probes,
          states,
          msme_stats,
          trace.get_resource("counters"),
        ],
        outputs: [history_valid, sh_probes, states, msme_stats],
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(context.probes_per_frame, 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "depth_update",
      "probe_sh_depth_moments_update",
      {
        inputs: [
          params,
          ray_hits,
          trace.get_resource("update_indices"),
          history_valid,
          this.get_resource("depth_moments"),
          trace.get_resource("depth_slot_indices"),
        ],
        outputs: [this.get_resource("depth_moments")],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(context.probes_per_frame / 8),
            Math.ceil(context.max_depth_word_count_per_probe / 8),
            1
          )
    );
  }

  _record_post_trace_passes(render_graph, context) {
    const params = this.get_resource("params");
    const states = this.get_resource("probe_states");
    this.add_compute_pass(
      render_graph,
      "classify",
      "probe_volume_classify",
      {
        inputs: [
          params,
          this.get_resource("update_indices"),
          this.get_resource("ray_hits"),
          states,
          this.get_resource("counters"),
        ],
        outputs: [states],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probes_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
  }

  _record_accumulation_resolve_passes(render_graph, context, branch) {
    const trace = this;
    const lighting = this.scene_lighting_data;
    const inputs = context.inputs;
    const sample_output = this.get_resource("diffuse_sample_output");
    this.add_compute_pass(
      render_graph,
      "sample",
      `probe_sh_sample_${context.ping_pong_frame}`,
      {
        inputs: [
          trace.get_resource("params"),
          this.get_resource("sh_probes"),
          trace.get_resource("probe_states"),
          this.get_resource("depth_moments"),
          inputs.hzb_texture,
          inputs.gbuffer_normal,
          sample_output,
          lighting.scene_lighting_buffer,
          lighting.skybox_image,
          trace.get_resource("depth_slot_indices"),
        ],
        outputs: [sample_output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(context.diffuse_sample_width / 16),
            Math.ceil(context.diffuse_sample_height / 16),
            1
          )
    );
    const diffuse_output = this.get_resource("diffuse_output");
    if (context.diffuse_sample_upscale_factor > 1) {
      this.add_compute_pass(
        render_graph,
        "resolve",
        `probe_sh_resolve_${context.ping_pong_frame}`,
        {
          inputs: [sample_output, inputs.hzb_texture, inputs.gbuffer_normal, diffuse_output],
          outputs: [diffuse_output],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
      );
    }

    let final_diffuse = diffuse_output;
    let read = diffuse_output;
    let write = this.get_resource("atrous_ping");
    const pass_count =
      context.config.diffuse_atrous_enabled === false
        ? 0
        : Math.max(0, Math.floor(context.config.diffuse_atrous_pass_count || 0));
    for (let pass_index = 0; pass_index < pass_count; pass_index++) {
      this.add_graph_local_pass(
        render_graph,
        `probe_sh_atrous_upload_params_${context.ping_pong_frame}_${pass_index}`,
        (graph) => {
          this.atrous_params_data[0] = Math.pow(2, pass_index);
          this.atrous_params_data[1] = Math.max(
            0.0001,
            context.config.diffuse_atrous_phi_depth || 0.04
          );
          this.atrous_params_data[2] = Math.max(1, context.config.diffuse_atrous_phi_normal || 64);
          this.atrous_params_data[3] = Math.max(
            0.0001,
            context.config.diffuse_atrous_luma_sigma || 1
          );
          graph
            .get_physical_buffer(this.get_resource("atrous_params"))
            .write_raw(this.atrous_params_data);
        }
      );
      this.add_compute_pass(
        render_graph,
        "atrous",
        `probe_sh_atrous_${context.ping_pong_frame}_${pass_index}`,
        {
          inputs: [
            this.get_resource("atrous_params"),
            read,
            inputs.depth_texture,
            inputs.gbuffer_normal,
            write,
          ],
          outputs: [write],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
      );
      final_diffuse = write;
      read = write;
      write =
        write === this.get_resource("atrous_ping")
          ? this.get_resource("atrous_pong")
          : this.get_resource("atrous_ping");
    }
    this.import_resource("diffuse_output", final_diffuse);
  }

  record(render_graph, context, branch) {
    if (this.is_stage_enabled(GIPipelineStage.Trace))
      this._record_trace_passes(render_graph, context, branch);
    if (this.is_stage_enabled(GIPipelineStage.Shading))
      this._record_shading_passes(render_graph, context, branch);
    if (this.is_stage_enabled(GIPipelineStage.Accumulation))
      this._record_accumulation_passes(render_graph, context, branch);
    if (this.is_stage_enabled(GIPipelineStage.Trace))
      this._record_post_trace_passes(render_graph, context, branch);
    if (this.is_stage_enabled(GIPipelineStage.Accumulation))
      this._record_accumulation_resolve_passes(render_graph, context, branch);
  }

  _record_accumulation_debug_passes(render_graph, context, branch) {
    if (context.debug_view !== DebugDrawType.GI_Probes) return null;

    const trace = this;
    const output = this.create_image(render_graph, "debug_output", {
      name: "probe_sh_debug_output",
      format: "rgba16float",
      width: context.width,
      height: context.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: context.force_recreate,
    });
    this.add_compute_pass(
      render_graph,
      "debug",
      "probe_sh_debug",
      {
        inputs: [
          trace.get_resource("params"),
          this.get_resource("sh_probes"),
          trace.get_resource("probe_states"),
          trace.get_resource("surface_flags"),
          context.inputs.scene_color,
          context.inputs.depth_texture,
          output,
        ],
        outputs: [output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );
    return output;
  }

  record_debug(render_graph, context, branch) {
    if (!this.is_stage_enabled(GIPipelineStage.Accumulation)) return null;
    return this._record_accumulation_debug_passes(render_graph, context, branch);
  }

  get_stats(context = this.frame_context) {
    if (!this.counters_data || !context?.probe_count) return null;
    const update_count = this.counters_data[5] || 0;
    return {
      probe_grid_dims: context.grid_dims,
      cascade_count: context.cascade_count,
      probes_per_cascade: context.probes_per_cascade,
      total_probe_count: context.probe_count,
      probes_per_frame: context.probes_per_frame,
      max_rays_per_probe: context.max_rays_per_probe,
      max_ray_length: context.max_ray_length,
      total_rays_fired: update_count * context.max_rays_per_probe,
      active_probe_count: this.counters_data[2] || 0,
      probe_update_count: update_count,
      light_count: this.counters_data[0] || 0,
      probe_spacing: context.config.probe_spacing,
      probe_radius: context.config.probe_radius,
      depth_slot_count: context.depth_slot_count,
      depth_slot_retention_frames: context.config.probe_depth_slot_retention_frames ?? 120,
      depth_sparse_bytes: context.depth_slot_count * context.depth_words_per_slot * 4,
      depth_sparse_metadata_bytes: context.probe_count * 4 + context.depth_slot_count * 12 + 16,
      depth_dense_packed_bytes: context.total_depth_texel_count * 2,
      depth_dense_previous_bytes: context.total_depth_texel_count * 4,
    };
  }
}

/**
 * Hashed surface tracing, hit shading, and SH accumulation.
 * Disable shading and accumulation to record only the tracing portion.
 */
export class SurfaceRadianceCache extends GIModule {
  constructor({ stages = {} } = {}) {
    super({
      name: "surface-radiance-cache",
      representation: "surface-radiance-cache",
      stages,
      shader_setups: {
        compact_emissive: compute_shader("system_compute/compact_emissive_lights.wgsl"),
        detect_scene_change: compute_shader("gi/surface_cache_detect_scene_change.wgsl"),
        prepare_dispatch: compute_shader("gi/surface_cache_prepare_dispatch.wgsl"),
        feedback: compute_shader("gi/surface_cache_feedback.wgsl"),
        trace_hit: compute_shader("gi/surface_cache_trace_hit.wgsl"),
        shade: compute_shader("gi/surface_cache_trace_shade.wgsl"),
        shadow: compute_shader("gi/surface_cache_trace_shadow.wgsl"),
        accumulate: compute_shader("gi/surface_cache_accumulate.wgsl"),
        accumulate_bootstrap: compute_shader("gi/surface_cache_accumulate_bootstrap.wgsl"),
        resolve: compute_shader("gi/surface_cache_resolve.wgsl"),
        temporal: compute_shader("gi/surface_cache_temporal.wgsl"),
        debug: compute_shader("gi/surface_cache_debug.wgsl"),
      },
    });
    this.params_data = new Float32Array(24);
    this.params_u32_data = new Uint32Array(this.params_data.buffer);
    this.temporal_params_data = new Float32Array(8);
    this.counters_reset_data = new Uint32Array(9);
    this.counters_buffer = null;
    this.counters_data = null;
    this.stats_enabled = false;
  }

  setup(render_graph, context, branch) {
    this._setup_trace_resources(render_graph, context, branch);
    this._setup_shading_resources(render_graph, context, branch);
    this._setup_accumulation_resources(render_graph, context, branch);
  }

  record(render_graph, context, branch) {
    if (this.is_stage_enabled(GIPipelineStage.Trace))
      this._record_trace_passes(render_graph, context, branch);
    if (this.is_stage_enabled(GIPipelineStage.Shading))
      this._record_shading_passes(render_graph, context, branch);
    if (this.is_stage_enabled(GIPipelineStage.Accumulation))
      this._record_accumulation_passes(render_graph, context, branch);
  }

  record_debug(render_graph, context, branch) {
    if (!this.is_stage_enabled(GIPipelineStage.Accumulation)) return null;
    return this._record_accumulation_debug_passes(render_graph, context, branch);
  }

  set_stats_enabled(enabled) {
    const next_enabled = !!enabled;
    if (this.stats_enabled === next_enabled) return;

    this.stats_enabled = next_enabled;
    if (this.counters_buffer) {
      this.counters_buffer.config.own_readback = next_enabled;
    }
  }

  _setup_trace_resources(render_graph, context, branch) {
    const { config, width, height, force_recreate } = context;

    context.total_patches = Math.max(16, floor_to_multiple(config.surface_cache_size, 16));
    context.rays_per_patch = Math.max(
      Math.floor(config.rays_per_patch ?? 1),
      1,
    );
    context.total_ray_count = context.total_patches * context.rays_per_patch;
    context.bootstrap_rays_per_patch = Math.max(
      Math.floor(config.bootstrap_rays_per_patch ?? context.rays_per_patch),
      context.rays_per_patch,
    );
    context.bootstrap_patch_capacity = clamp(
      Math.floor(config.bootstrap_patch_capacity ?? 0),
      0,
      context.total_patches
    );
    context.bootstrap_enabled =
      context.bootstrap_patch_capacity > 0 &&
      context.bootstrap_rays_per_patch > context.rays_per_patch;
    context.mature_patch_update_period = Math.max(
      Math.floor(config.mature_patch_update_period ?? 1),
      1
    );
    context.maximum_ray_count_per_frame = clamp(
      Math.floor(config.maximum_ray_count_per_frame ?? context.total_ray_count),
      context.rays_per_patch,
      context.total_ray_count
    );

    this.create_buffer(render_graph, "params", {
      name: "surface_cache_params",
      size: this.params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "surface_cache", {
      name: "surface_cache_elements",
      size: context.total_patches * 20,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "hashmap_entries", {
      name: "surface_cache_hashmap_entries",
      size: context.total_patches * 3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "update_indices", {
      name: "surface_cache_update_indices",
      size: context.total_patches,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "bootstrap_indices", {
      name: "surface_cache_bootstrap_indices",
      size: Math.max(1, context.bootstrap_patch_capacity),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "hit_info", {
      name: "surface_cache_hit_info",
      size: context.total_ray_count * SURFACE_CACHE_HIT_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "emissive_lights", {
      name: "surface_cache_emissive_lights",
      size: 4 + Math.max(1, Math.floor(config.max_emissive_lights ?? 32768)) * 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "dispatch_args", {
      name: "surface_cache_dispatch_args",
      size: SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
      force: force_recreate,
    });
    this.import_resource(
      "entity_index_lookup",
      render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name)
    );
    this.import_resource(
      "entity_flags",
      render_graph.register_buffer(FragmentGpuBuffer.entity_flags_buffer.buffer.config.name)
    );
    if (this.stats_enabled || this.counters_buffer) {
      const needs_stats_buffer = this.stats_enabled && !this.counters_buffer;
      this._ensure_counters(force_recreate || needs_stats_buffer);

      this.counters_buffer.config.own_readback = this.stats_enabled;
      this.import_resource("counters", render_graph.register_buffer(SURFACE_CACHE_COUNTERS_NAME));
    } else {
      this.create_buffer(render_graph, "counters", {
        name: SURFACE_CACHE_COUNTERS_NAME,
        size: 9,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: force_recreate,
      });
    }
  }

  _setup_shading_resources(render_graph, context, _branch) {
    this.create_buffer(render_graph, "radiance_info", {
      name: "surface_cache_radiance_info",
      size: context.total_ray_count * SURFACE_CACHE_RADIANCE_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.material_buffers = register_material_buffers(render_graph);
    this.texture_pools = register_texture_pools(render_graph);
    this.scene_lighting_data = register_scene_lighting_data(render_graph);
  }

  _setup_accumulation_resources(render_graph, context, branch) {
    const { width, height, total_patches, force_recreate } = context;
    const sh_size = total_patches * 6;

    const black_output = this.create_image(render_graph, "direct_output", {
      name: "surface_cache_black_output",
      format: "rgba16float",
      width: 1,
      height: 1,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.import_resource("specular_output", black_output);
    this.create_image(render_graph, "diffuse_output", {
      name: "surface_cache_diffuse_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "surface_cache_sh", {
      name: "surface_cache_sh",
      size: sh_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "temporal_params", {
      name: "surface_cache_temporal_params",
      size: this.temporal_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    if (context.config.screen_reconstruction_enabled !== false) {
      for (let frame = 0; frame < 2; frame++) {
        this.create_image(render_graph, `diffuse_history_${frame}`, {
          name: `surface_cache_diffuse_history_${frame}`,
          format: "rgba16float",
          width,
          height,
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
          clear_value: { r: 0, g: 0, b: 0, a: 0 },
          force: force_recreate,
        });
      }
    }
  }

  _record_trace_passes(render_graph, context, branch) {
    const { config, width, height, rays_per_patch, total_patches, inputs } = context;
    const params = this.get_resource("params");
    const surface_cache = this.get_resource("surface_cache");
    const hashmap_entries = this.get_resource("hashmap_entries");
    const update_indices = this.get_resource("update_indices");
    const bootstrap_indices = this.get_resource("bootstrap_indices");
    const counters = this.get_resource("counters");
    const dispatch_args = this.get_resource("dispatch_args");
    const hit_info = this.get_resource("hit_info");
    const entity_index_lookup = this.get_resource("entity_index_lookup");
    const entity_flags = this.get_resource("entity_flags");

    this.add_graph_local_pass(render_graph, "surface_cache_upload_params", (graph) => {
      this.params_data[0] = total_patches;
      this.params_data[1] = total_patches;
      this.params_data[2] = width;
      this.params_data[3] = height;
      this.params_data[4] = SharedFrameInfoBuffer.get_frame_index();
      this.params_data[5] = config.max_ray_length;
      this.params_data[6] = config.history_hysteresis;
      this.params_data[7] = config.max_history_samples;
      this.params_data[8] = config.indirect_boost;
      this.params_u32_data[9] = context.rays_per_patch;
      this.params_data[10] = config.cache_entry_lifetime;
      this.params_u32_data[11] = context.bootstrap_rays_per_patch;
      this.params_data[12] = clamp(config.bootstrap_ray_budget_fraction ?? 0.5, 0.0, 1.0);
      this.params_data[13] = Math.max(config.cache_pixel_footprint ?? 3, 1);
      this.params_data[14] = clamp(Math.floor(config.hash_search_count ?? 10), 1, 64);
      this.params_data[15] = Math.max(config.cache_normal_bias ?? 0, 0);
      this.params_data[16] = Math.max(config.history_footprint_start_samples ?? 4, 0);
      this.params_data[17] = Math.max(
        config.history_footprint_end_samples ?? 32,
        this.params_data[16] + 1
      );
      this.params_data[18] = Math.max(config.history_footprint_max_scale ?? 1, 1);
      this.params_data[19] = context.bootstrap_enabled ? context.bootstrap_patch_capacity : 0;
      this.params_u32_data[20] = context.mature_patch_update_period;
      this.params_u32_data[21] = context.maximum_ray_count_per_frame;
      this.params_u32_data[22] = 0;
      this.params_u32_data[23] = 0;
      graph.get_physical_buffer(params).write_raw(this.params_data);
      this.counters_reset_data.fill(0);
      this.counters_reset_data[5] = context.force_full_update ? 1 : 0;
      graph.get_physical_buffer(counters).write_raw(this.counters_reset_data);
    });

    if (context.mature_patch_update_period > 1) {
      this.add_compute_pass(
        render_graph,
        "detect_scene_change",
        "surface_cache_detect_scene_change",
        {
          inputs: [
            counters,
            inputs.tlas_bvh2_bounds,
            inputs.tlas_bvh_info,
            entity_index_lookup,
            entity_flags,
          ],
          outputs: [counters],
        },
        (graph, frame_data) => {
          const bounds_buffer = graph.get_physical_buffer(inputs.tlas_bvh2_bounds);
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(
              Math.ceil(Math.floor(bounds_buffer.config.size / 32) / COMPUTE_WORKGROUP_SIZE),
              1,
              1
            );
        }
      );
    }

    this.add_compute_pass(
      render_graph,
      "feedback",
      "surface_cache_feedback",
      {
        inputs: [
          params,
          surface_cache,
          counters,
          update_indices,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          hashmap_entries,
          bootstrap_indices,
        ],
        outputs: [surface_cache, counters, update_indices, hashmap_entries, bootstrap_indices],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    this.add_compute_pass(
      render_graph,
      "prepare_dispatch",
      "surface_cache_prepare_dispatch",
      {
        inputs: [params, counters, dispatch_args],
        outputs: [counters, dispatch_args],
      },
      (graph, frame_data) => graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1)
    );

    this.add_compute_pass(
      render_graph,
      "trace_hit",
      "surface_cache_trace_hits",
      {
        inputs: [
          params,
          surface_cache,
          update_indices,
          bootstrap_indices,
          counters,
          hit_info,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          inputs.compact_transforms,
          inputs.index_buffer,
          entity_index_lookup,
          dispatch_args,
        ],
        outputs: [hit_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_TRACE_DISPATCH_OFFSET
          )
    );
  }

  _record_shading_passes(render_graph, context, branch) {
    const trace = this;
    const params = trace.get_resource("params");
    const surface_cache = trace.get_resource("surface_cache");
    const hashmap_entries = trace.get_resource("hashmap_entries");
    const update_indices = trace.get_resource("update_indices");
    const bootstrap_indices = trace.get_resource("bootstrap_indices");
    const counters = trace.get_resource("counters");
    const dispatch_args = trace.get_resource("dispatch_args");
    const hit_info = trace.get_resource("hit_info");
    const emissive_lights = trace.get_resource("emissive_lights");
    const entity_index_lookup = trace.get_resource("entity_index_lookup");
    const radiance_info = this.get_resource("radiance_info");
    const sh = this.get_resource("surface_cache_sh");
    const { inputs } = context;
    const materials = this.material_buffers;
    const textures = this.texture_pools;
    const lighting = this.scene_lighting_data;

    this.add_compute_pass(
      render_graph,
      "compact_emissive",
      "surface_cache_compact_emissive_lights",
      {
        inputs: [
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_directory,
          inputs.index_buffer,
          inputs.entity_transforms,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          entity_index_lookup,
          emissive_lights,
          textures.albedo,
          textures.emission,
        ],
        outputs: [emissive_lights],
      },
      (graph, frame_data) => {
        const bounds_buffer = graph.get_physical_buffer(inputs.tlas_bvh2_bounds);
        graph.get_physical_buffer(emissive_lights).write_raw(EMPTY_EMISSIVE_LIGHT_HEADER, 0);
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(Math.floor(bounds_buffer.config.size / 32) / COMPUTE_WORKGROUP_SIZE),
            1,
            1
          );
      }
    );

    this.add_compute_pass(
      render_graph,
      "shade",
      "surface_cache_shade_hits",
      {
        inputs: [
          params,
          lighting.scene_lighting_buffer,
          surface_cache,
          sh,
          update_indices,
          bootstrap_indices,
          counters,
          hit_info,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          inputs.compact_transforms,
          inputs.dense_lights,
          emissive_lights,
          textures.albedo,
          textures.normal,
          textures.emission,
          lighting.skybox_image,
          radiance_info,
          hashmap_entries,
          dispatch_args,
        ],
        outputs: [radiance_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_TRACE_DISPATCH_OFFSET
          )
    );
    this.add_compute_pass(
      render_graph,
      "shadow",
      "surface_cache_trace_shadows",
      {
        inputs: [
          params,
          counters,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          inputs.compact_transforms,
          inputs.index_buffer,
          entity_index_lookup,
          radiance_info,
          dispatch_args,
        ],
        outputs: [radiance_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_TRACE_DISPATCH_OFFSET
          )
    );
  }

  _record_accumulation_passes(render_graph, context, branch) {
    const trace = this;
    const shade = this;
    const params = trace.get_resource("params");
    const surface_cache = trace.get_resource("surface_cache");
    const hashmap_entries = trace.get_resource("hashmap_entries");
    const update_indices = trace.get_resource("update_indices");
    const bootstrap_indices = trace.get_resource("bootstrap_indices");
    const counters = trace.get_resource("counters");
    const dispatch_args = trace.get_resource("dispatch_args");
    const hit_info = trace.get_resource("hit_info");
    const radiance_info = shade.get_resource("radiance_info");
    const sh = this.get_resource("surface_cache_sh");
    const direct = this.get_resource("direct_output");
    const diffuse = this.get_resource("diffuse_output");
    const { width, height, inputs } = context;

    this.add_compute_pass(
      render_graph,
      "accumulate",
      "surface_cache_sh_accumulate",
      {
        inputs: [
          params,
          surface_cache,
          sh,
          update_indices,
          bootstrap_indices,
          counters,
          hit_info,
          radiance_info,
          dispatch_args,
        ],
        outputs: [surface_cache, sh],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_UPDATE_DISPATCH_OFFSET
          )
    );
    this.add_compute_pass(
      render_graph,
      "accumulate_bootstrap",
      "surface_cache_bootstrap_sh_accumulate",
      {
        inputs: [
          params,
          surface_cache,
          sh,
          bootstrap_indices,
          counters,
          hit_info,
          radiance_info,
          dispatch_args,
        ],
        outputs: [surface_cache, sh],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch_indirect(
            graph.get_physical_buffer(dispatch_args),
            SURFACE_CACHE_BOOTSTRAP_UPDATE_DISPATCH_OFFSET
          )
    );
    this.add_compute_pass(
      render_graph,
      "resolve",
      "surface_cache_sh_resolve",
      {
        inputs: [
          params,
          surface_cache,
          sh,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          diffuse,
          direct,
          hashmap_entries,
        ],
        outputs: [diffuse, direct],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    if (context.config.screen_reconstruction_enabled === false) {
      return;
    }

    const history_frame = SharedFrameInfoBuffer.get_frame_index() & 1;
    const history_prev = this.get_resource(`diffuse_history_${history_frame}`);
    const history_curr = this.get_resource(`diffuse_history_${1 - history_frame}`);

    const temporal_params = this.get_resource("temporal_params");
    const temporal_response = Math.fround(
      clamp(context.config.temporal_response ?? 0.02, 0.0, 1)
    );
    const temporal_max_history_frames = Math.max(
      1,
      Math.floor(context.config.temporal_max_history_frames ?? 64)
    );
    const temporal_depth_threshold = Math.fround(
      Math.max(context.config.temporal_depth_threshold ?? 0.03, 0.0001)
    );
    const temporal_normal_threshold = Math.fround(
      clamp(context.config.temporal_normal_threshold ?? 0.9, 0, 0.9999)
    );
    const spatial_filter_radius = clamp(
      Math.floor((context.config.cache_pixel_footprint ?? 3) * 0.5),
      1,
      8
    );
    const temporal_params_need_upload =
      context.force_recreate ||
      this.temporal_params_data[0] !== temporal_response ||
      this.temporal_params_data[1] !== temporal_max_history_frames ||
      this.temporal_params_data[2] !== temporal_depth_threshold ||
      this.temporal_params_data[3] !== temporal_normal_threshold ||
      this.temporal_params_data[4] !== spatial_filter_radius;
    if (temporal_params_need_upload) {
      this.temporal_params_data[0] = temporal_response;
      this.temporal_params_data[1] = temporal_max_history_frames;
      this.temporal_params_data[2] = temporal_depth_threshold;
      this.temporal_params_data[3] = temporal_normal_threshold;
      this.temporal_params_data[4] = spatial_filter_radius;
      this.temporal_params_data[5] = 0;
      this.temporal_params_data[6] = 0;
      this.temporal_params_data[7] = 0;
      this.add_graph_local_pass(render_graph, "surface_cache_temporal_upload_params", (graph) => {
        graph.get_physical_buffer(temporal_params).write_raw(this.temporal_params_data);
      });
    }
    this.add_compute_pass(
      render_graph,
      "temporal",
      `surface_cache_temporal_${history_frame}`,
      {
        inputs: [
          temporal_params,
          diffuse,
          history_prev,
          inputs.depth_texture,
          inputs.prev_depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_normal_prev,
          inputs.gbuffer_motion_emissive,
          history_curr,
        ],
        outputs: [history_curr],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );

    this.import_resource("diffuse_output", history_curr);
  }

  _record_accumulation_debug_passes(render_graph, context, branch) {
    if (context.debug_view !== DebugDrawType.GI_SurfaceCache) return null;

    const trace = this;
    const output = this.create_image(render_graph, "debug_output", {
      name: "surface_cache_debug_output",
      format: "rgba16float",
      width: context.width,
      height: context.height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: context.force_recreate,
    });
    this.add_compute_pass(
      render_graph,
      "debug",
      "surface_cache_debug",
      {
        inputs: [
          trace.get_resource("params"),
          trace.get_resource("surface_cache"),
          this.get_resource("surface_cache_sh"),
          context.inputs.depth_texture,
          context.inputs.gbuffer_normal,
          context.inputs.scene_color,
          output,
          trace.get_resource("hashmap_entries"),
        ],
        outputs: [output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );
    return output;
  }

  _ensure_counters(force_recreate) {
    if (!force_recreate && this.counters_buffer) {
      return;
    }

    if (this.counters_buffer) {
      this.counters_buffer.destroy();
    }

    this.counters_data = new Uint32Array(9);
    this.counters_buffer = Buffer.create({
      name: SURFACE_CACHE_COUNTERS_NAME,
      raw_data: this.counters_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
      force: force_recreate,
    });
  }

  get_stats(context = this.frame_context) {
    if (!this.counters_data || !context?.total_patches) return null;

    const active_patch_count = Math.min(this.counters_data[0] || 0, context.total_patches);
    const update_patch_count = Math.min(this.counters_data[1] || 0, active_patch_count);
    const bootstrap_patch_count = Math.min(
      this.counters_data[2] || 0,
      context.bootstrap_patch_capacity
    );
    const available_bootstrap_patch_count = Math.min(
      this.counters_data[6] || bootstrap_patch_count,
      context.bootstrap_patch_capacity
    );
    const bootstrap_rays_per_patch = bootstrap_patch_count > 0
      ? clamp(
          this.counters_data[3] || context.rays_per_patch,
          context.rays_per_patch,
          context.bootstrap_rays_per_patch
        )
      : 0;
    const deferred_patch_count = Math.max(
      active_patch_count - bootstrap_patch_count - update_patch_count,
      0
    );
    const surface_cache_bytes = context.total_patches * 20 * 4;
    const hashmap_bytes = context.total_patches * 3 * 4;
    const sh_bytes = context.total_patches * 6 * 4;
    const ray_working_set_bytes =
      context.total_ray_count *
      (SURFACE_CACHE_HIT_WORD_COUNT + SURFACE_CACHE_RADIANCE_WORD_COUNT) *
      4;
    const emissive_light_bytes =
      (4 + Math.max(1, Math.floor(context.config.max_emissive_lights ?? 32768)) * 12) * 4;
    const scheduling_bytes =
      this.params_data.byteLength +
      (context.total_patches + context.bootstrap_patch_capacity) * 4 +
      this.counters_data.byteLength +
      SURFACE_CACHE_DISPATCH_ARGS_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT;
    const output_texture_count = context.config.screen_reconstruction_enabled === false ? 1 : 3;
    const output_bytes = context.width * context.height * 8 * output_texture_count + 8;

    return {
      strategy: "scgi",
      width: context.width,
      height: context.height,
      total_patch_count: context.total_patches,
      active_patch_count,
      update_patch_count,
      bootstrap_patch_count,
      available_bootstrap_patch_count,
      pending_bootstrap_patch_count: Math.max(
        available_bootstrap_patch_count - bootstrap_patch_count,
        0
      ),
      deferred_patch_count,
      rays_per_patch: context.rays_per_patch,
      bootstrap_rays_per_patch,
      maximum_bootstrap_rays_per_patch: context.bootstrap_rays_per_patch,
      bootstrap_ray_budget_fraction: clamp(
        context.config.bootstrap_ray_budget_fraction ?? 0.5,
        0.0,
        1.0
      ),
      bootstrap_patch_capacity: context.bootstrap_patch_capacity,
      total_rays_fired:
        update_patch_count * context.rays_per_patch +
        bootstrap_patch_count * bootstrap_rays_per_patch,
      active_set_ray_budget: active_patch_count * context.rays_per_patch,
      maximum_ray_count_per_frame: context.maximum_ray_count_per_frame,
      maximum_ray_count: context.total_ray_count,
      max_ray_length: context.config.max_ray_length,
      cache_entry_lifetime: context.config.cache_entry_lifetime,
      cache_pixel_footprint: context.config.cache_pixel_footprint,
      history_footprint_start_samples: context.config.history_footprint_start_samples ?? 0,
      history_footprint_end_samples: context.config.history_footprint_end_samples ?? 0,
      history_footprint_max_scale: context.config.history_footprint_max_scale ?? 1,
      cache_normal_bias: context.config.cache_normal_bias,
      hash_search_count: context.config.hash_search_count,
      history_hysteresis: context.config.history_hysteresis,
      max_history_samples: context.config.max_history_samples,
      mature_patch_update_period: context.mature_patch_update_period,
      force_full_update: (this.counters_data[5] || 0) !== 0,
      surface_cache_bytes,
      hashmap_bytes,
      sh_bytes,
      ray_working_set_bytes,
      emissive_light_bytes,
      scheduling_bytes,
      output_bytes,
      total_memory_bytes:
        surface_cache_bytes +
        hashmap_bytes +
        sh_bytes +
        ray_working_set_bytes +
        emissive_light_bytes +
        scheduling_bytes +
        output_bytes,
    };
  }
}

/**
 * Per-pixel tracing, hit shading, and temporal/spatial RGB accumulation.
 * Disable shading and accumulation to record only the tracing portion.
 */
export class PerPixelRadianceCache extends GIModule {
  constructor({ stages = {} } = {}) {
    super({
      name: "per-pixel-radiance-cache",
      representation: "per-pixel-radiance-cache",
      stages,
      shader_setups: {
        reset: compute_shader("gi/gi_reset.wgsl"),
        compact_emissive: compute_shader("system_compute/compact_emissive_lights.wgsl"),
        trace_init: compute_shader("gi/pixel_trace_init.wgsl"),
        trace_hit: compute_shader("gi/pixel_trace_hit.wgsl"),
        shade: compute_shader("gi/pixel_trace_shade.wgsl"),
        temporal: compute_shader("gi/pixel_temporal_reservoir.wgsl"),
        spatial_wide: compute_shader("gi/pixel_spatial_reservoir_wide.wgsl"),
        spatial_narrow: compute_shader("gi/pixel_spatial_reservoir_narrow.wgsl"),
        accumulate: compute_shader("gi/pixel_accumulate.wgsl"),
        resolve: compute_shader("gi/pixel_upscale_final.wgsl"),
        atrous: compute_shader("gi/ddgi_atrous_diffuse.wgsl"),
      },
    });
    this.params_data = new Float32Array(12);
  }

  _setup_trace_resources(render_graph, context) {
    const { config, rays_per_frame, force_recreate } = context;
    this.create_buffer(render_graph, "params", {
      name: "pixel_trace_params",
      size: this.params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "counters", {
      name: "pixel_trace_counters",
      size: 6,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "path_state", {
      name: "gi_pixel_path_state",
      size: rays_per_frame * 15 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "ray_queue", {
      name: "gi_pixel_ray_queue",
      size: rays_per_frame,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "emissive_lights", {
      name: "pixel_trace_emissive_lights",
      size: 4 + Math.max(1, Math.floor(config.max_emissive_lights)) * 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.import_resource(
      "entity_index_lookup",
      render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name)
    );
    this.import_resource(
      "blue_noise",
      render_graph.register_image(Texture.default_blue_noise().config.name)
    );
  }

  _setup_shading_resources(render_graph) {
    this.material_buffers = register_material_buffers(render_graph);
    this.texture_pools = register_texture_pools(render_graph);
    this.scene_lighting_data = register_scene_lighting_data(render_graph);
    this.import_resource(
      "entity_index_lookup",
      render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name)
    );
  }

  _setup_accumulation_resources(render_graph, context) {
    const { width, height, gi_width, gi_height, force_recreate } = context;
    const reservoir_size = gi_width * gi_height * 28;
    this.create_buffer(render_graph, "temporal_reservoir_0", {
      name: "gi_temporal_reservoir_0",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "temporal_reservoir_1", {
      name: "gi_temporal_reservoir_1",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "spatial_reservoir_0", {
      name: "gi_spatial_reservoir_0",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "spatial_reservoir_1", {
      name: "gi_spatial_reservoir_1",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "spatial_reservoir_stage", {
      name: "gi_spatial_reservoir_stage",
      size: reservoir_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_image(render_graph, "radiance_direct_0", {
      name: "gi_low_radiance_direct_0",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "radiance_direct_1", {
      name: "gi_low_radiance_direct_1",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "radiance_diffuse_0", {
      name: "gi_low_radiance_indirect_diffuse_0",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "radiance_diffuse_1", {
      name: "gi_low_radiance_indirect_diffuse_1",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "radiance_specular_0", {
      name: "gi_low_radiance_indirect_specular_0",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "radiance_specular_1", {
      name: "gi_low_radiance_indirect_specular_1",
      format: "rgba16float",
      width: gi_width,
      height: gi_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "direct_output", {
      name: "pixel_rgb_direct_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "diffuse_output", {
      name: "pixel_rgb_diffuse_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "specular_output", {
      name: "pixel_rgb_specular_output",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "atrous_ping", {
      name: "gi_diffuse_atrous_ping",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.create_image(render_graph, "atrous_pong", {
      name: "gi_diffuse_atrous_pong",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    this.atrous_params_data = new Float32Array([1, 0.04, 64.0, 1.0]);
    this.create_buffer(render_graph, "atrous_params", {
      name: "gi_diffuse_atrous_params",
      size: this.atrous_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
  }

  setup(render_graph, context, branch) {
    this._setup_trace_resources(render_graph, context, branch);
    this._setup_shading_resources(render_graph, context, branch);
    this._setup_accumulation_resources(render_graph, context, branch);
  }

  _record_trace_passes(render_graph, context, branch) {
    const inputs = context.inputs;
    const params = this.get_resource("params");
    const counters = this.get_resource("counters");
    const emissive_lights = this.get_resource("emissive_lights");
    const entity_index_lookup = this.get_resource("entity_index_lookup");
    const path_state = this.get_resource("path_state");
    const ray_queue = this.get_resource("ray_queue");
    const materials = this.material_buffers;
    const textures = this.texture_pools;

    this.add_graph_local_pass(render_graph, "pixel_trace_upload_params", (graph) => {
      this.params_data[0] = context.config.screen_ray_count;
      this.params_data[1] = context.total_pixels;
      this.params_data[2] = context.frame_index;
      this.params_data[3] = context.config.indirect_boost;
      this.params_data[4] = context.safe_upscale_factor;
      this.params_data[5] = context.width;
      this.params_data[6] = context.height;
      this.params_data[7] = context.gi_width;
      this.params_data[8] = context.gi_height;
      this.params_data[9] = context.config.max_ray_length;
      this.params_data[10] = 0;
      this.params_data[11] = 0;
      graph.get_physical_buffer(params).write_raw(this.params_data);
    });

    this.add_compute_pass(
      render_graph,
      "compact_emissive",
      "pixel_trace_compact_emissive_lights",
      {
        inputs: [
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_directory,
          inputs.index_buffer,
          inputs.entity_transforms,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          entity_index_lookup,
          emissive_lights,
          textures.albedo,
          textures.emission,
        ],
        outputs: [emissive_lights],
      },
      (graph, frame_data) => {
        graph.get_physical_buffer(emissive_lights).write_raw(new Uint32Array([0, 0, 0, 0]), 0);
        const bounds = graph.get_physical_buffer(inputs.tlas_bvh2_bounds);
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(Math.floor(bounds.config.size / 32) / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );
    this.add_compute_pass(
      render_graph,
      "reset",
      "pixel_trace_reset_counters",
      { inputs: [counters, inputs.dense_lights], outputs: [counters] },
      (graph, frame_data) => graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1)
    );

    this.add_compute_pass(
      render_graph,
      "trace_init",
      `pixel_trace_init_${context.ping_pong_frame}`,
      {
        inputs: [
          params,
          counters,
          path_state,
          ray_queue,
          inputs.dense_lights,
          emissive_lights,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_albedo,
          inputs.gbuffer_smra,
          inputs.gbuffer_motion_emissive,
          this.get_resource("blue_noise"),
        ],
        outputs: [path_state, ray_queue],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.rays_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "trace_hit",
      "pixel_trace_hits",
      {
        inputs: [
          params,
          counters,
          path_state,
          ray_queue,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          inputs.compact_transforms,
          inputs.index_buffer,
          entity_index_lookup,
        ],
        outputs: [path_state],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.rays_per_frame / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
  }

  _record_shading_passes(render_graph, context, branch) {
    const surface = branch.get_dependency("radiance_cache").module;
    const trace = this;
    const materials = this.material_buffers;
    const textures = this.texture_pools;
    const lighting = this.scene_lighting_data;
    const path_state = trace.get_resource("path_state");
    this.add_compute_pass(
      render_graph,
      "shade",
      "pixel_rgb_shade_hits",
      {
        inputs: [
          trace.get_resource("params"),
          lighting.scene_lighting_buffer,
          path_state,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          surface.get_resource("surface_cache"),
          this.get_resource("entity_index_lookup"),
          textures.albedo,
          textures.normal,
          textures.roughness,
          textures.metallic,
          textures.ao,
          textures.height,
          textures.specular,
          textures.emission,
          lighting.skybox_image,
          surface.get_resource("params"),
          surface.get_resource("surface_cache_sh"),
          surface.get_resource("hashmap_entries"),
        ],
        outputs: [path_state],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.rays_per_frame / 128), 1, 1)
    );
  }

  _record_accumulation_passes(render_graph, context, branch) {
    const trace = this;
    const inputs = context.inputs;
    const params = trace.get_resource("params");
    const path_state = trace.get_resource("path_state");
    const frame = context.ping_pong_frame;
    const temporal_prev = this.get_resource(`temporal_reservoir_${frame}`);
    const temporal_curr = this.get_resource(`temporal_reservoir_${1 - frame}`);
    const spatial_curr = this.get_resource(`spatial_reservoir_${1 - frame}`);
    const spatial_stage = this.get_resource("spatial_reservoir_stage");
    const previous = {
      direct: this.get_resource(`radiance_direct_${frame}`),
      diffuse: this.get_resource(`radiance_diffuse_${frame}`),
      specular: this.get_resource(`radiance_specular_${frame}`),
    };
    const current = {
      direct: this.get_resource(`radiance_direct_${1 - frame}`),
      diffuse: this.get_resource(`radiance_diffuse_${1 - frame}`),
      specular: this.get_resource(`radiance_specular_${1 - frame}`),
    };
    this.add_compute_pass(
      render_graph,
      "temporal",
      `pixel_rgb_temporal_reservoir_${frame}`,
      {
        inputs: [
          params,
          path_state,
          temporal_prev,
          temporal_curr,
          inputs.depth_texture,
          inputs.prev_depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_motion_emissive,
          inputs.gbuffer_normal_prev,
        ],
        outputs: [temporal_curr],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.gi_width / 16), Math.ceil(context.gi_height / 16), 1)
    );
    this.add_compute_pass(
      render_graph,
      "spatial_wide",
      `pixel_rgb_spatial_reservoir_wide_${frame}`,
      {
        inputs: [
          params,
          temporal_curr,
          spatial_stage,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_smra,
        ],
        outputs: [spatial_stage],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.gi_width / 16), Math.ceil(context.gi_height / 16), 1)
    );
    this.add_compute_pass(
      render_graph,
      "spatial_narrow",
      `pixel_rgb_spatial_reservoir_narrow_${frame}`,
      {
        inputs: [
          params,
          spatial_stage,
          spatial_curr,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_smra,
        ],
        outputs: [spatial_curr],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.gi_width / 16), Math.ceil(context.gi_height / 16), 1)
    );
    this.add_compute_pass(
      render_graph,
      "accumulate",
      `pixel_rgb_accumulate_${frame}`,
      {
        inputs: [
          params,
          spatial_curr,
          previous.direct,
          previous.diffuse,
          previous.specular,
          inputs.depth_texture,
          inputs.prev_depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_normal_prev,
          inputs.gbuffer_motion_emissive,
          current.direct,
          current.diffuse,
          current.specular,
        ],
        outputs: [current.direct, current.diffuse, current.specular],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.gi_width / 8), Math.ceil(context.gi_height / 8), 1)
    );

    const direct_output = this.get_resource("direct_output");
    const diffuse_output = this.get_resource("diffuse_output");
    const specular_output = this.get_resource("specular_output");
    this.add_compute_pass(
      render_graph,
      "resolve",
      `pixel_rgb_resolve_${frame}`,
      {
        inputs: [
          params,
          current.direct,
          current.diffuse,
          current.specular,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_smra,
          direct_output,
          diffuse_output,
          specular_output,
        ],
        outputs: [direct_output, diffuse_output, specular_output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );

    let final_diffuse = diffuse_output;
    let read = diffuse_output;
    let write = this.get_resource("atrous_ping");
    const pass_count =
      context.config.diffuse_atrous_enabled === false
        ? 0
        : Math.max(0, Math.floor(context.config.diffuse_atrous_pass_count || 0));
    for (let pass_index = 0; pass_index < pass_count; pass_index++) {
      this.add_graph_local_pass(
        render_graph,
        `pixel_rgb_atrous_upload_params_${frame}_${pass_index}`,
        (graph) => {
          this.atrous_params_data[0] = Math.pow(2, pass_index);
          this.atrous_params_data[1] = Math.max(
            0.0001,
            context.config.diffuse_atrous_phi_depth || 0.04
          );
          this.atrous_params_data[2] = Math.max(1, context.config.diffuse_atrous_phi_normal || 64);
          this.atrous_params_data[3] = Math.max(
            0.0001,
            context.config.diffuse_atrous_luma_sigma || 1
          );
          graph
            .get_physical_buffer(this.get_resource("atrous_params"))
            .write_raw(this.atrous_params_data);
        }
      );
      this.add_compute_pass(
        render_graph,
        "atrous",
        `pixel_rgb_atrous_${frame}_${pass_index}`,
        {
          inputs: [
            this.get_resource("atrous_params"),
            read,
            inputs.depth_texture,
            inputs.gbuffer_normal,
            write,
          ],
          outputs: [write],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
      );
      final_diffuse = write;
      read = write;
      write =
        write === this.get_resource("atrous_ping")
          ? this.get_resource("atrous_pong")
          : this.get_resource("atrous_ping");
    }
    this.import_resource("diffuse_output", final_diffuse);
  }

  record(render_graph, context, branch) {
    if (this.is_stage_enabled(GIPipelineStage.Trace))
      this._record_trace_passes(render_graph, context, branch);
    if (this.is_stage_enabled(GIPipelineStage.Shading))
      this._record_shading_passes(render_graph, context, branch);
    if (this.is_stage_enabled(GIPipelineStage.Accumulation))
      this._record_accumulation_passes(render_graph, context, branch);
  }
}

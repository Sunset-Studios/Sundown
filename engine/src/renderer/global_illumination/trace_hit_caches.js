import { GITraceHitCache } from "./gi_pipeline.js";
import { SharedFrameInfoBuffer, SharedViewBuffer } from "../../core/shared_data.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";
import { Texture } from "../texture.js";
import { Buffer } from "../buffer.js";
import { ispot, npot } from "../../utility/math.js";

const COMPUTE_WORKGROUP_SIZE = 128;
const PROBE_SCHEDULER_PRIORITY_COUNT = 2;
const MAX_PROBE_CASCADES = 6;
const PROBE_COUNTERS_NAME = "probe_volume_gi_counters";

const shader = (path) => ({ pipeline_shaders: { compute: { path } } });

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

export const GIHitRepresentation = Object.freeze({
  PROBE_RAYS: "probe-rays-v1",
  SURFACE_PATCH_RAYS: "surface-patch-rays-v1",
  PIXEL_PATHS: "pixel-paths-v1",
});

export class ScrollingProbeVolumeTraceHitCache extends GITraceHitCache {
  constructor(options = {}) {
    super({
      name: "scrolling-cascaded-probe-volume",
      representation: "scrolling-cascaded-probe-volume",
      hit_representation: GIHitRepresentation.PROBE_RAYS,
      shader_setups: {
        reset: shader("gi/ddgi_reset.wgsl"),
        scroll_reset: shader("gi/ddgi_probe_scroll_reset.wgsl"),
        feedback_clear: shader("gi/ddgi_probe_surface_feedback_clear.wgsl"),
        feedback: shader("gi/ddgi_probe_surface_feedback.wgsl"),
        active_mark: shader("gi/ddgi_probe_active_mark.wgsl"),
        active_prefix_sum: shader("gi/ddgi_probe_active_prefix_sum.wgsl"),
        active_block_scan: shader("gi/ddgi_probe_active_block_prefix_scan.wgsl"),
        compact: shader("gi/ddgi_probe_indices_init.wgsl"),
        trace_init: shader("gi/ddgi_probe_trace_init.wgsl"),
        trace_hit: shader("gi/ddgi_probe_trace_hit.wgsl"),
        classify: shader("gi/ddgi_probe_state_classify.wgsl"),
        compact_emissive: shader("system_compute/compact_emissive_lights.wgsl"),
        ...options.shader_setups,
      },
    });
    this.params_data = new Float32Array(32 + 16 * MAX_PROBE_CASCADES);
    this.snapped_origins = null;
    this.scroll_offsets = null;
    this.initialized = null;
    this.counters_buffer = null;
    this.counters_data = null;
  }

  setup(render_graph, context) {
    const config = context.config;
    const requested_dims = config.probe_grid_dimensions ?? [32, 32, 32];
    const grid_dims = requested_dims.map((value) => {
      const dimension = Math.max(1, Math.floor(Number(value) || 1));
      return ispot(dimension) ? dimension : npot(dimension);
    });
    const cascade_count = Math.min(
      MAX_PROBE_CASCADES,
      Math.max(1, Math.floor(config.cascade_count || 1))
    );
    const probes_per_cascade = grid_dims[0] * grid_dims[1] * grid_dims[2];
    const probe_count = probes_per_cascade * cascade_count;
    const configured_budget = Math.max(0, Math.floor(config.probes_per_frame));
    const probes_per_frame =
      configured_budget === 0 ? probe_count : Math.min(probe_count, configured_budget);
    const max_rays_per_probe = Math.max(1, Math.floor(config.max_rays_per_probe));
    const probe_total_ray_count = probes_per_frame * max_rays_per_probe;
    const depth_resolutions = Array.from({ length: cascade_count }, (_, index) => {
      const configured = config.probe_depth_resolutions?.[index] ?? 16;
      const resolution = Math.max(4, Math.min(64, Math.floor(configured)));
      return ispot(resolution) ? resolution : npot(resolution);
    });

    if (!this.snapped_origins || this.snapped_origins.length !== cascade_count) {
      this.snapped_origins = Array.from({ length: cascade_count }, () => new Float32Array(3));
      this.scroll_offsets = Array.from({ length: cascade_count }, () => new Int32Array(3));
      this.initialized = Array.from({ length: cascade_count }, () => false);
    }

    const view = SharedViewBuffer.get_view_data(SharedFrameInfoBuffer.get_view_index());
    const camera = view.view_position;
    const cascade_data = new Float32Array(cascade_count * 16);
    let total_depth_texel_count = 0;
    let max_depth_texel_count_per_probe = 1;
    let cascade_has_scroll = false;
    for (let cascade = 0; cascade < cascade_count; cascade++) {
      const spacing =
        config.probe_spacing * Math.pow(Math.max(1, config.cascade_spacing_multiplier), cascade);
      const half = grid_dims.map((dimension) => (dimension - 1) * 0.5 * spacing);
      const next = camera
        .slice(0, 3)
        .map((position, axis) => Math.floor(position / spacing) * spacing - half[axis]);
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

    Object.assign(context, {
      grid_dims,
      cascade_count,
      probes_per_cascade,
      probe_count,
      probes_per_frame,
      max_rays_per_probe,
      probe_total_ray_count,
      total_depth_texel_count,
      max_depth_texel_count_per_probe,
      cascade_data,
      cascade_has_scroll,
      grid_log2: grid_dims.map((value) => Math.round(Math.log2(value))),
      grid_mask: grid_dims.map((value) => value - 1),
      frame_index: SharedFrameInfoBuffer.get_frame_index(),
    });
    context.ping_pong_frame = context.frame_index % 2;

    const create = (
      semantic,
      name,
      size,
      usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    ) =>
      this.create_buffer(render_graph, semantic, {
        name,
        size,
        usage,
        force: context.force_recreate,
      });
    create(
      "params",
      "probe_volume_params",
      this.params_data.length,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    create("update_indices", "probe_volume_update_indices", probes_per_frame);
    create("surface_flags", "probe_volume_surface_flags", probe_count);
    create("active_flags", "probe_volume_active_flags", probe_count);
    create(
      "active_prefix_sum",
      "probe_volume_active_prefix_sum",
      probe_count * PROBE_SCHEDULER_PRIORITY_COUNT
    );
    const block_count = Math.ceil(probe_count / COMPUTE_WORKGROUP_SIZE);
    create(
      "active_block_sums",
      "probe_volume_active_block_sums",
      block_count * PROBE_SCHEDULER_PRIORITY_COUNT
    );
    create(
      "active_block_prefixes",
      "probe_volume_active_block_prefixes",
      block_count * PROBE_SCHEDULER_PRIORITY_COUNT + PROBE_SCHEDULER_PRIORITY_COUNT
    );
    create("ray_hits", "probe_volume_ray_hits", 4 + probe_total_ray_count * 40);
    create("probe_states", "probe_volume_states", probe_count * 2);
    create(
      "emissive_lights",
      "probe_volume_emissive_lights",
      4 + Math.max(1, Math.floor(config.max_emissive_lights)) * 12
    );
    this.import_resource(
      "entity_index_lookup",
      render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name)
    );
    this._ensure_counters(context.force_recreate);
    this.import_resource("counters", render_graph.register_buffer(PROBE_COUNTERS_NAME));
  }

  _ensure_counters(force_recreate) {
    if (!force_recreate && this.counters_buffer) return;
    if (this.counters_buffer) this.counters_buffer.destroy();
    this.counters_data = new Uint32Array(6);
    this.counters_buffer = Buffer.create({
      name: PROBE_COUNTERS_NAME,
      raw_data: this.counters_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
    });
  }

  record(render_graph, context, branch) {
    const { config, inputs } = context;
    const accumulator = branch.accumulator;
    const shade = branch.shading_strategy;
    const resource = (name) => this.get_resource(name);
    const params = resource("params");
    const counters = resource("counters");
    const ray_hits = resource("ray_hits");
    const states = resource("probe_states");
    const update_indices = resource("update_indices");
    const surface_flags = resource("surface_flags");
    const active_flags = resource("active_flags");
    const active_prefix_sum = resource("active_prefix_sum");
    const active_block_sums = resource("active_block_sums");
    const active_block_prefixes = resource("active_block_prefixes");
    const emissive_lights = resource("emissive_lights");
    const entity_index_lookup = resource("entity_index_lookup");
    const sh_probes = accumulator.get_resource("sh_probes");
    const depth_moments = accumulator.get_resource("depth_moments");
    const msme_stats = accumulator.get_resource("msme_stats");
    const materials = shade.material_buffers;
    const textures = shade.texture_pools;
    const permutation = permutation_params(context.probe_count);

    this.add_graph_local_pass(render_graph, "probe_volume_upload_params", (graph) => {
      const primary_origin = this.snapped_origins[0];
      const values = [
        context.probe_count,
        context.max_rays_per_probe,
        context.probes_per_frame,
        config.probe_spacing,
        ...context.grid_dims,
        config.probe_radius,
        ...primary_origin,
        0,
        ...context.grid_log2,
        0,
        ...context.grid_mask,
        0,
        0,
        0,
        0,
        0,
        context.frame_index,
        config.indirect_boost,
        context.cascade_count,
        0,
        permutation.stride,
        permutation.base_offset,
        permutation.frame_stride,
        0,
      ];
      this.params_data.set(values, 0);
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
        graph.get_physical_buffer(emissive_lights).write_raw(new Uint32Array([0, 0, 0, 0]), 0);
        const bounds = graph.get_physical_buffer(inputs.tlas_bvh2_bounds);
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(Math.floor(bounds.config.size / 32) / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    const dispatch = (semantic, name, pass_inputs, outputs, x, y = 1) =>
      this.add_compute_pass(
        render_graph,
        semantic,
        name,
        { inputs: pass_inputs, outputs },
        (graph, frame_data) => graph.get_physical_pass(frame_data.current_pass).dispatch(x, y, 1)
      );
    dispatch(
      "reset",
      "probe_volume_reset",
      [counters, inputs.dense_lights, ray_hits],
      [counters, ray_hits],
      1
    );
    if (context.cascade_has_scroll) {
      dispatch(
        "scroll_reset",
        "probe_volume_scroll_reset",
        [params, sh_probes, depth_moments, states, msme_stats],
        [sh_probes, depth_moments, states, msme_stats],
        Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE)
      );
    }
    dispatch(
      "feedback_clear",
      "probe_volume_feedback_clear",
      [params, surface_flags],
      [surface_flags],
      Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE)
    );
    dispatch(
      "feedback",
      "probe_volume_feedback",
      [params, inputs.hzb_texture, inputs.gbuffer_normal, surface_flags],
      [surface_flags],
      Math.ceil(context.width / 8),
      Math.ceil(context.height / 8)
    );
    dispatch(
      "active_mark",
      "probe_volume_active_mark",
      [params, states, surface_flags, active_flags],
      [active_flags, states],
      Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE)
    );
    dispatch(
      "active_prefix_sum",
      "probe_volume_active_prefix_sum",
      [active_flags, active_prefix_sum, active_block_sums],
      [active_prefix_sum, active_block_sums],
      Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE)
    );
    dispatch(
      "active_block_scan",
      "probe_volume_active_block_prefix_scan",
      [active_block_sums, active_block_prefixes, counters, params],
      [active_block_prefixes, counters],
      1
    );
    dispatch(
      "compact",
      "probe_volume_compact_active_probes",
      [params, update_indices, active_flags, active_prefix_sum, active_block_prefixes, counters],
      [update_indices],
      Math.ceil(context.probe_count / COMPUTE_WORKGROUP_SIZE)
    );
    dispatch(
      "trace_init",
      "probe_volume_trace_init",
      [params, update_indices, ray_hits, counters],
      [ray_hits],
      Math.ceil(context.max_rays_per_probe / 16),
      Math.ceil(context.probes_per_frame / 16)
    );
    dispatch(
      "trace_hit",
      "probe_volume_trace_hits",
      [
        params,
        ray_hits,
        inputs.tlas_bvh2_bounds,
        inputs.tlas_bvh_info,
        inputs.blas_bvh2_nodes,
        inputs.blas_directory,
        inputs.entity_transforms,
        inputs.index_buffer,
        inputs.dense_lights,
        emissive_lights,
        entity_index_lookup,
      ],
      [ray_hits],
      Math.ceil(context.probe_total_ray_count / COMPUTE_WORKGROUP_SIZE)
    );
  }

  record_post_accumulation(render_graph, context) {
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
      total_rays_fired: update_count * context.max_rays_per_probe,
      active_probe_count: this.counters_data[2] || 0,
      probe_update_count: update_count,
      light_count: this.counters_data[0] || 0,
      probe_spacing: context.config.probe_spacing,
      probe_radius: context.config.probe_radius,
    };
  }
}

export class HashedSurfaceTraceHitCache extends GITraceHitCache {
  constructor({ shader_setups = {} } = {}) {
    super({
      name: "hashed-surface-cache",
      representation: "hashed-surface-cache",
      hit_representation: GIHitRepresentation.SURFACE_PATCH_RAYS,
      shader_setups: {
        evict: shader("gi/surface_cache_evict.wgsl"),
        feedback: shader("gi/surface_cache_feedback.wgsl"),
        trace_hit: shader("gi/surface_cache_trace_hit.wgsl"),
        prepare_transforms: shader("gi/ray_instance_transform_prepare.wgsl"),
        ...shader_setups,
      },
    });
    this.params_data = new Float32Array(16);
  }

  setup(render_graph, context, branch) {
    const { config, width, height, force_recreate } = context;
    const cache_size = Math.max(16, Math.floor(config.surface_cache_size / 16) * 16);
    const lod_count = Math.max(1, Math.floor(config.surface_cache_lod_count));
    const total_patches = cache_size * lod_count;
    const entity_transform_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      "transforms"
    );
    const entity_transform_count = entity_transform_buffer.max_rows;

    Object.assign(context, { cache_size, lod_count, total_patches, entity_transform_count });

    this.create_buffer(render_graph, "params", {
      name: "surface_cache_params",
      size: this.params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "surface_cache", {
      name: "surface_cache_elements",
      size: total_patches * 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "active_indices", {
      name: "surface_cache_active_indices",
      size: total_patches,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "counters", {
      name: "surface_cache_counters",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "hit_info", {
      name: "surface_cache_hit_info",
      size: total_patches * 28,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "ray_instance_transforms", {
      name: "surface_cache_ray_instance_transforms",
      size: entity_transform_count * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.import_resource(
      "entity_index_lookup",
      render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name)
    );
  }

  record(render_graph, context, branch) {
    const { config, width, height, total_patches, entity_transform_count, inputs } = context;
    const params = this.get_resource("params");
    const surface_cache = this.get_resource("surface_cache");
    const active_indices = this.get_resource("active_indices");
    const counters = this.get_resource("counters");
    const hit_info = this.get_resource("hit_info");
    const ray_instance_transforms = this.get_resource("ray_instance_transforms");
    const entity_index_lookup = this.get_resource("entity_index_lookup");
    const sh = branch.accumulator.get_resource("surface_cache_sh");
    const sh_filtered = branch.accumulator.get_resource("surface_cache_sh_filtered");

    this.add_graph_local_pass(render_graph, "surface_cache_upload_params", (graph) => {
      this.params_data[0] = context.cache_size;
      this.params_data[1] = config.surface_cache_cell_size;
      this.params_data[2] = context.lod_count;
      this.params_data[3] = total_patches;
      this.params_data[4] = width;
      this.params_data[5] = height;
      this.params_data[6] = SharedFrameInfoBuffer.get_frame_index();
      this.params_data[7] = config.max_ray_length;
      this.params_data[8] = config.history_hysteresis;
      this.params_data[9] = config.max_history_samples;
      this.params_data[10] = config.indirect_boost;
      this.params_data[11] = Math.max(2, Math.min(16, Math.floor(config.importance_sample_count)));
      this.params_data[12] = config.cache_entry_lifetime;
      this.params_data[13] = Math.max(0.01, Math.min(1.0, config.importance_exploration));
      this.params_data[14] = 0;
      this.params_data[15] = 0;
      graph.get_physical_buffer(params).write_raw(this.params_data);
    });

    this.add_compute_pass(
      render_graph,
      "prepare_transforms",
      "surface_cache_prepare_ray_transforms",
      {
        inputs: [inputs.entity_transforms, ray_instance_transforms],
        outputs: [ray_instance_transforms],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(entity_transform_count / 128), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "evict",
      "surface_cache_evict",
      {
        inputs: [params, surface_cache, sh, sh_filtered, counters],
        outputs: [surface_cache, sh, sh_filtered, counters],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(total_patches / 128), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "feedback",
      "surface_cache_feedback",
      {
        inputs: [
          params,
          surface_cache,
          sh,
          counters,
          active_indices,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_albedo,
          inputs.gbuffer_smra,
          inputs.gbuffer_motion_emissive,
        ],
        outputs: [surface_cache, sh, counters, active_indices],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );
    this.add_compute_pass(
      render_graph,
      "trace_hit",
      "surface_cache_trace_hits",
      {
        inputs: [
          params,
          surface_cache,
          sh,
          active_indices,
          counters,
          hit_info,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          ray_instance_transforms,
          inputs.index_buffer,
          entity_index_lookup,
        ],
        outputs: [hit_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(total_patches / 128), 1, 1)
    );
  }
}

export class PerPixelTraceHitCache extends GITraceHitCache {
  constructor(options = {}) {
    super({
      name: "per-pixel-path-cache",
      representation: "per-pixel",
      hit_representation: GIHitRepresentation.PIXEL_PATHS,
      shader_setups: {
        reset: shader("gi/gi_reset.wgsl"),
        compact_emissive: shader("system_compute/compact_emissive_lights.wgsl"),
        prepare_transforms: shader("gi/ray_instance_transform_prepare.wgsl"),
        trace_init: shader("gi/pixel_trace_init.wgsl"),
        trace_hit: shader("gi/pixel_trace_hit.wgsl"),
        ...options.shader_setups,
      },
    });
    this.params_data = new Float32Array(16);
  }

  setup(render_graph, context) {
    const { config, rays_per_frame, force_recreate } = context;
    const entity_transform_buffer = EntityManager.get_fragment_gpu_buffer(
      TransformFragment,
      "transforms"
    );
    this.entity_transform_count = entity_transform_buffer.max_rows;
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
    this.create_buffer(render_graph, "ray_instance_transforms", {
      name: "pixel_trace_ray_instance_transforms",
      size: this.entity_transform_count * 32,
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

  record(render_graph, context, branch) {
    const inputs = context.inputs;
    const params = this.get_resource("params");
    const counters = this.get_resource("counters");
    const emissive_lights = this.get_resource("emissive_lights");
    const ray_instance_transforms = this.get_resource("ray_instance_transforms");
    const entity_index_lookup = this.get_resource("entity_index_lookup");
    const path_state = this.get_resource("path_state");
    const ray_queue = this.get_resource("ray_queue");
    const materials = branch.shading_strategy.material_buffers;
    const textures = branch.shading_strategy.texture_pools;

    this.add_graph_local_pass(render_graph, "pixel_trace_upload_params", (graph) => {
      this.params_data[0] = context.config.screen_ray_count;
      this.params_data[1] = context.config.surface_cache_size;
      this.params_data[2] = context.config.surface_cache_cell_size;
      this.params_data[3] = context.total_pixels;
      this.params_data[4] = context.frame_index;
      this.params_data[5] = context.config.indirect_boost;
      this.params_data[6] = context.safe_upscale_factor;
      this.params_data[7] = context.config.surface_cache_lod_count;
      this.params_data[8] = context.width;
      this.params_data[9] = context.height;
      this.params_data[10] = context.gi_width;
      this.params_data[11] = context.gi_height;
      this.params_data[12] = context.config.max_ray_length;
      graph.get_physical_buffer(params).write_raw(this.params_data);
    });

    this.add_compute_pass(
      render_graph,
      "prepare_transforms",
      "pixel_trace_prepare_ray_transforms",
      {
        inputs: [inputs.entity_transforms, ray_instance_transforms],
        outputs: [ray_instance_transforms],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(this.entity_transform_count / COMPUTE_WORKGROUP_SIZE), 1, 1)
    );
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
          ray_instance_transforms,
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
}

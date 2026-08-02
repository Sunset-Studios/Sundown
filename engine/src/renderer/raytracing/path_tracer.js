// =============================================================================
// Path Tracer - Simple Monte Carlo Implementation
// =============================================================================
// Unbiased Monte Carlo path tracing with BRDF importance sampling.
// - Supports multiple bounces with Russian Roulette termination
// - Progressive frame accumulation for noise reduction
// - Next Event Estimation (NEE) for direct lighting
// - Optional G-buffer mode for hybrid rasterization/ray tracing
// - Configurable samples per pixel per frame
// =============================================================================

import { RenderPassFlags } from "../renderer_types.js";
import { RayTracer } from "./raytracer.js";
import {
  SharedFrameInfoBuffer,
  SharedViewBuffer,
  SharedEnvironmentData,
} from "../../core/shared_data.js";
import { MaterialAllocationTable } from "../material_allocation_table.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";
import { ResourceCache } from "../resource_cache.js";
import { CacheTypes } from "../renderer_types.js";
import { Name } from "../../utility/names.js";
import { Texture } from "../texture.js";

// ─────────────────────────────────────────────────────────────────────────────
// Constants & Configuration
// ─────────────────────────────────────────────────────────────────────────────
const material_offsets_name = "material_table_offset";
const texture_pool_albedo_name = Name.from("texture_pool_albedo");
const texture_pool_normal_name = Name.from("texture_pool_normal");
const texture_pool_roughness_name = Name.from("texture_pool_roughness");
const texture_pool_metallic_name = Name.from("texture_pool_metallic");
const texture_pool_ao_name = Name.from("texture_pool_ao");
const texture_pool_height_name = Name.from("texture_pool_height");
const texture_pool_specular_name = Name.from("texture_pool_specular");
const texture_pool_emission_name = Name.from("texture_pool_emission");

const COMPUTE_WORKGROUP_SIZE = 128;

function get_sampling_grid(trace_rate) {
  const rate = Math.max(1, Math.floor(trace_rate));
  let tile_height = Math.floor(Math.sqrt(rate));
  while (rate % tile_height !== 0) {
    tile_height--;
  }

  return {
    rate,
    tile_width: rate / tile_height,
    tile_height,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader Configurations
// ─────────────────────────────────────────────────────────────────────────────
const path_tracer_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "raytracing/path_trace_init.wgsl" },
  },
};

const path_tracer_hit_shader_setup = {
  pipeline_shaders: {
    compute: { path: "raytracing/path_trace_hit.wgsl" },
  },
};

const path_tracer_gbuffer_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "raytracing/path_trace_gbuffer_shade.wgsl" },
  },
};

const path_tracer_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "raytracing/path_trace_shade.wgsl" },
  },
};

const path_tracer_output_shader_setup = {
  pipeline_shaders: {
    compute: { path: "raytracing/path_trace_output.wgsl" },
  },
};

// =============================================================================
// PathTracer Class
// =============================================================================
export class PathTracer extends RayTracer {
  // ─────────────────────────────────────────────────────────────────────────
  // Parameters buffer layout:
  // [max_bounces, reset_accum, use_gbuffer, trace_rate, sampling_frame, samples_per_pixel, sample_index, sampling_tile_width, max_accumulation_frames]
  // ─────────────────────────────────────────────────────────────────────────
  params = new Uint32Array([
    0, // max_bounces
    0, // reset_accum_flag
    0, // use_gbuffer
    0, // trace_rate
    0, // sampling_frame
    1, // samples_per_pixel
    0, // sample_index
    1, // sampling_tile_width
    0, // max_accumulation_frames (0 = infinite)
  ]);
  frame_phase = 0;

  constructor() {
    super();
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Add Render Passes
  // ═══════════════════════════════════════════════════════════════════════════
  add_passes(
    render_graph,
    width,
    height,
    max_bounces = 2,
    trace_rate = 1,
    samples_per_pixel = 1,
    use_gbuffer = false,
    tlas_bvh2_bounds = null,
    tlas_bvh_info = null,
    blas_bvh2_nodes = null,
    blas_directory = null,
    entity_transforms = null,
    index_buffer = null,
    dense_lights = null,
    visibility_entity = null,
    visibility_surface = null,
    meshlet_buffer = null,
    meshlet_vertex_buffer = null,
    meshlet_triangle_buffer = null,
    depth_texture = null,
    gbuffer_normal = null,
    gbuffer_albedo = null,
    gbuffer_smra = null,
    gbuffer_emissive = null,
    entity_index_lookup = null,
    force_recreate = false,
    max_accumulation_frames = Infinity
  ) {
    if (
      max_accumulation_frames !== Infinity &&
      (!Number.isInteger(max_accumulation_frames) ||
        max_accumulation_frames < 1 ||
        max_accumulation_frames > 0xffffffff)
    ) {
      throw new RangeError(
        "max_accumulation_frames must be a positive uint32 integer or Infinity"
      );
    }

    super.setup(render_graph, width, height, force_recreate);

    const num_bounce_passes = max_bounces;
    const num_rays = width * height;
    const sampling_grid = get_sampling_grid(trace_rate);
    const normalized_trace_rate = sampling_grid.rate;
    const active_pixel_count =
      Math.ceil(width / sampling_grid.tile_width) *
      Math.ceil(height / sampling_grid.tile_height);
    const sampling_frame = this.frame_phase;

    const view_index = SharedFrameInfoBuffer.get_view_index();
    const view_moved = SharedViewBuffer.was_moved(view_index);

    // ─────────────────────────────────────────────────────────────────────────
    // Material System Buffers
    // ─────────────────────────────────────────────────────────────────────────
    const params_gpu = MaterialAllocationTable.params_buffer;
    const material_palette = MaterialAllocationTable.palette_buffer;
    const material_palette_offsets = EntityManager.get_fragment_gpu_buffer(
      StaticMeshFragment,
      material_offsets_name
    );

    const params_gpu_buffer = render_graph.register_buffer(params_gpu.config.name);
    const material_palette_buffer = render_graph.register_buffer(material_palette.config.name);
    const material_palette_offsets_buffer = render_graph.register_buffer(
      material_palette_offsets.buffer.config.name
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Texture Pools
    // ─────────────────────────────────────────────────────────────────────────
    const default_texture = Texture.default_array();
    const albedo_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_albedo_name);
    const normal_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_normal_name);
    const roughness_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_roughness_name);
    const metallic_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_metallic_name);
    const ao_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_ao_name);
    const height_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_height_name);
    const specular_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_specular_name);
    const emission_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, texture_pool_emission_name);

    const default_texture_buffer = render_graph.register_image(default_texture.config.name);
    const albedo_pool_buffer = albedo_pool
      ? render_graph.register_image(albedo_pool.config.name)
      : default_texture_buffer;
    const normal_pool_buffer = normal_pool
      ? render_graph.register_image(normal_pool.config.name)
      : default_texture_buffer;
    const roughness_pool_buffer = roughness_pool
      ? render_graph.register_image(roughness_pool.config.name)
      : default_texture_buffer;
    const metallic_pool_buffer = metallic_pool
      ? render_graph.register_image(metallic_pool.config.name)
      : default_texture_buffer;
    const ao_pool_buffer = ao_pool
      ? render_graph.register_image(ao_pool.config.name)
      : default_texture_buffer;
    const height_pool_buffer = height_pool
      ? render_graph.register_image(height_pool.config.name)
      : default_texture_buffer;
    const specular_pool_buffer = specular_pool
      ? render_graph.register_image(specular_pool.config.name)
      : default_texture_buffer;
    const emission_pool_buffer = emission_pool
      ? render_graph.register_image(emission_pool.config.name)
      : default_texture_buffer;

    // ─────────────────────────────────────────────────────────────────────────
    // Environment Sky Setup
    // ─────────────────────────────────────────────────────────────────────────
    const skydome_data = SharedEnvironmentData.get_skydome_data();
    const skydome_data_buffer = render_graph.register_buffer(skydome_data.config.name);

    const skybox = SharedEnvironmentData.get_skybox();
    const skybox_texture_buffer = render_graph.register_image(skybox.config.name);

    // ─────────────────────────────────────────────────────────────────────────
    // Path Tracing Buffers
    // PathState: 13 vec4<f32> = 52 floats per ray (includes primary_albedo for demodulation)
    // ─────────────────────────────────────────────────────────────────────────
    const path_state = render_graph.create_buffer({
      name: "pt_path_state",
      size: num_rays * 52,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const pt_params = render_graph.create_buffer({
      name: "pt_params",
      size: this.params.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    // ═══════════════════════════════════════════════════════════════════════
    // SAMPLE LOOP: Multiple samples per pixel per frame
    // ═══════════════════════════════════════════════════════════════════════
    for (let sample_idx = 0; sample_idx < samples_per_pixel; sample_idx++) {
      // ═════════════════════════════════════════════════════════════════════
      // PASS: Parameter Update for this sample
      // ═════════════════════════════════════════════════════════════════════
      render_graph.add_pass(
        `path_trace_params_${sample_idx}`,
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          const params_buffer = graph.get_physical_buffer(pt_params);
          this.params[0] = num_bounce_passes;
          this.params[1] = view_moved ? 1 : 0;
          this.params[2] = use_gbuffer ? 1 : 0;
          this.params[3] = normalized_trace_rate;
          this.params[4] = sampling_frame;
          this.params[5] = samples_per_pixel;
          this.params[6] = sample_idx;
          this.params[7] = sampling_grid.tile_width;
          this.params[8] =
            max_accumulation_frames === Infinity ? 0 : max_accumulation_frames;
          params_buffer.write_raw(this.params);
        }
      );

      // ═════════════════════════════════════════════════════════════════════
      // PASS: Initialize Path State for this sample
      // ═════════════════════════════════════════════════════════════════════
      render_graph.add_pass(
        `path_trace_init_${sample_idx}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            pt_params,
            path_state,
            visibility_entity,
            visibility_surface,
            entity_transforms,
            meshlet_buffer,
            meshlet_vertex_buffer,
            meshlet_triangle_buffer,
            depth_texture,
            gbuffer_normal,
            gbuffer_albedo,
            gbuffer_smra,
            gbuffer_emissive,
            this.output_texture,
          ],
          outputs: [path_state],
          shader_setup: path_tracer_init_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          const pixel_count = view_moved ? num_rays : active_pixel_count;
          pass.dispatch(Math.ceil(pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
        }
      );

      // ═════════════════════════════════════════════════════════════════════
      // PASS: G-Buffer Initial Shade (Bounce 0, when use_gbuffer enabled)
      // ═════════════════════════════════════════════════════════════════════
      if (use_gbuffer) {
        render_graph.add_pass(
          `path_trace_gbuffer_shade_${sample_idx}`,
          RenderPassFlags.Compute,
          {
            inputs: [
              pt_params,
              skydome_data_buffer,
              path_state,
              dense_lights,
              skybox_texture_buffer,
              this.output_texture,
            ],
            outputs: [path_state],
            shader_setup: path_tracer_gbuffer_shade_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
          }
        );
      }

      // ═════════════════════════════════════════════════════════════════════
      // BOUNCE LOOP: Hit (shadow + bounce) → Shade
      // ═════════════════════════════════════════════════════════════════════
      for (let b = 0; b < num_bounce_passes; b++) {
        // ───────────────────────────────────────────────────────────────────
        // PASS: Combined Hit (Shadow Ray + Bounce Ray Intersection)
        // ───────────────────────────────────────────────────────────────────
        render_graph.add_pass(
          `path_trace_hit_${sample_idx}_${b}`,
          RenderPassFlags.Compute,
          {
            inputs: [
              pt_params,
              path_state,
              tlas_bvh2_bounds,
              tlas_bvh_info,
              blas_bvh2_nodes,
              blas_directory,
              entity_transforms,
              index_buffer,
              entity_index_lookup,
              this.output_texture,
            ],
            outputs: [path_state],
            shader_setup: path_tracer_hit_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
          }
        );

        // ───────────────────────────────────────────────────────────────────
        // PASS: Shade and Generate Next Bounce
        // ───────────────────────────────────────────────────────────────────
        const shade_inputs_list = [
          pt_params,
          skydome_data_buffer,
          path_state,
          params_gpu_buffer,
          material_palette_offsets_buffer,
          material_palette_buffer,
          dense_lights,
          entity_index_lookup,
          albedo_pool_buffer,
          normal_pool_buffer,
          roughness_pool_buffer,
          metallic_pool_buffer,
          ao_pool_buffer,
          height_pool_buffer,
          specular_pool_buffer,
          emission_pool_buffer,
          skybox_texture_buffer,
          this.output_texture,
        ];

        render_graph.add_pass(
          `path_trace_shade_${sample_idx}_${b}`,
          RenderPassFlags.Compute,
          {
            inputs: shade_inputs_list,
            outputs: [path_state],
            shader_setup: path_tracer_shade_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
          }
        );
      }
    }

    // Keep a monotonic frame so each completed coverage cycle gets a new scramble.
    this.frame_phase = (this.frame_phase + 1) >>> 0;

    // ═══════════════════════════════════════════════════════════════════════
    // PASS: Output Final Result
    // ═══════════════════════════════════════════════════════════════════════
    render_graph.add_pass(
      "path_trace_output",
      RenderPassFlags.Compute,
      {
        inputs: [pt_params, path_state, this.output_texture],
        outputs: [this.output_texture],
        shader_setup: path_tracer_output_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }
}

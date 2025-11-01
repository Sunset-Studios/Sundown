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

const path_tracer_hit_visibility_shader_setup = {
  pipeline_shaders: {
    compute: { path: "raytracing/path_trace_hit_visibility.wgsl" },
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

export class PathTracer extends RayTracer {
  params = new Uint32Array([
    0, // max_bounces
    0, // spp_per_frame
    0, // reset_accum_flag
    0, // use_gbuffer
    0, // trace_rate
    0, // frame_phase
    0, // indirect_boost
    0, // padding
  ]);
  frame_phase = 0;

  constructor() {
    super();
  }

  add_passes(
    render_graph,
    width,
    height,
    max_bounces = 2,
    spp_per_frame = 4,
    trace_rate = 1,
    indirect_boost = 2.0,
    use_gbuffer = false,
    tlas_bvh2_bounds = null,
    tlas_bvh4_nodes = null,
    blas_atlas = null,
    entity_transforms = null,
    mesh_asset_ids = null,
    index_buffer = null,
    dense_lights = null,
    light_count = null,
    gbuffer_position = null,
    gbuffer_normal = null,
    gbuffer_albedo = null,
    gbuffer_smra = null,
    gbuffer_emissive = null,
    force_recreate = false
  ) {
    super.setup(render_graph, width, height, force_recreate);

    const num_bounce_passes = max_bounces;
    const num_rays = width * height;

    const view_index = SharedFrameInfoBuffer.get_view_index();
    const view_moved = SharedViewBuffer.was_moved(view_index);

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

    // === Environment Sky Setup (Skybox or Skydome) ===
    const skydome_data = SharedEnvironmentData.get_skydome_data();
    const skydome_data_buffer = render_graph.register_buffer(skydome_data.config.name);

    const skybox = SharedEnvironmentData.get_skybox();
    const skybox_texture_buffer = render_graph.register_image(skybox.config.name);

    const path_state = render_graph.create_buffer({
      name: "pt_path_state",
      size: num_rays * 36 * 4, // 9 vec4<f32> ≈ PathState
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const path_shade = render_graph.create_buffer({
      name: "pt_path_shade",
      size: num_rays * 20 * 4, // 5 vec4<f32> ≈ PathShade (3 original + 2 for reservoir)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    // no separate TLAS hits buffer; hit stage writes into path_state
    const pt_params = render_graph.create_buffer({
      name: "pt_params",
      size: this.params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    render_graph.add_pass(
      "path_trace_reset",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const params_buffer = graph.get_physical_buffer(pt_params);
        this.params[0] = num_bounce_passes; // max_bounces
        this.params[1] = spp_per_frame; // spp_per_frame
        this.params[2] = view_moved ? 1 : 0; // reset_accum_flag
        this.params[3] = use_gbuffer ? 1 : 0; // use_gbuffer
        this.params[4] = trace_rate; // trace_rate
        this.params[5] = this.frame_phase; // frame_phase
        this.params[6] = indirect_boost; // indirect_boost
        this.params[7] = 0; // padding
        params_buffer.write_raw(this.params);
        // Cycle frame phase for next frame
        this.frame_phase = (this.frame_phase + 1) % Math.max(1, trace_rate);
      }
    );

    const spp = Math.max(1, spp_per_frame | 0);

    for (let s = 0; s < spp; s++) {
      render_graph.add_pass(
        `path_trace_init_${s}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            pt_params,
            path_state,
            path_shade,
            gbuffer_position,
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
          const pixel_count = view_moved ? num_rays : Math.ceil(num_rays / Math.max(1, trace_rate));
          pass.dispatch(Math.ceil(pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
        }
      );

      // ═════════════════════════════════════════════════════════════════════════
      // G-Buffer Initial Shade Pass (Bounce 0 only, when use_gbuffer is enabled)
      // - Handles direct lighting and spawns first indirect ray
      // - Eliminates need for G-buffer checks in main bounce loop
      // ═════════════════════════════════════════════════════════════════════════
      if (use_gbuffer) {
        render_graph.add_pass(
          `path_trace_gbuffer_shade_${s}`,
          RenderPassFlags.Compute,
          {
            inputs: [
              pt_params,
              path_state,
              path_shade,
              dense_lights,
              light_count,
              this.output_texture,
            ],
            outputs: [path_state, path_shade],
            shader_setup: path_tracer_gbuffer_shade_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const active_pixel_count = Math.ceil(num_rays / Math.max(1, trace_rate));
            pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
          }
        );
      }

      for (let b = 0; b < num_bounce_passes; b++) {
        render_graph.add_pass(
          `path_trace_hit_visibility_${s}_${b}`,
          RenderPassFlags.Compute,
          {
            inputs: [
              pt_params,
              path_state,
              tlas_bvh2_bounds,
              tlas_bvh4_nodes,
              blas_atlas,
              entity_transforms,
              index_buffer,
              mesh_asset_ids,
              this.output_texture,
            ],
            outputs: [path_state],
            shader_setup: path_tracer_hit_visibility_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const active_pixel_count = Math.ceil(num_rays / Math.max(1, trace_rate));
            pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
          }
        );

        render_graph.add_pass(
          `path_trace_hit_${s}_${b}`,
          RenderPassFlags.Compute,
          {
            inputs: [
              pt_params,
              path_state,
              tlas_bvh2_bounds,
              tlas_bvh4_nodes,
              blas_atlas,
              entity_transforms,
              index_buffer,
              mesh_asset_ids,
              this.output_texture,
            ],
            outputs: [path_state],
            shader_setup: path_tracer_hit_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const active_pixel_count = Math.ceil(num_rays / Math.max(1, trace_rate));
            pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
          }
        );

        // Build shader inputs
        const shade_inputs_list = [
          pt_params,
          skydome_data_buffer,
          path_state,
          path_shade,
          params_gpu_buffer,
          material_palette_offsets_buffer,
          material_palette_buffer,
          dense_lights,
          light_count,
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
          `path_trace_shade_${s}_${b}`,
          RenderPassFlags.Compute,
          {
            inputs: shade_inputs_list,
            outputs: [path_state, path_shade],
            shader_setup: path_tracer_shade_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const active_pixel_count = Math.ceil(num_rays / Math.max(1, trace_rate));
            pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
          }
        );
      }
    }

    render_graph.add_pass(
      "path_trace_output",
      RenderPassFlags.Compute,
      {
        inputs: [path_shade, this.output_texture],
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

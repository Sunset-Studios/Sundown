import { RenderPassFlags } from "../renderer_types.js";
import { RayTracer } from "../raytracing/raytracer.js";
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

const world_cache_path_tracer_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/path_trace_world_cache_shade.wgsl" },
  },
};

const world_cache_path_tracer_update_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/path_trace_world_cache_update.wgsl" },
  },
};

const path_tracer_output_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/path_trace_world_cache_output.wgsl" },
  },
};

export class WorldCachePathTracer extends RayTracer {
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
    spp_per_frame = 1,
    trace_rate = 4,
    indirect_boost = 1.0,
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
    gi_params = null,
    world_cache = null,
    force_recreate = false
  ) {
    super.setup(render_graph, width, height, force_recreate);

    const num_bounce_passes = max_bounces;
    const num_rays = width * height;

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

    const skydome_data = SharedEnvironmentData.get_skydome_data();
    const skydome_data_buffer = render_graph.register_buffer(skydome_data.config.name);

    const skybox = SharedEnvironmentData.get_skybox();
    const skybox_texture_buffer = render_graph.register_image(skybox.config.name);

    const path_state = render_graph.create_buffer({
      name: "gi_pt_path_state",
      size: num_rays * 36 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const path_shade = render_graph.create_buffer({
      name: "gi_pt_path_shade",
      size: num_rays * 20 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const pt_params = render_graph.create_buffer({
      name: "gi_pt_params",
      size: this.params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    render_graph.add_pass(
      "gi_pt_reset",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const params_buffer = graph.get_physical_buffer(pt_params);
        this.params[0] = num_bounce_passes;
        this.params[1] = spp_per_frame;
        this.params[2] = 1; // reset_accum_flag
        this.params[3] = 1; // use_gbuffer
        this.params[4] = trace_rate; // trace_rate
        this.params[5] = this.frame_phase; // frame_phase
        this.params[6] = indirect_boost; // indirect_boost
        this.params[7] = 0; // padding
        params_buffer.write_raw(this.params);
        this.frame_phase = (this.frame_phase + 1) % Math.max(1, trace_rate);
      }
    );

    const spp = Math.max(1, spp_per_frame | 0);

    for (let s = 0; s < spp; s++) {
      render_graph.add_pass(
        `gi_pt_init_${s}`,
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
          //const pixel_count = Math.ceil(num_rays / Math.max(1, trace_rate));
          const pixel_count = num_rays;
          pass.dispatch(Math.ceil(pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
        }
      );

      render_graph.add_pass(
        `gi_pt_gbuffer_shade_${s}`,
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

      for (let b = 0; b < num_bounce_passes; b++) {
        render_graph.add_pass(
          `gi_pt_hit_visibility_${s}_${b}`,
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
          `gi_pt_hit_${s}_${b}`,
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
          world_cache,
          gi_params,
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
          `gi_pt_shade_${s}_${b}`,
          RenderPassFlags.Compute,
          {
            inputs: shade_inputs_list,
            outputs: [path_state, path_shade],
            shader_setup: world_cache_path_tracer_shade_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            const active_pixel_count = Math.ceil(num_rays / Math.max(1, trace_rate));
            pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
          }
        );
      }

      render_graph.add_pass(
        `gi_pt_world_cache_update_${s}`,
        RenderPassFlags.Compute,
        {
          inputs: [gi_params, pt_params, path_state, path_shade, world_cache, this.output_texture],
          outputs: [world_cache],
          shader_setup: world_cache_path_tracer_update_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          const active_pixel_count = Math.ceil(num_rays / Math.max(1, trace_rate));
          pass.dispatch(Math.ceil(active_pixel_count / COMPUTE_WORKGROUP_SIZE), 1, 1);
        }
      );
    }

    render_graph.add_pass(
      "gi_pt_output",
      RenderPassFlags.Compute,
      {
        inputs: [gi_params, pt_params, path_state, path_shade, world_cache, this.output_texture],
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

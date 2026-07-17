import { GIShadingStrategy } from "./gi_pipeline.js";
import { GIHitRepresentation } from "./trace_hit_caches.js";
import {
  register_material_buffers,
  register_texture_pools,
  register_scene_lighting_data,
} from "../render_graph_utils.js";
import { FragmentGpuBuffer } from "../../core/ecs/solar/memory.js";

const shader = (path) => ({ pipeline_shaders: { compute: { path } } });

export const GIRadianceRepresentation = Object.freeze({
  PROBE_SH: "probe-sh-radiance-v1",
  SURFACE_SH: "surface-sh-radiance-v1",
  PIXEL_RGB: "pixel-weighted-rgb-v1",
});

export class ProbeSHShadingStrategy extends GIShadingStrategy {
  constructor(options = {}) {
    super({
      name: "probe-sh-shading",
      representation: "probe-sh",
      accepted_hit_representations: [GIHitRepresentation.PROBE_RAYS],
      radiance_representation: GIRadianceRepresentation.PROBE_SH,
      shader_setups: {
        shade: shader("gi/ddgi_probe_trace_shade.wgsl"),
        ...options.shader_setups,
      },
    });
  }

  setup(render_graph) {
    this.material_buffers = register_material_buffers(render_graph);
    this.texture_pools = register_texture_pools(render_graph);
    this.scene_lighting_data = register_scene_lighting_data(render_graph);
  }

  record(render_graph, context, branch) {
    const trace = branch.trace_hit_cache;
    const accumulator = branch.accumulator;
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
          trace.get_resource("probe_states"),
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          accumulator.get_resource("sh_probes"),
          accumulator.get_resource("depth_moments"),
          trace.get_resource("entity_index_lookup"),
          textures.albedo,
          textures.normal,
          textures.roughness,
          textures.metallic,
          textures.ao,
          textures.height,
          textures.specular,
          textures.emission,
          lighting.skybox_image,
        ],
        outputs: [ray_hits],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.probe_total_ray_count / 128), 1, 1)
    );
  }
}

export class SurfaceCacheSHShadingStrategy extends GIShadingStrategy {
  constructor({ shader_setups = {} } = {}) {
    super({
      name: "surface-cache-sh-shading",
      representation: "surface-cache-sh",
      accepted_hit_representations: [GIHitRepresentation.SURFACE_PATCH_RAYS],
      radiance_representation: GIRadianceRepresentation.SURFACE_SH,
      shader_setups: {
        shade: shader("gi/surface_cache_trace_shade.wgsl"),
        shadow: shader("gi/surface_cache_trace_shadow.wgsl"),
        ...shader_setups,
      },
    });
  }

  setup(render_graph, context, _branch) {
    this.create_buffer(render_graph, "radiance_info", {
      name: "surface_cache_radiance_info",
      size: context.total_patches * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    this.material_buffers = register_material_buffers(render_graph);
    this.texture_pools = register_texture_pools(render_graph);
    this.scene_lighting_data = register_scene_lighting_data(render_graph);
  }

  record(render_graph, context, branch) {
    const trace = branch.trace_hit_cache;
    const params = trace.get_resource("params");
    const surface_cache = trace.get_resource("surface_cache");
    const active_indices = trace.get_resource("active_indices");
    const counters = trace.get_resource("counters");
    const hit_info = trace.get_resource("hit_info");
    const entity_index_lookup = trace.get_resource("entity_index_lookup");
    const radiance_info = this.get_resource("radiance_info");
    const sh = branch.accumulator.get_resource("surface_cache_sh");
    const ray_instance_transforms = trace.get_resource("ray_instance_transforms");
    const { inputs, total_patches } = context;
    const materials = this.material_buffers;
    const textures = this.texture_pools;
    const lighting = this.scene_lighting_data;

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
          active_indices,
          counters,
          hit_info,
          materials.params_gpu_buffer,
          materials.material_offsets_buffer,
          materials.material_palette_buffer,
          entity_index_lookup,
          inputs.dense_lights,
          textures.albedo,
          textures.normal,
          textures.roughness,
          textures.metallic,
          textures.ao,
          textures.height,
          textures.specular,
          textures.emission,
          lighting.skybox_image,
          radiance_info,
        ],
        outputs: [hit_info, radiance_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(total_patches / 128), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "shadow",
      "surface_cache_trace_shadows",
      {
        inputs: [
          params,
          counters,
          hit_info,
          inputs.tlas_bvh2_bounds,
          inputs.tlas_bvh_info,
          inputs.blas_bvh2_nodes,
          inputs.blas_directory,
          ray_instance_transforms,
          inputs.index_buffer,
          entity_index_lookup,
          radiance_info,
        ],
        outputs: [radiance_info],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(total_patches / 128), 1, 1)
    );
  }
}

export class PerPixelRGBShadingStrategy extends GIShadingStrategy {
  constructor(options = {}) {
    super({
      name: "per-pixel-weighted-rgb-shading",
      representation: "per-pixel-rgb",
      accepted_hit_representations: [GIHitRepresentation.PIXEL_PATHS],
      radiance_representation: GIRadianceRepresentation.PIXEL_RGB,
      shader_setups: {
        shade: shader("gi/pixel_trace_shade.wgsl"),
        ...options.shader_setups,
      },
    });
  }

  setup(render_graph) {
    this.material_buffers = register_material_buffers(render_graph);
    this.texture_pools = register_texture_pools(render_graph);
    this.scene_lighting_data = register_scene_lighting_data(render_graph);
    this.import_resource(
      "entity_index_lookup",
      render_graph.register_buffer(FragmentGpuBuffer.entity_index_map_buffer.buffer.config.name)
    );
  }

  record(render_graph, context, branch) {
    const surface_branch = branch.get_dependency("radiance_cache");
    const surface = surface_branch.trace_hit_cache;
    const trace = branch.trace_hit_cache;
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
          surface_branch.accumulator.get_resource("surface_cache_sh_filtered"),
        ],
        outputs: [path_state],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.rays_per_frame / 128), 1, 1)
    );
  }
}

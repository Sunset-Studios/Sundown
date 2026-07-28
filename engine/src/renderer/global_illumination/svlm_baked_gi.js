import { RenderPassFlags } from "../renderer_types.js";
import { SharedFrameInfoBuffer, SharedViewBuffer } from "../../core/shared_data.js";

const svlm_baked_resolve_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_baked_resolve.wgsl" },
  },
};

const diffuse_image_config = {
  name: "svlm_baked_indirect_diffuse",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

const black_image_config = {
  name: "svlm_baked_black",
  format: "rgba16float",
  width: 1,
  height: 1,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

/**
 * Runtime GI strategy for the GPU-resident SVLM bake artifact.
 *
 * Unlike DDGI/PTGI, this performs no tracing, temporal accumulation, or cache
 * maintenance. One full-resolution compute pass traverses the baked hierarchy,
 * interpolates eight packed SH probes, and writes diffuse irradiance.
 */
export class SVLMBakedGI {
  final_gi_texture_direct = null;
  final_gi_texture_indirect_diffuse = null;
  final_gi_texture_indirect_specular = null;
  debug_texture = null;

  constructor(svlm) {
    this.svlm = svlm;
  }

  add_passes(
    render_graph,
    width,
    height,
    depth_texture,
    _prev_depth_texture,
    gbuffer_normal,
    _gbuffer_normal_prev,
    _gbuffer_albedo,
    _gbuffer_smra,
    _gbuffer_motion_emissive,
    _tlas_bvh2_bounds,
    _tlas_bvh_info,
    _blas_bvh2_nodes,
    _blas_directory,
    _entity_transforms,
    _index_buffer,
    _dense_lights,
    draw_count,
    _hzb_texture,
    force_recreate = false
  ) {
    const view_index = SharedFrameInfoBuffer.get_view_index();
    if (view_index >= 0 && view_index < SharedViewBuffer.get_view_data_count()) {
      this.svlm?.update_tile_streaming(SharedViewBuffer.get_view_data(view_index).view_position);
    }

    const artifact = this.svlm?.get_bake_artifact();
    if (
      draw_count <= 0 ||
      !artifact?.usable ||
      !artifact.buffers.params ||
      !artifact.buffers.nodes ||
      !artifact.buffers.leaf_bricks ||
      !artifact.buffers.irradiance
    ) {
      this.reset();
      return;
    }

    diffuse_image_config.width = width;
    diffuse_image_config.height = height;
    diffuse_image_config.force = force_recreate;
    black_image_config.force = force_recreate;

    const diffuse_output = render_graph.create_image(diffuse_image_config);
    const black_output = render_graph.create_image(black_image_config);
    const params = render_graph.register_buffer(artifact.buffers.params.config.name);
    const nodes = render_graph.register_buffer(artifact.buffers.nodes.config.name);
    const leaves = render_graph.register_buffer(artifact.buffers.leaf_bricks.config.name);
    const irradiance = render_graph.register_buffer(artifact.buffers.irradiance.config.name);

    render_graph.add_pass(
      "svlm_baked_resolve",
      RenderPassFlags.Compute,
      {
        inputs: [
          depth_texture,
          gbuffer_normal,
          params,
          nodes,
          leaves,
          irradiance,
          diffuse_output,
          black_output,
        ],
        outputs: [diffuse_output, black_output],
        shader_setup: svlm_baked_resolve_shader_setup,
      },
      (graph, frame_data) => {
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    // Deferred lighting requires all three GI bindings. Direct lighting stays
    // in the deferred light loop and baked SVLM currently contains no glossy
    // lobe, so both unused slots share a single 1x1 zero texture.
    this.final_gi_texture_direct = black_output;
    this.final_gi_texture_indirect_diffuse = diffuse_output;
    this.final_gi_texture_indirect_specular = black_output;
  }

  add_debug_passes(
    render_graph,
    width,
    height,
    _gbuffer_normal,
    depth_texture,
    scene_color,
    _debug_view,
    force_recreate = false
  ) {
    this.svlm?.add_probe_debug_passes(
      render_graph,
      width,
      height,
      depth_texture,
      scene_color,
      force_recreate
    );
    this.debug_texture = this.svlm?.debug_texture ?? null;
    return this.debug_texture;
  }

  get_stats() {
    return this.svlm?.get_stats() ?? null;
  }

  reset() {
    this.final_gi_texture_direct = null;
    this.final_gi_texture_indirect_diffuse = null;
    this.final_gi_texture_indirect_specular = null;
    this.debug_texture = null;
  }
}

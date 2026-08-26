import { rgba16float_format } from "../../utility/config_permutations.js";
import { draw_quad } from "../draw_helpers.js";
import { RenderPassFlags } from "../renderer_types.js";

const lighting_pass_name = "lighting_pass";

const deferred_lighting_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "deferred_lighting.wgsl",
      defines: {
        GI_ENABLED: false,
        SHADOWS_ENABLED: false,
        AO_ENABLED: false,
      },
    },
    fragment: {
      path: "deferred_lighting.wgsl",
      defines: {
        GI_ENABLED: false,
        SHADOWS_ENABLED: false,
        AO_ENABLED: false,
      },
    },
  },
};

const post_lighting_image_config = {
  name: "post_lighting",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  force: false,
};

export class DeferredLightingPipeline {
  output_image_config = post_lighting_image_config;
  gi_lighting_enabled = false;
  previous_gi_lighting_enabled = null;

  add_pass(
    render_graph,
    {
      image_extent,
      current_buffered_frame,
      force_recreate = false,
      skybox_image,
      main_albedo_image,
      main_smra_image,
      main_normal_image,
      main_motion_emissive_image,
      main_depth_image,
      dense_lights,
      gi_enabled,
      gi_direct_texture,
      gi_diffuse_texture,
      gi_specular_texture,
      shadows_enabled,
      shadow_atlas,
      shadow_page_table,
      shadow_page_offset,
      shadow_settings,
      ao_enabled,
      ao_texture,
      bent_normal_texture,
    }
  ) {
    this._update_gi_lighting_state(
      render_graph,
      gi_enabled,
      gi_direct_texture,
      gi_diffuse_texture,
      gi_specular_texture
    );

    deferred_lighting_shader_setup.force_recreate = force_recreate;
    this._set_shader_features(this.gi_lighting_enabled, shadows_enabled, ao_enabled);

    const lighting_inputs = [
      skybox_image,
      main_albedo_image,
      main_smra_image,
      main_normal_image,
      main_motion_emissive_image,
      main_depth_image,
      dense_lights,
    ];

    if (this.gi_lighting_enabled) {
      lighting_inputs.push(gi_direct_texture, gi_diffuse_texture, gi_specular_texture);
    }
    if (shadows_enabled) {
      lighting_inputs.push(
        shadow_atlas,
        shadow_page_table,
        shadow_page_offset,
        shadow_settings
      );
    }
    if (ao_enabled) {
      lighting_inputs.push(ao_texture, bent_normal_texture);
    }

    this.output_image_config.width = image_extent.width;
    this.output_image_config.height = image_extent.height;
    this.output_image_config.force = force_recreate;
    const output_image = render_graph.create_image(this.output_image_config);

    render_graph.add_pass(
      lighting_pass_name,
      RenderPassFlags.Graphics,
      {
        inputs: lighting_inputs,
        outputs: [output_image],
        shader_setup: deferred_lighting_shader_setup,
        // Some implementations alternate history targets; buffer bindings while reusing one pipeline layout.
        bind_group_cache_key: `${lighting_pass_name}_${current_buffered_frame}`,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        draw_quad(pass);
      }
    );

    return output_image;
  }

  _update_gi_lighting_state(
    render_graph,
    gi_enabled,
    gi_direct_texture,
    gi_diffuse_texture,
    gi_specular_texture
  ) {
    this.gi_lighting_enabled =
      gi_enabled &&
      gi_direct_texture != null &&
      gi_diffuse_texture != null &&
      gi_specular_texture != null;

    if (
      this.previous_gi_lighting_enabled != null &&
      this.previous_gi_lighting_enabled !== this.gi_lighting_enabled
    ) {
      render_graph.recreate_pipeline_states();
    }
    this.previous_gi_lighting_enabled = this.gi_lighting_enabled;
  }

  _set_shader_features(gi_enabled, shadows_enabled, ao_enabled) {
    deferred_lighting_shader_setup.pipeline_shaders.vertex.defines.GI_ENABLED = gi_enabled;
    deferred_lighting_shader_setup.pipeline_shaders.vertex.defines.SHADOWS_ENABLED = shadows_enabled;
    deferred_lighting_shader_setup.pipeline_shaders.vertex.defines.AO_ENABLED = ao_enabled;
    deferred_lighting_shader_setup.pipeline_shaders.fragment.defines.GI_ENABLED = gi_enabled;
    deferred_lighting_shader_setup.pipeline_shaders.fragment.defines.SHADOWS_ENABLED = shadows_enabled;
    deferred_lighting_shader_setup.pipeline_shaders.fragment.defines.AO_ENABLED = ao_enabled;
  }
}

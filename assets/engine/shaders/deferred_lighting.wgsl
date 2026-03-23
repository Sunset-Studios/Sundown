#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_sampling.wgsl"

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var skybox_texture: texture_2d<f32>;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;
@group(1) @binding(2) var smra_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;
@group(1) @binding(4) var motion_emissive_texture: texture_2d<f32>;
@group(1) @binding(5) var depth_texture: texture_2d<f32>;
@group(1) @binding(6) var<storage, read> dense_lights_buffer: DenseLightsBuffer;

#if GI_ENABLED
  @group(1) @binding(7) var gi_direct_texture: texture_2d<f32>;
  @group(1) @binding(8) var gi_indirect_diffuse_texture: texture_2d<f32>;
  @group(1) @binding(9) var gi_indirect_specular_texture: texture_2d<f32>;
  #if SHADOWS_ENABLED
    @group(1) @binding(10) var<storage, read> shadow_atlas_depth: array<u32>;
    @group(1) @binding(11) var page_table: texture_storage_2d_array<r32uint, read>;
    @group(1) @binding(12) var page_offset: texture_storage_2d_array<rgba32float, read>;
    @group(1) @binding(13) var<uniform> vsm_settings: ASVSMSettings;
    #if AO_ENABLED
      @group(1) @binding(14) var ao_texture: texture_2d<f32>;
      @group(1) @binding(15) var bent_normal_texture: texture_2d<f32>;
    #endif
  #else
    #if AO_ENABLED
      @group(1) @binding(10) var ao_texture: texture_2d<f32>;
      @group(1) @binding(11) var bent_normal_texture: texture_2d<f32>;
    #endif
  #endif
#else
  #if SHADOWS_ENABLED
    @group(1) @binding(7) var<storage, read> shadow_atlas_depth: array<u32>;
    @group(1) @binding(8) var page_table: texture_storage_2d_array<r32uint, read>;
    @group(1) @binding(9) var page_offset: texture_storage_2d_array<rgba32float, read>;
    @group(1) @binding(10) var<uniform> vsm_settings: ASVSMSettings;

    #if AO_ENABLED
      @group(1) @binding(11) var ao_texture: texture_2d<f32>;
      @group(1) @binding(12) var bent_normal_texture: texture_2d<f32>;
    #endif
  #else
    #if AO_ENABLED
      @group(1) @binding(7) var ao_texture: texture_2d<f32>;
      @group(1) @binding(8) var bent_normal_texture: texture_2d<f32>;
    #endif
  #endif
#endif

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------
// Vertex Shader
// ------------------------------------------------------------------------------------ 

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var output : VertexOutput;

    output.position = vec4<f32>(vertex_buffer[vi].position);
    output.uv = vertex_buffer[vi].uv;

    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    let ambient = vec3<f32>(0.2, 0.2, 0.2);
    let uv = vec2<f32>(v_out.uv);

    var tex_sky = textureSample(skybox_texture, global_sampler, uv);

    var tex_albedo = textureSample(albedo_texture, global_sampler, uv);
    var albedo = tex_albedo.rgb;

    var tex_motion = textureSample(motion_emissive_texture, global_sampler, uv);
    var emissive = tex_motion.w;

	  var tex_normal = textureSample(normal_texture, global_sampler, uv);
    var normal = tex_normal.xyz;
	  var normal_length = length(normal);
	  var normalized_normal = normal / normal_length;
    var deferred_standard_lighting = tex_normal.w;

    var tex_smra = textureSample(smra_texture, global_sampler, uv);
    var reflectance = tex_smra.r;
    var roughness = tex_smra.g;
    var metallic = tex_smra.b;
    var ao = tex_smra.a;

#if AO_ENABLED
    ao = textureSample(ao_texture, non_filtering_sampler, uv).r;
    let bent_normal = textureSample(bent_normal_texture, non_filtering_sampler, uv).xyz;
    //normalized_normal = normalize(bent_normal);
#endif

    let view_index = u32(frame_info.view_index);
    let full_resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    let depth = textureLoad(depth_texture, uv_to_coord(uv, full_resolution), 0).r;
    var position = reconstruct_world_position(uv, depth, view_index);
    var position4 = vec4<f32>(position, 1.0);
    var view_dir = normalize(view_buffer[view_index].view_position.xyz - position);

    let unlit = min(1u, u32(normal_length <= 0.0) + u32(1.0 - deferred_standard_lighting));

    var color = f32(unlit) * tex_sky.rgb * mix(vec3<f32>(1.0), albedo, tex_albedo.a);

    var gi_direct = vec3<f32>(0.0);
    var gi_indirect_diffuse = vec3<f32>(0.0);
    var gi_indirect_specular = vec3<f32>(0.0);
#if GI_ENABLED
    gi_direct = textureSample(gi_direct_texture, global_sampler, uv).rgb;
    gi_indirect_diffuse = textureSample(gi_indirect_diffuse_texture, global_sampler, uv).rgb;
    gi_indirect_specular = textureSample(gi_indirect_specular_texture, global_sampler, uv).rgb;
#endif

    // Only use split GI as *indirect* when doing classic deferred direct lighting.
    let num_lights = dense_lights_buffer.header.light_count * (1u - unlit);
    for (var light_index = 0u; light_index < num_lights; light_index++) {
        var light = dense_lights_buffer.lights[light_index];
        let light_view_index = u32(light.view_index);
        let light_shadow_index = u32(light.shadow_index);
        let light_dir = get_light_dir(light, position);

#if SHADOWS_ENABLED
        let depth         = vsm_shadow_depth(
                                position4,
                                normalized_normal,
                                light_dir,
                                light_view_index,
                                light_shadow_index,
                                page_offset,
                                vsm_settings,
                            );
        let filter_res    = vsm_sample_shadow(
                              depth,
                              position4,
                              normalized_normal,
                              light_dir,
                              light_view_index,
                              light_shadow_index,
                              page_table,
                              vsm_settings);

        let shadow_factor = 1.0 - select(1.0, filter_res.depth, filter_res.valid);
#else
        let shadow_factor = 0.0;
#endif

        color += calculate_direct_brdf(
            light,
            normalized_normal,
            view_dir,
            light_dir,
            position,
            albedo,
            roughness,
            metallic,
            reflectance,
            0.0, // clear coat
            0.0, // clear coat roughness 
            shadow_factor,
        );
    }

    color += calculate_indirect_brdf(
        normalized_normal,
        view_dir,
        albedo,
        roughness,
        metallic,
        reflectance,
        ao,
        gi_indirect_diffuse,
        vec3<f32>(0.01, 0.01, 0.01),
        vec2<f32>(1.0, 1.0),
    );

    // Add indirect specular term after direct lighting evaluation.
    color += gi_indirect_specular;

    color += (emissive * albedo);

    return FragmentOutput(vec4<f32>(vec4<f32>(color, 1.0)));
}

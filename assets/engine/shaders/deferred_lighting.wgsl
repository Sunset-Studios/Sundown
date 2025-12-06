#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_sampling.wgsl"

#define USE_RADIANCE_CACHE_AS_DEFERRED_LIGHTING

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var skybox_texture: texture_2d<f32>;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;
@group(1) @binding(2) var smra_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;
@group(1) @binding(4) var position_texture: texture_2d<f32>;
@group(1) @binding(5) var motion_emissive_texture: texture_2d<f32>;
@group(1) @binding(6) var depth_texture: texture_depth_2d;
@group(1) @binding(7) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(8) var<storage, read> light_count_buffer: array<u32>;

#if GI_ENABLED
  @group(1) @binding(9) var gi_texture: texture_2d<f32>;
  #if SHADOWS_ENABLED
    @group(1) @binding(10) var<storage, read> shadow_atlas_depth: array<u32>;
    @group(1) @binding(11) var page_table: texture_storage_2d_array<r32uint, read>;
    @group(1) @binding(12) var page_offset: texture_storage_2d_array<rgba32float, read>;
    @group(1) @binding(13) var<uniform> vsm_settings: ASVSMSettings;
    #if GTAO_ENABLED
      @group(1) @binding(14) var ao_texture: texture_2d<f32>;
      @group(1) @binding(15) var bent_normal_texture: texture_2d<f32>;
    #endif
  #else
    #if GTAO_ENABLED
      @group(1) @binding(10) var ao_texture: texture_2d<f32>;
      @group(1) @binding(11) var bent_normal_texture: texture_2d<f32>;
    #endif
  #endif
#else
  #if SHADOWS_ENABLED
    @group(1) @binding(9) var<storage, read> shadow_atlas_depth: array<u32>;
    @group(1) @binding(10) var page_table: texture_storage_2d_array<r32uint, read>;
    @group(1) @binding(11) var page_offset: texture_storage_2d_array<rgba32float, read>;
    @group(1) @binding(12) var<uniform> vsm_settings: ASVSMSettings;

    #if GTAO_ENABLED
      @group(1) @binding(13) var ao_texture: texture_2d<f32>;
      @group(1) @binding(14) var bent_normal_texture: texture_2d<f32>;
    #endif
  #else
    #if GTAO_ENABLED
      @group(1) @binding(9) var ao_texture: texture_2d<f32>;
      @group(1) @binding(10) var bent_normal_texture: texture_2d<f32>;
    #endif
  #endif
#endif

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
};

struct FragmentOutput {
    @location(0) color: vec4<precision_float>,
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
    let ambient = vec3<precision_float>(0.2, 0.2, 0.2);
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

#if GTAO_ENABLED
    ao = textureSample(ao_texture, non_filtering_sampler, uv).r;
    let bent_normal = textureSample(bent_normal_texture, non_filtering_sampler, uv).xyz;
    //normalized_normal = normalize(bent_normal);
#endif

    var tex_position = textureSample(position_texture, non_filtering_sampler, uv);
    var position = tex_position.xyz;
    var position4 = vec4<f32>(position, 1.0);

    let view_index = u32(frame_info.view_index);
    var view_dir = normalize(view_buffer[view_index].view_position.xyz - position);

    let unlit = min(1u, u32(normal_length <= 0.0) + u32(1.0 - deferred_standard_lighting));

    var color = f32(unlit) * tex_sky.rgb * mix(vec3f(1.0), albedo, tex_albedo.a);

    var irradiance = vec3f(0.0);
#if GI_ENABLED
    irradiance = textureSample(gi_texture, global_sampler, uv).rgb;
#endif


#if USE_RADIANCE_CACHE_AS_DEFERRED_LIGHTING
    let gi_contribution = select(irradiance * ao, irradiance, ao <= 0.0);
    color += gi_contribution * albedo;
#else
    let num_lights = light_count_buffer[0] * (1u - unlit);
    for (var light_index = 0u; light_index < num_lights; light_index++) {
        var light = dense_lights_buffer[light_index];
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

        color += calculate_brdf(
            light_view_index,
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
            ao,
            irradiance, // irradiance
            vec3f(0.01, 0.01, 0.01), // prefilter color 
            vec2f(1.0, 1.0), // env brdf
            shadow_factor,
        );
    }
#endif

    color += (emissive * albedo);

    return FragmentOutput(vec4<precision_float>(vec4<f32>(color, 1.0)));
}
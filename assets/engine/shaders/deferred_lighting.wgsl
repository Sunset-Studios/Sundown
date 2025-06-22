#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var skybox_texture: texture_2d<f32>;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;
@group(1) @binding(2) var emissive_texture: texture_2d<f32>;
@group(1) @binding(3) var smra_texture: texture_2d<f32>;
@group(1) @binding(4) var normal_texture: texture_2d<f32>;
@group(1) @binding(5) var position_texture: texture_2d<f32>;
@group(1) @binding(6) var depth_texture: texture_depth_2d;
@group(1) @binding(7) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(8) var<storage, read> light_count_buffer: array<u32>;

#if GI_ENABLED
@group(1) @binding(9) var<uniform> gi_params: GIParams;
@group(1) @binding(10) var gi_irradiance: texture_3d<f32>;

#if SHADOWS_ENABLED
@group(1) @binding(11) var<storage, read> shadow_atlas_depth: array<u32>;
@group(1) @binding(12) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(13) var<uniform> vsm_settings: ASVSMSettings;
#endif

#else

#if SHADOWS_ENABLED
@group(1) @binding(9) var<storage, read> shadow_atlas_depth: array<u32>;
@group(1) @binding(10) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(11) var<uniform> vsm_settings: ASVSMSettings;
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

fn sample_probe_irradiance(world_pos: vec3<f32>) -> vec3<f32> {
#if GI_ENABLED
    let uvw = (world_pos - gi_params.origin) / gi_params.spacing;
    let tex = textureSampleLevel(gi_irradiance, global_sampler, uvw, 0.0);
    return tex.rgb;
#else
    return vec3<f32>(0.0);
#endif
}

#if SHADOWS_ENABLED
fn sample_shadow_vsm(
    world_pos: vec4<f32>,
    view_idx: u32,
    shadow_idx: u32,
) -> f32 {
  // Unpack useful constants
  let tile_size              = vsm_settings.tile_size;
  let phys_tiles_per_row     = u32(vsm_settings.physical_tiles_per_row);
  let one_over_atlas_size    = 1.0 / vec2<f32>(vsm_settings.physical_dim);
  let camera_vp              = view_buffer[frame_info.view_index].view_projection_matrix;
  let light_vp               = view_buffer[view_idx].view_projection_matrix;

  // --------------------------------------------------
  // Select clip-map level & compute virtual-texture coords
  // --------------------------------------------------
  let vtile_info = vsm_world_to_virtual_tile(world_pos, camera_vp, light_vp, vsm_settings);

  // Resolve PTE for this virtual tile (single-light slice assumption)
  let entry = textureLoad(page_table, vtile_info.tile_coords, vtile_info.clipmap_index + shadow_idx * u32(vsm_settings.max_lods)).r;
  if (!vsm_pte_is_valid(entry)) {
    return 1.0;
  }

  let clip_pos  = vsm_calculate_render_clip_value_from_world_pos(world_pos, vtile_info.clipmap_index, light_vp);
  // Convert NDC depth [-1,1] to [0,1] for comparison
  let depth_ndc = clip_pos.z;
  let depth_ref = depth_ndc * 0.5 + 0.5;

  // Decode physical tile & pool
  let physical_xy_offset = vsm_pte_get_phys_xy(entry);
  let memory_pool_index = vsm_pte_get_memory_pool_index(entry);

  // Build atlas UV
  let local_pixel_f = fract(vtile_info.tile_xy_f) * tile_size;
  let local_pixel = vec2<u32>(local_pixel_f);
  let physical_pixel = physical_xy_offset * u32(tile_size) + local_pixel;
  let base_pixel = vec2<i32>(physical_pixel);

  let phys_dim_u32 = u32(vsm_settings.physical_dim);

  // 3×3 PCF sampling via storage buffer
  // var sum_visible = 0.0;
  // for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
  //   for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
  //     let sx = clamp(base_pixel.x + ox, 0, i32(phys_dim_u32) - 1);
  //     let sy = clamp(base_pixel.y + oy, 0, i32(phys_dim_u32) - 1);

  //     let sample_index = memory_pool_index * phys_dim_u32 * phys_dim_u32 + 
  //                        u32(sy) * phys_dim_u32 + u32(sx);
  //     let depth_bits      = shadow_atlas_depth[sample_index];
  //     let depth_sample    = unpack_depth(depth_bits);

  //     sum_visible += select(0.0, 1.0, depth_ref < depth_sample);
  //   }
  // }

  let sample_index = memory_pool_index * phys_dim_u32 * phys_dim_u32 + 
                 u32(base_pixel.y) * phys_dim_u32 + u32(base_pixel.x);
  let depth_bits      = shadow_atlas_depth[sample_index];
  let depth_sample    = unpack_depth(depth_bits);

  let sum_visible = select(0.0, 1.0, depth_ref < depth_sample);


  return 1.0 - sum_visible / 9.0;
}
#endif

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

    var tex_emissive = textureSample(emissive_texture, global_sampler, uv);
    var emissive = tex_emissive.r;

	var tex_normal = textureSample(normal_texture, global_sampler, uv);
    var normal = tex_normal.xyz;
	var normal_length = length(normal);
	var normalized_normal = normal / normal_length;
    var deferred_standard_lighting = tex_normal.w;

    var tex_smra = textureSample(smra_texture, global_sampler, uv);
    var reflectance = tex_smra.r * 0.0009765625 /* 1.0f / 1024 */;
    var metallic = tex_smra.g;
    var roughness = tex_smra.b;
    var ao = tex_smra.a;

    var tex_position = textureSample(position_texture, global_sampler, uv);
    var position = tex_position.xyz;

    let view_index = frame_info.view_index;
    var view_dir = normalize(view_buffer[view_index].view_position.xyz - position);

    let unlit = min(1u, u32(normal_length <= 0.0) + u32(1.0 - deferred_standard_lighting));

    var color = f32(unlit) * tex_sky.rgb * mix(vec3f(1.0), albedo, tex_albedo.a);

    let irradiance = sample_probe_irradiance(position);

    let num_lights = light_count_buffer[0] * (1u - unlit);
    for (var light_index = 0u; light_index < num_lights; light_index++) {
        var light = dense_lights_buffer[light_index];
        let light_view_index = u32(light.view_index);

#if SHADOWS_ENABLED
        let shadow_factor = sample_shadow_vsm(vec4<f32>(position, 1.0), light_view_index, u32(light.shadow_index));
#else
        let shadow_factor = 0.0;
#endif

        color += calculate_brdf(
            light_view_index,
            light,
            normalized_normal,
            view_dir,
            position,
            albedo,
            roughness,
            metallic,
            reflectance,
            0.0, // clear coat
            1.0, // clear coat roughness 
            ao,
            irradiance,
            vec3f(1.0, 1.0, 1.0), // prefilter color 
            vec2f(1.0, 1.0), // env brdf
            shadow_factor,
        );
    }

    color += (emissive * albedo);

    return FragmentOutput(vec4<precision_float>(vec4<f32>(color, 1.0)));
}
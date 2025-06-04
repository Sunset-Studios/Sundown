#include "common.wgsl"
#include "lighting_common.wgsl"

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
@group(1) @binding(8) var<storage, read> dense_shadow_casting_lights_buffer: array<u32>;
@group(1) @binding(9) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(10) var<uniform> gi_params: GIParams;
@group(1) @binding(11) var gi_irradiance: texture_3d<f32>;
@group(1) @binding(12) var shadow_atlas: texture_depth_2d_array;
@group(1) @binding(13) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(14) var<storage, read> vsm_settings: ASVSMSettings;

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
    let uvw = (world_pos - gi_params.origin) / gi_params.spacing;
    let tex = textureSampleLevel(gi_irradiance, global_sampler, uvw, 0.0);
    return tex.rgb;
}

// PCF sample with page table lookup
fn sample_shadow(world_pos: vec3<f32>, view_idx: u32, light_idx: u32) -> f32 {
  // unpack AS-VSM settings
  let tile_size = vsm_settings.tile_size;
  let virtual_dim = vsm_settings.virtual_dim;
  let atlas_size = textureDimensions(shadow_atlas, 0).xy;
  
  let one_over_tile_size = 1.0 / f32(tile_size);
  let one_over_atlas_size = 1.0 / vec2<f32>(atlas_size);
  let phys_tiles_per_row = u32(f32(atlas_size.x) * one_over_tile_size);

  // project into light clip space
  let clip = view_buffer[view_idx].view_projection_matrix * vec4<f32>(world_pos, 1.0);
  let ndc = clip.xyz / clip.w;
  let depth_ref = ndc.z;

  // compute virtual UV
  let virtual_uv = (ndc.xy * 0.5 + vec2<f32>(0.5)) * virtual_dim;
  // virtual tile coords
  let tile_xy = virtual_uv * one_over_tile_size;

  // Map dense shadow casting light index to atlas array layer
  let layer = dense_shadow_casting_lights_buffer[light_idx];

  // fetch page table entry
  let entry = textureLoad(page_table, vec2<u32>(tile_xy), layer).x;
  if (entry == 0u) {
    return 1.0;
  }

  let phys_id = vsm_pte_get_physical_id(entry);
  // physical tile coords
  let phys_x = phys_id % phys_tiles_per_row;
  let phys_y = phys_id / phys_tiles_per_row;

  // base UV in atlas
  let base_uv = (vec2<f32>(f32(phys_x), f32(phys_y)) * tile_size) * one_over_atlas_size;
  // local UV within tile
  let local_uv = fract(tile_xy) * (tile_size * one_over_atlas_size);
  // final UV
  let final_uv = base_uv + local_uv;

  // PCF 3x3
  var sum: f32 = 0.0;
  for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
      let offset_uv = final_uv + one_over_atlas_size * vec2<f32>(f32(ox), f32(oy));
      sum += textureSampleCompareLevel(
          shadow_atlas,
          comparison_sampler,
          offset_uv,
          layer,
          depth_ref
      );
    }
  }
  return sum / 9.0;
}

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
    for (var i = 0u; i < num_lights; i++) {
        var light = dense_lights_buffer[i];
        let shadow_visible = sample_shadow(position, u32(light.view_index), i);
        let brdf = calculate_brdf(
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
            0, // shadow map index
        );
        color = color + brdf * shadow_visible;
    }

    color += (emissive * albedo);

    return FragmentOutput(vec4<precision_float>(vec4<f32>(color, 1.0)));
}
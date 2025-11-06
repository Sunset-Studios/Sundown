// =============================================================================
// World Cache Path Trace Output Pass
// - Blends accumulated path traced radiance with world cache data
// - Provides temporal stability by mixing cached indirect lighting
// - Outputs final composited result
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,      // 1=full res, 2=half res, 4=quarter res, etc.
    frame_phase: u32,     // cycles 0 to trace_rate-1
    indirect_boost: u32,  // Multiplier for indirect bounces
    padding: u32,
};

struct PathState {
    origin_tmin: vec4<f32>,                  // xyz = origin, w = tmin
    direction_tmax: vec4<f32>,                // xyz = direction, w = tmax
    normal_section_index: vec4<f32>,          // xyz = normal, w = section index
    state_u32: vec4<u32>, // x=bounce, y=alive(0/1), z=shadow_flag(0/1), w=tri_id
    hit_attr0: vec4<f32>, // xyz = world_tangent, w = uv.x
    hit_attr1: vec4<f32>, // xyz = world_bitangent, w = uv.y
    shadow_origin: vec4<f32>,      // xyz = origin, w = tmin
    shadow_direction: vec4<f32>,    // xyz = direction, w = tmax
    shadow_radiance: vec4<f32>,     // rgb = light contribution, a = needs_trace flag
};

struct PathShade {
    path_weight: vec4<f32>,                // xyz=throughput weight, w=source_pdf of current ray
    rng_sample_count_frame_stamp: vec4<f32>, // x=rng, y=sample count, z=frame stamp, w=padding
    throughput: vec4<f32>,                   // xyz=throughput, w=padding
    reservoir_radiance_m: vec4<f32>,       // xyz=BRDF estimate (importance hint), w=m (sample count)
    reservoir_direction_w: vec4<f32>,      // xyz=next bounce direction, w=final weight
};

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(2) var<storage, read> path_state: array<PathState>;
@group(1) @binding(3) var<storage, read> path_shade: array<PathShade>;
@group(1) @binding(4) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;

// Check if this pixel was traced in the current frame phase
// Following the article's approach: sparse pixels read from cache, traced pixels blend
fn is_pixel_traced_this_frame(coord: vec2<u32>, trace_rate: u32, frame_phase: u32) -> bool {
    if (trace_rate <= 1u) { return true; }
    
    // Checkerboard pattern based on frame phase
    let first_x = (frame_phase + trace_rate - (coord.y * 2u) % trace_rate) % trace_rate;
    return (coord.x >= first_x) && ((coord.x - first_x) % trace_rate == 0u);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    let pixel_coords = gid.xy;
    
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }

    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;
    let info = path_state[pixel_index];
    let shade = path_shade[pixel_index];

    // =============================================================================
    // Determine Pixel Trace Status
    // - Traced pixels: Have fresh path traced data this frame
    // - Untraced pixels: Need to rely more heavily on cache
    // =============================================================================
    let was_traced = is_pixel_traced_this_frame(pixel_coords, pt_params.trace_rate, pt_params.frame_phase);
    let sample_count = max(shade.rng_sample_count_frame_stamp.y, 1.0);
    let path_traced_radiance = safe_clamp_vec3(shade.throughput.xyz / sample_count);
    
    // =============================================================================
    // Query World Cache for Cached Indirect Lighting
    // Per the article: "terminate paths early into a data structure that
    // approximates the scene's lighting" for both performance and filtering
    // =============================================================================
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let camera_position = view.view_position.xyz;
    
    let cached_radiance = query_world_cache_cell(
        info.origin_tmin.xyz,
        info.normal_section_index.xyz,
        camera_position,
        u32(gi_params.world_cache_size),
        gi_params.world_cache_cell_size,
        u32(gi_params.world_cache_lod_count)
    );
    
    // =============================================================================
    // Adaptive Blending Based on Trace Status
    // - Untraced pixels: Heavily favor cache (acts as spatial filter)
    // - Traced pixels: Blend based on sample count (progressive refinement)
    // 
    // This implements the article's approach: "the noise is also greatly reduced
    // thanks to the filtering offered by the data structure"
    // =============================================================================
    let final_radiance = select(
        cached_radiance,
        path_traced_radiance,
        was_traced
    );
    
    // Write final composited result
    textureStore(output_tex, pixel_coords, vec4f(final_radiance, 1.0));
}


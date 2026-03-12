// =============================================================================
// Path Trace Output Pass (Simple Monte Carlo)
// =============================================================================
// Writes accumulated results and increments sample count.
// Uses simple averaging for progressive rendering with albedo demodulation:
//   1. accumulated_radiance stores demodulated radiance (without primary albedo)
//   2. output = (accumulated_radiance / sample_count) * primary_albedo
// This separation improves temporal stability and enables better denoising.
// =============================================================================

#include "common.wgsl"
#include "postprocess_common.wgsl"

// ─────────────────────────────────────────────────────────────────────────────
// Path Tracer Parameters
// ─────────────────────────────────────────────────────────────────────────────
struct PathTracerParams {
    max_bounces: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,
    frame_phase: u32,
    samples_per_pixel: u32,
    sample_index: u32,
    padding: u32,
};

// ─────────────────────────────────────────────────────────────────────────────
// Path State Structure
// ─────────────────────────────────────────────────────────────────────────────
struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
    path_weight: vec4<f32>,
    rng_sample_count: vec4<f32>,
    accumulated_radiance: vec4<f32>,
    primary_albedo: vec4<f32>,
};

// ─────────────────────────────────────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────────────────────────────────────
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var output_tex: texture_storage_2d<rgba16float, write>;

// =============================================================================
// Main Compute Shader
// =============================================================================
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    
    if (gid.x >= res.x || gid.y >= res.y) { return; }
    
    let pixel_index = gid.y * res.x + gid.x;
    var state = path_state[pixel_index];
    
    // ─────────────────────────────────────────────────────────────────────────
    // Determine if this pixel should be traced this frame (for trace_rate > 1)
    // ─────────────────────────────────────────────────────────────────────────
    let first_x_in_row = (pt_params.frame_phase + pt_params.trace_rate - (gid.y * 2u) % pt_params.trace_rate) % pt_params.trace_rate;
    let was_traced_this_frame = (pt_params.trace_rate <= 1u) || 
        ((gid.x >= first_x_in_row) && ((gid.x - first_x_in_row) % pt_params.trace_rate == 0u));

    if (was_traced_this_frame) {
        // ─────────────────────────────────────────────────────────────────────────
        // Increment sample count ONLY for traced pixels, by samples_per_pixel
        // ─────────────────────────────────────────────────────────────────────────
        path_state[pixel_index].rng_sample_count.y += f32(pt_params.samples_per_pixel);

        // ─────────────────────────────────────────────────────────────────────────
        // Compute progressive average and remodulate with primary albedo
        // ─────────────────────────────────────────────────────────────────────────
        // accumulated_radiance stores demodulated radiance (without primary hit albedo)
        // We multiply by primary_albedo here to get the final result
        let sample_count = max(path_state[pixel_index].rng_sample_count.y, 1.0);
        let demod_radiance = path_state[pixel_index].accumulated_radiance.xyz / sample_count;
        
        // Remodulate: multiply by primary albedo to restore correct color
        let primary_albedo = path_state[pixel_index].primary_albedo.xyz;
        let remodulated_radiance = demod_radiance * primary_albedo;
        let averaged_radiance = safe_clamp_vec3(remodulated_radiance);

        // Tonemap the output
        let exposure = 1.2;
        let tonemapped_color = reinhard_tonemapping(averaged_radiance, exposure);
        
        // ─────────────────────────────────────────────────────────────────────────
        // Write to output texture
        // ─────────────────────────────────────────────────────────────────────────
        textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(tonemapped_color, 1.0));
    }

}

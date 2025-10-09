// =============================================================================
// Path Tracer - Init Pass
// - Initializes per-pixel path state and resets per-pixel accumulation when needed
// - Generates primary rays from the active camera (no GBuffer dependency)
// =============================================================================
#include "common.wgsl"

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,      // 1=full res, 2=half res, 4=quarter res, etc.
    frame_phase: u32,     // cycles 0 to trace_rate-1
    ris_light_candidates: u32,    // Number of light candidates for RIS (M)
    ris_brdf_candidates: u32,     // Number of BRDF candidates for RIS (M)
    indirect_boost: u32,          // Multiplier for indirect bounces
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>, // x=bounce, y=alive(0/1), z=shadow_flag(0/1)
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,      // xyz = origin, w = tmin
    shadow_direction: vec4<f32>,    // xyz = direction, w = tmax
    shadow_radiance: vec4<f32>,     // rgb = light contribution, a = needs_trace flag
};

struct PathShade {
    path_weight: vec4<f32>,
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
    reservoir_light_index_m_w: vec4<f32>,  // x=light_idx, y=m, z=w, w=last_brdf_pdf
    reservoir_data: vec4<f32>,              // xyz=light_dir, w=attenuation
}

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read_write> path_shade: array<PathShade>;
@group(1) @binding(3) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_emissive: texture_2d<f32>;
@group(1) @binding(8) var output_tex: texture_storage_2d<rgba16float, write>;

// Helper function to compute the Nth pixel that matches the frame_phase pattern
// Optimized: ~3-5 iterations max, independent of resolution
fn compute_pixel_coords(linear_index: u32, res: vec2<u32>, trace_rate: u32, frame_phase: u32) -> vec2<u32> {
    if (trace_rate <= 1u) {
        return vec2<u32>(linear_index % res.x, linear_index / res.x);
    }
    
    // Estimate which row the pixel is in
    // Most rows have approx res.x / trace_rate pixels
    let avg_pixels_per_row = res.x / trace_rate;
    let estimated_row = linear_index / max(avg_pixels_per_row, 1u);
    
    // Search a small window around the estimate (max ~5 iterations)
    let search_start = select(0u, estimated_row - 1u, estimated_row >= 1u);
    let search_end = min(estimated_row + 4u, res.y);
    
    // Estimate cumulative pixels before search_start
    var cumulative_pixels = search_start * avg_pixels_per_row;
    
    for (var y = search_start; y < search_end; y = y + 1u) {
        let first_x = (frame_phase + trace_rate - (y * 2u) % trace_rate) % trace_rate;
        let pixels_in_row = (res.x + trace_rate - 1u - first_x) / trace_rate;
        
        if (linear_index < cumulative_pixels + pixels_in_row) {
            let offset_in_row = linear_index - cumulative_pixels;
            let x = first_x + offset_in_row * trace_rate;
            return vec2<u32>(x, y);
        }
        
        cumulative_pixels += pixels_in_row;
    }
    
    return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    
    // Compute actual pixel coordinates based on linear thread index and trace pattern
    let pixel_coords = compute_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    // Early exit if we're out of bounds
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }
    
    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    
    // Start a new path
    if (pt_params.use_gbuffer != 0u) {
        // G-buffer mode: Read from rasterized G-buffer instead of shooting primary rays
        let pixel_coord = vec2<i32>(i32(pixel_coords.x), i32(pixel_coords.y));
        
        let gbuffer_pos = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
        let gbuffer_norm_data = textureLoad(gbuffer_normal, pixel_coord, 0);
        let gbuffer_norm = gbuffer_norm_data.xyz;
        
        // Check if this pixel has valid geometry (normal length > 0 means geometry was rasterized)
        var normalized_normal = safe_normalize(gbuffer_norm);

        // Read pre-computed material properties from G-buffer
        let albedo_data = textureLoad(gbuffer_albedo, pixel_coord, 0);
        let smra_data = textureLoad(gbuffer_smra, pixel_coord, 0);
        let emissive_data = textureLoad(gbuffer_emissive, pixel_coord, 0);
        
        // Store view direction in direction_tmax (for BRDF evaluation)
        let view_dir = normalize(view.view_position.xyz - gbuffer_pos);
        // Ensure normal faces the camera's incoming view direction (similar to path_trace_hit.wgsl)
        // Ray direction is from camera towards surface (opposite of view_dir)
        let ray_dir = -view_dir;
        normalized_normal = select(normalized_normal, -normalized_normal, dot(normalized_normal, ray_dir) > 0.0);
        
        // Store G-buffer hit data in path state
        // Use origin_tmin to store world position with small normal offset
        path_state[pixel_index].origin_tmin = vec4f(gbuffer_pos + normalized_normal * 0.001, 0.0001);
        
        path_state[pixel_index].direction_tmax = vec4f(-view_dir, 0.0);
        
        // Store world-space normal
        path_state[pixel_index].normal_section_index = vec4f(normalized_normal, 0.0);
        
        // Pack material properties into hit_attr fields for shade pass to read
        // hit_attr0: rgb = albedo, w = roughness
        path_state[pixel_index].hit_attr0 = vec4f(albedo_data.rgb, smra_data.g);
        
        // hit_attr1: x = metallic, y = specular, z = emissive, w = ao
        let specular = smra_data.r * 0.0009765625; // 1.0f / 1024
        path_state[pixel_index].hit_attr1 = vec4f(smra_data.b, specular, emissive_data.r, smra_data.a);
        
        // Mark as having a valid G-buffer hit (state_u32.w = 0x0 for G-buffer mode)
        // bounce=0, alive=1, shadow_flag=0, tri_id=0x0 (special marker for G-buffer hit)
        path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0x0u);
    } else {
        // Traditional ray tracing mode: Generate primary rays from camera
        let dims = vec2<f32>(f32(res.x), f32(res.y));
        let pixel_center = vec2<f32>(f32(pixel_coords.x) + 0.5, f32(pixel_coords.y) + 0.5);
        let uv = pixel_center / dims;                // [0,1]
        let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); // [-1,1], y-up

        let tan_half_fov = tan(0.5 * view.fov);
        let sensor_x = ndc.x * view.aspect_ratio * tan_half_fov;
        let sensor_y = ndc.y * tan_half_fov;

        let forward = normalize(view.view_direction.xyz);
        let right = normalize(view.view_right.xyz);
        let up = normalize(cross(right, forward));
        var ray_dir = normalize(forward + right * sensor_x + up * sensor_y);
        let ray_origin = view.view_position.xyz;

        path_state[pixel_index].origin_tmin = vec4f(ray_origin + ray_dir * 0.001, 0.0001);
        path_state[pixel_index].direction_tmax = vec4f(ray_dir, 1e30);
        path_state[pixel_index].normal_section_index = vec4f(0.0, 0.0, 0.0, 0.0);
        path_state[pixel_index].hit_attr0 = vec4f(0.0, 0.0, 0.0, 0.0);
        path_state[pixel_index].hit_attr1 = vec4f(0.0, 0.0, 0.0, 0.0);
        path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu);
    }

    // Reset accumulation only when camera moves (view changed)
    // Only reset accumulation-related state once per frame when requested
    if (pt_params.reset_accum_flag != 0u) {
        let frame_id = u32(frame_info.frame_index);
        if (u32(path_shade[pixel_index].rng_sample_count_frame_stamp.z) != frame_id) {
            let rng_seed = hash(pixel_index ^ frame_id);
            path_shade[pixel_index].rng_sample_count_frame_stamp = vec4f(
                f32(rng_seed), 0.0, f32(frame_id), 0.0
            );
            path_shade[pixel_index].throughput = vec4f(0.0);
            path_shade[pixel_index].reservoir_light_index_m_w = vec4f(0.0);
            path_shade[pixel_index].reservoir_data = vec4f(0.0);
        }
    }

    path_state[pixel_index].shadow_origin = vec4f(0.0, 0.0, 0.0, 0.0);
    path_state[pixel_index].shadow_direction = vec4f(0.0, 0.0, 0.0, 0.0);
    path_state[pixel_index].shadow_radiance = vec4f(0.0, 0.0, 0.0, 0.0);
    path_shade[pixel_index].path_weight = vec4f(1.0, 1.0, 1.0, 0.0);
}



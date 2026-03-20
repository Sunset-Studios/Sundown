// =============================================================================
// Path Tracer - Init Pass (Simple Monte Carlo)
// =============================================================================
// Initializes per-pixel path state for unbiased Monte Carlo path tracing.
// Generates primary rays from the active camera or reads G-buffer data.
// Resets accumulation buffer when camera moves for progressive rendering.
// =============================================================================

#include "common.wgsl"

// ─────────────────────────────────────────────────────────────────────────────
// Path Tracer Parameters
// ─────────────────────────────────────────────────────────────────────────────
struct PathTracerParams {
    max_bounces: u32,          // Maximum number of light bounces
    reset_accum_flag: u32,     // 1 = camera moved, reset accumulation
    use_gbuffer: u32,          // 1 = hybrid mode (raster first hit)
    trace_rate: u32,           // 1=full res, 2=half, 4=quarter, etc.
    frame_phase: u32,          // Cycles 0 to trace_rate-1
    samples_per_pixel: u32,    // Number of samples per pixel per frame
    sample_index: u32,         // Current sample index (0 to samples_per_pixel-1)
    padding: u32,
};

// ─────────────────────────────────────────────────────────────────────────────
// Path State - Per-ray geometric and accumulation information
// ─────────────────────────────────────────────────────────────────────────────
struct PathState {
    origin_tmin: vec4<f32>,            // xyz = ray origin, w = t_min
    direction_tmax: vec4<f32>,         // xyz = ray direction, w = t_max or prim_store
    normal_section_index: vec4<f32>,   // xyz = surface normal, w = section index
    state_u32: vec4<u32>,              // x = bounce, y = alive, z = shadow_visible, w = tri_id
    hit_attr0: vec4<f32>,              // xyz = tangent (or albedo for gbuffer), w = uv.x (or roughness)
    hit_attr1: vec4<f32>,              // xyz = bitangent (or metallic,refl,emissive), w = uv.y
    shadow_origin: vec4<f32>,          // xyz = shadow ray origin, w = t_min
    shadow_direction: vec4<f32>,       // xyz = shadow ray direction, w = t_max
    shadow_radiance: vec4<f32>,        // rgb = potential light contribution, a = needs_trace
    path_weight: vec4<f32>,            // xyz = current path throughput, w = unused
    rng_sample_count: vec4<f32>,       // x = rng state, y = sample count, zw = unused
    accumulated_radiance: vec4<f32>,   // xyz = total accumulated radiance (demodulated), w = unused
    primary_albedo: vec4<f32>,         // xyz = primary hit albedo for demodulation, w = unused
};

// ─────────────────────────────────────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────────────────────────────────────
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var depth_texture: texture_2d<f32>;
@group(1) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_motion_emissive: texture_2d<f32>;
@group(1) @binding(7) var output_tex: texture_storage_2d<rgba16float, write>;

// =============================================================================
// Main Compute Shader
// =============================================================================
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    
    // Compute pixel coordinates based on trace pattern
    let pixel_coords = select(
        compute_phased_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase),
        vec2<u32>(gid.x % res.x, gid.x / res.x),
        pt_params.reset_accum_flag != 0u
    );

    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }
    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    
    // ─────────────────────────────────────────────────────────────────────────
    // Determine if this pixel should be traced this frame (for trace_rate > 1)
    // ─────────────────────────────────────────────────────────────────────────
    let first_x_in_row = (pt_params.frame_phase + pt_params.trace_rate - (pixel_coords.y * 2u) % pt_params.trace_rate) % pt_params.trace_rate;
    let should_trace_this_pixel = (pt_params.trace_rate <= 1u) || 
        ((pixel_coords.x >= first_x_in_row) && ((pixel_coords.x - first_x_in_row) % pt_params.trace_rate == 0u));
    
    // ─────────────────────────────────────────────────────────────────────────
    // Reset accumulation when camera moves (only on first sample)
    // ─────────────────────────────────────────────────────────────────────────
    if (pt_params.reset_accum_flag != 0u && pt_params.sample_index == 0u) {
        let frame_id = u32(frame_info.frame_index);
        let rng_seed = hash(pixel_index ^ frame_id);
        
        path_state[pixel_index].rng_sample_count = vec4<f32>(f32(rng_seed), 0.0, 0.0, 0.0);
        path_state[pixel_index].accumulated_radiance = vec4<f32>(0.0);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Initialize path for pixels being traced this frame
    // ─────────────────────────────────────────────────────────────────────────
    if (should_trace_this_pixel && pt_params.use_gbuffer != 0u) {
        // =====================================================================
        // G-Buffer Mode: Read first hit from rasterized G-buffer
        // =====================================================================
        let pixel_coord = vec2<i32>(i32(pixel_coords.x), i32(pixel_coords.y));
        let uv = coord_to_uv(pixel_coord, res);
        let depth = textureLoad(depth_texture, pixel_coord, 0).r;
        
        let gbuffer_pos = reconstruct_world_position(uv, depth, view_index);
        let gbuffer_norm_data = textureLoad(gbuffer_normal, pixel_coord, 0);
        let gbuffer_norm = gbuffer_norm_data.xyz;
        let gbuffer_norm_length = length(gbuffer_norm);
        
        if (gbuffer_norm_length > 0.0) {
            // Valid geometry hit - read material properties
            var normalized_normal = safe_normalize(gbuffer_norm);
            let albedo_data = textureLoad(gbuffer_albedo, pixel_coord, 0);
            let smra_data = textureLoad(gbuffer_smra, pixel_coord, 0);
            let emissive_data = textureLoad(gbuffer_motion_emissive, pixel_coord, 0).w;
            
            let ray_dir = normalize(gbuffer_pos - view.view_position.xyz);
            
            path_state[pixel_index].origin_tmin = vec4<f32>(gbuffer_pos + normalized_normal * 0.001, 0.0001);
            path_state[pixel_index].direction_tmax = vec4<f32>(ray_dir, 0.0);
            path_state[pixel_index].normal_section_index = vec4<f32>(normalized_normal, 0.0);
            path_state[pixel_index].hit_attr0 = vec4<f32>(albedo_data.rgb, smra_data.g); // albedo, roughness
            path_state[pixel_index].hit_attr1 = vec4<f32>(smra_data.b, smra_data.r, emissive_data, smra_data.a); // metallic, reflectance, emissive, ao
            path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0x0u); // bounce=0, alive=1, gbuffer marker
            path_state[pixel_index].primary_albedo = vec4<f32>(albedo_data.rgb, 1.0);
        } else {
            // No geometry - shoot ray to evaluate sky
            let dims = vec2<f32>(f32(res.x), f32(res.y));
            let pixel_center = vec2<f32>(f32(pixel_coords.x) + 0.5, f32(pixel_coords.y) + 0.5);
            let uv = pixel_center / dims;
            let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

            let tan_half_fov = tan(0.5 * view.fov);
            let sensor_x = ndc.x * view.aspect_ratio * tan_half_fov;
            let sensor_y = ndc.y * tan_half_fov;

            let forward = normalize(view.view_direction.xyz);
            let right = normalize(view.view_right.xyz);
            let up = normalize(cross(right, forward));
            var ray_dir = normalize(forward + right * sensor_x + up * sensor_y);
            let ray_origin = view.view_position.xyz;

            path_state[pixel_index].origin_tmin = vec4<f32>(ray_origin + ray_dir * 0.001, 0.0001);
            path_state[pixel_index].direction_tmax = vec4<f32>(ray_dir, 1e30);
            path_state[pixel_index].normal_section_index = vec4<f32>(0.0);
            path_state[pixel_index].hit_attr0 = vec4<f32>(0.0);
            path_state[pixel_index].hit_attr1 = vec4<f32>(0.0);
            path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu); // miss marker
            path_state[pixel_index].primary_albedo = vec4<f32>(1.0, 1.0, 1.0, 1.0);
        }
    } else if (should_trace_this_pixel) {
        // =====================================================================
        // Traditional Mode: Generate primary rays from camera
        // =====================================================================
        let dims = vec2<f32>(f32(res.x), f32(res.y));
        let pixel_center = vec2<f32>(f32(pixel_coords.x) + 0.5, f32(pixel_coords.y) + 0.5);
        let uv = pixel_center / dims;
        let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

        let tan_half_fov = tan(0.5 * view.fov);
        let sensor_x = ndc.x * view.aspect_ratio * tan_half_fov;
        let sensor_y = ndc.y * tan_half_fov;

        let forward = normalize(view.view_direction.xyz);
        let right = normalize(view.view_right.xyz);
        let up = normalize(cross(right, forward));
        var ray_dir = normalize(forward + right * sensor_x + up * sensor_y);
        let ray_origin = view.view_position.xyz;

        path_state[pixel_index].origin_tmin = vec4<f32>(ray_origin + ray_dir * 0.001, 0.0001);
        path_state[pixel_index].direction_tmax = vec4<f32>(ray_dir, 1e30);
        path_state[pixel_index].normal_section_index = vec4<f32>(0.0);
        path_state[pixel_index].hit_attr0 = vec4<f32>(0.0);
        path_state[pixel_index].hit_attr1 = vec4<f32>(0.0);
        path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu);
        path_state[pixel_index].primary_albedo = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Initialize fresh path state for this sample
    // ─────────────────────────────────────────────────────────────────────────
    if (should_trace_this_pixel || pt_params.reset_accum_flag != 0u) {
        path_state[pixel_index].shadow_origin = vec4<f32>(0.0);
        path_state[pixel_index].shadow_direction = vec4<f32>(0.0);
        path_state[pixel_index].shadow_radiance = vec4<f32>(0.0);
        path_state[pixel_index].path_weight = vec4<f32>(1.0, 1.0, 1.0, 0.0);
        
        // Advance RNG for this sample
        var rng = u32(path_state[pixel_index].rng_sample_count.x);
        if (rng == 0u) {
            let frame_id = u32(frame_info.frame_index);
            rng = hash(pixel_index ^ frame_id ^ pt_params.sample_index);
        } else {
            rng = random_seed(rng);
        }
        path_state[pixel_index].rng_sample_count.x = f32(rng);
    }
}

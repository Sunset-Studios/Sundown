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

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }
    
    // Temporal ray tracing: only trace a subset of pixels this frame
    // Use interleaved pattern that cycles over frames
    if (pt_params.trace_rate > 1u) {
        let pixel_pattern = (gid.x + gid.y * 2u) % pt_params.trace_rate;
        if (pixel_pattern != pt_params.frame_phase) { return; }
    }
    
    let pixel_index = gid.y * res.x + gid.x;

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    
    // Start a new path
    if (pt_params.use_gbuffer != 0u) {
        // G-buffer mode: Read from rasterized G-buffer instead of shooting primary rays
        let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
        
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
        
        path_state[pixel_index].direction_tmax = vec4f(view_dir, 0.0);
        
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
        let pixel_center = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
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
        }
    }

    path_state[pixel_index].shadow_origin = vec4f(0.0, 0.0, 0.0, 0.0);
    path_state[pixel_index].shadow_direction = vec4f(0.0, 0.0, 0.0, 0.0);
    path_state[pixel_index].shadow_radiance = vec4f(0.0, 0.0, 0.0, 0.0);
    path_shade[pixel_index].path_weight = vec4f(1.0, 1.0, 1.0, 0.0);
}



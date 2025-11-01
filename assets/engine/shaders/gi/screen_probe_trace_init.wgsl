// =============================================================================
// GI-1.0 Screen Probe Ray Tracing - Init Pass
// - Initializes rays for screen probes marked active this frame
// - Generates multiple rays per probe for variance reduction
// - Inactive probes (not updated this frame) get dead rays
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var<storage, read_write> probe_path_state: array<ProbePathState>;
@group(1) @binding(4) var<storage, read_write> probe_path_shade: array<ProbePathShade>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Use total_screen_probes (derived from grid dimensions)
    // Some probes may be inactive (not updated this frame), which is fine
    let probe_count = u32(gi_params.total_screen_probes);
    let rays_per_probe = u32(gi_params.screen_ray_count);
    let total_rays = probe_count * rays_per_probe;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    let probe_index = gid.x / rays_per_probe;
    let ray_index = gid.x % rays_per_probe;
    
    let probe = screen_probes[probe_index];
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    
    // Check if probe is active (being updated this frame)
    // Inactive probes get dead rays and won't be traced
    // Also check if probe was actually spawned (has valid data from spawn pass)
    if (probe.state.x == 0.0 || probe.state.w == 0.0) {
        // Mark all rays and shade data for this inactive probe as dead
        probe_path_state[gid.x].state_u32 = vec4<u32>(0u, 0u, 0u, 0xffffffffu);
        probe_path_state[gid.x].origin_tmin = vec4<f32>(0.0);
        probe_path_state[gid.x].direction_tmax = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }
    
    let position = probe.position_radius.xyz;
    let normal = probe.normal_frame.xyz;
    let frame_id = u32(gi_params.frame_index);
    
    // Initialize RNG for this probe ray
    var rng = u32(probe_path_state[gid.x].rng_sample_count_frame_stamp.x);
    if (rng == 0u) { rng = hash(gid.x ^ u32(gi_params.frame_index)); }
    else { rng = random_seed(rng); }
    
    // Generate random ray direction (cosine-weighted hemisphere)
    rng = random_seed(rng);
    let u1 = rand_float(rng);
    rng = random_seed(rng);
    let u2 = rand_float(rng);
    
    //let ray_dir = sample_cosine_hemisphere(u1, u2, normal);
    let ray_dir = normalize(position - view.view_position.xyz);
    let ray_origin = position + normal * 0.001; // Offset along normal to avoid self-intersection
    
    // Initialize path state
    probe_path_state[gid.x].origin_tmin = vec4<f32>(ray_origin, 0.0001);
    probe_path_state[gid.x].direction_tmax = vec4<f32>(ray_dir, 1e30);
    probe_path_state[gid.x].normal_section_index = vec4<f32>(normal, 0.0);
    probe_path_state[gid.x].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu);
    probe_path_state[gid.x].hit_attr0 = vec4<f32>(0.0);
    probe_path_state[gid.x].hit_attr1 = vec4<f32>(0.0);
    probe_path_state[gid.x].shadow_origin = vec4<f32>(0.0);
    probe_path_state[gid.x].shadow_direction = vec4<f32>(0.0);
    probe_path_state[gid.x].shadow_radiance = vec4<f32>(0.0);
    probe_path_state[gid.x].rng_sample_count_frame_stamp = vec4<f32>(f32(rng), 0.0, f32(frame_id), 0.0);
    probe_path_state[gid.x].path_weight = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    probe_path_state[gid.x].throughput = vec4<f32>(0.0);
    probe_path_shade[gid.x].reservoir_radiance_m = vec4<f32>(0.0);
    probe_path_shade[gid.x].reservoir_direction_w = vec4<f32>(0.0);
}


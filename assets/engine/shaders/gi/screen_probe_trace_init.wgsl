// =============================================================================
// GI-1.0 Screen Probe Ray Tracing - Init Pass
// - Initializes rays for screen probes marked active this frame
// - Generates multiple rays per probe for variance reduction
// - Inactive probes (not updated this frame) get dead rays
// =============================================================================
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "gi/gi_common.wgsl"

// Path state for screen probe rays
struct ProbePathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,        // x=bounce, y=alive, z=shadow_flag, w=tri_id
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
};

// Shading state with ReSTIR reservoir
struct ProbePathShade {
    path_weight: vec4<f32>,
    rng_sample_count_frame_stamp: vec4<f32>,
    throughput: vec4<f32>,
    reservoir_radiance_m: vec4<f32>,
    reservoir_direction_w: vec4<f32>,
};

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read> screen_probes: array<ScreenProbe>;
@group(1) @binding(3) var<storage, read_write> probe_path_state: array<ProbePathState>;
@group(1) @binding(4) var<storage, read_write> probe_path_shade: array<ProbePathShade>;

// Generate cosine-weighted hemisphere sample
fn sample_cosine_hemisphere(u1: f32, u2: f32, normal: vec3<f32>) -> vec3<f32> {
    let r = sqrt(u1);
    let theta = 2.0 * PI * u2;
    
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u1));
    
    // Build TBN frame
    // When normal is aligned with Y-axis, use X-axis as up; otherwise use Y-axis
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.y) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = normalize(cross(normal, tangent));
    
    return normalize(tangent * x + bitangent * y + normal * z);
}

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
    
    // Check if probe is active (being updated this frame)
    // Inactive probes get dead rays and won't be traced
    // Also check if probe was actually spawned (has valid data from spawn pass)
    if (probe.state.x == 0.0) {
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
    let ray_seed = hash(probe_index ^ (ray_index << 16u) ^ frame_id);
    var rng_state = ray_seed;
    
    // Generate random ray direction (cosine-weighted hemisphere)
    rng_state = random_seed(rng_state);
    let u1 = rand_float(rng_state);
    rng_state = random_seed(rng_state);
    let u2 = rand_float(rng_state);
    
    let ray_dir = sample_cosine_hemisphere(u1, u2, normal);
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
    
    // New spawn so reinitialize shading state from scratch
    //if (probe.state.w == 1.0) { 
        probe_path_shade[gid.x].rng_sample_count_frame_stamp = vec4<f32>(f32(rng_state), 0.0, f32(frame_id), 0.0);
        probe_path_shade[gid.x].path_weight = vec4<f32>(1.0, 1.0, 1.0, 1.0);
        probe_path_shade[gid.x].throughput = vec4<f32>(0.0);
        probe_path_shade[gid.x].reservoir_radiance_m = vec4<f32>(0.0);
        probe_path_shade[gid.x].reservoir_direction_w = vec4<f32>(0.0);
    //}
}


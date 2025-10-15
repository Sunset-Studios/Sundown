// =============================================================================
// GI-1.0 Screen Probe Ray Tracing - Init Pass
// - Initializes rays for each screen probe
// - Generates multiple rays per probe for variance reduction
// =============================================================================
#include "common.wgsl"

struct GIParams {
    screen_probe_spawn_rate: u32,
    screen_probe_size: u32,
    screen_ray_count: u32,
    world_cache_size: u32,
    max_screen_probes: u32,
    frame_index: u32,
    reset_caches: u32,
    indirect_boost: u32,
    upscale_x: u32,
    upscale_y: u32,
    cell_size_heuristic: u32,
    padding: u32,
};

struct ScreenProbe {
    position_radius: vec4<f32>,
    normal_frame: vec4<f32>,
    radiance_m: vec4<f32>,
    albedo_roughness: vec4<f32>,
    state: vec4<u32>,
};

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
@group(1) @binding(1) var<storage, read> screen_probes: array<ScreenProbe>;
@group(1) @binding(2) var<storage, read> screen_probe_counter: array<u32>;
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
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    
    return normalize(tangent * x + bitangent * y + normal * z);
}

@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = screen_probe_counter[0];
    let rays_per_probe = gi_params.screen_ray_count;
    let total_rays = probe_count * rays_per_probe;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    let probe_index = gid.x / rays_per_probe;
    let ray_index = gid.x % rays_per_probe;
    
    let probe = screen_probes[probe_index];
    
    // Check if probe is active
    if (probe.state.x == 0u) {
        probe_path_state[gid.x].state_u32 = vec4<u32>(0u, 0u, 0u, 0xffffffffu);
        return;
    }
    
    let position = probe.position_radius.xyz;
    let normal = probe.normal_frame.xyz;
    let frame_id = gi_params.frame_index;
    
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
    
    // Initialize shading state
    probe_path_shade[gid.x].path_weight = vec4<f32>(1.0, 1.0, 1.0, 1.0 / PI); // cosine-weighted PDF
    probe_path_shade[gid.x].rng_sample_count_frame_stamp = vec4<f32>(f32(rng_state), 0.0, f32(frame_id), 0.0);
    probe_path_shade[gid.x].throughput = vec4<f32>(0.0);
    probe_path_shade[gid.x].reservoir_radiance_m = vec4<f32>(0.0);
    probe_path_shade[gid.x].reservoir_direction_w = vec4<f32>(0.0);
}


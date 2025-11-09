// =============================================================================
// World Cache Path Trace Update Pass
// - Inserts radiance gathered from world-space path tracing into the shared
//   spatial hash cache so future frames can reuse the data.
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

struct PathTracerParams {
    max_bounces: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,      // 1=full res, 2=half res, 4=quarter res, etc.
    frame_phase: u32,     // cycles 0 to trace_rate-1
    indirect_boost: u32,          // Multiplier for indirect bounces
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

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    let pixel_coords = compute_phased_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }

    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;

    let sample_count = max(path_shade[pixel_index].rng_sample_count_frame_stamp.y, 1.0);
    let accumulated_avg = path_shade[pixel_index].throughput.xyz / sample_count;
    let radiance = safe_clamp_vec3(accumulated_avg);

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let camera_position = view.view_position.xyz;
}

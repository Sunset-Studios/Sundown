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
    max_spp: u32,
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal: vec4<f32>,
    throughput: vec4<f32>,
    state_u32: vec4<u32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    padding: vec4<f32>,
};

struct PixelHitInfo {
    rng: f32,
    sample_count: f32,
    prim_id: f32,
    frame_stamp: f32,
    accum_color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read_write> pixel_info: array<PixelHitInfo>;
@group(1) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }
    let pixel_index = gid.y * res.x + gid.x;

    // Build primary ray from camera for this pixel (always initialize rays)
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];

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
    path_state[pixel_index].normal = vec4f(0.0, 0.0, 0.0, 0.0);
    path_state[pixel_index].throughput = vec4f(1.0, 1.0, 1.0, 0.0);
    path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu);
    path_state[pixel_index].hit_attr0 = vec4f(0.0, 0.0, 0.0, 0.0);
    path_state[pixel_index].hit_attr1 = vec4f(0.0, 0.0, 0.0, 0.0);

    // Only reset accumulation-related state once per frame when requested
    if (pt_params.reset_accum_flag != 0u) {
        let frame_id = f32(u32(frame_info.frame_index));
        if (pixel_info[pixel_index].frame_stamp != frame_id) {
            let rng_seed = hash(pixel_index ^ u32(frame_info.frame_index));
            pixel_info[pixel_index].rng = f32(rng_seed);
            pixel_info[pixel_index].sample_count = 0.0;
            pixel_info[pixel_index].prim_id = -1.0;
            pixel_info[pixel_index].frame_stamp = frame_id; // stamp to avoid multiple resets in same frame
            pixel_info[pixel_index].accum_color = vec4f(0.0);
        }
    }
}



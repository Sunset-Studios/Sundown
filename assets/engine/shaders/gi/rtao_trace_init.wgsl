// =============================================================================
// RTAO RAY INITIALIZATION (SIMPLIFIED)
// =============================================================================
//
// Initializes rays for Ray Traced Ambient Occlusion only:
//   - One or more rays per pixel (screen_ray_count)
//   - Cosine-weighted hemisphere sampling for AO directions
//   - No NEE, no BRDF sampling, no ReSTIR, no light buffers
//   - Output: ray origin, direction, tmax and pixel mapping for hit/resolve
//
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> pixel_path_state: array<AOPixelPathState>;
@group(1) @binding(3) var<storage, read_write> ray_work_queue: array<u32>;
@group(1) @binding(4) var depth_texture: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var blue_noise: texture_2d_array<f32>;

// =============================================================================
// BLUE NOISE HELPERS
// =============================================================================

const BLUE_NOISE_LAYER_COUNT: u32 = 64u;

fn rtao_blue_noise_2d(gi_pixel_coord: vec2<u32>, frame_index: u32, dimension: u32) -> vec2<f32> {
    let dims = textureDimensions(blue_noise);
    let layer = (frame_index + dimension) % BLUE_NOISE_LAYER_COUNT;
    let offset = vec2<u32>((dimension * 17u) % dims.x, (dimension * 31u) % dims.y);
    let coord = vec2<i32>(
        i32((gi_pixel_coord.x + offset.x) % dims.x),
        i32((gi_pixel_coord.y + offset.y) % dims.y)
    );
    let texel = textureLoad(blue_noise, coord, i32(layer), 0);
    let scramble = f32(hash(dimension ^ (frame_index * 0x9E3779B9u))) * one_over_float_max;
    return vec2<f32>(fract(texel.r + scramble), fract(texel.g + scramble));
}

// =============================================================================
// MAIN COMPUTE
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = vec2<u32>(u32(gi_params.full_resolution_x), u32(gi_params.full_resolution_y));
    let gi_resolution = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    let frame_id = u32(gi_params.frame_index);
    let rays_per_pixel = u32(gi_params.screen_ray_count);
    let total_pixels = gi_resolution.x * gi_resolution.y;
    let total_rays = total_pixels * rays_per_pixel;

    if (gid.x >= total_rays) {
        return;
    }

    let ray_slot = gid.x;
    let pixel_index = ray_slot / rays_per_pixel;
    let gi_x = pixel_index % gi_resolution.x;
    let gi_y = pixel_index / gi_resolution.x;
    let gi_pixel_coord = vec2<u32>(gi_x, gi_y);

    let upscale_factor = u32(gi_params.upscale_factor);
    let full_pixel_coord = gi_pixel_to_full_res_pixel_coord(gi_pixel_coord, upscale_factor, full_resolution);

    // Sample G-buffer
    let normal_data = textureLoad(gbuffer_normal, full_pixel_coord, 0u);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);

    if (normal_length <= 0.0) {
        pixel_path_state[ray_slot].state_u32 = vec4<u32>(0u, 0u, 0u, 0xffffffffu);
        return;
    }

    let full_uv = coord_to_uv(vec2<i32>(full_pixel_coord), full_resolution);
    let depth = textureLoad(depth_texture, full_pixel_coord, 0u).r;
    let position = reconstruct_world_position(full_uv, depth, u32(frame_info.view_index));
    let uv = rtao_blue_noise_2d(gi_pixel_coord, frame_id, ray_slot);
    let ray_dir = sample_cosine_hemisphere(uv.x, uv.y, normal);

    // Ray for AO: origin offset along normal, max length from config
    let origin = position + normal * 0.001;
    let t_max = gi_params.max_ray_length;

    pixel_path_state[ray_slot].origin_tmin = vec4<f32>(origin, 0.0001);
    pixel_path_state[ray_slot].direction_tmax = vec4<f32>(ray_dir, t_max);
    pixel_path_state[ray_slot].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu);

    let queue_index = atomicAdd(&gi_counters.ray_queue_count, 1u);
    ray_work_queue[queue_index] = ray_slot;
}

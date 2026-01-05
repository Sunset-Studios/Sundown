#include "gi/gi_common.wgsl"
// =============================================================================
// Probe octahedral atlas layout
// - Each probe stores directional irradiance in an octahedral map (paper: 8x8)
// - Each probe tile is padded with a 1-texel duplicated gutter to avoid bilinear
//   filtering artifacts across atlas tile boundaries.
// - Each probe stores depth moments in an octahedral map (paper: 16x16)
// =============================================================================
const DDGI_PROBE_IRRADIANCE_RES = 8u;
const DDGI_PROBE_DEPTH_RES = 16u;
const DDGI_PROBE_ATLAS_GUTTER = 1u;
const PROBE_SAMPLE_CAP = 32.0;
const GOLDEN_RATIO_CONJUGATE = 0.6180339887498948;

struct DDGIParams {
    probe_counts: vec4<f32>,      // x=probe_count, y=rays_per_probe, z=probes_per_frame, w=probe_spacing
    probe_grid_dims: vec4<f32>,   // x=dim_x, y=dim_y, z=dim_z, w=probe_radius
    probe_grid_origin: vec4<f32>, // xyz = grid origin, w = unused
    probe_grid_log2: vec4<f32>,   // xyz = log2(dim_*), w = unused
    probe_grid_mask: vec4<f32>,   // xyz = (dim_*-1), w = unused
    probe_grid_snap_delta: vec4<f32>, // xyz = delta in probe cells, w = active (1/0)
};

struct DDGIProbeRayHit {
    hit_pos_t: vec4<f32>,         // xyz = world hit position, w = t (>=0) or -1 for miss
    ray_dir_prim: vec4<f32>,      // xyz = ray direction, w = prim_store as f32 (undefined if miss)
    world_n_section: vec4<f32>,   // xyz = world geometric normal, w = section_index as f32
    world_t_uvx: vec4<f32>,       // xyz = world tangent, w = uv.x
    world_b_uvy: vec4<f32>,       // xyz = world bitangent, w = uv.y
    state_u32: vec4<u32>,         // x = lobe_type (0 = diffuse, 1 = specular), y = alive, z = shadow_visible, w = tri_id
};

fn ddgi_probe_atlas_tile_size() -> u32 {
    return DDGI_PROBE_IRRADIANCE_RES + 2u * DDGI_PROBE_ATLAS_GUTTER;
}

fn ddgi_probe_depth_atlas_tile_size() -> u32 {
    return DDGI_PROBE_DEPTH_RES + 2u * DDGI_PROBE_ATLAS_GUTTER;
}

fn ddgi_probe_coord_from_index(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> vec3<u32> {
    let shift_x = u32((*ddgi_params).probe_grid_log2.x);
    let shift_y = u32((*ddgi_params).probe_grid_log2.y);

    let mask_x = u32((*ddgi_params).probe_grid_mask.x);
    let mask_y = u32((*ddgi_params).probe_grid_mask.y);
    let mask_z = u32((*ddgi_params).probe_grid_mask.z);

    let x = probe_index & mask_x;
    let y = (probe_index >> shift_x) & mask_y;
    let z = (probe_index >> (shift_x + shift_y)) & mask_z;

    return vec3<u32>(x, y, z);
}

fn ddgi_probe_index_from_coord(ddgi_params: ptr<uniform, DDGIParams>, coord: vec3<u32>) -> u32 {
    let shift_x = u32((*ddgi_params).probe_grid_log2.x);
    let shift_y = u32((*ddgi_params).probe_grid_log2.y);
    return coord.x | (coord.y << shift_x) | (coord.z << (shift_x + shift_y));
}

fn ddgi_probe_atlas_cell_from_coord(ddgi_params: ptr<uniform, DDGIParams>, coord: vec3<u32>) -> vec2<u32> {
    let dim_x = u32((*ddgi_params).probe_grid_dims.x);
    let cell_x = coord.x + coord.z * dim_x;
    let cell_y = coord.y;
    return vec2<u32>(cell_x, cell_y);
}

fn ddgi_probe_atlas_base_pixel(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> vec2<u32> {
    let coord = ddgi_probe_coord_from_index(ddgi_params, probe_index);
    let cell = ddgi_probe_atlas_cell_from_coord(ddgi_params, coord);
    let tile_size = ddgi_probe_atlas_tile_size();
    return cell * tile_size;
}

fn ddgi_probe_depth_atlas_base_pixel(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> vec2<u32> {
    let coord = ddgi_probe_coord_from_index(ddgi_params, probe_index);
    let cell = ddgi_probe_atlas_cell_from_coord(ddgi_params, coord);
    let tile_size = ddgi_probe_depth_atlas_tile_size();
    return cell * tile_size;
}

fn ddgi_probe_world_position_from_index(ddgi_params: ptr<uniform, DDGIParams>, probe_index: u32) -> vec3<f32> {
    let spacing = (*ddgi_params).probe_counts.w;
    let origin = (*ddgi_params).probe_grid_origin.xyz;
    let coord = ddgi_probe_coord_from_index(ddgi_params, probe_index);
    return origin + vec3<f32>(f32(coord.x), f32(coord.y), f32(coord.z)) * spacing;
}

fn ddgi_probe_world_position_from_coord(ddgi_params: ptr<uniform, DDGIParams>, coord: vec3<u32>) -> vec3<f32> {
    let spacing = ddgi_params.probe_counts.w;
    let origin = ddgi_params.probe_grid_origin.xyz;
    return origin + vec3<f32>(f32(coord.x), f32(coord.y), f32(coord.z)) * spacing;
}

fn ddgi_sample_probe_irradiance(
    ddgi_params: ptr<uniform, DDGIParams>,
    atlas: texture_2d<f32>,
    probe_index: u32,
    dir_ws: vec3<f32>
) -> vec3<f32> {
    let oct_uv = encode_octahedral(dir_ws);
    let res = f32(DDGI_PROBE_IRRADIANCE_RES);
    let gutter = f32(DDGI_PROBE_ATLAS_GUTTER);

    let base_px_u32 = ddgi_probe_atlas_base_pixel(ddgi_params, probe_index);
    let base_px = vec2<f32>(f32(base_px_u32.x), f32(base_px_u32.y));

    let atlas_dims_u32 = textureDimensions(atlas);
    let atlas_dims = vec2<f32>(f32(atlas_dims_u32.x), f32(atlas_dims_u32.y));

    // Map [0,1] oct UV into the interior [0,res-1] texel range, and sample with
    // bilinear filtering. The duplicated gutter prevents cross-tile bleeding.
    let interior_px = base_px + vec2<f32>(gutter, gutter) + oct_uv * (res - 1.0);
    let sample_uv = (interior_px + vec2<f32>(0.5)) / atlas_dims;

    return textureSampleLevel(atlas, global_sampler, sample_uv, 0.0).xyz;
}

// =============================================================================
// DDGI paper-inspired probe interpolation weights
// - Backface culling (soft)
// - Perceptual low-irradiance reduction (light leak robustness)
// - Chebyshev visibility from depth moments atlas (VSM-inspired)
// - Shading point bias for visibility query stability
// - Standard trilinear interpolation in probe grid space
// =============================================================================
fn ddgi_sample_probe_depth_moments(
    ddgi_params_ptr: ptr<uniform, DDGIParams>,
    atlas: texture_2d<f32>,
    probe_index: u32,
    dir_ws: vec3<f32>
) -> vec2<f32> {
    let oct_uv = encode_octahedral(dir_ws);
    let res = f32(DDGI_PROBE_DEPTH_RES);
    let gutter = f32(DDGI_PROBE_ATLAS_GUTTER);

    let base_px_u32 = ddgi_probe_depth_atlas_base_pixel(ddgi_params_ptr, probe_index);
    let base_px = vec2<f32>(f32(base_px_u32.x), f32(base_px_u32.y));

    let atlas_dims_u32 = textureDimensions(atlas);
    let atlas_dims = vec2<f32>(f32(atlas_dims_u32.x), f32(atlas_dims_u32.y));

    let interior_px = base_px + vec2<f32>(gutter, gutter) + oct_uv * (res - 1.0);
    let sample_uv = (interior_px + vec2<f32>(0.5)) / atlas_dims;

    return textureSampleLevel(atlas, global_sampler, sample_uv, 0.0).xy;
}

fn ddgi_visibility_chebyshev(moments: vec2<f32>, dist: f32, mean_bias: f32, variance_bias_sq: f32) -> f32 {
    // Depth moments:
    // - moments.x = E[d]
    // - moments.y = E[d^2]
    let mean_d = max(moments.x, 0.0);
    let mean_d2 = max(moments.y, 0.0);
    let variance = max(mean_d2 - mean_d * mean_d, 0.0);

    // Biasing (VSM-style) to reduce light leaks:
    // - Move the mean closer (more conservative occlusion)
    // - Add variance to widen the filter conservatively
    let mean_d_biased = max(mean_d - mean_bias, 0.0);
    let variance_biased = max(variance + variance_bias_sq, 1e-6);

    // If the point is closer than the (biased) mean hit distance, treat as visible.
    let delta = max(dist - mean_d_biased, 0.0);
    let p_max = variance_biased / (variance_biased + delta * delta);
    return clamp(p_max, 0.0, 1.0);
}

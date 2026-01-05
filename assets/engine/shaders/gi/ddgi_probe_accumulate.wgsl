// =============================================================================
// DDGI Probe Accumulation
// - Projects shaded rays into an 8x8 octahedral irradiance map per probe
// - Projects hit distances into a 16x16 octahedral depth moments map
// - Packs all probe maps into a single 2D atlas with duplicated gutter texels
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(2) var<storage, read> probe_ray_hits: array<DDGIProbeRayHit>;
@group(1) @binding(3) var<storage, read> probe_ray_radiance: array<vec4<f32>>;
@group(1) @binding(4) var probe_irradiance_prev: texture_2d<f32>;
@group(1) @binding(5) var probe_irradiance_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var probe_depth_prev: texture_2d<f32>;
@group(1) @binding(7) var probe_depth_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probes_per_frame = u32(ddgi_params.probe_counts.z);
    let rays_per_probe = u32(ddgi_params.probe_counts.y);

    if (gid.x >= probes_per_frame) {
        return;
    }

    let probe_index = probe_update_indices[gid.x];
    let ray_base = gid.x * rays_per_probe;

    // -------------------------------------------------------------------------
    // Atlas layout
    // -------------------------------------------------------------------------
    let gutter = i32(DDGI_PROBE_ATLAS_GUTTER);
    let res = i32(DDGI_PROBE_IRRADIANCE_RES);
    let depth_res = i32(DDGI_PROBE_DEPTH_RES);

    let dst_base_u32 = ddgi_probe_atlas_base_pixel(&ddgi_params, probe_index);
    let dst_base = vec2<i32>(i32(dst_base_u32.x), i32(dst_base_u32.y));

    let dst_base_depth_u32 = ddgi_probe_depth_atlas_base_pixel(&ddgi_params, probe_index);
    let dst_base_depth = vec2<i32>(i32(dst_base_depth_u32.x), i32(dst_base_depth_u32.y));

    // Reprojection for snapped probe grids: read from prev atlas at shifted coord
    let dims = vec3<i32>(
        i32(ddgi_params.probe_grid_dims.x),
        i32(ddgi_params.probe_grid_dims.y),
        i32(ddgi_params.probe_grid_dims.z)
    );
    let dst_coord = ddgi_probe_coord_from_index(&ddgi_params, probe_index);
    let dst_coord_i32 = vec3<i32>(dst_coord);
    let src_coord_i32 = dst_coord_i32 + vec3<i32>(ddgi_params.probe_grid_snap_delta.xyz);
    let in_bounds =
        src_coord_i32.x >= 0 && src_coord_i32.x < dims.x &&
        src_coord_i32.y >= 0 && src_coord_i32.y < dims.y &&
        src_coord_i32.z >= 0 && src_coord_i32.z < dims.z;
    let src_coord = vec3<u32>(src_coord_i32);
    let src_probe_index = ddgi_probe_index_from_coord(&ddgi_params, src_coord);
    let src_base_u32 = ddgi_probe_atlas_base_pixel(&ddgi_params, src_probe_index);
    let src_base = vec2<i32>(i32(src_base_u32.x), i32(src_base_u32.y));

    let src_base_depth_u32 = ddgi_probe_depth_atlas_base_pixel(&ddgi_params, src_probe_index);
    let src_base_depth = vec2<i32>(i32(src_base_depth_u32.x), i32(src_base_depth_u32.y));

    let atlas_dims_u32 = textureDimensions(probe_irradiance_prev);
    let atlas_dims = vec2<f32>(f32(atlas_dims_u32.x), f32(atlas_dims_u32.y));

    let depth_atlas_dims_u32 = textureDimensions(probe_depth_prev);
    let depth_atlas_dims = vec2<f32>(f32(depth_atlas_dims_u32.x), f32(depth_atlas_dims_u32.y));

    // -------------------------------------------------------------------------
    // Accumulate per-texel samples in local memory (no atomics)
    // -------------------------------------------------------------------------
    var accum_radiance: array<vec3<f32>, 64>;
    var accum_weight: array<f32, 64>;
    for (var t = 0u; t < 64u; t = t + 1u) {
        accum_radiance[t] = vec3<f32>(0.0);
        accum_weight[t] = 0.0;
    }

    // Depth moments (E[d], E[d^2]) stored in RG16F
    var accum_depth_sum: array<f32, 256>;
    var accum_depth_sq_sum: array<f32, 256>;
    var accum_depth_weight: array<f32, 256>;
    for (var t = 0u; t < 256u; t = t + 1u) {
        accum_depth_sum[t] = 0.0;
        accum_depth_sq_sum[t] = 0.0;
        accum_depth_weight[t] = 0.0;
    }

    let ddgi_probe_max_hit_distance = 65504.0;
    let ddgi_probe_max_hit_distance_sq = ddgi_probe_max_hit_distance * ddgi_probe_max_hit_distance;

    for (var i = 0u; i < rays_per_probe; i = i + 1u) {
        let ray_dir = probe_ray_hits[ray_base + i].ray_dir_prim.xyz;
        let hit_t_raw = probe_ray_hits[ray_base + i].hit_pos_t.w;
        let oct_uv = encode_octahedral(ray_dir);

        // --- Irradiance: octahedral 8x8 ---
        let tx = u32(clamp(round(oct_uv.x * f32(res - 1)), 0.0, f32(res - 1)));
        let ty = u32(clamp(round(oct_uv.y * f32(res - 1)), 0.0, f32(res - 1)));
        let texel_index = ty * u32(res) + tx;

        accum_radiance[texel_index] += probe_ray_radiance[ray_base + i].xyz;
        accum_weight[texel_index] += 1.0;

        // --- Depth: octahedral 16x16 ---
        let dtx = u32(clamp(round(oct_uv.x * f32(depth_res - 1)), 0.0, f32(depth_res - 1)));
        let dty = u32(clamp(round(oct_uv.y * f32(depth_res - 1)), 0.0, f32(depth_res - 1)));
        let depth_texel_index = dty * u32(depth_res) + dtx;

        let hit_t = clamp(hit_t_raw, 0.0, ddgi_probe_max_hit_distance);
        accum_depth_sum[depth_texel_index] += hit_t;
        accum_depth_sq_sum[depth_texel_index] += hit_t * hit_t;
        accum_depth_weight[depth_texel_index] += 1.0;
    }

    // -------------------------------------------------------------------------
    // Write interior texels (res x res) + duplicate gutters
    // -------------------------------------------------------------------------
    for (var y = 0; y < res; y = y + 1) {
        for (var x = 0; x < res; x = x + 1) {
            let texel_index = u32(y * res + x);
            let w_new = accum_weight[texel_index];
            let rgb_new = accum_radiance[texel_index] / max(w_new, 1.0);

            // Fetch previous history (if in bounds); otherwise treat as empty history.
            let src_px = src_base + vec2<i32>(gutter + x, gutter + y);
            let src_uv = (vec2<f32>(f32(src_px.x), f32(src_px.y)) + vec2<f32>(0.5)) / atlas_dims;
            let prev = select(
                vec4<f32>(0.0, 0.0, 0.0, 0.0),
                textureSampleLevel(probe_irradiance_prev, non_filtering_sampler, src_uv, 0.0),
                in_bounds
            );

            let combined_count = clamp(prev.w + w_new, 0.0, PROBE_SAMPLE_CAP);
            let alpha = w_new / max(prev.w + w_new, 1.0);
            var out_rgb = mix(prev.xyz, rgb_new, alpha);

            let dst_px = dst_base + vec2<i32>(gutter + x, gutter + y);
            textureStore(probe_irradiance_out, dst_px, vec4<f32>(out_rgb, combined_count));

            // Duplicate 1-texel gutter to allow bilinear filtering without
            // bleeding across atlas tiles.
            if (x == 0) {
                textureStore(probe_irradiance_out, dst_base + vec2<i32>(0, gutter + y), vec4<f32>(out_rgb, combined_count));
            }
            if (x == (res - 1)) {
                textureStore(probe_irradiance_out, dst_base + vec2<i32>(gutter + res, gutter + y), vec4<f32>(out_rgb, combined_count));
            }
            if (y == 0) {
                textureStore(probe_irradiance_out, dst_base + vec2<i32>(gutter + x, 0), vec4<f32>(out_rgb, combined_count));
            }
            if (y == (res - 1)) {
                textureStore(probe_irradiance_out, dst_base + vec2<i32>(gutter + x, gutter + res), vec4<f32>(out_rgb, combined_count));
            }
            if (x == 0 && y == 0) {
                textureStore(probe_irradiance_out, dst_base + vec2<i32>(0, 0), vec4<f32>(out_rgb, combined_count));
            }
            if (x == (res - 1) && y == 0) {
                textureStore(probe_irradiance_out, dst_base + vec2<i32>(gutter + res, 0), vec4<f32>(out_rgb, combined_count));
            }
            if (x == 0 && y == (res - 1)) {
                textureStore(probe_irradiance_out, dst_base + vec2<i32>(0, gutter + res), vec4<f32>(out_rgb, combined_count));
            }
            if (x == (res - 1) && y == (res - 1)) {
                textureStore(probe_irradiance_out, dst_base + vec2<i32>(gutter + res, gutter + res), vec4<f32>(out_rgb, combined_count));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Write depth moments (16x16) + duplicate gutters
    // - R: mean(hit_distance)
    // - G: mean(hit_distance^2)
    // -------------------------------------------------------------------------
    for (var y = 0; y < depth_res; y = y + 1) {
        for (var x = 0; x < depth_res; x = x + 1) {
            let texel_index = u32(y * depth_res + x);
            let w_new = accum_depth_weight[texel_index];
            let inv_w = 1.0 / max(w_new, 1.0);
            let moments_new = vec4<f32>(
                accum_depth_sum[texel_index] * inv_w,
                accum_depth_sq_sum[texel_index] * inv_w,
                0.0,
                w_new
            );

            // Reprojection for snapped probe grids:
            // If this texel received no rays this frame, keep history from the
            // reprojected previous atlas coordinate.
            let src_px = src_base_depth + vec2<i32>(gutter + x, gutter + y);
            let src_uv = (vec2<f32>(f32(src_px.x), f32(src_px.y)) + vec2<f32>(0.5)) / depth_atlas_dims;
            let moments_prev = select(
                vec4<f32>(ddgi_probe_max_hit_distance, ddgi_probe_max_hit_distance_sq, 0.0, 0.0),
                textureSampleLevel(probe_depth_prev, non_filtering_sampler, src_uv, 0.0),
                in_bounds
            );

            let combined_count = clamp(moments_prev.w + w_new, 0.0, PROBE_SAMPLE_CAP);
            let alpha = w_new / max(moments_prev.w + w_new, 1.0);
            var out_moments = mix(moments_prev.xy, moments_new.xy, alpha);

            let dst_px = dst_base_depth + vec2<i32>(gutter + x, gutter + y);
            textureStore(probe_depth_out, dst_px, vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));

            // Duplicate gutters (same pattern as irradiance atlas).
            if (x == 0) {
                textureStore(probe_depth_out, dst_base_depth + vec2<i32>(0, gutter + y), vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));
            }
            if (x == (depth_res - 1)) {
                textureStore(probe_depth_out, dst_base_depth + vec2<i32>(gutter + depth_res, gutter + y), vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));
            }
            if (y == 0) {
                textureStore(probe_depth_out, dst_base_depth + vec2<i32>(gutter + x, 0), vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));
            }
            if (y == (depth_res - 1)) {
                textureStore(probe_depth_out, dst_base_depth + vec2<i32>(gutter + x, gutter + depth_res), vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));
            }
            if (x == 0 && y == 0) {
                textureStore(probe_depth_out, dst_base_depth + vec2<i32>(0, 0), vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));
            }
            if (x == (depth_res - 1) && y == 0) {
                textureStore(probe_depth_out, dst_base_depth + vec2<i32>(gutter + depth_res, 0), vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));
            }
            if (x == 0 && y == (depth_res - 1)) {
                textureStore(probe_depth_out, dst_base_depth + vec2<i32>(0, gutter + depth_res), vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));
            }
            if (x == (depth_res - 1) && y == (depth_res - 1)) {
                textureStore(probe_depth_out, dst_base_depth + vec2<i32>(gutter + depth_res, gutter + depth_res), vec4<f32>(out_moments.x, out_moments.y, 0.0, combined_count));
            }
        }
    }
}

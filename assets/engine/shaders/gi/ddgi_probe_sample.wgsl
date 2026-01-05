// =============================================================================
// DDGI Probe Sampling
// - Samples probe radiance for diffuse GI
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(2) var probe_irradiance_atlas: texture_2d<f32>;
@group(1) @binding(3) var probe_depth_atlas: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var output_diffuse: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_diffuse);
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);

    if (length(normal_data.xyz) <= 0.001) {
        textureStore(output_diffuse, pixel_coord, vec4f(0.0));
        return;
    }

    let spacing = ddgi_params.probe_counts.w;
    let dims = vec3<u32>(
        u32(ddgi_params.probe_grid_dims.x),
        u32(ddgi_params.probe_grid_dims.y),
        u32(ddgi_params.probe_grid_dims.z)
    );
    let origin = ddgi_params.probe_grid_origin.xyz;

    let rel = (position - origin) / spacing;
    let base_f = floor(rel);
    let frac = rel - base_f;

    let max_base = vec3<f32>(
        f32(select(0u, dims.x - 2u, dims.x > 1u)),
        f32(select(0u, dims.y - 2u, dims.y > 1u)),
        f32(select(0u, dims.z - 2u, dims.z > 1u))
    );
    let base_clamped = clamp(base_f, vec3<f32>(0.0), max_base);
    let base = vec3<u32>(base_clamped);
    let frac_clamped = clamp(frac, vec3<f32>(0.0), vec3<f32>(1.0));

    let normal_ws = safe_normalize(normal_data.xyz);

    // Conservative epsilons / biases (paper-inspired; tune as needed):
    let probe_radius = ddgi_params.probe_grid_dims.w;
    let normal_bias = probe_radius * 0.20;
    let view_bias = probe_radius * 0.10;
    let mean_bias = probe_radius * 0.20;
    let variance_bias_sq = (probe_radius * 0.20) * (probe_radius * 0.20);

    // Perceptual weighting threshold: 5% of a "representable" luminance scale.
    // We use MAX_RADIANCE_LUMINANCE (GI common) as a practical HDR reference.
    let perceptual_threshold = max(0.05 * MAX_RADIANCE_LUMINANCE, 1e-4);

    var irradiance_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var z = 0u; z < 2u; z = z + 1u) {
        for (var y = 0u; y < 2u; y = y + 1u) {
            for (var x = 0u; x < 2u; x = x + 1u) {
                let coord = base + vec3<u32>(x, y, z);
                let idx = ddgi_probe_index_from_coord(&ddgi_params, coord);
                let tri_weight =
                    mix(1.0 - frac_clamped.x, frac_clamped.x, f32(x)) *
                    mix(1.0 - frac_clamped.y, frac_clamped.y, f32(y)) *
                    mix(1.0 - frac_clamped.z, frac_clamped.z, f32(z));

                let probe_pos = ddgi_probe_world_position_from_coord(&ddgi_params, coord);
                let dir_to_probe = safe_normalize(probe_pos - position);

                // --- Backface cull (soft): probes "below" the tangent plane contribute less ---
                let backface = clamp(dot(normal_ws, dir_to_probe), 0.0, 1.0);
                let backface_weight = backface * backface;

                // --- Bias the shading point for visibility query stability ---
                let biased_pos = position + normal_ws * normal_bias + dir_to_probe * view_bias;
                let dir_from_probe = safe_normalize(biased_pos - probe_pos);
                let dist = length(biased_pos - probe_pos);

                // --- Visibility (Chebyshev / VSM-inspired) from depth moments atlas ---
                let moments = ddgi_sample_probe_depth_moments(&ddgi_params, probe_depth_atlas, idx, dir_from_probe);
                let visibility_weight = ddgi_visibility_chebyshev(moments, dist, mean_bias, variance_bias_sq);

                // --- Irradiance sample + perceptual weighting (reduce very low irradiance) ---
                let probe_irradiance = ddgi_sample_probe_irradiance(&ddgi_params, probe_irradiance_atlas, idx, normal_ws);
                let probe_luma = dot(probe_irradiance, vec3<f32>(0.2126, 0.7152, 0.0722));
                let perceptual_linear = clamp(probe_luma / perceptual_threshold, 0.0, 1.0);
                let perceptual_weight = perceptual_linear * perceptual_linear;

                // --- Final normalized weight ---
                let weight = tri_weight * backface_weight * visibility_weight * perceptual_weight;

                irradiance_sum = irradiance_sum + probe_irradiance * weight;
                weight_sum = weight_sum + weight;
            }
        }
    }

    let inv_weight_sum = select(0.0, 1.0 / weight_sum, weight_sum > 1e-6);
    var irradiance = irradiance_sum * inv_weight_sum;
    irradiance = irradiance * gi_params.indirect_boost;
    textureStore(output_diffuse, pixel_coord, vec4f(irradiance, 1.0));
}

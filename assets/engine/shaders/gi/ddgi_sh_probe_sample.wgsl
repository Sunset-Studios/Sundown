// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               DDGI SPHERICAL HARMONICS PROBE SAMPLING                     ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Samples diffuse irradiance from SH-encoded probe grid.                   ║
// ║  Uses trilinear interpolation of SH coefficients for smooth results.      ║
// ║                                                                           ║
// ║  Advantages over octahedral sampling:                                     ║
// ║  • SH coefficients interpolate linearly (no octahedral edge artifacts)    ║
// ║  • More compact representation (12 floats vs 64 for 8x8 octahedral)       ║
// ║  • Natural low-frequency filtering (L1 captures dominant direction)       ║
// ║  • Efficient irradiance calculation via cosine lobe convolution           ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(2) var<storage, read> sh_probes: array<u32>;
@group(1) @binding(3) var probe_depth_atlas: texture_2d_array<f32>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var output_diffuse: texture_storage_2d<rgba16float, write>;

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ─────────────────────────────────────────────────────────────────────────
    // Bounds check
    // ─────────────────────────────────────────────────────────────────────────
    let res = textureDimensions(output_diffuse);
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Skip sky pixels (no geometry)
    // ─────────────────────────────────────────────────────────────────────────
    if (length(normal_data.xyz) <= 0.001) {
        textureStore(output_diffuse, pixel_coord, vec4f(0.0));
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Extract probe grid parameters
    // ─────────────────────────────────────────────────────────────────────────
    let spacing = ddgi_params.probe_counts.w;
    let dims = vec3<u32>(
        u32(ddgi_params.probe_grid_dims.x),
        u32(ddgi_params.probe_grid_dims.y),
        u32(ddgi_params.probe_grid_dims.z)
    );
    let origin = ddgi_params.probe_grid_origin.xyz;
    let probe_radius = ddgi_params.probe_grid_dims.w;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute probe grid coordinates and interpolation weights
    // ─────────────────────────────────────────────────────────────────────────
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
    
    // ─────────────────────────────────────────────────────────────────────────
    // Visibility/occlusion bias parameters
    // Based on DDGI paper recommendations
    // ─────────────────────────────────────────────────────────────────────────
    let normal_bias = probe_radius * 0.20;
    let view_bias = probe_radius * 0.10;
    let mean_bias = probe_radius * 0.20;
    let variance_bias_sq = (probe_radius * 0.20) * (probe_radius * 0.20);
    
    // Perceptual threshold for low-luminance filtering
    let perceptual_threshold = max(0.05 * MAX_RADIANCE_LUMINANCE, 1e-4);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Accumulate irradiance from 8 neighboring probes with weighting
    // Uses SH interpolation + visibility from depth atlas
    // ─────────────────────────────────────────────────────────────────────────
    var sh_sum = sh_l1_rgb_zero();
    var weight_sum = 0.0;
    
    for (var z = 0u; z < 2u; z = z + 1u) {
        for (var y = 0u; y < 2u; y = y + 1u) {
            for (var x = 0u; x < 2u; x = x + 1u) {
                let coord = base + vec3<u32>(x, y, z);
                let clamped_coord = clamp(coord, vec3<u32>(0u), dims - vec3<u32>(1u));
                let idx = ddgi_probe_index_from_coord(&ddgi_params, clamped_coord);
                
                // Trilinear interpolation weight
                let tri_weight =
                    select(1.0 - frac_clamped.x, frac_clamped.x, x == 1u) *
                    select(1.0 - frac_clamped.y, frac_clamped.y, y == 1u) *
                    select(1.0 - frac_clamped.z, frac_clamped.z, z == 1u);
                
                let probe_pos = ddgi_probe_world_position_from_coord(&ddgi_params, clamped_coord);
                let dir_to_probe = safe_normalize(probe_pos - position);
                
                // ─────────────────────────────────────────────────────────────
                // Backface culling: probes below tangent plane contribute less
                // ─────────────────────────────────────────────────────────────
                let backface = clamp(dot(normal_ws, dir_to_probe), 0.0, 1.0);
                let backface_weight = backface * backface;
                
                // ─────────────────────────────────────────────────────────────
                // Visibility query using depth moments (VSM-inspired)
                // ─────────────────────────────────────────────────────────────
                let biased_pos = position + normal_ws * normal_bias + dir_to_probe * view_bias;
                let dir_from_probe = safe_normalize(biased_pos - probe_pos);
                let dist = length(biased_pos - probe_pos);
                
                let moments = ddgi_sample_probe_depth_moments(&ddgi_params, probe_depth_atlas, idx, dir_from_probe);
                let visibility_weight = ddgi_visibility_chebyshev(moments, dist, mean_bias, variance_bias_sq);
                
                // ─────────────────────────────────────────────────────────────
                // Read SH probe and evaluate preview irradiance for perceptual weighting
                // ─────────────────────────────────────────────────────────────
                let probe_sh = ddgi_sh_probe_read(&sh_probes, idx);
                let preview_irradiance = ddgi_sh_evaluate_irradiance(probe_sh, normal_ws);
                let probe_luma = dot(preview_irradiance, vec3<f32>(0.2126, 0.7152, 0.0722));
                let perceptual_linear = clamp(probe_luma / perceptual_threshold, 0.0, 1.0);
                let perceptual_weight = perceptual_linear * perceptual_linear;
                
                // ─────────────────────────────────────────────────────────────
                // Combined weight for this probe
                // ─────────────────────────────────────────────────────────────
                let weight = tri_weight * backface_weight * visibility_weight * perceptual_weight;
                
                // Accumulate weighted SH coefficients
                sh_sum = sh_l1_rgb_add(sh_sum, sh_l1_rgb_multiply_scalar(probe_sh, weight));
                weight_sum = weight_sum + weight;
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Normalize accumulated SH and evaluate final irradiance
    // ─────────────────────────────────────────────────────────────────────────
    let inv_weight_sum = select(0.0, 1.0 / weight_sum, weight_sum > 1e-6);
    let sh_interpolated = sh_l1_rgb_multiply_scalar(sh_sum, inv_weight_sum);
    
    // Evaluate irradiance in the surface normal direction
    var irradiance = ddgi_sh_evaluate_irradiance(sh_interpolated, normal_ws);
    
    // Apply indirect boost
    irradiance = irradiance * gi_params.indirect_boost;
    
    // Ensure non-negative output
    irradiance = max(irradiance, vec3<f32>(0.0));
    
    textureStore(output_diffuse, pixel_coord, vec4f(irradiance, 1.0));
}


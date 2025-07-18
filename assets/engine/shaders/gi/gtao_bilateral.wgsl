// -----------------------------------------------------------------------------
// GTAO Bilateral Blur Compute Shader (full)
// -----------------------------------------------------------------------------
// Edge-aware denoiser for GTAO accessibility (AO) output.
// Preserves edges via depth & normal gating.
// -----------------------------------------------------------------------------

#include "common.wgsl"  // frame_info, view_buffer, non_filtering_sampler, etc.

// -----------------------------------------------------------------------------
// Uniforms
// -----------------------------------------------------------------------------
struct BilateralBlurSettings {
    radius_px:    f32,   // blur radius in pixels (half-kernel). e.g. 3 => 7x7 kernel.
    normal_power: f32,   // exponent for NdotN weight (higher => sharper edge stop). e.g. 32.
    sigma_depth:  f32,   // world-space depth sigma. controls falloff across depth breaks.
    sigma_spatial:f32,   // gaussian sigma in pixels for spatial weight.
    sigma_ao:     f32,   // range sigma in AO units (0..1). set <=0 to disable.
    _pad0:        f32,   // padding
};

// -----------------------------------------------------------------------------
// Bindings (match your engine however you like; these mirror GTAO-ish layout)
// -----------------------------------------------------------------------------
@group(1) @binding(0) var position_tex: texture_2d<f32>;                 // world pos G-buffer
@group(1) @binding(1) var normal_tex:   texture_2d<f32>;                 // world normal G-buffer
@group(1) @binding(2) var ao_src:       texture_2d<f32>;                 // GTAO AO (sample view)
@group(1) @binding(3) var ao_dst:       texture_storage_2d<r32float, write>; // blurred AO out
@group(1) @binding(4) var<uniform> blur_settings: BilateralBlurSettings;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
fn clamp_i32(v:i32, lo:i32, hi:i32) -> i32 {
    return max(lo, min(v, hi));
}

fn gauss_from_var(x:f32, sigma:f32) -> f32 {
    if (sigma <= 0.0) { return 1.0; }
    let s2 = sigma * sigma * 2.0; // 2*sigma^2 in denom
    return exp(- (x * x) / s2);
}

fn gauss_from_var2(x2:f32, sigma:f32) -> f32 {
    if (sigma <= 0.0) { return 1.0; }
    let s2 = sigma * sigma * 2.0;
    return exp(- x2 / s2);
}

// -----------------------------------------------------------------------------
// Main bilateral blur kernel
// -----------------------------------------------------------------------------
@compute @workgroup_size(8,8,1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ao_src);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let xy = gid.xy;
    let dims_f = vec2f(dims);
    let uv = (vec2f(xy) + 0.5) / dims_f;

    // camera
    let view = view_buffer[u32(frame_info.view_index)];

    // center sample
    let c_pos = textureLoad(position_tex, xy, 0).xyz;
    var c_nrm = textureSampleLevel(normal_tex, non_filtering_sampler, uv, 0.0).xyz;
    c_nrm = normalize(c_nrm);
    let c_ao = textureLoad(ao_src, xy, 0).x; // accessibility 0..1

    // view-space z for depth gating
    let c_vs = (view.view_matrix * vec4f(c_pos, 1.0)).z;

    let r = i32(max(0, min(u32(blur_settings.radius_px), 32))); // clamp sanity
    let npow = max(blur_settings.normal_power, 0.0);

    var w_sum = 0.0;
    var ao_sum = 0.0;

    // include center pixel explicitly so we never divide by 0 and preserve energy
    var w_center = 1.0;
    w_sum += w_center;
    ao_sum += w_center * c_ao;

    for (var dy:i32 = -r; dy <= r; dy = dy + 1) {
        for (var dx:i32 = -r; dx <= r; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            let sx = clamp_i32(i32(xy.x) + dx, 0, i32(dims.x) - 1);
            let sy = clamp_i32(i32(xy.y) + dy, 0, i32(dims.y) - 1);
            let sxy = vec2u(u32(sx), u32(sy));
            let suv = (vec2f(sxy) + 0.5) / dims_f;

            let s_pos = textureLoad(position_tex, sxy, 0).xyz;
            var s_nrm = textureSampleLevel(normal_tex, non_filtering_sampler, suv, 0.0).xyz;
            s_nrm = normalize(s_nrm);
            let s_ao = textureLoad(ao_src, sxy, 0).x;
            let s_vs = (view.view_matrix * vec4f(s_pos, 1.0)).z;

            // spatial weight (pixel distance)
            let d2 = f32(dx*dx + dy*dy);
            let w_spatial = gauss_from_var2(d2, blur_settings.sigma_spatial);

            // depth weight (view-space z difference)
            let dz = abs(s_vs - c_vs);
            let w_depth = gauss_from_var(dz, blur_settings.sigma_depth);

            // normal weight
            let nd = max(dot(c_nrm, s_nrm), 0.0);
            let w_normal = select(pow(nd, npow), 1.0, npow > 0.0);

            // AO range weight (optional)
            let da = abs(s_ao - c_ao);
            let w_ao = gauss_from_var(da, blur_settings.sigma_ao);

            let w = w_spatial * w_depth * w_normal * w_ao;
            ao_sum += w * s_ao;
            w_sum += w;
        }
    }

    var out_ao = c_ao;
    if (w_sum > 0.0) {
        out_ao = ao_sum / w_sum;
    }
    out_ao = clamp(out_ao, 0.0, 1.0);

    textureStore(ao_dst, vec2i(xy), vec4f(out_ao, out_ao, out_ao, 1.0));
}

// -----------------------------------------------------------------------------
// Suggested defaults when populating BilateralBlurSettings from host:
//   radius_px    = 3 (=> 7x7 kernel)
//   normal_power = 32
//   sigma_depth  = settings.radius * 0.25  (tie to GTAO radius)
//   sigma_spatial= f32(radius_px) * 0.5
//   sigma_ao     = 0.25
// Larger radii: run multiple passes ping-pong.
// -----------------------------------------------------------------------------

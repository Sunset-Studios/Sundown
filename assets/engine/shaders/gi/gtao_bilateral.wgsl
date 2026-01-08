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
const max_halo = 8;
const load_extent = 8 + 2 * max_halo;

var<workgroup> shared_ao: array<f32, load_extent * load_extent>;
var<workgroup> shared_vs: array<f32, load_extent * load_extent>;
var<workgroup> shared_nrm: array<vec4<f32>, load_extent * load_extent>;

@compute @workgroup_size(8,8,1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
    let dims = textureDimensions(ao_src);
    let dims_f = vec2f(dims);

    let xy = gid.xy;

    // blur_settings.radius_px is the half-kernel radius in pixels.
    let r = i32(max(0, min(u32(blur_settings.radius_px + 0.5), u32(max_halo))));
    let this_load_w = 8 + 2 * r;

    let tile_left = i32(wgid.x * 8u);
    let tile_top = i32(wgid.y * 8u);

    let load_left = tile_left - r;
    let load_top = tile_top - r;

    let total_load_pixels = this_load_w * this_load_w;
    let loads_per_thread = total_load_pixels / 64;
    let remainder_loads = total_load_pixels % 64;

    let thread_idx = i32(lid.y * 8u + lid.x);

    for (var k = 0; k < loads_per_thread; k = k + 1) {
        let flat_idx = thread_idx * loads_per_thread + k;
        if (flat_idx >= total_load_pixels) { continue; }
        let lx = flat_idx % this_load_w;
        let ly = flat_idx / this_load_w;
        let sx = load_left + lx;
        let sy = load_top + ly;
        let clamped_sx = clamp(sx, 0, i32(dims.x) - 1);
        let clamped_sy = clamp(sy, 0, i32(dims.y) - 1);
        let sxy = vec2u(u32(clamped_sx), u32(clamped_sy));
        let suv = (vec2f(sxy) + 0.5) / dims_f;

        let s_pos = textureLoad(position_tex, sxy, 0).xyz;
        let s_nrm = textureLoad(normal_tex, sxy, 0).xyz;
        let s_ao = textureLoad(ao_src, sxy, 0).x;

        let view = view_buffer[u32(frame_info.view_index)];
        let s_vs = (view.view_matrix * vec4f(s_pos, 1.0)).z;

        let shared_idx = ly * load_extent + lx;
        shared_ao[shared_idx] = s_ao;
        shared_vs[shared_idx] = s_vs;
        shared_nrm[shared_idx] = vec4f(normalize(s_nrm), 0.0);
    }

    if (thread_idx < remainder_loads) {
        let flat_idx = 64 * loads_per_thread + thread_idx;
        if (flat_idx < total_load_pixels) {
            let lx = flat_idx % this_load_w;
            let ly = flat_idx / this_load_w;
            let sx = load_left + lx;
            let sy = load_top + ly;
            let clamped_sx = clamp(sx, 0, i32(dims.x) - 1);
            let clamped_sy = clamp(sy, 0, i32(dims.y) - 1);
            let sxy = vec2u(u32(clamped_sx), u32(clamped_sy));
            let suv = (vec2f(sxy) + 0.5) / dims_f;

            let s_pos = textureLoad(position_tex, sxy, 0).xyz;
            let s_nrm = textureLoad(normal_tex, sxy, 0).xyz;
            let s_ao = textureLoad(ao_src, sxy, 0).x;

            let view = view_buffer[u32(frame_info.view_index)];
            let s_vs = (view.view_matrix * vec4f(s_pos, 1.0)).z;

            let shared_idx = ly * load_extent + lx;
            shared_ao[shared_idx] = s_ao;
            shared_vs[shared_idx] = s_vs;
            shared_nrm[shared_idx] = vec4f(normalize(s_nrm), 0.0);
        }
    }

    workgroupBarrier();

    let local_x = i32(lid.x);
    let local_y = i32(lid.y);

    let center_shared_x = r + local_x;
    let center_shared_y = r + local_y;
    let center_idx = center_shared_y * load_extent + center_shared_x;

    let c_ao = shared_ao[center_idx];
    let c_vs = shared_vs[center_idx];
    let c_nrm = shared_nrm[center_idx].xyz;

    var w_sum = 0.0;
    var ao_sum = 0.0;

    var w_center = 1.0;
    w_sum += w_center;
    ao_sum += w_center * c_ao;

    for (var dy:i32 = -r; dy <= r; dy = dy + 1) {
        for (var dx:i32 = -r; dx <= r; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }

            let s_shared_x = center_shared_x + dx;
            let s_shared_y = center_shared_y + dy;
            let s_idx = s_shared_y * load_extent + s_shared_x;

            let s_ao = shared_ao[s_idx];
            let s_vs = shared_vs[s_idx];
            let s_nrm = shared_nrm[s_idx].xyz;

            let dz = abs(s_vs - c_vs);
            if (dz > blur_settings.sigma_depth * 3.0) { continue; } // Bilateral rejection

            let d2 = f32(dx*dx + dy*dy);
            let w_spatial = gauss_from_var2(d2, blur_settings.sigma_spatial);

            let w_depth = gauss_from_var(dz, blur_settings.sigma_depth);

            let da = abs(s_ao - c_ao);
            let w_ao = gauss_from_var(da, blur_settings.sigma_ao);

            let ndot = max(dot(c_nrm, s_nrm), 0.0);
            let w_nrm = pow(ndot, max(blur_settings.normal_power, 0.0));

            let w = w_spatial * w_depth * w_ao * w_nrm;
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

// -----------------------------------------------------------------------------
// GTAO Compute Shader (Slice/Horizon Version)
// -----------------------------------------------------------------------------
// This implementation follows the high-level algorithm described in the GTAO
// report you provided: multi-slice horizon search in the normal's tangent plane,
// cosine-weighted integration, and bent-normal estimation.
//
// Conventions:
//   * All calculations done in WORLD space using position_tex + normal_tex.
//   * Camera transforms accessed via view_buffer[frame_info.view_index].
//   * settings.radius is a WORLD-space distance.
//   * Output AO is "accessibility": 1 = fully open (no occlusion), 0 = fully blocked.
//   * Bent normal is written in WORLD space to match world-space lighting inputs.
//
// Performance notes:
//   * This version does direct world projection per tap (clear & correct, not fastest).
//   * You can later add a depth pyramid + screen-space stepping if needed.
// -----------------------------------------------------------------------------

#include "common.wgsl"   // must define: frame_info, view_buffer, non_filtering_sampler, etc.

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------
struct GTAOSettings {
    radius: f32,        // world units
    bias: f32,          // small depth slack (world units) when testing horizons
    sample_count: f32,  // total tap budget (distributed across slices)
    _pad: f32
};

// -----------------------------------------------------------------------------
// Bindings
// -----------------------------------------------------------------------------
@group(1) @binding(0) var position_tex: texture_2d<f32>;                  // world position G-buffer
@group(1) @binding(1) var normal_tex:   texture_2d<f32>;                  // world normal G-buffer
@group(1) @binding(2) var ao_texture:   texture_storage_2d<r32float, write>;
@group(1) @binding(3) var bent_output:  texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var<uniform> settings: GTAOSettings;

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------
const PI:        f32 = 3.14159265358979323846;
const HALF_PI:   f32 = 1.57079632679489661923;
const TWO_PI:    f32 = 6.28318530717958647692;
const NUM_SLICES: u32 = 3u;      // 3 evenly spaced directions (XeGTAO-style)
const MIN_STEPS_PER_SLICE: u32 = 2u;

// -----------------------------------------------------------------------------
// Utility hashes
// -----------------------------------------------------------------------------
fn hash_u32(x: u32) -> u32 {
    var v = x * 1664525u + 1013904223u;
    v ^= v >> 16u;
    v *= 2246822519u;
    v ^= v >> 13u;
    v *= 3266489917u;
    v ^= v >> 16u;
    return v;
}

fn hash2(ixy: vec2<u32>, seed: u32) -> vec2<f32> {
    let h0 = hash_u32(ixy.x ^ (seed * 374761393u) ^ (ixy.y * 668265263u));
    let h1 = hash_u32(ixy.y ^ (seed * 2246822519u) ^ (ixy.x * 3266489917u));
    return vec2f(
        f32(h0 & 0x00ffffffu) * (1.0 / 16777215.0),
        f32(h1 & 0x00ffffffu) * (1.0 / 16777215.0)
    );
}

// Random rotation angle in [0, 2π) per pixel
fn random_angle(ixy: vec2<u32>, frame_idx: u32) -> f32 {
    return hash2(ixy, frame_idx).x * TWO_PI;
}

// -----------------------------------------------------------------------------
// Robust tangent basis from world-space normal
// -----------------------------------------------------------------------------
fn make_tbn(n_in: vec3<f32>) -> mat3x3<f32> {
    let n = normalize(n_in);
    let use_alt = abs(n.y) > 0.99;
    let up = select(world_up, world_right, use_alt);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n); // columns: T,B,N
}

// -----------------------------------------------------------------------------
// Fast acos approximation used by XeGTAO. Good enough for our purposes.
// -----------------------------------------------------------------------------
fn fast_acos(x: f32) -> f32 {
    let ax = abs(x);
    var res = -0.156583 * ax + HALF_PI;
    res *= sqrt(max(0.0, 1.0 - ax));
    return select(PI - res, res, x >= 0.0);
}

// -----------------------------------------------------------------------------
// Construct rotation matrix that rotates `v1` vector to `v2` vector.
// Adapted from XeGTAO reference implementation.
// -----------------------------------------------------------------------------
fn rot_from_to_matrix(v1: vec3<f32>, v2: vec3<f32>) -> mat3x3<f32> {
    let e = dot(v1, v2);
    let f = abs(e);
    if (f > 1.0 - 0.0003) {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }

    let v = cross(v1, v2);
    let h = 1.0 / (1.0 + e);
    let hvx = h * v.x;
    let hvz = h * v.z;
    let hvxy = hvx * v.y;
    let hvxz = hvx * v.z;
    let hvyz = hvz * v.y;

    return mat3x3<f32>(
        vec3<f32>(e + hvx * v.x, hvxy - v.z, hvxz + v.y),
        vec3<f32>(hvxy + v.z, e + h * v.y * v.y, hvyz - v.x),
        vec3<f32>(hvxz - v.y, hvyz + v.x, e + hvz * v.z),
    );
}

// -----------------------------------------------------------------------------
// Integrate visibility using XeGTAO analytic formulation. Also used for bent
// normal computation so returns both visibility contribution and bent normal
// contribution in a vec4 (xyz=bent, w=visibility).
// h0/h1 are signed horizon angles around the slice normal as described in the
// XeGTAO paper.
// -----------------------------------------------------------------------------
fn integrate_slice(
    dir2: vec2<f32>,
    n_angle: f32,
    cos_norm: f32,
    h0: f32,
    h1: f32,
    proj_len: f32,
    view_vec: vec3<f32>,
) -> vec4<f32> {
    let iarc0 = (cos_norm + 2.0 * h0 * sin(n_angle) - cos(2.0 * h0 - n_angle)) *
        0.25;
    let iarc1 = (cos_norm + 2.0 * h1 * sin(n_angle) - cos(2.0 * h1 - n_angle)) *
        0.25;
    let vis = proj_len * (iarc0 + iarc1);

    // Bent normal contribution (Algorithm 2 in XeGTAO paper)
    let t0 = (
        6.0 * sin(h0 - n_angle) - sin(3.0 * h0 - n_angle) +
        6.0 * sin(h1 - n_angle) - sin(3.0 * h1 - n_angle) +
        16.0 * sin(n_angle) -
        3.0 * (sin(h0 + n_angle) + sin(h1 + n_angle))
    ) / 12.0;
    let t1 = (
        -cos(3.0 * h0 - n_angle) - cos(3.0 * h1 - n_angle) +
        8.0 * cos(n_angle) -
        3.0 * (cos(h0 + n_angle) + cos(h1 + n_angle))
    ) / 12.0;

    var local_bent = vec3<f32>(dir2.x * t0, dir2.y * t0, -t1);
    local_bent = rot_from_to_matrix(vec3f(0.0, 0.0, -1.0), view_vec) * local_bent;
    local_bent *= proj_len;

    return vec4<f32>(local_bent, vis);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
@compute @workgroup_size(8,8,1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(position_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let xy   = gid.xy;
    let uv   = (vec2<f32>(xy) + 0.5) / vec2<f32>(dims);

    // Camera/view
    var view = view_buffer[u32(frame_info.view_index)];

    // G-buffer fetches
    let world_pos  = textureLoad(position_tex, xy, 0).xyz;
    var normal     = textureSampleLevel(normal_tex, non_filtering_sampler, uv, 0.0).xyz;
    normal         = normalize(normal);

    // TBN
    let tbn = make_tbn(normal);
    let T = tbn[0];
    let B = tbn[1];

    // Random per-pixel rotation to avoid banding
    let rand_ang = random_angle(xy, u32(frame_info.frame_index));
    let cos_r = cos(rand_ang);
    let sin_r = sin(rand_ang);

    // Slice base vectors in tangent plane (evenly spaced)
    // slice_k = rotate2D(rand) * baseAngle(k)
    // We'll build each slice direction in TB plane, then convert to WORLD (already world).
    var slice_dirs: array<vec3<f32>, NUM_SLICES>;
    for (var k = 0u; k < NUM_SLICES; k = k + 1u) {
        let base_ang = f32(k) * (TWO_PI / f32(NUM_SLICES));
        let ca = cos(base_ang);
        let sa = sin(base_ang);
        // rotate base_ang by rand_ang:
        let x = ca * cos_r - sa * sin_r;
        let y = ca * sin_r + sa * cos_r;
        // TB plane -> world
        slice_dirs[k] = normalize(T * x + B * y);
    }

    // Distribute taps across slices
    let total_taps = max(1u, u32(settings.sample_count));
    let steps_per_slice = max(MIN_STEPS_PER_SLICE, total_taps / (NUM_SLICES * 2u)); // *2 because +/- sides
    // Guarantee at least 1
    let steps_final = max(1u, steps_per_slice);

    // View direction for this pixel
    let view_vec = normalize(view.view_position.xyz - world_pos);

    // Horizon search per slice using XeGTAO analytic model
    var vis_accum: f32 = 0.0;
    var bent_accum = vec3<f32>(0.0);

    for (var s = 0u; s < NUM_SLICES; s = s + 1u) {
        let dir_s = slice_dirs[s];

        // Basis vectors relative to the view direction
        let ortho_dir = dir_s - dot(dir_s, view_vec) * view_vec;
        let axis_vec = normalize(cross(ortho_dir, view_vec));
        let proj_norm = normal - axis_vec * dot(normal, axis_vec);

        var sign_norm: f32 = 1.0;
        if (dot(ortho_dir, proj_norm) < 0.0) { sign_norm = -1.0; }

        let proj_len = max(length(proj_norm), 1e-4);
        let cos_norm = clamp(dot(proj_norm, view_vec) / proj_len, -1.0, 1.0);
        let n_angle = sign_norm * fast_acos(cos_norm);

        var horizon_cos0 = cos(n_angle + HALF_PI);
        var horizon_cos1 = cos(n_angle - HALF_PI);

        // search samples along this slice (+/- directions)
        let step_count_f = f32(steps_final);
        let sdir = normalize(dir_s);
        for (var i:u32 = 1u; i <= steps_final; i = i + 1u) {
            let t = f32(i) / step_count_f;
            let dist = t * settings.radius;

            let sample_w0 = world_pos + sdir * dist;
            let sample_view0 = (view.view_matrix * vec4f(sample_w0, 1.0)).xyz;
            let clip0 = view.projection_matrix * vec4f(sample_view0, 1.0);
            var uv0 = (clip0.xy / clip0.w) * 0.5 + 0.5;
            uv0.y = 1.0 - uv0.y;
            if (uv0.x >= 0.0 && uv0.x <= 1.0 && uv0.y >= 0.0 && uv0.y <= 1.0) {
                let scene0 = textureSampleLevel(position_tex, non_filtering_sampler, uv0, 0.0).xyz;
                let delta0 = scene0 - world_pos;
                if (dot(delta0, sdir) > 0.0) {
                    let shv0 = normalize(delta0);
                    let shc0 = dot(shv0, view_vec);
                    horizon_cos0 = max(horizon_cos0, shc0);
                }
            }

            let sample_w1 = world_pos - sdir * dist;
            let sample_view1 = (view.view_matrix * vec4f(sample_w1, 1.0)).xyz;
            let clip1 = view.projection_matrix * vec4f(sample_view1, 1.0);
            var uv1 = (clip1.xy / clip1.w) * 0.5 + 0.5;
            uv1.y = 1.0 - uv1.y;
            if (uv1.x >= 0.0 && uv1.x <= 1.0 && uv1.y >= 0.0 && uv1.y <= 1.0) {
                let scene1 = textureSampleLevel(position_tex, non_filtering_sampler, uv1, 0.0).xyz;
                let delta1 = scene1 - world_pos;
                if (dot(delta1, -sdir) > 0.0) {
                    let shv1 = normalize(delta1);
                    let shc1 = dot(shv1, view_vec);
                    horizon_cos1 = max(horizon_cos1, shc1);
                }
            }
        }

        let h0 = -fast_acos(horizon_cos1);
        let h1 = fast_acos(horizon_cos0);
        let contrib = integrate_slice(dir_s.xy, n_angle, cos_norm, h0, h1, proj_len, view_vec);
        vis_accum += contrib.w;
        bent_accum += contrib.xyz;
    }

    // after slice loop
    let ao_vis = clamp(vis_accum / f32(NUM_SLICES), 0.0, 1.0);

    // Get a non-normalized mean bent direction
    var bent_dir = vec3<f32>(0.0);
    if (vis_accum > 0.0) {
        bent_dir = bent_accum / vis_accum;
    }

    // Precompute some epsilons
    let len2 = dot(bent_dir, bent_dir);
    let eps_dir2 = 1e-6;        // squared-length threshold
    let eps_vis  = 0.05;       // require at least 5% “openness” for bent to be meaningful

    // Start with fallback = the surface normal
    var final_bent = normal;

    // If there is *enough* open hemisphere AND the integrated bent has non-zero length...
    if (ao_vis > eps_vis && len2 > eps_dir2) {
        final_bent = bent_dir / sqrt(len2);  // stable normalize
    }

    // Optional: you can blend fallback+ bent so it doesn’t snap harshly
    final_bent = mix(normal, final_bent, smoothstep(eps_vis, 1.0, ao_vis));

    textureStore(ao_texture, vec2<i32>(xy), vec4f(ao_vis, ao_vis, ao_vis, 1.0));
    textureStore(bent_output, vec2<i32>(xy), vec4f(final_bent, 1.0));
}

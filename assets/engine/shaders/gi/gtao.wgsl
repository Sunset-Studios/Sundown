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
@group(1) @binding(0) var depth_tex:    texture_2d<f32>;                  // bound; unused in base path
@group(1) @binding(1) var position_tex: texture_2d<f32>;                  // world position G-buffer
@group(1) @binding(2) var normal_tex:   texture_2d<f32>;                  // world normal G-buffer
@group(1) @binding(3) var ao_texture:   texture_storage_2d<r32float, write>;
@group(1) @binding(4) var bent_output:  texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var<uniform> settings: GTAOSettings;

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
// Integrate slice visibility from two horizon elevation angles.
// Angles are measured downward from N toward +/- slice axis in [0, π/2].
// Result ~fraction (0..1) of cosine-weighted hemisphere visible in this slice.
// This is a simplified analytic fit that behaves well for common ranges.
// -----------------------------------------------------------------------------
fn integrate_slice(h_neg: f32, h_pos: f32) -> f32 {
    // Clamp
    let hn = clamp(h_neg, 0.0, HALF_PI);
    let hp = clamp(h_pos, 0.0, HALF_PI);

    // Cosine-weighted visibility for each side approximated by cos(h)
    // (cos(0)=1 open, cos(π/2)=0 blocked). Average two sides.
    // Empirically scale slightly toward open to reduce over-occlusion.
    let v_neg = cos(hn);
    let v_pos = cos(hp);
    let v = 0.5 * (v_neg + v_pos);

    // Gentle contrast curve (empirical)
    return pow(v, 1.0); // change exponent to taste ( <1 brightens, >1 darkens )
}

// -----------------------------------------------------------------------------
// Bent-normal contribution from slice horizons.
// We choose the midpoint of the open cone on each side and weight by visibility.
// For +D side: open elev range = [0, π/2 - h_pos]; midpoint elev = half that.
// For -D side: open elev range = [0, π/2 - h_neg]; midpoint elev = half that.
// Direction = N*cos(mid) + +/-slice_dir*sin(mid).
// -----------------------------------------------------------------------------
fn bent_dir_for_side(n: vec3<f32>, slice_dir: vec3<f32>, h: f32, sign_dir: f32) -> vec3<f32> {
    let h_clamped = clamp(h, 0.0, HALF_PI);
    let open_elev = HALF_PI - h_clamped;
    let mid_elev = 0.5 * open_elev;
    // Build direction in the slice plane
    let sdir = normalize(slice_dir * sign_dir);
    let d = normalize(n * cos(mid_elev) + sdir * sin(mid_elev));
    return d;
}

// -----------------------------------------------------------------------------
// Horizon search along a slice direction.
// We march N steps out to radius along WORLD-space slice_dir and sample
// the G-buffer to estimate the maximum elevation angle of occluders on that side.
//
// Returns elevation angle in [0, π/2], measured downward from N toward slice_dir.
// -----------------------------------------------------------------------------
fn search_horizon_side(
    slice_dir: vec3<f32>,
    normal: vec3<f32>,
    world_pos: vec3<f32>,
    view: ptr<function, View>,
    max_radius: f32,
    steps: u32,
    sign_dir: f32                 // +1 for +slice_dir, -1 for -slice_dir
) -> f32 {

    let sdir = normalize(slice_dir * sign_dir);
    let step_count_f = f32(steps);

    var max_angle: f32 = 0.0;

    // Step distances (world). Uniform spacing; could switch to quadratic for QoL.
    for (var i:u32 = 1u; i <= steps; i = i + 1u) {
        let t = f32(i) / step_count_f;
        let dist = t * max_radius;

        let sample_world = world_pos + sdir * dist;
        let sample_view  = (view.view_matrix * vec4f(sample_world, 1.0)).xyz;
        let clip         = view.projection_matrix * vec4f(sample_view, 1.0);
        let uv           = (clip.xy / clip.w) * 0.5 + 0.5;

        // Off-screen -> treat as open; continue
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            continue;
        }

        // Fetch scene sample at that screen location
        let scene_world  = textureSampleLevel(position_tex, non_filtering_sampler, uv, 0.0).xyz;
        let v            = scene_world - world_pos;

        // Project v into slice plane components
        let dist_along   = dot(v, sdir);         // signed along slice side (should be >=0 to matter)
        if (dist_along <= 0.0) {
            continue;
        }

        let height       = dot(v, normal);            // height above surface plane
        if (height <= -settings.bias) {
            // Occluder below the plane; ignore (prevents self-darkening slopes)
            continue;
        }

        // Elevation angle downward from N toward slice_dir (0=open, π/2=horizon at 90° down)
        // Actually, if height < dist_along, angle < 45°, etc. Use atan2(height, dist).
        let ang = atan2(height, dist_along);     // 0..π/2+ (should clamp)
        max_angle = max(max_angle, ang);
    }

    // Clamp to hemisphere
    return clamp(max_angle, 0.0, HALF_PI);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
@compute @workgroup_size(8,8,1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(depth_tex);
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

    // Horizon search per slice
    var vis_accum: f32 = 0.0;
    var bent_accum = vec3<f32>(0.0);
    var weight_accum: f32 = 0.0;

    for (var s = 0u; s < NUM_SLICES; s = s + 1u) {
        let dir_s = slice_dirs[s];

        // +side and -side horizon elevation angles
        let h_pos = search_horizon_side(dir_s, normal, world_pos, &view, settings.radius, steps_final,  1.0);
        let h_neg = search_horizon_side(dir_s, normal, world_pos, &view, settings.radius, steps_final, -1.0);

        // Slice visibility (cosine-weighted approx)
        let v_slice = integrate_slice(h_neg, h_pos);
        vis_accum += v_slice;

        // Bent-normal contribution: midpoint directions for +/- sides, weight by each side's vis
        let side_weight_pos = cos(clamp(h_pos, 0.0, HALF_PI)); // same weighting as slice vis
        let side_weight_neg = cos(clamp(h_neg, 0.0, HALF_PI));

        let bent_pos = bent_dir_for_side(normal, dir_s, h_pos,  1.0);
        let bent_neg = bent_dir_for_side(normal, dir_s, h_neg, -1.0);

        bent_accum += bent_pos * side_weight_pos;
        bent_accum += bent_neg * side_weight_neg;
        weight_accum += (side_weight_pos + side_weight_neg);
    }

    // Final AO accessibility
    let ao_vis = vis_accum / f32(NUM_SLICES);

    // Final bent normal
    var bent = normal;
    if (weight_accum > 1e-5) {
        bent = normalize(bent_accum / weight_accum);
    }

    // Write outputs
    textureStore(ao_texture, vec2<i32>(xy), vec4f(ao_vis, ao_vis, ao_vis, 1.0));
    textureStore(bent_output, vec2<i32>(xy), vec4f(bent, 1.0));
}

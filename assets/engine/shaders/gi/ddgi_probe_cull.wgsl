// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                  DDGI PROBE FRUSTUM & OCCLUSION CULLING                   ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Performs frustum AND occlusion culling on DDGI probes to identify which  ║
// ║  probes are visible from the current camera view.                         ║
// ║                                                                           ║
// ║  Culling Strategy:                                                        ║
// ║  1. Frustum Test: Is the probe's influence sphere inside the view frustum?║
// ║  2. Occlusion Test: Is the probe visible in the Hierarchical Z-Buffer?    ║
// ║                                                                           ║
// ║  Output:                                                                  ║
// ║  - probe_cull_flags[word_index] = 32 bits (bit i = visible for            ║
// ║    probe word*32+i)                                                       ║
// ║    Write-only, no probe_states RMW; 32x fewer writes than per-probe.      ║
// ║                                                                           ║
// ║  This information is used by the probe scheduling system to prioritize    ║
// ║  updating probes that are visible to the camera before culled probes.     ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var hzb_texture: texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> probe_cull_flags: array<u32>;

const OCCLUSION_TESTING_GRID_SIZE: u32 = 2u;
// Each thread writes one u32 (32 visibility bits). No read-modify-write; minimal write traffic.
const PROBES_PER_WORD: u32 = 32u;

// =============================================================================
// FRUSTUM CULLING HELPERS
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Test if a sphere is inside or intersects the view frustum
// More conservative than point test - allows margin for probe influence radius
// ─────────────────────────────────────────────────────────────────────────────
fn is_sphere_in_frustum(center: vec3<f32>, radius: f32, view_index: u32) -> bool {
    let view = view_buffer[view_index];
    
    // Test against all 6 frustum planes with sphere radius margin
    let check0 = (dot(view.frustum[0].xyz, center) + view.frustum[0].w) >= -radius;
    let check1 = (dot(view.frustum[1].xyz, center) + view.frustum[1].w) >= -radius;
    let check2 = (dot(view.frustum[2].xyz, center) + view.frustum[2].w) >= -radius;
    let check3 = (dot(view.frustum[3].xyz, center) + view.frustum[3].w) >= -radius;
    let check4 = (dot(view.frustum[4].xyz, center) + view.frustum[4].w) >= -radius;
    let check5 = (dot(view.frustum[5].xyz, center) + view.frustum[5].w) >= -radius;
    
    return view.culling_enabled < 0.5 || (check0 && check1 && check2 && check3 && check4 && check5);
}

// =============================================================================
// OCCLUSION CULLING HELPERS (HZB-based)
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Project a sphere to screen-space UV rectangle
// Returns false if the sphere is completely off-screen or behind camera
// ─────────────────────────────────────────────────────────────────────────────
fn sphere_project_to_screen(
    center: vec3<f32>,
    radius: f32,
    view: View,
    out_uv_rect: ptr<function, vec4<f32>>
) -> bool {
    // Transform center to view space
    let center_view = view.view_matrix * vec4<f32>(center, 1.0);
    
    // If sphere is completely behind camera, skip
    if (center_view.z + radius >= 0.0) {
        return false;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Project sphere bounding box corners to get screen-space extent
    // This is a conservative approximation - the actual screen-space projection
    // of a sphere is an ellipse, but a box is sufficient for occlusion culling
    // ─────────────────────────────────────────────────────────────────────────
    let right = vec3<f32>(view.view_matrix[0][0], view.view_matrix[1][0], view.view_matrix[2][0]);
    let up = vec3<f32>(view.view_matrix[0][1], view.view_matrix[1][1], view.view_matrix[2][1]);
    
    // Build 8 corners of the sphere's bounding box in world space
    let corners = array<vec4<f32>, 8>(
        vec4<f32>(center + vec3<f32>(-radius, -radius, -radius), 1.0),
        vec4<f32>(center + vec3<f32>( radius, -radius, -radius), 1.0),
        vec4<f32>(center + vec3<f32>(-radius,  radius, -radius), 1.0),
        vec4<f32>(center + vec3<f32>( radius,  radius, -radius), 1.0),
        vec4<f32>(center + vec3<f32>(-radius, -radius,  radius), 1.0),
        vec4<f32>(center + vec3<f32>( radius, -radius,  radius), 1.0),
        vec4<f32>(center + vec3<f32>(-radius,  radius,  radius), 1.0),
        vec4<f32>(center + vec3<f32>( radius,  radius,  radius), 1.0)
    );
    
    // Initialize NDC min/max
    var ndc_min = vec2<f32>(1.0, 1.0);
    var ndc_max = vec2<f32>(-1.0, -1.0);
    var valid_corners = 0u;
    
    // Project each corner into NDC
    for (var i = 0u; i < 8u; i = i + 1u) {
        let cv = view.view_matrix * corners[i];
        // Skip corners behind camera
        if (cv.z >= 0.0) {
            continue;
        }
        let clip = view.projection_matrix * cv;
        let ndc = clip.xy / clip.w;
        ndc_min = min(ndc_min, ndc);
        ndc_max = max(ndc_max, ndc);
        valid_corners = valid_corners + 1u;
    }
    
    // If completely off-screen or no valid corners, no occlusion test needed
    if (valid_corners == 0u || ndc_max.x < -1.0 || ndc_min.x > 1.0 ||
        ndc_max.y < -1.0 || ndc_min.y > 1.0) {
        return false;
    }
    
    // Convert NDC box to [0..1] UV (0,0=top-left)
    let u_min = clamp(ndc_min.x * 0.5 + 0.5, 0.0, 1.0);
    let u_max = clamp(ndc_max.x * 0.5 + 0.5, 0.0, 1.0);
    let v_min = clamp(-ndc_max.y * 0.5 + 0.5, 0.0, 1.0);
    let v_max = clamp(-ndc_min.y * 0.5 + 0.5, 0.0, 1.0);
    
    *out_uv_rect = vec4<f32>(u_min, v_min, u_max, v_max);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test if a sphere is occluded using the Hierarchical Z-Buffer
// Returns true if the sphere is fully occluded (not visible)
// ─────────────────────────────────────────────────────────────────────────────
fn is_sphere_occluded(center: vec3<f32>, radius: f32, view_index: u32) -> bool {
    let view = view_buffer[view_index];
    
    // Skip occlusion test if disabled in view settings
    if (view.occlusion_enabled < 0.5) {
        return false;
    }
    
    // Project sphere to screen space
    var uv_rect: vec4<f32>;
    if (!sphere_project_to_screen(center, radius, view, &uv_rect)) {
        // If projection fails (behind camera or off-screen), not occluded
        // (frustum test handles off-screen, here we're conservative)
        return false;
    }
    
    // Guard degenerate or inverted rects
    if (uv_rect.z <= uv_rect.x || uv_rect.w <= uv_rect.y) {
        return false;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compute sphere depth in view-space (closest point to camera)
    // ─────────────────────────────────────────────────────────────────────────
    let center_view = view.view_matrix * vec4<f32>(center, 1.0);
    let sphere_depth = -center_view.z - radius;  // Closest point of sphere
    
    // If closest point is behind camera, not visible anyway
    if (sphere_depth <= 0.0) {
        return false;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Determine appropriate HZB mip level based on screen-space size
    // ─────────────────────────────────────────────────────────────────────────
    let hzb_dims = textureDimensions(hzb_texture);
    let width = (uv_rect.z - uv_rect.x) * f32(hzb_dims.x);
    let height = (uv_rect.w - uv_rect.y) * f32(hzb_dims.y);
    let screen_size = max(width, height);
    
    let level_floor = floor(log2(max(screen_size, 1.0)));
    let max_level = f32(textureNumLevels(hzb_texture) - 1u);
    let level = clamp(level_floor, 0.0, max_level);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Sample HZB at multiple points to get conservative depth
    // We use the maximum depth (furthest) from the samples
    // ─────────────────────────────────────────────────────────────────────────
    let u_min = uv_rect.x;
    let v_min = uv_rect.y;
    let u_max = uv_rect.z;
    let v_max = uv_rect.w;
    
    // Sample a NxN grid for robust occlusion testing
    var max_depth: f32 = 0.0;
    
    for (var ix = 0u; ix < OCCLUSION_TESTING_GRID_SIZE; ix = ix + 1u) {
        for (var iy = 0u; iy < OCCLUSION_TESTING_GRID_SIZE; iy = iy + 1u) {
            let uv = vec2<f32>(
                mix(u_min, u_max, f32(ix) / f32(OCCLUSION_TESTING_GRID_SIZE - 1u)),
                mix(v_min, v_max, f32(iy) / f32(OCCLUSION_TESTING_GRID_SIZE - 1u))
            );
            let raw_d = textureSampleLevel(hzb_texture, non_filtering_sampler, uv, level).r;
            let lin_d = linearize_depth(raw_d, view.near, view.far, view_index);
            max_depth = max(max_depth, lin_d);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Compare sphere depth with HZB depth
    // Add small bias to reduce flickering at depth discontinuities
    // ─────────────────────────────────────────────────────────────────────────
    let depth_bias = radius * 0.5;  // Bias proportional to sphere size
    let is_visible = sphere_depth < (max_depth + depth_bias);
    
    return !is_visible;
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================
// Each thread writes one u32 (32 probes' visibility bits). No probe_states read/write.
// Dispatch: ceil(probe_count / 32 / 256) workgroups.
// =============================================================================

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let probe_count = u32(ddgi_params.probe_counts.x);
    let view_index = u32(frame_info.view_index);
    let word_index = gid.x;
    let base_index = word_index * PROBES_PER_WORD;

    var bits: u32 = 0u;
    for (var i = 0u; i < PROBES_PER_WORD; i = i + 1u) {
        let probe_index = base_index + i;
        if (probe_index >= probe_count) {
            break;
        }
        let probe_pos = ddgi_probe_world_position_from_index(&ddgi_params, probe_index);
        let probe_spacing = ddgi_probe_spacing_from_index(&ddgi_params, probe_index);
        let influence_radius = probe_spacing * 0.75;
        var is_visible = is_sphere_in_frustum(probe_pos, influence_radius, view_index);
        if (is_visible) {
            is_visible = !is_sphere_occluded(probe_pos, influence_radius, view_index);
        }
        if (is_visible) {
            bits = bits | (1u << i);
        }
    }
    
    probe_cull_flags[word_index] = bits;
}

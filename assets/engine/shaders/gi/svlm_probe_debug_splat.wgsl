#include "gi/svlm_common.wgsl"
#include "sh_common.wgsl"

// Probe debug splat pass.
//
// Leaf bricks already contain origin and size, so probe centers are inferred as
// a 4x4x4 lattice instead of reading a separate probe-position buffer. Each
// thread owns one local probe from the camera-local gather list and splats a
// small screen-space SDF sphere into the packed debug depth buffer.

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;
@group(1) @binding(2) var<storage, read> leaf_bricks: array<SVLMLeafBrick>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var<storage, read_write> debug_depth: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read> debug_leaf_indices: array<u32>;
@group(1) @binding(6) var<storage, read> irradiance_probes: array<u32>;
@group(1) @binding(7) var<storage, read> streamed_probe_validity: array<u32>;

const SVLM_DEBUG_MAX_RADIUS_PX = 18.0;
const SVLM_PROBE_DEBUG_WORKGROUP_Y = 8u;

fn svlm_probe_visible_for_debug_level(leaf_level: u32) -> bool {
    let debug_level = i32(svlm_params.debug_level);
    return debug_level < 0 || leaf_level == u32(debug_level);
}

fn svlm_probe_debug_read_sh(probe_index: u32) -> SH_L1_RGB {
    let base = probe_index * SVLM_SH_WORDS_PER_PROBE;
    var packed: SH_L1_RGB_Packed;
    for (var i = 0u; i < SVLM_SH_WORDS_PER_PROBE; i = i + 1u) {
        packed.data[i] = irradiance_probes[base + i];
    }
    return sh_l1_rgb_unpack(packed);
}

fn svlm_probe_debug_streamed_index(
    leaf_index: u32,
    leaf: SVLMLeafBrick,
    local_probe: u32
) -> u32 {
    let validity_base = leaf_index * 2u;
    let word_index = local_probe >> 5u;
    if (validity_base + word_index >= arrayLength(&streamed_probe_validity)) {
        return INVALID_IDX;
    }
    let validity_word = streamed_probe_validity[validity_base + word_index];
    let bit = 1u << (local_probe & 31u);
    if ((validity_word & bit) == 0u) {
        return INVALID_IDX;
    }
    var rank = countOneBits(validity_word & (bit - 1u));
    if (word_index != 0u) {
        rank += countOneBits(streamed_probe_validity[validity_base]);
    }
    return leaf.probe_base + rank;
}

fn svlm_probe_debug_pack_rgb565(color: vec3<f32>) -> u32 {
    let saturated = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let r = u32(round(saturated.r * 31.0));
    let g = u32(round(saturated.g * 63.0));
    let b = u32(round(saturated.b * 31.0));
    return (r << 11u) | (g << 5u) | b;
}

// Depth must dominate the packed value so atomicMin keeps the nearest splat.
// The low 16 bits carry a display-mapped copy of the baked irradiance.
fn svlm_probe_debug_pack_distance(distance: f32, view_index: u32, color: vec3<f32>) -> u32 {
    let far_plane = max(view_buffer[view_index].far, 1.0);
    let depth16 = u32(clamp(distance / far_plane, 0.0, 1.0) * 65534.0);
    return (depth16 << 16u) | svlm_probe_debug_pack_rgb565(color);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_probe = gid.x;
    let page_groups_y = max(u32(max(svlm_params.debug_leaf_page_groups_y, 0.0)), 1u);
    // Dispatch dimensions are paged so we can cover large leaf buffers without
    // exceeding maxComputeWorkgroupsPerDimension in Y.
    let selected_leaf_slot = gid.y + gid.z * page_groups_y * SVLM_PROBE_DEBUG_WORKGROUP_Y;
    if (local_probe >= SVLM_PROBES_PER_BRICK) {
        return;
    }

    let selected_leaf_count = min(debug_leaf_indices[0], arrayLength(&debug_leaf_indices) - 1u);
    if (selected_leaf_slot >= selected_leaf_count) {
        return;
    }

    let leaf_index = debug_leaf_indices[selected_leaf_slot + 1u];
    if (leaf_index == 0xffffffffu) {
        return;
    }

    if (leaf_index >= arrayLength(&leaf_bricks)) {
        return;
    }

    let leaf = leaf_bricks[leaf_index];
    let leaf_level = leaf.level;
    if (!svlm_probe_visible_for_debug_level(leaf_level)) {
        return;
    }

    let leaf_size = leaf.size;
    let leaf_origin = vec3<f32>(
        leaf.origin_x,
        leaf.origin_y,
        leaf.origin_z
    );
    let local_coord = vec3<u32>(
        local_probe & 3u,
        (local_probe >> 2u) & 3u,
        (local_probe >> 4u) & 3u
    );
    // Cell-centered probes avoid placing entire probe planes on leaf boundaries
    // and thin geometry. This must exactly match bake and runtime sampling.
    let probe_spacing = max(leaf_size / 4.0, 0.0001);
    let probe_position =
        leaf_origin + (vec3<f32>(local_coord) + vec3<f32>(0.5)) * probe_spacing;
    let radius = max(probe_spacing * 0.075, 0.025);
    let probe_index = select(
        leaf.probe_base + local_probe,
        svlm_probe_debug_streamed_index(leaf_index, leaf, local_probe),
        svlm_params.tile_streaming_enabled > 0.5
    );
    if (probe_index == INVALID_IDX) {
        return;
    }
    let last_probe_word =
        probe_index * SVLM_SH_WORDS_PER_PROBE + (SVLM_SH_WORDS_PER_PROBE - 1u);
    if (last_probe_word >= arrayLength(&irradiance_probes)) {
        return;
    }
    let completed_probe_samples = atomicLoad(
        &svlm_counters.irradiance_completed_probe_samples
    );
    let probe_has_sample = completed_probe_samples > probe_index;
    var display_color = vec3<f32>(1.0, 0.0, 1.0);
    if (probe_has_sample) {
        let probe_sh = svlm_probe_debug_read_sh(probe_index);
        let baked_irradiance = max(
            probe_sh.c[0] * (PI * SH_BASIS_L0),
            vec3<f32>(0.0)
        );
        // A gentle photographic mapping preserves dim bounce light. Keep a
        // small visible marker for valid-but-zero SH so black probes cannot be
        // mistaken for missing splats; unwritten probes remain bright magenta.
        let exposed = baked_irradiance * 2.0;
        display_color = pow(
            exposed / (vec3<f32>(1.0) + exposed),
            vec3<f32>(1.0 / 2.2)
        );
        let peak = max(max(display_color.r, display_color.g), display_color.b);
        if (peak <= 1e-5) {
            display_color = vec3<f32>(0.10, 0.025, 0.10);
        } else if (peak < 0.15) {
            display_color *= 0.15 / peak;
        }
    }
    let view_index = u32(frame_info.view_index);
    let view_position = view_buffer[view_index].view_matrix * vec4<f32>(probe_position, 1.0);
    let center_clip = view_buffer[view_index].projection_matrix * view_position;
    if (center_clip.w <= 0.00001) {
        return;
    }

    let edge_clip = view_buffer[view_index].projection_matrix * vec4<f32>(view_position.xyz + vec3<f32>(radius, 0.0, 0.0), 1.0);
    if (edge_clip.w <= 0.00001) {
        return;
    }

    let res = textureDimensions(depth_texture);
    let center_ndc = center_clip.xy / center_clip.w;
    let edge_ndc = edge_clip.xy / edge_clip.w;
    let center_px = vec2<f32>(
        (center_ndc.x * 0.5 + 0.5) * f32(res.x),
        (1.0 - (center_ndc.y * 0.5 + 0.5)) * f32(res.y)
    );
    let radius_px = clamp(abs(edge_ndc.x - center_ndc.x) * 0.5 * f32(res.x), 1.0, SVLM_DEBUG_MAX_RADIUS_PX);
    let min_px = vec2<i32>(max(vec2<f32>(0.0), floor(center_px - vec2<f32>(radius_px))));
    let max_pixel_f = vec2<f32>(f32(res.x - 1u), f32(res.y - 1u));
    let max_px = vec2<i32>(min(max_pixel_f, ceil(center_px + vec2<f32>(radius_px))));
    if (min_px.x > max_px.x || min_px.y > max_px.y) {
        return;
    }

    let view_distance = max(length(view_position.xyz), 0.0001);
    let center_depth = center_clip.z / center_clip.w;
    if (center_depth < 0.0 || center_depth > 1.0) {
        return;
    }

    let light_dir = safe_normalize(vec3<f32>(0.35, -0.55, 0.75));
    for (var py = min_px.y; py <= max_px.y; py = py + 1) {
        for (var px = min_px.x; px <= max_px.x; px = px + 1) {
            let pixel_center = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);
            let screen_delta = (pixel_center - center_px) / radius_px;
            let sdf_sq = dot(screen_delta, screen_delta);
            if (sdf_sq > 1.0) {
                continue;
            }

            let scene_depth = textureLoad(depth_texture, vec2<i32>(px, py), 0).r;
            if (scene_depth < 1.0 && scene_depth + 0.0005 < center_depth) {
                continue;
            }

            let sphere_normal = safe_normalize(vec3<f32>(screen_delta.x, -screen_delta.y, sqrt(max(0.0, 1.0 - sdf_sq))));
            let shade = 0.72 + 0.28 * max(dot(sphere_normal, light_dir), 0.0);
            let surface_distance = max(0.0, view_distance - sphere_normal.z * radius);
            let packed = svlm_probe_debug_pack_distance(
                surface_distance,
                view_index,
                display_color * shade
            );
            let pixel_index = u32(py) * res.x + u32(px);
            // Lower packed depth wins; the low appearance bits survive for resolve.
            atomicMin(&debug_depth[pixel_index], packed);
        }
    }
}

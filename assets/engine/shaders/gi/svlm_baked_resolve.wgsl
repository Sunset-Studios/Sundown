#include "gi/svlm_common.wgsl"
#include "sh_common.wgsl"

// Screen-space resolve for the static SVLM bake. Most pixels use one leaf-local
// trilinear SH sample. Pixels within one probe spacing of an adaptive leaf
// boundary additionally blend the adjacent leaf estimates to remove brick
// seams without paying that cost throughout leaf interiors.

@group(1) @binding(0) var depth_texture: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(3) var<storage, read> svlm_lookup_data: array<u32>;
@group(1) @binding(4) var<storage, read> leaf_bricks: array<SVLMLeafBrick>;
@group(1) @binding(5) var<storage, read> irradiance_probes: array<u32>;
@group(1) @binding(6) var<storage, read> coarse_lookup_data: array<u32>;
@group(1) @binding(7) var<storage, read> svlm_local_pages: array<u32>;
@group(1) @binding(8) var<storage, read> streamed_probe_validity: array<u32>;
@group(1) @binding(9) var output_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(10) var output_black: texture_storage_2d<rgba16float, write>;

const SVLM_NODE_WORD_STRIDE = 7u;
const SVLM_COVERAGE_DIRECTORY_WORD_STRIDE = 8u;
const SVLM_LOCAL_PAGE_WORD_STRIDE = 5u;
const SVLM_PAGE_TABLE_MAX_PROBES = 16u;
const SVLM_DIRECTORY_TOMBSTONE = 0xfffffffeu;
const SVLM_COARSE_LOOKUP_WORD_STRIDE = 10u;
const SVLM_COARSE_LOOKUP_MAX_PROBES = 16u;

struct SVLMTiledLeafLookup {
    leaf_index: u32,
    fade_start_time: f32,
};

fn svlm_sample_output_to_full_res_coord(
    sample_coord: vec2<u32>,
    sample_size: vec2<u32>,
    full_size: vec2<u32>
) -> vec2<i32> {
    let uv =
        (vec2<f32>(sample_coord) + vec2<f32>(0.5)) /
        vec2<f32>(sample_size);
    let full_coord = min(
        vec2<u32>(uv * vec2<f32>(full_size)),
        full_size - vec2<u32>(1u)
    );
    return vec2<i32>(full_coord);
}

fn svlm_read_node(node_index: u32) -> SVLMNode {
    let base = node_index * SVLM_NODE_WORD_STRIDE;
    return SVLMNode(
        svlm_lookup_data[base],
        svlm_lookup_data[base + 1u],
        svlm_lookup_data[base + 2u],
        svlm_lookup_data[base + 3u],
        svlm_lookup_data[base + 4u],
        svlm_lookup_data[base + 5u],
        svlm_lookup_data[base + 6u]
    );
}

fn svlm_find_monolithic_leaf(position: vec3<f32>) -> u32 {
    let world_min = svlm_world_min(&svlm_params);
    let root_size = max(svlm_params.root_size, 0.0001);
    let root_dims = svlm_root_dims(&svlm_params);
    if (any(root_dims == vec3<u32>(0u))) {
        return INVALID_IDX;
    }
    let root_extent = vec3<f32>(root_dims) * root_size;
    let local_position = position - world_min;

    if (
        any(local_position < vec3<f32>(0.0)) ||
        any(local_position >= root_extent)
    ) {
        return INVALID_IDX;
    }

    let root_coord = min(
        vec3<u32>(floor(local_position / root_size)),
        root_dims - vec3<u32>(1u)
    );
    var node_index =
        root_coord.x +
        root_coord.y * root_dims.x +
        root_coord.z * root_dims.x * root_dims.y;

    for (var step = 0u; step < 10u; step = step + 1u) {
        if (
            node_index >=
                arrayLength(&svlm_lookup_data) / SVLM_NODE_WORD_STRIDE
        ) {
            return INVALID_IDX;
        }

        let node = svlm_read_node(node_index);
        if (
            (node.flags & SVLM_FLAG_LEAF) != 0u &&
            node.leaf_index != INVALID_IDX
        ) {
            return node.leaf_index;
        }
        if (node.child_base == INVALID_IDX) {
            return INVALID_IDX;
        }

        let node_size = svlm_brick_size(&svlm_params, node.level);
        let node_origin =
            world_min +
            vec3<f32>(f32(node.coord_x), f32(node.coord_y), f32(node.coord_z)) * node_size;
        let child_size = node_size * 0.5;
        let child_coord = select(
            vec3<u32>(0u),
            vec3<u32>(1u),
            position >= node_origin + vec3<f32>(child_size)
        );
        let child_offset =
            child_coord.x +
            (child_coord.y << 1u) +
            (child_coord.z << 2u);
        node_index = node.child_base + child_offset;
    }

    return INVALID_IDX;
}

fn svlm_hash_lookup_words(
    word_0: u32,
    word_1: u32,
    word_2: u32,
    word_3: u32
) -> u32 {
    var hash = 0x811c9dc5u;
    hash = (hash ^ word_0) * 0x01000193u;
    hash = (hash ^ word_1) * 0x01000193u;
    hash = (hash ^ word_2) * 0x01000193u;
    hash = (hash ^ word_3) * 0x01000193u;
    hash ^= hash >> 16u;
    hash *= 0x7feb352du;
    hash ^= hash >> 15u;
    hash *= 0x846ca68bu;
    return hash ^ (hash >> 16u);
}

fn svlm_hash_coverage_coord(tile_coord: vec3<i32>) -> u32 {
    return svlm_hash_lookup_words(
        bitcast<u32>(tile_coord.x),
        bitcast<u32>(tile_coord.y),
        bitcast<u32>(tile_coord.z),
        0u
    );
}

fn svlm_hash_local_page_key(
    level: u32,
    leaf_coord: vec3<u32>
) -> u32 {
    return svlm_hash_lookup_words(level, leaf_coord.x, leaf_coord.y, leaf_coord.z);
}

fn svlm_lookup_tiled_leaf(
    tile_coord: vec3<i32>,
    level: u32,
    leaf_coord: vec3<u32>
) -> SVLMTiledLeafLookup {
    let directory_entry_count =
        arrayLength(&svlm_lookup_data) /
        SVLM_COVERAGE_DIRECTORY_WORD_STRIDE;
    if (directory_entry_count == 0u) {
        return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
    }
    let directory_mask = directory_entry_count - 1u;
    var directory_index = svlm_hash_coverage_coord(tile_coord) & directory_mask;
    var page_offset = INVALID_IDX;
    var page_mask = 0u;
    var fade_start_time = 0.0;
    for (
        var probe = 0u;
        probe < SVLM_PAGE_TABLE_MAX_PROBES;
        probe = probe + 1u
    ) {
        let base = directory_index * SVLM_COVERAGE_DIRECTORY_WORD_STRIDE;
        let candidate_page_offset = svlm_lookup_data[base + 3u];
        if (candidate_page_offset == INVALID_IDX) {
            return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
        }
        if (
            candidate_page_offset != SVLM_DIRECTORY_TOMBSTONE &&
            bitcast<i32>(svlm_lookup_data[base]) == tile_coord.x &&
            bitcast<i32>(svlm_lookup_data[base + 1u]) == tile_coord.y &&
            bitcast<i32>(svlm_lookup_data[base + 2u]) == tile_coord.z
        ) {
            page_offset = candidate_page_offset;
            page_mask = svlm_lookup_data[base + 4u];
            fade_start_time = bitcast<f32>(svlm_lookup_data[base + 5u]);
            break;
        }
        directory_index = (directory_index + 1u) & directory_mask;
    }
    if (page_offset == INVALID_IDX) {
        return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
    }

    var local_entry_index = svlm_hash_local_page_key(level, leaf_coord) & page_mask;
    for (
        var probe = 0u;
        probe < SVLM_PAGE_TABLE_MAX_PROBES;
        probe = probe + 1u
    ) {
        let entry_index = page_offset + local_entry_index;
        let base = entry_index * SVLM_LOCAL_PAGE_WORD_STRIDE;
        if (base + 4u >= arrayLength(&svlm_local_pages)) {
            return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
        }
        let leaf_index = svlm_local_pages[base + 4u];
        if (leaf_index == INVALID_IDX) {
            return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
        }
        if (
            svlm_local_pages[base] == level &&
            all(vec3<u32>(
                svlm_local_pages[base + 1u],
                svlm_local_pages[base + 2u],
                svlm_local_pages[base + 3u]
            ) == leaf_coord)
        ) {
            return SVLMTiledLeafLookup(leaf_index, fade_start_time);
        }
        local_entry_index = (local_entry_index + 1u) & page_mask;
    }
    return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
}

fn svlm_find_tiled_leaf(position: vec3<f32>) -> SVLMTiledLeafLookup {
    if (svlm_params.resident_tile_count <= 0.0) {
        return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
    }
    let world_min = svlm_world_min(&svlm_params);
    let root_size = max(svlm_params.root_size, 0.0001);
    let root_dims = svlm_root_dims(&svlm_params);
    let local_position = position - world_min;
    if (
        any(root_dims == vec3<u32>(0u)) ||
        any(local_position < vec3<f32>(0.0)) ||
        any(local_position >= vec3<f32>(root_dims) * root_size)
    ) {
        return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
    }

    let tile_size = max(svlm_params.world_tile_size, 0.0001);
    let tile_coord = vec3<i32>(floor(position / tile_size));
    var level = u32(max(svlm_params.max_level, 0.0));
    loop {
        let leaf_size = svlm_brick_size(&svlm_params, level);
        let leaf_coord = vec3<u32>(floor(local_position / leaf_size));
        let lookup =
            svlm_lookup_tiled_leaf(tile_coord, level, leaf_coord);
        if (lookup.leaf_index != INVALID_IDX) {
            return lookup;
        }
        if (level == 0u) {
            break;
        }
        level -= 1u;
    }
    return SVLMTiledLeafLookup(INVALID_IDX, 0.0);
}

fn svlm_find_leaf_sample(position: vec3<f32>) -> SVLMTiledLeafLookup {
    if (svlm_params.tile_streaming_enabled > 0.5) {
        return svlm_find_tiled_leaf(position);
    }
    return SVLMTiledLeafLookup(svlm_find_monolithic_leaf(position), 0.0);
}

fn svlm_find_leaf(position: vec3<f32>) -> u32 {
    return svlm_find_leaf_sample(position).leaf_index;
}

fn svlm_tiled_leaf_fade(lookup: SVLMTiledLeafLookup) -> f32 {
    if (
        svlm_params.tile_streaming_enabled <= 0.5 ||
        svlm_params.streaming_fade_seconds <= 1e-4
    ) {
        return 1.0;
    }
    let fade_elapsed = max(frame_info.time - lookup.fade_start_time, 0.0);
    return smoothstep(
        0.0,
        svlm_params.streaming_fade_seconds,
        fade_elapsed
    );
}

fn svlm_read_probe_sh(probe_index: u32) -> SH_L1_RGB {
    let base = probe_index * SVLM_SH_WORDS_PER_PROBE;
    var packed: SH_L1_RGB_Packed;
    packed.data[0] = irradiance_probes[base];
    packed.data[1] = irradiance_probes[base + 1u];
    packed.data[2] = irradiance_probes[base + 2u];
    packed.data[3] = irradiance_probes[base + 3u];
    packed.data[4] = irradiance_probes[base + 4u];
    packed.data[5] = irradiance_probes[base + 5u];
    return sh_l1_rgb_unpack(packed);
}

fn svlm_probe_is_valid(probe_index: u32) -> bool {
    let base = probe_index * SVLM_SH_WORDS_PER_PROBE;
    return !(
        irradiance_probes[base] == SVLM_INVALID_PROBE_WORD_0 &&
        (irradiance_probes[base + 1u] &
            SVLM_INVALID_PROBE_WORD_1_MASK) ==
            SVLM_INVALID_PROBE_WORD_1_VALUE
    );
}

struct SVLMFineSample {
    irradiance: SH_L1_RGB,
    validity: f32,
};

struct SVLMCoarseSample {
    irradiance: SH_L1_RGB,
    validity: f32,
};

fn svlm_sh_multiply_add(
    accumulator: SH_L1_RGB,
    value: SH_L1_RGB,
    weight: f32
) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    result.c[0] = accumulator.c[0] + value.c[0] * weight;
    result.c[1] = accumulator.c[1] + value.c[1] * weight;
    result.c[2] = accumulator.c[2] + value.c[2] * weight;
    result.c[3] = accumulator.c[3] + value.c[3] * weight;
    return result;
}

fn svlm_streamed_probe_index(
    leaf_index: u32,
    leaf: SVLMLeafBrick,
    local_probe_index: u32
) -> u32 {
    let validity_base = leaf_index * 2u;
    let validity_word_index = local_probe_index >> 5u;
    if (validity_base + validity_word_index >= arrayLength(&streamed_probe_validity)) {
        return INVALID_IDX;
    }
    let validity_word = streamed_probe_validity[validity_base + validity_word_index];
    let bit_index = local_probe_index & 31u;
    let bit = 1u << bit_index;
    if ((validity_word & bit) == 0u) {
        return INVALID_IDX;
    }
    var rank = countOneBits(validity_word & (bit - 1u));
    if (validity_word_index != 0u) {
        rank += countOneBits(streamed_probe_validity[validity_base]);
    }
    return leaf.probe_base + rank;
}

fn svlm_leaf_probe_index(
    leaf_index: u32,
    leaf: SVLMLeafBrick,
    local_probe_index: u32
) -> u32 {
    if (svlm_params.tile_streaming_enabled > 0.5) {
        return svlm_streamed_probe_index(leaf_index, leaf, local_probe_index);
    }
    return leaf.probe_base + local_probe_index;
}

fn svlm_hash_coarse_lookup_key(tile_coord: vec3<i32>, lod: u32) -> u32 {
    var hash = 0x811c9dc5u;
    hash = (hash ^ bitcast<u32>(tile_coord.x)) * 0x01000193u;
    hash = (hash ^ bitcast<u32>(tile_coord.y)) * 0x01000193u;
    hash = (hash ^ bitcast<u32>(tile_coord.z)) * 0x01000193u;
    hash = (hash ^ lod) * 0x01000193u;
    hash ^= hash >> 16u;
    hash *= 0x7feb352du;
    hash ^= hash >> 15u;
    hash *= 0x846ca68bu;
    return hash ^ (hash >> 16u);
}

fn svlm_lookup_coarse_entry(tile_coord: vec3<i32>, lod: u32) -> u32 {
    let entry_count =
        arrayLength(&coarse_lookup_data) /
        SVLM_COARSE_LOOKUP_WORD_STRIDE;
    if (entry_count == 0u) {
        return INVALID_IDX;
    }
    let entry_mask = entry_count - 1u;
    var entry_index =
        svlm_hash_coarse_lookup_key(tile_coord, lod) & entry_mask;
    for (
        var probe = 0u;
        probe < SVLM_COARSE_LOOKUP_MAX_PROBES;
        probe = probe + 1u
    ) {
        let base = entry_index * SVLM_COARSE_LOOKUP_WORD_STRIDE;
        let entry_lod = coarse_lookup_data[base + 3u];
        if (entry_lod == INVALID_IDX) {
            return INVALID_IDX;
        }
        if (
            bitcast<i32>(coarse_lookup_data[base]) == tile_coord.x &&
            bitcast<i32>(coarse_lookup_data[base + 1u]) == tile_coord.y &&
            bitcast<i32>(coarse_lookup_data[base + 2u]) == tile_coord.z &&
            entry_lod == lod
        ) {
            return base;
        }
        entry_index = (entry_index + 1u) & entry_mask;
    }
    return INVALID_IDX;
}

fn svlm_read_coarse_sh(entry_base: u32) -> SH_L1_RGB {
    var packed: SH_L1_RGB_Packed;
    packed.data[0] = coarse_lookup_data[entry_base + 4u];
    packed.data[1] = coarse_lookup_data[entry_base + 5u];
    packed.data[2] = coarse_lookup_data[entry_base + 6u];
    packed.data[3] = coarse_lookup_data[entry_base + 7u];
    packed.data[4] = coarse_lookup_data[entry_base + 8u];
    packed.data[5] = coarse_lookup_data[entry_base + 9u];
    return sh_l1_rgb_unpack(packed);
}

fn svlm_sample_coarse_lod(position: vec3<f32>, lod: u32) -> SVLMCoarseSample {
    let tile_size =
        max(svlm_params.world_tile_size, 0.0001) *
        exp2(f32(lod));
    let grid_position = position / tile_size - vec3<f32>(0.5);
    let base_coord = vec3<i32>(floor(grid_position));
    let fraction = fract(grid_position);
    var result = sh_l1_rgb_zero();
    var weight_sum = 0.0;
    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        let offset = vec3<i32>(
            i32(corner & 1u),
            i32((corner >> 1u) & 1u),
            i32((corner >> 2u) & 1u)
        );
        let entry_base = svlm_lookup_coarse_entry(base_coord + offset, lod);
        if (entry_base == INVALID_IDX) {
            continue;
        }
        let weight_axis = select(
            vec3<f32>(1.0) - fraction,
            fraction,
            offset == vec3<i32>(1)
        );
        let weight = weight_axis.x * weight_axis.y * weight_axis.z;
        result = svlm_sh_multiply_add(
            result,
            svlm_read_coarse_sh(entry_base),
            weight
        );
        weight_sum += weight;
    }
    if (weight_sum <= 1e-6) {
        return SVLMCoarseSample(sh_l1_rgb_zero(), 0.0);
    }
    return SVLMCoarseSample(
        sh_l1_rgb_multiply_scalar(result, 1.0 / weight_sum),
        1.0
    );
}

fn svlm_sample_coarse(position: vec3<f32>) -> SVLMCoarseSample {
    let min_lod = u32(max(svlm_params.coarse_min_lod, 1.0));
    let max_lod = u32(max(svlm_params.coarse_max_lod, f32(min_lod)));
    for (var lod = min_lod; lod <= max_lod; lod = lod + 1u) {
        let sample = svlm_sample_coarse_lod(position, lod);
        if (sample.validity > 0.5 || lod == max_lod) {
            return sample;
        }
    }
    return SVLMCoarseSample(sh_l1_rgb_zero(), 0.0);
}

fn svlm_leaf_contains(
    leaf: SVLMLeafBrick,
    position: vec3<f32>
) -> bool {
    let leaf_min = vec3<f32>(
        leaf.origin_x,
        leaf.origin_y,
        leaf.origin_z
    );
    let leaf_size = max(leaf.size, 0.0001);
    let epsilon = max(leaf_size * 1e-5, 1e-5);
    return
        all(position >= leaf_min - vec3<f32>(epsilon)) &&
        all(position <= leaf_min + vec3<f32>(leaf_size + epsilon));
}

fn svlm_probe_surface_weight(
    probe_position: vec3<f32>,
    surface_position: vec3<f32>,
    surface_normal: vec3<f32>,
    probe_spacing: f32
) -> f32 {
    let signed_plane_distance =
        dot(surface_normal, probe_position - surface_position) /
        max(probe_spacing, 0.0001);
    // Unlike DDGI, SVLM has no per-direction depth moments. Reject probes on
    // the back side of the local surface plane so thin walls and concave seams
    // cannot borrow bright irradiance from the opposite side.
    return smoothstep(-0.2, 0.2, signed_plane_distance);
}

fn svlm_sample_leaf_sh(
    leaf_index: u32,
    leaf: SVLMLeafBrick,
    sample_position: vec3<f32>,
    surface_position: vec3<f32>,
    surface_normal: vec3<f32>
) -> SVLMFineSample {
    let available_probe_count =
        arrayLength(&irradiance_probes) /
        SVLM_SH_WORDS_PER_PROBE;
    if (leaf.probe_base >= available_probe_count) {
        return SVLMFineSample(sh_l1_rgb_zero(), 0.0);
    }
    if (
        svlm_params.tile_streaming_enabled <= 0.5 &&
        available_probe_count - leaf.probe_base <
        SVLM_PROBES_PER_BRICK
    ) {
        return SVLMFineSample(sh_l1_rgb_zero(), 0.0);
    }

    let origin = vec3<f32>(leaf.origin_x, leaf.origin_y, leaf.origin_z);
    let size = max(leaf.size, 0.0001);
    let probe_spacing = size / 4.0;
    let probe_grid_position = clamp(
        (sample_position - origin) * (4.0 / size) - vec3<f32>(0.5),
        vec3<f32>(0.0),
        vec3<f32>(3.0)
    );
    let probe_base_coord = min(
        vec3<u32>(floor(probe_grid_position)),
        vec3<u32>(2u)
    );
    let probe_fraction = probe_grid_position - vec3<f32>(probe_base_coord);
    var result = sh_l1_rgb_zero();
    var weight_sum = 0.0;
    var validity_sum = 0.0;

    for (
        var corner = 0u;
        corner < 8u;
        corner = corner + 1u
    ) {
        let offset = vec3<u32>(
            corner & 1u,
            (corner >> 1u) & 1u,
            (corner >> 2u) & 1u
        );
        let coord = probe_base_coord + offset;
        let local_probe_index = coord.x + coord.y * 4u + coord.z * 16u;
        let probe_index = svlm_leaf_probe_index(leaf_index, leaf, local_probe_index);
        if (
            probe_index == INVALID_IDX ||
            probe_index >= available_probe_count ||
            (svlm_params.tile_streaming_enabled <= 0.5 && !svlm_probe_is_valid(probe_index))
        ) {
            continue;
        }
        let weight_axis = select(
            vec3<f32>(1.0) - probe_fraction,
            probe_fraction,
            offset == vec3<u32>(1u)
        );
        let trilinear_weight =
            weight_axis.x *
            weight_axis.y *
            weight_axis.z;
        let probe_position = svlm_probe_position(
            leaf,
            coord.x + coord.y * 4u + coord.z * 16u
        );
        let surface_weight = svlm_probe_surface_weight(
            probe_position,
            surface_position,
            surface_normal,
            probe_spacing
        );
        let weight = trilinear_weight * surface_weight;
        result = svlm_sh_multiply_add(
            result,
            svlm_read_probe_sh(probe_index),
            weight
        );
        weight_sum += weight;
        validity_sum += weight;
    }

    // Invalid in-geometry probes can leave a local 2x2x2 cell with only a tiny
    // surviving weight, which amplifies one noisy probe into a dark splotch.
    // Recover from nearby probes in the same brick only in that exceptional
    // case. The compact Gaussian radius keeps the normal path at eight reads
    // and avoids crossing an adaptive brick boundary or a surface plane.
    if (validity_sum < 0.5) {
        var recovery = sh_l1_rgb_zero();
        var recovery_weight_sum = 0.0;
        for (
            var local_probe_index = 0u;
            local_probe_index < SVLM_PROBES_PER_BRICK;
            local_probe_index = local_probe_index + 1u
        ) {
            let probe_index = svlm_leaf_probe_index(leaf_index, leaf, local_probe_index);
            if (
                probe_index == INVALID_IDX ||
                probe_index >= available_probe_count ||
                (svlm_params.tile_streaming_enabled <= 0.5 && !svlm_probe_is_valid(probe_index))
            ) {
                continue;
            }
            let probe_position = svlm_probe_position(
                leaf,
                local_probe_index
            );
            let probe_delta =
                (probe_position - sample_position) /
                max(probe_spacing, 0.0001);
            let distance_squared = dot(probe_delta, probe_delta);
            if (distance_squared > 6.25) {
                continue;
            }
            let surface_weight = svlm_probe_surface_weight(
                probe_position,
                surface_position,
                surface_normal,
                probe_spacing
            );
            let weight = exp2(-distance_squared) * surface_weight;
            if (weight <= 1e-6) {
                continue;
            }
            recovery = svlm_sh_multiply_add(
                recovery,
                svlm_read_probe_sh(probe_index),
                weight
            );
            recovery_weight_sum += weight;
        }
        if (recovery_weight_sum > 1e-6) {
            return SVLMFineSample(
                sh_l1_rgb_multiply_scalar(
                    recovery,
                    1.0 / recovery_weight_sum
                ),
                1.0
            );
        }
    }

    if (weight_sum <= 1e-6) {
        return SVLMFineSample(sh_l1_rgb_zero(), 0.0);
    }
    return SVLMFineSample(
        sh_l1_rgb_multiply_scalar(result, 1.0 / weight_sum),
        saturate(validity_sum)
    );
}

// Returns (coordinate just across the nearest boundary, neighbor weight).
// Each leaf contributes fully outside the transition band and reaches a 50/50
// blend exactly at a shared boundary. Repeating this independently on XYZ
// produces continuous face, edge, and corner transitions.
fn svlm_leaf_boundary_blend_axis(
    position: f32,
    leaf_min: f32,
    leaf_max: f32,
    blend_width: f32,
    epsilon: f32
) -> vec2<f32> {
    let distance_to_low = max(position - leaf_min, 0.0);
    let distance_to_high = max(leaf_max - position, 0.0);
    if (
        distance_to_low <= distance_to_high &&
        distance_to_low < blend_width
    ) {
        let weight = 0.5 * (
            1.0 - clamp(distance_to_low / blend_width, 0.0, 1.0)
        );
        return vec2<f32>(leaf_min - epsilon, weight);
    }
    if (distance_to_high < blend_width) {
        let weight = 0.5 * (
            1.0 - clamp(distance_to_high / blend_width, 0.0, 1.0)
        );
        return vec2<f32>(leaf_max + epsilon, weight);
    }
    return vec2<f32>(position, 0.0);
}

fn svlm_sample_blended_sh(
    base_leaf_index: u32,
    sample_position: vec3<f32>,
    surface_position: vec3<f32>,
    surface_normal: vec3<f32>
) -> SVLMFineSample {
    let base_leaf = leaf_bricks[base_leaf_index];
    let leaf_min = vec3<f32>(
        base_leaf.origin_x,
        base_leaf.origin_y,
        base_leaf.origin_z
    );
    let leaf_size = max(base_leaf.size, 0.0001);
    let leaf_max = leaf_min + vec3<f32>(leaf_size);
    let blend_width = max(leaf_size / 4.0, 0.0001);
    let epsilon = max(leaf_size * 0.0001, 0.0001);
    let blend_x = svlm_leaf_boundary_blend_axis(
        sample_position.x,
        leaf_min.x,
        leaf_max.x,
        blend_width,
        epsilon
    );
    let blend_y = svlm_leaf_boundary_blend_axis(
        sample_position.y,
        leaf_min.y,
        leaf_max.y,
        blend_width,
        epsilon
    );
    let blend_z = svlm_leaf_boundary_blend_axis(
        sample_position.z,
        leaf_min.z,
        leaf_max.z,
        blend_width,
        epsilon
    );
    let neighbor_weight = vec3<f32>(blend_x.y, blend_y.y, blend_z.y);
    if (all(neighbor_weight <= vec3<f32>(1e-6))) {
        return svlm_sample_leaf_sh(
            base_leaf_index,
            base_leaf,
            sample_position,
            surface_position,
            surface_normal
        );
    }
    let neighbor_position = vec3<f32>(blend_x.x, blend_y.x, blend_z.x);
    var result = sh_l1_rgb_zero();
    var weight_sum = 0.0;
    var validity_sum = 0.0;

    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        let use_neighbor = vec3<bool>(
            (corner & 1u) != 0u,
            ((corner >> 1u) & 1u) != 0u,
            ((corner >> 2u) & 1u) != 0u
        );
        let weight_axis = select(
            vec3<f32>(1.0) - neighbor_weight,
            neighbor_weight,
            use_neighbor
        );
        var weight = weight_axis.x * weight_axis.y * weight_axis.z;
        if (weight <= 1e-6) {
            continue;
        }

        let lookup_position = select(
            sample_position,
            neighbor_position,
            use_neighbor
        );
        var leaf_index = base_leaf_index;
        if (corner != 0u) {
            let adjacent_lookup = svlm_find_leaf_sample(lookup_position);
            let adjacent_leaf_index = adjacent_lookup.leaf_index;
            if (
                adjacent_leaf_index != INVALID_IDX &&
                adjacent_leaf_index < arrayLength(&leaf_bricks)
            ) {
                leaf_index = adjacent_leaf_index;
                weight *= svlm_tiled_leaf_fade(adjacent_lookup);
            }
        }
        if (weight <= 1e-6) {
            continue;
        }
        let leaf_sample = svlm_sample_leaf_sh(
            leaf_index,
            leaf_bricks[leaf_index],
            sample_position,
            surface_position,
            surface_normal
        );
        let sample_weight = weight * leaf_sample.validity;
        if (sample_weight <= 1e-6) {
            continue;
        }
        result = svlm_sh_multiply_add(
            result,
            leaf_sample.irradiance,
            sample_weight
        );
        weight_sum += sample_weight;
        validity_sum += sample_weight;
    }

    return SVLMFineSample(
        sh_l1_rgb_multiply_scalar(
            result,
            1.0 / max(weight_sum, 1e-6)
        ),
        saturate(validity_sum)
    );
}

fn svlm_evaluate_irradiance(
    irradiance_sh: SH_L1_RGB,
    normal: vec3<f32>
) -> vec3<f32> {
    let directional_irradiance = max(
        sh_l1_rgb_calculate_irradiance(irradiance_sh, normal),
        vec3<f32>(0.0)
    );
    // L1 can undershoot opposite a strong dominant direction. Retaining a
    // small L0 floor avoids turning valid low-frequency light into black.
    let l0_irradiance = max(
        irradiance_sh.c[0] * (PI * SH_BASIS_L0),
        vec3<f32>(0.0)
    );
    return max(directional_irradiance, l0_irradiance * 0.15);
}

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u && gid.y == 0u) {
        textureStore(output_black, vec2<i32>(0), vec4<f32>(0.0));
    }

    let sample_size = textureDimensions(output_diffuse);
    if (gid.x >= sample_size.x || gid.y >= sample_size.y) {
        return;
    }

    let sample_pixel = vec2<i32>(gid.xy);
    let full_size = textureDimensions(gbuffer_normal);
    let full_pixel = svlm_sample_output_to_full_res_coord(
        gid.xy,
        sample_size,
        full_size
    );
    let normal_data = textureLoad(gbuffer_normal, full_pixel, 0);
    if (dot(normal_data.xyz, normal_data.xyz) <= 1e-10) {
        textureStore(output_diffuse, sample_pixel, vec4<f32>(0.0));
        return;
    }

    let uv =
        (vec2<f32>(full_pixel) + vec2<f32>(0.5)) /
        vec2<f32>(full_size);
    let depth = textureLoad(depth_texture, full_pixel, 0).r;
    let position = reconstruct_world_position(
        uv,
        depth,
        u32(frame_info.view_index)
    );
    let normal = safe_normalize(normal_data.xyz);
    let surface_lookup = svlm_find_leaf_sample(position);
    let surface_leaf_index = surface_lookup.leaf_index;
    let surface_leaf_valid =
        surface_leaf_index != INVALID_IDX &&
        surface_leaf_index < arrayLength(&leaf_bricks);
    var coarse_sample = SVLMCoarseSample(sh_l1_rgb_zero(), 0.0);
    if (svlm_params.tile_streaming_enabled > 0.5) {
        coarse_sample = svlm_sample_coarse(position);
    }
    if (!surface_leaf_valid && coarse_sample.validity <= 0.5) {
        textureStore(output_diffuse, sample_pixel, vec4<f32>(0.0));
        return;
    }

    var fine_irradiance = vec3<f32>(0.0);
    var fine_validity = 0.0;
    var tile_fade = 1.0;
    if (surface_leaf_valid) {
        var sample_lookup = surface_lookup;
        var sample_leaf_index = surface_leaf_index;
        let leaf = leaf_bricks[sample_leaf_index];
        // Keep the lookup bias small and on the known air side. The surface
        // plane filter above does the heavy lifting without moving a crease or
        // thin frame halfway across a probe cell.
        let probe_spacing = max(leaf.size / 4.0, 0.0001);
        let sample_position =
            position + normal * max(probe_spacing * 0.25, 0.001);
        if (!svlm_leaf_contains(leaf, sample_position)) {
            let biased_lookup = svlm_find_leaf_sample(sample_position);
            let biased_leaf_index = biased_lookup.leaf_index;
            if (
                biased_leaf_index != INVALID_IDX &&
                biased_leaf_index < arrayLength(&leaf_bricks)
            ) {
                sample_lookup = biased_lookup;
                sample_leaf_index = biased_leaf_index;
            }
        }
        let fine_sample = svlm_sample_blended_sh(
            sample_leaf_index,
            sample_position,
            position,
            normal
        );
        fine_validity = fine_sample.validity;
        tile_fade = svlm_tiled_leaf_fade(sample_lookup);
        if (fine_validity > 1e-6) {
            fine_irradiance = svlm_evaluate_irradiance(
                fine_sample.irradiance,
                normal
            );
        }
    }
    if (fine_validity <= 1e-6 && coarse_sample.validity <= 0.5) {
        textureStore(output_diffuse, sample_pixel, vec4<f32>(0.0));
        return;
    }

    var irradiance = fine_irradiance * tile_fade;
    if (coarse_sample.validity > 0.5) {
        let coarse_irradiance = svlm_evaluate_irradiance(
            coarse_sample.irradiance,
            normal
        );
        let fine_weight = select(0.0, tile_fade, fine_validity > 1e-6);
        irradiance = mix(coarse_irradiance, fine_irradiance, fine_weight);
    }
    textureStore(
        output_diffuse,
        sample_pixel,
        vec4<f32>(irradiance, 1.0)
    );
}

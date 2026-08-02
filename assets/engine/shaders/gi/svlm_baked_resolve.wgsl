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
@group(1) @binding(7) var output_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(8) var output_black: texture_storage_2d<rgba16float, write>;

const SVLM_NODE_WORD_STRIDE = 7u;
const SVLM_TILED_LOOKUP_WORD_STRIDE = 8u;
const SVLM_TILED_LOOKUP_MAX_PROBES = 16u;
const SVLM_COARSE_LOOKUP_WORD_STRIDE = 10u;
const SVLM_COARSE_LOOKUP_MAX_PROBES = 16u;
const SVLM_LOOKUP_TOMBSTONE = 0xfffffffeu;

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

fn svlm_hash_tiled_lookup_key(
    tile_coord: vec3<i32>,
    level: u32,
    leaf_coord: vec3<u32>
) -> u32 {
    var hash = 0x811c9dc5u;
    hash = (hash ^ bitcast<u32>(tile_coord.x)) * 0x01000193u;
    hash = (hash ^ bitcast<u32>(tile_coord.y)) * 0x01000193u;
    hash = (hash ^ bitcast<u32>(tile_coord.z)) * 0x01000193u;
    hash = (hash ^ level) * 0x01000193u;
    hash = (hash ^ leaf_coord.x) * 0x01000193u;
    hash = (hash ^ leaf_coord.y) * 0x01000193u;
    hash = (hash ^ leaf_coord.z) * 0x01000193u;
    hash ^= hash >> 16u;
    hash *= 0x7feb352du;
    hash ^= hash >> 15u;
    hash *= 0x846ca68bu;
    return hash ^ (hash >> 16u);
}

fn svlm_lookup_tiled_leaf(
    tile_coord: vec3<i32>,
    level: u32,
    leaf_coord: vec3<u32>
) -> u32 {
    // CPU packing guarantees that every resident leaf is reachable within this
    // fixed probe budget. This bound is the critical difference from the old
    // per-pixel linear scan over all leaves intersecting a world tile.
    let entry_count =
        arrayLength(&svlm_lookup_data) /
        SVLM_TILED_LOOKUP_WORD_STRIDE;
    if (entry_count == 0u) {
        return INVALID_IDX;
    }

    let entry_mask = entry_count - 1u;
    var entry_index =
        svlm_hash_tiled_lookup_key(tile_coord, level, leaf_coord) &
        entry_mask;
    for (
        var probe = 0u;
        probe < SVLM_TILED_LOOKUP_MAX_PROBES;
        probe = probe + 1u
    ) {
        let base = entry_index * SVLM_TILED_LOOKUP_WORD_STRIDE;
        let leaf_index = svlm_lookup_data[base + 7u];
        if (leaf_index == INVALID_IDX) {
            return INVALID_IDX;
        }
        if (
            leaf_index != SVLM_LOOKUP_TOMBSTONE &&
            bitcast<i32>(svlm_lookup_data[base]) == tile_coord.x &&
            bitcast<i32>(svlm_lookup_data[base + 1u]) == tile_coord.y &&
            bitcast<i32>(svlm_lookup_data[base + 2u]) == tile_coord.z &&
            svlm_lookup_data[base + 3u] == level &&
            all(
                vec3<u32>(
                    svlm_lookup_data[base + 4u],
                    svlm_lookup_data[base + 5u],
                    svlm_lookup_data[base + 6u]
                ) == leaf_coord
            )
        ) {
            return leaf_index;
        }
        entry_index = (entry_index + 1u) & entry_mask;
    }
    return INVALID_IDX;
}

fn svlm_find_tiled_leaf(position: vec3<f32>) -> u32 {
    if (svlm_params.resident_tile_count <= 0.0) {
        return INVALID_IDX;
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
        return INVALID_IDX;
    }

    let tile_size = max(svlm_params.world_tile_size, 0.0001);
    let tile_coord = vec3<i32>(floor(position / tile_size));
    var level = u32(max(svlm_params.max_level, 0.0));
    loop {
        let leaf_size = svlm_brick_size(&svlm_params, level);
        let leaf_coord = vec3<u32>(floor(local_position / leaf_size));
        let leaf_index =
            svlm_lookup_tiled_leaf(tile_coord, level, leaf_coord);
        if (leaf_index != INVALID_IDX) {
            return leaf_index;
        }
        if (level == 0u) {
            break;
        }
        level -= 1u;
    }
    return INVALID_IDX;
}

fn svlm_find_leaf(position: vec3<f32>) -> u32 {
    if (svlm_params.tile_streaming_enabled > 0.5) {
        return svlm_find_tiled_leaf(position);
    }
    return svlm_find_monolithic_leaf(position);
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

struct SVLMCoarseSample {
    irradiance: SH_L1_RGB,
    validity: f32,
    lod: u32,
};

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
        return SVLMCoarseSample(sh_l1_rgb_zero(), 0.0, lod);
    }
    return SVLMCoarseSample(
        sh_l1_rgb_multiply_scalar(result, 1.0 / weight_sum),
        1.0,
        lod
    );
}

fn svlm_sample_coarse_from_lod(
    position: vec3<f32>,
    first_lod: u32,
    max_lod: u32
) -> SVLMCoarseSample {
    for (var lod = first_lod; lod <= max_lod; lod = lod + 1u) {
        let sample = svlm_sample_coarse_lod(position, lod);
        if (sample.validity > 0.5 || lod == max_lod) {
            return sample;
        }
    }
    return SVLMCoarseSample(sh_l1_rgb_zero(), 0.0, max_lod);
}

fn svlm_sample_coarse(position: vec3<f32>) -> SVLMCoarseSample {
    let min_lod = u32(max(svlm_params.coarse_min_lod, 1.0));
    let max_lod = u32(max(svlm_params.coarse_max_lod, f32(min_lod)));
    let camera_position =
        view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let fine_extent = max(
        svlm_params.world_tile_size * svlm_params.streaming_radius,
        svlm_params.world_tile_size
    );
    let camera_distance = distance(position, camera_position);
    let distance_lod = u32(max(
        floor(log2(max(camera_distance / fine_extent, 1.0))),
        0.0
    ));
    let desired_lod = min(min_lod + distance_lod, max_lod);
    let fine_sample = svlm_sample_coarse_from_lod(
        position,
        desired_lod,
        max_lod
    );
    if (
        fine_sample.validity <= 0.5 ||
        fine_sample.lod != desired_lod ||
        desired_lod >= max_lod
    ) {
        return fine_sample;
    }

    let coarse_sample = svlm_sample_coarse_from_lod(
        position,
        desired_lod + 1u,
        max_lod
    );
    if (coarse_sample.validity <= 0.5) {
        return fine_sample;
    }

    // Every coarse shell doubles both its reach and its transition width. This
    // extends the fine-to-coarse crossfade across the full hierarchy instead
    // of snapping at the logarithmic distance boundaries after the first LOD.
    let cascade_scale = exp2(f32(distance_lod + 1u));
    let outer_distance = fine_extent * cascade_scale;
    let transition_distance =
        svlm_params.streaming_transition_tiles *
        svlm_params.world_tile_size *
        cascade_scale;
    let inner_distance = max(outer_distance - transition_distance, 0.0);
    let coarse_weight = smoothstep(
        inner_distance,
        max(outer_distance, inner_distance + 0.0001),
        camera_distance
    );
    return SVLMCoarseSample(
        sh_l1_rgb_lerp(
            fine_sample.irradiance,
            coarse_sample.irradiance,
            coarse_weight
        ),
        1.0,
        fine_sample.lod
    );
}

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

fn svlm_sample_leaf_sh(
    leaf: SVLMLeafBrick,
    position: vec3<f32>
) -> SH_L1_RGB {
    let available_probe_count =
        arrayLength(&irradiance_probes) /
        SVLM_SH_WORDS_PER_PROBE;
    if (leaf.probe_base >= available_probe_count) {
        return sh_l1_rgb_zero();
    }
    if (
        available_probe_count - leaf.probe_base <
        SVLM_PROBES_PER_BRICK
    ) {
        return sh_l1_rgb_zero();
    }

    let origin = vec3<f32>(leaf.origin_x, leaf.origin_y, leaf.origin_z);
    let size = max(leaf.size, 0.0001);
    let probe_grid_position = clamp(
        (position - origin) * (4.0 / size) - vec3<f32>(0.5),
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
        let probe_index =
            leaf.probe_base +
            coord.x +
            coord.y * 4u +
            coord.z * 16u;
        if (!svlm_probe_is_valid(probe_index)) {
            continue;
        }
        let weight_axis = select(
            vec3<f32>(1.0) - probe_fraction,
            probe_fraction,
            offset == vec3<u32>(1u)
        );
        let weight =
            weight_axis.x *
            weight_axis.y *
            weight_axis.z;
        result = svlm_sh_multiply_add(
            result,
            svlm_read_probe_sh(probe_index),
            weight
        );
        weight_sum += weight;
    }

    if (weight_sum <= 1e-6) {
        return sh_l1_rgb_zero();
    }
    return sh_l1_rgb_multiply_scalar(result, 1.0 / weight_sum);
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
    position: vec3<f32>
) -> SH_L1_RGB {
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
        position.x,
        leaf_min.x,
        leaf_max.x,
        blend_width,
        epsilon
    );
    let blend_y = svlm_leaf_boundary_blend_axis(
        position.y,
        leaf_min.y,
        leaf_max.y,
        blend_width,
        epsilon
    );
    let blend_z = svlm_leaf_boundary_blend_axis(
        position.z,
        leaf_min.z,
        leaf_max.z,
        blend_width,
        epsilon
    );
    let neighbor_weight = vec3<f32>(blend_x.y, blend_y.y, blend_z.y);
    if (all(neighbor_weight <= vec3<f32>(1e-6))) {
        return svlm_sample_leaf_sh(base_leaf, position);
    }
    let neighbor_position = vec3<f32>(blend_x.x, blend_y.x, blend_z.x);
    var result = sh_l1_rgb_zero();
    var weight_sum = 0.0;

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
        let weight = weight_axis.x * weight_axis.y * weight_axis.z;
        if (weight <= 1e-6) {
            continue;
        }

        let lookup_position = select(position, neighbor_position, use_neighbor);
        var leaf_index = base_leaf_index;
        if (corner != 0u) {
            let adjacent_leaf_index = svlm_find_leaf(lookup_position);
            if (
                adjacent_leaf_index != INVALID_IDX &&
                adjacent_leaf_index < arrayLength(&leaf_bricks)
            ) {
                leaf_index = adjacent_leaf_index;
            }
        }
        result = svlm_sh_multiply_add(
            result,
            svlm_sample_leaf_sh(leaf_bricks[leaf_index], position),
            weight
        );
        weight_sum += weight;
    }

    return sh_l1_rgb_multiply_scalar(
        result,
        1.0 / max(weight_sum, 1e-6)
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
    let leaf_index = svlm_find_leaf(position);
    let fine_valid =
        leaf_index != INVALID_IDX &&
        leaf_index < arrayLength(&leaf_bricks);
    var coarse_sample = SVLMCoarseSample(sh_l1_rgb_zero(), 0.0, 0u);
    if (svlm_params.tile_streaming_enabled > 0.5) {
        coarse_sample = svlm_sample_coarse(position);
    }
    if (!fine_valid && coarse_sample.validity <= 0.5) {
        textureStore(output_diffuse, sample_pixel, vec4<f32>(0.0));
        return;
    }

    var fine_irradiance = vec3<f32>(0.0);
    if (fine_valid) {
        var sample_leaf_index = leaf_index;
        let leaf = leaf_bricks[sample_leaf_index];
        // Resolve from the air side of the surface. Adaptive refinement places
        // the visible surface in a geometry-overlapping leaf, while the useful
        // probes are commonly in its neighboring empty-space leaf.
        let probe_spacing = max(leaf.size / 4.0, 0.0001);
        let sample_position =
            position + normal * max(probe_spacing * 0.5, 0.001);
        if (!svlm_leaf_contains(leaf, sample_position)) {
            let biased_leaf_index = svlm_find_leaf(sample_position);
            if (
                biased_leaf_index != INVALID_IDX &&
                biased_leaf_index < arrayLength(&leaf_bricks)
            ) {
                sample_leaf_index = biased_leaf_index;
            }
        }
        fine_irradiance = svlm_evaluate_irradiance(
            svlm_sample_blended_sh(sample_leaf_index, sample_position),
            normal
        );
    }

    var irradiance = fine_irradiance;
    if (coarse_sample.validity > 0.5) {
        let coarse_irradiance = svlm_evaluate_irradiance(
            coarse_sample.irradiance,
            normal
        );
        let camera_position =
            view_buffer[u32(frame_info.view_index)].view_position.xyz;
        let outer_distance =
            svlm_params.streaming_radius * svlm_params.world_tile_size;
        let transition_distance =
            svlm_params.streaming_transition_tiles *
            svlm_params.world_tile_size;
        let inner_distance = max(outer_distance - transition_distance, 0.0);
        let fine_weight = select(
            0.0,
            1.0 - smoothstep(
                inner_distance,
                max(outer_distance, inner_distance + 0.0001),
                distance(position, camera_position)
            ),
            fine_valid
        );
        irradiance = mix(coarse_irradiance, fine_irradiance, fine_weight);
    }
    textureStore(
        output_diffuse,
        sample_pixel,
        vec4<f32>(irradiance, 1.0)
    );
}

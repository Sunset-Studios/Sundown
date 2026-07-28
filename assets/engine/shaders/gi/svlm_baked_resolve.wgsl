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
@group(1) @binding(6) var output_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var output_black: texture_storage_2d<rgba16float, write>;

const SVLM_NODE_WORD_STRIDE = 7u;
const SVLM_TILE_DIRECTORY_WORD_STRIDE = 8u;

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

fn svlm_find_tiled_leaf(position: vec3<f32>) -> u32 {
    let tile_size = max(svlm_params.world_tile_size, 0.0001);
    let tile_coord = vec3<i32>(floor(position / tile_size));
    let available_tile_count =
        arrayLength(&svlm_lookup_data) /
        SVLM_TILE_DIRECTORY_WORD_STRIDE;
    let tile_count = min(
        u32(max(svlm_params.resident_tile_count, 0.0)),
        available_tile_count
    );

    for (
        var tile_index = 0u;
        tile_index < tile_count;
        tile_index = tile_index + 1u
    ) {
        let base = tile_index * SVLM_TILE_DIRECTORY_WORD_STRIDE;
        let entry_coord = vec3<i32>(
            bitcast<i32>(svlm_lookup_data[base]),
            bitcast<i32>(svlm_lookup_data[base + 1u]),
            bitcast<i32>(svlm_lookup_data[base + 2u])
        );
        if (any(entry_coord != tile_coord)) {
            continue;
        }

        let leaf_offset = svlm_lookup_data[base + 3u];
        let leaf_count = svlm_lookup_data[base + 4u];
        var best_leaf = INVALID_IDX;
        var best_size = 1e30;

        for (
            var local_leaf = 0u;
            local_leaf < leaf_count;
            local_leaf = local_leaf + 1u
        ) {
            let leaf_index = leaf_offset + local_leaf;
            if (leaf_index >= arrayLength(&leaf_bricks)) {
                break;
            }

            let leaf = leaf_bricks[leaf_index];
            let leaf_min = vec3<f32>(
                leaf.origin_x,
                leaf.origin_y,
                leaf.origin_z
            );
            let leaf_size = max(leaf.size, 0.0001);
            let leaf_max = leaf_min + vec3<f32>(leaf_size);
            let epsilon = max(leaf_size * 1e-5, 1e-5);
            if (
                all(position >= leaf_min - vec3<f32>(epsilon)) &&
                all(position <= leaf_max + vec3<f32>(epsilon)) &&
                leaf_size < best_size
            ) {
                best_leaf = leaf_index;
                best_size = leaf_size;
            }
        }
        return best_leaf;
    }

    return INVALID_IDX;
}

fn svlm_find_leaf(position: vec3<f32>) -> u32 {
    if (svlm_params.resident_tile_count > 0.0) {
        return svlm_find_tiled_leaf(position);
    }
    return svlm_find_monolithic_leaf(position);
}

fn svlm_read_probe_sh(probe_index: u32) -> SH_L1_RGB {
    let base = probe_index * SVLM_SH_WORDS_PER_PROBE;
    var packed: SH_L1_RGB_Packed;
    for (var i = 0u; i < SVLM_SH_WORDS_PER_PROBE; i = i + 1u) {
        packed.data[i] = irradiance_probes[base + i];
    }
    return sh_l1_rgb_unpack(packed);
}

fn svlm_sample_leaf_sh(
    leaf: SVLMLeafBrick,
    position: vec3<f32>
) -> SH_L1_RGB {
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

    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
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
        let weight_axis = select(
            vec3<f32>(1.0) - probe_fraction,
            probe_fraction,
            offset == vec3<u32>(1u)
        );
        let weight = weight_axis.x * weight_axis.y * weight_axis.z;

        if (
            probe_index * SVLM_SH_WORDS_PER_PROBE +
                (SVLM_SH_WORDS_PER_PROBE - 1u) <
                arrayLength(&irradiance_probes)
        ) {
            result = sh_l1_rgb_add(
                result,
                sh_l1_rgb_multiply_scalar(
                    svlm_read_probe_sh(probe_index),
                    weight
                )
            );
        }
    }

    return result;
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
        result = sh_l1_rgb_add(
            result,
            sh_l1_rgb_multiply_scalar(
                svlm_sample_leaf_sh(leaf_bricks[leaf_index], position),
                weight
            )
        );
        weight_sum += weight;
    }

    return sh_l1_rgb_multiply_scalar(
        result,
        1.0 / max(weight_sum, 1e-6)
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u && gid.y == 0u) {
        textureStore(output_black, vec2<i32>(0), vec4<f32>(0.0));
    }

    let output_size = textureDimensions(output_diffuse);
    if (gid.x >= output_size.x || gid.y >= output_size.y) {
        return;
    }

    let pixel = vec2<i32>(gid.xy);
    let normal_data = textureLoad(gbuffer_normal, pixel, 0);
    if (dot(normal_data.xyz, normal_data.xyz) <= 1e-10) {
        textureStore(output_diffuse, pixel, vec4<f32>(0.0));
        return;
    }

    let uv =
        (vec2<f32>(gid.xy) + vec2<f32>(0.5)) /
        vec2<f32>(output_size);
    let depth = textureLoad(depth_texture, pixel, 0).r;
    let position = reconstruct_world_position(
        uv,
        depth,
        u32(frame_info.view_index)
    );
    let normal = safe_normalize(normal_data.xyz);
    let leaf_index = svlm_find_leaf(position);
    if (leaf_index == INVALID_IDX || leaf_index >= arrayLength(&leaf_bricks)) {
        textureStore(output_diffuse, pixel, vec4<f32>(0.0));
        return;
    }

    var sample_leaf_index = leaf_index;
    var leaf = leaf_bricks[sample_leaf_index];
    // Resolve from the air side of the surface. Adaptive refinement places the
    // visible surface in a geometry-overlapping leaf, while the useful probes
    // are commonly in its neighboring empty-space leaf. The previous 1% bias
    // was clamped back into the original leaf and produced isolated patches.
    let probe_spacing = max(leaf.size / 4.0, 0.0001);
    let sample_position =
        position + normal * max(probe_spacing * 0.5, 0.001);
    let biased_leaf_index = svlm_find_leaf(sample_position);
    if (
        biased_leaf_index != INVALID_IDX &&
        biased_leaf_index < arrayLength(&leaf_bricks)
    ) {
        sample_leaf_index = biased_leaf_index;
        leaf = leaf_bricks[sample_leaf_index];
    }
    let interpolated_sh = svlm_sample_blended_sh(
        sample_leaf_index,
        sample_position
    );
    let directional_irradiance = max(
        sh_l1_rgb_calculate_irradiance(interpolated_sh, normal),
        vec3<f32>(0.0)
    );
    // L1 can undershoot below zero opposite a strong dominant direction. Keep
    // a small fraction of the non-directional band so ringing does not turn
    // valid low-frequency bounce light into completely black shadow pixels.
    let l0_irradiance = max(
        interpolated_sh.c[0] * (PI * SH_BASIS_L0),
        vec3<f32>(0.0)
    );
    let irradiance = max(directional_irradiance, l0_irradiance * 0.15);
    textureStore(output_diffuse, pixel, vec4<f32>(irradiance, 1.0));
}

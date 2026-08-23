diagnostic(off, subgroup_uniformity);

#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(3) var<storage, read_write> update_indices: array<u32>;
@group(1) @binding(4) var depth_texture: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var<storage, read_write> surface_cache_hashmap: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> bootstrap_indices: array<u32>;

const SURFACE_CACHE_FEEDBACK_SUBGROUP_GROUP_LIMIT: u32 = 4u;

fn append_regular_patch(patch_index: u32) {
    let update_index = atomicAdd(&counters.update_patch_count, 1u);
    if (update_index < arrayLength(&update_indices)) {
        update_indices[update_index] = patch_index;
    }
}

fn surface_cache_patch_is_due(patch_index: u32) -> bool {
    let surface_patch = surface_cache[patch_index];
    let history_is_mature =
        surface_patch.history.x >= surface_cache_params.max_history_samples &&
        surface_patch.metadata.w >= surface_cache_params.history_footprint_end_samples;
    if (!history_is_mature) {
        return true;
    }

    // Once both radiance and footprint history are saturated, distribute
    // maintenance updates across frames. Invalidation lowers radiance maturity
    // before this test, so unfinished refresh work remains eligible.
    let update_period = max(
        u32(surface_cache_params.mature_patch_update_period),
        1u
    );
    return update_period == 1u ||
        hash(patch_index) % update_period ==
            u32(surface_cache_params.frame_index) % update_period;
}

fn append_active_patch(patch_index: u32, bootstrap: bool) {
    atomicAdd(&counters.active_patch_count, 1u);

    let bootstrap_capacity = min(
        u32(surface_cache_params.bootstrap_patch_capacity),
        arrayLength(&bootstrap_indices)
    );
    if (bootstrap && bootstrap_capacity > 0u) {
        let bootstrap_index = atomicAdd(&counters.bootstrap_patch_count, 1u);
        if (bootstrap_index < bootstrap_capacity) {
            bootstrap_indices[bootstrap_index] = patch_index;
            return;
        }
    }
    // Capacity normally covers the complete cache. If a caller deliberately
    // limits it, overflow still receives the regular batch instead of going black.
    if (bootstrap || surface_cache_patch_is_due(patch_index)) {
        append_regular_patch(patch_index);
    }
}

fn initialize_patch(
    patch_index: u32,
    position: vec3<f32>,
    normal: vec3<f32>,
    grid_key: vec4<i32>,
    cell_exponent: i32
) {
    surface_cache[patch_index].position_frame = vec4<f32>(
        position,
        surface_cache_params.frame_index
    );
    surface_cache[patch_index].normal_cell_exponent = vec4<f32>(
        normal,
        f32(cell_exponent)
    );
    surface_cache[patch_index].grid_key = grid_key;
    surface_cache[patch_index].metadata = vec4<f32>(
        0.0,
        0.0,
        surface_cache_params.frame_index,
        0.0
    );
    surface_cache[patch_index].history = vec4<f32>(0.0);
}

fn feedback_surface_descriptor(
    position: vec3<f32>,
    normal: vec3<f32>,
    quantized_position: vec3<i32>,
    grid_key: vec4<i32>,
    frame: u32,
    cell_exponent: i32,
    directional_bin: u32,
    capacity: u32,
    search_count: u32,
    lifetime: u32
) -> f32 {
    let key = surface_cache_hash_key(
        quantized_position,
        directional_bin,
        cell_exponent
    );
    let result = hashmap_find_or_claim(
        &surface_cache_hashmap,
        key,
        capacity,
        search_count,
        frame,
        lifetime
    );

    if (result.status == HASHMAP_RESULT_CLAIMED) {
        initialize_patch(
            result.index,
            position,
            normal,
            grid_key,
            cell_exponent
        );
        append_active_patch(result.index, true);
        return 0.0;
    } else if (result.status == HASHMAP_RESULT_FOUND) {
        let metadata = surface_cache[result.index].metadata;
        let sample_count = metadata.w;
        var history = surface_cache[result.index].history;

        surface_cache[result.index].metadata = vec4<f32>(
            metadata.x,
            sample_count,
            surface_cache_params.frame_index,
            metadata.w
        );
        surface_cache[result.index].position_frame = vec4<f32>(
            position,
            surface_cache_params.frame_index
        );
        surface_cache[result.index].normal_cell_exponent = vec4<f32>(
            normal,
            f32(cell_exponent)
        );
        let requires_bootstrap = history.x <= 0.0 && metadata.w <= 0.0;
        append_active_patch(result.index, requires_bootstrap);
        return sample_count;
    } else if (result.status == HASHMAP_RESULT_ALREADY_UPDATED) {
        return surface_cache[result.index].metadata.w;
    }
    atomicAdd(&counters.feedback_miss_count, 1u);
    return 0.0;
}

fn feedback_surface_level_deduplicated(
    position: vec3<f32>,
    normal: vec3<f32>,
    descriptor_bias: vec3<f32>,
    frame: u32,
    cell_exponent: i32,
    directional_bin: u32,
    capacity: u32,
    search_count: u32,
    lifetime: u32,
    descriptor_valid: bool,
    subgroup_lane: u32,
    subgroup_size: u32
) -> f32 {
    var quantized_position = vec3<i32>(0);
    var grid_key = vec4<i32>(0);
    if (descriptor_valid) {
        let cell_size = surface_cache_cell_size(cell_exponent);
        let descriptor_position =
            position + descriptor_bias * cell_size;
        quantized_position = vec3<i32>(floor(
            descriptor_position / cell_size
        ));
        grid_key = surface_cache_make_grid_key(
            quantized_position,
            directional_bin,
            cell_exponent
        );
    }

    // Coherent screen-space pixels usually share a cache descriptor. Electing
    // one lane per exact key removes redundant global probes and atomic traffic.
    var pending = descriptor_valid;
    var sample_count = 0.0;
    for (
        var group_index = 0u;
        group_index < SURFACE_CACHE_FEEDBACK_SUBGROUP_GROUP_LIMIT;
        group_index = group_index + 1u
    ) {
        if (!subgroupAny(pending)) {
            break;
        }

        let leader_lane = subgroupMin(select(
            subgroup_size,
            subgroup_lane,
            pending
        ));
        let leader_grid_key = vec4<i32>(
            bitcast<i32>(subgroupShuffle(
                bitcast<u32>(grid_key.x),
                leader_lane
            )),
            bitcast<i32>(subgroupShuffle(
                bitcast<u32>(grid_key.y),
                leader_lane
            )),
            bitcast<i32>(subgroupShuffle(
                bitcast<u32>(grid_key.z),
                leader_lane
            )),
            bitcast<i32>(subgroupShuffle(
                bitcast<u32>(grid_key.w),
                leader_lane
            ))
        );
        let descriptor_matches = pending && all(grid_key == leader_grid_key);

        var leader_sample_count = 0.0;
        if (subgroup_lane == leader_lane) {
            leader_sample_count = feedback_surface_descriptor(
                position,
                normal,
                quantized_position,
                grid_key,
                frame,
                cell_exponent,
                directional_bin,
                capacity,
                search_count,
                lifetime
            );
        }
        let shared_sample_count = subgroupShuffle(
            leader_sample_count,
            leader_lane
        );
        sample_count = select(
            sample_count,
            shared_sample_count,
            descriptor_matches
        );
        pending = pending && !descriptor_matches;
    }

    // Highly discontinuous subgroups retain the original per-pixel behavior
    // after the coherent groups have been removed. The fixed bound prevents
    // pathological geometry from turning deduplication into a long shuffle loop.
    if (pending) {
        sample_count = feedback_surface_descriptor(
            position,
            normal,
            quantized_position,
            grid_key,
            frame,
            cell_exponent,
            directional_bin,
            capacity,
            search_count,
            lifetime
        );
    }
    return sample_count;
}

@compute @workgroup_size(8, 8, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_lane: u32,
    @builtin(subgroup_size) subgroup_size: u32
) {
    let pixel_coord = gid.xy;
    let full_resolution = surface_cache_full_resolution(surface_cache_params);
    let frame = u32(surface_cache_params.frame_index);
    let capacity = max(u32(surface_cache_params.total_patch_count), 1u);
    let search_count = surface_cache_hash_search_count(surface_cache_params);
    let lifetime = max(u32(surface_cache_params.cache_entry_lifetime), 1u);
    var pixel_valid =
        pixel_coord.x < full_resolution.x &&
        pixel_coord.y < full_resolution.y;
    var position = vec3<f32>(0.0);
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var descriptor_bias = vec3<f32>(0.0);
    var directional_bin = 0u;
    var base_exponent_value = f32(SURFACE_CACHE_MIN_CELL_EXPONENT);
    var base_fine_exponent = SURFACE_CACHE_MIN_CELL_EXPONENT;
    var base_coarse_exponent = SURFACE_CACHE_MIN_CELL_EXPONENT;

    if (pixel_valid) {
        let normal_data = textureLoad(
            gbuffer_normal,
            vec2<i32>(pixel_coord),
            0
        );
        pixel_valid = dot(normal_data.xyz, normal_data.xyz) > 1e-8;
        if (pixel_valid) {
            normal = safe_normalize(normal_data.xyz);
            let descriptor_normal = safe_normalize(normal);
            descriptor_bias = surface_cache_descriptor_offset_normalized(
                descriptor_normal,
                1.0,
                surface_cache_params
            );
            directional_bin = surface_cache_directional_bin_normalized(
                descriptor_normal
            );
            position = reconstruct_world_position(
                coord_to_uv(vec2<i32>(pixel_coord), full_resolution),
                textureLoad(depth_texture, vec2<i32>(pixel_coord), 0).r,
                u32(frame_info.view_index)
            );
            base_exponent_value = surface_cache_cell_exponent_value(
                position,
                surface_cache_params
            );
            base_fine_exponent = i32(floor(base_exponent_value));
            base_coarse_exponent = min(
                base_fine_exponent + 1,
                SURFACE_CACHE_MAX_CELL_EXPONENT
            );
        }
    }

    // Always request the native footprint so it can accumulate history. While
    // either native level is new or underconverged, also request a temporary
    // coarser footprint used by lookup to hide its initial gathering noise.
    let fine_history = feedback_surface_level_deduplicated(
        position,
        normal,
        descriptor_bias,
        frame,
        base_fine_exponent,
        directional_bin,
        capacity,
        search_count,
        lifetime,
        pixel_valid,
        subgroup_lane,
        subgroup_size
    );
    var cell_history = fine_history;
    let base_coarse_valid = pixel_valid &&
        base_coarse_exponent != base_fine_exponent;
    let coarse_history = feedback_surface_level_deduplicated(
        position,
        normal,
        descriptor_bias,
        frame,
        base_coarse_exponent,
        directional_bin,
        capacity,
        search_count,
        lifetime,
        base_coarse_valid,
        subgroup_lane,
        subgroup_size
    );
    if (base_coarse_valid) {
        cell_history = min(cell_history, coarse_history);
    }

    let history_exponent_value = clamp(
        base_exponent_value + log2(surface_cache_history_footprint_scale(
            cell_history,
            surface_cache_params
        )),
        f32(SURFACE_CACHE_MIN_CELL_EXPONENT),
        f32(SURFACE_CACHE_MAX_CELL_EXPONENT)
    );
    let history_fine_exponent = i32(floor(history_exponent_value));
    let history_coarse_exponent = min(
        history_fine_exponent + 1,
        SURFACE_CACHE_MAX_CELL_EXPONENT
    );
    let history_fine_valid = pixel_valid &&
        history_fine_exponent != base_fine_exponent &&
        history_fine_exponent != base_coarse_exponent;
    feedback_surface_level_deduplicated(
        position,
        normal,
        descriptor_bias,
        frame,
        history_fine_exponent,
        directional_bin,
        capacity,
        search_count,
        lifetime,
        history_fine_valid,
        subgroup_lane,
        subgroup_size
    );

    let history_coarse_valid = pixel_valid &&
        history_coarse_exponent != history_fine_exponent &&
        history_coarse_exponent != base_fine_exponent &&
        history_coarse_exponent != base_coarse_exponent;
    feedback_surface_level_deduplicated(
        position,
        normal,
        descriptor_bias,
        frame,
        history_coarse_exponent,
        directional_bin,
        capacity,
        search_count,
        lifetime,
        history_coarse_valid,
        subgroup_lane,
        subgroup_size
    );
}

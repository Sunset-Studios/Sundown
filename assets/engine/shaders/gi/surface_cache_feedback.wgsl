#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

// Full-resolution visibility feedback performs the article's bounded
// find-or-insert operation. The jittered position selects a neighboring cache
// cell, while the unjittered surface remains the ray origin stored as payload.
@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read_write> surface_cache_sh_filtered: array<u32>;
@group(1) @binding(4) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(5) var<storage, read_write> active_indices: array<u32>;
@group(1) @binding(6) var<storage, read_write> update_indices: array<u32>;
@group(1) @binding(7) var depth_texture: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(9) var<storage, read_write> surface_cache_hashmap: array<atomic<u32>>;

// Keep spatial residency independent from tracing frequency. Young and noisy
// patches converge continuously; mature patches rotate through bounded update
// phases while an age guard prevents intermittently visible entries starving.
fn patch_requires_ray_update(patch_index: u32) -> bool {
    let surface_patch = surface_cache[patch_index];
    let minimum_samples = max(surface_cache_params.stable_update_min_samples, 0.0);
    if (surface_patch.history.x < minimum_samples) {
        return true;
    }

    let interval = max(u32(surface_cache_params.stable_update_interval), 1u);
    if (interval <= 1u) {
        return true;
    }

    let variance = max(surface_patch.history.w - surface_patch.history.z * surface_patch.history.z, 0.0);
    let normalized_variance = variance / max(surface_patch.history.z * surface_patch.history.z, 0.01);
    let variance_threshold = max(
        surface_cache_params.stable_update_variance_threshold,
        0.0
    );
    if (variance_threshold > 0.0 && normalized_variance >= variance_threshold) {
        return true;
    }

    let frame = u32(surface_cache_params.frame_index);
    let last_update_frame = min(u32(surface_patch.metadata.x), frame);
    let update_age = frame - last_update_frame;
    let update_phase = hash(patch_index ^ 0x85ebca6bu) % interval;
    return frame % interval == update_phase || update_age >= interval * 2u;
}

fn append_active_patch(patch_index: u32) {
    let active_index = atomicAdd(&counters.active_patch_count, 1u);
    if (active_index < u32(surface_cache_params.total_patch_count)) {
        active_indices[active_index] = patch_index;
    }

    if (patch_requires_ray_update(patch_index)) {
        let update_index = atomicAdd(&counters.update_patch_count, 1u);
        if (update_index < u32(surface_cache_params.total_patch_count)) {
            update_indices[update_index] = patch_index;
        }
    }
}

fn initialize_patch(
    patch_index: u32,
    position: vec3<f32>,
    normal: vec3<f32>,
    quantized_position: vec3<i32>,
    quantized_normal: vec3<i32>,
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
    surface_cache[patch_index].grid_key = surface_cache_make_grid_key(
        quantized_position,
        quantized_normal,
        cell_exponent
    );
    surface_cache[patch_index].metadata = vec4<f32>(0.0);
    surface_cache[patch_index].history = vec4<f32>(0.0);
    surface_cache_sh_patch_write(&surface_cache_sh, patch_index, sh_l1_rgb_zero());
    surface_cache_sh_patch_write(&surface_cache_sh_filtered, patch_index, sh_l1_rgb_zero());
}

fn feedback_surface_level(
    position: vec3<f32>,
    normal: vec3<f32>,
    frame: u32,
    cell_exponent: i32
) {
    let quantized_position = surface_cache_quantize_position(
        position,
        normal,
        cell_exponent,
        surface_cache_params
    );
    let quantized_normal = surface_cache_quantize_normal(normal);
    let key = surface_cache_hash_key(
        quantized_position,
        quantized_normal,
        cell_exponent
    );
    let capacity = max(u32(surface_cache_params.total_patch_count), 1u);
    let search_count = surface_cache_hash_search_count(surface_cache_params);
    let lifetime = max(u32(surface_cache_params.cache_entry_lifetime), 1u);
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
            quantized_position,
            quantized_normal,
            cell_exponent
        );
        append_active_patch(result.index);
    } else if (result.status == HASHMAP_RESULT_FOUND) {
        let previous_normal = safe_normalize(
            surface_cache[result.index].normal_cell_exponent.xyz
        );
        if (dot(previous_normal, normal) < 0.999999) {
            let raw_sh = surface_cache_rotate_sh_between_hemispheres(
                surface_cache_sh_patch_read(&surface_cache_sh, result.index),
                previous_normal,
                normal
            );
            let filtered_sh = surface_cache_rotate_sh_between_hemispheres(
                surface_cache_sh_patch_read(
                    &surface_cache_sh_filtered,
                    result.index
                ),
                previous_normal,
                normal
            );
            surface_cache_sh_patch_write(
                &surface_cache_sh,
                result.index,
                raw_sh
            );
            surface_cache_sh_patch_write(
                &surface_cache_sh_filtered,
                result.index,
                filtered_sh
            );
        }
        surface_cache[result.index].position_frame = vec4<f32>(
            position,
            surface_cache_params.frame_index
        );
        surface_cache[result.index].normal_cell_exponent.x = normal.x;
        surface_cache[result.index].normal_cell_exponent.y = normal.y;
        surface_cache[result.index].normal_cell_exponent.z = normal.z;

        append_active_patch(result.index);
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_coord = gid.xy;
    let full_resolution = surface_cache_full_resolution(surface_cache_params);
    if (pixel_coord.x >= full_resolution.x || pixel_coord.y >= full_resolution.y) {
        return;
    }

    let normal_data = textureLoad(gbuffer_normal, vec2<i32>(pixel_coord), 0);
    if (dot(normal_data.xyz, normal_data.xyz) <= 1e-8) {
        return;
    }

    let frame = u32(surface_cache_params.frame_index);
    let normal = safe_normalize(normal_data.xyz);
    let position = reconstruct_world_position(
        coord_to_uv(vec2<i32>(pixel_coord), full_resolution),
        textureLoad(depth_texture, vec2<i32>(pixel_coord), 0).r,
        u32(frame_info.view_index)
    );

    let levels = surface_cache_cell_levels(position, surface_cache_params);
    feedback_surface_level(
        position,
        normal,
        frame,
        levels.fine_exponent
    );
    if (levels.coarse_exponent != levels.fine_exponent) {
        feedback_surface_level(
            position,
            normal,
            frame,
            levels.coarse_exponent
        );
    }
}

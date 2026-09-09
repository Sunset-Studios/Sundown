#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(3) var<storage, read_write> update_indices: array<u32>;
@group(1) @binding(4) var depth_texture: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var<storage, read_write> surface_cache_hashmap: array<atomic<u32>>;

fn append_patch(patch_index: u32) {
    let update_index = atomicAdd(&counters.update_patch_count, 1u);
    if (update_index < arrayLength(&update_indices)) {
        update_indices[update_index] = patch_index;
    }
}

fn append_active_patch(patch_index: u32) {
    atomicAdd(&counters.active_patch_count, 1u);
    append_patch(patch_index);
}

fn feedback_surface_patch(
    position: vec3<f32>,
    normal: vec3<f32>,
    frame: u32,
    cell_exponent: i32,
    directional_bin: u32,
    capacity: u32,
    search_count: u32,
    lifetime: u32
) {
    let cell_size = surface_cache_cell_size(cell_exponent);
    let quantized_position = vec3<i32>(floor(
        position / cell_size
    ));
    let result = hashmap_find_or_claim(
        &surface_cache_hashmap,
        surface_cache_hash_key(
            quantized_position,
            directional_bin,
            cell_exponent
        ),
        capacity,
        search_count,
        frame,
        lifetime
    );

    if (result.status == HASHMAP_RESULT_CLAIMED) {
        surface_cache[result.index].position_frame = vec4<f32>(
            position,
            surface_cache_params.frame_index
        );
        surface_cache[result.index].normal_cell_exponent = vec4<f32>(
            normal,
            f32(cell_exponent)
        );
        surface_cache[result.index].grid_key = surface_cache_make_grid_key(
            quantized_position,
            directional_bin,
            cell_exponent
        );
        surface_cache[result.index].metadata = vec4<f32>(
            0.0,
            0.0,
            surface_cache_params.frame_index,
            0.0
        );
        surface_cache[result.index].history = vec4<f32>(0.0);
        append_active_patch(result.index);
        return;
    } else if (result.status == HASHMAP_RESULT_FOUND) {
        let metadata = surface_cache[result.index].metadata;
        let sample_count = metadata.w;
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
        append_active_patch(result.index);
        return;
    } else if (result.status == HASHMAP_RESULT_ALREADY_UPDATED) {
        return;
    }
    atomicAdd(&counters.feedback_miss_count, 1u);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = surface_cache_full_resolution(surface_cache_params);
    if (gid.x >= full_resolution.x || gid.y >= full_resolution.y) {
        return;
    } 

    let normal_data = textureLoad(
        gbuffer_normal,
        vec2<i32>(gid.xy),
        0
    );
    if (dot(normal_data.xyz, normal_data.xyz) <= 1e-8) {
        return;
    }

    let normal = safe_normalize(normal_data.xyz);
    let directional_bin = surface_cache_directional_bin(
        normal
    );
    let position = reconstruct_world_position(
        coord_to_uv(vec2<i32>(gid.xy), full_resolution),
        textureLoad(depth_texture, vec2<i32>(gid.xy), 0).r,
        u32(frame_info.view_index)
    );

    let search_count = surface_cache_hash_search_count(surface_cache_params);

    feedback_surface_patch(
        position,
        normal,
        u32(surface_cache_params.frame_index),
        surface_cache_cell_exponent(position, surface_cache_params),
        directional_bin,
        u32(surface_cache_params.total_patch_count),
        search_count,
        u32(surface_cache_params.cache_entry_lifetime)
    );
}

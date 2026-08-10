#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(3) var<storage, read_write> update_indices: array<u32>;
@group(1) @binding(4) var depth_texture: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var<storage, read_write> surface_cache_hashmap: array<atomic<u32>>;

fn append_active_patch(patch_index: u32) {
    atomicAdd(&counters.active_patch_count, 1u);

    let update_index = atomicAdd(&counters.update_patch_count, 1u);
    if (update_index < u32(surface_cache_params.total_patch_count)) {
        update_indices[update_index] = patch_index;
    }
}

fn initialize_patch(
    patch_index: u32,
    position: vec3<f32>,
    normal: vec3<f32>,
    quantized_position: vec3<i32>,
    directional_bin: u32,
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
        directional_bin,
        cell_exponent
    );
    surface_cache[patch_index].metadata = vec4<f32>(0.0);
    surface_cache[patch_index].history = vec4<f32>(0.0);
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
    let directional_bin = surface_cache_directional_bin(normal);
    let key = surface_cache_hash_key(
        quantized_position,
        directional_bin,
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
            directional_bin,
            cell_exponent
        );
        append_active_patch(result.index);
    } else if (result.status == HASHMAP_RESULT_FOUND) {
        surface_cache[result.index].position_frame = vec4<f32>(
            position,
            surface_cache_params.frame_index
        );
        surface_cache[result.index].normal_cell_exponent.x = normal.x;
        surface_cache[result.index].normal_cell_exponent.y = normal.y;
        surface_cache[result.index].normal_cell_exponent.z = normal.z;
        surface_cache[result.index].normal_cell_exponent.w = f32(cell_exponent);
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

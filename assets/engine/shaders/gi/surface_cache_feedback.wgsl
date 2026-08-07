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
@group(1) @binding(6) var depth_texture: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(8) var<storage, read_write> surface_cache_hashmap: array<atomic<u32>>;

fn append_active_patch(patch_index: u32) {
    let active_index = atomicAdd(&counters.active_patch_count, 1u);
    if (active_index < u32(surface_cache_params.total_patch_count)) {
        active_indices[active_index] = patch_index;
    }
}

fn initialize_patch(
    patch_index: u32,
    position: vec3<f32>,
    normal: vec3<f32>,
    quantized_position: vec3<i32>,
    quantized_normal: vec3<i32>,
    lod: u32
) {
    surface_cache[patch_index].position_frame = vec4<f32>(
        position,
        surface_cache_params.frame_index
    );
    surface_cache[patch_index].normal_lod = vec4<f32>(normal, f32(lod));
    surface_cache[patch_index].grid_key = surface_cache_make_grid_key(
        quantized_position,
        quantized_normal,
        lod
    );
    surface_cache[patch_index].metadata = vec4<f32>(0.0);
    surface_cache[patch_index].history = vec4<f32>(0.0);
    surface_cache_sh_patch_write(&surface_cache_sh, patch_index, sh_l1_rgb_zero());
    surface_cache_sh_patch_write(&surface_cache_sh_filtered, patch_index, sh_l1_rgb_zero());
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

    let lod = surface_cache_select_lod(position, surface_cache_params);
    let cell_size = surface_cache_lod_cell_size(lod, surface_cache_params);
    let lookup_position = surface_cache_jitter_lookup_position(
        position,
        normal,
        pixel_coord,
        frame,
        cell_size,
        surface_cache_params
    );
    let quantized_position = surface_cache_quantize_position(
        lookup_position,
        lod,
        surface_cache_params
    );
    let quantized_normal = surface_cache_quantize_normal(normal);
    let key = surface_cache_hash_key(
        quantized_position,
        quantized_normal,
        lod,
        surface_cache_params
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
            lod
        );
        hashmap_publish_claim(
            &surface_cache_hashmap,
            result.index,
            key.checksum,
            frame
        );
        append_active_patch(result.index);
    } else if (result.status == HASHMAP_RESULT_FOUND) {
        surface_cache[result.index].position_frame.w = surface_cache_params.frame_index;
        if (!surface_cache_sample_limit_reached(
            surface_cache[result.index],
            surface_cache_params
        )) {
            append_active_patch(result.index);
        }
    }
}

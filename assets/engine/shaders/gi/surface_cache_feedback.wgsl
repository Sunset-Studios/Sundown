#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

// Full-resolution visibility feedback admits world-space surface descriptors
// and produces a unique stream of patches to update this frame. Patch geometry
// is fixed for the lifetime of an entry; only last-seen metadata changes. That
// prevents sub-pixel camera motion from moving every ray origin in the cache.
@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read_write> surface_cache_sh_filtered: array<u32>;
@group(1) @binding(4) var<storage, read_write> counters: SurfaceCacheCounters;
@group(1) @binding(5) var<storage, read_write> active_indices: array<u32>;
@group(1) @binding(6) var depth_texture: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_normal: texture_2d<f32>;

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
    quantized_normal: vec2<i32>,
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

fn try_touch_patch(patch_index: u32, frame: u32) -> bool {
    let previous_frame = atomicLoad(&surface_cache[patch_index].update_frame);
    if (previous_frame == SURFACE_CACHE_UPDATE_LOCKED) {
        return false;
    }
    let touch = atomicCompareExchangeWeak(
        &surface_cache[patch_index].update_frame,
        previous_frame,
        frame
    );
    if (!touch.exchanged) {
        return false;
    }
    if (previous_frame != frame) {
        surface_cache[patch_index].position_frame.w = surface_cache_params.frame_index;
        append_active_patch(patch_index);
    }
    return true;
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
    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let normal = safe_normalize(normal_data.xyz);
    let position = reconstruct_world_position(
        coord_to_uv(vec2<i32>(pixel_coord), full_resolution),
        textureLoad(depth_texture, vec2<i32>(pixel_coord), 0).r,
        u32(frame_info.view_index)
    );

    let lod = surface_cache_select_lod(position, camera_position, surface_cache_params);
    let quantized_position = surface_cache_quantize_position(position, lod, surface_cache_params);
    let quantized_normal = surface_cache_quantize_normal(normal);
    let bucket_start = surface_cache_bucket_start(
        quantized_position,
        quantized_normal,
        lod,
        surface_cache_params
    );
    let fingerprint = surface_cache_hash_fingerprint(
        quantized_position,
        quantized_normal,
        lod
    );

    var empty_index = -1;
    var oldest_index = -1;
    var oldest_frame = frame;

    // Find an exact key first. Fingerprints accelerate rejection but never
    // define identity by themselves.
    for (var probe = 0u; probe < SURFACE_CACHE_BUCKET_SIZE; probe = probe + 1u) {
        let patch_index = bucket_start + probe;
        let patch_fingerprint = atomicLoad(&surface_cache[patch_index].fingerprint);
        if (patch_fingerprint == fingerprint && surface_cache_patch_descriptor_matches(
            surface_cache[patch_index].grid_key,
            quantized_position,
            quantized_normal,
            lod
        )) {
            try_touch_patch(patch_index, frame);
            return;
        }

        let patch_frame = atomicLoad(&surface_cache[patch_index].update_frame);
        // A descriptor is being published in this bucket. Deferring this
        // invocation avoids racing the non-atomic exact key and allocating a
        // duplicate. The next frame retries the bucket.
        if (patch_frame == SURFACE_CACHE_UPDATE_LOCKED) {
            return;
        }
        if (patch_fingerprint == SURFACE_CACHE_PATCH_EMPTY && patch_frame == 0u) {
            if (empty_index < 0) {
                empty_index = i32(patch_index);
            }
        } else if (patch_frame < oldest_frame) {
            oldest_frame = patch_frame;
            oldest_index = i32(patch_index);
        }
    }

    var claim_index = empty_index;
    var expected_frame = 0u;
    if (claim_index < 0) {
        // Bucket-local replacement gives a busy region a way to recover from
        // hash pressure without globally flushing useful radiance. Recently
        // visible entries are never eligible.
        let replacement_age = max(8u, u32(surface_cache_params.cache_entry_lifetime * 0.5));
        if (oldest_index < 0 || frame - oldest_frame <= replacement_age) {
            return;
        }
        claim_index = oldest_index;
        expected_frame = oldest_frame;
    }

    let patch_index = u32(claim_index);
    let claim = atomicCompareExchangeWeak(
        &surface_cache[patch_index].update_frame,
        expected_frame,
        SURFACE_CACHE_UPDATE_LOCKED
    );
    if (!claim.exchanged) {
        return;
    }

    // The update-frame lock remains held while descriptor and radiance data
    // are reset. Publish the fingerprint last, then release the slot.
    initialize_patch(
        patch_index,
        position,
        normal,
        quantized_position,
        quantized_normal,
        lod
    );
    atomicStore(&surface_cache[patch_index].fingerprint, fingerprint);
    atomicStore(&surface_cache[patch_index].update_frame, frame);
    append_active_patch(patch_index);
}

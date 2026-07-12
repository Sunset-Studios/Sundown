#include "common.wgsl"
#include "gi/scgi_common.wgsl"

// Full-resolution depth feedback is both cache admission and active-stream
// generation. Every visible descriptor either claims an empty bucket slot or
// refreshes its existing world-space patch. Exactly one invocation per patch
// appends that patch to active_indices for the current frame.
@group(1) @binding(0) var<uniform> scgi_params: SCGIParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatch>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read_write> counters: SCGICounters;
@group(1) @binding(4) var<storage, read_write> active_indices: array<u32>;
@group(1) @binding(5) var depth_texture: texture_2d<f32>;
@group(1) @binding(6) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_motion_emissive: texture_2d<f32>;

fn append_active_patch(patch_index: u32) {
    let active_index = atomicAdd(&counters.active_patch_count, 1u);
    if (active_index < u32(scgi_params.total_patch_count)) {
        active_indices[active_index] = patch_index;
    }
}

fn initialize_patch(
    patch_index: u32,
    pixel_coord: vec2<u32>,
    position: vec3<f32>,
    normal: vec3<f32>,
    lod: u32,
    reset: bool
) {
    let albedo = textureLoad(gbuffer_albedo, pixel_coord.xy, 0).rgb;
    let smra = textureLoad(gbuffer_smra, pixel_coord.xy, 0);
    let motion_emissive = textureLoad(gbuffer_motion_emissive, pixel_coord.xy, 0);
    surface_cache[patch_index].position_frame = vec4<f32>(position, scgi_params.frame_index);
    surface_cache[patch_index].normal_unused = vec4<f32>(normal, 0.0);
    surface_cache[patch_index].albedo_roughness = vec4<f32>(albedo, smra.g);
    surface_cache[patch_index].material_props = vec4<f32>(smra.b, smra.r, motion_emissive.w, f32(lod));
    if (reset) {
        surface_cache[patch_index].history = vec4<f32>(0.0);
        scgi_sh_patch_write(&surface_cache_sh, patch_index, sh_l1_rgb_zero());
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_coord = gid.xy;
    let full_resolution = scgi_full_resolution(scgi_params);
    if (pixel_coord.x >= full_resolution.x || pixel_coord.y >= full_resolution.y) {
        return;
    }

    let normal_data = textureLoad(gbuffer_normal, vec2<i32>(pixel_coord), 0);
    if (dot(normal_data.xyz, normal_data.xyz) <= 1e-8) {
        return;
    }

    let frame = u32(scgi_params.frame_index);

    let camera_position = view_buffer[u32(frame_info.view_index)].view_position.xyz;
    let normal = safe_normalize(normal_data.xyz);
    let position = reconstruct_world_position(
        coord_to_uv(vec2<i32>(pixel_coord), full_resolution),
        textureLoad(depth_texture, vec2<i32>(pixel_coord), 0).r,
        u32(frame_info.view_index)
    );

    let lod = scgi_select_lod(position, camera_position, scgi_params);
    let quantized_position = scgi_quantize_position(position, lod, scgi_params);
    let quantized_normal = scgi_quantize_normal(normal);

    let bucket_start = scgi_bucket_start(quantized_position, quantized_normal, lod, scgi_params);
    let fingerprint = scgi_hash_fingerprint(quantized_position, quantized_normal, lod);

    for (var probe = 0u; probe < SCGI_BUCKET_SIZE; probe = probe + 1u) {
        let patch_index = bucket_start + probe;
        let claim = atomicCompareExchangeWeak(
            &surface_cache[patch_index].fingerprint,
            SCGI_PATCH_EMPTY,
            fingerprint
        );

        if (claim.exchanged) {
            atomicStore(&surface_cache[patch_index].update_frame, frame);
            initialize_patch(
                patch_index,
                pixel_coord,
                position,
                normal,
                lod,
                true
            );
            append_active_patch(patch_index);
            return;
        }

        if (claim.old_value == fingerprint) {
            if (scgi_patch_descriptor_matches(
                surface_cache[patch_index].position_frame.xyz,
                surface_cache[patch_index].normal_unused.xyz,
                quantized_position,
                quantized_normal,
                lod,
                scgi_params
            )) {
                let previous_frame = atomicExchange(&surface_cache[patch_index].update_frame, frame);
                if (previous_frame != frame) {
                    atomicStore(&surface_cache[patch_index].update_frame, frame);
                    initialize_patch(
                        patch_index,
                        pixel_coord,
                        position,
                        normal,
                        lod,
                        false
                    );
                    append_active_patch(patch_index);
                }
                return;
            }
        }
    }
}

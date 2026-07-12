#include "common.wgsl"
#include "gi/scgi_common.wgsl"

@group(1) @binding(0) var<uniform> scgi_params: SCGIParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var scene_color: texture_2d<f32>;
@group(1) @binding(6) var output_debug: texture_storage_2d<rgba16float, write>;

#include "gi/scgi_cache_lookup.wgsl"

// Displays the single cache patch addressed by the current surface. Unlike
// scgi_resolve, this deliberately performs no neighboring-cell lookup or
// geometry/confidence weighting, exposing the raw spatial cache population.
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(output_debug);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let pixel_coord = vec2<i32>(gid.xy);
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    if (dot(normal_data.xyz, normal_data.xyz) <= 1e-8) {
        textureStore(output_debug, pixel_coord, textureLoad(scene_color, pixel_coord, 0));
        return;
    }

    let view_index = u32(frame_info.view_index);
    let normal = safe_normalize(normal_data.xyz);
    let position = reconstruct_world_position(
        coord_to_uv(pixel_coord, resolution),
        textureLoad(depth_texture, pixel_coord, 0).r,
        view_index
    );
    let camera_position = view_buffer[view_index].view_position.xyz;
    let lod = scgi_select_lod(position, camera_position, scgi_params);
    let descriptor_position = scgi_quantize_position(position, lod, scgi_params);
    let descriptor_normal = scgi_quantize_normal(normal);
    let patch_index = scgi_find_patch(descriptor_position, descriptor_normal, lod);

    var cached_radiance = vec3<f32>(0.0);
    if (patch_index >= 0) {
        cached_radiance = scgi_evaluate_local_sh_irradiance(
            scgi_sh_patch_read(&surface_cache_sh, u32(patch_index))
        ) * scgi_params.indirect_boost;
    }

    textureStore(
        output_debug,
        pixel_coord,
        vec4<f32>(safe_clamp_vec3_max(cached_radiance, SCGI_MAX_RADIANCE), 1.0)
    );
}

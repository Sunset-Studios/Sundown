#include "common.wgsl"
#include "gi/scgi_common.wgsl"

@group(1) @binding(0) var<uniform> scgi_params: SCGIParams;
@group(1) @binding(1) var<storage, read_write> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read_write> surface_cache_sh: array<u32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var out_direct: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var out_indirect_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var out_indirect_specular: texture_storage_2d<rgba16float, write>;

#include "gi/scgi_cache_lookup.wgsl"

fn store_zero(pixel_coord: vec2<i32>) {
    textureStore(out_direct, pixel_coord, vec4<f32>(0.0));
    textureStore(out_indirect_diffuse, pixel_coord, vec4<f32>(0.0));
    textureStore(out_indirect_specular, pixel_coord, vec4<f32>(0.0));
}

// Pure reconstruction pass. Noise and geometry-aware filtering have already
// been handled in surface-cache space; this pass only resolves three filtered
// patch samples into the full-resolution indirect-lighting texture.
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = scgi_full_resolution(scgi_params);
    if (gid.x >= full_resolution.x || gid.y >= full_resolution.y) {
        return;
    }

    let pixel_coord = vec2<i32>(gid.xy);
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    if (dot(normal_data.xyz, normal_data.xyz) <= 1e-8) {
        store_zero(pixel_coord);
        return;
    }

    let view_index = u32(frame_info.view_index);
    let camera_position = view_buffer[view_index].view_position.xyz;
    let normal = safe_normalize(normal_data.xyz);
    let position = reconstruct_world_position(
        coord_to_uv(pixel_coord, full_resolution),
        textureLoad(depth_texture, pixel_coord, 0).r,
        view_index
    );
    let cached_radiance = scgi_sample_surface_cache(position, normal, camera_position);

    textureStore(out_direct, pixel_coord, vec4<f32>(0.0));
    textureStore(
        out_indirect_diffuse,
        pixel_coord,
        vec4<f32>(safe_clamp_vec3_max(cached_radiance.xyz, SCGI_MAX_RADIANCE), cached_radiance.w)
    );
    textureStore(out_indirect_specular, pixel_coord, vec4<f32>(0.0));
}

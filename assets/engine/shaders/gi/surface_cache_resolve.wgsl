#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read> surface_cache_sh: array<u32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var out_indirect_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var out_black: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var<storage, read> surface_cache_hashmap: array<HashMapEntry>;

#include "gi/surface_cache_lookup.wgsl"

fn store_zero(pixel_coord: vec2<i32>) {
    textureStore(out_indirect_diffuse, pixel_coord, vec4<f32>(0.0));
}

fn surface_cache_resolve_clamp_radiance(radiance: vec3<f32>) -> vec3<f32> {
    let sanitized = safe_clamp_vec3(radiance);
    let radiance_luminance = dot(
        sanitized,
        vec3<f32>(0.2126, 0.7152, 0.0722)
    );
    // The generic select-based helper evaluates its division for every pixel.
    // Resolve radiance is almost always below the firefly ceiling, so branch
    // around that division and pay it only for the exceptional clamped path.
    if (radiance_luminance > SURFACE_CACHE_MAX_RADIANCE) {
        return sanitized * (
            SURFACE_CACHE_MAX_RADIANCE / radiance_luminance
        );
    }
    return sanitized;
}

// Pure reconstruction pass. Noise and geometry-aware filtering have already
// been handled in surface-cache space; this pass reconstructs geometry-aware
// filtered patch samples into the full-resolution indirect-lighting texture.
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = surface_cache_full_resolution(surface_cache_params);
    if (gid.x >= full_resolution.x || gid.y >= full_resolution.y) {
        return;
    }
    if (gid.x == 0u && gid.y == 0u) {
        textureStore(out_black, vec2<i32>(0), vec4<f32>(0.0));
    }

    let pixel_coord = vec2<i32>(gid.xy);
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0).xyz;
    if (dot(normal_data, normal_data) <= 1e-8) {
        store_zero(pixel_coord);
        return;
    }

    let view_index = u32(frame_info.view_index);
    // The validity check guarantees normalize follows safe_normalize's
    // nonzero path while avoiding its redundant length calculation.
    let normal = normalize(normal_data);
    let position = reconstruct_world_position(
        coord_to_uv(pixel_coord, full_resolution),
        textureLoad(depth_texture, pixel_coord, 0).r,
        view_index
    );
    let cached_radiance = surface_cache_sample_normalized(position, normal);

    textureStore(
        out_indirect_diffuse,
        pixel_coord,
        vec4<f32>(
            surface_cache_resolve_clamp_radiance(cached_radiance.xyz),
            cached_radiance.w
        )
    );
}

#include "common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "gi/surface_cache_lookup.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read> surface_cache_sh: array<u32>;
@group(1) @binding(3) var depth_texture: texture_2d<f32>;
@group(1) @binding(4) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(5) var out_indirect_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var<storage, read> surface_cache_hashmap: array<HashMapEntry>;
@group(1) @binding(7) var out_resolve_aux: texture_storage_2d<rgba16float, write>;

fn store_zero(pixel_coord: vec2<i32>) {
    textureStore(out_indirect_diffuse, pixel_coord, vec4<f32>(0.0));
    textureStore(out_resolve_aux, pixel_coord, vec4<f32>(0.0));
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

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_resolution = surface_cache_full_resolution(surface_cache_params);
    if (gid.x >= full_resolution.x || gid.y >= full_resolution.y) {
        return;
    }

    let pixel_coord = vec2<i32>(gid.xy);
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0).xyz;
    if (dot(normal_data, normal_data) <= 1e-8) {
        store_zero(pixel_coord);
        return;
    }

    let view_index = u32(frame_info.view_index);
    let normal = safe_normalize(normal_data);
    let position = reconstruct_world_position(
        coord_to_uv(pixel_coord, full_resolution),
        textureLoad(depth_texture, pixel_coord, 0).r,
        view_index
    );
    let presented = surface_cache_presentation_sample_normalized(
        position,
        normal
    );

    textureStore(
        out_indirect_diffuse,
        pixel_coord,
        vec4<f32>(
            surface_cache_resolve_clamp_radiance(presented.value.xyz),
            presented.value.w
        )
    );
    textureStore(
        out_resolve_aux,
        pixel_coord,
        vec4<f32>(
            presented.confidence,
            presented.standard_error,
            presented.fallback_weight,
            select(0.0, 1.0, presented.value.w > 0.0)
        )
    );
}

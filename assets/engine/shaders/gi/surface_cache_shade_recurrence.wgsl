#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "gi/surface_cache_lookup.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<storage, read> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(2) var<storage, read> surface_cache_sh: array<u32>;
@group(1) @binding(3) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(4) var<storage, read_write> radiance_info: array<SurfaceCacheRadianceInfo>;
@group(1) @binding(5) var<storage, read> surface_cache_hashmap: array<HashMapEntry>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= surface_cache_total_ray_count(counters)) {
        return;
    }
    let ray_data_index = gid.x;
    if (radiance_info[ray_data_index].shadow_direction.w == 0.0) {
        return;
    }
    let hit_position = radiance_info[ray_data_index].shadow_origin.xyz;
    let shading_normal = radiance_info[ray_data_index].shadow_direction.xyz;
    // Follow DDGI's diffuse-only recurrence. The cache lookup returns incident
    // irradiance here; material response is applied exactly once at this hit.
    // The 1 / (2 PI) factor intentionally matches DDGI's stable diffuse seed.
    let diffuse_response = radiance_info[ray_data_index].shadow_radiance.xyz;
    let albedo = radiance_info[ray_data_index].sample_radiance.xyz;
    let emissive = radiance_info[ray_data_index].shadow_origin.w;
    let recurrent_irradiance = surface_cache_sample_nearest_irradiance(
        hit_position,
        shading_normal
    ).xyz;
    var shaded_radiance = safe_clamp_vec3_max(
        recurrent_irradiance,
        SURFACE_CACHE_MAX_RADIANCE
    ) * diffuse_response;
    if (emissive > 0.0) {
        shaded_radiance += stabilize_surface_cache_emissive_radiance(
            emissive * albedo
        );
    }
    radiance_info[ray_data_index].sample_radiance = vec4<f32>(
        shaded_radiance,
        1.0
    );
}

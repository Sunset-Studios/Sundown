#include "gi/svlm_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"

// Shades traced probe rays with the same environment/material/light sources as
// DDGI. The result is outgoing radiance along each probe ray, ready for SH
// projection in the following pass.

@group(1) @binding(0) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(1) var<storage, read_write> ray_data: SVLMProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(2) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(3) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(4) var<storage, read> material_palette: array<u32>;
@group(1) @binding(5) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(6) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(7) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(8) var skybox_texture: texture_cube<f32>;

const SVLM_EMISSIVE_HIT_LUMA_SOFT_CAP = 2.0;
const SVLM_EMISSIVE_HIT_OVERFLOW_SCALE = 0.1;
fn svlm_stabilize_emissive_radiance(value: vec3<f32>) -> vec3<f32> {
    let clamped = safe_clamp_vec3_max(
        value,
        SVLM_MAX_RADIANCE_LUMINANCE
    );
    let value_luminance = max(luminance(clamped), 1e-6);
    let compressed_luminance = select(
        value_luminance,
        SVLM_EMISSIVE_HIT_LUMA_SOFT_CAP +
            (value_luminance - SVLM_EMISSIVE_HIT_LUMA_SOFT_CAP) *
            SVLM_EMISSIVE_HIT_OVERFLOW_SCALE,
        value_luminance > SVLM_EMISSIVE_HIT_LUMA_SOFT_CAP
    );
    return clamped * (compressed_luminance / value_luminance);
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (
        gid.x >= ray_data.header.active_ray_count ||
        gid.x >= arrayLength(&ray_data.rays)
    ) {
        return;
    }

    let hit = ray_data.rays[gid.x];
    let direction = hit.ray_direction.xyz;
    if (hit.hit_payload_t.w < 0.0) {
        ray_data.rays[gid.x].radiance = vec2<u32>(0u);
        return;
    }

    let is_miss = hit.state_u32.w == INVALID_IDX;
    if (is_miss) {
        let light_view_index = u32(scene_lighting_data.view_index);
        let sun_direction = normalize(
            -view_buffer[light_view_index].view_direction.xyz
        );
        let environment = evaluate_environment(
            direction,
            sun_direction,
            scene_lighting_data,
            skybox_texture
        );
        ray_data.rays[gid.x].radiance = svlm_pack_ray_radiance(
            environment
        );
        return;
    }

    let prim_store = hit.state_u32.x;
    if (prim_store >= arrayLength(&entity_index_lookup)) {
        ray_data.rays[gid.x].radiance = vec2<u32>(0u);
        return;
    }
    let entity_index = entity_index_lookup[prim_store];
    if (
        entity_index == INVALID_IDX ||
        entity_index >= arrayLength(&material_table_offset)
    ) {
        ray_data.rays[gid.x].radiance = vec2<u32>(0u);
        return;
    }

    let section_index = u32(hit.hit_payload_t.z);
    let palette_base = material_table_offset[entity_index];
    if (palette_base + section_index >= arrayLength(&material_palette)) {
        ray_data.rays[gid.x].radiance = vec2<u32>(0u);
        return;
    }
    let material_index = material_palette[palette_base + section_index];
    if (material_index >= arrayLength(&material_params)) {
        ray_data.rays[gid.x].radiance = vec2<u32>(0u);
        return;
    }
    let material = material_params[material_index];
    let uv = hit.hit_payload_t.xy *
        material.emission_roughness_metallic_tiling.w;
    let lod = 0.0;
    let albedo = sample_texture_or_vec4_param_handle(
        u32(material.albedo_handle),
        uv,
        material.albedo,
        u32(material.texture_flags1.x),
        texture_pool_albedo,
        lod
    ).xyz;
    let emission = sample_texture_or_float_param_handle(
        u32(material.emission_handle),
        uv,
        material.emission_roughness_metallic_tiling.x,
        u32(material.texture_flags2.w),
        texture_pool_emission,
        lod
    );

    let visibility = select(0.0, 1.0, hit.state_u32.z == 1u);
    var radiance = safe_clamp_vec3_max(
        svlm_unpack_ray_radiance(hit.nee_light_radiance) * visibility,
        SVLM_MAX_RADIANCE_LUMINANCE
    );
    // Match DDGI's NEE seed convention. The hit pass has already selected and
    // visibility-tested a light sample; applying another cosine here strongly
    // suppresses the sparse one-sample seed, especially on the secondary
    // surfaces that are responsible for lighting shadowed regions.
    radiance *= max(albedo, vec3<f32>(0.0)) * (1.0 / (2.0 * PI));

    if (emission > 0.0) {
        radiance += svlm_stabilize_emissive_radiance(
            emission * max(albedo, vec3<f32>(0.0))
        );
    }
    ray_data.rays[gid.x].radiance = svlm_pack_ray_radiance(radiance);
}

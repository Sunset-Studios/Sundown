#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/surface_cache_common.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(3) var<storage, read> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(4) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(5) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(6) var<storage, read> material_palette: array<u32>;
@group(1) @binding(7) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(8) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(9) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(11) var skybox_texture: texture_cube<f32>;
@group(1) @binding(12) var<storage, read_write> radiance_info: array<SurfaceCacheRadianceInfo>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= surface_cache_total_ray_count(counters)) {
        return;
    }
    let work = surface_cache_ray_work(
        gid.x,
        counters
    );
    let ray_data_index = work.data_index;

    let ray_direction = hit_info[ray_data_index].ray_direction_sampling_weight.xyz;
    let hit_identity = hit_info[ray_data_index].hit_identity;
    let hit_barycentrics = hit_info[ray_data_index].hit_barycentrics_t.xy;

    // Backfaces stop traversal but are excluded from accumulation. Treating
    // them as misses leaks environment radiance through thin geometry, while
    // shading them as valid samples biases otherwise converged patches black.
    if (hit_identity.x == SURFACE_CACHE_BACKFACE_HIT) {
        radiance_info[ray_data_index].shadow_direction = vec4<f32>(0.0);
        radiance_info[ray_data_index].sample_radiance = vec4<f32>(0.0);
        radiance_info[ray_data_index].shadow_radiance = vec4<f32>(0.0);
        return;
    }

    if (hit_identity.x == INVALID_IDX) {
        radiance_info[ray_data_index].shadow_direction = vec4<f32>(0.0);
        let light_view_index = u32(scene_lighting_data.view_index);
        let sun_direction = normalize(-view_buffer[light_view_index].view_direction.xyz);
        let environment_radiance = evaluate_environment(
            ray_direction,
            sun_direction,
            scene_lighting_data,
            skybox_texture
        );
        radiance_info[ray_data_index].sample_radiance = vec4<f32>(
            safe_clamp_vec3_max(environment_radiance, SURFACE_CACHE_MAX_RADIANCE),
            1.0
        );
        radiance_info[ray_data_index].shadow_radiance = vec4<f32>(0.0);
        return;
    }

    let entity_resolved = hit_identity.x;
    let instance_transform = compact_transforms[entity_resolved];
    let vertex0 = decode_vertex(vertex_buffer[hit_identity.y]);
    let vertex1 = decode_vertex(vertex_buffer[hit_identity.z]);
    let vertex2 = decode_vertex(vertex_buffer[hit_identity.w]);
    let bary_u = hit_barycentrics.x;
    let bary_v = hit_barycentrics.y;
    let bary_w = 1.0 - bary_u - bary_v;
    let uv = vertex0.uv * bary_w + vertex1.uv * bary_u + vertex2.uv * bary_v;
    let position_local = vertex0.position.xyz * bary_w
        + vertex1.position.xyz * bary_u
        + vertex2.position.xyz * bary_v;
    let hit_position = transform_local_point_from_instance(
        instance_transform,
        position_local
    );
    let normal_local = vertex0.normal.xyz * bary_w
        + vertex1.normal.xyz * bary_u
        + vertex2.normal.xyz * bary_v;
    var world_normal = safe_normalize(transform_local_direction_from_instance(
        instance_transform,
        normal_local
    ));

    let palette_base = material_table_offset[entity_resolved];
    let material_index = material_palette[palette_base + u32(vertex0.section_index)];
    let material = material_params[material_index];
    let base_uv = uv * material.emission_roughness_metallic_tiling.w;
    let albedo = sample_texture_or_vec4_param_handle(
        u32(material.albedo_handle), base_uv, material.albedo,
        u32(material.texture_flags1.x), texture_pool_albedo, 0.0
    ).xyz;
    let emissive = sample_emission_handle(
        u32(material.emission_handle), base_uv, material.emission_roughness_metallic_tiling.x,
        u32(material.texture_flags2.w), texture_pool_emission, 0.0
    );

    var shading_normal = world_normal;
    let normal_flags = u32(material.texture_flags1.y);
    if ((normal_flags & NORMAL_TEXTURE_FLAG_PRESENT) != 0u) {
        let tangent_local = vertex0.tangent.xyz * bary_w
            + vertex1.tangent.xyz * bary_u
            + vertex2.tangent.xyz * bary_v;
        let bitangent_local = vertex0.bitangent.xyz * bary_w
            + vertex1.bitangent.xyz * bary_u
            + vertex2.bitangent.xyz * bary_v;
        let world_tangent = safe_normalize(transform_local_direction_from_instance(
            instance_transform,
            tangent_local
        ));
        let world_bitangent = safe_normalize(transform_local_direction_from_instance(
            instance_transform,
            bitangent_local
        ));
        let tangent_frame = mat3x3<f32>(world_tangent, world_bitangent, world_normal);
        let normal_sample = decode_normal_texture_sample(
            sample_handle_rgba(u32(material.normal_handle), base_uv, texture_pool_normal, 0.0).xyz,
            normal_flags
        );
        shading_normal = normalize(tangent_frame * normal_sample);
    }

    // Shadow fields carry material attributes until recurrence and light sampling
    // consume them. Reusing this record keeps the per-ray working set unchanged.
    radiance_info[ray_data_index].shadow_origin = vec4<f32>(hit_position, emissive);
    radiance_info[ray_data_index].shadow_direction = vec4<f32>(shading_normal, 1.0);
    radiance_info[ray_data_index].shadow_radiance = vec4<f32>(albedo * (1.0 / (2.0 * PI)), 0.0);
    radiance_info[ray_data_index].sample_radiance = vec4<f32>(albedo, 1.0);
}

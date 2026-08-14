#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/surface_cache_common.wgsl"
#include "gi/surface_cache_lookup.wgsl"

@group(1) @binding(0) var<uniform> surface_cache_params: SurfaceCacheParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read> surface_cache: array<SurfacePatchReadOnly>;
@group(1) @binding(3) var<storage, read> surface_cache_sh: array<u32>;
@group(1) @binding(4) var<storage, read> update_indices: array<u32>;
@group(1) @binding(5) var<storage, read> bootstrap_indices: array<u32>;
@group(1) @binding(6) var<storage, read> counters: SurfaceCacheCountersReadOnly;
@group(1) @binding(7) var<storage, read> hit_info: array<SurfaceCacheHitInfo>;
@group(1) @binding(8) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(9) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(10) var<storage, read> material_palette: array<u32>;
@group(1) @binding(11) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(12) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(13) var<storage, read> emissive_lights_buffer: EmissiveLightsBuffer;
@group(1) @binding(14) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(16) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(17) var skybox_texture: texture_cube<f32>;
@group(1) @binding(18) var<storage, read_write> radiance_info: array<SurfaceCacheRadianceInfo>;
@group(1) @binding(19) var<storage, read> surface_cache_hashmap: array<HashMapEntry>;

fn sample_weighted_surface_cache_emissive_light(
    rng: ptr<function, u32>,
    emissive_light_count: u32,
    emissive_pdf: ptr<function, f32>
) -> u32 {
    let safe_emissive_count = max(emissive_light_count, 1u);
    (*emissive_pdf) = 1.0 / f32(safe_emissive_count);

    (*rng) = random_seed((*rng));
    let uniform_rand = rand_float((*rng));
    var selected_emissive_index =
        u32(uniform_rand * f32(safe_emissive_count)) % safe_emissive_count;
    if (emissive_light_count == 0u) {
        return selected_emissive_index;
    }

    let total_sampling_weight =
        f32(emissive_lights_buffer.header._pad0) * EMISSIVE_WEIGHT_QUANTIZATION_INV;
    let max_sampling_weight =
        f32(emissive_lights_buffer.header._pad1) * EMISSIVE_WEIGHT_QUANTIZATION_INV;
    let can_use_weighted_sampling =
        total_sampling_weight > 0.0 && max_sampling_weight > 0.0;

    var accepted = false;
    var selected_weight = 0.0;
    if (can_use_weighted_sampling) {
        for (
            var attempt_index = 0u;
            attempt_index < EMISSIVE_WEIGHTED_SAMPLE_ATTEMPTS;
            attempt_index = attempt_index + 1u
        ) {
            (*rng) = random_seed((*rng));
            let candidate_rand = rand_float((*rng));
            let candidate_index =
                u32(candidate_rand * f32(emissive_light_count)) % emissive_light_count;

            selected_weight = max(
                emissive_lights_buffer.lights[candidate_index].radiance_weight.w,
                0.0
            );
            (*rng) = random_seed((*rng));
            if (rand_float((*rng)) <= min(selected_weight / max_sampling_weight, 1.0)) {
                selected_emissive_index = candidate_index;
                accepted = true;
                break;
            }
        }
    }

    if (accepted) {
        (*emissive_pdf) = selected_weight / max(total_sampling_weight, 1e-6);
    }
    return selected_emissive_index;
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= surface_cache_total_ray_count(counters, surface_cache_params)) {
        return;
    }
    let work = surface_cache_ray_work(
        gid.x,
        arrayLength(&radiance_info),
        counters,
        surface_cache_params
    );
    let ray_data_index = work.data_index;

    let ray_direction = hit_info[ray_data_index].ray_direction_sampling_weight.xyz;
    let hit_identity = hit_info[ray_data_index].hit_identity;
    let hit_barycentrics = hit_info[ray_data_index].hit_barycentrics_t.xy;

    // Backfaces stop traversal but are excluded from accumulation. Treating
    // them as misses leaks environment radiance through thin geometry, while
    // shading them as valid samples biases otherwise converged patches black.
    if (hit_identity.x == SURFACE_CACHE_BACKFACE_HIT) {
        radiance_info[ray_data_index].sample_radiance = vec4<f32>(0.0);
        radiance_info[ray_data_index].shadow_radiance = vec4<f32>(0.0);
        return;
    }

    if (hit_identity.x == INVALID_IDX) {
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
    let emissive = sample_texture_or_float_param_handle(
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

    // Follow DDGI's diffuse-only recurrence. The cache lookup returns incident
    // irradiance here; material response is applied exactly once at this hit.
    // The 1 / (2 PI) factor intentionally matches DDGI's stable diffuse seed.
    let diffuse_response = albedo * (1.0 / (2.0 * PI));
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

    let analytic_light_count = dense_lights_buffer.header.light_count;
    let emissive_light_count = min(
        emissive_lights_buffer.header.light_count,
        arrayLength(&emissive_lights_buffer.lights)
    );
    let total_light_count = analytic_light_count + emissive_light_count;
    if (total_light_count == 0u) {
        radiance_info[ray_data_index].shadow_radiance = vec4<f32>(0.0);
        return;
    }

    var patch_index = 0u;
    if (work.bootstrap_batch != 0u) {
        patch_index = bootstrap_indices[
            surface_cache_bootstrap_schedule_index(
                work.active_index,
                counters
            )
        ];
    } else {
        patch_index = update_indices[surface_cache_regular_schedule_index(
            work.active_index,
            counters
        )];
    }
    let surface_patch = surface_cache[patch_index];

    // Follow the patch's progressive sample sequence rather than absolute
    // frame time. Intermittently updated patches therefore retain the same
    // light-sampling sequence regardless of their scheduling phase.
    let light_sample_index = u32(surface_patch.history.y) + work.ray_index_in_patch;
    var rng = random_seed(surface_cache_patch_rng(
        patch_index,
        surface_patch.grid_key
    ) ^ hash(
        (light_sample_index ^ 0xa24baeddu) * 0x9e3779b9u
    ));
    rng = random_seed(rng);
    let light_bucket_rand = rand_float(rng);
    let emissive_bucket_pdf = f32(emissive_light_count) / f32(total_light_count);
    let analytic_bucket_pdf = 1.0 - emissive_bucket_pdf;
    let select_emissive = emissive_light_count > 0u && (
        analytic_light_count == 0u || light_bucket_rand >= analytic_bucket_pdf
    );

    var light_direction = vec3<f32>(0.0, 1.0, 0.0);
    var shadow_distance = 0.0;
    var direct_irradiance = vec3<f32>(0.0);
    if (!select_emissive) {
        rng = random_seed(rng);
        let analytic_rand = rand_float(rng);
        let selected_light_index =
            u32(analytic_rand * f32(analytic_light_count)) % max(analytic_light_count, 1u);
        let light = dense_lights_buffer.lights[selected_light_index];
        light_direction = get_light_dir(light, hit_position);
        let attenuation = get_light_attenuation(light, hit_position);
        let analytic_light_pdf =
            analytic_bucket_pdf / max(f32(analytic_light_count), 1.0);
        direct_irradiance =
            light.color.rgb * light.intensity * attenuation /
            max(analytic_light_pdf, 1e-6);
        var light_distance = surface_cache_params.max_ray_length;
        if (light.light_type != 0.0) {
            light_distance = length(light.position.xyz - hit_position);
        }
        shadow_distance = min(
            light_distance * 0.999,
            surface_cache_params.max_ray_length
        );
    } else {
        var emissive_pdf = 0.0;
        let emissive_light_index = sample_weighted_surface_cache_emissive_light(
            &rng,
            emissive_light_count,
            &emissive_pdf
        );
        let emissive_light = emissive_lights_buffer.lights[emissive_light_index];
        let to_emissive = emissive_light.position_radius.xyz - hit_position;
        let distance_squared = max(dot(to_emissive, to_emissive), 1e-6);
        let light_distance = sqrt(distance_squared);
        light_direction = to_emissive / light_distance;
        let light_facing = max(
            dot(emissive_light.normal_area.xyz, -light_direction),
            0.0
        );
        let solid_angle_scale = emissive_light.normal_area.w / distance_squared;
        let emissive_light_pdf = emissive_bucket_pdf * emissive_pdf;
        direct_irradiance =
            emissive_light.radiance_weight.xyz * light_facing * solid_angle_scale /
            max(emissive_light_pdf, 1e-6);
        shadow_distance = min(
            max(0.0, light_distance - emissive_light.position_radius.w) * 0.999,
            surface_cache_params.max_ray_length
        );
    }

    let direct_radiance = safe_clamp_vec3_max(
        direct_irradiance,
        SURFACE_CACHE_MAX_RADIANCE
    ) * diffuse_response;
    radiance_info[ray_data_index].shadow_origin = vec4<f32>(
        hit_position + light_direction * 0.001,
        0.0
    );
    radiance_info[ray_data_index].shadow_direction = vec4<f32>(
        light_direction,
        shadow_distance
    );
    // w = 1 means pending visibility; the shadow pass changes it to 2 only
    // when the light is visible.
    radiance_info[ray_data_index].shadow_radiance = vec4<f32>(
        direct_radiance,
        1.0
    );
}

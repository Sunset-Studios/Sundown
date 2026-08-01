// =============================================================================
// DDGI Probe Ray Trace - Hit Pass
// - Traces primary rays from world-space probes against the BVH
// - Writes compact hit attributes for the shade pass
// - Intentionally avoids ALL material + texture bindings to reduce binding count
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "gi/ddgi_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> probe_ray_data: DDGIProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(2) var<storage, read> probe_update_indices: array<u32>;
@group(1) @binding(3) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(4) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(5) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(6) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(7) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(8) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(9) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(10) var<storage, read> emissive_lights_buffer: EmissiveLightsBuffer;
@group(1) @binding(11) var<storage, read> entity_index_lookup: array<u32>;

// =============================================================================
// HELPER: Process a shadow ray and write result
// =============================================================================
fn trace_shadow_visibility(ray_origin: vec3<f32>, ray_dir: vec3<f32>, t_max: f32) -> bool {
    var ray: Ray;
    ray.origin_and_tmin = vec4<f32>(ray_origin + ray_dir * 0.001, 0.0);
    ray.direction_and_tmax = vec4<f32>(ray_dir, t_max);
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray.direction_and_tmax.x), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.x < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.y), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.y < 0.0),
        1.0 / max(abs(ray.direction_and_tmax.z), 1e-8) * select(1.0, -1.0, ray.direction_and_tmax.z < 0.0),
        0.0
    );

    return !trace_ray_any(&ray);
}

fn sample_weighted_emissive_light(
    rng: ptr<function, u32>,
    num_emissive_lights: u32,
    emissive_pdf: ptr<function, f32>
) -> u32 {
    let safe_emissive_count = max(num_emissive_lights, 1u);
    let uniform_pdf = 1.0 / f32(safe_emissive_count);
    (*emissive_pdf) = uniform_pdf;

    (*rng) = random_seed((*rng));
    let uniform_rand = rand_float((*rng));
    var selected_emissive_idx = u32(uniform_rand * f32(safe_emissive_count)) % safe_emissive_count;

    if (num_emissive_lights == 0u) {
        return selected_emissive_idx;
    }

    let total_sampling_weight =
        f32(emissive_lights_buffer.header._pad0) * EMISSIVE_WEIGHT_QUANTIZATION_INV;
    let max_sampling_weight =
        f32(emissive_lights_buffer.header._pad1) * EMISSIVE_WEIGHT_QUANTIZATION_INV;
    let can_use_weighted_sampling =
        total_sampling_weight > 0.0 && max_sampling_weight > 0.0;

    var accepted = false;
    if (can_use_weighted_sampling) {
        for (var attempt_idx = 0u; attempt_idx < EMISSIVE_WEIGHTED_SAMPLE_ATTEMPTS; attempt_idx = attempt_idx + 1u) {
            (*rng) = random_seed((*rng));
            let candidate_rand = rand_float((*rng));
            let candidate_idx = u32(candidate_rand * f32(num_emissive_lights)) % num_emissive_lights;
            let candidate_weight = max(emissive_lights_buffer.lights[candidate_idx].radiance_weight.w, 0.0);
            let accept_prob = min(candidate_weight / max_sampling_weight, 1.0);

            (*rng) = random_seed((*rng));
            let accept_rand = rand_float((*rng));
            if (accept_rand <= accept_prob) {
                selected_emissive_idx = candidate_idx;
                accepted = true;
                break;
            }
        }
    }

    if (accepted) {
        let selected_weight = max(emissive_lights_buffer.lights[selected_emissive_idx].radiance_weight.w, 0.0);
        (*emissive_pdf) = selected_weight / max(total_sampling_weight, 1e-6);
    }

    return selected_emissive_idx;
}

// =============================================================================
// HELPER: Process a primary ray and write hit attributes
// =============================================================================
fn process_primary_ray(
    index: u32,
    probe_position: vec3<f32>,
    ray_dir: vec3<f32>,
    probe_index: u32,
    ray_index_in_probe: u32,
) {
    probe_ray_data.rays[index].ray_dir_x = ray_dir.x;
    probe_ray_data.rays[index].ray_dir_y = ray_dir.y;
    probe_ray_data.rays[index].ray_dir_z = ray_dir.z;
    probe_ray_data.rays[index].hit_distance = 0.0;
    probe_ray_data.rays[index].prim_store = INVALID_IDX;
    probe_ray_data.rays[index].vertex_index_0 = 0u;
    probe_ray_data.rays[index].vertex_index_1 = 0u;
    probe_ray_data.rays[index].vertex_index_2 = 0u;
    probe_ray_data.rays[index].barycentric_u = 0.0;
    probe_ray_data.rays[index].barycentric_v = 0.0;
    ddgi_probe_ray_set_radiance(&probe_ray_data.rays[index], vec3<f32>(0.0));

    var ray: Ray;
    ray.origin_and_tmin = vec4<f32>(probe_position + ray_dir * 0.001, 0.0);
    ray.direction_and_tmax = vec4<f32>(ray_dir, ddgi_params.max_ray_length);
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(ray_dir.x), 1e-8) * select(1.0, -1.0, ray_dir.x < 0.0),
        1.0 / max(abs(ray_dir.y), 1e-8) * select(1.0, -1.0, ray_dir.y < 0.0),
        1.0 / max(abs(ray_dir.z), 1e-8) * select(1.0, -1.0, ray_dir.z < 0.0),
        0.0
    );

    let hit_result = trace_ray_closest(&ray);

    if (hit_result.has_hit != 0u) {
        let prim_store = hit_result.prim_store;
        let entity_resolved = entity_index_lookup[prim_store];
        let entity_transform = entity_transforms[entity_resolved];

        var ray_local = build_local_ray(
            &ray,
            entity_transform.transform,
            entity_transform.transpose_inverse_model_matrix
        );

        let t_tri = hit_result.t_hit;
        let p_local = ray_local.origin_and_tmin.xyz + ray_local.direction_and_tmax.xyz * t_tri;
        let p_world = (entity_transform.transform * vec4<f32>(p_local, 1.0)).xyz;

        let v0i = hit_result.tri_indices.x;
        let v1i = hit_result.tri_indices.y;
        let v2i = hit_result.tri_indices.z;

        let vertex0 = decode_vertex(vertex_buffer[v0i]);
        let vertex1 = decode_vertex(vertex_buffer[v1i]);
        let vertex2 = decode_vertex(vertex_buffer[v2i]);
        let v0 = vertex0.position.xyz;
        let v1 = vertex1.position.xyz;
        let v2 = vertex2.position.xyz;

        let e0 = v1 - v0;
        let e1 = v2 - v0;
        let vp = p_local - v0;
        let d00 = dot(e0, e0);
        let d01 = dot(e0, e1);
        let d11 = dot(e1, e1);
        let d20 = dot(vp, e0);
        let d21 = dot(vp, e1);
        let denom = max(d00 * d11 - d01 * d01, 1e-8);
        let v_bc = (d00 * d21 - d01 * d20) / denom;
        let u_bc = (d11 * d20 - d01 * d21) / denom;
        let w_bc = 1.0 - u_bc - v_bc;

        let n_local = vertex0.normal.xyz * w_bc +
            vertex1.normal.xyz * u_bc +
            vertex2.normal.xyz * v_bc;
        let world_n = safe_normalize(
            (entity_transform.transpose_inverse_model_matrix * vec4<f32>(n_local, 0.0)).xyz
        );

        let ray_is_backfacing = dot(world_n, ray_dir) > 0.0;

        // Backface rays:
        // - Mark with NEGATIVE distance so the shade pass can zero irradiance (leak reduction).
        // - Shorten their stored depth by 80% (multiply by 0.2) for conservative visibility.
        //   (World hit position stays unmodified; we only adjust the stored "t".)
        // - Using negative distance allows efficient backface counting without extra flags,
        //   enabling robust dead probe detection even with non-manifold geometry.
        let stored_t = select(t_tri, -t_tri, ray_is_backfacing);
        probe_ray_data.rays[index].hit_distance = stored_t;
        probe_ray_data.rays[index].prim_store = prim_store;
        probe_ray_data.rays[index].vertex_index_0 = v0i;
        probe_ray_data.rays[index].vertex_index_1 = v1i;
        probe_ray_data.rays[index].vertex_index_2 = v2i;
        probe_ray_data.rays[index].barycentric_u = u_bc;
        probe_ray_data.rays[index].barycentric_v = v_bc;

        // One-sample NEE visibility test at the primary hit point.
        // We do the expensive shadow trace here (hit pass has BVH bindings),
        // and retain only its visible radiance contribution for the shade pass.
        let num_lights = dense_lights_buffer.header.light_count;
        let num_emissive_lights = emissive_lights_buffer.header.light_count;
        let total_light_count = num_lights + num_emissive_lights;
        if (total_light_count > 0u) {
            var nee_rng = hash(
                probe_index
                    ^ (ray_index_in_probe * 0xA24BAEDDu)
                    ^ (u32(ddgi_params.frame_index) * 0x9E3779B9u)
            );
            nee_rng = random_seed(nee_rng);
            let light_rand = rand_float(nee_rng);
            let emissive_bucket_pdf = f32(num_emissive_lights) / f32(total_light_count);
            let analytic_bucket_pdf = 1.0 - emissive_bucket_pdf;
            let select_emissive =
                num_emissive_lights > 0u && (num_lights == 0u || light_rand >= analytic_bucket_pdf);

            if (!select_emissive) {
                nee_rng = random_seed(nee_rng);
                let analytic_rand = rand_float(nee_rng);
                let selected_light_idx = u32(analytic_rand * f32(num_lights)) % max(num_lights, 1u);
                let light = dense_lights_buffer.lights[selected_light_idx];
                let shadow_dir = get_light_dir(light, p_world);
                let attenuation = get_light_attenuation(light, p_world);
                let light_distance = select(1e30, length(light.position.xyz - p_world), light.light_type != 0.0);
                let shadow_t_max = light_distance * 0.999;
                let analytic_light_pdf = analytic_bucket_pdf * (1.0 / max(f32(num_lights), 1.0));
                let analytic_light_scale = 1.0 / max(analytic_light_pdf, 1e-6);

                let nee_radiance =
                    light.color.rgb * light.intensity * attenuation * analytic_light_scale;
                if (trace_shadow_visibility(p_world, shadow_dir, shadow_t_max)) {
                    ddgi_probe_ray_set_radiance(&probe_ray_data.rays[index], nee_radiance);
                }
            } else {
                var emissive_pdf = 0.0;
                let emissive_idx = sample_weighted_emissive_light(
                    &nee_rng,
                    num_emissive_lights,
                    &emissive_pdf
                );
                let emissive_light = emissive_lights_buffer.lights[emissive_idx];
                let to_emissive = emissive_light.position_radius.xyz - p_world;
                let distance_sq = max(dot(to_emissive, to_emissive), 1e-6);
                let distance = sqrt(distance_sq);
                let shadow_dir = to_emissive / distance;
                let light_facing = max(dot(emissive_light.normal_area.xyz, -shadow_dir), 0.0);
                let solid_angle_scale = emissive_light.normal_area.w / distance_sq;
                let shadow_t_max = max(0.0, distance - emissive_light.position_radius.w) * 0.999;
                let emissive_light_pdf = emissive_bucket_pdf * emissive_pdf;
                let emissive_light_scale = 1.0 / max(emissive_light_pdf, 1e-6);

                let nee_radiance =
                    emissive_light.radiance_weight.xyz * light_facing * solid_angle_scale * emissive_light_scale;
                if (trace_shadow_visibility(p_world, shadow_dir, shadow_t_max)) {
                    ddgi_probe_ray_set_radiance(&probe_ray_data.rays[index], nee_radiance);
                }
            }
        }
    }
}

// =============================================================================
// Main
// =============================================================================
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let active_ray_count = probe_ray_data.header.active_ray_count;

    if (gid.x >= active_ray_count) {
        return;
    }

    let rays_per_probe = ddgi_max_rays_per_probe(&ddgi_params);
    let probe_slot = gid.x / rays_per_probe;
    let ray_index_in_probe = gid.x - probe_slot * rays_per_probe;
    let probe_index = probe_update_indices[probe_slot];
    let probe_position = ddgi_probe_world_position_from_index(&ddgi_params, probe_index);

    let ray_dir = ddgi_probe_ray_direction(
        &ddgi_params,
        probe_index,
        ray_index_in_probe,
        rays_per_probe
    );
    process_primary_ray(gid.x, probe_position, ray_dir, probe_index, ray_index_in_probe);
}


#include "gi/svlm_common.wgsl"
#include "lighting_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

// Traces the transient SVLM probe-ray batch and records compact hit attributes.
// Material/texture evaluation remains in a separate pass to keep this BVH-heavy
// binding set within WebGPU per-stage limits.

@group(1) @binding(0) var<storage, read_write> ray_data: SVLMProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(1) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(2) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(3) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(4) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(5) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(6) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(7) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(8) var<storage, read> emissive_lights_buffer: EmissiveLightsBuffer;
@group(1) @binding(9) var<storage, read> entity_index_lookup: array<u32>;

fn svlm_safe_inverse_direction(direction: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        1.0 / max(abs(direction.x), 1e-8) * select(1.0, -1.0, direction.x < 0.0),
        1.0 / max(abs(direction.y), 1e-8) * select(1.0, -1.0, direction.y < 0.0),
        1.0 / max(abs(direction.z), 1e-8) * select(1.0, -1.0, direction.z < 0.0)
    );
}

fn svlm_process_shadow_visibility(
    ray_index: u32,
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_max: f32
) {
    var ray: Ray;
    ray.origin_and_tmin = vec4<f32>(origin + direction * 0.001, 0.0);
    ray.direction_and_tmax = vec4<f32>(direction, max(t_max, 0.0));
    ray.inv_direction = vec4<f32>(
        svlm_safe_inverse_direction(direction),
        0.0
    );
    if (!trace_ray_any(&ray)) {
        ray_data.rays[ray_index].state_u32.z = 1u;
    }
}

fn svlm_sample_weighted_emissive_light(
    rng: ptr<function, u32>,
    light_count: u32,
    output_pdf: ptr<function, f32>
) -> u32 {
    let safe_light_count = max(light_count, 1u);
    (*output_pdf) = 1.0 / f32(safe_light_count);
    (*rng) = random_seed((*rng));
    var selected_index = u32(
        rand_float((*rng)) * f32(safe_light_count)
    ) % safe_light_count;

    if (light_count == 0u) {
        return selected_index;
    }

    let total_weight =
        f32(emissive_lights_buffer.header._pad0) *
        EMISSIVE_WEIGHT_QUANTIZATION_INV;
    let max_weight =
        f32(emissive_lights_buffer.header._pad1) *
        EMISSIVE_WEIGHT_QUANTIZATION_INV;
    if (total_weight <= 0.0 || max_weight <= 0.0) {
        return selected_index;
    }

    var accepted = false;
    for (
        var attempt = 0u;
        attempt < EMISSIVE_WEIGHTED_SAMPLE_ATTEMPTS;
        attempt = attempt + 1u
    ) {
        (*rng) = random_seed((*rng));
        let candidate = u32(
            rand_float((*rng)) * f32(light_count)
        ) % light_count;
        let candidate_weight = max(
            emissive_lights_buffer.lights[candidate].radiance_weight.w,
            0.0
        );
        (*rng) = random_seed((*rng));
        if (rand_float((*rng)) <= min(candidate_weight / max_weight, 1.0)) {
            selected_index = candidate;
            accepted = true;
            break;
        }
    }

    if (accepted) {
        let selected_weight = max(
            emissive_lights_buffer.lights[selected_index].radiance_weight.w,
            0.0
        );
        (*output_pdf) = selected_weight / max(total_weight, 1e-6);
    }
    return selected_index;
}

fn svlm_trace_primary_ray(ray_index: u32) {
    let record = ray_data.rays[ray_index];
    let probe_position = record.hit_payload_t.xyz;
    let direction = record.ray_direction.xyz;
    let max_ray_distance = max(bitcast<f32>(record.state_u32.y), 1.0);

    var ray: Ray;
    ray.origin_and_tmin = vec4<f32>(probe_position + direction * 0.001, 0.0);
    ray.direction_and_tmax = vec4<f32>(direction, max_ray_distance);
    ray.inv_direction = vec4<f32>(
        svlm_safe_inverse_direction(direction),
        0.0
    );

    let hit_result = trace_ray_closest(&ray);
    if (hit_result.has_hit == 0u) {
        return;
    }

    let tri_id_local = hit_result.tri_id_local;
    let prim_store = hit_result.prim_store;
    if (prim_store >= arrayLength(&entity_index_lookup)) {
        return;
    }
    let entity_index = entity_index_lookup[prim_store];
    if (
        entity_index == INVALID_IDX ||
        entity_index >= arrayLength(&compact_transforms)
    ) {
        return;
    }
    let instance_transform = compact_transforms[entity_index];
    var local_ray = build_local_ray_from_instance(
        &ray,
        instance_transform
    );

    let t_hit = hit_result.t_hit;
    let local_position =
        local_ray.origin_and_tmin.xyz +
        local_ray.direction_and_tmax.xyz * t_hit;
    let world_position = transform_local_point_from_instance(
        instance_transform,
        local_position
    );

    let vertex0 = decode_vertex(vertex_buffer[hit_result.tri_indices.x]);
    let vertex1 = decode_vertex(vertex_buffer[hit_result.tri_indices.y]);
    let vertex2 = decode_vertex(vertex_buffer[hit_result.tri_indices.z]);
    let p0 = vertex0.position.xyz;
    let edge0 = vertex1.position.xyz - p0;
    let edge1 = vertex2.position.xyz - p0;
    let to_hit = local_position - p0;
    let d00 = dot(edge0, edge0);
    let d01 = dot(edge0, edge1);
    let d11 = dot(edge1, edge1);
    let d20 = dot(to_hit, edge0);
    let d21 = dot(to_hit, edge1);
    let denominator = max(d00 * d11 - d01 * d01, 1e-8);
    let bary_v = (d00 * d21 - d01 * d20) / denominator;
    let bary_u = (d11 * d20 - d01 * d21) / denominator;
    let bary_w = 1.0 - bary_u - bary_v;

    let uv =
        vertex0.uv * bary_w +
        vertex1.uv * bary_u +
        vertex2.uv * bary_v;
    let normal_local =
        vertex0.normal.xyz * bary_w +
        vertex1.normal.xyz * bary_u +
        vertex2.normal.xyz * bary_v;
    let world_normal = safe_normalize(
        transform_local_direction_from_instance(instance_transform, normal_local)
    );
    let backfacing = dot(world_normal, direction) > 0.0;

    ray_data.rays[ray_index].hit_payload_t = vec4<f32>(
        uv,
        vertex0.section_index,
        select(t_hit, -t_hit, backfacing)
    );
    ray_data.rays[ray_index].state_u32.x = prim_store;
    ray_data.rays[ray_index].state_u32.w = tri_id_local;

    let analytic_count = dense_lights_buffer.header.light_count;
    let emissive_count = emissive_lights_buffer.header.light_count;
    let total_light_count = analytic_count + emissive_count;
    if (total_light_count == 0u || backfacing) {
        return;
    }

    var rng = hash(
        (ray_data.header.probe_cursor +
            ray_index / max(ray_data.header.rays_per_probe, 1u)) ^
        ((ray_index % max(ray_data.header.rays_per_probe, 1u)) * 0xa24baeddu) ^
        (ray_data.header.sample_index * 0x9e3779b9u)
    );
    rng = random_seed(rng);
    let emissive_bucket_pdf =
        f32(emissive_count) / f32(total_light_count);
    let analytic_bucket_pdf = 1.0 - emissive_bucket_pdf;
    let choose_emissive =
        emissive_count > 0u &&
        (
            analytic_count == 0u ||
            rand_float(rng) >= analytic_bucket_pdf
        );

    if (!choose_emissive) {
        rng = random_seed(rng);
        let selected_index = u32(
            rand_float(rng) * f32(analytic_count)
        ) % max(analytic_count, 1u);
        let light = dense_lights_buffer.lights[selected_index];
        let light_direction = get_light_dir(light, world_position);
        let attenuation = get_light_attenuation(light, world_position);
        let light_distance = select(
            max_ray_distance,
            length(light.position.xyz - world_position),
            light.light_type != 0.0
        );
        let light_pdf =
            analytic_bucket_pdf / max(f32(analytic_count), 1.0);

        ray_data.rays[ray_index].nee_light_radiance = svlm_pack_ray_radiance(
            light.color.rgb *
            light.intensity *
            attenuation /
            max(light_pdf, 1e-6)
        );
        svlm_process_shadow_visibility(
            ray_index,
            world_position,
            light_direction,
            min(light_distance * 0.999, max_ray_distance)
        );
        return;
    }

    var emissive_pdf = 0.0;
    let selected_index = svlm_sample_weighted_emissive_light(
        &rng,
        emissive_count,
        &emissive_pdf
    );
    let light = emissive_lights_buffer.lights[selected_index];
    let to_light = light.position_radius.xyz - world_position;
    let distance_squared = max(dot(to_light, to_light), 1e-6);
    let distance = sqrt(distance_squared);
    let light_direction = to_light / distance;
    let light_facing = max(
        dot(light.normal_area.xyz, -light_direction),
        0.0
    );
    let solid_angle_scale = light.normal_area.w / distance_squared;
    let light_pdf = emissive_bucket_pdf * emissive_pdf;

    ray_data.rays[ray_index].nee_light_radiance = svlm_pack_ray_radiance(
        light.radiance_weight.xyz *
        light_facing *
        solid_angle_scale /
        max(light_pdf, 1e-6)
    );
    svlm_process_shadow_visibility(
        ray_index,
        world_position,
        light_direction,
        min(
            max(0.0, distance - light.position_radius.w) * 0.999,
            max_ray_distance
        )
    );
}

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    bvh_stack_lane = local_idx;
    if (
        gid.x >= ray_data.header.active_ray_count ||
        gid.x >= arrayLength(&ray_data.rays)
    ) {
        return;
    }
    svlm_trace_primary_ray(gid.x);
}

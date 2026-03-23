#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(1) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(2) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(3) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(4) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(5) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(6) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(7) var<storage, read> material_palette: array<u32>;
@group(1) @binding(8) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(9) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(10) var<storage, read_write> emissive_lights_buffer: EmissiveLightsBufferA;

const MAX_TRIANGLES_PER_LEAF: u32 = 32u;
fn triangle_area_and_normal(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec4<f32> {
    let edge0 = p1 - p0;
    let edge1 = p2 - p0;
    let cross_val = cross(edge0, edge1);
    let area2 = length(cross_val);
    let area = 0.5 * area2;
    let normal_ws = select(vec3<f32>(0.0, 1.0, 0.0), cross_val / area2, area2 > 1e-7);
    return vec4<f32>(normal_ws, area);
}

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_index = global_id.x;
    if (node_index >= tlas_bvh_info.bvh2_count) {
        return;
    }

    let node = tlas_bvh2_bounds[node_index];
    if (!is_leaf(node)) {
        return;
    }

    let mesh_id = u32(node.min.w);
    let prim_store = u32(-node.max.w - 1.0);
    if (mesh_id == INVALID_IDX) {
        return;
    }

    let mesh_directory_entry = blas_directory[mesh_id];
    let tri_count = mesh_directory_entry.leaf_count;
    if (tri_count == 0u) {
        return;
    }

    let sample_count = min(MAX_TRIANGLES_PER_LEAF, tri_count);

    // Deterministic temporal sweep:
    // each frame shifts by sample_count, so bounded per-frame work still covers all triangles over time.
    let frame_index_u32 = u32(frame_info.frame_index);
    let leaf_hash = hash(node_index ^ (prim_store * 0x9E3779B9u) ^ (mesh_id * 0x85EBCA6Bu));
    let start_tri = (leaf_hash + frame_index_u32 * sample_count) % tri_count;

    let entity_transform = entity_transforms[prim_store];
    let entity_palette_base = material_table_offset[prim_store];

    for (var sample_idx = 0u; sample_idx < sample_count; sample_idx = sample_idx + 1u) {
        let tri_id_local = (start_tri + sample_idx) % tri_count;
        let tri_base = mesh_directory_entry.first_index + tri_id_local * 3u;

        let v0i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 0u];
        let v1i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 1u];
        let v2i = mesh_directory_entry.first_vertex + index_buffer[tri_base + 2u];

        let section_index = u32(vertex_section_index(vertex_buffer[v0i]));
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let emissive_scalar = material.emission_roughness_metallic_tiling.x;
        if (emissive_scalar <= 0.0) {
            continue;
        }

        let p0_local = vertex_position(vertex_buffer[v0i]);
        let p1_local = vertex_position(vertex_buffer[v1i]);
        let p2_local = vertex_position(vertex_buffer[v2i]);

        let p0_world = (entity_transform.transform * vec4<f32>(p0_local, 1.0)).xyz;
        let p1_world = (entity_transform.transform * vec4<f32>(p1_local, 1.0)).xyz;
        let p2_world = (entity_transform.transform * vec4<f32>(p2_local, 1.0)).xyz;

        let area_and_normal = triangle_area_and_normal(p0_world, p1_world, p2_world);
        let tri_area = area_and_normal.w;

        // Estimate emission at triangle centroid UV so emissive NEE better matches
        // textured emitters and near-field intensity.
        let tiling = material.emission_roughness_metallic_tiling.w;
        let uv0 = vertex_uv(vertex_buffer[v0i]);
        let uv1 = vertex_uv(vertex_buffer[v1i]);
        let uv2 = vertex_uv(vertex_buffer[v2i]);
        let centroid_uv = ((uv0 + uv1 + uv2) * (1.0 / 3.0)) * tiling;
        let lod = 0.0;
        let centroid_albedo = sample_texture_or_vec4_param_handle(
            u32(material.albedo_handle),
            centroid_uv,
            material.albedo,
            u32(material.texture_flags1.x),
            texture_pool_albedo,
            lod
        ).xyz;
        let centroid_emissive = sample_texture_or_float_param_handle(
            u32(material.emission_handle),
            centroid_uv,
            emissive_scalar,
            u32(material.texture_flags2.w),
            texture_pool_emission,
            lod
        );
        let emissive_radiance = max(centroid_albedo, vec3<f32>(0.0)) * centroid_emissive;
        let sampling_weight = luminance(emissive_radiance) * tri_area;
        if (sampling_weight <= 0.0) {
            continue;
        }

        let dst = atomicAdd(&emissive_lights_buffer.header.light_count, 1u);
        if (dst < arrayLength(&emissive_lights_buffer.lights)) {
            let centroid = (p0_world + p1_world + p2_world) * (1.0 / 3.0);
            let extent_radius = max(
                max(length(p0_world - centroid), length(p1_world - centroid)),
                length(p2_world - centroid)
            );

            emissive_lights_buffer.lights[dst].position_radius = vec4<f32>(centroid, extent_radius);
            emissive_lights_buffer.lights[dst].normal_area = vec4<f32>(area_and_normal.xyz, tri_area);
            emissive_lights_buffer.lights[dst].radiance_weight = vec4<f32>(emissive_radiance, sampling_weight);
            let sampling_weight_q = u32(min(sampling_weight * EMISSIVE_WEIGHT_QUANTIZATION, 4294967295.0));
            _ = atomicAdd(&emissive_lights_buffer.header._pad0, sampling_weight_q);
            _ = atomicMax(&emissive_lights_buffer.header._pad1, sampling_weight_q);
            emissive_lights_buffer.lights[dst].instance_tri_section = vec4<u32>(
                prim_store,
                mesh_id,
                tri_id_local,
                section_index
            );
        }
    }
}

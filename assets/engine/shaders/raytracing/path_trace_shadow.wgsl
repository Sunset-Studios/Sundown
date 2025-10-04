// =============================================================================
// Path Tracer - Shadow Pass
// - Traces shadow rays written by shade pass for Next Event Estimation
// - Accumulates direct lighting contribution if unoccluded
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"

const TLAS_CANDIDATES = 4;

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    max_spp: u32,
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    throughput: vec4<f32>,
    state_u32: vec4<u32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    rng: f32,
    sample_count: f32,
    prim_id: f32,
    frame_stamp: f32,
    // Shadow ray state for Next Event Estimation
    shadow_origin: vec4<f32>,      // xyz = origin, w = tmin
    shadow_direction: vec4<f32>,    // xyz = direction, w = tmax
    shadow_radiance: vec4<f32>,     // rgb = light contribution, a = needs_trace flag
    path_weight: vec4<f32>,         // rgb = cumulative BRDF weight along path, a = unused
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(3) var<storage, read> tlas_bvh4_nodes: array<BVH4Node>;
@group(1) @binding(4) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(5) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(6) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(7) var<storage, read> mesh_asset_ids: array<u32>;
@group(1) @binding(8) var output_tex: texture_storage_2d<rgba16float, write>;

fn trace_tlas(ray: ptr<function, Ray>) -> vec4<f32> {
    // Returns: vec4(prim_index as f32, hit_flag, t_entry, t_exit)
    var result = vec4<f32>(-1.0, 0.0, 0.0, 0.0);

    var node_stack: array<u32, 32>;
    node_stack[0] = 0u;
    var stack_size = 1u;

    var t_entry_min = 1e38;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let current_node = tlas_bvh4_nodes[node_idx];
        let t_aabb = intersect_aabb(*ray, current_node.min.xyz, current_node.max.xyz);

        if (t_aabb >= (*ray).origin_and_tmin.w && t_aabb < (*ray).direction_and_tmax.w) {
            if (stack_size < 32u) {
                var min_index = 0u;
                var min_t = 1e38;
                let leaf_mask = bitcast<u32>(current_node.min.w);

                for (var i = 0u; i < 4u; i = i + 1u) {
                    let child_raw = current_node.children[i];
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        let leaf_bounds = tlas_bvh2_bounds[child_idx];
                        let entry_exit = aabb_entry_exit_for_ray(ray, leaf_bounds);
                        let t_entry = entry_exit.x;
                        let t_exit = entry_exit.y;
                        if (t_entry <= t_exit && t_entry >= (*ray).origin_and_tmin.w && t_entry < t_entry_min) {
                            // Select closest TLAS leaf
                            t_entry_min = t_entry;
                            result = vec4<f32>(f32(u32(leaf_bounds.min.w)), 1.0, t_entry, t_exit);
                        }
                    } else {
                        let child_node = tlas_bvh4_nodes[child_idx];
                        let t_aabb_child = intersect_aabb(*ray, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child >= (*ray).origin_and_tmin.w && t_aabb_child < t_entry_min) {
                            min_index = stack_size;
                            min_t = t_aabb_child;
                            node_stack[stack_size] = child_idx;
                            stack_size = stack_size + 1u;
                        }
                    }
                }

                let tmp_node = node_stack[min_index];
                node_stack[min_index] = node_stack[stack_size - 1u];
                node_stack[stack_size - 1u] = tmp_node;
            }
        }
    }

    return result;
}

fn trace_blas_any_hit(ray_world: ptr<function, Ray>, ray_local: ptr<function, Ray>, entity_transform: mat4x4f, mesh_asset_id: u32) -> bool {
    let mesh_directory_entry = atlas_load_directory_entry(mesh_asset_id);
    let bvh4_base = mesh_directory_entry.bvh4_base;
    let first_vertex = mesh_directory_entry.first_vertex;
    let first_index = mesh_directory_entry.first_index;

    var node_stack: array<u32, 32>;
    node_stack[0] = bvh4_base;
    var stack_size = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = atlas_load_bvh4_node(node_idx);
        let t_aabb = intersect_aabb(*ray_local, node.min.xyz, node.max.xyz);

        if (t_aabb >= ray_local.origin_and_tmin.w && t_aabb < ray_local.direction_and_tmax.w) {
            if (stack_size < 32u) {
                let leaf_mask = bitcast<u32>(node.min.w);

                for (var i = 0u; i < 4u; i = i + 1u) {
                    let child_raw = node.children[i];
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        let leaf_bounds = atlas_load_aabb(child_idx);
                        let t_leaf = intersect_aabb(*ray_local, leaf_bounds.min.xyz, leaf_bounds.max.xyz);
                        if (t_leaf >= ray_local.origin_and_tmin.w && t_leaf < ray_local.direction_and_tmax.w) {
                            // Triangle intersection in local space
                            let tri_id_local = u32(leaf_bounds.min.w);
                            let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
                            let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
                            let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];

                            let v0 = vertex_buffer[first_vertex + i0].position.xyz;
                            let v1 = vertex_buffer[first_vertex + i1].position.xyz;
                            let v2 = vertex_buffer[first_vertex + i2].position.xyz;

                            let t_tri = intersect_triangle(*ray_local, v0, v1, v2);
                            if (t_tri >= ray_local.origin_and_tmin.w && t_tri < ray_local.direction_and_tmax.w) {
                                return true;
                            }
                        }
                    } else {
                        let child_node = atlas_load_bvh4_node(child_idx);
                        let t_aabb_child = intersect_aabb(*ray_local, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child >= ray_local.origin_and_tmin.w && t_aabb_child < ray_local.direction_and_tmax.w) {
                            node_stack[stack_size] = child_idx;
                            stack_size = stack_size + 1u;
                        }
                    }
                }
            }
        }
    }

    return false;
}


@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }
    
    let pixel_index = gid.y * res.x + gid.x;
    var info = path_state[pixel_index];
    
    // Check if shadow ray is active (a > 0.0 means needs trace)
    if (info.shadow_radiance.a > 0.0) {
        var ray: Ray;
        ray.origin_and_tmin = info.shadow_origin;
        ray.direction_and_tmax = info.shadow_direction;
        let d = ray.direction_and_tmax.xyz;
        ray.inv_direction = vec4f(
            1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
            1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
            1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
            0.0
        );

        var original_ray = ray;

        for (var k = 0u; k < TLAS_CANDIDATES; k = k + 1u) {
            let tlas_result = trace_tlas(&ray);
            if (tlas_result.y == 0.0) {
                // Accumulate direct lighting contribution
                info.throughput += vec4f(info.shadow_radiance.rgb, 0.0);
                break;
            }

            let prim_store = u32(tlas_result.x);
            let mesh_id = mesh_asset_ids[prim_store];
            var entity_transform = entity_transforms[prim_store];

            var ray_local = build_local_ray(&original_ray, entity_transform.transform);
            let is_occluded = trace_blas_any_hit(&original_ray, &ray_local, entity_transform.transform, mesh_id);
            if (is_occluded) {
                break;
            }

            // Advance tmax to t_exit to skip this TLAS leaf entirely
            ray.origin_and_tmin.w = tlas_result.w + 0.001;
        }

        path_state[pixel_index] = info;
    }
}


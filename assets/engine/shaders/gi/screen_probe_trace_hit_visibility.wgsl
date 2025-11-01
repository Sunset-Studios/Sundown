// =============================================================================
// GI-1.0 Screen Probe Ray Tracing - Shadow Ray Pass
// - Separate pass for shadow visibility (reduces register pressure)
// - Based on path_trace_hit_visibility.wgsl for consistency
// =============================================================================
diagnostic(off,subgroup_uniformity);

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"
#include "gi/gi_common.wgsl"

const NODE_STACK_SIZE = 12;

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> probe_path_state: array<ProbePathState>;
@group(1) @binding(3) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(4) var<storage, read> tlas_bvh4_nodes: array<BVH4Node>;
@group(1) @binding(5) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(6) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> mesh_asset_ids: array<u32>;

fn trace_blas_any_hit(
    ray_world: ptr<function, Ray>,
    ray_local: ptr<function, Ray>,
    entity_transform: mat4x4f,
    mesh_asset_id: u32,
) -> bool {
    let mesh_directory_entry = atlas_load_directory_entry(mesh_asset_id);
    let bvh4_base = mesh_directory_entry.bvh4_base;
    let first_vertex = mesh_directory_entry.first_vertex;
    let first_index = mesh_directory_entry.first_index;

    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = bvh4_base;
    var stack_size = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = atlas_load_bvh4_node(node_idx);
        
        // Node already tested before push - no redundant AABB test here!
        if (stack_size < NODE_STACK_SIZE) {
            let leaf_mask = bitcast<u32>(node.min.w);

            for (var i = 0u; i < 4u; i = i + 1u) {
                if (node.children[i] < 0.0) { continue; }

                let child_idx = u32(node.children[i]);

                if (((leaf_mask >> i) & 1u) != 0u) { // Is leaf?
                    // Load vertex indices from co-located leaf data (cache-adjacent to node!)
                    let leaf_indices = atlas_load_bvh4_leaf_indices(node_idx, i);
                    let v0 = vertex_buffer[leaf_indices.x].position.xyz;
                    let v1 = vertex_buffer[leaf_indices.y].position.xyz;
                    let v2 = vertex_buffer[leaf_indices.z].position.xyz;
                    let t_tri = intersect_triangle(ray_local, v0, v1, v2);
                    if (t_tri >= ray_local.origin_and_tmin.w && t_tri < ray_local.direction_and_tmax.w) {
                        return true;
                    }
                } else {
                    // Only test AABB before pushing - guarantees single test per node
                    let child_node = atlas_load_bvh4_node(child_idx);
                    let t_aabb_child = intersect_aabb(ray_local, child_node.min.xyz, child_node.max.xyz);

                    if (t_aabb_child.x <= t_aabb_child.y && t_aabb_child.x >= ray_local.origin_and_tmin.w && t_aabb_child.x < ray_local.direction_and_tmax.w) {
                        node_stack[stack_size] = child_idx;
                        stack_size = stack_size + 1u;
                    }
                }
            }
        }
    }

    return false;
}

fn trace_hit_any(ray: ptr<function, Ray>) -> bool {
    var node_stack: array<u32, NODE_STACK_SIZE>;
    node_stack[0] = 0u;
    var stack_size = 1u;

    var current_ray = *ray;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let current_node = tlas_bvh4_nodes[node_idx];
        
        // Node already tested before push - no redundant AABB test here!
        if (stack_size < NODE_STACK_SIZE) {
            let leaf_mask = bitcast<u32>(current_node.min.w);

            for (var i = 0u; i < 4u; i = i + 1u) {
                if (current_node.children[i] < 0.0) { continue; }

                let child_idx = u32(current_node.children[i]);

                if (((leaf_mask >> i) & 1u) != 0u) { // Is leaf?
                    let leaf_bounds = tlas_bvh2_bounds[child_idx];
                    let t_leaf = intersect_aabb(&current_ray, leaf_bounds.min.xyz, leaf_bounds.max.xyz);

                    if (t_leaf.x <= t_leaf.y && max(t_leaf.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                        let prim_store = u32(leaf_bounds.min.w);
                        let mesh_id = mesh_asset_ids[prim_store];
                        let entity_transform = entity_transforms[prim_store];

                        var ray_local = build_local_ray(
                            &current_ray,
                            entity_transform.transform,
                            entity_transform.transpose_inverse_model_matrix
                        );
                        
                        // Wave-optimized shadow ray test (massive win here!)
                        if (trace_blas_any_hit(
                            &current_ray,
                            &ray_local,
                            entity_transform.transform,
                            mesh_id
                        )) {
                            return true;
                        } 
                    }
                } else {
                    // Only test AABB before pushing - guarantees single test per node
                    let child_node = tlas_bvh4_nodes[child_idx];
                    let t_aabb_child = intersect_aabb(&current_ray, child_node.min.xyz, child_node.max.xyz);

                    if (t_aabb_child.x <= t_aabb_child.y && max(t_aabb_child.x, current_ray.origin_and_tmin.w) < current_ray.direction_and_tmax.w) {
                        node_stack[stack_size] = child_idx;
                        stack_size = stack_size + 1u;
                    }
                }
            }
        }
    }

    return false;
}

@compute @workgroup_size(128, 1, 1)
fn cs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) lane_id: u32,
    @builtin(subgroup_size) warp_size: u32
) {
    let probe_count = u32(gi_params.total_screen_probes);
    let rays_per_probe = u32(gi_params.screen_ray_count);
    let total_rays = probe_count * rays_per_probe;
    
    if (gid.x >= total_rays) {
        return;
    }

    var ps = probe_path_state[gid.x];

    // ONLY process shadow rays in this pass
    if (ps.shadow_radiance.a > 0.0) {
        var ray: Ray;
        ray.origin_and_tmin = ps.shadow_origin;
        ray.direction_and_tmax = ps.shadow_direction;
        let d = ray.direction_and_tmax.xyz;
        ray.inv_direction = vec4f(
            1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
            1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
            1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
            0.0
        );
        
        if (!trace_hit_any(&ray)) {
            ps.state_u32.z = 1u; // No shadow hit - light is visible
        }
    }

    probe_path_state[gid.x] = ps;
}


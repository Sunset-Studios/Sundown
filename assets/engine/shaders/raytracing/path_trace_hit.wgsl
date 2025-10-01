// =============================================================================
// Path Tracer - Hit Pass (TLAS only)
// - Reads current path rays from `path_state`
// - Traverses TLAS to find closest leaf per pixel
// - Writes TLAS hit info (prim index, t_entry/t_exit) to `tlas_hits`
// - BLAS traversal is deferred to the Shade pass to respect storage limits
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
    origin_tmin: vec4<f32>,      // xyz = origin, w = tmin
    direction_tmax: vec4<f32>,   // xyz = direction, w = tmax
    normal: vec4<f32>,           // xyz = normal, w = unused
    throughput: vec4<f32>,       // rgb = throughput, a unused or prim_store
    state_u32: vec4<u32>,        // x=bounce, y=alive(0/1), z=mesh_id, w=tri_id
    hit_attr0: vec4<f32>,        // xyz = world_tangent, w = uv.x
    hit_attr1: vec4<f32>,        // xyz = world_bitangent, w = uv.y
    padding: vec4<f32>,
};

// Packed TLAS hit per pixel
// data.x = prim_index (u32-as-f32), data.y = hit_flag (0.0/1.0)
// data.z = t_entry, data.w = t_exit
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(3) var<storage, read> tlas_bvh4_nodes: array<BVH4Node>;
@group(1) @binding(4) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(5) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(6) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(7) var<storage, read> mesh_asset_ids: array<u32>;
@group(1) @binding(8) var output_tex: texture_storage_2d<rgba16float, write>; // for dimensions only

fn aabb_entry_exit_for_ray(ray: ptr<function, Ray>, bounds: AABB) -> vec2<f32> {
    var tmin_local = (*ray).origin_and_tmin.w;
    var tmax_local = (*ray).direction_and_tmax.w;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let inv_d = (*ray).inv_direction.xyz[i32(axis)];
        var t1 = (bounds.min.xyz[i32(axis)] - (*ray).origin_and_tmin.xyz[i32(axis)]) * inv_d;
        var t2 = (bounds.max.xyz[i32(axis)] - (*ray).origin_and_tmin.xyz[i32(axis)]) * inv_d;
        if (inv_d < 0.0) {
            let tmp = t1; t1 = t2; t2 = tmp;
        }
        tmin_local = max(tmin_local, t1);
        tmax_local = min(tmax_local, t2);
    }
    return vec2<f32>(tmin_local, tmax_local);
}

fn build_local_ray(ray_world: ptr<function, Ray>, entity_transform: mat4x4f) -> Ray {
    let inv_m = inverse4x4(entity_transform);
    let ro_world = (*ray_world).origin_and_tmin.xyz;
    let rd_world = (*ray_world).direction_and_tmax.xyz;

    var ray_local: Ray;
    ray_local.origin_and_tmin = vec4f((inv_m * vec4f(ro_world, 1.0)).xyz, (*ray_world).origin_and_tmin.w);
    ray_local.direction_and_tmax = vec4f((inv_m * vec4f(rd_world, 0.0)).xyz, (*ray_world).direction_and_tmax.w);

    let d = ray_local.direction_and_tmax.xyz;
    ray_local.inv_direction = vec4f(
        1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
        1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
        1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
        0.0
    );
    return ray_local;
}

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

fn trace_blas(ray_world: ptr<function, Ray>, ray_local: ptr<function, Ray>, entity_transform: mat4x4f, mesh_asset_id: u32) -> RayHit {
    let mesh_directory_entry = atlas_load_directory_entry(mesh_asset_id);
    let bvh4_base = mesh_directory_entry.bvh4_base;
    let first_vertex = mesh_directory_entry.first_vertex;
    let first_index = mesh_directory_entry.first_index;

    var hit: RayHit;
    hit.position_and_t = vec4f((*ray_world).origin_and_tmin.xyz, (*ray_world).direction_and_tmax.w);
    hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

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

        if (t_aabb >= ray_local.origin_and_tmin.w && t_aabb < hit.position_and_t.w) {
            if (stack_size < 32u) {
                var min_index = 0u;
                var min_t = 1e38;

                let leaf_mask = bitcast<u32>(node.min.w);

                for (var i = 0u; i < 4u; i = i + 1u) {
                    let child_raw = node.children[i];
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        let leaf_bounds = atlas_load_aabb(child_idx);
                        let t_leaf = intersect_aabb(*ray_local, leaf_bounds.min.xyz, leaf_bounds.max.xyz);
                        if (t_leaf >= ray_local.origin_and_tmin.w && t_leaf < hit.position_and_t.w) {
                            // Triangle intersection in local space
                            let tri_id_local = u32(leaf_bounds.min.w);
                            let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
                            let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
                            let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];

                            let v0 = vertex_buffer[first_vertex + i0].position.xyz;
                            let v1 = vertex_buffer[first_vertex + i1].position.xyz;
                            let v2 = vertex_buffer[first_vertex + i2].position.xyz;

                            let t_tri = intersect_triangle(*ray_local, v0, v1, v2);
                            if (t_tri >= ray_local.origin_and_tmin.w && t_tri < hit.position_and_t.w) {
                                let p_local = ray_local.origin_and_tmin.xyz + ray_local.direction_and_tmax.xyz * t_tri;
                                let p_world = (entity_transform * vec4f(p_local, 1.0)).xyz;
                                let n_local = normalize(cross(v1 - v0, v2 - v0));
                                let n_world = normalize((transpose(inverse4x4(entity_transform)) * vec4f(n_local, 0.0)).xyz);
                                hit.position_and_t = vec4<f32>(p_world, t_tri);
                                hit.normal_and_user_data = vec4<f32>(n_world, f32(tri_id_local));
                            }
                        }
                    } else {
                        let child_node = atlas_load_bvh4_node(child_idx);
                        let t_aabb_child = intersect_aabb(*ray_local, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child >= ray_local.origin_and_tmin.w && t_aabb_child < hit.position_and_t.w) {
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

    return hit;
}

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let pixel_index = gid.y * res.x + gid.x;

    // Read path state
    let ps = path_state[pixel_index];
    let alive = ps.state_u32.y;
    if (alive == 0u) {
        // mark as miss
        return;
    }

    var ray: Ray;
    ray.origin_and_tmin = ps.origin_tmin;
    ray.direction_and_tmax = ps.direction_tmax;
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
            break;
        }

        let prim_store = u32(tlas_result.x);
        let mesh_id = mesh_asset_ids[prim_store];
        var entity_transform = entity_transforms[prim_store];

        var ray_local = build_local_ray(&original_ray, entity_transform.transform);
        let blas_hit = trace_blas(&original_ray, &ray_local, entity_transform.transform, mesh_id);
        if (blas_hit.normal_and_user_data.w >= 0.0) {
            // Compute barycentric coordinates at the hit point in mesh-local space
            let tri_id_local = u32(blas_hit.normal_and_user_data.w);

            // Get triangle indices to fetch per-vertex attributes
            let mesh_entry = atlas_load_directory_entry(mesh_id);
            let first_vertex = mesh_entry.first_vertex;
            let first_index = mesh_entry.first_index;

            let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
            let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
            let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];
            let vertex0 = vertex_buffer[first_vertex + i0];
            let vertex1 = vertex_buffer[first_vertex + i1];
            let vertex2 = vertex_buffer[first_vertex + i2];

            let v0 = vertex0.position.xyz;
            let v1 = vertex1.position.xyz;
            let v2 = vertex2.position.xyz;
            let uv0 = vertex0.uv.xy;
            let uv1 = vertex1.uv.xy;
            let uv2 = vertex2.uv.xy;
            let n0 = vertex0.normal.xyz;
            let n1 = vertex1.normal.xyz;
            let n2 = vertex2.normal.xyz;
            let t0 = vertex0.tangent.xyz;
            let t1 = vertex1.tangent.xyz;
            let t2 = vertex2.tangent.xyz;
            let b0 = vertex0.bitangent.xyz;
            let b1 = vertex1.bitangent.xyz;
            let b2 = vertex2.bitangent.xyz;

            // Compute barycentric coordinates at the hit point in mesh-local space
            let t_tri = blas_hit.position_and_t.w;
            let p_local = ray_local.origin_and_tmin.xyz + ray_local.direction_and_tmax.xyz * t_tri;
            let p_world = (entity_transform.transform * vec4f(p_local, 1.0)).xyz;

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

            // Interpolate per-vertex attributes and transform T/B/N to world
            let uv_hit = uv0 * (1.0 - u_bc - v_bc) + uv1 * u_bc + uv2 * v_bc;

            let t_local = normalize(t0 * (1.0 - u_bc - v_bc) + t1 * u_bc + t2 * v_bc);
            let b_local = normalize(b0 * (1.0 - u_bc - v_bc) + b1 * u_bc + b2 * v_bc);
            let n_local = normalize(n0 * (1.0 - u_bc - v_bc) + n1 * u_bc + n2 * v_bc);

            var world_n = safe_normalize((entity_transform.transpose_inverse_model_matrix * vec4<f32>(n_local, 0.0)).xyz);
            let world_t = safe_normalize((entity_transform.transform * vec4<f32>(t_local, 0.0)).xyz);
            let world_b = safe_normalize((entity_transform.transform * vec4<f32>(b_local, 0.0)).xyz);

            // Ensure normal faces the incoming ray direction
            let ray_dir = ray.direction_and_tmax.xyz;
            world_n = select(world_n, -world_n, dot(world_n, ray_dir) > 0.0);

            // Store world-space hit position with small offset along normal to avoid self-intersection
            path_state[pixel_index].origin_tmin = vec4f(p_world + world_n * 0.001, 0.0001);
            // Store world-space hit normal directly
            path_state[pixel_index].direction_tmax = vec4f(ray_dir, f32(prim_store));
            // Store world-space hit normal directly
            path_state[pixel_index].normal = vec4f(world_n, 0.0);
            // Write world tangent
            path_state[pixel_index].hit_attr0 = vec4f(world_t, uv_hit.x);
            // Write world bitangent
            path_state[pixel_index].hit_attr1 = vec4f(world_b, uv_hit.y);
            // Mesh and triangle ids for shading
            path_state[pixel_index].state_u32.z = mesh_id;
            // Pack triangle id in state_u32.w
            path_state[pixel_index].state_u32.w = tri_id_local;

            break;
        }

        // Advance tmax to t_exit to skip this TLAS leaf entirely
		ray.origin_and_tmin.w = tlas_result.w + 0.001;
    }
}



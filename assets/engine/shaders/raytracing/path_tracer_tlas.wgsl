// ============================================================================
// TLAS Path Tracer Helpers (Candidate Collection)
// - Collects up to TLAS_CANDIDATES closest TLAS leaf hits per pixel
// - Strictly formatting and documentation improvements; no behavior changes
// ============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"

// ----------------------------------------------------------------------------
// Parameters and Per-Pixel Accumulation
// ----------------------------------------------------------------------------
struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    max_spp: u32,
};

struct PixelHitInfo {
    rng: f32,
    sample_count: f32,
    prim_id: f32,
    mesh_asset_id: f32,
    accum_color: vec4<f32>,
};

// Number of TLAS candidates to collect per pixel
const TLAS_CANDIDATES: u32 = 4u;

// ----------------------------------------------------------------------------
// Bindings (group 1)
// ----------------------------------------------------------------------------
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;          // Path tracer control params
@group(1) @binding(1) var<storage, read_write> pixel_info: array<PixelHitInfo>; // Per-pixel state
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;      // TLAS leaf bounds
@group(1) @binding(3) var<storage, read> tlas_bvh4_nodes: array<BVH4Node>;   // TLAS BVH4 nodes
@group(1) @binding(4) var<storage, read> mesh_asset_ids: array<u32>;         // Primitive -> mesh id
@group(1) @binding(5) var position_tex: texture_2d<f32>;                     // GBuffer world position
@group(1) @binding(6) var normal_tex: texture_2d<f32>;      

// ----------------------------------------------------------------------------
// Debug: generate a stable color per primitive id
// ----------------------------------------------------------------------------
// fn primitive_debug_color(mesh_asset_id: u32, prim_id: u32) -> vec3f {
//     let seed0 = hash(mesh_asset_id ^ (prim_id * 0x9e3779b9u));
//     let seed1 = hash(seed0 ^ 0x85ebca6bu);
//     let seed2 = hash(seed1 ^ 0xc2b2ae35u);
//     let r = f32(seed0) * one_over_float_max;
//     let g = f32(seed1) * one_over_float_max;
//     let b = f32(seed2) * one_over_float_max;
//     return vec3f(r, g, b);
// }

// ----------------------------------------------------------------------------
// Helper: Compute slab entry/exit for an AABB against a ray in TLAS space
// Returns vec2(t_entry, t_exit). No side effects; uses the ray's current tmin/tmax.
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Helper: Build a ray in mesh-local space from a world-space ray
// - Preserves tmin/tmax; returns fully-populated Ray with inv_direction
// ----------------------------------------------------------------------------
// fn build_local_ray(ray_world: ptr<function, Ray>, entity_transform: ptr<function, mat4x4f>) -> Ray {
//     let inv_m = inverse4x4(*entity_transform);
//     let ro_world = (*ray_world).origin_and_tmin.xyz;
//     let rd_world = (*ray_world).direction_and_tmax.xyz;

//     var ray_local: Ray;
//     ray_local.origin_and_tmin = vec4f((inv_m * vec4f(ro_world, 1.0)).xyz, (*ray_world).origin_and_tmin.w);
//     ray_local.direction_and_tmax = vec4f((inv_m * vec4f(rd_world, 0.0)).xyz, (*ray_world).direction_and_tmax.w);

//     let d = ray_local.direction_and_tmax.xyz;
//     ray_local.inv_direction = vec4f(
//         1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
//         1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
//         1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
//         0.0
//     );
//     return ray_local;
// }

// ----------------------------------------------------------------------------
// TLAS traversal: returns closest TLAS-leaf intersection (position_and_t.w is t)
// - normal_and_user_data.x holds t_exit for advancing the caller's ray tmin
// - normal_and_user_data.w holds primitive index (u32 as float)
// ----------------------------------------------------------------------------
fn trace_tlas(ray: ptr<function, Ray>) -> RayHit {
    var hit: RayHit;
    hit.position_and_t = vec4f(ray.origin_and_tmin.xyz, ray.direction_and_tmax.w);
    hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

    var node_stack: array<u32, 32>;
    node_stack[0] = 0u;
    var stack_size = 1u;

    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let current_node = tlas_bvh4_nodes[node_idx];
        let t_aabb = intersect_aabb(*ray, current_node.min.xyz, current_node.max.xyz);

        if (t_aabb >= ray.origin_and_tmin.w && t_aabb < hit.position_and_t.w) {
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
						if (t_entry <= t_exit && t_entry >= (*ray).origin_and_tmin.w && t_entry < hit.position_and_t.w) {
							let prim = u32(leaf_bounds.min.w);
							hit.position_and_t = vec4<f32>(
								(*ray).origin_and_tmin.xyz + (*ray).direction_and_tmax.xyz * t_entry,
								t_entry
							);
							// Store t_exit in x for caller to advance beyond leaf AABB
							hit.normal_and_user_data = vec4<f32>(t_exit, 0.0, 0.0, f32(prim));
						}
					} else {
                        let child_node = tlas_bvh4_nodes[child_idx];
                        let t_aabb_child = intersect_aabb(*ray, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child >= ray.origin_and_tmin.w && t_aabb_child < hit.position_and_t.w) {
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

// ----------------------------------------------------------------------------
// BLAS traversal: returns closest BLAS-leaf intersection (position_and_t.w is t)
// ----------------------------------------------------------------------------
// fn trace_blas(ray_world: ptr<function, Ray>, entity_transform: ptr<function, mat4x4f>, mesh_asset_id: u32) -> RayHit {
//     let mesh_directory_entry = blas_directory[mesh_asset_id];
//     let bvh2_base = mesh_directory_entry.bvh2_base;
//     let bvh4_base = mesh_directory_entry.bvh4_base;
//     let first_vertex = mesh_directory_entry.first_vertex;
//     let first_index = mesh_directory_entry.first_index;

//     // Build local ray from world ray and entity transform
//     var ray_local = build_local_ray(ray_world, entity_transform);

//     var hit: RayHit;
//     hit.position_and_t = vec4f((*ray_world).origin_and_tmin.xyz, (*ray_world).direction_and_tmax.w);
//     hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

//     var node_stack: array<u32, 32>;
//     node_stack[0] = bvh4_base;
//     var stack_size = 1u;
    
//     loop {
//         if (stack_size == 0u) { break; }
//         stack_size = stack_size - 1u;

//         var node_idx = node_stack[stack_size];
//         if (node_idx == INVALID_IDX) { continue; }

//         let node = blas_bvh4_nodes[node_idx];
//         let t_aabb = intersect_aabb(ray_local, node.min.xyz, node.max.xyz);

//         if (t_aabb >= ray_local.origin_and_tmin.w && t_aabb < hit.position_and_t.w) {
//             if (stack_size < 32u) {
//                 var min_index = 0u;
//                 var min_t = 1e38;

//                 let leaf_mask = bitcast<u32>(node.min.w);

//                 for (var i = 0u; i < 4u; i = i + 1u) {
//                     let child_raw = node.children[i];
//                     if (child_raw < 0.0) { continue; }

//                     let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
//                     let child_idx = u32(child_raw);

//                     if (is_leaf_child) {
//                         let leaf_bounds = blas_bvh2_bounds[child_idx];
//                         let t_leaf = intersect_aabb(ray_local, leaf_bounds.min.xyz, leaf_bounds.max.xyz);
//                         if (t_leaf >= ray_local.origin_and_tmin.w && t_leaf < hit.position_and_t.w) {
//                             // Triangle intersection in local space
//                             let tri_id_local = u32(leaf_bounds.min.w);
//                             let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
//                             let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
//                             let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];

//                             let v0 = vertex_buffer[first_vertex + i0].position.xyz;
//                             let v1 = vertex_buffer[first_vertex + i1].position.xyz;
//                             let v2 = vertex_buffer[first_vertex + i2].position.xyz;

//                             let t_tri = intersect_triangle(ray_local, v0, v1, v2);
//                             if (t_tri >= ray_local.origin_and_tmin.w && t_tri < hit.position_and_t.w) {
//                                 let p_local = ray_local.origin_and_tmin.xyz + ray_local.direction_and_tmax.xyz * t_tri;
//                                 let p_world = (*entity_transform * vec4f(p_local, 1.0)).xyz;
//                                 hit.position_and_t = vec4<f32>(p_world, t_tri);
//                                 hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, f32(tri_id_local));
//                             }
//                         }
//                     } else {
//                         let child_node = blas_bvh4_nodes[child_idx];
//                         let t_aabb_child = intersect_aabb(ray_local, child_node.min.xyz, child_node.max.xyz);

//                         if (t_aabb_child >= ray_local.origin_and_tmin.w && t_aabb_child < hit.position_and_t.w) {
//                             min_index = stack_size;
//                             min_t = t_aabb_child;
//                             node_stack[stack_size] = child_idx;
//                             stack_size = stack_size + 1u;
//                         }
//                     }
//                 }

//                 let tmp_node = node_stack[min_index];
//                 node_stack[min_index] = node_stack[stack_size - 1u];
//                 node_stack[stack_size - 1u] = tmp_node;
//             }
//         }
//     }

//     return hit;
// }

// ----------------------------------------------------------------------------
// Compute Kernel: Collect TLAS leaf candidates per pixel
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(position_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let pixel_index = (gid.y * res.x + gid.x) * TLAS_CANDIDATES;

    if (pt_params.reset_accum_flag != 0u) {
        let rng = hash(pixel_index ^ u32(frame_info.frame_index));
        for (var k = 0u; k < TLAS_CANDIDATES; k = k + 1u) {
            pixel_info[pixel_index + k].rng = f32(rng);
            pixel_info[pixel_index + k].sample_count = 0.0;
            pixel_info[pixel_index + k].prim_id = -1.0;
            pixel_info[pixel_index + k].mesh_asset_id = -1.0;
            pixel_info[pixel_index + k].accum_color = vec4f(0.0);
        }
        return;
    }

    for (var k = 0u; k < TLAS_CANDIDATES; k = k + 1u) {
        var rng = u32(pixel_info[pixel_index + k].rng);
        if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
        else           { rng = random_seed(rng); }
        pixel_info[pixel_index + k].rng = f32(rng);
    }

    let world_pos = textureLoad(position_tex, vec2<i32>(gid.xy), 0).xyz;
    let world_norm = textureLoad(normal_tex, vec2<i32>(gid.xy), 0).xyz;

	// Use the surface normal as a stable probe direction for TLAS sampling
	var ray_dir = safe_normalize(world_norm);
    var ray: Ray;
    ray.origin_and_tmin = vec4f(world_pos, 0.001);
    ray.direction_and_tmax = vec4f(ray_dir, 1e30);
    ray.inv_direction = vec4f(1.0 / max(abs(ray_dir.x), 1e-8) * select(1.0, -1.0, ray_dir.x < 0.0),
                              1.0 / max(abs(ray_dir.y), 1e-8) * select(1.0, -1.0, ray_dir.y < 0.0),
                              1.0 / max(abs(ray_dir.z), 1e-8) * select(1.0, -1.0, ray_dir.z < 0.0),
                              0.0);

    var write_count = 0u;
	for (var k = 0u; k < TLAS_CANDIDATES; k = k + 1u) {
        let hit = trace_tlas(&ray);
        let has_hit = hit.normal_and_user_data.w >= 0.0;
        if (!has_hit) {
            break;
        }
        let prim_store = u32(hit.normal_and_user_data.w);
        let mesh_asset_id = mesh_asset_ids[prim_store];
        pixel_info[pixel_index + k].prim_id = f32(prim_store);
        pixel_info[pixel_index + k].mesh_asset_id = f32(mesh_asset_id);
        write_count = write_count + 1u;
		// Advance tmin beyond the TLAS leaf's exit to avoid reselecting same leaf
		let t_exit = hit.normal_and_user_data.x;
		ray.origin_and_tmin.w = t_exit + 1e-4;
    }

    // Fill the rest with invalids
    for (var k = write_count; k < TLAS_CANDIDATES; k = k + 1u) {
        pixel_info[pixel_index + k].prim_id = -1.0;
        pixel_info[pixel_index + k].mesh_asset_id = -1.0;
    }
}

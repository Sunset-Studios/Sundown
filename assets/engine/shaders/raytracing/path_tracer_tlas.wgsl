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

struct PixelInfo {
    position_and_rng: vec4<f32>,
    normal_and_sample_count: vec4<f32>,
    accum_color: vec4<f32>,
};

// ----------------------------------------------------------------------------
// TLAS Hit Output
//  - t_bits:    ieee-754 bits of t (bitcast from f32)
//  - prim_id:   TLAS leaf primitive index (u32)
//  - mesh_asset_id: mesh id for the primitive
// ----------------------------------------------------------------------------
struct TlasHit {
    t_bits: u32,
    prim_id: u32,
    mesh_asset_id: u32,
    pad1: u32,
};

// Number of TLAS candidates to collect per pixel
const TLAS_CANDIDATES: u32 = 4u;

// ----------------------------------------------------------------------------
// Bindings (group 1)
// ----------------------------------------------------------------------------
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;          // Path tracer control params
@group(1) @binding(1) var<storage, read_write> pixel_info: array<PixelInfo>; // Per-pixel state
@group(1) @binding(2) var<storage, read> tlas_bvh2_bounds: array<AABB>;      // TLAS leaf bounds
@group(1) @binding(3) var<storage, read> tlas_bvh4_nodes: array<BVH4Node>;   // TLAS BVH4 nodes
@group(1) @binding(4) var<storage, read_write> tlas_hits: array<TlasHit>;    // Output candidates
@group(1) @binding(5) var<storage, read> mesh_asset_ids: array<u32>;         // Primitive -> mesh id
@group(1) @binding(6) var position_tex: texture_2d<f32>;                     // GBuffer world position
@group(1) @binding(7) var normal_tex: texture_2d<f32>;                        // GBuffer world normal

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
// Compute Kernel: Collect TLAS leaf candidates per pixel
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(position_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let pixel_index = gid.y * res.x + gid.x;

    if (pt_params.reset_accum_flag != 0u) {
        let rng = hash(pixel_index ^ u32(frame_info.frame_index));
        pixel_info[pixel_index].position_and_rng = vec4f(0.0, 0.0, 0.0, f32(rng));
        pixel_info[pixel_index].normal_and_sample_count = vec4f(0.0, 0.0, 0.0, 0.0);
        pixel_info[pixel_index].accum_color = vec4f(0.0);
        let base = pixel_index * TLAS_CANDIDATES;
        for (var k = 0u; k < TLAS_CANDIDATES; k = k + 1u) {
            tlas_hits[base + k] = TlasHit(bitcast<u32>(-1.0), INVALID_IDX, 0u, 0u);
        }
        return;
    }

    var rng = u32(pixel_info[pixel_index].position_and_rng.w);
    if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }

    let world_pos = textureLoad(position_tex, vec2<i32>(gid.xy), 0).xyz;
    let world_norm = textureLoad(normal_tex, vec2<i32>(gid.xy), 0).xyz;

    pixel_info[pixel_index].position_and_rng = vec4f(world_pos, f32(rng));
    pixel_info[pixel_index].normal_and_sample_count = vec4f(world_norm, pixel_info[pixel_index].normal_and_sample_count.w);

    rng = random_seed(rng);
    pixel_info[pixel_index].position_and_rng.w = f32(rng);

	// Use the surface normal as a stable probe direction for TLAS sampling
	var ray_dir = safe_normalize(world_norm);
    var ray: Ray;
    ray.origin_and_tmin = vec4f(world_pos, 0.001);
    ray.direction_and_tmax = vec4f(ray_dir, 1e30);
    ray.inv_direction = vec4f(1.0 / max(abs(ray_dir.x), 1e-8) * select(1.0, -1.0, ray_dir.x < 0.0),
                              1.0 / max(abs(ray_dir.y), 1e-8) * select(1.0, -1.0, ray_dir.y < 0.0),
                              1.0 / max(abs(ray_dir.z), 1e-8) * select(1.0, -1.0, ray_dir.z < 0.0),
                              0.0);

    let base = pixel_index * TLAS_CANDIDATES;
    var write_count = 0u;
	for (var k = 0u; k < TLAS_CANDIDATES; k = k + 1u) {
        let hit = trace_tlas(&ray);
        let has_hit = hit.normal_and_user_data.w >= 0.0;
        if (!has_hit) {
            break;
        }
        let t_store = hit.position_and_t.w;
        let prim_store = u32(hit.normal_and_user_data.w);
        let mesh_asset_id = mesh_asset_ids[prim_store];
        tlas_hits[base + k] = TlasHit(bitcast<u32>(t_store), prim_store, mesh_asset_id, 0u);
        write_count = write_count + 1u;
		// Advance tmin beyond the TLAS leaf's exit to avoid reselecting same leaf
		let t_exit = hit.normal_and_user_data.x;
		ray.origin_and_tmin.w = t_exit + 1e-4;
    }
    // Fill the rest with invalids
    for (var k2 = write_count; k2 < TLAS_CANDIDATES; k2 = k2 + 1u) {
        tlas_hits[base + k2] = TlasHit(bitcast<u32>(-1.0), INVALID_IDX, 0u, 0u);
    }
}

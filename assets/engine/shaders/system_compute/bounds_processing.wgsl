#include "common.wgsl"
#include "acceleration_common.wgsl"

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------ 

const bounds_padding = 1.0;

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read_write> entity_flags: array<u32>;
@group(1) @binding(2) var<storage, read_write> aabb_bounds: array<AABB>;
@group(1) @binding(3) var<storage, read_write> scene_aabb: array<atomic<u32>, 8>;
@group(1) @binding(4) var<storage, read> entity_mesh_ids: array<u32>;
@group(1) @binding(5) var<storage, read> mesh_local_bounds: array<AABB>;

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

fn atomic_min_f32(target_val: ptr<storage, atomic<u32>, read_write>, value: f32) {
	var old_bits = atomicLoad(target_val);
	loop {
		let old_f = bitcast<f32>(old_bits);
		let new_f = min(old_f, value);
		if (new_f == old_f) { break; }
		let new_bits = bitcast<u32>(new_f);
		let result = atomicCompareExchangeWeak(target_val, old_bits, new_bits);
		if (result.exchanged) { break; }
		old_bits = result.old_value;
	}
}

fn atomic_max_f32(target_val: ptr<storage, atomic<u32>, read_write>, value: f32) {
	var old_bits = atomicLoad(target_val);
	loop {
		let old_f = bitcast<f32>(old_bits);
		let new_f = max(old_f, value);
		if (new_f == old_f) { break; }
		let new_bits = bitcast<u32>(new_f);
		let result = atomicCompareExchangeWeak(target_val, old_bits, new_bits);
		if (result.exchanged) { break; }
		old_bits = result.old_value;
	}
}

var<workgroup> wg_min_points: array<vec3<f32>, 256>;
var<workgroup> wg_max_points: array<vec3<f32>, 256>;
var<workgroup> wg_valid_counts: array<u32, 256>;
var<workgroup> wg_scene_min: vec3<f32>;
var<workgroup> wg_scene_max: vec3<f32>;

@compute @workgroup_size(256)
fn cs(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
#if HAS_SUBGROUPS
  @builtin(subgroup_invocation_id) sid: u32,
  @builtin(subgroup_size) ss: u32
#endif
) {
	let idx = global_id.x;
	let lid = local_id.x;

	let num_rows = arrayLength(&entity_transforms);

	var min_point = vec3<f32>(pos_inf, pos_inf, pos_inf);
	var max_point = vec3<f32>(neg_inf, neg_inf, neg_inf);

	var min_node_bounds = vec4<f32>(0.0, 0.0, 0.0, -1.0);
	var max_node_bounds = vec4<f32>(0.0, 0.0, 0.0, -1.0);

    let entity_id_offset = idx;
    let transform = entity_transforms[entity_id_offset].transform;
    let position = transform[3].xyz;
	let scale = vec3<f32>(length(transform[0].xyz), length(transform[1].xyz), length(transform[2].xyz));

	let is_active = idx < num_rows && transform[3].w != 0.0;
	if (is_active) {
		// Prefer mesh-local bounds via shared mesh data if mesh id is valid; otherwise fall back to scale-based cube.
		let mesh_id = entity_mesh_ids[entity_id_offset];
		let has_mesh_bounds = mesh_id != INVALID_IDX;

		if (has_mesh_bounds) {
			let mesh_min_local = mesh_local_bounds[mesh_id].min.xyz;
			let mesh_max_local = mesh_local_bounds[mesh_id].max.xyz;
			let center_local = 0.5 * (mesh_min_local + mesh_max_local);
			let half_local = 0.5 * (mesh_max_local - mesh_min_local);

			let world_center = (transform * vec4<f32>(center_local, 1.0)).xyz;
			let r0 = abs(transform[0].xyz) * 0.5;
			let r1 = abs(transform[1].xyz) * 0.5;
			let r2 = abs(transform[2].xyz) * 0.5;
			let world_half = vec3<f32>(
				dot(r0, half_local),
				dot(r1, half_local),
				dot(r2, half_local),
			);
			let padding = world_half * bounds_padding;
			min_point = world_center - (world_half + padding);
			max_point = world_center + (world_half + padding);
		} else {
			let half_size = vec3<f32>(
			  abs(scale[0]) * 0.5,
			  abs(scale[1]) * 0.5,
			  abs(scale[2]) * 0.5,
			);
			let padding = vec3<f32>(
			  half_size[0] * bounds_padding,
			  half_size[1] * bounds_padding,
			  half_size[2] * bounds_padding,
			);
			min_point = vec3<f32>(
			  position[0] - half_size[0] - padding[0],
			  position[1] - half_size[1] - padding[1],
			  position[2] - half_size[2] - padding[2],
			);
			max_point = vec3<f32>(
			  position[0] + half_size[0] + padding[0],
			  position[1] + half_size[1] + padding[1],
			  position[2] + half_size[2] + padding[2],
			);
		}

        min_node_bounds = vec4<f32>(min_point, f32(mesh_id));
        max_node_bounds = vec4<f32>(max_point, -1.0 - f32(entity_id_offset));

		entity_flags[entity_id_offset] |= EF_AABB_DIRTY;
	}

	aabb_bounds[entity_id_offset].min = min_node_bounds;
	aabb_bounds[entity_id_offset].max = max_node_bounds;

	wg_min_points[lid] = min_point;
	wg_max_points[lid] = max_point;
	wg_valid_counts[lid] = select(0u, 1u, is_active);
	workgroupBarrier();

	if (lid == 0u) {
		wg_scene_min = vec3<f32>(
			bitcast<f32>(atomicLoad(&scene_aabb[0])),
			bitcast<f32>(atomicLoad(&scene_aabb[1])),
			bitcast<f32>(atomicLoad(&scene_aabb[2]))
		);
		wg_scene_max = vec3<f32>(
			bitcast<f32>(atomicLoad(&scene_aabb[4])),
			bitcast<f32>(atomicLoad(&scene_aabb[5])),
			bitcast<f32>(atomicLoad(&scene_aabb[6]))
		);
		wg_min_points[0u] = min(wg_min_points[0u], wg_scene_min);
		wg_max_points[0u] = max(wg_max_points[0u], wg_scene_max);
	}
	workgroupBarrier();

#if HAS_SUBGROUPS
  let warp_ctx = make_warp_ctx(lid, lid, ss);
#else
  let warp_ctx = make_warp_ctx(lid, lid, LOGICAL_WARP_SIZE);
#endif

	let sub_min = vec3<f32>(
		warp_min_f32(warp_ctx, wg_min_points[lid].x),
		warp_min_f32(warp_ctx, wg_min_points[lid].y),
		warp_min_f32(warp_ctx, wg_min_points[lid].z)
	);
	let sub_max = vec3<f32>(
		warp_max_f32(warp_ctx, wg_max_points[lid].x),
		warp_max_f32(warp_ctx, wg_max_points[lid].y),
		warp_max_f32(warp_ctx, wg_max_points[lid].z)
	);
	let sub_valid: u32 = warp_reduce_add_u32(warp_ctx, wg_valid_counts[lid]);

	if (warp_ctx.lane_id == 0u) {
		wg_min_points[warp_ctx.warp_id] = sub_min;
		wg_max_points[warp_ctx.warp_id] = sub_max;
		wg_valid_counts[warp_ctx.warp_id] = sub_valid;
	}
	workgroupBarrier();

	if (lid == 0u) {
		var gmin = wg_min_points[0u];
		var gmax = wg_max_points[0u];
		var gvalid = wg_valid_counts[0u];

		var i: u32 = 1u;
		loop {
			if (i >= warp_ctx.warp_size) { break; }
			gmin = min(gmin, wg_min_points[i]);
			gmax = max(gmax, wg_max_points[i]);
			gvalid = gvalid + wg_valid_counts[i];
			i = i + 1u;
		}

		if (gvalid > 0u) {
			atomic_min_f32(&scene_aabb[0], gmin.x);
			atomic_min_f32(&scene_aabb[1], gmin.y);
			atomic_min_f32(&scene_aabb[2], gmin.z);
			atomic_max_f32(&scene_aabb[4], gmax.x);
			atomic_max_f32(&scene_aabb[5], gmax.y);
			atomic_max_f32(&scene_aabb[6], gmax.z);
		}
	}
}
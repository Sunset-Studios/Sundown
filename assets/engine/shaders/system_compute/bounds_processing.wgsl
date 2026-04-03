#include "common.wgsl"
#include "acceleration_common.wgsl"

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------ 

const bounds_padding = 0.01;

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read_write> entity_flags: array<u32>;
@group(1) @binding(2) var<storage, read_write> aabb_bounds: array<AABB>;
@group(1) @binding(3) var<storage, read_write> scene_aabb: AABB;
@group(1) @binding(4) var<storage, read> entity_mesh_ids: array<u32>;
@group(1) @binding(5) var<storage, read> mesh_local_bounds: array<AABB>;

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
	let idx = global_id.x;

	let num_rows = arrayLength(&entity_transforms);
	let in_bounds = idx < num_rows;

	var min_point = vec3<f32>(pos_inf, pos_inf, pos_inf);
	var max_point = vec3<f32>(neg_inf, neg_inf, neg_inf);

	var min_node_bounds = vec4<f32>(0.0, 0.0, 0.0, -1.0);
	var max_node_bounds = vec4<f32>(0.0, 0.0, 0.0, -1.0);

    let entity_id_offset = idx;
	var transform = identity_matrix;
	var flags = 0u;
	var mesh_id = INVALID_IDX;

	if (in_bounds) {
		transform = entity_transforms[entity_id_offset].transform;
		flags = entity_flags[entity_id_offset];
		if ((flags & EF_HAS_MESH) != 0u) {
			mesh_id = entity_mesh_ids[entity_id_offset];
		}
	}

	let has_mesh_bounds = (flags & EF_HAS_MESH) != 0u && mesh_id != INVALID_IDX;
	let is_active = in_bounds && transform[3].w != 0.0 && has_mesh_bounds;
	if (is_active) {
		let mesh_min_local = mesh_local_bounds[mesh_id].min.xyz;
		let mesh_max_local = mesh_local_bounds[mesh_id].max.xyz;
		let center_local = 0.5 * (mesh_min_local + mesh_max_local);
		let half_local = 0.5 * (mesh_max_local - mesh_min_local);

		let world_center = (transform * vec4<f32>(center_local, 1.0)).xyz;
		let c0 = abs(transform[0].xyz);
		let c1 = abs(transform[1].xyz);
		let c2 = abs(transform[2].xyz);
		let world_half = vec3<f32>(
			c0.x * half_local.x + c1.x * half_local.y + c2.x * half_local.z,
			c0.y * half_local.x + c1.y * half_local.y + c2.y * half_local.z,
			c0.z * half_local.x + c1.z * half_local.y + c2.z * half_local.z,
		);
		let padding = world_half * bounds_padding;
		min_point = world_center - (world_half + padding);
		max_point = world_center + (world_half + padding);

        min_node_bounds = vec4<f32>(min_point, f32(mesh_id));
        max_node_bounds = vec4<f32>(max_point, -1.0 - f32(entity_id_offset));

		entity_flags[entity_id_offset] |= EF_AABB_DIRTY;
	}

	if (in_bounds) {
		aabb_bounds[entity_id_offset].min = min_node_bounds;
		aabb_bounds[entity_id_offset].max = max_node_bounds;
	}

}

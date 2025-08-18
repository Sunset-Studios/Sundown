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
@group(1) @binding(3) var<storage, read> entity_aabb_node_indices: array<u32>;
@group(1) @binding(4) var<storage, read_write> scene_aabb: array<atomic<u32>, 8>;

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

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= arrayLength(&entity_aabb_node_indices)) {
        return;
    }

    let entity_id_offset = get_entity_row(global_id.x);
    let node_index = entity_aabb_node_indices[entity_id_offset];

    // if there's no valid AABB node, skip processing entirely
    if (node_index == 0u) {
        return;
    }

    // Get the entity's world transform
    let transform = entity_transforms[entity_id_offset].transform;
    let position = transform[3].xyz;
    let scale = vec3f(length(transform[0].xyz), length(transform[1].xyz), length(transform[2].xyz));

    // Calculate bounds based on position and scale
    // This is a simple axis-aligned box, but could be more sophisticated
    // based on the entity's mesh or collider
    let half_size = vec3f(
      abs(scale[0]) * 0.5,
      abs(scale[1]) * 0.5,
      abs(scale[2]) * 0.5,
    );

    // Add padding
    let padding = vec3f(
      half_size[0] * bounds_padding,
      half_size[1] * bounds_padding,
      half_size[2] * bounds_padding,
    );

    let min_point = vec3f(
      position[0] - half_size[0] - padding[0],
      position[1] - half_size[1] - padding[1],
      position[2] - half_size[2] - padding[2],
    );

    let max_point = vec3f(
      position[0] + half_size[0] + padding[0],
      position[1] + half_size[1] + padding[1],
      position[2] + half_size[2] + padding[2],
    );

    // write to full array for other GPU consumers
    aabb_bounds[node_index].min = vec4f(min_point, f32(entity_id_offset));
    aabb_bounds[node_index].max = vec4f(max_point, 1.0 /* can be used for other purposes */);

	  atomic_min_f32(&scene_aabb[0], min_point.x);
	  atomic_min_f32(&scene_aabb[1], min_point.y);
	  atomic_min_f32(&scene_aabb[2], min_point.z);
	  atomic_max_f32(&scene_aabb[4], max_point.x);
	  atomic_max_f32(&scene_aabb[5], max_point.y);
	  atomic_max_f32(&scene_aabb[6], max_point.z);

    entity_flags[entity_id_offset] |= EF_AABB_DIRTY;
}
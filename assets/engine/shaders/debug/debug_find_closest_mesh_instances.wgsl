#include "common.wgsl"
#include "acceleration_common.wgsl"

// ============================================================================
// CLOSEST MESH INSTANCE FINDER - Compact BLAS Debug Preprocessing 
// ============================================================================
// 
// Finds the closest instance per mesh asset by storing the resolved entity
// index of the closest instance in a compact per-mesh array.
//
// ============================================================================

@group(1) @binding(0) var<storage, read_write> closest_entities_per_mesh: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> closest_distances_per_mesh: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(3) var<storage, read> visible_object_instances: array<i32>;
@group(1) @binding(4) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(5) var<storage, read> mesh_asset_ids: array<u32>;
@group(1) @binding(6) var<storage, read> entity_index_lookup: array<u32>;

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let instance_idx = gid.x;
    
    // Bounds check for visible_object_instances array
    if (instance_idx >= arrayLength(&visible_object_instances)) { 
        return; 
    }
    
    let instance_index = visible_object_instances[instance_idx];
    if (instance_index < 0) { 
        return; 
    }
    
    // Bounds check for object_instances array
    if (u32(instance_index) >= arrayLength(&object_instances)) { 
        return; 
    }
    
    let entity_resolved = entity_index_lookup[get_entity_row(object_instances[instance_index].row)];
    
    // Bounds check for mesh_asset_ids and entity_transforms arrays
    if (entity_resolved >= arrayLength(&mesh_asset_ids) || 
        entity_resolved >= arrayLength(&entity_transforms)) { 
        return; 
    }
    
    let mesh_asset_id = mesh_asset_ids[entity_resolved];
    
    // Bounds check for output arrays
    if (mesh_asset_id >= arrayLength(&closest_entities_per_mesh) || 
        mesh_asset_id >= arrayLength(&closest_distances_per_mesh)) { 
        return; 
    }
    
    let instance_world_pos = entity_transforms[entity_resolved].transform[3].xyz;
    let camera_position = view_buffer[u32(frame_info.view_index)].view_position;
    
    // Calculate distance to camera
    let pos = instance_world_pos - camera_position.xyz;
    let distance_squared = dot(pos, pos);
    
    // Convert distance to u32 bits for atomic storage (preserves precision)
    let new_distance_bits = bitcast<u32>(distance_squared);
    
    // Atomic compare-and-swap loop to find minimum distance
    var success = false;
    var attempts = 0u;
    while (!success && attempts < 8u) {
        attempts += 1u;
        
        let current_distance_bits = atomicLoad(&closest_distances_per_mesh[mesh_asset_id]);
        let current_distance = bitcast<f32>(current_distance_bits);
        
        // If we're closer, try to update both distance and entity atomically
        if (distance_squared < current_distance) {
            let old_distance_bits = atomicCompareExchangeWeak(
                &closest_distances_per_mesh[mesh_asset_id],
                current_distance_bits,
                new_distance_bits
            );
            
            // Only update entity if we successfully updated the distance
            if (old_distance_bits.exchanged) {
                atomicStore(&closest_entities_per_mesh[mesh_asset_id], entity_resolved);
                success = true;
            }
            // If exchange failed, retry (another thread may have updated)
        } else {
            // We're not closer, no need to continue
            success = true;
        }
    }
}

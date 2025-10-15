// =============================================================================
// GI-1.0 World Cache - Common Functions
// - Spatial hash-based radiance cache for secondary bounces
// - Based on GI-1.0 paper section 2.2
// =============================================================================

// World cache cell - stores outgoing radiance at secondary vertices
struct WorldCacheCell {
    position_frame: vec4<f32>,      // xyz = world position, w = frame stamp
    normal_count: vec4<f32>,        // xyz = normal, w = sample count
    radiance_w: vec4<f32>,          // xyz = radiance, w = confidence weight
    data: vec4<u32>,                // x = hash_key, y = occupied, z,w = padding
};

// Spatial hash function - maps 3D position to hash table index
// Using the approach from GI-1.0 paper
fn spatial_hash(position: vec3<f32>, cell_size: f32) -> u32 {
    // Quantize position to grid
    let grid_pos = vec3<i32>(floor(position / cell_size));
    
    // Large primes for good distribution
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    
    let h = (u32(grid_pos.x) * p1) ^ (u32(grid_pos.y) * p2) ^ (u32(grid_pos.z) * p3);
    
    return h;
}

// Compute grid cell from position
fn get_grid_cell(position: vec3<f32>, cell_size: f32) -> vec3<i32> {
    return vec3<i32>(floor(position / cell_size));
}

// Check if two positions are in the same grid cell
fn same_grid_cell(pos_a: vec3<f32>, pos_b: vec3<f32>, cell_size: f32) -> bool {
    let cell_a = get_grid_cell(pos_a, cell_size);
    let cell_b = get_grid_cell(pos_b, cell_size);
    return all(cell_a == cell_b);
}

// Distance-based weight for cache lookup
fn cache_lookup_weight(
    query_pos: vec3<f32>,
    cache_pos: vec3<f32>,
    query_normal: vec3<f32>,
    cache_normal: vec3<f32>,
    cell_size: f32
) -> f32 {
    // Distance falloff
    let dist = length(query_pos - cache_pos);
    let dist_weight = max(0.0, 1.0 - dist / cell_size);
    
    // Normal similarity
    let normal_dot = max(0.0, dot(query_normal, cache_normal));
    let normal_weight = normal_dot * normal_dot; // Squared for sharper falloff
    
    return dist_weight * normal_weight;
}

// Probe world cache with position and normal
fn query_world_cache(
    position: vec3<f32>,
    normal: vec3<f32>,
    cache: ptr<storage, array<WorldCacheCell>, read>,
    cache_size: u32,
    cell_size: f32
) -> vec3<f32> {
    let hash_key = spatial_hash(position, cell_size);
    let base_index = hash_key % cache_size;
    
    var total_radiance = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    // Probe nearby cells (linear probing with max search distance)
    let max_probes = 8u;
    for (var i = 0u; i < max_probes; i = i + 1u) {
        let probe_index = (base_index + i) % cache_size;
        let cell = (*cache)[probe_index];
        
        // Check if cell is occupied and valid
        if (cell.data.y == 0u) {
            continue;
        }
        
        // Check if hash matches
        if (cell.data.x != hash_key) {
            continue;
        }
        
        // Compute weight based on distance and normal similarity
        let weight = cache_lookup_weight(
            position,
            cell.position_frame.xyz,
            normal,
            cell.normal_count.xyz,
            cell_size
        );
        
        if (weight > 0.001) {
            total_radiance += cell.radiance_w.xyz * weight;
            total_weight += weight;
        }
    }
    
    // Normalize by total weight
    if (total_weight > 0.001) {
        return total_radiance / total_weight;
    }
    
    return vec3<f32>(0.0);
}

// Insert or update entry in world cache
fn insert_world_cache(
    position: vec3<f32>,
    normal: vec3<f32>,
    radiance: vec3<f32>,
    cache: ptr<storage, array<WorldCacheCell>, read_write>,
    cache_size: u32,
    cell_size: f32,
    frame_index: u32
) -> bool {
    let hash_key = spatial_hash(position, cell_size);
    let base_index = hash_key % cache_size;
    
    // Try to find existing entry or empty slot
    let max_probes = 16u;
    for (var i = 0u; i < max_probes; i = i + 1u) {
        let probe_index = (base_index + i) % cache_size;
        
        // Check if slot is empty
        if ((*cache)[probe_index].data.y == 0u) {
            // Claim this slot
            (*cache)[probe_index].position_frame = vec4<f32>(position, f32(frame_index));
            (*cache)[probe_index].normal_count = vec4<f32>(normal, 1.0);
            (*cache)[probe_index].radiance_w = vec4<f32>(radiance, 1.0);
            (*cache)[probe_index].data = vec4<u32>(hash_key, 1u, 0u, 0u);
            return true;
        }
        
        // Check if this is the same cell (update existing)
        if ((*cache)[probe_index].data.x == hash_key &&
            same_grid_cell(position, (*cache)[probe_index].position_frame.xyz, cell_size)) {
            // Update with exponential moving average
            let alpha = 0.1; // Blend factor
            let old_radiance = (*cache)[probe_index].radiance_w.xyz;
            let new_radiance = mix(old_radiance, radiance, alpha);
            
            (*cache)[probe_index].radiance_w = vec4<f32>(new_radiance, (*cache)[probe_index].radiance_w.w);
            (*cache)[probe_index].normal_count.w += 1.0; // Increment sample count
            (*cache)[probe_index].position_frame.w = f32(frame_index); // Update frame stamp
            return true;
        }
    }
    
    // Failed to find slot (cache is full or too many collisions)
    return false;
}


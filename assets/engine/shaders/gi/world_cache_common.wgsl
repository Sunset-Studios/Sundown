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
// Using the approach from GI-1.0 paper with proper negative coordinate handling
fn spatial_hash(position: vec3<f32>, cell_size: f32) -> u32 {
    // Quantize position to grid
    let grid_pos = vec3<i32>(floor(position / cell_size));
    
    // Convert signed integers to unsigned using bitwise reinterpretation
    // This ensures negative coordinates hash correctly without huge unsigned values
    let ux = bitcast<u32>(grid_pos.x);
    let uy = bitcast<u32>(grid_pos.y);
    let uz = bitcast<u32>(grid_pos.z);
    
    // Large primes for good distribution
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    
    let h = (ux * p1) ^ (uy * p2) ^ (uz * p3);
    
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

// Query world cache for entire grid cell
// Returns uniform radiance across entire cell without distance falloff
fn query_world_cache_cell(
    position: vec3<f32>,
    cache_size: u32,
    cell_size: f32
) -> vec3<f32> {
    let hash_key = spatial_hash(position, cell_size);
    let base_index = hash_key % cache_size;
    
    var total_radiance = vec3<f32>(0.0);
    var count = 0.0;
    
    // Probe linear chain to find all entries in this grid cell
    let max_probes = 8u;
    for (var i = 0u; i < max_probes; i = i + 1u) {
        let probe_index = (base_index + i) % cache_size;
        let cell = world_cache[probe_index];
        
        // Check if cell is occupied
        if (cell.data.y == 0u) {
            continue;
        }
        
        // Check if hash matches
        if (cell.data.x != hash_key) {
            continue;
        }
        
        // Check if in same grid cell
        if (same_grid_cell(position, cell.position_frame.xyz, cell_size)) {
            total_radiance += cell.radiance_w.xyz;
            count += 1.0;
        }
    }
    
    // Return average radiance for all entries in this cell
    if (count > 0.0) {
        return total_radiance / count;
    }
    
    return vec3<f32>(0.0);
}

// =============================================================================
// Eviction Priority Scoring
// Lower score = higher priority to evict
// Combines: age (older = evict), distance (farther = evict), confidence (lower = evict)
// =============================================================================
fn compute_eviction_score(
    cell: WorldCacheCell,
    camera_position: vec3<f32>,
    current_frame: u32,
    max_age_frames: f32
) -> f32 {
    // Age factor: normalized age [0, 1], higher = older
    let cell_frame = u32(cell.position_frame.w);
    let age = f32(current_frame - cell_frame);
    let age_factor = min(age / max_age_frames, 1.0);
    
    // Distance factor: normalized distance from camera [0, 1]
    // Clamp max distance to prevent overflow
    let distance = length(cell.position_frame.xyz - camera_position);
    let max_distance = 3000.0; // Tune based on your world scale
    let distance_factor = min(distance / max_distance, 1.0);
    
    // Confidence factor: inverse of confidence [0, 1]
    // Lower confidence = higher eviction priority
    let confidence = cell.radiance_w.w;
    let max_confidence = 10.0; // Tune based on typical confidence values
    let confidence_factor = 1.0 - min(confidence / max_confidence, 1.0);
    
    // Sample count factor: inverse of sample count [0, 1]
    // Fewer samples = higher eviction priority
    let sample_count = cell.normal_count.w;
    let sample_factor = 1.0 - min(sample_count / 32.0, 1.0);
    
    // Weighted combination (tune weights based on importance)
    // Higher weights = more important in eviction decision
    let weight_age = 2.0;
    let weight_distance = 0.5;
    let weight_confidence = 0.5;
    let weight_samples = 1.0;
    
    let total_weight = weight_age + weight_distance + weight_confidence + weight_samples;
    let score = (age_factor * weight_age + 
                 distance_factor * weight_distance + 
                 confidence_factor * weight_confidence +
                 sample_factor * weight_samples) / total_weight;
    
    return score;
}

// =============================================================================
// Insert or update entry in world cache with adaptive eviction
// When cache is full, evicts the lowest-priority entry (oldest, farthest, lowest confidence)
// =============================================================================
fn insert_world_cache(
    position: vec3<f32>,
    normal: vec3<f32>,
    radiance: vec3<f32>,
    cache_size: u32,
    cell_size: f32,
    frame_index: u32,
    camera_position: vec3<f32>
) {
    let hash_key = spatial_hash(position, cell_size);
    let base_index = hash_key % cache_size;
    
    // Try to find existing entry or empty slot
    let max_probes = 8u;
    
    // Track best eviction candidate while probing
    var evict_index = base_index;
    var evict_score = -1.0; // Start with invalid score
    var found_empty = false;
    var found_update = false;
    
    for (var i = 0u; i < max_probes; i = i + 1u) {
        let probe_index = (base_index + i) % cache_size;
        
        // Check if slot is empty
        if (world_cache[probe_index].data.y == 0u) {
            // Claim this slot immediately
            world_cache[probe_index].position_frame = vec4<f32>(position, f32(frame_index));
            world_cache[probe_index].normal_count = vec4<f32>(normal, 1.0);
            world_cache[probe_index].radiance_w = vec4<f32>(radiance, 1.0);
            world_cache[probe_index].data = vec4<u32>(hash_key, 1u, 0u, 0u);
            return;
        }

        let same_cell = same_grid_cell(position, world_cache[probe_index].position_frame.xyz, cell_size);
        
        // Check if this is the same cell (update existing)
        if (world_cache[probe_index].data.x == hash_key && same_cell) {
            // Update with exponential moving average for temporal stability
            let old_radiance = world_cache[probe_index].radiance_w.xyz;
            let old_count = world_cache[probe_index].normal_count.w;
            
            // Use EMA with adaptive alpha based on sample count (converges over time)
            let alpha = 1.0 / min(old_count + 1.0, 32.0); // Max history of 32 samples
            let new_radiance = mix(old_radiance, radiance, alpha);
            
            world_cache[probe_index].radiance_w = vec4<f32>(new_radiance, world_cache[probe_index].radiance_w.w);
            world_cache[probe_index].normal_count.w = old_count + 1.0; // Increment sample count
            world_cache[probe_index].position_frame.w = f32(frame_index); // Update frame stamp
            return;
        }

        // All occupied slots are potential eviction candidates
        // This ensures we can always insert even when hash keys don't match
        let score = compute_eviction_score(
            world_cache[probe_index],
            camera_position,
            frame_index,
            120.0 // Max age in frames (~2 seconds at 60fps)
        );
        
        // Track slot with highest eviction score (worst priority)
        if (score > evict_score) {
            evict_score = score;
            evict_index = probe_index;
        }
    }
    
    // =============================================================================
    // No empty slot or exact match found - evict lowest priority entry
    // =============================================================================
    if (evict_score >= 0.0) {
        // Evict and replace with new entry
        world_cache[evict_index].position_frame = vec4<f32>(position, f32(frame_index));
        world_cache[evict_index].normal_count = vec4<f32>(normal, 1.0);
        world_cache[evict_index].radiance_w = vec4<f32>(radiance, 1.0);
        world_cache[evict_index].data = vec4<u32>(hash_key, 1u, 0u, 0u);
    }
}


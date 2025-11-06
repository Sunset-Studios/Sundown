// =============================================================================
// GI-1.0 World Cache - Common Functions (Bucket + Fingerprint Approach)
// =============================================================================
//
// Two-level hashing scheme for efficient radiance cache lookups:
//
// DESCRIPTOR CONSTRUCTION:
//   Each cache entry is indexed by a descriptor containing:
//   1. Quantized world position (3D grid cell at LOD level)
//   2. Quantized ray direction (octahedral projection, 32x32 resolution)
//   3. LOD level (adaptive quantization based on distance from camera)
//
// HASHING STRATEGY:
//   1st Hash: descriptor → bucket_index (using hash function #1)
//   2nd Hash: descriptor → fingerprint  (using hash function #2)
//
//   Two hash functions chosen to minimize collisions (Jarzynski & Olano 2020)
//
// INSERTION/QUERY ALGORITHM:
//   1. Create descriptor from position, direction, and distance-to-camera
//   2. Apply 1st hash → get bucket index
//   3. Apply 2nd hash → get fingerprint
//   4. Linear probe within bucket (up to BUCKET_SIZE entries):
//      a. Quick fingerprint comparison (fast reject)
//      b. Full descriptor match verification (if fingerprint matches)
//   5. Insert/update entry or evict oldest/farthest entry on collision
//
// ADAPTIVE QUANTIZATION (LOD):
//   - Quantization level adapts with distance from camera
//   - Near camera: fine quantization (small cells) → high spatial precision
//   - Far from camera: coarse quantization (large cells) → roughly constant
//     sample density across view frustum
//
// BENEFITS:
//   - View-dependent caching (position + direction)
//   - Fast lookups via fingerprint comparison
//   - Efficient collision resolution with linear probing
//   - Adaptive resolution maintains consistent quality at all distances
//
// =============================================================================

// Hash function constants (from Jarzynski and Olano 2020)
const p1 = 73856093;
const p2 = 19349663;
const p3 = 83492791;
const p4 = 50331653;  // For direction hashing
const p5 = 25165843;  // For LOD hashing

// Bucket configuration
const BUCKET_SIZE = 8u;           // Number of cells per bucket
const MAX_LINEAR_PROBE = 8u;      // Max cells to check within bucket

// LOD configuration (adaptive quantization based on distance)
const NUM_LOD_LEVELS = 6u;
const LOD_EXTENT = 128.0;

// Quantization resolution for direction hashing
const QUANTIZATION_RESOLUTION = 8;
const RADIANCE_UPDATE_SAMPLE_CAP = 64.0;

// World cache cell - stores outgoing radiance at secondary vertices
// Indexed by descriptor: quantized_position + quantized_direction + LOD
struct WorldCacheCell {
    position_frame: vec4<f32>,      // xyz = world position, w = frame stamp
    normal_count: vec4<f32>,        // xyz = normal (direction), w = sample count
    radiance_w: vec4<f32>,          // xyz = radiance, w = confidence weight
    data: vec4<u32>,                // x = fingerprint hash, y = occupied flag, z = LOD level, w = padding
};

// =============================================================================
// LOD Level Selection and Quantization
// Adaptive quantization: coarser cells at distance for constant sample density
// =============================================================================

// Determine LOD level based on distance from camera
// Higher LOD = coarser quantization (larger cells)
fn select_lod_level(distance_from_camera: f32, base_cell_size: f32, lod_count: u32) -> u32 {
    let normalized_distance = distance_from_camera / (base_cell_size * LOD_EXTENT);
    let raw_level = log2(max(normalized_distance, 0.001));
    let level = clamp(i32(ceil(raw_level)), 0, i32(lod_count - 1u));
    return u32(level);
}

// Get cell size for a given LOD level
fn get_lod_cell_size(lod_level: u32, base_cell_size: f32) -> f32 {
    return base_cell_size * f32(1u << lod_level); // base_size * 2^lod
}

// Quantize world position to grid cell at specified LOD level
// Returns quantized grid cell coordinates
fn quantize_position(position: vec3<f32>, lod_level: u32, base_cell_size: f32) -> vec3<i32> {
    let cell_size = get_lod_cell_size(lod_level, base_cell_size);
    return vec3<i32>(floor(position / cell_size));
}

// Quantize direction (normal) to discrete hemisphere directions
// Uses octahedral mapping for compact representation
// Returns quantized direction as integer coordinates
fn quantize_direction(direction: vec3<f32>) -> vec2<i32> {
    // Octahedral projection (maps sphere to square)
    let l1norm = abs(direction.x) + abs(direction.y) + abs(direction.z);
    let oct = direction.xy / max(l1norm, 0.0001);
    
    // Quantize to 32x32 grid for reasonable angular resolution
    let quantized = vec2<i32>(
        i32(floor(oct.x * f32(QUANTIZATION_RESOLUTION) + 0.5)),
        i32(floor(oct.y * f32(QUANTIZATION_RESOLUTION) + 0.5))
    );
    
    return quantized;
}

// =============================================================================
// Two-Level Hashing: Bucket Hash + Fingerprint Hash
// Based on Jarzynski and Olano 2020 for minimal collisions
// =============================================================================

// FIRST HASH: Descriptor → Bucket Index
// Hashes the complete descriptor (position + direction + LOD) to a bucket
fn hash_descriptor_to_bucket(
    quantized_pos: vec3<i32>,
    quantized_dir: vec2<i32>,
    lod_level: u32,
    cache_size: u32
) -> u32 {
    // Combine all descriptor components with different primes
    let hash_pos = (quantized_pos.x * p1) ^ (quantized_pos.y * p2) ^ (quantized_pos.z * p3);
    let hash_dir = (quantized_dir.x * p4) ^ (quantized_dir.y * p5);
    let hash_lod = i32(lod_level) * 196613; // Another prime
    
    let combined_hash = bitcast<u32>(hash_pos ^ hash_dir ^ hash_lod);
    
    // Map to bucket index (each bucket contains BUCKET_SIZE cells)
    let num_buckets = cache_size / BUCKET_SIZE;
    return combined_hash % num_buckets;
}

// SECOND HASH: Descriptor → Fingerprint
// Creates a compact fingerprint for fast comparison within bucket
// Uses different hash function to minimize collisions with bucket hash
fn hash_descriptor_to_fingerprint(
    quantized_pos: vec3<i32>,
    quantized_dir: vec2<i32>,
    lod_level: u32
) -> u32 {
    // Use different mixing pattern than bucket hash
    let hash_pos = (quantized_pos.x * p5) ^ (quantized_pos.y * p4) ^ (quantized_pos.z * p1);
    let hash_dir = (quantized_dir.x * p3) ^ (quantized_dir.y * p2);
    let hash_lod = i32(lod_level) * 393241; // Different prime
    
    let fingerprint = bitcast<u32>(hash_pos ^ hash_dir ^ hash_lod);
    
    // Keep fingerprint non-zero (0 reserved for empty slots)
    return select(fingerprint, 1u, fingerprint == 0u);
}

// Helper: Get bucket start index in cache array
fn get_bucket_start_index(bucket_index: u32) -> u32 {
    return bucket_index * BUCKET_SIZE;
}

// Helper: Check if two descriptors match
fn descriptors_match(
    pos_a: vec3<i32>,
    dir_a: vec2<i32>,
    lod_a: u32,
    pos_b: vec3<i32>,
    dir_b: vec2<i32>,
    lod_b: u32
) -> bool {
    return all(pos_a == pos_b) && all(dir_a == dir_b) && lod_a == lod_b;
}

// =============================================================================
// Query World Cache (Bucket + Fingerprint Linear Probing)
// Returns cached radiance for a given position and direction descriptor
// =============================================================================
fn query_world_cache_cell(
    position: vec3<f32>,
    normal: vec3<f32>,
    camera_position: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    lod_count: u32
) -> vec3<f32> {
    // Determine LOD level and create descriptor
    let distance_from_camera = length(position - camera_position);
    let lod_level = select_lod_level(distance_from_camera, base_cell_size, lod_count);
    
    let quantized_pos = quantize_position(position, lod_level, base_cell_size);
    let quantized_dir = quantize_direction(normal);
    
    // FIRST HASH: Get bucket index
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        cache_size
    );
    
    // SECOND HASH: Get fingerprint for linear probing
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level
    );
    
    // Linear probe within bucket using fingerprint matching
    let bucket_start = get_bucket_start_index(bucket_index);
    
    for (var probe = 0u; probe < min(BUCKET_SIZE, MAX_LINEAR_PROBE); probe = probe + 1u) {
        let cell_index = bucket_start + probe;
        let cell = world_cache[cell_index];
        
        // Check if slot is occupied
        if (cell.data.y == 0u) {
            continue; // Empty slot
        }
        
        // Fast fingerprint comparison first
        if (cell.data.x != target_fingerprint) {
            continue; // Fingerprint mismatch
        }
        
        // Fingerprint matches - verify full descriptor
        let stored_pos = quantize_position(cell.position_frame.xyz, cell.data.z, base_cell_size);
        let stored_dir = quantize_direction(cell.normal_count.xyz);
        let stored_lod = cell.data.z;
        
        if (descriptors_match(quantized_pos, quantized_dir, lod_level, stored_pos, stored_dir, stored_lod)) {
            // Found matching entry - return averaged radiance
            return cell.radiance_w.xyz;
        }
    }
    
    return vec3<f32>(0.0); // No match found
}

// =============================================================================
// Query World Cache with Interpolation (Optional Higher Quality)
// Queries with slight direction perturbations to provide smooth falloff
// Note: More expensive than direct query - use for final gather if needed
// =============================================================================
fn query_world_cache_interpolated(
    position: vec3<f32>,
    normal: vec3<f32>,
    camera_position: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    lod_count: u32
) -> vec3<f32> {
    // For now, just use direct query
    // Can be extended to sample neighboring directions for smoother results
    return query_world_cache_cell(position, normal, camera_position, cache_size, base_cell_size, lod_count);
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
    let sample_factor = 1.0 - min(sample_count / 16.0, 1.0);
    
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
// Insert or Update World Cache (Bucket + Fingerprint with Eviction)
// Uses two-level hashing with descriptor-based indexing
// =============================================================================
fn insert_world_cache(
    position: vec3<f32>,
    normal: vec3<f32>,
    radiance: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    frame_index: u32,
    camera_position: vec3<f32>,
    lod_count: u32
) {
    // Determine LOD level and create descriptor
    let distance_from_camera = length(position - camera_position);
    let lod_level = select_lod_level(distance_from_camera, base_cell_size, lod_count);
    
    let quantized_pos = quantize_position(position, lod_level, base_cell_size);
    let quantized_dir = quantize_direction(normal);
    
    // FIRST HASH: Get bucket index
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        cache_size
    );
    
    // SECOND HASH: Get fingerprint for linear probing
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level
    );
    
    // Linear probe within bucket to find slot
    let bucket_start = get_bucket_start_index(bucket_index);
    
    var best_slot_index = bucket_start;
    var best_eviction_score = -1.0;
    var found_empty = false;
    var found_match = false;
    
    for (var probe = 0u; probe < min(BUCKET_SIZE, MAX_LINEAR_PROBE); probe = probe + 1u) {
        let cell_index = bucket_start + probe;
        let existing_cell = world_cache[cell_index];
        let is_occupied = existing_cell.data.y != 0u;
        
        // Case 1: Found empty slot - use it immediately
        if (!is_occupied) {
            best_slot_index = cell_index;
            found_empty = true;
            break;
        }
        
        // Case 2: Fingerprint matches - check for exact descriptor match
        if (is_occupied && existing_cell.data.x == target_fingerprint) {
            let stored_pos = quantize_position(existing_cell.position_frame.xyz, existing_cell.data.z, base_cell_size);
            let stored_dir = quantize_direction(existing_cell.normal_count.xyz);
            let stored_lod = existing_cell.data.z;
            
            if (descriptors_match(quantized_pos, quantized_dir, lod_level, stored_pos, stored_dir, stored_lod)) {
                // Exact match - update with accumulated radiance
                let old_radiance = existing_cell.radiance_w.xyz;
                let old_count = existing_cell.normal_count.w;

                let alpha = 1.0 / min(old_count + 1.0, RADIANCE_UPDATE_SAMPLE_CAP); // Max history of 32 samples
                let new_radiance = mix(old_radiance, radiance, alpha);

                world_cache[cell_index].position_frame.w = f32(frame_index);
                world_cache[cell_index].normal_count.w = old_count + 1.0;
                world_cache[cell_index].radiance_w = vec4<f32>(new_radiance, existing_cell.radiance_w.w);
                world_cache[cell_index].data = vec4<u32>(target_fingerprint, 1u, lod_level, 0u);
                found_match = true;
                break;
            }
        }
        
        // Case 3: Track worst entry for potential eviction
        if (is_occupied) {
            let score = compute_eviction_score(
                existing_cell,
                camera_position,
                frame_index,
                60.0
            );
            
            if (score > best_eviction_score) {
                best_eviction_score = score;
                best_slot_index = cell_index;
            }
        }
    }
    
    // If we found a match and updated it, we're done
    if (found_match) {
        return;
    }
    
    // Insert new entry if we found empty slot OR eviction candidate
    if (found_empty || best_eviction_score >= 0.0) {
        world_cache[best_slot_index].position_frame = vec4<f32>(position, f32(frame_index));
        world_cache[best_slot_index].normal_count = vec4<f32>(normal, 1.0);
        world_cache[best_slot_index].radiance_w = vec4<f32>(radiance, 1.0);
        world_cache[best_slot_index].data = vec4<u32>(target_fingerprint, 1u, lod_level, 0u);
    }
}
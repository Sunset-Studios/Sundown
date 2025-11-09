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

const BUCKET_SIZE = 8u;            // Number of cells per bucket
const LOD_EXTENT = 128.0;          // Size of first LOD level
const QUANTIZATION_RESOLUTION = 8; // Quantization resolution for direction hashing
const RADIANCE_UPDATE_SAMPLE_CAP = 64.0; // Maximum sample count for radiance update
const WORLD_CACHE_CELL_LIFETIME = 60.0; // Maximum lifetime of a cell in frames
const WORLD_CACHE_CELL_EMPTY = 0u;

// World cache cell - stores outgoing radiance at secondary vertices
// Indexed by descriptor: quantized_position + quantized_direction + LOD
struct WorldCacheCell {
    position_frame: vec4<f32>,      // xyz = world position, w = frame stamp
    normal_count: vec4<f32>,        // xyz = normal (direction), w = sample count
    radiance_w: vec4<f32>,          // xyz = radiance, w = confidence weight
    data: vec4<u32>,                // x = fingerprint hash, y = LOD level, z = padding, w = padding
    albedo_roughness: vec4<f32>,    // xyz = albedo, w = roughness
    material_props: vec4<f32>,      // x = metallic, y = reflectance, z = emissive, w = unused
};

// =============================================================================
// LOD Level Selection and Quantization
// Adaptive quantization: coarser cells at distance for constant sample density
// =============================================================================

// Determine LOD level using a square (Chebyshev on XY) metric from camera
// Higher LOD = coarser quantization (larger cells)
fn select_lod_level(position: vec3<f32>, camera_position: vec3<f32>, base_cell_size: f32, lod_count: u32) -> u32 {
    let delta = position - camera_position;
    let square_distance = max(max(abs(delta.x), abs(delta.y)), abs(delta.z));
    let normalized_distance = square_distance / (LOD_EXTENT);
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
    return vec3<i32>(floor(position / cell_size + 0.0001));
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
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    reflectance: f32,
    emissive: f32,
    camera_position: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    lod_count: u32
) -> vec3<f32> {
    // Determine LOD level (square thresholds) and create descriptor
    let lod_level = select_lod_level(position, camera_position, base_cell_size, lod_count);
    
    let quantized_pos = quantize_position(position, lod_level, base_cell_size);
    let quantized_dir = quantize_direction(normal);
    
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        cache_size
    );
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level
    );
    
    // Linear probe within bucket using fingerprint matching
    let bucket_start = get_bucket_start_index(bucket_index);
    
    for (var probe = 0u; probe < BUCKET_SIZE; probe = probe + 1u) {
        let cell_index = bucket_start + probe;
        // Fast fingerprint comparison first
        if (world_cache[cell_index].data.x == target_fingerprint) {
            world_cache[cell_index].position_frame.w = WORLD_CACHE_CELL_LIFETIME;
            return world_cache[cell_index].radiance_w.xyz;
        } else if (world_cache[cell_index].data.x == 0u) {
            world_cache[cell_index].position_frame = vec4<f32>(position, WORLD_CACHE_CELL_LIFETIME);
            world_cache[cell_index].normal_count = vec4<f32>(normal, 1.0);
            world_cache[cell_index].albedo_roughness = vec4<f32>(albedo, roughness);
            world_cache[cell_index].material_props = vec4<f32>(metallic, reflectance, emissive, 0.0);
            return vec3<f32>(0.0);
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
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    reflectance: f32,
    emissive: f32,
    camera_position: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    lod_count: u32
) -> vec3<f32> {
    // For now, just use direct query
    // Can be extended to sample neighboring directions for smoother results
    return query_world_cache_cell(
        position,
        normal,
        albedo,
        roughness,
        metallic,
        reflectance,
        emissive,
        camera_position,
        cache_size,
        base_cell_size,
        lod_count
    );
}

fn read_world_cache_cell_radiance(
    position: vec3<f32>,
    normal: vec3<f32>,
    camera_position: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    lod_count: u32
) -> vec3<f32> {
    // Determine LOD level (square thresholds) and create descriptor
    let lod_level = select_lod_level(position, camera_position, base_cell_size, lod_count);
    let quantized_pos = quantize_position(position, lod_level, base_cell_size);
    let quantized_dir = quantize_direction(normal);
    
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        cache_size
    );
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level
    );
    
    // Linear probe within bucket using fingerprint matching
    let bucket_start = get_bucket_start_index(bucket_index);
    for (var probe = 0u; probe < BUCKET_SIZE; probe = probe + 1u) {
        let cell_index = bucket_start + probe;
        if (world_cache[cell_index].data.x == target_fingerprint) {
            return world_cache[cell_index].radiance_w.xyz;
        }
    }
    
    return vec3<f32>(0.0); // No match found
}

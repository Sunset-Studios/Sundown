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
//   4. PCG-based pseudorandom probe within bucket (up to BUCKET_SIZE entries):
//      a. Initialize PCG state from fingerprint (deterministic seed)
//      b. Generate probe sequence using PCG random numbers
//      c. Quick fingerprint comparison at each probe location (fast reject)
//      d. Full descriptor match verification (if fingerprint matches)
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
//   - Efficient collision resolution with PCG-based pseudorandom probing
//   - Better hash distribution than linear probing (reduces clustering)
//   - Adaptive resolution maintains consistent quality at all distances
//
// =============================================================================

// Hash function constants (from Jarzynski and Olano 2020)
const p1 = 73856093;
const p2 = 19349663;
const p3 = 83492791;
const p4 = 50331653;  // For direction hashing
const p5 = 25165843;  // For LOD hashing
const p6 = 12582923;  // For direction hashing

const BUCKET_SIZE = 8u;            // Number of cells per bucket
const LOD_EXTENT = 128.0;          // Size of first LOD level
const QUANTIZATION_RESOLUTION = 1; // Quantization resolution for direction hashing
const WORLD_CACHE_RADIANCE_UPDATE_SAMPLE_CAP = 32.0; // Maximum sample count for radiance update
const WORLD_CACHE_CELL_LIFETIME = 32.0; // Maximum lifetime of a cell in frames
const WORLD_CACHE_CELL_EMPTY = 0u;
const PCG_MULTIPLIER = 747796405u;
const PCG_INCREMENT = 2891336453u;

// PCG random state for generating probe sequence
struct PcgHashState {
    state: u32,
}

// World cache cell - stores outgoing radiance at secondary vertices
// Indexed by descriptor: quantized_position + quantized_direction + LOD
struct WorldCacheCell {
    position_frame: vec4<f32>,      // xyz = world position, w = frame stamp
    normal_rank: vec4<f32>,        // xyz = normal (direction), w = rank
    radiance_m: vec4<f32>,          // xyz = radiance, w = sample count
    albedo_roughness: vec4<f32>,    // xyz = albedo, w = roughness
    material_props: vec4<f32>,      // x = metallic, y = reflectance, z = emissive, w = unused
    fingerprint: atomic<u32>,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

// Initialize PCG hash state from seed value
// Performs two rounds of PCG advancement for better mixing
fn pcg_hash_init(seed: u32) -> PcgHashState {
    var state = seed * PCG_MULTIPLIER + PCG_INCREMENT;
    state = state * PCG_MULTIPLIER + PCG_INCREMENT;
    return PcgHashState(state);
}

// Generate next probe index using PCG random number generator
// Returns value in range [0, max_val)
// Updates internal state for next call
fn pcg_hash_next(state_ptr: ptr<function, PcgHashState>, max_val: u32) -> u32 {
    let state = (*state_ptr).state;
    
    // PCG XSH-RR output function (32-bit)
    // XSH = xorshift, RR = random rotation
    let xorshifted = ((state >> 18u) ^ state) >> 27u;
    let rot = state >> 27u;
    let result = (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    
    // Advance internal state (LCG step)
    (*state_ptr).state = state * PCG_MULTIPLIER + PCG_INCREMENT;
    
    return result % max_val;
}

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
fn quantize_direction(direction: vec3<f32>) -> vec3<i32> {
    // Octahedral projection (maps sphere to square)
    let l1norm = abs(direction.x) + abs(direction.y) + abs(direction.z);
    let oct = direction.xyz / max(l1norm, 0.0001);
    
    // Quantize to 32x32 grid for reasonable angular resolution
    let quantized = vec3<i32>(
        i32(floor(oct.x * f32(QUANTIZATION_RESOLUTION) + 0.5)),
        i32(floor(oct.y * f32(QUANTIZATION_RESOLUTION) + 0.5)),
        i32(floor(oct.z * f32(QUANTIZATION_RESOLUTION) + 0.5))
    );
    
    return quantized;
}

// FIRST HASH: Descriptor → Bucket Index
// Hashes the complete descriptor (position + direction + LOD + short_ray flag) to a bucket
// The is_short_ray flag separates rays that haven't traveled beyond a cell size,
// preventing light leaking when secondary rays are shorter than cell dimensions
fn hash_descriptor_to_bucket(
    quantized_pos: vec3<i32>,
    quantized_dir: vec3<i32>,
    lod_level: u32,
    is_short_ray: bool,
    cache_size: u32,
    lod_count: u32
) -> u32 {
    // Combine all descriptor components with different primes
    let hash_pos = (quantized_pos.x * p1) ^ (quantized_pos.y * p2) ^ (quantized_pos.z * p3);
    let hash_dir = (quantized_dir.x * p4) ^ (quantized_dir.y * p5) ^ (quantized_dir.z * p6);
    let hash_lod = i32(lod_level) * 196613; // Another prime
    // Hash the short ray boolean to separate short/long ray events into distinct cells
    let hash_short = select(0, 786433, is_short_ray); // Prime for short ray flag
    
    let combined_hash = bitcast<u32>(hash_pos ^ hash_dir ^ hash_lod ^ hash_short);
    
    // Map to bucket index (each bucket contains BUCKET_SIZE cells)
    // Hash across entire cache including all LOD levels
    let total_cache_size = cache_size * lod_count;
    let num_buckets = total_cache_size / BUCKET_SIZE;
    return combined_hash % num_buckets;
}

// SECOND HASH: Descriptor → Fingerprint
// Creates a compact fingerprint for fast comparison within bucket
// Uses different hash function to minimize collisions with bucket hash
// Includes is_short_ray flag to match bucket hash descriptor
fn hash_descriptor_to_fingerprint(
    quantized_pos: vec3<i32>,
    quantized_dir: vec3<i32>,
    lod_level: u32,
    is_short_ray: bool
) -> u32 {
    // Use different mixing pattern than bucket hash
    let hash_pos = (quantized_pos.x * p5) ^ (quantized_pos.y * p4) ^ (quantized_pos.z * p1);
    let hash_dir = (quantized_dir.x * p3) ^ (quantized_dir.y * p2) ^ (quantized_dir.z * p6);
    let hash_lod = i32(lod_level) * 393241; // Different prime
    // Hash the short ray boolean with different prime than bucket hash
    let hash_short = select(0, 1572869, is_short_ray); // Different prime for fingerprint
    
    let fingerprint = bitcast<u32>(hash_pos ^ hash_dir ^ hash_lod ^ hash_short);
    
    // Keep fingerprint non-zero (0 reserved for empty slots)
    return select(fingerprint, 1u, fingerprint == 0u);
}

// Helper: Get bucket start index in cache array
fn get_bucket_start_index(bucket_index: u32) -> u32 {
    return bucket_index * BUCKET_SIZE;
}

// =============================================================================
// Query World Cache (Bucket + Fingerprint with PCG Probing)
// Returns cached radiance for a given position and direction descriptor
// Uses PCG-based pseudorandom probing for better collision resolution
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
    lod_count: u32,
    ray_length: f32,
    rank: u32
) -> vec3<f32> {
    // Determine LOD level (square thresholds) and create descriptor
    let lod_level = select_lod_level(position, camera_position, base_cell_size, lod_count);
    let cell_size = get_lod_cell_size(lod_level, base_cell_size);
    
    // Light-leak prevention: separate short rays (< cell size) from long rays
    // Short rays haven't traveled far enough to be in a geometrically distinct region
    let is_short_ray = ray_length < cell_size;
    
    let quantized_pos = quantize_position(position, lod_level, base_cell_size);
    let quantized_dir = quantize_direction(normal);
    
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray,
        cache_size,
        lod_count
    );
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray
    );
    
    // Initialize PCG state for pseudorandom probing within bucket
    // Seed with fingerprint to get deterministic but well-distributed probe sequence
    let bucket_start = get_bucket_start_index(bucket_index);
    var pcg_state = pcg_hash_init(target_fingerprint);
    
    var cell_index = bucket_start;
    // PCG-based probing: each collision jumps to pseudorandom location in bucket
    for (var probe = 0u; probe < BUCKET_SIZE; probe = probe + 1u) {
        // Fast fingerprint comparison first
        let existing_fingerprint = atomicCompareExchangeWeak(
            &world_cache[cell_index].fingerprint, 
            WORLD_CACHE_CELL_EMPTY, 
            target_fingerprint
        ).old_value;
        let existing_rank = u32(world_cache[cell_index].normal_rank.w);

        if (existing_fingerprint == target_fingerprint && rank <= existing_rank) {
            // Cache hit: found matching entry, refresh lifetime and return radiance
            world_cache[cell_index].position_frame = vec4<f32>(position, WORLD_CACHE_CELL_LIFETIME);
            world_cache[cell_index].normal_rank = vec4<f32>(normal, world_cache[cell_index].normal_rank.w);
            world_cache[cell_index].albedo_roughness = vec4<f32>(albedo, roughness);
            world_cache[cell_index].material_props = vec4<f32>(metallic, reflectance, emissive, 0.0);
            return world_cache[cell_index].radiance_m.xyz;
        } else if (existing_fingerprint == WORLD_CACHE_CELL_EMPTY) {
            // Empty slot: initialize new cache entry
            world_cache[cell_index].position_frame = vec4<f32>(position, WORLD_CACHE_CELL_LIFETIME);
            world_cache[cell_index].normal_rank = vec4<f32>(normal, world_cache[cell_index].normal_rank.w + 1.0);
            world_cache[cell_index].albedo_roughness = vec4<f32>(albedo, roughness);
            world_cache[cell_index].material_props = vec4<f32>(metallic, reflectance, emissive, 0.0);
            world_cache[cell_index].radiance_m = vec4<f32>(0.0);
            return vec3<f32>(0.0);
        }

        // Probe next slot within bucket (stay within bucket boundaries)
        cell_index = bucket_start + ((probe + pcg_hash_next(&pcg_state, BUCKET_SIZE)) % BUCKET_SIZE);
    }

    return vec3<f32>(0.0);
}

// =============================================================================
// Probabilistic Query with Distance-Based Allocation Probability
// Queries world cache with allocation probability inversely proportional to distance
// Provides smooth falloff: nearby hits allocate often, distant hits rarely
// Excellent for temporal amortization while preserving camera coverage
// =============================================================================
fn query_world_cache_cell_probabilistic(
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
    lod_count: u32,
    allocation_radius: f32,  // Distance at which allocation probability = 0.5
    random_value: f32,       // Random value [0,1] for probabilistic decision
    ray_length: f32,          // Secondary ray length for light-leak prevention
    rank: u32
) -> vec3<f32> {
    let distance_from_camera = length(position - camera_position);
    
    // Compute allocation probability: exponential falloff with distance
    // probability = exp(-distance / radius)
    // At distance = 0: prob = 1.0 (always allocate)
    // At distance = radius: prob = 0.37 (occasionally allocate)  
    // At distance = 3*radius: prob ~= 0.05 (rarely allocate)
    let allocation_probability = exp(-distance_from_camera / allocation_radius);
    
    // Decide whether to allow allocation based on probability
    if (random_value < allocation_probability) {
        // Probabilistic allocation: allows gradual propagation to distant regions
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
            lod_count,
            ray_length,
            rank
        );
    }
    
    // Read without allocating instead
    return read_world_cache_cell_radiance(
        position,
        normal,
        camera_position,
        cache_size,
        base_cell_size,
        lod_count,
        ray_length,
        rank
    );
}

// =============================================================================
// Query World Cache with Interpolation (Higher Quality Spatial Filtering)
// Performs trilinear interpolation across 8 neighboring cells with intelligent
// falloff based on distance, normal alignment, and cell confidence.
// 
// INTERPOLATION STRATEGY:
//   1. Compute fractional position within current LOD cell
//   2. Sample all 8 corner cells (neighbors in 3D grid)
//   3. Weight each neighbor by:
//      - Trilinear distance weight (standard interpolation)
//      - Normal similarity (dot product with query normal)
//      - Confidence from sample count (cells with more samples = higher weight)
//   4. Normalize and return weighted radiance sum
//
// PERFORMANCE NOTES:
//   - ~8x more cache reads than direct query
//   - Uses read-only queries (no allocation) for neighbors
//   - Allocates only the primary cell to maintain cache population
//   - Best used for final gather / screen-space integration
// =============================================================================

// Helper: Read a neighboring cell's radiance and metadata for interpolation
// Returns vec4: xyz = radiance, w = validity weight (0 if empty, else confidence)
fn read_neighbor_cell_for_interpolation(
    quantized_pos: vec3<i32>,
    quantized_dir: vec3<i32>,
    lod_level: u32,
    is_short_ray: bool,
    query_normal: vec3<f32>,
    cache_size: u32,
    lod_count: u32,
    rank: u32
) -> vec4<f32> {
    // Hash to find bucket and fingerprint for this neighbor cell
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray,
        cache_size,
        lod_count
    );
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray
    );
    
    // PCG-based probing within bucket
    let bucket_start = get_bucket_start_index(bucket_index);
    var pcg_state = pcg_hash_init(target_fingerprint);
    
    var cell_index = bucket_start;
    for (var probe = 0u; probe < BUCKET_SIZE; probe = probe + 1u) {
        let existing_fingerprint = atomicLoad(&world_cache[cell_index].fingerprint);
        
        if (existing_fingerprint == target_fingerprint && rank <= u32(world_cache[cell_index].normal_rank.w)) {
            // Found matching cell - extract radiance and compute validity weight
            let radiance = world_cache[cell_index].radiance_m.xyz;
            let cell_normal = world_cache[cell_index].normal_rank.xyz;
            let sample_count = world_cache[cell_index].radiance_m.w;
            
            // Normal alignment factor: prefer cells facing similar direction
            // Use saturated dot product for hemisphere compatibility
            let normal_alignment = max(dot(cell_normal, query_normal), 0.0);
            
            // Confidence factor: cells with more samples are more reliable
            // Smooth ramp from 0 to 1 over first few samples
            let confidence = saturate(sample_count / 4.0);
            
            // Combined validity weight
            let validity = normal_alignment * confidence;
            
            return vec4<f32>(radiance, validity);
        }
        
        // Probe next slot within bucket
        cell_index = bucket_start + (probe + pcg_hash_next(&pcg_state, BUCKET_SIZE)) % BUCKET_SIZE;
    }
    
    // Cell not found - return zero with zero validity
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

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
    lod_count: u32,
    ray_length: f32,
    rank: u32
) -> vec3<f32> {
    // -------------------------------------------------------------------------
    // Step 1: Determine LOD level and cell size
    // -------------------------------------------------------------------------
    let lod_level = select_lod_level(position, camera_position, base_cell_size, lod_count);
    let cell_size = get_lod_cell_size(lod_level, base_cell_size);
    let is_short_ray = ray_length < cell_size;
    let quantized_dir = quantize_direction(normal);
    
    // -------------------------------------------------------------------------
    // Step 2: Compute fractional position within cell for trilinear weights
    // -------------------------------------------------------------------------
    // Get continuous cell coordinates (not quantized)
    let cell_coords = position / cell_size;
    
    // Floor gives us the base cell, fract gives interpolation weights
    let base_cell = vec3<i32>(floor(cell_coords));
    let frac = fract(cell_coords);
    
    // Trilinear interpolation weights for the 8 corners
    // Weight for corner (i,j,k) = product of (1-frac) or frac per axis
    let w000 = (1.0 - frac.x) * (1.0 - frac.y) * (1.0 - frac.z);
    let w001 = (1.0 - frac.x) * (1.0 - frac.y) * frac.z;
    let w010 = (1.0 - frac.x) * frac.y * (1.0 - frac.z);
    let w011 = (1.0 - frac.x) * frac.y * frac.z;
    let w100 = frac.x * (1.0 - frac.y) * (1.0 - frac.z);
    let w101 = frac.x * (1.0 - frac.y) * frac.z;
    let w110 = frac.x * frac.y * (1.0 - frac.z);
    let w111 = frac.x * frac.y * frac.z;
    
    // -------------------------------------------------------------------------
    // Step 3: Query primary cell with allocation (maintains cache population)
    // -------------------------------------------------------------------------
    let primary_radiance = query_world_cache_cell(
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
        lod_count,
        ray_length,
        rank
    );
    
    // -------------------------------------------------------------------------
    // Step 4: Sample all 8 neighboring cells (read-only, no allocation)
    // -------------------------------------------------------------------------
    // Sample each corner of the interpolation cube
    let n000 = read_neighbor_cell_for_interpolation(
        base_cell + vec3<i32>(0, 0, 0), quantized_dir, lod_level, is_short_ray, normal, cache_size, lod_count, rank
    );
    let n001 = read_neighbor_cell_for_interpolation(
        base_cell + vec3<i32>(0, 0, 1), quantized_dir, lod_level, is_short_ray, normal, cache_size, lod_count, rank
    );
    let n010 = read_neighbor_cell_for_interpolation(
        base_cell + vec3<i32>(0, 1, 0), quantized_dir, lod_level, is_short_ray, normal, cache_size, lod_count, rank
    );
    let n011 = read_neighbor_cell_for_interpolation(
        base_cell + vec3<i32>(0, 1, 1), quantized_dir, lod_level, is_short_ray, normal, cache_size, lod_count, rank
    );
    let n100 = read_neighbor_cell_for_interpolation(
        base_cell + vec3<i32>(1, 0, 0), quantized_dir, lod_level, is_short_ray, normal, cache_size, lod_count, rank
    );
    let n101 = read_neighbor_cell_for_interpolation(
        base_cell + vec3<i32>(1, 0, 1), quantized_dir, lod_level, is_short_ray, normal, cache_size, lod_count, rank
    );
    let n110 = read_neighbor_cell_for_interpolation(
        base_cell + vec3<i32>(1, 1, 0), quantized_dir, lod_level, is_short_ray, normal, cache_size, lod_count, rank
    );
    let n111 = read_neighbor_cell_for_interpolation(
        base_cell + vec3<i32>(1, 1, 1), quantized_dir, lod_level, is_short_ray, normal, cache_size, lod_count, rank
    );
    
    // -------------------------------------------------------------------------
    // Step 5: Compute weighted radiance sum with combined weights
    // -------------------------------------------------------------------------
    // Combined weight = trilinear_weight * validity_weight (from normal + confidence)
    let cw000 = w000 * n000.w;
    let cw001 = w001 * n001.w;
    let cw010 = w010 * n010.w;
    let cw011 = w011 * n011.w;
    let cw100 = w100 * n100.w;
    let cw101 = w101 * n101.w;
    let cw110 = w110 * n110.w;
    let cw111 = w111 * n111.w;
    
    // Accumulate weighted radiance
    var weighted_radiance = vec3<f32>(0.0);
    weighted_radiance += n000.xyz * cw000;
    weighted_radiance += n001.xyz * cw001;
    weighted_radiance += n010.xyz * cw010;
    weighted_radiance += n011.xyz * cw011;
    weighted_radiance += n100.xyz * cw100;
    weighted_radiance += n101.xyz * cw101;
    weighted_radiance += n110.xyz * cw110;
    weighted_radiance += n111.xyz * cw111;
    
    // Total weight for normalization
    let total_weight = cw000 + cw001 + cw010 + cw011 + cw100 + cw101 + cw110 + cw111;
    
    // -------------------------------------------------------------------------
    // Step 6: Normalize or fallback to primary cell
    // -------------------------------------------------------------------------
    // If we have enough valid neighbor data, use interpolated result
    // Otherwise fall back to the primary cell's direct query result
    let min_weight_threshold = 0.01;
    
    if (total_weight > min_weight_threshold) {
        // Blend interpolated result with primary for stability
        // Higher total weight = more confidence in interpolation
        let interp_confidence = saturate(total_weight * 2.0);
        let interpolated = weighted_radiance / total_weight;
        return mix(primary_radiance, interpolated, interp_confidence);
    }
    
    // Fallback to direct query result
    return primary_radiance;
}

fn read_world_cache_cell_radiance(
    position: vec3<f32>,
    normal: vec3<f32>,
    camera_position: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    lod_count: u32,
    ray_length: f32,
    rank: u32
) -> vec3<f32> {
    // Determine LOD level (square thresholds) and create descriptor
    let lod_level = select_lod_level(position, camera_position, base_cell_size, lod_count);
    let cell_size = get_lod_cell_size(lod_level, base_cell_size);
    
    // Light-leak prevention: separate short rays (< cell size) from long rays
    let is_short_ray = ray_length < cell_size;
    
    let quantized_pos = quantize_position(position, lod_level, base_cell_size);
    let quantized_dir = quantize_direction(normal);
    
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray,
        cache_size,
        lod_count
    );
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray
    );
    
    // Linear probe within bucket using fingerprint matching
    let bucket_start = get_bucket_start_index(bucket_index);
    var pcg_state = pcg_hash_init(target_fingerprint);
    
    var cell_index = bucket_start;
    for (var cell = 0u; cell < BUCKET_SIZE; cell = cell + 1u) {
        if (atomicLoad(&world_cache[cell_index].fingerprint) == target_fingerprint && rank <= u32(world_cache[cell_index].normal_rank.w)) {
            return world_cache[cell_index].radiance_m.xyz;
        }
        // Probe next slot within bucket (stay within bucket boundaries)
        cell_index = bucket_start + ((cell + pcg_hash_next(&pcg_state, BUCKET_SIZE)) % BUCKET_SIZE);
    }
    return vec3<f32>(0.0);
}

fn read_world_cache_cell_rank(
    position: vec3<f32>,
    normal: vec3<f32>,
    camera_position: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    lod_count: u32,
    ray_length: f32
) -> u32 {
    let lod_level = select_lod_level(position, camera_position, base_cell_size, lod_count);
    let cell_size = get_lod_cell_size(lod_level, base_cell_size);
    
    // Light-leak prevention: separate short rays (< cell size) from long rays
    let is_short_ray = ray_length < cell_size;
    
    let quantized_pos = quantize_position(position, lod_level, base_cell_size);
    let quantized_dir = quantize_direction(normal);
    
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray,
        cache_size,
        lod_count
    );
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray
    );
    
    let bucket_start = get_bucket_start_index(bucket_index);
    var pcg_state = pcg_hash_init(target_fingerprint);
    
    var cell_index = bucket_start;
    for (var cell = 0u; cell < BUCKET_SIZE; cell = cell + 1u) {
        if (atomicLoad(&world_cache[cell_index].fingerprint) == target_fingerprint) {
            return u32(world_cache[cell_index].normal_rank.w);
        }
        // Probe next slot within bucket (stay within bucket boundaries)
        cell_index = bucket_start + ((cell + pcg_hash_next(&pcg_state, BUCKET_SIZE)) % BUCKET_SIZE);
    }
    return 0u;
}

fn validate_world_cache_cell(
    position: vec3<f32>,
    normal: vec3<f32>,
    camera_position: vec3<f32>,
    cache_size: u32,
    base_cell_size: f32,
    lod_count: u32,
    ray_length: f32,
    rank: u32
) -> bool {
    let lod_level = select_lod_level(position, camera_position, base_cell_size, lod_count);
    let cell_size = get_lod_cell_size(lod_level, base_cell_size);
    
    // Light-leak prevention: separate short rays (< cell size) from long rays
    let is_short_ray = ray_length < cell_size;
    
    let quantized_pos = quantize_position(position, lod_level, base_cell_size);
    let quantized_dir = quantize_direction(normal);
    
    let bucket_index = hash_descriptor_to_bucket(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray,
        cache_size,
        lod_count
    );
    let target_fingerprint = hash_descriptor_to_fingerprint(
        quantized_pos,
        quantized_dir,
        lod_level,
        is_short_ray
    );

    // Linear probe within bucket using fingerprint matching
    let bucket_start = get_bucket_start_index(bucket_index);
    var pcg_state = pcg_hash_init(target_fingerprint);
    
    var cell_index = bucket_start;
    for (var cell = 0u; cell < BUCKET_SIZE; cell = cell + 1u) {
        if (atomicLoad(&world_cache[cell_index].fingerprint) == target_fingerprint && rank <= u32(world_cache[cell_index].normal_rank.w)) {
            return true;
        }
        // Probe next slot within bucket (stay within bucket boundaries)
        cell_index = bucket_start + ((cell + pcg_hash_next(&pcg_state, BUCKET_SIZE)) % BUCKET_SIZE);
    }
    return false;
}

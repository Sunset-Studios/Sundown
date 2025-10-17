// =============================================================================
// Path Tracer - Combined Bin and Sort Pass
// - Counts rays per bin and writes sorted indices in one pass
// - Uses subgroup operations for efficient parallel prefix sum
// - Minimizes memory access and synchronization overhead
// =============================================================================
diagnostic(off,subgroup_uniformity);

#include "common.wgsl"

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,
    frame_phase: u32,
    indirect_boost: u32,
    padding: u32,
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
};

// Consolidated binning information structure
struct RayBinInfo {
    // Bin 0-7 counters (atomics for counting phase)
    bin_counts: array<atomic<u32>, 8>,
    // Bin 0-7 offsets (computed via prefix sum)
    bin_offsets: array<u32, 8>,
    // Sorted ray indices (pixel indices in binned order)
    // Dynamic size: max width * height
    ray_indices: array<u32>,
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read_write> bin_info: RayBinInfo;
@group(1) @binding(3) var output_tex: texture_storage_2d<rgba16float, read>;

// Compute which octant bin a ray belongs to based on direction
fn compute_ray_bin(direction: vec3<f32>) -> u32 {
    var bin = 0u;
    if (direction.x >= 0.0) { bin |= 1u; }
    if (direction.y >= 0.0) { bin |= 2u; }
    if (direction.z >= 0.0) { bin |= 4u; }
    return bin;
}

// Helper function to compute pixel coordinates from linear index
fn compute_pixel_coords(linear_index: u32, res: vec2<u32>, trace_rate: u32, frame_phase: u32) -> vec2<u32> {
    if (trace_rate <= 1u) {
        return vec2<u32>(linear_index % res.x, linear_index / res.x);
    }
    
    let avg_pixels_per_row = res.x / trace_rate;
    let estimated_row = linear_index / max(avg_pixels_per_row, 1u);
    
    let search_start = select(0u, estimated_row - 1u, estimated_row >= 1u);
    let search_end = min(estimated_row + 4u, res.y);
    
    var cumulative_pixels = search_start * avg_pixels_per_row;
    
    for (var y = search_start; y < search_end; y = y + 1u) {
        let first_x = (frame_phase + trace_rate - (y * 2u) % trace_rate) % trace_rate;
        let pixels_in_row = (res.x + trace_rate - 1u - first_x) / trace_rate;
        
        if (linear_index < cumulative_pixels + pixels_in_row) {
            let offset_in_row = linear_index - cumulative_pixels;
            let x = first_x + offset_in_row * trace_rate;
            return vec2<u32>(x, y);
        }
        
        cumulative_pixels += pixels_in_row;
    }
    
    return vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
}

// Workgroup-local storage for wave-level aggregation
var<workgroup> wg_bin_counts: array<atomic<u32>, 8>;
var<workgroup> wg_bin_offsets: array<u32, 8>;

// Combined pass: Count + Prefix Sum + Bin (single kernel, wave-optimized)
@compute @workgroup_size(128, 1, 1)
fn count_and_bin_combined(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>,
    @builtin(subgroup_invocation_id) lane_id: u32,
    @builtin(subgroup_size) wave_size: u32
) {
    let res = textureDimensions(output_tex);
    let total_pixels = res.x * res.y;
    
    // === PHASE 1: Count rays per bin (wave-aggregated) ===
    
    // Initialize workgroup counters
    if (local_idx < 8u) {
        atomicStore(&wg_bin_counts[local_idx], 0u);
    }
    workgroupBarrier();
    
    let pixel_coords = compute_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    var pixel_index = 0u;
    var bin = 0u;
    var is_valid = false;
    
    if (pixel_coords.x < res.x && pixel_coords.y < res.y) {
        pixel_index = pixel_coords.y * res.x + pixel_coords.x;
        let ps = path_state[pixel_index];
        
        let skip_main_hit = (pt_params.use_gbuffer != 0u) && (ps.state_u32.x == 0u) && (ps.state_u32.w == 0x0u);
        let is_alive = ps.state_u32.y != 0u;
        
        if (is_alive && !skip_main_hit) {
            let ray_dir = ps.direction_tmax.xyz;
            bin = compute_ray_bin(ray_dir);
            is_valid = true;
            
            // Accumulate to workgroup counters
            atomicAdd(&wg_bin_counts[bin], 1u);
        }
    }
    
    workgroupBarrier();
    
    // Aggregate to global counters (one thread per bin)
    if (local_idx < 8u) {
        let wg_count = atomicLoad(&wg_bin_counts[local_idx]);
        if (wg_count > 0u) {
            atomicAdd(&bin_info.bin_counts[local_idx], wg_count);
        }
    }
    
    // === PHASE 2: Compute prefix sum (only last workgroup) ===
    // Use wave ops for ultra-fast prefix sum
    let is_last_wg = (wg_id.x == (num_wgs.x - 1u));
    
    if (is_last_wg && local_idx < 8u) {
        // All workgroups have finished counting at this point
        storageBarrier();
        
        let count = atomicLoad(&bin_info.bin_counts[local_idx]);
        
        // Wave-level exclusive prefix sum
        let prefix = subgroupExclusiveAdd(count);
        
        bin_info.bin_offsets[local_idx] = prefix;
        wg_bin_offsets[local_idx] = prefix;
    }
    
    // Wait for prefix sum to complete
    workgroupBarrier();
    
    // === PHASE 3: Write sorted indices ===
    // Reset workgroup counters for use as write pointers
    if (local_idx < 8u) {
        atomicStore(&wg_bin_counts[local_idx], 0u);
        // Load global offsets computed in phase 2
        if (!is_last_wg) {
            wg_bin_offsets[local_idx] = bin_info.bin_offsets[local_idx];
        }
    }
    workgroupBarrier();
    
    if (is_valid) {
        // Allocate slot within this bin
        let base_offset = wg_bin_offsets[bin];
        let slot = atomicAdd(&bin_info.bin_counts[bin], 1u);
        let write_index = base_offset + slot;
        
        // Write pixel index to sorted array
        bin_info.ray_indices[write_index] = pixel_index;
    }
}


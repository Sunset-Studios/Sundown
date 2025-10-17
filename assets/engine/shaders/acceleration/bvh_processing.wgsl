diagnostic(off,subgroup_uniformity);

#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC: GPU-based acceleration structure construction.

// -----------------------------------------------------------------------------
// H-PLOC constants
// -----------------------------------------------------------------------------
const SEARCH_RADIUS: u32 = 8u;

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct Cluster {
    aabb_min_and_node_idx: vec4<f32>,
    aabb_max_and_is_active: vec4<f32>,
};

struct BVHData {
    leaf_count: u32,
    bvh2_count: atomic<u32>,
    prim_count: u32,
    prim_base: u32,
    node_base: u32,
    is_blas: u32,
};

struct IndexPair {
    hi: u32,
    lo: u32,
};

// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read_write> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read_write> cluster_idx: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> bvh_data: BVHData;
@group(1) @binding(3) var<storage, read> morton_codes: array<u32>;
@group(1) @binding(4) var<storage, read_write> parent_idx: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> index_pairs: array<IndexPair>;

//------------------------------------------------------------------------------
// HPLOC Helpers
//------------------------------------------------------------------------------
fn delta_pair(a: u32, b: u32) -> u32 {
    let x = morton_codes[a] ^ morton_codes[b];
    let y = a ^ b;
    return select(x, y, x == 0u);
}

fn delta_less(a0: u32, b0: u32, a1: u32, b1: u32) -> bool {
    let d0 = delta_pair(a0, b0);
    let d1 = delta_pair(a1, b1);
    return d0 < d1;
}

fn find_parent_id(left: u32, right: u32, prim_count: u32) -> u32 {
    let cond = (left == 0u) || ((right != prim_count - 1u) && delta_less(right, right + 1u, left - 1u, left));
    return select(left - 1u, right, cond);
}

fn first_set_lane(mask: vec4<u32>) -> u32 {
    let ctz = countTrailingZeros(mask);
    let has_x = ctz.x < 32u;
    let has_y = ctz.y < 32u;
    let has_z = ctz.z < 32u;
    let has_w = ctz.w < 32u;

    var result: u32 = 128;
    result = select(result, ctz.x, has_x);
    result = select(result, ctz.y + 32u, (!has_x) && has_y);
    result = select(result, ctz.z + 64u, (!has_x) && (!has_y) && has_z);
    result = select(result, ctz.w + 96u, (!has_x) && (!has_y) && (!has_z) && has_w);
    return result;
}

// Clear the least-significant set bit across a 128-bit mask vec4<u32>
// Treats mask as little-endian words: x (bits 0..31), y (32..63), z (64..95), w (96..127)
fn clear_lsb(mask: vec4<u32>) -> vec4<u32> {
    if (mask.x != 0u) {
        return vec4<u32>(mask.x & (mask.x - 1u), mask.y, mask.z, mask.w);
    }
    if (mask.y != 0u) {
        return vec4<u32>(mask.x, mask.y & (mask.y - 1u), mask.z, mask.w);
    }
    if (mask.z != 0u) {
        return vec4<u32>(mask.x, mask.y, mask.z & (mask.z - 1u), mask.w);
    }
    if (mask.w != 0u) {
        return vec4<u32>(mask.x, mask.y, mask.z, mask.w & (mask.w - 1u));
    }
    return mask;
}

// Return the 0-based index of the n-th set bit in a 128-bit mask vec4<u32>.
fn rank_from_mask(mask: vec4<u32>, lane: u32) -> u32 {
    let word = lane >> 5u;
    let bit  = lane & 31u;

    var rank = 0u;
    if (word > 0u) { rank += countOneBits(mask.x); }
    if (word > 1u) { rank += countOneBits(mask.y); }
    if (word > 2u) { rank += countOneBits(mask.z); }

    // pick the correct 32-bit word for this lane
    var cur: u32 = mask.x;
    if (word == 1u) { cur = mask.y; }
    if (word == 2u) { cur = mask.z; }
    if (word == 3u) { cur = mask.w; }

    let below = select((1u << bit) - 1u, 0u, bit == 0u);
    rank += countOneBits(cur & below);
    return rank;
}

// Return the 0-based index of the n-th set bit in a 32-bit word.
// If k >= popcount(word), returns -1.
fn kth_set_bit_in_word(word: u32, k: u32) -> u32 {
    let total = countOneBits(word);
    if (k >= total) {
        return INVALID_IDX;
    }
    // Clear the LSB k times, then the current LSB is the answer.
    var w = word;
    var times = k;
    while (times > 0u) {
        w = w & (w - 1u);
        times -= 1u;
    }
    let pos = countTrailingZeros(w);
    return pos;
}

// Find the n-th set bit across a vec4<u32> mask (x=lowest 32 bits).
// Returns -1 if n is out of range.
fn find_nth_set_bit(mask: vec4<u32>, n: u32) -> u32 {
    // Scan per word using prefix popcounts to jump directly to the containing word.
    var remaining: u32 = n;

    // word 0: x (bits 0..31)
    var pc = countOneBits(mask.x);
    if (remaining < pc) {
        return kth_set_bit_in_word(mask.x, remaining);
    }
    remaining -= pc;

    // word 1: y (bits 32..63)
    pc = countOneBits(mask.y);
    if (remaining < pc) {
        let local = kth_set_bit_in_word(mask.y, remaining);
        return select(local + 32u, INVALID_IDX, local == INVALID_IDX);
    }
    remaining -= pc;

    // word 2: z (bits 64..95)
    pc = countOneBits(mask.z);
    if (remaining < pc) {
        let local = kth_set_bit_in_word(mask.z, remaining);
        return select(local + 64u, INVALID_IDX, local == INVALID_IDX);
    }
    remaining -= pc;

    // word 3: w (bits 96..127)
    pc = countOneBits(mask.w);
    if (remaining < pc) {
        let local = kth_set_bit_in_word(mask.w, remaining);
        return select(local + 96u, INVALID_IDX, local == INVALID_IDX);
    }

    return INVALID_IDX;
}

// Safe lane shuffles: guard against out-of-range source lanes
fn warp_safe_shuffle_u32(warp_ctx: WarpCtx, value: u32, src_lane: u32) -> u32 {
    let lane = src_lane & (warp_ctx.warp_size - 1u);
    return warp_shuffle_u32(warp_ctx, value, lane);
}

// Safe lane shuffles: guard against out-of-range source lanes
fn warp_safe_shuffle_f32(warp_ctx: WarpCtx, value: f32, src_lane: u32) -> f32 {
    let lane = src_lane & (warp_ctx.warp_size - 1u);
    return warp_shuffle_f32(warp_ctx, value, lane);
}

fn load_indices(warp_ctx: WarpCtx, start: u32, end: u32, cluster_index: ptr<function, u32>, offset: u32, merging_threshold: u32) -> u32 {
    let lane = warp_ctx.lane_id;

    let index = lane - offset;
    let lane_valid = index < min(end - start, merging_threshold);

    if (lane_valid) {
        *cluster_index = atomicLoad(&cluster_idx[start + index]);
    }

    let have_valid = lane_valid && (*cluster_index != INVALID_IDX);
    let ballot = warp_ballot_u32(warp_ctx, have_valid);

    return mask_popcount(ballot);
}

fn store_indices(warp_ctx: WarpCtx, previous_num_prim: u32, cluster_index: u32, l_start: u32) {
    let lane = warp_ctx.lane_id;
    if (lane < previous_num_prim) {
        atomicStore(&cluster_idx[l_start + lane], cluster_index);
    }
}

fn find_nearest_neighbor(
    warp_ctx: WarpCtx,
    num_prim: u32,
    cluster_value: u32,
    cmin: vec3<f32>,
    cmax: vec3<f32>
) -> u32 {
    let lane = warp_ctx.lane_id;

    var best_area = INVALID_IDX;
    var best_idx = INVALID_IDX;

    for (var r = 1u; r <= SEARCH_RADIUS; r = r + 1u) {
        let neighbor_idx = lane + r;
        let neighbor_valid = neighbor_idx < num_prim;
        var area = INVALID_IDX;

        let n_min_x = warp_safe_shuffle_f32(warp_ctx, cmin.x, neighbor_idx);
        let n_min_y = warp_safe_shuffle_f32(warp_ctx, cmin.y, neighbor_idx);
        let n_min_z = warp_safe_shuffle_f32(warp_ctx, cmin.z, neighbor_idx);
        let n_max_x = warp_safe_shuffle_f32(warp_ctx, cmax.x, neighbor_idx);
        let n_max_y = warp_safe_shuffle_f32(warp_ctx, cmax.y, neighbor_idx);
        let n_max_z = warp_safe_shuffle_f32(warp_ctx, cmax.z, neighbor_idx);

        if (neighbor_valid) {
            let mmin = vec3<f32>(min(n_min_x, cmin.x), min(n_min_y, cmin.y), min(n_min_z, cmin.z));
            let mmax = vec3<f32>(max(n_max_x, cmax.x), max(n_max_y, cmax.y), max(n_max_z, cmax.z));
            area = bitcast<u32>(calculate_aabb_surface_area(mmin, mmax));

            if (area < best_area) {
                best_area = area;
                best_idx = neighbor_idx;
            }
        }

        var nearest_neighbor_area = warp_safe_shuffle_u32(warp_ctx, best_area, neighbor_idx);
        var nearest_neighbor_idx = warp_safe_shuffle_u32(warp_ctx, best_idx, neighbor_idx);

        if (area < nearest_neighbor_area) {
            nearest_neighbor_area = area;
            nearest_neighbor_idx = lane;
        }

        best_area = warp_safe_shuffle_u32(warp_ctx, nearest_neighbor_area, lane - r);
        best_idx = warp_safe_shuffle_u32(warp_ctx, nearest_neighbor_idx, lane - r);
    }

    return best_idx;
}

fn merge_clusters_create_bvh2_node(
    warp_ctx: WarpCtx,
    num_prim: u32,
    nearest_neighbor: u32,
    cluster_index: ptr<function, u32>,
    cmin_in: ptr<function, vec3<f32>>,
    cmax_in: ptr<function, vec3<f32>>
) -> u32 {
    let lane = warp_ctx.lane_id;
    let lane_active = lane < num_prim;

    let nn_of_nn = warp_safe_shuffle_u32(warp_ctx, nearest_neighbor, nearest_neighbor);
    let mutual_neighbor = lane_active && (lane == nn_of_nn);
    let do_merge = mutual_neighbor && (lane < nearest_neighbor);

    let merge_mask = warp_ballot_u32(warp_ctx, do_merge);
    let merge_count = mask_popcount(merge_mask);

    var base_idx = 0u;
    if (is_warp_leader(warp_ctx)) {
        base_idx = atomicAdd(&bvh_data.bvh2_count, merge_count);
    }
    base_idx = warp_safe_shuffle_u32(warp_ctx, base_idx, 0u);

    // Rank among merging lanes strictly before this lane
    let rank = rank_from_mask(merge_mask, lane);

    let neighbor_cluster_index = warp_safe_shuffle_u32(warp_ctx, *cluster_index, nearest_neighbor);
    let nb_min_x = warp_safe_shuffle_f32(warp_ctx, cmin_in.x, nearest_neighbor);
    let nb_min_y = warp_safe_shuffle_f32(warp_ctx, cmin_in.y, nearest_neighbor);
    let nb_min_z = warp_safe_shuffle_f32(warp_ctx, cmin_in.z, nearest_neighbor);
    let nb_max_x = warp_safe_shuffle_f32(warp_ctx, cmax_in.x, nearest_neighbor);
    let nb_max_y = warp_safe_shuffle_f32(warp_ctx, cmax_in.y, nearest_neighbor);
    let nb_max_z = warp_safe_shuffle_f32(warp_ctx, cmax_in.z, nearest_neighbor);

    let node_index = base_idx + rank;

    var merged_min = vec4<f32>(
        min(cmin_in.x, nb_min_x), min(cmin_in.y, nb_min_y), min(cmin_in.z, nb_min_z),
        f32(bvh_data.prim_base + *cluster_index)
    );
    var merged_max = vec4<f32>(
        max(cmax_in.x, nb_max_x), max(cmax_in.y, nb_max_y), max(cmax_in.z, nb_max_z),
        f32(bvh_data.prim_base + neighbor_cluster_index)
    );
    if (do_merge) {
        *cmin_in = merged_min.xyz;
        *cmax_in = merged_max.xyz;
        *cluster_index = node_index;
        // Grow the current cluster to include the neighbor
        bounds[bvh_data.prim_base + node_index].min = merged_min;
        bounds[bvh_data.prim_base + node_index].max = merged_max;
    }

    // Compaction
    let valid_mask = warp_ballot_u32(warp_ctx, (warp_ctx.lane_id < num_prim) && (do_merge || !mutual_neighbor));
    let shift_lane = find_nth_set_bit(valid_mask, warp_ctx.lane_id);

    *cluster_index = warp_safe_shuffle_u32(warp_ctx, *cluster_index, u32(shift_lane));
    cmin_in.x = warp_safe_shuffle_f32(warp_ctx, cmin_in.x, u32(shift_lane));
    cmin_in.y = warp_safe_shuffle_f32(warp_ctx, cmin_in.y, u32(shift_lane));
    cmin_in.z = warp_safe_shuffle_f32(warp_ctx, cmin_in.z, u32(shift_lane));
    cmax_in.x = warp_safe_shuffle_f32(warp_ctx, cmax_in.x, u32(shift_lane));
    cmax_in.y = warp_safe_shuffle_f32(warp_ctx, cmax_in.y, u32(shift_lane));
    cmax_in.z = warp_safe_shuffle_f32(warp_ctx, cmax_in.z, u32(shift_lane));

    if (shift_lane == INVALID_IDX) {
        *cluster_index = INVALID_IDX;
    }

    return num_prim - merge_count;
}

fn ploc_merge(
    warp_ctx: WarpCtx,
    lane_id_selected: u32,
    left: u32,
    right: u32,
    split: u32,
    final_lane: bool,
    merging_threshold: u32
) {
    // Share current lane's LBVH node with other threads in the warp
    let l_start = warp_safe_shuffle_u32(warp_ctx, left, lane_id_selected);
    let l_end   = warp_safe_shuffle_u32(warp_ctx, split, lane_id_selected);
    let r_start = l_end;
    let r_end   = warp_safe_shuffle_u32(warp_ctx, right, lane_id_selected) + 1u;

    let lane_id = warp_ctx.lane_id;
    var cluster_index = INVALID_IDX;

    // Load left and right child cluster indices
    let num_left = load_indices(warp_ctx, l_start, l_end, &cluster_index, 0u, merging_threshold);
    let num_right = load_indices(warp_ctx, r_start, r_end, &cluster_index, num_left, merging_threshold);
    var num_prim = num_left + num_right;

    let valid_lane = lane_id < num_prim;
    let node_bounds = bounds[bvh_data.prim_base + cluster_index];
    var cmin = select(zero_vec4.xyz, node_bounds.min.xyz, valid_lane);
    var cmax = select(zero_vec4.xyz, node_bounds.max.xyz, valid_lane);

    let sync_final = warp_safe_shuffle_u32(warp_ctx, u32(final_lane), lane_id_selected) != 0u;
    let threshold = select(merging_threshold, 1u, sync_final);

    while (num_prim > threshold) {
        let nearest_neighbor = find_nearest_neighbor(warp_ctx, num_prim, cluster_index, cmin, cmax);
        num_prim = merge_clusters_create_bvh2_node(warp_ctx, num_prim, nearest_neighbor, &cluster_index, &cmin, &cmax);
    }

    store_indices(warp_ctx, num_left + num_right, cluster_index, l_start);
}

//------------------------------------------------------------------------------
// HPLOC Kernels 
//------------------------------------------------------------------------------
@compute @workgroup_size(HPLOC_WAVE_SIZE)
fn build_bvh2_hploc(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
#if HAS_SUBGROUPS
    @builtin(subgroup_invocation_id)  subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32
#endif
) {
    let total = bvh_data.leaf_count;

#if HAS_SUBGROUPS
    let lane = subgroup_id;
    let warp_ctx = make_warp_ctx(local_id.x, lane, subgroup_size);
#else
    let lane = lane_id(local_id.x, LOGICAL_WARP_SIZE);
    let warp_ctx = make_warp_ctx(local_id.x, lane, LOGICAL_WARP_SIZE);
#endif

    let merging_threshold = warp_ctx.warp_size / 2u;
    let idx = gid.x;
    var left = idx;
    var right = idx;
    var split = 0u;
    var lane_active = (idx < total);

    while (warp_any(warp_ctx, lane_active)) {
        // Any active lanes in this warp?
        if (lane_active) {
            var previous_id: u32 = INVALID_IDX;
            if (find_parent_id(left, right, total) == right) {
                previous_id = atomicExchange(&parent_idx[right], left);
                if (previous_id != INVALID_IDX) {
                    split = right + 1u;
                    right = previous_id;
                }
            } else {
                previous_id = atomicExchange(&parent_idx[left - 1u], right);
                if (previous_id != INVALID_IDX) {
                    split = left;
                    left = previous_id;
                }
            }
            if (previous_id == INVALID_IDX) {
                lane_active = false;
            }
        }

        let size = right - left + 1u;
        let final_lane = lane_active && (size == total);
        let do_merge = (lane_active && (size > merging_threshold)) || final_lane;

        // Ballot lanes requesting merge (supports up to 128 lanes)
        var mask = warp_ballot_u32(warp_ctx, do_merge);
        while (any(mask != vec4<u32>(0u))) {
            let sel = first_set_lane(mask);
            ploc_merge(warp_ctx, sel, left, right, split, final_lane, merging_threshold);
            mask = clear_lsb(mask);
        }
    }
}

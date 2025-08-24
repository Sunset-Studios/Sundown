diagnostic(off,subgroup_uniformity);

#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC Step 2: GPU-based acceleration structure construction.

// -----------------------------------------------------------------------------
// H-PLOC constants
// -----------------------------------------------------------------------------
const SEARCH_RADIUS: u32 = 8u;
const MERGING_THRESHOLD: u32 = 16u;

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------
struct Cluster {
    aabb_min_and_node_idx: vec4<f32>,
    aabb_max_and_is_active: vec4<f32>,
};

struct HPLOCUniforms {
    primitive_count: u32,
    pass_num: u32,
};

struct Counters {
    bvh2_count: atomic<u32>,
    bvh4_count: atomic<u32>,
};

// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read_write> cluster_idx: array<u32>;
@group(1) @binding(2) var<storage, read_write> bvh2_nodes: array<BVH2Node>;
@group(1) @binding(3) var<storage, read_write> counters: Counters;
@group(1) @binding(4) var<storage, read_write> clusters: array<Cluster>;
@group(1) @binding(5) var<storage, read> morton_codes: array<u32>;
@group(1) @binding(6) var<storage, read_write> parent_idx: array<atomic<u32>>;

//------------------------------------------------------------------------------
// HPLOC Helpers
//------------------------------------------------------------------------------

// Atomic exchange helper (WGSL lacks atomicExchange)
fn atomic_exchange_u32(dst: ptr<storage, atomic<u32>, read_write>, new_value: u32) -> u32 {
    var expected = atomicLoad(dst);
    loop {
        let res = atomicCompareExchangeWeak(dst, expected, new_value);
        if (res.exchanged) { return res.old_value; }
        expected = res.old_value;
    }
}

fn delta_pair(a: u32, b: u32) -> vec2<u32> {
    let x = morton_codes[a] ^ morton_codes[b];
    return vec2<u32>(x, a ^ b);
}

fn delta_less(a0: u32, b0: u32, a1: u32, b1: u32) -> bool {
    let d0 = delta_pair(a0, b0);
    let d1 = delta_pair(a1, b1);
    // Lexicographic compare: (code_xor, index_xor)
    let upper_lt = d0.x < d1.x;
    let lower_lt = d0.y < d1.y;
    return select(upper_lt, lower_lt, d0.x == d1.x);
}

fn find_parent_id(left: u32, right: u32, prim_count: u32) -> u32 {
    let cond = (left == 0u) || ((right != prim_count - 1u) && delta_less(right, right + 1u, left - 1u, left));
    return select(left - 1u, right, cond);
}

fn mask_popcount(mask: vec4<u32>) -> u32 {
    let ones = countOneBits(mask);
    return ones.x + ones.y + ones.z + ones.w;
}

fn first_set_lane(mask: vec4<u32>) -> i32 {
    var trailing = select(-1, i32(firstTrailingBit(mask.x)), mask.x != 0u);
    trailing = select(trailing, i32(firstTrailingBit(mask.y) + 32), trailing == -1);
    trailing = select(trailing, i32(firstTrailingBit(mask.z) + 64), trailing == -1);
    trailing = select(trailing, i32(firstTrailingBit(mask.w) + 96), trailing == -1);
    return trailing;
}

fn clear_bit(mask: vec4<u32>, index: u32) -> vec4<u32> {
    var m = mask;
    let part = index >> 5u;
    let bit  = 1u << (index & 31u);
    m[part] = m[part] & ~bit;
    return m;
}

// Return the 0-based index of the n-th set bit in a 32-bit word.
// If k >= popcount(word), returns -1.
fn kth_set_bit_in_word(word: u32, k: u32) -> i32 {
    let total = countOneBits(word);
    if (k >= total) {
        return -1;
    }

    // Binary-search the position using popcounts over shrinking chunks.
    var pos: u32 = 0u;
    var rem: u32 = k;

    // width sequence: 16, 8, 4, 2, 1
    {
        var width: u32 = 16u;
        var chunk = (word >> pos) & ((1u << width) - 1u);
        var cnt = countOneBits(chunk);
        if (rem >= cnt) { rem -= cnt; pos += width; }

        width = 8u;
        chunk = (word >> pos) & ((1u << width) - 1u);
        cnt = countOneBits(chunk);
        if (rem >= cnt) { rem -= cnt; pos += width; }

        width = 4u;
        chunk = (word >> pos) & ((1u << width) - 1u);
        cnt = countOneBits(chunk);
        if (rem >= cnt) { rem -= cnt; pos += width; }

        width = 2u;
        chunk = (word >> pos) & ((1u << width) - 1u);
        cnt = countOneBits(chunk);
        if (rem >= cnt) { rem -= cnt; pos += width; }

        width = 1u;
        chunk = (word >> pos) & 1u;
        // cnt is 0 or 1 here
        cnt = countOneBits(chunk);
        if (rem >= cnt) { /* rem would be 0 and cnt==0 here only if word==0, but we guarded above */ 
            // If we somehow get here with cnt==1 and rem>=1, move to the next bit.
            rem -= cnt; 
            pos += width; 
        }
    }

    // pos now points to the exact bit
    return i32(pos);
}

// Find the n-th set bit across a vec4<u32> mask (x=lowest 32 bits).
// Returns -1 if n is out of range.
fn find_nth_set_bit(mask: vec4<u32>, n: u32) -> i32 {
    // Scan per word using prefix popcounts to jump directly to the containing word.
    var remaining: u32 = n;

    // word 0: x (bits 0..31)
    var pc = countOneBits(mask.x);
    if (remaining < pc) {
        let local = kth_set_bit_in_word(mask.x, remaining);
        return local; // 0..31
    }
    remaining -= pc;

    // word 1: y (bits 32..63)
    pc = countOneBits(mask.y);
    if (remaining < pc) {
        let local = kth_set_bit_in_word(mask.y, remaining);
        return select(local + 32, -1, local < 0);
    }
    remaining -= pc;

    // word 2: z (bits 64..95)
    pc = countOneBits(mask.z);
    if (remaining < pc) {
        let local = kth_set_bit_in_word(mask.z, remaining);
        return select(local + 64, -1, local < 0);
    }
    remaining -= pc;

    // word 3: w (bits 96..127)
    pc = countOneBits(mask.w);
    if (remaining < pc) {
        let local = kth_set_bit_in_word(mask.w, remaining);
        return select(local + 96, -1, local < 0);
    }

    return -1;
}

fn load_indices(warp_ctx: WarpCtx, start: u32, end_: u32, cluster_index: ptr<function, u32>, offset: u32) -> u32 {
    let lane = warp_ctx.lane_id;

    let index = lane - offset;
    let lane_valid = index < min(end_ - start, MERGING_THRESHOLD);

    if (lane_valid) {
        *cluster_index = cluster_idx[start + index];
    }

    let have_valid = lane_valid && (*cluster_index != INVALID_IDX);
    let ballot = warp_ballot_u32(warp_ctx, have_valid);

    return mask_popcount(ballot);
}

fn store_indices(warp_ctx: WarpCtx, previous_num_prim: u32, cluster_index: u32, l_start: u32) {
    if (warp_ctx.lane_id < previous_num_prim) {
        cluster_idx[l_start + warp_ctx.lane_id] = cluster_index;
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

    var min_area = INVALID_IDX;
    var min_index = INVALID_IDX;
    for (var r = 1u; r <= SEARCH_RADIUS; r = r + 1u) {
        let neighbor_idx = lane + r;
        // Load neighbor bounds via shuffle (safe source lane)
        let nb_min_x = warp_shuffle_f32(warp_ctx, cmin.x, neighbor_idx);
        let nb_min_y = warp_shuffle_f32(warp_ctx, cmin.y, neighbor_idx);
        let nb_min_z = warp_shuffle_f32(warp_ctx, cmin.z, neighbor_idx);
        let nb_max_x = warp_shuffle_f32(warp_ctx, cmax.x, neighbor_idx);
        let nb_max_y = warp_shuffle_f32(warp_ctx, cmax.y, neighbor_idx);
        let nb_max_z = warp_shuffle_f32(warp_ctx, cmax.z, neighbor_idx);

        var area = INVALID_IDX;
        if (neighbor_idx < num_prim) {
            // Grow the current cluster to include the neighbor
            let mmin = vec3<f32>(min(nb_min_x, cmin.x), min(nb_min_y, cmin.y), min(nb_min_z, cmin.z));
            let mmax = vec3<f32>(max(nb_max_x, cmax.x), max(nb_max_y, cmax.y), max(nb_max_z, cmax.z));
            // Calculate the surface area of the new cluster
            area = bitcast<u32>(calculate_aabb_surface_area(mmin, mmax));
            // Update the nearest neighbor if the new cluster is smaller
            if (area < min_area) {
                min_area = area;
                min_index = neighbor_idx;
            }
        }

        // Read neighbor's current NN and update it if we are closer (safe lane)
        let neigh_min_area = warp_shuffle_u32(warp_ctx, min_area, neighbor_idx);
        let neigh_min_index = warp_shuffle_u32(warp_ctx, min_index, neighbor_idx);

        // Update the nearest neighbor if we are closer
        var nn_area = neigh_min_area;
        var nn_index = neigh_min_index;
        if (area < neigh_min_area) {
            nn_area = area;
            nn_index = lane;
        }

        // Get result back from cluster i - r
        min_area = warp_shuffle_u32(warp_ctx, nn_area, lane - r);
        min_index  = warp_shuffle_u32(warp_ctx, nn_index,   lane - r);
    }

    return min_index;
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

    let nn_of_nn = warp_shuffle_u32(warp_ctx, nearest_neighbor, nearest_neighbor);
    let mutual_neighbor = lane_active && (lane == nn_of_nn);
    let do_merge = mutual_neighbor && (lane < nearest_neighbor);

    let merge_mask = warp_ballot_u32(warp_ctx, do_merge);
    let merge_count = mask_popcount(merge_mask);

    var base_idx = 0u;
    if (lane == 0u) {
        base_idx = atomicAdd(&counters.bvh2_count, merge_count);
    }
    base_idx = warp_shuffle_u32(warp_ctx, base_idx, 0u);

    // Rank among merging lanes strictly before this lane
    let shifted_mask = vec4<u32>(
        merge_mask[0] << (warp_ctx.warp_size - lane),
        merge_mask[1] << (warp_ctx.warp_size - lane),
        merge_mask[2] << (warp_ctx.warp_size - lane),
        merge_mask[3] << (warp_ctx.warp_size - lane)
    );
    let rank = mask_popcount(shifted_mask);

    let neighbor_cluster_index = warp_shuffle_u32(warp_ctx, *cluster_index, nn_of_nn);
    let nb_min_x = warp_shuffle_f32(warp_ctx, cmin_in.x, nn_of_nn);
    let nb_min_y = warp_shuffle_f32(warp_ctx, cmin_in.y, nn_of_nn);
    let nb_min_z = warp_shuffle_f32(warp_ctx, cmin_in.z, nn_of_nn);
    let nb_max_x = warp_shuffle_f32(warp_ctx, cmax_in.x, nn_of_nn);
    let nb_max_y = warp_shuffle_f32(warp_ctx, cmax_in.y, nn_of_nn);
    let nb_max_z = warp_shuffle_f32(warp_ctx, cmax_in.z, nn_of_nn);

    let node_index = base_idx + rank;
    let merged_min = vec3<f32>(min(cmin_in.x, nb_min_x), min(cmin_in.y, nb_min_y), min(cmin_in.z, nb_min_z));
    let merged_max = vec3<f32>(max(cmax_in.x, nb_max_x), max(cmax_in.y, nb_max_y), max(cmax_in.z, nb_max_z));
    if (do_merge) {
        // Grow the current cluster to include the neighbor
        bvh2_nodes[node_index].min_and_left_child = vec4<f32>(merged_min, f32(*cluster_index));
        bvh2_nodes[node_index].max_and_right_child = vec4<f32>(merged_max, f32(neighbor_cluster_index));

        *cmin_in = merged_min;
        *cmax_in = merged_max;
        *cluster_index = node_index;
    }

    // Compaction
    let valid_mask = warp_ballot_u32(warp_ctx, do_merge || !mutual_neighbor);
    let shift_lane = find_nth_set_bit(valid_mask, lane + 1u);

    *cluster_index = warp_shuffle_u32(warp_ctx, *cluster_index, u32(shift_lane));
    if (shift_lane < 0) {
        *cluster_index = INVALID_IDX;
    }

    cmin_in.x = warp_shuffle_f32(warp_ctx, cmin_in.x, u32(shift_lane));
    cmin_in.y = warp_shuffle_f32(warp_ctx, cmin_in.y, u32(shift_lane));
    cmin_in.z = warp_shuffle_f32(warp_ctx, cmin_in.z, u32(shift_lane));
    cmax_in.x = warp_shuffle_f32(warp_ctx, cmax_in.x, u32(shift_lane));
    cmax_in.y = warp_shuffle_f32(warp_ctx, cmax_in.y, u32(shift_lane));
    cmax_in.z = warp_shuffle_f32(warp_ctx, cmax_in.z, u32(shift_lane));

    return num_prim - merge_count;
}

fn ploc_merge(
    warp_ctx: WarpCtx,
    lane_id_selected: u32,
    left: u32,
    right: u32,
    split: u32,
    final_lane: bool
) {
    // Share current lane's LBVH node with other threads in the warp
    let l_start = warp_shuffle_u32(warp_ctx, left, lane_id_selected);
    let r_end   = warp_shuffle_u32(warp_ctx, right, lane_id_selected) + 1u;
    let l_end   = warp_shuffle_u32(warp_ctx, split, lane_id_selected);
    let r_start = l_end;

    var cluster_index = INVALID_IDX;

    // Load left and right child cluster indices
    let num_left = load_indices(warp_ctx, l_start, l_end, &cluster_index, 0u);
    let num_right = load_indices(warp_ctx, r_start, r_end, &cluster_index, num_left);
    var num_prim = num_left + num_right;

    let valid_lane = warp_ctx.lane_id < num_prim;
    var cmin = select(zero_vec4.xyz, clusters[cluster_index].aabb_min_and_node_idx.xyz, valid_lane);
    var cmax = select(zero_vec4.xyz, clusters[cluster_index].aabb_max_and_is_active.xyz, valid_lane);

    let sync_final = warp_shuffle_u32(warp_ctx, u32(final_lane), lane_id_selected);
    let threshold = select(MERGING_THRESHOLD, 1u, sync_final != 0u);

    // while (num_prim > threshold) {
    //     let nearest_neighbor = find_nearest_neighbor(warp_ctx, num_prim, cluster_index, cmin, cmax);
    //     num_prim = merge_clusters_create_bvh2_node(warp_ctx, num_prim, nearest_neighbor, &cluster_index, &cmin, &cmax);
    // }

    store_indices(warp_ctx, num_left + num_right, cluster_index, l_start);
}

//------------------------------------------------------------------------------
// HPLOC Kernels 
//------------------------------------------------------------------------------

@compute @workgroup_size(HPLOC_WAVE_SIZE)
fn build_bvh2_hploc(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
#if HAS_SUBGROUPS
    @builtin(subgroup_invocation_id)  subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32
#endif
) {
    let total = atomicLoad(&counters.bvh2_count);

    // Limit to first warp per workgroup for deterministic behavior
#if HAS_SUBGROUPS
    let lane = subgroup_id;
    let warp_ctx = make_warp_ctx(local_id.x, lane, subgroup_size);
#else
    let lane = lane_id(local_id.x, LOGICAL_WARP_SIZE);
    let warp_ctx = make_warp_ctx(local_id.x, lane, LOGICAL_WARP_SIZE);
#endif

    // Global index in sorted order handled by this lane
    let idx = group_id.x * HPLOC_WAVE_SIZE + local_id.x;

    var left = idx;
    var right = idx;
    var split = 0u;
    var lane_active = (idx < total);

    while (warp_any(warp_ctx, lane_active)) {
        // Any active lanes in this warp?
        if (lane_active) {
            var previous_id: u32 = INVALID_IDX;
            let parent_right = (find_parent_id(left, right, total) == right);
            if (parent_right) {
                previous_id = atomic_exchange_u32(&parent_idx[right], left);
                if (previous_id != INVALID_IDX) {
                    split = right + 1u;
                    right = previous_id;
                }
            } else {
                previous_id = atomic_exchange_u32(&parent_idx[left - 1u], right);
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
        let do_merge = (lane_active && (size > MERGING_THRESHOLD)) || final_lane;

        // Ballot lanes requesting merge (supports up to 128 lanes)
        var mask = warp_ballot_u32(warp_ctx, do_merge);
        var sel = first_set_lane(mask);
        // while (sel != -1) {
        //     ploc_merge(warp_ctx, u32(sel), left, right, split, final_lane);
        //     mask = clear_bit(mask, u32(sel));
        //     sel = first_set_lane(mask);
        // }
    }
}

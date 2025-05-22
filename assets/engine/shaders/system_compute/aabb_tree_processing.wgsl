#include "common.wgsl"
// A GPU‐based dynamic AABB tree processor in WGSL.

//------------------------------------------------------------------------------
// Constants & enums
//------------------------------------------------------------------------------
const MAX_NODES:        u32   = 1024u;
const QUEUE_CAPACITY:   u32   = MAX_NODES + 1u;
const QUEUE_EMPTY:      u32   = 0xffffffffu;
const FAT_MARGIN_FACTOR: f32  = 0.3;

const AABB_NODE_FLAG_MOVED: u32 = 1u << 0u;
const AABB_NODE_FLAG_FREE:  u32 = 1u << 1u;
const AABB_NODE_TYPE_INTERNAL: u32 = 0u;
const AABB_NODE_TYPE_LEAF:     u32 = 1u;

//------------------------------------------------------------------------------
// Surface‐area helper
//------------------------------------------------------------------------------
fn calculate_surface_area(min_pt: vec3<f32>, max_pt: vec3<f32>) -> f32 {
    let w = max_pt.x - min_pt.x;
    let h = max_pt.y - min_pt.y;
    let d = max_pt.z - min_pt.z;
    return 2.0 * (w * h + w * d + h * d);
}

//------------------------------------------------------------------------------
// GPU memory layouts (must match JS‐side)
//------------------------------------------------------------------------------
struct AABBNode {
    // word0: flags_and_type (bit0=MOVED, bit1=FREE, bits2+=node_type)
    flags_and_type: atomic<u32>;
    left:           atomic<u32>;
    right:          atomic<u32>;
    parent:         atomic<u32>;
    user_data:      u32;

    // AABB bounds + fat‐bounds (each 2× vec4<f32>)
    min_point: vec4<f32>;
    max_point: vec4<f32>;
    fat_min:   vec4<f32>;
    fat_max:   vec4<f32>;

    // cached subtree‐height
    height: atomic<u32>;
};

struct RingQueue {
    head: atomic<u32>;
    tail: atomic<u32>;
    data: array<u32, QUEUE_CAPACITY>;
};

//------------------------------------------------------------------------------
// Bindings
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read_write> nodes:              array<AABBNode>;
@group(1) @binding(1) var<storage, read_write> bounds:             array<AABBNodeBounds>;
@group(1) @binding(2) var<storage, read_write> dirty_queue:        RingQueue;
@group(1) @binding(3) var<storage, read_write> reinsertion_queue:  RingQueue;
@group(1) @binding(4) var<storage, read_write> balance_queue:      RingQueue;
@group(1) @binding(5) var<storage, read_write> free_list_queue:    RingQueue;
@group(1) @binding(5) var<storage, read_write> root_node:          atomic<u32>;

//------------------------------------------------------------------------------
// Ring‐buffer push/pop (lock‐free, drops on full/empty)
//------------------------------------------------------------------------------
fn ring_queue_push(q: ptr<storage, RingQueue>, idx: u32) {
    loop {
        let tail     = atomicLoad(&(*q).tail);
        let next_tail= (tail + 1u) % QUEUE_CAPACITY;
        let head     = atomicLoad(&(*q).head);
        if (next_tail == head) { return; }            // full
        let old = atomicCompareExchangeWeak(&(*q).tail, tail, next_tail);
        if (old.exchanged) {
            (*q).data[tail] = idx;
            return;
        }
    }
}
fn ring_queue_pop(q: ptr<storage, RingQueue>) -> u32 {
    loop {
        let head = atomicLoad(&(*q).head);
        let tail = atomicLoad(&(*q).tail);
        if (head == tail) { return QUEUE_EMPTY; }     // empty
        let next_head = (head + 1u) % QUEUE_CAPACITY;
        let val = (*q).data[head];
        let old = atomicCompareExchangeWeak(&(*q).head, head, next_head);
        if (old.exchanged) { return val; }
    }
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------
fn clear_moved(idx: u32) {
    let f = atomicLoad(&nodes[idx].flags_and_type);
    atomicStore(&nodes[idx].flags_and_type, f & ~AABB_NODE_FLAG_MOVED);
}

//------------------------------------------------------------------------------
// remove_leaf  (CPU _remove_leaf → GPU)
//------------------------------------------------------------------------------
fn remove_leaf(node_idx: u32) {
    let parent = atomicLoad(&nodes[node_idx].parent);
    atomicStore(&nodes[node_idx].parent, 0u);
    if (parent == 0u) {
        atomicStore(&root_node, 0u);
        return;
    }
    let left    = atomicLoad(&nodes[parent].left);
    let right   = atomicLoad(&nodes[parent].right);
    let sibling = select(right, left, left == node_idx);
    let grand   = atomicLoad(&nodes[parent].parent);

    if (grand == 0u) {
        atomicStore(&root_node, sibling);
        atomicStore(&nodes[sibling].parent, 0u);
    } else {
        let gp_left = atomicLoad(&nodes[grand].left);
        if (gp_left == parent) {
            atomicStore(&nodes[grand].left, sibling);
        } else {
            atomicStore(&nodes[grand].right, sibling);
        }
        atomicStore(&nodes[sibling].parent, grand);
    }

    ring_queue_push(&free_list_queue, parent);
    ring_queue_push(&balance_queue, sibling);
}

//------------------------------------------------------------------------------
// find_best_sibling  (CPU _find_best_sibling → GPU)
//------------------------------------------------------------------------------
fn find_best_sibling(node_idx: u32) -> u32 {
    let node_min = nodes[node_idx].min_point.xyz;
    let node_max = nodes[node_idx].max_point.xyz;

    var best = atomicLoad(&root_node);
    if (best == 0u) { return 0u; }

    // initial cost vs. root
    let rmin = nodes[best].min_point.xyz;
    let rmax = nodes[best].max_point.xyz;
    var comb_min = vec3<f32>(
      min(node_min.x, rmin.x),
      min(node_min.y, rmin.y),
      min(node_min.z, rmin.z)
    );
    var comb_max = vec3<f32>(
      max(node_max.x, rmax.x),
      max(node_max.y, rmax.y),
      max(node_max.z, rmax.z)
    );
    var best_cost = calculate_surface_area(comb_min, comb_max);

    var current = best;
    loop {
        if (current == 0u) { break; }
        let f_and_t = atomicLoad(&nodes[current].flags_and_type);
        let node_type = f_and_t >> 2u;
        let cmin = nodes[current].min_point.xyz;
        let cmax = nodes[current].max_point.xyz;

        // cost if we pair here
        comb_min = vec3<f32>(
          min(node_min.x, cmin.x),
          min(node_min.y, cmin.y),
          min(node_min.z, cmin.z)
        );
        comb_max = vec3<f32>(
          max(node_max.x, cmax.x),
          max(node_max.y, cmax.y),
          max(node_max.z, cmax.z)
        );
        let cost = calculate_surface_area(comb_min, comb_max);
        if (cost < best_cost) {
            best_cost = cost;
            best = current;
        }

        if (node_type == AABB_NODE_TYPE_LEAF) { break; }

        // pick cheapest child to descend
        let left_idx  = atomicLoad(&nodes[current].left);
        let right_idx = atomicLoad(&nodes[current].right);

        var left_cost  = 1e30;
        if (left_idx  != 0u) {
            let lmin = nodes[left_idx].min_point.xyz;
            let lmax = nodes[left_idx].max_point.xyz;
            let lm_min = vec3<f32>(
              min(node_min.x, lmin.x),
              min(node_min.y, lmin.y),
              min(node_min.z, lmin.z)
            );
            let lm_max = vec3<f32>(
              max(node_max.x, lmax.x),
              max(node_max.y, lmax.y),
              max(node_max.z, lmax.z)
            );
            left_cost = calculate_surface_area(lm_min, lm_max)
                      - calculate_surface_area(lmin, lmax);
        }

        var right_cost = 1e30;
        if (right_idx != 0u) {
            let rmin = nodes[right_idx].min_point.xyz;
            let rmax = nodes[right_idx].max_point.xyz;
            let rm_min = vec3<f32>(
              min(node_min.x, rmin.x),
              min(node_min.y, rmin.y),
              min(node_min.z, rmin.z)
            );
            let rm_max = vec3<f32>(
              max(node_max.x, rmax.x),
              max(node_max.y, rmax.y),
              max(node_max.z, rmax.z)
            );
            right_cost = calculate_surface_area(rm_min, rm_max)
                       - calculate_surface_area(rmin, rmax);
        }

        current = select(right_idx, left_idx, left_cost < right_cost);
    }
    return best;
}

//------------------------------------------------------------------------------
// insert_leaf  (CPU _insert_leaf → GPU)
//------------------------------------------------------------------------------
fn insert_leaf(node_idx: u32) {
    // clear flags, set type=LEAF
    atomicStore(&nodes[node_idx].flags_and_type, AABB_NODE_TYPE_LEAF << 2u);

    let root = atomicLoad(&root_node);
    if (root == 0u) {
        atomicStore(&root_node, node_idx);
        atomicStore(&nodes[node_idx].parent, 0u);
        return;
    }

    let sibling = find_best_sibling(node_idx);
    let sib_parent = atomicLoad(&nodes[sibling].parent);

    // grab a free internal node
    let new_parent = ring_queue_pop(&free_list_queue);
    atomicStore(&nodes[new_parent].flags_and_type, AABB_NODE_TYPE_INTERNAL << 2u);
    atomicStore(&nodes[new_parent].parent, sib_parent);

    // wire up children
    atomicStore(&nodes[new_parent].left, sibling);
    atomicStore(&nodes[new_parent].right, node_idx);
    atomicStore(&nodes[sibling].parent, new_parent);
    atomicStore(&nodes[node_idx].parent, new_parent);

    // compute and store combined bounds
    let smin = nodes[sibling].min_point.xyz;
    let smax = nodes[sibling].max_point.xyz;
    let nmin = nodes[node_idx].min_point.xyz;
    let nmax = nodes[node_idx].max_point.xyz;
    let cmin = vec4<f32>(
      min(smin.x, nmin.x),
      min(smin.y, nmin.y),
      min(smin.z, nmin.z),
      0.0
    );
    let cmax = vec4<f32>(
      max(smax.x, nmax.x),
      max(smax.y, nmax.y),
      max(smax.z, nmax.z),
      0.0
    );
    nodes[new_parent].min_point = cmin;
    nodes[new_parent].max_point = cmax;

    // attach new_parent into the old tree
    if (sib_parent == 0u) {
        atomicStore(&root_node, new_parent);
    } else {
        let gp_left = atomicLoad(&nodes[sib_parent].left);
        if (gp_left == sibling) {
            atomicStore(&nodes[sib_parent].left, new_parent);
        } else {
            atomicStore(&nodes[sib_parent].right, new_parent);
        }
    }

    // schedule a refit+balance
    ring_queue_push(&balance_queue, new_parent);
}

//------------------------------------------------------------------------------
// refit & update heights
//------------------------------------------------------------------------------
fn refit_ancestors(start_idx: u32) {
    var cur = start_idx;
    loop {
        if (cur == 0u) { break; }
        let f_and_t  = atomicLoad(&nodes[cur].flags_and_type);
        let node_t   = f_and_t >> 2u;
        if (node_t == AABB_NODE_TYPE_LEAF) {
            cur = atomicLoad(&nodes[cur].parent);
            continue;
        }
        let l = atomicLoad(&nodes[cur].left);
        let r = atomicLoad(&nodes[cur].right);
        var mn = vec3<f32>(1e30,1e30,1e30);
        var mx = vec3<f32>(-1e30,-1e30,-1e30);
        if (l != 0u) {
            mn = vec3<f32>(
              min(mn.x, nodes[l].min_point.x),
              min(mn.y, nodes[l].min_point.y),
              min(mn.z, nodes[l].min_point.z)
            );
            mx = vec3<f32>(
              max(mx.x, nodes[l].max_point.x),
              max(mx.y, nodes[l].max_point.y),
              max(mx.z, nodes[l].max_point.z)
            );
        }
        if (r != 0u) {
            mn = vec3<f32>(
              min(mn.x, nodes[r].min_point.x),
              min(mn.y, nodes[r].min_point.y),
              min(mn.z, nodes[r].min_point.z)
            );
            mx = vec3<f32>(
              max(mx.x, nodes[r].max_point.x),
              max(mx.y, nodes[r].max_point.y),
              max(mx.z, nodes[r].max_point.z)
            );
        }
        nodes[cur].min_point = vec4<f32>(mn,0.0);
        nodes[cur].max_point = vec4<f32>(mx,0.0);
        ring_queue_push(&balance_queue, cur);
        cur = atomicLoad(&nodes[cur].parent);
    }
}
fn update_node_heights(start_idx: u32) {
    var cur = start_idx;
    loop {
        if (cur == 0u) { break; }
        let f_and_t = atomicLoad(&nodes[cur].flags_and_type);
        let t       = f_and_t >> 2u;
        var new_h: u32 = 0u;
        if (t == AABB_NODE_TYPE_LEAF) {
            new_h = 1u;
        } else {
            let l = atomicLoad(&nodes[cur].left);
            let r = atomicLoad(&nodes[cur].right);
            let lh= select(0u, atomicLoad(&nodes[l].height), l != 0u);
            let rh= select(0u, atomicLoad(&nodes[r].height), r != 0u);
            new_h = max(lh, rh) + 1u;
        }
        atomicStore(&nodes[cur].height, new_h);
        cur = atomicLoad(&nodes[cur].parent);
    }
}

//------------------------------------------------------------------------------
// process_leaf_changes  (CPU _process_node_changes → GPU)
//------------------------------------------------------------------------------
@compute @workgroup_size(256)
fn process_leaf_changes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node_idx = ring_queue_pop(&dirty_queue);
    if (node_idx == QUEUE_EMPTY || node_idx >= MAX_NODES) { return; }
    let f = atomicLoad(&nodes[node_idx].flags_and_type);
    if ((f & AABB_NODE_FLAG_FREE) != 0u) { return; }
    if ((f & AABB_NODE_FLAG_MOVED) == 0u) { return; }
    clear_moved(node_idx);

    let mn = nodes[node_idx].min_point.xyz;
    let mx = nodes[node_idx].max_point.xyz;
    let fm = nodes[node_idx].fat_min.xyz;
    let fM = nodes[node_idx].fat_max.xyz;
    if (mn.x < fm.x || mn.y < fm.y || mn.z < fm.z ||
        mx.x > fM.x || mx.y > fM.y || mx.z > fM.z) {
        ring_queue_push(&reinsertion_queue, node_idx);
    }
}

//------------------------------------------------------------------------------
// process_reinsertions  (CPU _process_leaf_reinsert → GPU)
//------------------------------------------------------------------------------
@compute @workgroup_size(256)
fn process_reinsertions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node_idx = ring_queue_pop(&reinsertion_queue);
    if (node_idx == QUEUE_EMPTY || node_idx >= MAX_NODES) { return; }

    // remove
    remove_leaf(node_idx);

    // recompute fat bounds
    let mn = nodes[node_idx].min_point.xyz;
    let mx = nodes[node_idx].max_point.xyz;
    let ext = vec3<f32>(mx - mn) * FAT_MARGIN_FACTOR;
    nodes[node_idx].fat_min = vec4<f32>(mn - ext, 0.0);
    nodes[node_idx].fat_max = vec4<f32>(mx + ext, 0.0);

    // insert
    insert_leaf(node_idx);

    // refit & update heights
    let p = atomicLoad(&nodes[node_idx].parent);
    refit_ancestors(p);
    update_node_heights(node_idx);
}

//------------------------------------------------------------------------------
// rotate & balance (CPU _process_incremental_balancing → GPU)
//------------------------------------------------------------------------------
fn rotate_left(node_idx: u32) {
    let n    = nodes[node_idx];
    let right= atomicLoad(&n.right);
    if (right == 0u) { return; }
    let rt = atomicLoad(&nodes[right].flags_and_type) >> 2u;
    if (rt != AABB_NODE_TYPE_INTERNAL) { return; }

    let rl = atomicLoad(&nodes[right].left);
    let np = atomicLoad(&nodes[node_idx].parent);

    // re-link
    atomicStore(&nodes[right].parent, np);
    atomicStore(&nodes[node_idx].parent, right);

    atomicStore(&nodes[right].left,  node_idx);
    atomicStore(&nodes[node_idx].right, rl);
    if (rl != 0u) {
        atomicStore(&nodes[rl].parent, node_idx);
    }

    if (np == 0u) {
        atomicStore(&root_node, right);
    } else {
        let c = atomicLoad(&nodes[np].left);
        if (c == node_idx) {
            atomicStore(&nodes[np].left, right);
        } else {
            atomicStore(&nodes[np].right, right);
        }
    }

    refit_ancestors(node_idx);
    update_node_heights(node_idx);
}
fn rotate_right(node_idx: u32) {
    let n    = nodes[node_idx];
    let left = atomicLoad(&n.left);
    if (left == 0u) { return; }
    let lt = atomicLoad(&nodes[left].flags_and_type) >> 2u;
    if (lt != AABB_NODE_TYPE_INTERNAL) { return; }

    let lr = atomicLoad(&nodes[left].right);
    let np = atomicLoad(&nodes[node_idx].parent);

    atomicStore(&nodes[left].parent, np);
    atomicStore(&nodes[node_idx].parent, left);

    atomicStore(&nodes[left].right, node_idx);
    atomicStore(&nodes[node_idx].left, lr);
    if (lr != 0u) {
        atomicStore(&nodes[lr].parent, node_idx);
    }

    if (np == 0u) {
        atomicStore(&root_node, left);
    } else {
        let c = atomicLoad(&nodes[np].left);
        if (c == node_idx) {
            atomicStore(&nodes[np].left, left);
        } else {
            atomicStore(&nodes[np].right, left);
        }
    }

    refit_ancestors(node_idx);
    update_node_heights(node_idx);
}

@compute @workgroup_size(256)
fn process_balancing(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node_idx = ring_queue_pop(&balance_queue);
    if (node_idx == QUEUE_EMPTY || node_idx >= MAX_NODES) { return; }
    let t = atomicLoad(&nodes[node_idx].flags_and_type) >> 2u;
    if (t != AABB_NODE_TYPE_INTERNAL) { return; }

    let l = atomicLoad(&nodes[node_idx].left);
    let r = atomicLoad(&nodes[node_idx].right);
    let lh= select(0u, atomicLoad(&nodes[l].height), l != 0u);
    let rh= select(0u, atomicLoad(&nodes[r].height), r != 0u);
    let diff = select(lh - rh, rh - lh, rh > lh);

    if (diff <= 1u) { return; }
    if (lh > rh) { rotate_right(node_idx); }
    else         { rotate_left (node_idx); }
}
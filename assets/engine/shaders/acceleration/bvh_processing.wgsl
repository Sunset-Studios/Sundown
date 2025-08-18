#include "common.wgsl"
#include "acceleration_common.wgsl"

// H-PLOC Step 2: GPU-based acceleration structure construction.

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

//------------------------------------------------------------------------------
// Bindings & Uniforms
//------------------------------------------------------------------------------
@group(1) @binding(0) var<storage, read> bounds: array<AABB>;
@group(1) @binding(1) var<storage, read> aabb_node_indices: array<u32>;
@group(1) @binding(2) var<storage, read_write> sorted_indices: array<u32>;
@group(1) @binding(3) var<storage, read_write> bvh2_nodes: array<BVH2Node>;
@group(1) @binding(4) var<storage, read_write> counters: Counters;
@group(1) @binding(5) var<storage, read_write> clusters_in: array<Cluster>;
@group(1) @binding(6) var<storage, read_write> clusters_out: array<Cluster>;
@group(1) @binding(7) var<uniform> hploc_uniforms: HPLOCUniforms;

//------------------------------------------------------------------------------
// Kernel 1: BVH2 Construction using H-PLOC
//------------------------------------------------------------------------------

var<workgroup> wg_clusters: array<Cluster, HPLOC_WAVE_SIZE>;
var<workgroup> wg_best_costs: array<f32, HPLOC_WAVE_SIZE>;
var<workgroup> wg_best_pairs: array<vec2<i32>, HPLOC_WAVE_SIZE>;
var<workgroup> wg_selected_pair: vec2<i32>;

@compute @workgroup_size(256)
fn initialize_leaf_clusters(@builtin(global_invocation_id) gid: vec3<u32>) {
    let prim_idx = gid.x;
    if (prim_idx >= hploc_uniforms.primitive_count) { return; }

    // TODO: Is this too much indirection? Should we treat the prim index buffer as a sorted aabb node index buffer instead?
    let sorted_prim_idx = sorted_indices[prim_idx];
    let aabb_node_index = aabb_node_indices[sorted_prim_idx];
    if (aabb_node_index == 0) { return; }

    let bound = bounds[aabb_node_index];

    let leaf_node_idx = atomicAdd(&counters.bvh2_count, 1u);

    // q_min_max will be filled later when this leaf is attached to a parent
    bvh2_nodes[leaf_node_idx].q_min_max = vec2<u32>(0u, 0u);
    bvh2_nodes[leaf_node_idx].children = vec2<u32>(0x80000000u | sorted_prim_idx, 0xffffffffu);

    clusters_in[leaf_node_idx].aabb_min_and_node_idx = vec4<f32>(bound.min.xyz, f32(leaf_node_idx));
    clusters_in[leaf_node_idx].aabb_max_and_is_active = vec4<f32>(bound.max.xyz, 1.0);
}

@compute @workgroup_size(HPLOC_WAVE_SIZE)
fn build_bvh2_hploc(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let wave_idx = group_id.x;
    let thread_idx = local_id.x;
    let num_clusters_in = hploc_uniforms.primitive_count;
    let quant_scale = f32(QUANT_MAX);

    // 1. Load wave's clusters into shared memory
    let cluster_global_idx = wave_idx * HPLOC_WAVE_SIZE + thread_idx;
    if (cluster_global_idx < num_clusters_in) {
        wg_clusters[thread_idx] = clusters_in[cluster_global_idx];
    } else {
        wg_clusters[thread_idx].aabb_max_and_is_active.w = 0.0;
    }
    workgroupBarrier();

    // 2. Merge clusters within the wave and selected best pair using parallel reduction
    var active_count = min(HPLOC_WAVE_SIZE, num_clusters_in - wave_idx * HPLOC_WAVE_SIZE);
    for (var i = 0u; i < HPLOC_WAVE_SIZE - 1u && active_count > 1u; i++) {
        var local_best_cost: f32 = 1e38;
        var local_best_pair: vec2<i32> = vec2<i32>(-1, -1);

        // Each active thread evaluates pairs between its cluster and all other active clusters
        if (wg_clusters[thread_idx].aabb_max_and_is_active.w != 0.0) {
            for (var other_thread_idx = 0u; other_thread_idx < HPLOC_WAVE_SIZE; other_thread_idx = other_thread_idx + 1u) {
                if (other_thread_idx == thread_idx || wg_clusters[other_thread_idx].aabb_max_and_is_active.w == 0.0) { continue; }

                let merged = merge_aabbs(
                    wg_clusters[thread_idx].aabb_min_and_node_idx.xyz,
                    wg_clusters[thread_idx].aabb_max_and_is_active.xyz,
                    wg_clusters[other_thread_idx].aabb_min_and_node_idx.xyz,
                    wg_clusters[other_thread_idx].aabb_max_and_is_active.xyz
                );
                let cost = calculate_aabb_surface_area(merged.min.xyz, merged.max.xyz);

                if (cost < local_best_cost) {
                    local_best_cost = cost;
                    local_best_pair = vec2<i32>(i32(thread_idx), i32(other_thread_idx));
                }
            }
        }

        // Write to shared memory
        wg_best_costs[thread_idx] = local_best_cost;
        wg_best_pairs[thread_idx] = local_best_pair; 

        workgroupBarrier();

        // Reduction: stride halves each iteration
        var stride = HPLOC_WAVE_SIZE / 2u;
        while (stride > 0u) {
            if (thread_idx < stride) {
                if (wg_best_costs[thread_idx + stride] < wg_best_costs[thread_idx]) {
                    wg_best_costs[thread_idx] = wg_best_costs[thread_idx + stride];
                    wg_best_pairs[thread_idx] = wg_best_pairs[thread_idx + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }

        // Thread 0 publishes the selected pair
        if (thread_idx == 0u) {
            wg_selected_pair = wg_best_pairs[0];
        }
        workgroupBarrier();

        // Merge the selected pair into a new parent node
        var merge_idx1 = u32(wg_selected_pair.x);
        var merge_idx2 = u32(wg_selected_pair.y);

        let merged = merge_aabbs(
            wg_clusters[merge_idx1].aabb_min_and_node_idx.xyz,
            wg_clusters[merge_idx1].aabb_max_and_is_active.xyz,
            wg_clusters[merge_idx2].aabb_min_and_node_idx.xyz,
            wg_clusters[merge_idx2].aabb_max_and_is_active.xyz
        );
        let child1 = u32(wg_clusters[merge_idx1].aabb_min_and_node_idx.w);
        let child2 = u32(wg_clusters[merge_idx2].aabb_min_and_node_idx.w);

        if (thread_idx == u32(wg_selected_pair.x)) {
          let parent_node_idx = atomicAdd(&counters.bvh2_count, 1u);

          // Write the merged AABB to the parent node
          let parent_min = merged.min.xyz;
          let parent_max = merged.max.xyz;
          let parent_extent = parent_max - parent_min;

          bvh2_nodes[parent_node_idx].q_min_max = vec2<u32>(0u, 0u);
          bvh2_nodes[parent_node_idx].children = vec2<u32>(child1, child2);

          // Encode child 0 quantisation
          let c0_min = wg_clusters[merge_idx1].aabb_min_and_node_idx.xyz;
          let c0_max = wg_clusters[merge_idx1].aabb_max_and_is_active.xyz;
          let rel0_min = clamp((c0_min - parent_min) / parent_extent, vec3f(0.0), vec3f(1.0));
          let rel0_max = clamp((c0_max - parent_min) / parent_extent, vec3f(0.0), vec3f(1.0));
          let q0_min = pack_quant3(
              u32(round(rel0_min.x * quant_scale)),
              u32(round(rel0_min.y * quant_scale)),
              u32(round(rel0_min.z * quant_scale))
          );
          let q0_max = pack_quant3(
              u32(round(rel0_max.x * quant_scale)),
              u32(round(rel0_max.y * quant_scale)),
              u32(round(rel0_max.z * quant_scale))
          );

          // Encode child 1 quantisation
          let c1_min = wg_clusters[merge_idx2].aabb_min_and_node_idx.xyz;
          let c1_max = wg_clusters[merge_idx2].aabb_max_and_is_active.xyz;
          let rel1_min = clamp((c1_min - parent_min) / parent_extent, vec3f(0.0), vec3f(1.0));
          let rel1_max = clamp((c1_max - parent_min) / parent_extent, vec3f(0.0), vec3f(1.0));
          let q1_min = pack_quant3(
              u32(round(rel1_min.x * quant_scale)),
              u32(round(rel1_min.y * quant_scale)),
              u32(round(rel1_min.z * quant_scale))
          );
          let q1_max = pack_quant3(
              u32(round(rel1_max.x * quant_scale)),
              u32(round(rel1_max.y * quant_scale)),
              u32(round(rel1_max.z * quant_scale))
          );

          // Write the quantised AABBs (relative to parent) to the children
          bvh2_nodes[child1].q_min_max = vec2<u32>(q0_min, q0_max);
          bvh2_nodes[child2].q_min_max = vec2<u32>(q1_min, q1_max);

          // Update the surviving/merged clusters in shared memory
          wg_clusters[merge_idx1].aabb_min_and_node_idx = vec4<f32>(merged.min.xyz, f32(parent_node_idx));
          wg_clusters[merge_idx1].aabb_max_and_is_active = vec4<f32>(merged.max.xyz, 1.0);
          wg_clusters[merge_idx2].aabb_max_and_is_active.w = 0.0;
        }
        workgroupBarrier();
    }

    // 3. Write surviving cluster back to global memory
    if (wg_clusters[thread_idx].aabb_max_and_is_active.w > 0.0) {
        clusters_out[wave_idx] = wg_clusters[thread_idx]; // Each wave produces one cluster
    }
}

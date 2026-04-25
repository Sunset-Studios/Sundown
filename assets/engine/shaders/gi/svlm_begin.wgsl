#include "gi/svlm_common.wgsl"

// Bake bootstrap pass.
//
// This is the only pass that derives the SVLM volume from scene state. It uses
// the TLAS root bounds, pads them for probe coverage around the scene, chooses a
// root brick size, and resets all build counters before root nodes are seeded.

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;
@group(1) @binding(2) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(3) var<storage, read> tlas_nodes: array<AABB>;

fn svlm_ceil_div_extent(extent: vec3<f32>, root_size: f32) -> vec3<u32> {
    return vec3<u32>(
        max(1u, u32(ceil(max(extent.x, 0.0) / root_size))),
        max(1u, u32(ceil(max(extent.y, 0.0) / root_size))),
        max(1u, u32(ceil(max(extent.z, 0.0) / root_size)))
    );
}

fn svlm_root_count(dims: vec3<u32>) -> u32 {
    return dims.x * dims.y * dims.z;
}

fn svlm_clear_counters() {
    atomicStore(&svlm_counters.node_count, 0u);
    atomicStore(&svlm_counters.curr_count, 0u);
    atomicStore(&svlm_counters.next_count, 0u);
    atomicStore(&svlm_counters.leaf_count, 0u);
    atomicStore(&svlm_counters.probe_count, 0u);
    atomicStore(&svlm_counters.debug_line_count, 0u);
    atomicStore(&svlm_counters.status, 0u);
    atomicStore(&svlm_counters.max_level_reached, 0u);
    atomicStore(&svlm_counters.current_level, 0u);

    for (var i = 0u; i < 16u; i = i + 1u) {
        atomicStore(&svlm_counters.split_counts[i], 0u);
        atomicStore(&svlm_counters.level_counts[i], 0u);
    }
}

@compute @workgroup_size(1, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    svlm_clear_counters();

    if (tlas_bvh_info.bvh2_count == 0u) {
        return;
    }

    let root_node = tlas_nodes[tlas_bvh_info.bvh2_count - 1u];
    let scene_min = root_node.min.xyz;
    let scene_max = root_node.max.xyz;
    let requested_root_size = svlm_params.requested_root_size;
    let bake_padding = max(svlm_params.bake_padding, 0.0);
    var root_size = max(1.0, requested_root_size);

    var world_min = scene_min - vec3<f32>(max(bake_padding, root_size * 0.25));
    var world_max = scene_max + vec3<f32>(max(bake_padding, root_size * 0.25));
    var dims = svlm_ceil_div_extent(world_max - world_min, root_size);
    var root_count = svlm_root_count(dims);
    var status = 0u;

    // If the requested root grid cannot fit in the node pool, grow the root
    // brick size instead of truncating immediately. The JS side may still grow
    // buffers later, but this keeps the initial root coverage complete.
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (root_count <= svlm_params.max_nodes) {
            break;
        }
        root_size *= 1.25;
        world_min = scene_min - vec3<f32>(max(bake_padding, root_size * 0.25));
        world_max = scene_max + vec3<f32>(max(bake_padding, root_size * 0.25));
        dims = svlm_ceil_div_extent(world_max - world_min, root_size);
        root_count = svlm_root_count(dims);
        status |= SVLM_STATUS_ROOT_OVERFLOW;
    }

    if (root_count > svlm_params.max_nodes) {
        root_count = svlm_params.max_nodes;
        status |= SVLM_STATUS_ROOT_OVERFLOW;
    }

    // Persist the derived volume into the parameter block so later passes and
    // the stats UI all agree on the exact bake bounds used by the GPU.
    svlm_params.world_min_x = world_min.x;
    svlm_params.world_min_y = world_min.y;
    svlm_params.world_min_z = world_min.z;
    svlm_params.root_size = root_size;
    svlm_params.root_dim_x = dims.x;
    svlm_params.root_dim_y = dims.y;
    svlm_params.root_dim_z = dims.z;
    svlm_params.root_count = root_count;
    svlm_params.scene_min_x = scene_min.x;
    svlm_params.scene_min_y = scene_min.y;
    svlm_params.scene_min_z = scene_min.z;
    svlm_params.scene_max_x = scene_max.x;
    svlm_params.scene_max_y = scene_max.y;
    svlm_params.scene_max_z = scene_max.z;

    atomicStore(&svlm_counters.node_count, root_count);
    atomicStore(&svlm_counters.curr_count, root_count);
    atomicStore(&svlm_counters.next_count, 0u);
    atomicStore(&svlm_counters.current_level, 0u);
    atomicStore(&svlm_counters.status, status);
}

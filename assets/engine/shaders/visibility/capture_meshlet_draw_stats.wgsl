#include "visibility/visibility_common.wgsl"

@group(1) @binding(0) var<storage, read> in_draw_command: array<MeshletDrawCommandNoAtomics>;
@group(1) @binding(1) var<storage, read_write> out_stats: array<u32>;

@compute @workgroup_size(1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    out_stats[0] = in_draw_command[0].vertex_count;
    out_stats[1] = in_draw_command[0].instance_count;
    out_stats[2] = in_draw_command[0].first_vertex;
    out_stats[3] = in_draw_command[0].first_instance;
}

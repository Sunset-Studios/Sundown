#include "common.wgsl"

struct DispatchArgs {
    workgroup_count_x: atomic<u32>,
    workgroup_count_y: u32,
    workgroup_count_z: u32,
};

@group(1) @binding(0) var<storage, read_write> dispatch_args: array<DispatchArgs>;

@compute @workgroup_size(256, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&dispatch_args)) {
        return;
    }

    atomicStore(&dispatch_args[gid.x].workgroup_count_x, 0u);
    dispatch_args[gid.x].workgroup_count_y = 1u;
    dispatch_args[gid.x].workgroup_count_z = 1u;
}

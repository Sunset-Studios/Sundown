#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(1) var<storage, read_write> visible_object_instances_no_occlusion: array<i32>;

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let g_id = global_id.x;
    visible_object_instances[g_id] = -1;
    visible_object_instances_no_occlusion[g_id] = -1;
}

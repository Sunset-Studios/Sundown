#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let g_id = global_id.x;
    let length = arrayLength(&draw_indirect_buffer);
    if (g_id >= length) {
        return;
    }
    atomicStore(&draw_indirect_buffer[g_id].instance_count, 0u);
}

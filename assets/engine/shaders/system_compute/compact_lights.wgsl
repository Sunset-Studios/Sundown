#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var<storage, read> lights_buffer: array<Light>;
@group(1) @binding(1) var<storage, read_write> dense_lights_buffer: array<Light>;
@group(1) @binding(2) var<storage, read_write> light_count_buffer: array<atomic<u32>>;

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&lights_buffer)) {
        return;
    }
    let light = lights_buffer[idx];
    if (light.activated > 0.0) {
        let dst = atomicAdd(&light_count_buffer[0], 1u);
        dense_lights_buffer[dst] = light;
    }
} 
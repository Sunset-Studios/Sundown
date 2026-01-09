#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var<storage, read> lights_buffer: array<Light>;
@group(1) @binding(1) var<storage, read_write> dense_lights_buffer: DenseLightsBufferA;

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&lights_buffer)) {
        return;
    }
    let light = lights_buffer[idx];
    if (light.activated > 0.0) {
        let dst = atomicAdd(&dense_lights_buffer.header.light_count, 1u);
        if (light.shadow_casting > 0.0) {
            _ = atomicAdd(&dense_lights_buffer.header.shadow_casting_light_count, 1u);
        }

        // Guard against overflow (drops excess lights, but still increments the counter).
        if (dst < arrayLength(&dense_lights_buffer.lights)) {
            dense_lights_buffer.lights[dst] = light;
        }
    }
} 
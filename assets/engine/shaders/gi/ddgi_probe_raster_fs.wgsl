#include "common.wgsl"
#include "lighting_common.wgsl"

struct ProbeParams {
    probe_view_index: u32;
};
@group(1) @binding(4) var<uniform> probe_params: ProbeParams;

struct VertexOutput {
    @builtin(position) position: vec4<precision_float>;
    @location(0) world_pos: vec3<precision_float>;
};

@fragment fn main_fs(input: VertexOutput) -> @location(0) vec4<precision_float> {
    // TODO: replace this with your direct-light PBR shading logic
    // Sample normals, albedo, etc. from G-Buffer or material data
    // Then for each light in dense_lights, accumulate direct lighting

    // Placeholder: encode world position direction as color
    let color = normalize(input.world_pos);
    return vec4<f32>(abs(color), 1.0);
} 
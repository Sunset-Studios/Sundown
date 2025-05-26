#include "common.wgsl"

// Uniform to index into SharedViewBuffer
struct ProbeParams {
    probe_view_index: u32;
};
@group(1) @binding(4) var<uniform> probe_params: ProbeParams;

struct VertexOutput {
    @builtin(position) position: vec4<precision_float>;
    @location(0) world_pos: vec3<precision_float>;
};

@vertex fn main_vs(
    @builtin(vertex_index) vi: u32
) -> VertexOutput {
    // Fetch the view/projection for this probe from SharedViewBuffer
    let view_data = view_buffer[probe_params.probe_view_index];

    var output: VertexOutput;
    let pos_world4 = vertex_buffer[vi].position;
    output.position = view_data.view_projection_matrix * pos_world4;
    output.world_pos = pos_world4.xyz;
    return output;
} 
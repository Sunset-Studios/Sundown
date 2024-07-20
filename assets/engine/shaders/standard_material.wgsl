#define CUSTOM_VS
#define CUSTOM_FS

#include "gbuffer_base.wgsl"

fn vertex(v_out: VertexOutput) -> VertexOutput {
    return v_out;
}

fn fragment(v_out: VertexOutput, f_out: FragmentOutput) -> FragmentOutput {
    var out : FragmentOutput = f_out;
    out.albedo = vec4f(1.0, 1.0, 0.0, 1.0);
    return out;
}
#define CUSTOM_VS
#define CUSTOM_FS

#include "gbuffer_base.wgsl"

fn vertex(v_out: VertexOutput) -> VertexOutput {
    return v_out;
}

fn fragment(v_out: VertexOutput, f_out: FragmentOutput) -> FragmentOutput {
    var out : FragmentOutput = f_out;
    out.albedo = vec4f(0.2, 1.0, 0.2, 0.0);
    out.smra.r = 0.2;
    out.smra.g = 0.1;
    out.smra.b = 0.7;
    out.smra.a = 0.0;
    return out;
}
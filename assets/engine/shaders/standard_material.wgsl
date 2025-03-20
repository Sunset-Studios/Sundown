#define CUSTOM_FS

#include "gbuffer_base.wgsl"

struct MaterialParams {
    color: vec4<precision_float>,
    emission: vec4<precision_float>,
}

@group(2) @binding(0) var<uniform> material_params: MaterialParams;

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    f_out.albedo = material_params.color;
    f_out.emissive = material_params.emission;

    f_out.smra.r = 255.0;
    f_out.smra.g = 0.7;
    f_out.smra.b = 0.3;
    f_out.smra.a = 0.0;

    return *f_out;
}
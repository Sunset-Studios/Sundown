#define CUSTOM_VS
#define CUSTOM_FS

#include "gbuffer_base.wgsl"

// ------------------------------------------------------------------------------------
// Vertex Shader
// ------------------------------------------------------------------------------------ 
fn vertex(v_out: ptr<function, VertexOutput>) -> VertexOutput {
    // Generate random color based on instance_id
    let h1 = hash(u32(v_out.instance_id));
    let h2 = hash(h1);
    let h3 = hash(h2);
    
    let r = uint_to_normalized_float(h1);
    let g = uint_to_normalized_float(h2);
    let b = uint_to_normalized_float(h3);

    v_out.color = vec4f(r, g, b, 1.0);

    return *v_out;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    f_out.albedo = vec4f(v_out.color.rgb, 0.3);
    f_out.emissive = vec4f(0.1, 0.0, 0.0, 0.0);

    f_out.smra.r = 255.0;
    f_out.smra.g = 0.7;
    f_out.smra.b = 0.3;
    f_out.smra.a = 0.0;

    return *f_out;
}
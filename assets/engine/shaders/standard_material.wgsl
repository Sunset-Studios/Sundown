#define CUSTOM_VS
#define CUSTOM_FS

#include "gbuffer_base.wgsl"

const one_over_float_max = 1.0 / 4294967296.0;

// Improved hash function
fn hash(x: u32) -> u32 {
    var y = x;
    y = y ^ (y >> u32(16));
    y = y * 0x85ebca6bu;
    y = y ^ (y >> u32(13));
    y = y * 0xc2b2ae35u;
    y = y ^ (y >> u32(16));
    return y;
}

// Convert uint to float in [0, 1) range
fn uint_to_float(x: u32) -> f32 {
    return f32(x) * one_over_float_max;
}

fn vertex(v_out: ptr<function, VertexOutput>) -> VertexOutput {
    // Generate random color based on instance_id
    let h1 = hash(u32(v_out.instance_id));
    let h2 = hash(h1);
    let h3 = hash(h2);
    
    let r = uint_to_float(h1);
    let g = uint_to_float(h2);
    let b = uint_to_float(h3);

    v_out.color = vec4f(r, g, b, 1.0);

    return *v_out;
}

fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    f_out.albedo = vec4f(v_out.color.rgb, 0.5);
    f_out.emissive = vec4f(0.7, 0.0, 0.0, 1.0);

    f_out.smra.r = 0.9;
    f_out.smra.g = 0.9;
    f_out.smra.b = 0.1;
    f_out.smra.a = 0.0;

    return *f_out;
}
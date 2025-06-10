#define CUSTOM_FS

#include "gbuffer_base.wgsl"

//------------------------------------------------------------------------------------
// Data Structures
//------------------------------------------------------------------------------------
struct ElementData {
    element_color: vec4f,
    element_emissive: f32,
    element_rounding: f32,
};

//------------------------------------------------------------------------------------
// Buffers / Textures
//------------------------------------------------------------------------------------
@group(2) @binding(0) var<storage, read> element_data: array<ElementData>;

//------------------------------------------------------------------------------------
// Fragment Shader
//------------------------------------------------------------------------------------
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    let element_rounding = element_data[f_out.entity_id].element_rounding;
    let element_emissive = element_data[f_out.entity_id].element_emissive;
    var element_color = element_data[f_out.entity_id].element_color;
    
    // Calculate distance from edges
    let uv = v_out.uv;
    let dx = min(uv.x, 1.0 - uv.x);
    let dy = min(uv.y, 1.0 - uv.y);
    
    // Calculate corner distance in normalized space
    let corner_distance = length(vec2f(
        max(0.0, element_rounding - dx),
        max(0.0, element_rounding - dy)
    )); 
    
    // Apply smoothed alpha based on corner distance
    element_color.a *= 1.0 - smoothstep(0.0, element_rounding, corner_distance);
    
    f_out.albedo = element_color;
    f_out.emissive = vec4f(element_emissive, element_emissive, element_emissive, 0.0);

    return *f_out;
}
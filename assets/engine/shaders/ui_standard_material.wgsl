#define CUSTOM_FS
#define TRANSPARENT

#include "gbuffer_base.wgsl"


//------------------------------------------------------------------------------------
// Data Structures
//------------------------------------------------------------------------------------
struct ElementData {
    color: vec4f,
    emissive: f32,
    rounding: f32,
};

//------------------------------------------------------------------------------------
// Buffers / Textures
//------------------------------------------------------------------------------------
@group(2) @binding(0) var<storage, read> element_data: array<ElementData>;

//------------------------------------------------------------------------------------
// Fragment Shader
//------------------------------------------------------------------------------------
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    let rounding = element_data[f_out.entity_id.y].rounding;
    let emissive = element_data[f_out.entity_id.y].emissive;
    var color = element_data[f_out.entity_id.y].color;
    
    // Calculate distance from edges
    let uv = v_out.uv;
    let dx = min(uv.x, 1.0 - uv.x);
    let dy = min(uv.y, 1.0 - uv.y);
    
    // Calculate corner distance in normalized space
    let corner_distance = length(vec2f(
        max(0.0, rounding - dx),
        max(0.0, rounding - dy)
    )); 
    
    // Apply smoothed alpha based on corner distance
    color.a *= 1.0 - smoothstep(0.0, rounding, corner_distance);
    
    f_out.albedo = color;
    f_out.emissive = vec4f(emissive, emissive, emissive, 0.0);

    return *f_out;
}
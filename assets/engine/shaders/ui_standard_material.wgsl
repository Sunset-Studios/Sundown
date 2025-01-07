#define CUSTOM_FS
#define TRANSPARENT

#include "gbuffer_base.wgsl"


//------------------------------------------------------------------------------------
// Data Structures
//------------------------------------------------------------------------------------
struct ElementData {
    color: vec4f,
};

//------------------------------------------------------------------------------------
// Buffers / Textures
//------------------------------------------------------------------------------------
@group(2) @binding(0) var<storage, read> element_data: array<ElementData>;

//------------------------------------------------------------------------------------
// Fragment Shader
//------------------------------------------------------------------------------------
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    f_out.albedo = element_data[f_out.entity_id.y].color;
    f_out.emissive = vec4f(1.0, 0.0, 0.0, 0.0);

    return *f_out;
}
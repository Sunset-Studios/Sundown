#define CUSTOM_FS
#define CUSTOM_VS

#include "gbuffer_base.wgsl"

//------------------------------------------------------------------------------------
// Data Structures
//------------------------------------------------------------------------------------
struct StringData {
    page_texture_size: vec2<precision_float>,
    color: vec4<precision_float>,
    emissive: precision_float,

};

struct GlyphData {
    width:   u32,
    height:  u32,
    x: i32,
    y: i32,
};

//------------------------------------------------------------------------------------
// Buffers / Textures
//------------------------------------------------------------------------------------
@group(2) @binding(0) var<storage, read> text:            array<u32>;
@group(2) @binding(1) var<storage, read> string_data:     array<StringData>;
@group(2) @binding(2) var<storage, read> font_glyph_data: array<GlyphData>;
@group(2) @binding(3) var font_page_texture: texture_2d<f32>;

//------------------------------------------------------------------------------------
// VERTEX STAGE
// We assume each glyph is drawn with 4 vertices (2 triangles), so we derive
// the corner from `vertex_index & 3` and place it accordingly.
//
//   corner_id | offset (in normalized quad space)
//     0       | (0,0)
//     1       | (1,0)
//     2       | (0,1)
//     3       | (1,1)
//------------------------------------------------------------------------------------
fn vertex(v_out: ptr<function, VertexOutput>) -> VertexOutput {
    let entity               = v_out.base_instance_id;
    let entity_row           = get_entity_row(entity);
    let local_instance_index = compacted_object_instances[v_out.instance_index].entity_instance;

    // Find which character in the text we are drawing
    let string = string_data[entity_row];
    let text_index = entity_row + local_instance_index;
    let ch         = text[text_index];
    let glyph_data = font_glyph_data[ch];

    // Use the provided UV coordinates for corner offset
    var corner_offset = v_out.uv.xy;
    corner_offset.y = 1.0 - corner_offset.y;

    // Calculate UV coordinates for the glyph in the texture atlas
    var uv_top_left = vec2<precision_float>(
        precision_float(glyph_data.x),
        precision_float(glyph_data.y)
    ) / string.page_texture_size;

    let uv_size = vec2<precision_float>(
        precision_float(glyph_data.width),
        precision_float(glyph_data.height)
    ) / string.page_texture_size;

    // Flip Y coordinate and apply the corner offset
    uv_top_left.y = 1.0 - uv_top_left.y - uv_size.y;
    v_out.uv = uv_top_left + corner_offset * uv_size;

    return *v_out;
}

//------------------------------------------------------------------------------------
// FRAGMENT STAGE
// This is where we do the MSDF decoding.  We take the median of the R, G, and B channels,
// offset it by 0.5 (the boundary), and use smoothstep/fwidth for anti-aliasing.
//
// We store the final color/alpha in f_out.albedo (and .a), though you can customize
// how you blend or store this in the g-buffer pipeline.
//------------------------------------------------------------------------------------
fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    // Sample the MSDF texture
    let entity_row = get_entity_row(v_out.base_instance_id);

    let sample_color = vec4<precision_float>(textureSample(font_page_texture, global_sampler, vec2<f32>(v_out.uv)));
    let string_color = string_data[entity_row].color;
    let emissive = string_data[entity_row].emissive;

    let r = sample_color.r;
    let g = sample_color.g;
    let b = sample_color.b;

    // Compute MSDF distance
    let dist = median3(r, g, b);
    // Shift by 0.5 so that the isocontour for the glyph edge is at 0.5
    let sd = f32(dist) - 0.5;

    // fwidth() calculates how quickly 'sd' changes across the pixel,
    // and we use this to create a smooth transition.
    let w = fwidth(sd);

    // Anti-aliased alpha
    let alpha = smoothstep(-w, w, sd);
    if (alpha <= 0.0) {
        discard;
    }

    f_out.albedo = vec4<precision_float>(string_color.rgb, precision_float(alpha));
    f_out.emissive = vec4<precision_float>(emissive, emissive, emissive, 0.0);


    f_out.smra.r = 2555.0;
    f_out.smra.g = 0.5;
    f_out.smra.b = 0.3;
    f_out.smra.a = 0.0;

    return *f_out;
}

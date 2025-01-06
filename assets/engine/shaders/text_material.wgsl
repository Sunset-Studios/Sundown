#define CUSTOM_FS
#define CUSTOM_VS
#define TRANSPARENT

#include "gbuffer_base.wgsl"

//------------------------------------------------------------------------------------
// Data Structures
//------------------------------------------------------------------------------------
struct StringData {
    start: u32,
    count: u32,
    page_texture_size: vec2i,
    font_size: u32,
};

struct GlyphData {
    width:   u32,
    height:  u32,
    offset_x: i32,
    offset_y: i32,
    x: i32,
    y: i32,
    advance: i32,
    page:   u32,
};

//------------------------------------------------------------------------------------
// Buffers / Textures
//------------------------------------------------------------------------------------
@group(2) @binding(0) var<storage, read> text:            array<u32>;
@group(2) @binding(1) var<storage, read> offsets:         array<f32>;
@group(2) @binding(2) var<storage, read> string_data:     array<StringData>;
@group(2) @binding(3) var<storage, read> font_glyph_data: array<GlyphData>;
@group(2) @binding(4) var font_page_texture: texture_2d<f32>;

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
    let local_instance_index = v_out.instance_index - compacted_object_instances[v_out.instance_index].base_instance;

    let string = string_data[entity];
    if (string.count > 0u) {
        // Find which character in the text we are drawing
        let text_index = string.start + local_instance_index;
        let ch         = text[text_index];
        let glyph_data = font_glyph_data[ch];

        // Retrieve the transform for this "string"
        let transform = entity_transforms[entity];

        // Use the provided UV coordinates for corner offset
        let corner_offset = v_out.uv.xy;

        // Calculate the total width of the text block (using the last offset)
        let total_width = offsets[string.start + string.count - 1u];
        
        // Calculate glyph position with proper scaling and offset
        let glyph_local_pos = (vec2f(
            // Center horizontally by subtracting the total width
            offsets[text_index] + f32(glyph_data.offset_x) - (total_width),
            // Center vertically by offsetting by the font size
            f32(glyph_data.offset_y) + f32(glyph_data.height) - f32(string.font_size)
        ) + corner_offset * vec2f(
            f32(glyph_data.width),
            f32(glyph_data.height)
        )) / vec2f(string.page_texture_size) * f32(string.font_size);

        // Transform this local 2D position by the entity's transform
        v_out.world_position = transform.transform * vec4f(glyph_local_pos.x, glyph_local_pos.y, 0.0, 1.0);

        // Calculate UV coordinates for the glyph in the texture atlas
        var uv_top_left = vec2f(
            f32(glyph_data.x),
            f32(glyph_data.y)
        ) / vec2f(string.page_texture_size);

        let uv_size = vec2f(
            f32(glyph_data.width),
            f32(glyph_data.height)
        ) / vec2f(string.page_texture_size);

        // Flip Y coordinate and apply the corner offset
        uv_top_left.y = 1.0 - uv_top_left.y - uv_size.y;
        v_out.uv = uv_top_left + corner_offset * uv_size;
    }

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
    let sample_color = textureSample(font_page_texture, global_sampler, v_out.uv);

    let r = sample_color.r;
    let g = sample_color.g;
    let b = sample_color.b;

    // Compute MSDF distance
    let dist = median3(r, g, b);
    // Shift by 0.5 so that the isocontour for the glyph edge is at 0.5
    let sd = dist - 0.5;

    // fwidth() calculates how quickly 'sd' changes across the pixel,
    // and we use this to create a smooth transition.
    let w = fwidth(sd);

    // Anti-aliased alpha
    let alpha = smoothstep(-w, w, sd);
    if (alpha == 0.0) {
        discard;
    }

    // Write out to the G-Buffer's albedo channel.
    f_out.albedo = vec4f(1.0, 0.1, 0.1, alpha);
    // If you have custom emissive, normal, etc. you can set them as needed:
    f_out.emissive = vec4f(3.0, 0.0, 0.0, 0.0);

    return *f_out;
}

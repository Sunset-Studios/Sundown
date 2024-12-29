#define CUSTOM_FS
#define CUSTOM_VS
#define FINAL_POSITION_WRITE
#define TRANSPARENT

#include "gbuffer_base.wgsl"

struct StringData {
    start: u32,
    count: u32,
}

struct GlyphSizeAndOffset {
    width: u32,
    height: u32,
    offset_x: u32,
    offset_y: u32,
}

struct GlyphPositionAdvanceAndPage {
    x: u32,
    y: u32,
    advance: u32,
    page: u32,
}

struct GlyphData {
    width: u32,
    height: u32,
    offset_x: i32,
    offset_y: i32,
    x: i32,
    y: i32,
    advance: i32,
    page: u32,
}

@group(2) @binding(0) var<storage, read> text: array<u32>;
@group(2) @binding(1) var<storage, read> offsets: array<f32>;
@group(2) @binding(2) var<storage, read> string_data: array<StringData>;
@group(2) @binding(3) var<storage, read> font_glyph_data: array<GlyphData>;
@group(2) @binding(4) var font_page_texture: texture_2d<f32>;

fn vertex(v_out: ptr<function, VertexOutput>) -> VertexOutput {
    let base_instance = compacted_object_instances[v_out.instance_index].base_instance;
    let local_instance_index = v_out.instance_index - base_instance;

    let string = string_data[base_instance];

    if (string.count > 0) {
        let text_index = string.start + local_instance_index;
        let char = text[text_index];
        let strided_char = char * 4;
        let glyph_data = font_glyph_data[strided_char];

        let transform = entity_transforms[base_instance];

        v_out.position = transform.transform * vec4f(offsets[text_index], 0.0, 0.0, 1.0);
    }

    return *v_out;
}

fn fragment(v_out: VertexOutput, f_out: ptr<function, FragmentOutput>) -> FragmentOutput {
    f_out.albedo = vec4f(0.0, 0.1, 0.4, 0.7);
    f_out.emissive = vec4f(2.0, 0.0, 0.0, 0.0);

    // Set the normal w component to 0.0 to indicate that custom lighting is used
    f_out.normal.w = 0.0;

    return *f_out;
}
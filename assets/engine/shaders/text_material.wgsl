#define CUSTOM_DEPTH_VS
#define CUSTOM_RASTER_VS
#define CUSTOM_DEPTH_FRAGMENT_MASK
#define CUSTOM_RASTER_FRAGMENT_MASK
#define CUSTOM_RESOLVE_FRAGMENT

#include "visibility/visibility_draw_base.wgsl"

struct StringData {
    text_color: vec4<f32>,
    page_texture_size: vec2<f32>,
    text_emissive: f32,
};

struct GlyphData {
    width: u32,
    height: u32,
    x: i32,
    y: i32,
};

@group(2) @binding(0) var<storage, read> text: array<u32>;
@group(2) @binding(1) var<storage, read> string_data: array<StringData>;
@group(2) @binding(2) var<storage, read> font_glyph_data: array<GlyphData>;
@group(2) @binding(3) var font_page_texture: texture_2d<f32>;

fn atlas_uv(entity_row: u32, corner_uv: vec2<f32>) -> vec2<f32> {
    let string = string_data[entity_row];
    let glyph_data = font_glyph_data[text[entity_row]];

    var corner_offset = corner_uv;
    corner_offset.y = 1.0 - corner_offset.y;

    var uv_top_left = vec2<f32>(f32(glyph_data.x), f32(glyph_data.y)) / string.page_texture_size;
    let uv_size = vec2<f32>(f32(glyph_data.width), f32(glyph_data.height)) / string.page_texture_size;

    uv_top_left.y = 1.0 - uv_top_left.y - uv_size.y;
    return uv_top_left + corner_offset * uv_size;
}

fn text_alpha(entity_row: u32, uv: vec2<f32>) -> f32 {
    let sample_color = textureSample(font_page_texture, global_sampler, uv);

    let dist = median3(sample_color.r, sample_color.g, sample_color.b);
    let sd = dist - 0.5;
    let w = fwidth(sd);

    return smoothstep(-w, w, sd);
}

fn depth_vertex(v_out: ptr<function, DepthVertexOutput>) -> DepthVertexOutput {
    v_out.uv = atlas_uv(v_out.entity_id, v_out.uv);
    return *v_out;
}

fn raster_vertex(v_out: ptr<function, RasterVertexOutput>) -> RasterVertexOutput {
    v_out.uv = atlas_uv(v_out.entity_id, v_out.uv);
    return *v_out;
}

fn depth_fragment_mask(input: DepthVertexOutput) -> f32 {
    return text_alpha(input.entity_id, input.uv);
}

fn raster_fragment_mask(input: RasterVertexOutput) -> f32 {
    return text_alpha(input.entity_id, input.uv);
}

fn resolve_fragment(
    input: ResolveFragmentInput,
    f_out: ptr<function, ResolveFragmentOutput>
) -> ResolveFragmentOutput {
    let entity_row = input.entity_id;
    let glyph_uv = atlas_uv(entity_row, input.uv);
    let alpha = text_alpha(entity_row, glyph_uv);
    let string = string_data[entity_row];

    f_out.albedo = vec4<f32>(string.text_color.rgb, alpha);
    f_out.smra = vec4<f32>(1.0, 0.5, 0.1, 1.0);
    f_out.motion_emissive.a = string.text_emissive;

    return *f_out;
}

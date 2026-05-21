#define CUSTOM_DEPTH_VS
#define CUSTOM_RASTER_VS
#define CUSTOM_DEPTH_FRAGMENT_MASK
#define CUSTOM_RASTER_FRAGMENT_MASK
#define CUSTOM_FORWARD_FRAGMENT
#define CUSTOM_RESOLVE_FRAGMENT
#define CUSTOM_RESOLVE_WORLD_POSITION

#include "visibility/visibility_draw_base.wgsl"

struct GlyphData {
    width: u32,
    height: u32,
    x: i32,
    y: i32,
};

struct UIData {
    origin: vec4<f32>,
    x_axis: vec4<f32>,
    y_axis: vec4<f32>,
    uv_rect: vec4<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    params: vec4<f32>,
};

@group(2) @binding(0) var<storage, read> ui_text_glyphs: array<u32>;
@group(2) @binding(1) var<storage, read> font_glyph_data: array<GlyphData>;
@group(2) @binding(2) var font_page_texture: texture_2d<f32>;
@group(2) @binding(3) var<storage, read> ui_data: array<UIData>;

fn atlas_uv(entity_row: u32, corner_uv: vec2<f32>) -> vec2<f32> {
    let data = ui_data[entity_row];
    let glyph_data = font_glyph_data[ui_text_glyphs[entity_row]];

    var corner_offset = corner_uv;
    corner_offset.y = 1.0 - corner_offset.y;

    let page_texture_size = max(data.uv_rect.zw, vec2<f32>(1.0));
    var uv_top_left = vec2<f32>(f32(glyph_data.x), f32(glyph_data.y)) / page_texture_size;
    let uv_size = vec2<f32>(f32(glyph_data.width), f32(glyph_data.height)) / page_texture_size;

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

fn text_coverage_mask(position: vec4<f32>, entity_row: u32, uv: vec2<f32>) -> f32 {
    let alpha = text_alpha(entity_row, atlas_uv(entity_row, uv));
#if TRANSPARENT
    return alpha;
#else
    return alpha - dither_mask(position.xy / frame_info.resolution, frame_info.resolution);
#endif
}

fn ui_3d_local_position(uv: vec2<f32>, entity_id: u32) -> vec4<f32> {
    let instance = ui_data[entity_id];
    return vec4<f32>(
        instance.origin.xyz +
        instance.x_axis.xyz * uv.x +
        instance.y_axis.xyz * uv.y,
        1.0
    );
}

fn ui_3d_world_position(uv: vec2<f32>, entity_id: u32, entity_transform: EntityTransform) -> vec4<f32> {
    return entity_transform.transform * ui_3d_local_position(uv, entity_id);
}

#if MESHLET_DEPTH_PASS
fn depth_vertex(v_out: ptr<function, DepthVertexOutput>) -> DepthVertexOutput {
    var output = *v_out;
    if (output.entity_id == INVALID_IDX) {
        return output;
    }
    let view_index = u32(frame_info.view_index);
    output.position = view_buffer[view_index].view_projection_matrix *
        ui_3d_world_position(output.uv, output.entity_id, entity_transforms[output.entity_id]);
    return output;
}

fn depth_fragment_mask(input: DepthVertexOutput) -> f32 {
    return text_coverage_mask(input.position, input.entity_id, input.uv);
}

#endif

#if MESHLET_RASTER_PASS
fn raster_vertex(v_out: ptr<function, RasterVertexOutput>) -> RasterVertexOutput {
    var output = *v_out;
    if (output.entity_id == INVALID_IDX) {
        return output;
    }
    let view_index = u32(frame_info.view_index);
    output.position = view_buffer[view_index].view_projection_matrix *
        ui_3d_world_position(output.uv, output.entity_id, entity_transforms[output.entity_id]);
    return output;
}

fn raster_fragment_mask(input: RasterVertexOutput) -> f32 {
    return text_coverage_mask(input.position, input.entity_id, input.uv);
}

fn forward_fragment(
    input: RasterVertexOutput,
    f_out: ptr<function, ForwardFragmentOutput>
) -> ForwardFragmentOutput {
    let data = ui_data[input.entity_id];
    let alpha = text_alpha(input.entity_id, atlas_uv(input.entity_id, input.uv)) * data.color.a;
    f_out.color = vec4<f32>(data.color.rgb, alpha);
    return *f_out;
}

#endif

#if MESHLET_RESOLVE_PASS
fn resolve_world_position(
    decoded: DecodedVertex,
    entity_id: u32,
    entity_transform: EntityTransform
) -> vec4<f32> {
    return ui_3d_world_position(decoded.uv, entity_id, entity_transform);
}

fn resolve_fragment(
    input: ResolveFragmentInput,
    f_out: ptr<function, ResolveFragmentOutput>
) -> ResolveFragmentOutput {
    let entity_row = input.entity_id;
    let glyph_uv = atlas_uv(entity_row, input.uv);
    let alpha = text_alpha(entity_row, glyph_uv);
    let data = ui_data[entity_row];

    f_out.albedo = vec4<f32>(data.color.rgb, alpha * data.color.a);
    f_out.smra = vec4<f32>(1.0, 0.5, 0.1, 1.0);
    f_out.normal = vec4<f32>(
        safe_normalize(cross(data.x_axis.xyz, data.y_axis.xyz)),
        1.0
    );
    f_out.motion_emissive.a = data.params.z;

    return *f_out;
}

#endif

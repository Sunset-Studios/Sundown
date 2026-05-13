#define CUSTOM_RASTER_PASS_BINDINGS
#define CUSTOM_RESOLVE_PASS_BINDINGS
#define CUSTOM_DEPTH_VERTEX_OUTPUT
#define CUSTOM_RASTER_VERTEX_OUTPUT
#define CUSTOM_DEPTH_VS
#define CUSTOM_RASTER_VS
#define CUSTOM_DEPTH_FRAGMENT_MASK
#define CUSTOM_RASTER_FRAGMENT_MASK
#define CUSTOM_RASTER_FRAGMENT
#define CUSTOM_RESOLVE_VISIBILITY_FRAGMENT
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

struct Immediate3DUIInstance {
    origin: vec4<f32>,
    x_axis: vec4<f32>,
    y_axis: vec4<f32>,
    uv_rect: vec4<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    params: vec4<f32>,
};

struct Immediate3DUIVertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) entity_id: u32,
    @location(2) world_position: vec3<f32>,
    @location(3) normal: vec3<f32>,
};

struct Immediate3DUIRasterOutput {
    @location(0) entity_id: u32,
    @location(1) surface: u32,
    @location(2) bucket: u32,
};

@group(2) @binding(0) var<storage, read> text: array<u32>;
@group(2) @binding(1) var<storage, read> string_data: array<StringData>;
@group(2) @binding(2) var<storage, read> font_glyph_data: array<GlyphData>;
@group(2) @binding(3) var font_page_texture: texture_2d<f32>;
@group(2) @binding(4) var<storage, read> immediate_ui_instances: array<Immediate3DUIInstance>;

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

fn pack_immediate_3d_ui_uv(uv: vec2<f32>) -> u32 {
    let packed = vec2<u32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * 65535.0);
    return (packed.y << 16u) | packed.x;
}

fn unpack_immediate_3d_ui_uv(value: u32) -> vec2<f32> {
    return vec2<f32>(f32(value & 0xffffu), f32(value >> 16u)) / 65535.0;
}

fn immediate_3d_ui_world_position(vi: u32, ii: u32) -> vec4<f32> {
    let instance = immediate_ui_instances[ii];
    let uv = vertex_uv(vertex_buffer[vi]);
    return vec4<f32>(
        instance.origin.xyz +
        instance.x_axis.xyz * uv.x +
        instance.y_axis.xyz * uv.y,
        1.0
    );
}

#if MESHLET_DEPTH_PASS
fn depth_vertex(v_out: ptr<function, DepthVertexOutput>) -> DepthVertexOutput {
    return *v_out;
}

fn depth_fragment_mask(input: DepthVertexOutput) -> f32 {
    return text_alpha(input.entity_id, atlas_uv(input.entity_id, input.uv));
}

fn build_depth_vertex_output(vi: u32, ii: u32) -> DepthVertexOutput {
    let uv = vertex_uv(vertex_buffer[vi]);
    let view_index = u32(frame_info.view_index);

    var output: DepthVertexOutput;
    output.position = view_buffer[view_index].view_projection_matrix * immediate_3d_ui_world_position(vi, ii);
    output.uv = uv;
    output.entity_id = ii;
    output.section_index = 0u;
    return output;
}
#endif

#if MESHLET_RASTER_PASS
fn raster_vertex(v_out: ptr<function, RasterVertexOutput>) -> RasterVertexOutput {
    return *v_out;
}

fn raster_fragment_mask(input: RasterVertexOutput) -> f32 {
    return text_alpha(input.entity_id, atlas_uv(input.entity_id, input.uv));
}

fn build_raster_vertex_output(vi: u32, ii: u32) -> RasterVertexOutput {
    let uv = vertex_uv(vertex_buffer[vi]);
    let view_index = u32(frame_info.view_index);

    var output: RasterVertexOutput;
    output.position = view_buffer[view_index].view_projection_matrix * immediate_3d_ui_world_position(vi, ii);
    output.uv = uv;
    output.entity_id = ii;
    output.section_index = 0u;
    output.barycentric = vec2<f32>(0.0);
    output.meshlet_index = 0u;
    output.triangle_index = 0u;
    return output;
}

fn raster_fragment(input: RasterVertexOutput, f_out: ptr<function, RasterFragmentOutput>) -> RasterFragmentOutput {
    f_out.surface = pack_immediate_3d_ui_uv(input.uv);
    return *f_out;
}
#endif

#if MESHLET_RESOLVE_PASS
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

fn resolve_visibility_fragment(input: ResolveVertexOutput) -> ResolveFragmentOutput {
    let resolution = vec2<u32>(u32(frame_info.resolution.x), u32(frame_info.resolution.y));
    let pixel_coord = uv_to_coord(input.uv, resolution);
    let entity_id = textureLoad(visibility_entity_texture, pixel_coord, 0).x;
    if (entity_id == INVALID_IDX) {
        discard;
    }

    let visibility_bucket = textureLoad(visibility_bucket_texture, pixel_coord, 0).x;
    if (visibility_bucket != visibility_bucket_info.current_visibility_bucket) {
        discard;
    }

    let uv = unpack_immediate_3d_ui_uv(textureLoad(visibility_surface_texture, pixel_coord, 0).x);
    let instance = immediate_ui_instances[entity_id];
    let world_position = vec4<f32>(
        instance.origin.xyz +
        instance.x_axis.xyz * uv.x +
        instance.y_axis.xyz * uv.y,
        1.0
    );
    let view_index = u32(frame_info.view_index);
    let current_clip_pos = view_buffer[view_index].view_projection_matrix * world_position;
    let normal = safe_normalize(cross(instance.x_axis.xyz, instance.y_axis.xyz));

    var material_input: ResolveFragmentInput;
    material_input.screen_uv = input.uv;
    material_input.device_depth = textureLoad(depth_texture, pixel_coord, 0).x;
    material_input.entity_id = entity_id;
    material_input.section_index = 0u;
    material_input.meshlet_index = 0u;
    material_input.triangle_index = 0u;
    material_input.barycentric = vec3<f32>(0.0, 0.0, 1.0);
    material_input.local_position = vec4<f32>(uv, 0.0, 1.0);
    material_input.prev_local_position = material_input.local_position;
    material_input.world_position = world_position;
    material_input.prev_world_position = world_position;
    material_input.uv = uv;
    material_input.normal = vec4<f32>(normal, 0.0);
    material_input.tangent = vec4<f32>(safe_normalize(instance.x_axis.xyz), 0.0);
    material_input.bitangent = vec4<f32>(safe_normalize(instance.y_axis.xyz), 0.0);
    material_input.current_clip_pos = current_clip_pos;
    material_input.prev_clip_pos = current_clip_pos;

    var output: ResolveFragmentOutput;
    output.albedo = vec4<f32>(0.0);
    output.smra = vec4<f32>(1.0, 0.5, 0.1, 1.0);
    output.normal = vec4<f32>(normal, 1.0);
    output.motion_emissive = vec4<f32>(0.0);
    output = resolve_fragment(material_input, &output);

#if TRANSPARENT
    let alpha = clamp(output.albedo.a, 0.0, 1.0);
    let weight = clamp(pow(min(1.0, alpha * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - material_input.current_clip_pos.z * 0.9, 3.0), 1e-2, 3e3);
    output.albedo = vec4<f32>(output.albedo.rgb * alpha, alpha) * weight;
#endif

    return output;
}
#endif
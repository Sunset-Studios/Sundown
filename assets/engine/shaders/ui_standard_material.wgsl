#define CUSTOM_DEPTH_VS
#define CUSTOM_RASTER_VS
#define CUSTOM_DEPTH_FRAGMENT_MASK
#define CUSTOM_RASTER_FRAGMENT_MASK
#define CUSTOM_FORWARD_FRAGMENT
#define CUSTOM_RESOLVE_FRAGMENT
#define CUSTOM_RESOLVE_WORLD_POSITION

#include "visibility/visibility_draw_base.wgsl"

//------------------------------------------------------------------------------------
// Data Structures
//------------------------------------------------------------------------------------
struct UIData {
    origin: vec4<f32>,
    x_axis: vec4<f32>,
    y_axis: vec4<f32>,
    uv_rect: vec4<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    params: vec4<f32>,
};

//------------------------------------------------------------------------------------
// Buffers / Textures
//------------------------------------------------------------------------------------
@group(2) @binding(0) var<storage, read> ui_data: array<UIData>;

fn element_alpha(entity_id: u32, uv: vec2<f32>) -> f32 {
    let data = ui_data[entity_id];
    let element_rounding = data.params.x;
    let dx = min(uv.x, 1.0 - uv.x);
    let dy = min(uv.y, 1.0 - uv.y);
    let corner_distance = length(vec2<f32>(
        max(0.0, element_rounding - dx),
        max(0.0, element_rounding - dy)
    ));
    return data.color.a *
        (1.0 - smoothstep(0.0, element_rounding, corner_distance));
}

fn element_coverage_mask(position: vec4<f32>, entity_id: u32, uv: vec2<f32>) -> f32 {
    let alpha = element_alpha(entity_id, uv);
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
    return element_coverage_mask(input.position, input.entity_id, input.uv);
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
    return element_coverage_mask(input.position, input.entity_id, input.uv);
}

fn forward_fragment(
    input: RasterVertexOutput,
    f_out: ptr<function, ForwardFragmentOutput>
) -> ForwardFragmentOutput {
    let data = ui_data[input.entity_id];
    let alpha = element_alpha(input.entity_id, input.uv);
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
    let data = ui_data[input.entity_id];
    let normal = safe_normalize(cross(data.x_axis.xyz, data.y_axis.xyz));

    f_out.albedo = data.color;
    f_out.albedo.a *= element_alpha(input.entity_id, input.uv);
    f_out.smra = vec4<f32>(1.0, 0.5, 0.1, 1.0);
    f_out.normal = vec4<f32>(normal, 1.0);
    f_out.motion_emissive.a = data.params.z;

    return *f_out;
}
#endif

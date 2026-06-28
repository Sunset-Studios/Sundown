// =============================================================================
// DDGI Probe Surface Feedback
// Recomputes the current-frame probe active set from visible depth-buffer
// surfaces. Each visible surface sample marks its trilinear probe neighborhood.
// =============================================================================

#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var hzb_texture: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(3) var<storage, read_write> probe_surface_flags: array<u32>;

const DDGI_SURFACE_FEEDBACK_SKY_NORMAL_LENGTH_SQ_EPS: f32 = 1e-10;

fn ddgi_mark_surface_probe_neighborhood(
    position: vec3<f32>,
    normal_ws: vec3<f32>,
    cascade_index: u32
) {
    let dims = vec3<u32>(
        u32(ddgi_params.probe_grid_dims.x),
        u32(ddgi_params.probe_grid_dims.y),
        u32(ddgi_params.probe_grid_dims.z)
    );
    let spacing = ddgi_cascade_spacing(&ddgi_params, cascade_index);
    let origin = ddgi_cascade_origin(&ddgi_params, cascade_index);
    let view_index = u32(frame_info.view_index);
    let camera_position = view_buffer[view_index].view_position.xyz;
    let bias_offset = (normal_ws * 0.2 + safe_normalize(camera_position - position) * 0.8) * (0.75 * spacing);
    let offset_pos = position + bias_offset;
    let rel = (offset_pos - origin) / spacing;
    let base = floor(rel);

    let trilinear_index_offsets: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
        vec3<i32>(0, 0, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(1, 1, 0),
        vec3<i32>(1, 0, 0),
        vec3<i32>(0, 0, 1),
        vec3<i32>(0, 1, 1),
        vec3<i32>(1, 1, 1),
        vec3<i32>(1, 0, 1),
    );

    let max_coord = vec3<i32>(dims) - vec3<i32>(1);
    for (var i = 0; i < 8; i = i + 1) {
        let coord = vec3<i32>(base) + trilinear_index_offsets[i];
        let clamped_coord = vec3<u32>(clamp(coord, vec3<i32>(0), max_coord));
        let probe_index = ddgi_probe_index_from_coord(&ddgi_params, cascade_index, clamped_coord);
        probe_surface_flags[probe_index] = 1u;
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hzb_dims = textureDimensions(hzb_texture);
    let normal_dims = textureDimensions(gbuffer_normal);
    let pixel = vec2<u32>(gid.xy);
    if (pixel.x >= hzb_dims.x || pixel.y >= hzb_dims.y || pixel.x >= normal_dims.x || pixel.y >= normal_dims.y) {
        return;
    }

    let pixel_i = vec2<i32>(i32(pixel.x), i32(pixel.y));
    let normal_data = textureLoad(gbuffer_normal, pixel_i, 0);
    if (dot(normal_data.xyz, normal_data.xyz) <= DDGI_SURFACE_FEEDBACK_SKY_NORMAL_LENGTH_SQ_EPS) {
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(hzb_dims);
    let depth = textureLoad(hzb_texture, pixel_i, 0).r;
    let position = reconstruct_world_position(uv, depth, u32(frame_info.view_index));
    let normal_ws = safe_normalize(normal_data.xyz);
    let cascade_index = ddgi_cascade_index_for_position(&ddgi_params, position);

    ddgi_mark_surface_probe_neighborhood(position, normal_ws, cascade_index);

    let coarser_cascade_index = cascade_index + 1u;
    if (
        coarser_cascade_index < ddgi_cascade_count(&ddgi_params) &&
        ddgi_position_in_coarser_active_overlap(&ddgi_params, cascade_index, coarser_cascade_index, position)
    ) {
        ddgi_mark_surface_probe_neighborhood(position, normal_ws, coarser_cascade_index);
    }
}

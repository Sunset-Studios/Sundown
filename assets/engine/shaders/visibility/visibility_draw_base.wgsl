#include "common.wgsl"
#include "visibility/visibility_common.wgsl"

struct DepthVertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) entity_id: u32,
    @location(2) @interpolate(flat) section_index: u32,
};

struct RasterVertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) entity_id: u32,
    @location(2) @interpolate(flat) section_index: u32,
    @location(3) barycentric: vec2<f32>,
    @location(4) @interpolate(flat) meshlet_index: u32,
    @location(5) @interpolate(flat) triangle_index: u32,
};

struct RasterFragmentOutput {
    @location(0) entity_id: u32,
    @location(1) surface: u32,
    @location(2) barycentric: u32,
}

struct ResolveVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct ResolveFragmentInput {
    screen_uv: vec2<f32>,
    device_depth: f32,
    entity_id: u32,
    section_index: u32,
    meshlet_index: u32,
    triangle_index: u32,
    barycentric: vec3<f32>,
    local_position: vec4<f32>,
    prev_local_position: vec4<f32>,
    world_position: vec4<f32>,
    prev_world_position: vec4<f32>,
    uv: vec2<f32>,
    normal: vec4<f32>,
    tangent: vec4<f32>,
    bitangent: vec4<f32>,
    current_clip_pos: vec4<f32>,
    prev_clip_pos: vec4<f32>,
};

struct ResolveFragmentOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) smra: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) motion_emissive: vec4<f32>,
};

struct VisibilityBucketInfo {
    current_visibility_bucket: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

#include "visibility/visibility_draw_pipeline.wgsl"

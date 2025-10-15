#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct VertexOutput {
    @builtin(position) @invariant position: vec4f,
    @location(0) local_position: vec4<precision_float>,
    @location(1) view_position: vec4<f32>,
    @location(2) world_position: vec4<f32>,
    @location(3) color: vec4<precision_float>,
    @location(4) uv: vec2<precision_float>,
    @location(5) normal: vec4<precision_float>,
    @location(6) tangent: vec4<precision_float>,
    @location(7) bitangent: vec4<precision_float>,
    @location(8) prev_clip_pos: vec4<f32>,
    @location(9) current_clip_pos: vec4<f32>,
    @location(10) prev_world_position: vec4<f32>,
    @location(11) @interpolate(flat) instance_index: u32,
    @location(12) @interpolate(flat) instance_id: u32,
    @location(13) @interpolate(flat) vertex_index: u32,
};

#ifndef DEPTH_ONLY

struct FragmentOutput {
    @location(0) albedo: vec4<precision_float>,
    @location(1) smra: vec4<precision_float>,
    @location(2) position: vec4<f32>,
    @location(3) normal: vec4<precision_float>,
    @location(4) motion_emissive: vec4<precision_float>,
}

#else

struct FragmentOutput {
    @location(0) entity_id: u32,
}

#endif

#include "gbuffer_pipeline.wgsl"

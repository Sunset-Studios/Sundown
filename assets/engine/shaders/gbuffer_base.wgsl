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
    @location(8) @interpolate(flat) instance_index: u32,
    @location(9) @interpolate(flat) instance_id: u32,
    @location(10) @interpolate(flat) vertex_index: u32,
};

#ifndef DEPTH_ONLY

struct FragmentOutput {
    @location(0) albedo: vec4<precision_float>,
    @location(1) emissive: vec4<precision_float>,
    @location(2) smra: vec4<precision_float>,
    @location(3) position: vec4<f32>,
    @location(4) normal: vec4<precision_float>,
#ifndef SKIP_ENTITY_WRITES
    @location(5) entity_id: vec2<u32>,
#endif
#if TRANSPARENT
    @location(transparency_reveal_location) transparency_reveal: f32,
#endif
}

#ifndef SKIP_ENTITY_WRITES
const transparency_reveal_location = 6;
#else
const transparency_reveal_location = 5;
#endif

#endif

#include "gbuffer_pipeline.wgsl"

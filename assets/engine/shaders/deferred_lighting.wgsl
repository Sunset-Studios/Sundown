#include "common.wgsl"

@group(1) @binding(0) var skybox_texture: texture_2d<f32>;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;
@group(1) @binding(2) var smra_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;
@group(1) @binding(4) var position_texture: texture_2d<f32>;
@group(1) @binding(5) var depth_texture: texture_depth_2d;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct FragmentOutput {
    @location(0) color: vec4f,
};

@vertex fn vs(
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
) -> VertexOutput {
    var output : VertexOutput;
    output.position = vertex_buffer[vi].position;
    output.uv = vertex_buffer[vi].uv;
    return output;
}

@fragment fn fs(v_out: VertexOutput) -> FragmentOutput {
    var tex_sky = textureSample(skybox_texture, global_sampler, v_out.uv);
    var tex_albedo = textureSample(albedo_texture, global_sampler, v_out.uv);

	var tex_normal = textureSample(normal_texture, global_sampler, v_out.uv);
    var normal = tex_normal.xyz;
	var normal_length = length(normal);
	var normalized_normal = normal / normal_length;

    let unlit = f32(normal_length <= 0.0);

    var color = unlit * tex_sky;

    color += tex_albedo;

    return FragmentOutput(color);
}
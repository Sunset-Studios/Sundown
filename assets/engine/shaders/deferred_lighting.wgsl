#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var skybox_texture: texture_2d<f32>;
@group(1) @binding(1) var albedo_texture: texture_2d<f32>;
@group(1) @binding(2) var smra_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;
@group(1) @binding(4) var position_texture: texture_2d<f32>;
@group(1) @binding(5) var depth_texture: texture_depth_2d;
@group(1) @binding(6) var<storage, read> lights_buffer: array<Light>;

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
    let ambient = vec3f(0.1, 0.1, 0.1);

    var tex_sky = textureSample(skybox_texture, global_sampler, v_out.uv);

    var tex_albedo = textureSample(albedo_texture, global_sampler, v_out.uv);
    var albedo = tex_albedo.xyz;
    var emissive = tex_albedo.w;
    
	var tex_normal = textureSample(normal_texture, global_sampler, v_out.uv);
    var normal = tex_normal.xyz;
	var normal_length = length(normal);
	var normalized_normal = normal / normal_length;

    var tex_smra = textureSample(smra_texture, global_sampler, v_out.uv);
    var reflectance = tex_smra.r * 0.0009765625 /* 1.0f / 1024 */;
    var metallic = tex_smra.g;
    var roughness = tex_smra.b;
    var ao = tex_smra.a;

    var tex_position = textureSample(position_texture, global_sampler, v_out.uv);
    var position = tex_position.xyz;

    var view_dir = normalize(view_buffer[0].view_matrix[3].xyz - position);
    var reflection = reflect(-view_dir, normalized_normal);

    let unlit = f32(normal_length <= 0.0);

    var color = unlit * tex_sky.rgb;

    let num_lights = arrayLength(&lights_buffer);
    for (var i = 0u; i < num_lights; i++) {
        var light = lights_buffer[i];
        color += calculate_brdf(
            light,
            normal,
            view_dir,
            position,
            albedo,
            roughness,
            metallic,
            reflectance,
            0.0, // clear coat
            1.0, // clear coat roughness 
            ao,
            vec3f(1.0, 1.0, 1.0), // irradiance
            vec3f(1.0, 1.0, 1.0), // prefilter color 
            vec2f(1.0, 1.0), // env brdf
            0, // shadow map index
        );
    }

    color += (emissive * albedo);

    return FragmentOutput(vec4f(color, 1.0));
}
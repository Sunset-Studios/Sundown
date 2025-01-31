#include "common.wgsl"
#include "postprocess_common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct VhsParams {
    noise_intensity: f32,
    scanline_intensity: f32,
    color_bleeding: f32,
    distortion_frequency: f32,
    distortion_amplitude: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

struct FragmentOutput {
    @location(0) color: vec4<precision_float>,
};

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<uniform> vhs_params: VhsParams;
@group(1) @binding(1) var input_texture: texture_2d<f32>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

// VHS noise function
fn vhs_noise(uv: vec2f, time: f32) -> f32 {
    let noise1 = hash(u32(uv.x * 35.34 + uv.y * 2375.23 + time * 2));
    let noise2 = hash(u32(uv.x * 256.0 + uv.y * 1437.0 + time * 3.14));
    return mix(uint_to_normalized_float(noise1), uint_to_normalized_float(noise2), 0.5);
}

// Vertical color bleeding
fn color_bleeding(uv: vec2f, color: vec3f, intensity: f32) -> vec3f {
    let offset = vec2f(0.0, 0.008 * intensity);
    
    let r = textureSample(input_texture, global_sampler, uv + offset).r;
    let g = textureSample(input_texture, global_sampler, uv).g;
    let b = textureSample(input_texture, global_sampler, uv - offset).b;
    
    return vec3f(r, g, b);
}

// Scanline effect
fn scanline(uv: vec2f, time: f32) -> f32 {
    let scan_speed = 9090.0;
    let scan_size = 150.0;
    let moving_line = fract(time * scan_speed + uv.y * scan_size) * 0.25;
    return 1.0 - moving_line;
}

// Horizontal distortion
fn horizontal_distortion(uv: vec2f, time: f32, frequency: f32, amplitude: f32) -> vec2f {
    let wave = sin(uv.y * frequency + time) * amplitude;
    return vec2f(uv.x + wave, uv.y);
}

// 4x4 Bayer matrix for ordered dithering
fn bayer_matrix_from_coord(x: u32, y: u32) -> f32 {
    let index = (y & 3u) * 4u + (x & 3u);
    let pattern = array<f32, 16>(
        0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
        12.0/16.0, 4.0/16.0, 14.0/16.0,  6.0/16.0,
        3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
        15.0/16.0, 7.0/16.0, 13.0/16.0,  5.0/16.0
    );
    return pattern[index];
}

// Apply dithering
fn apply_dithering(color: vec3f, uv: vec2f, dither_amount: f32) -> vec3f {
    let screen_size = vec2u(textureDimensions(input_texture));
    let pixel_pos = vec2u(uv * vec2f(screen_size));
    
    let dither_value = (bayer_matrix_from_coord(pixel_pos.x, pixel_pos.y) - 0.5) * dither_amount;
    return color + vec3f(dither_value);
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 

@fragment
fn fs(v_out: VertexOutput) -> FragmentOutput {
    let time = frame_info.time;
    
    // Apply horizontal distortion
    var distorted_uv = horizontal_distortion(
        v_out.uv, 
        time, 
        vhs_params.distortion_frequency * 10.0,
        vhs_params.distortion_amplitude * 0.02
    );
    
    // Keep UVs in bounds
    distorted_uv = clamp(distorted_uv, vec2f(0.0), vec2f(1.0));
    
    // Sample base color
    var color = textureSample(input_texture, global_sampler, distorted_uv).rgb;
    
    // Apply color bleeding
    color = mix(
        color,
        color_bleeding(distorted_uv, color, vhs_params.color_bleeding),
        0.5
    );
    
    // Apply scanlines
    let scan = scanline(v_out.uv, time);
    color *= 1.0 - (vhs_params.scanline_intensity * 0.25 * scan);
    
    // Apply noise
    let noise = vhs_noise(v_out.uv, time);
    color = mix(
        color,
        vec3f(noise),
        vhs_params.noise_intensity * 0.1
    );
    
    // Add dithering before color tinting
    color = apply_dithering(color, v_out.uv, 0.1); // Adjust the 0.1 to control dither intensity
    
    // Add slight color tinting
    color *= vec3f(1.05, 1.0, 1.1); // Slight purple tint
    
    // Boost contrast slightly
    color = pow(color, vec3f(1.1));
    
    return FragmentOutput(vec4f(color, 1.0));
} 
#include "postprocess_common.wgsl"
#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct TiltShiftParams {
    focus_center: f32,       // Center of the focused area (0-1 on Y axis)
    focus_width: f32,        // Width of the focused area
    blur_strength: f32,      // Maximum blur amount
    saturation: f32,         // Color saturation boost
    exposure: f32,           // Exposure adjustment
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) @interpolate(flat) instance_index: u32,
};

struct FragmentOutput {
    @location(0) color: vec4f,
};

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<uniform> tilt_shift_params: TiltShiftParams;
@group(1) @binding(1) var input_texture: texture_2d<f32>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

// Calculate blur amount based on distance from focus center
fn get_blur_factor(uv: vec2f) -> f32 {
    let dist = abs(uv.y - tilt_shift_params.focus_center);
    let focus_edge = tilt_shift_params.focus_width * 0.5;
    
    // Smooth transition between focused and blurred areas
    return smoothstep(focus_edge, focus_edge * 2.0, dist);
}

// Gaussian blur with dynamic sample count based on blur amount
fn variable_blur(uv: vec2f, blur_factor: f32) -> vec3f {
    var color = vec3f(0.0);
    var total_weight = 0.0;
    let blur_size = blur_factor * tilt_shift_params.blur_strength * 0.01;
    
    // Improved gaussian blur with better sample distribution
    // Using a 13x13 kernel for higher quality
    for (var i = -6.0; i <= 6.0; i += 1.0) {
        for (var j = -6.0; j <= 6.0; j += 1.0) {
            let offset = vec2f(i, j) * blur_size;
            let sample_uv = uv + offset;
            
            // Improved Gaussian weight calculation with better sigma
            let sigma = 3.0;
            let weight = exp(-(i*i + j*j) / (2.0 * sigma * sigma));
            
            // Skip samples with negligible contribution
            if (weight > 0.01) {
                color += textureSample(input_texture, global_sampler, sample_uv).rgb * weight;
                total_weight += weight;
            }
        }
    }
    
    return color / total_weight;
}

// Adjust saturation
fn adjust_saturation(color: vec3f, saturation: f32) -> vec3f {
    let luminance = dot(color, vec3f(0.299, 0.587, 0.114));
    return mix(vec3f(luminance), color, saturation);
}

// Vignette effect
fn apply_vignette(uv: vec2f, color: vec3f) -> vec3f {
    let center = vec2f(0.5);
    let dist = distance(uv, center);
    let vignette = 1.0 - smoothstep(0.3, 0.7, dist);
    return color * vignette;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 

@fragment
fn fs(v_out: VertexOutput) -> FragmentOutput {
    let blur_factor = get_blur_factor(v_out.uv);
    
    // Apply variable blur based on distance from focus center
    var color = variable_blur(v_out.uv, blur_factor);
    
    // Increase saturation in focused areas
    let saturation_factor = mix(1.0, tilt_shift_params.saturation, 1.0 - blur_factor);
    color = adjust_saturation(color, saturation_factor);
    
    // Apply subtle vignette
    color = apply_vignette(v_out.uv, color);
    
    // Adjust exposure
    color *= exp2(tilt_shift_params.exposure);
    
    // Add subtle color grading
    // Slightly boost blues in shadows and yellows in highlights
    let luminance = dot(color, vec3f(0.299, 0.587, 0.114));
    let shadows = smoothstep(0.0, 0.3, luminance);
    let highlights = smoothstep(0.7, 1.0, luminance);
    color += vec3f(-0.05, -0.05, 0.1) * (1.0 - shadows);    // Blue tint in shadows
    color += vec3f(0.1, 0.05, -0.05) * highlights;          // Warm highlights
    
    return FragmentOutput(vec4f(color, 1.0));
} 
#include "common.wgsl"
#include "postprocess_common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct CrtParams {
    curvature: f32,          // Screen curvature amount
    vignette: f32,           // Vignette darkness
    scan_brightness: f32,     // Brightness of scanlines
    rgb_offset: f32,         // RGB subpixel separation
    bloom_strength: f32,     // Bloom/glow effect strength
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
};


// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<uniform> crt_params: CrtParams;
@group(1) @binding(1) var input_texture: texture_2d<f32>;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------ 

// Apply screen curvature distortion
fn curve_uv(uv: vec2<f32>, curvature: f32) -> vec2<f32> {
    // Convert UV to centered coordinates (-1 to 1)
    var curved_uv = uv * 2.0 - 1.0;
    
    // Apply barrel distortion
    let barrel = curved_uv * curved_uv * curved_uv;
    curved_uv += barrel * curvature;
    
    // Convert back to 0-1 range
    curved_uv = curved_uv * 0.5 + 0.5;
    
    return curved_uv;
}

// Create RGB subpixel pattern
fn rgb_split(uv: vec2<f32>, offset: f32) -> vec3<f32> {
    let pixel_offset = offset * 0.001;
    let r = textureSample(input_texture, global_sampler, vec2<f32>(uv.x + pixel_offset, uv.y)).r;
    let g = textureSample(input_texture, global_sampler, uv).g;
    let b = textureSample(input_texture, global_sampler, vec2<f32>(uv.x - pixel_offset, uv.y)).b;
    return vec3<f32>(r, g, b);
}

// Create scanline pattern
fn scanlines(uv: vec2<f32>, brightness: f32) -> f32 {
    let scan_size = 400.0;
    let scan = sin(uv.y * scan_size) * 0.5 + 0.5;
    return mix(1.0, scan, brightness);
}

// Vignette effect
fn vignette(uv: vec2<f32>, strength: f32) -> f32 {
    let center = vec2<f32>(0.5);
    let dist = distance(uv, center);
    return 1.0 - smoothstep(0.4, 0.7, dist * strength);
}

// Simple bloom effect
fn bloom(uv: vec2<f32>, strength: f32) -> vec3<f32> {
    let blur_size = 0.004;
    var bloom_color = vec3<f32>(0.0);
    
    // 9-tap gaussian blur
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let offset = vec2<f32>(f32(i), f32(j)) * blur_size;
            bloom_color += textureSample(
                input_texture, 
                global_sampler, 
                uv + offset
            ).rgb;
        }
    }
    
    bloom_color /= 9.0;
    return bloom_color * strength;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------ 

@fragment
fn fs(v_out: VertexOutput) -> FragmentOutput {
    // Apply screen curvature
    var curved_uv = curve_uv(v_out.uv, crt_params.curvature);
    
    // Get base color with RGB subpixel separation
    var color = rgb_split(curved_uv, crt_params.rgb_offset);
    
    // Apply scanlines
    color *= scanlines(curved_uv, crt_params.scan_brightness);
    
    // Add bloom
    color += bloom(curved_uv, crt_params.bloom_strength);
    
    // Apply vignette
    color *= vignette(curved_uv, crt_params.vignette);
    
    // Enhance contrast and colors
    color = pow(color, vec3<f32>(1.2));  // Contrast boost
    color *= 1.2;                    // Brightness boost
    
    // Add subtle color tinting to simulate phosphor colors
    color *= vec3<f32>(1.0, 0.97, 0.95); // Slightly warmer
    
    return FragmentOutput(vec4<f32>(color, 1.0));
} 
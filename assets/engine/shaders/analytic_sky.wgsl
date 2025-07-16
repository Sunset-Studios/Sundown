// ------------------------------------------------------------------------------------
// Analytic Sky Shader
// 
// Implements a physically-based atmospheric scattering model for realistic sky rendering.
// Based on the Preetham et al. model with Rayleigh and Mie scattering.
// ------------------------------------------------------------------------------------
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "postprocess_common.wgsl"

// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------
const E: f32 = 2.71828182845904523536028747135266249775724709369995957;
const EE: f32 = 1000.0;
const CUTOFF_ANGLE: f32 = 1.6110731556870734;
const STEEPNESS: f32 = 1.5;

// Rayleigh scattering coefficients (wavelength-dependent)
const TOTAL_RAYLEIGH: vec3f = vec3f(
    5.804542996261093E-6,  // Red
    1.3562911419845635E-5, // Green  
    3.0265902468824876E-5  // Blue
);

// Mie scattering constants
const MIE_CONST: vec3f = vec3f(
    1.8399918514433978E14,
    2.7798023919660528E14,
    4.0790479543861094E14
);

const THREE_OVER_SIXTEEN_PI: f32 = 0.05968310365946075;
const ONE_OVER_FOUR_PI: f32 = 0.07957747154594767;
const RAYLEIGH_ZENITH_LENGTH: f32 = 8.4E3;
const MIE_ZENITH_LENGTH: f32 = 1.25E3;
const WHITE_SCALE: f32 = 1.0748724675633854; // 1.0 / u2_filmic_tonemapping(1000.0)

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------
struct VertexInput {
    @builtin(vertex_index) vi : u32,
    @builtin(instance_index) ii: u32
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world_position: vec4f,
    @location(1) sun_dir: vec3f,
    @location(2) sun_e: f32,
    @location(3) sun_fade: f32,
    @location(4) beta_r: vec3f,
    @location(5) beta_m: vec3f,
};

struct FragmentOutput {
    @location(0) color: vec4f,
};

struct SceneLightingData {
    sunlight_intensity: f32,
    sunlight_angular_radius: f32,
    atmospheric_rayleigh: f32,
    atmospheric_turbidity: f32,
    mie_coefficient: f32,
    mie_directional_g: f32,
    view_index: f32,
    padding: f32,
};

// ------------------------------------------------------------------------------------
// Uniforms
// ------------------------------------------------------------------------------------

@group(1) @binding(0) var<uniform> scene_lighting_data: SceneLightingData;

// ------------------------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------------------------

/// Calculates sun intensity based on zenith angle
fn sun_intensity(zenith_angle_cos: f32) -> f32 {
    let clamped_zenith_angle_cos = clamp(zenith_angle_cos, -1.0, 1.0);
    return EE * max(0.0, 1.0 - pow(E, -((CUTOFF_ANGLE - acos(clamped_zenith_angle_cos)) / STEEPNESS)));
}

/// Calculates total Mie scattering coefficient
fn total_mie(t: f32) -> vec3f {
    let c = (0.2 * t) * 10E-18;
    return 0.434 * c * MIE_CONST;
}

/// Calculates Rayleigh phase function
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return THREE_OVER_SIXTEEN_PI * (1.0 + pow(cos_theta, 2.0));
}

/// Calculates Henyey-Greenstein phase function for Mie scattering
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let inverse = 1.0 / pow(1.0 - 2.0 * g * cos_theta + g2, 1.5);
    return ONE_OVER_FOUR_PI * ((1.0 - g2) * inverse);
}

// ------------------------------------------------------------------------------------
// Vertex Shader
// ------------------------------------------------------------------------------------
@vertex
fn vs(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Get camera position (fallback if camera_data is not available)
    let camera_pos = view_buffer[frame_info.view_index].view_position.xyz;
    let light_view = view_buffer[u32(scene_lighting_data.view_index)];
    
    // Create model matrix positioned at camera
    let model_matrix = mat4x4f(
        vec4f(1.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 1.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 1.0, 0.0),
        vec4f(camera_pos.x, camera_pos.y, camera_pos.z, 1.0)
    );
    
    let local_position = vertex_buffer[input.vi].position;

    output.world_position = model_matrix * vec4f(local_position.xyz, 1.0);
    // Calculate sun direction and intensity
    output.sun_dir = normalize(-light_view.view_direction.xyz);
    output.sun_e = sun_intensity(dot(output.sun_dir, world_up));
    // Calculate sun fade based on height
    output.sun_fade = 1.0 - clamp(1.0 - exp((light_view.view_direction.y / 450000.0)), 0.0, 1.0);
    
    // Calculate atmospheric scattering coefficients
    let rayleigh_coeff = scene_lighting_data.atmospheric_rayleigh - (1.0 * (1.0 - output.sun_fade));
    output.beta_r = TOTAL_RAYLEIGH * rayleigh_coeff;
    output.beta_m = total_mie(scene_lighting_data.atmospheric_turbidity) * scene_lighting_data.mie_coefficient;

    // Project to clip space
    let view_proj = view_buffer[frame_info.view_index].view_projection_matrix;

    output.position = view_proj * output.world_position;
    output.position.z = output.position.w;
    
    return output;
}

// ------------------------------------------------------------------------------------
// Fragment Shader
// ------------------------------------------------------------------------------------
@fragment
fn fs(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;
    
    // Get camera position and calculate view direction
    let camera_pos = view_buffer[frame_info.view_index].view_position.xyz;
    let view_dir = normalize(input.world_position.xyz - camera_pos);
    
    // Calculate optical length through atmosphere
    let zenith_angle = acos(max(0.0, dot(world_up, view_dir)));
    let inverse = (cos(zenith_angle) + 0.15 * pow(93.885 - ((zenith_angle * 180.0) / PI), -1.253));
    let sr = RAYLEIGH_ZENITH_LENGTH / inverse;
    let sm = MIE_ZENITH_LENGTH / inverse;
    
    // Calculate combined extinction factor
    let f_ex = exp(-(input.beta_r * sr + input.beta_m * sm));
    
    // Calculate inscattering
    let cos_theta = dot(view_dir, input.sun_dir);
    
    // Rayleigh scattering
    let r_phase = rayleigh_phase(cos_theta * 0.5 + 0.5);
    let beta_r_theta = input.beta_r * r_phase;
    
    // Mie scattering
    let m_phase = hg_phase(cos_theta, scene_lighting_data.mie_directional_g);
    let beta_m_theta = input.beta_m * m_phase;
    
    // Combine scattering contributions
    let beta_delta = (beta_r_theta + beta_m_theta) / (input.beta_r + input.beta_m);
    var lin = pow(input.sun_e * beta_delta * (1.0 - f_ex), vec3f(1.5));
    
    // Apply atmospheric perspective
    lin *= mix(
        vec3f(1.0), 
        pow(input.sun_e * beta_delta * f_ex, vec3f(0.5)), 
        clamp(pow(1.0 - dot(world_up, input.sun_dir), 5.0), 0.0, 1.0)
    );
    
    // Calculate night sky contribution
    let theta = acos(view_dir.y);
    let phi = atan2(view_dir.z, view_dir.x);
    let uv = vec2f(phi, theta) / vec2f(2.0 * PI, PI) + vec2f(0.5, 0.0);
    let l0 = vec3f(0.1) * f_ex;
    
    // Add solar disk
    let sundisk = smoothstep(
        scene_lighting_data.sunlight_angular_radius, 
        scene_lighting_data.sunlight_angular_radius + 0.00002, 
        cos_theta
    );
    let l0_final = l0 + (input.sun_e * 19000.0 * f_ex) * sundisk;
    
    // Combine all contributions
    let tex_color = (lin + l0_final) * 0.04 + vec3f(0.0, 0.0003, 0.00075);
    
    // Apply tonemapping and final adjustments
    let curr = u2_filmic_tonemapping(tex_color, log2(2.0 / (scene_lighting_data.sunlight_intensity / 2.0)));
    var color = curr * WHITE_SCALE;
    color = pow(color, vec3f(1.0 / (1.2 + (1.2 * input.sun_fade))));

    output.color = vec4f(color, 0.0);

    return output;
}
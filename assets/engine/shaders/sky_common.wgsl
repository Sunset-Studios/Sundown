// =============================================================================
// Sky Evaluation Common Functions
// Shared atmospheric scattering model for skybox and path tracing
// =============================================================================

// Constants for atmospheric scattering
const E: f32 = 2.71828182845904523536028747135266249775724709369995957;
const EE: f32 = 1000.0;
const CUTOFF_ANGLE: f32 = 1.6110731556870734;
const STEEPNESS: f32 = 1.5;

// Rayleigh scattering coefficients (wavelength-dependent)
const TOTAL_RAYLEIGH: vec3<f32> = vec3<f32>(
    5.804542996261093E-6,  // Red
    1.3562911419845635E-5, // Green  
    3.0265902468824876E-5  // Blue
);

// Mie scattering constants
const MIE_CONST: vec3<f32> = vec3<f32>(
    1.8399918514433978E14,
    2.7798023919660528E14,
    4.0790479543861094E14
);

const THREE_OVER_SIXTEEN_PI: f32 = 0.05968310365946075;
const ONE_OVER_FOUR_PI: f32 = 0.07957747154594767;
const RAYLEIGH_ZENITH_LENGTH: f32 = 8.4E3;
const MIE_ZENITH_LENGTH: f32 = 1.25E3;
const WHITE_SCALE: f32 = 1.0748724675633854; // 1.0 / u2_filmic_tonemapping(1000.0)

// Scene lighting parameters
struct SceneLightingData {
    skybox_color: vec4<f32>,
    sunlight_intensity: f32,
    sunlight_angular_radius: f32,
    atmospheric_rayleigh: f32,
    atmospheric_turbidity: f32,
    mie_coefficient: f32,
    mie_directional_g: f32,
    view_index: f32,
    sky_type: f32, // 0 = skybox, 1 = skydome
};

/// Calculates sun intensity based on zenith angle
fn sun_intensity(zenith_angle_cos: f32) -> f32 {
    let clamped_zenith_angle_cos = clamp(zenith_angle_cos, -1.0, 1.0);
    return EE * max(0.0, 1.0 - pow(E, -((CUTOFF_ANGLE - acos(clamped_zenith_angle_cos)) / STEEPNESS)));
}

/// Calculates total Mie scattering coefficient
fn total_mie(t: f32) -> vec3<f32> {
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

/// Evaluates the analytical sky color for a given view direction
/// Returns RGB sky radiance
fn evaluate_sky(
    view_dir: vec3<f32>,
    sun_dir: vec3<f32>,
    scene_lighting: SceneLightingData
) -> vec3<f32> {
    // Calculate sun intensity and fade
    let sun_e = sun_intensity(dot(sun_dir, world_up));
    let sun_fade = 1.0 - clamp(1.0 - exp((sun_dir.y / 450000.0)), 0.0, 1.0);
    
    // Calculate atmospheric scattering coefficients
    let rayleigh_coeff = scene_lighting.atmospheric_rayleigh - (1.0 * (1.0 - sun_fade));
    let beta_r = TOTAL_RAYLEIGH * rayleigh_coeff;
    let beta_m = total_mie(scene_lighting.atmospheric_turbidity) * scene_lighting.mie_coefficient;
    
    // Calculate optical length through atmosphere
    let zenith_angle = acos(max(0.0, dot(world_up, view_dir)));
    let inverse = (cos(zenith_angle) + 0.15 * pow(93.885 - ((zenith_angle * 180.0) / PI), -1.253));
    let sr = RAYLEIGH_ZENITH_LENGTH / inverse;
    let sm = MIE_ZENITH_LENGTH / inverse;
    
    // Calculate combined extinction factor
    let f_ex = exp(-(beta_r * sr + beta_m * sm));
    
    // Calculate inscattering
    let cos_theta = dot(view_dir, sun_dir);
    
    // Rayleigh scattering
    let r_phase = rayleigh_phase(cos_theta * 0.5 + 0.5);
    let beta_r_theta = beta_r * r_phase;
    
    // Mie scattering
    let m_phase = hg_phase(cos_theta, scene_lighting.mie_directional_g);
    let beta_m_theta = beta_m * m_phase;
    
    // Combine scattering contributions
    let beta_delta = (beta_r_theta + beta_m_theta) / (beta_r + beta_m);
    var lin = pow(sun_e * beta_delta * (1.0 - f_ex), vec3<f32>(1.5));
    
    // Apply atmospheric perspective
    lin *= mix(
        vec3<f32>(1.0), 
        pow(sun_e * beta_delta * f_ex, vec3<f32>(0.5)), 
        clamp(pow(1.0 - dot(world_up, sun_dir), 5.0), 0.0, 1.0)
    );
    
    // Calculate night sky contribution
    let l0 = vec3<f32>(0.1) * f_ex;
    
    // Add solar disk
    let sundisk = smoothstep(
        scene_lighting.sunlight_angular_radius, 
        scene_lighting.sunlight_angular_radius + 0.00002, 
        cos_theta
    );
    let l0_final = l0 + (sun_e * 19000.0 * f_ex) * sundisk;
    
    // Combine all contributions
    let tex_color = (lin + l0_final) * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075);
    
    // Apply tonemapping and final adjustments
    let curr = u2_filmic_tonemapping(tex_color, log2(2.0 / (scene_lighting.sunlight_intensity / 2.0)));
    var color = curr * WHITE_SCALE;
    color = pow(color, vec3<f32>(1.0 / (1.2 + (1.2 * sun_fade))));

    return color;
}

/// Samples a skybox cube texture for a given direction
/// Returns RGB sky color from the skybox
fn sample_skybox(
    view_dir: vec3<f32>,
    skybox_tex: texture_cube<f32>,
    scene_lighting: SceneLightingData 
) -> vec3<f32> {
    // Sample the cube texture with the view direction
    let sampled_color = textureSampleLevel(skybox_tex, global_sampler, view_dir, 0.0);
    return sampled_color.rgb * scene_lighting.skybox_color.rgb;
}

/// Unified environment sampling function
/// Chooses between skybox and skydome based on sky_type flag
fn evaluate_environment(
    view_dir: vec3<f32>,
    sun_dir: vec3<f32>,
    scene_lighting: SceneLightingData,
    skybox_tex: texture_cube<f32>,
) -> vec3<f32> {
    // sky_type: 0 = skybox, 1 = skydome
    return select(
        evaluate_sky(view_dir, sun_dir, scene_lighting),
        sample_skybox(view_dir, skybox_tex, scene_lighting),
        scene_lighting.sky_type < 1.0
    );
}


// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct Light {
    position: vec4<f32>,
    direction: vec4<f32>,
    color: vec4<f32>,
    light_type: f32,
    intensity: f32,
    radius: f32,
    attenuation: f32,
    outer_angle: f32,
    shadow_casting: f32,
    activated: f32,
    view_index: f32,
    shadow_index: f32,
    shadow_clipmaps: f32,
    is_primary_sun: f32,
    shadows_dirty: f32,
};

struct DenseLightsHeader {
    light_count: u32,
    shadow_casting_light_count: u32,
    _pad0: u32,
    _pad1: u32,
};

struct DenseLightsBuffer {
    header: DenseLightsHeader,
    lights: array<Light>,
};

struct DenseLightsHeaderA {
    light_count: atomic<u32>,
    shadow_casting_light_count: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
};

struct DenseLightsBufferA {
    header: DenseLightsHeaderA,
    lights: array<Light>,
};

struct EmissiveLight {
    position_radius: vec4<f32>,      // xyz = world centroid, w = equivalent radius
    normal_area: vec4<f32>,          // xyz = world normal, w = triangle area
    radiance_weight: vec4<f32>,      // rgb = emissive radiance estimate, w = sampling weight (luminance * area)
    instance_tri_section: vec4<u32>, // x = prim_store, y = mesh_id, z = tri_id_local, w = section_index
};

struct EmissiveLightsHeader {
    light_count: u32,
    _pad0: u32, // quantized total sampling weight (sum(w) * EMISSIVE_WEIGHT_QUANTIZATION)
    _pad1: u32, // quantized max sampling weight (max(w) * EMISSIVE_WEIGHT_QUANTIZATION)
    _pad2: u32,
};

struct EmissiveLightsBuffer {
    header: EmissiveLightsHeader,
    lights: array<EmissiveLight>,
};

struct EmissiveLightsHeaderA {
    light_count: atomic<u32>,
    _pad0: atomic<u32>, // quantized total sampling weight
    _pad1: atomic<u32>, // quantized max sampling weight
    _pad2: u32,
};

struct EmissiveLightsBufferA {
    header: EmissiveLightsHeaderA,
    lights: array<EmissiveLight>,
};

const EMISSIVE_WEIGHT_QUANTIZATION = 1024.0;
const EMISSIVE_WEIGHT_QUANTIZATION_INV = 1.0 / EMISSIVE_WEIGHT_QUANTIZATION;
const EMISSIVE_WEIGHTED_SAMPLE_ATTEMPTS: u32 = 4u;

// ------------------------------------------------------------------------------------
// Light Helpers
// ------------------------------------------------------------------------------------
const SLOPE_SCALE_BIAS: f32 = 0.00001;
fn get_light_dir(light: Light, fragment_pos: vec3<f32>) -> vec3<f32> {
    var light_dir: vec3<f32>;

    if (light.light_type == 0.0) { // Directional
        light_dir = normalize(light.position.xyz);
    } else if (light.light_type == 1.0) { // Point
        let light_vec = light.position.xyz - fragment_pos;
        light_dir = normalize(light_vec);
    } else if (light.light_type == 2.0) { // Spot
        let light_vec = light.position.xyz - fragment_pos;
        light_dir = normalize(light_vec);
    }
    return light_dir;
}

fn get_light_attenuation(light: Light, fragment_pos: vec3<f32>) -> f32 {
    var attenuation = 1.0;
    if (light.light_type == 1.0) { // Point
        let light_vec = light.position.xyz - fragment_pos;
        let distance_sq = dot(light_vec, light_vec);
        attenuation = compute_distance_attenuation(distance_sq, light.radius);
    } else if (light.light_type == 2.0) { // Spot
        let light_vec = light.position.xyz - fragment_pos;
        let distance_sq = dot(light_vec, light_vec);
        let dist_att = compute_distance_attenuation(distance_sq, light.radius);
        let cos_theta = dot(-light_vec, normalize(light.direction.xyz));
        let cos_inner = cos(light.direction.w);
        let cos_outer = cos(light.outer_angle);
        let angle_att = compute_spot_angle_attenuation(cos_theta, cos_inner, cos_outer);
        attenuation = dist_att * angle_att;
    }
    return clamp(attenuation, 0.0, 1.0);
}

// ------------------------------------------------------------------------------------
// Microfacet Distribution
// ------------------------------------------------------------------------------------
fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = n_dot_h * roughness;
    let k = roughness / max(0.001, 1.0 - n_dot_h * n_dot_h + a * a);
    return k * k * (1.0 / PI);
}

fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let h = vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );

    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);

    let sample_vec = tangent * h.x + bitangent * h.y + n * h.z;
    return normalize(sample_vec);
}

// ------------------------------------------------------------------------------------
// Visibility
// ------------------------------------------------------------------------------------
fn v_smith_ggx_height_correlated(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    let ggx_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let ggx_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / (ggx_v + ggx_l);
}

fn v_smith_ggx_height_correlated_fast(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return 0.5 / mix(2.0 * n_dot_l * n_dot_v, n_dot_l + n_dot_v, roughness);
}

// ------------------------------------------------------------------------------------
// Fresnel
// ------------------------------------------------------------------------------------
fn f_schlick_scalar(f0: f32, f90: f32, v_dot_h: f32) -> f32 {
    let one_minus_voh = 1.0 - v_dot_h;
    let one_minus_voh_2 = one_minus_voh * one_minus_voh;
    return f0 + (f90 - f0) * one_minus_voh_2 * one_minus_voh_2 * one_minus_voh;
}

fn f_schlick_vec3(f0: vec3<f32>, f90: f32, v_dot_h: f32) -> vec3<f32> {
    let one_minus_voh = 1.0 - v_dot_h;
    let one_minus_voh_2 = one_minus_voh * one_minus_voh;
    return f0 + (f90 - f0) * one_minus_voh_2 * one_minus_voh_2 * one_minus_voh;
}

fn f_schlick_roughness(n_dot_v: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - n_dot_v, 0.0, 1.0), 5.0);
}

// ------------------------------------------------------------------------------------
// Diffuse
// ------------------------------------------------------------------------------------
fn fd_lambert() -> f32 {
    return 1.0 / PI;
}

// ------------------------------------------------------------------------------------
// Clear Coat
// ------------------------------------------------------------------------------------
fn v_kelemen(l_dot_h: f32) -> f32 {
    return clamp(0.25 / (l_dot_h * l_dot_h), 0.0, 1.0);
}

// ------------------------------------------------------------------------------------
// Normal Mapping
// ------------------------------------------------------------------------------------
fn get_normal_from_normal_map(normal_map: texture_2d<f32>, uv: vec2<f32>, tbn_matrix: mat3x3<f32>) -> vec3<f32> {
    let tangent_normal = normalize(textureSample(normal_map, global_sampler, uv).xyz * 2.0 - 1.0);
    return tbn_matrix * tangent_normal;
}

// ------------------------------------------------------------------------------------
// Attenuation Helpers
// ------------------------------------------------------------------------------------
fn compute_distance_attenuation(distance_squared: f32, radius: f32) -> f32 {
    // If radius is zero or negative, skip attenuation (treat as infinite reach).
    let fade = max(1.0 - distance_squared / (radius * radius), 0.0);
    // Square for a smoother fall-off.
    return fade * fade;
}

fn compute_spot_angle_attenuation(cos_theta: f32, cos_inner: f32, cos_outer: f32) -> f32 {
    // Smooth Hermite blend between inner and outer cone.
    let scale = 1.0 / max(cos_inner - cos_outer, 0.0001);
    let att = clamp((cos_theta - cos_outer) * scale, 0.0, 1.0);
    return att * att;
}

// ------------------------------------------------------------------------------------
// PDF Sampling 
// ------------------------------------------------------------------------------------
fn uniform_hemisphere_pdf() -> f32 {
    return 1.0 / (2.0 * PI);
}

fn cosine_hemisphere_pdf(normal: vec3<f32>, sample_dir: vec3<f32>) -> f32 {
    return dot(normal, sample_dir) / PI;
}

fn ggx_pdf(normal: vec3<f32>, view_dir: vec3<f32>, sample_dir: vec3<f32>, roughness: f32) -> f32 {
    let h = normalize(view_dir + sample_dir);
    let n_dot_h = max(dot(normal, h), 0.0001);
    let d = d_ggx(n_dot_h, roughness);
    return d * n_dot_h / max(4.0 * max(dot(view_dir, h), 0.0001), 0.0001);
}

fn brdf_pdf(normal: vec3<f32>, view_dir: vec3<f32>, sample_dir: vec3<f32>, roughness: f32, mis_specular_prob: f32) -> f32 {
    let uniform_pdf = cosine_hemisphere_pdf(normal, sample_dir);
    let ggx_pdf = ggx_pdf(normal, view_dir, sample_dir, roughness);
    return mis_specular_prob * ggx_pdf + (1.0 - mis_specular_prob) * uniform_pdf;
}

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#UniformlySamplingaHemisphere
fn sample_uniform_hemisphere(normal: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
    let cos_theta = sqrt(1.0 - r2);
    let phi = 2.0 * PI * r1;
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let x = sin_theta * cos(phi);
    let y = sin_theta * sin(phi);
    let z = cos_theta;
    return orthonormalize(normal) * vec3<f32>(x, y, z);
}

fn sample_ggx(n: vec3<f32>, roughness: f32, r1: f32, r2: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let a2 = a * a;
    
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt((1.0 - r2) / (1.0 + (a2 - 1.0) * r2));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = normalize(cross(n, tangent));
    
    let h_local = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return normalize(tangent * h_local.x + bitangent * h_local.y + n * h_local.z);
}

// ------------------------------------------------------------------------------------
// Lighting
// ------------------------------------------------------------------------------------

fn calculate_blinn_phong(
    light: Light,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    fragment_pos: vec3<f32>,
    albedo: vec3<f32>,
    shininess: f32,
    ambient: vec3<f32>,
    shadow_factor: f32,
) -> vec3<f32> {
    // Attenuation
    let attenuation = get_light_attenuation(light, fragment_pos);

    // Ambient
    let ambient_color = albedo * ambient;

    // Diffuse
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse_color = light.color.rgb * albedo * n_dot_l;

    // Specular
    let halfway = safe_normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, halfway), 0.0);
    let specular = pow(n_dot_h, shininess);
    let specular_color = light.color.rgb * specular;

    // Final color
    let final_color = ambient_color + (diffuse_color + specular_color) * light.intensity * attenuation * (1.0 - shadow_factor);

    return final_color;
}

// ------------------------------------------------------------------------------------
// BRDF (Physically Based, Energy Conserving, Clear Coat Layering, AO-correct)
// ------------------------------------------------------------------------------------
fn calculate_brdf(
    light_view_index: u32,
    light: Light,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    fragment_pos: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    reflectance: f32,
    clear_coat: f32,
    clear_coat_roughness: f32,
    ao: f32,
    irradiance: vec3<f32>,
    prefiltered_color: vec3<f32>,
    env_brdf: vec2<f32>,
    shadow_factor: f32,
) -> vec3<f32> {
    // Compute attenuation for point/spot
    let attenuation = get_light_attenuation(light, fragment_pos);

    // Halfway vector and dot products
    let halfway = safe_normalize(light_dir + view_dir);
    let n_dot_v = max(dot(normal, view_dir), 0.0001);
    let n_dot_l = max(dot(normal, light_dir), 0.0001);
    let n_dot_h = max(dot(normal, halfway), 0.0001);
    let v_dot_h = max(dot(view_dir, halfway), 0.0001);
    let l_dot_h = max(dot(light_dir, halfway), 0.0001);

    // Surface parameters
    let a = roughness * roughness;
    let clamped_clear_coat_roughness = clamp(clear_coat_roughness, 0.089, 1.0);
    let cc_roughness = clamped_clear_coat_roughness * clamped_clear_coat_roughness;

    // F0 (specular at normal incidence) for base layer
    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);

    // Fresnel for specular
    let f = f_schlick_vec3(f0, 1.0, v_dot_h);

    // D and V for GGX
    let r = max(roughness, 0.089);
    let d = d_ggx(n_dot_h, r);
    let v = v_smith_ggx_height_correlated_fast(n_dot_v, n_dot_l, r);

    // Specular BRDF (Cook-Torrance)
    let specular_brdf = (d * v) * f;

    // Lambertian diffuse with energy conservation (1-F)
    let kd = (1.0 - metallic) * (vec3<f32>(1.0) - f);
    let diffuse_brdf = kd * albedo / 3.14159265359;

    // ---- Clear Coat Layer (Disney 2015 style) ----
    // Single specular lobe (GGX or GTR1), usually IOR ~1.5, F0 = 0.04
    let dc = d_ggx(n_dot_h, cc_roughness);
    let vc = v_kelemen(l_dot_h); // Kelemen visibility
    let fc = f_schlick_scalar(0.04, 1.0, v_dot_h);
    let clear_coat_brdf = dc * vc * fc * clear_coat;
    let clear_coat_energy_loss = fc * clear_coat;

    // Direct lighting
    let direct_light = (diffuse_brdf + specular_brdf) * (1.0 - clear_coat_energy_loss) * light.intensity * n_dot_l * attenuation * light.color.rgb;

    // Layered composition: base * (1 - clear_coat_energy_loss) + clear_coat
    let direct_brdf = direct_light + clear_coat_brdf * light.intensity * n_dot_l * attenuation * light.color.rgb;

    // ---- Indirect Lighting ----
    // Indirect lighting from irradiance cache texture (screen probes / skybox)
    var indirect_contribution = albedo * irradiance * ao;

    // Specular: prefiltered env map, split-sum approximation, modulated by AO
    let env_f = f_schlick_roughness(n_dot_v, f0, a);
    let indirect_specular = prefiltered_color * (f0 * env_brdf.x + (vec3<f32>(1.0) - f0) * env_brdf.y) * ao;

    // Clear coat from environment: usually just add a small reflection, not typical unless you precompute a clear coat IBL.
    // For simplicity, we omit indirect clear coat here.

    // ---- Combine all lighting ----
    let color = direct_brdf * (1.0 - shadow_factor) + indirect_contribution + indirect_specular;

    return color;
}

// ------------------------------------------------------------------------------------
// BRDF - Ray Tracing Step Variant
// - Light-agnostic evaluation for a sampled direction `light_dir`
// - Returns (diffuse + specular [+ clear coat]) * n_dot_l
// ------------------------------------------------------------------------------------
fn calculate_brdf_rt(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    reflectance: f32,
    clear_coat: f32,
    clear_coat_roughness: f32,
) -> vec3<f32> {
    // Halfway vector and dot products
    let halfway = safe_normalize(light_dir + view_dir);
    let n_dot_v = max(dot(normal, view_dir), 0.0001);
    let n_dot_l = max(dot(normal, light_dir), 0.0001);
    let n_dot_h = max(dot(normal, halfway), 0.0001);
    let v_dot_h = max(dot(view_dir, halfway), 0.0001);
    let l_dot_h = max(dot(light_dir, halfway), 0.0001);

    // Base layer Fresnel term
    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
    let f = f_schlick_vec3(f0, 1.0, v_dot_h);

    // Microfacet terms (GGX)
    let r = clamp(roughness, 0.001, 1.0);
    let d = d_ggx(n_dot_h, r);
    let v = v_smith_ggx_height_correlated_fast(n_dot_v, n_dot_l, r);
    let specular_brdf = (d * v) * f;

    // Diffuse (energy conserving)
    let kd = (1.0 - metallic) * (vec3<f32>(1.0) - f);
    let diffuse_brdf = kd * albedo * (1.0 / PI);

    // Optional clear coat lobe
    let cc_r_clamped = clamp(clear_coat_roughness, 0.089, 1.0);
    let dc = d_ggx(n_dot_h, cc_r_clamped * cc_r_clamped);
    let vc = v_kelemen(l_dot_h);
    let fc = f_schlick_scalar(0.04, 1.0, v_dot_h);
    let clear_coat_brdf = dc * vc * fc * clear_coat;
    let clear_coat_energy_loss = fc * clear_coat;

    // Layered combination: base scaled by energy loss + clear coat lobe
    let base = (diffuse_brdf + specular_brdf) * (1.0 - clear_coat_energy_loss);
    let layered = base + clear_coat_brdf;

    return layered * n_dot_l;
}

// ------------------------------------------------------------------------------------
// BRDF Signal - Ray Tracing Step Variant
// - Returns (diffuse + specular [+ clear coat]) * n_dot_l
// - Used for calculating the BRDF lighting-only term for the radiance output 
// ------------------------------------------------------------------------------------
fn calculate_brdf_lighting_rt(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    roughness: f32,
    metallic: f32,
    reflectance: f32,
    clear_coat: f32,
    clear_coat_roughness: f32,
) -> vec3<f32> {
    let halfway = safe_normalize(light_dir + view_dir);
    let n_dot_v = max(dot(normal, view_dir), 0.0001);
    let n_dot_l = max(dot(normal, light_dir), 0.0001);
    let n_dot_h = max(dot(normal, halfway), 0.0001);
    let v_dot_h = max(dot(view_dir, halfway), 0.0001);
    let l_dot_h = max(dot(light_dir, halfway), 0.0001);

    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), vec3<f32>(1.0), metallic);

    let f = f_schlick_vec3(f0, 1.0, v_dot_h);
    let r = clamp(roughness, 0.001, 1.0);
    let d = d_ggx(n_dot_h, r);
    let v = v_smith_ggx_height_correlated_fast(n_dot_v, n_dot_l, r);

    let specular_brdf = (d * v) * f;
    let kd = (1.0 - metallic) * (vec3<f32>(1.0) - f);
    let diffuse_brdf = kd * vec3<f32>(1.0) * (1.0 / PI);
    let clear_coat_r_clamped = clamp(clear_coat_roughness, 0.089, 1.0);
    let dc = d_ggx(n_dot_h, clear_coat_r_clamped * clear_coat_r_clamped);
    let vc = v_kelemen(l_dot_h);
    let fc = f_schlick_scalar(0.04, 1.0, v_dot_h);
    let clear_coat_brdf = dc * vc * fc * clear_coat;
    let clear_coat_energy_loss = fc * clear_coat;
    let base = (diffuse_brdf + specular_brdf) * (1.0 - clear_coat_energy_loss);
    let layered = base + clear_coat_brdf;

    return layered * n_dot_l;
}
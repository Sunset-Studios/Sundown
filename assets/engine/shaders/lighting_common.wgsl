const PI: f32 = 3.14159265359;

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 
struct Light {
    position: vec4f,
    direction: vec4f,
    color: vec4f,
    light_type: f32,
    intensity: f32,
    radius: f32,
    attenuation: f32,
    outer_angle: f32,
    activated: f32,
};

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

    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) < 0.999);
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
// Lighting
// ------------------------------------------------------------------------------------
fn calculate_blinn_phong(
    light: Light,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    fragment_pos: vec3<f32>,
    albedo: vec3<f32>,
    shininess: f32,
    ambient: vec3<f32>
) -> vec3<f32> {
    var light_dir: vec3<f32>;
    var attenuation = 1.0;

    if (light.light_type == 0.0) { // Directional
        light_dir = normalize(light.position.xyz);
    } else if (light.light_type == 1.0) { // Point
        let light_to_frag = fragment_pos - light.position.xyz;
        light_dir = normalize(light_to_frag);
        let distance = length(light_to_frag);
        let falloff = 1.0 - smoothstep(0.0, light.radius, distance);
        attenuation = falloff / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    } else if (light.light_type == 2.0) { // Spot
        let light_to_frag = fragment_pos - light.position.xyz;
        light_dir = normalize(light_to_frag);
        let distance = length(light_to_frag);
        let spot_effect = dot(light_dir, light.direction.xyz);
        let falloff = pow(spot_effect, 180.0 / max(1.0, light.outer_angle));
        attenuation = falloff / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    }

    attenuation = clamp(attenuation, 0.0, 255.0);

    // Ambient
    let ambient_color = albedo * ambient;

    // Diffuse
    let wrap_factor = 0.01;
    let n_dot_l = max((dot(normal, light_dir) + wrap_factor) / (1.0 + wrap_factor), 0.0);
    let diffuse_color = light.color.rgb * albedo * n_dot_l;

    // Specular
    let halfway = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, halfway), 0.0);
    let specular = pow(n_dot_h, shininess);
    let specular_color = light.color.rgb * specular;

    // Attenuation
    let final_color = ambient_color + (diffuse_color + specular_color) * light.intensity * attenuation;

    return final_color;
}

// ------------------------------------------------------------------------------------
// BRDF
// ------------------------------------------------------------------------------------
fn calculate_brdf(
    light: Light,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
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
    shadow_map_index: i32
) -> vec3<f32> {
    var light_dir: vec3<f32>;
    var attenuation = 1.0;

    if (light.light_type == 0.0) { // Directional
        light_dir = normalize(light.position.xyz);
    } else if (light.light_type == 1.0) { // Point
        let light_to_frag = light.position.xyz - fragment_pos;
        light_dir = normalize(light_to_frag);
        let distance_squared = dot(light_to_frag, light_to_frag);
        let light_inv_radius = 1.0 / light.radius;
        let factor = distance_squared * light_inv_radius * light_inv_radius;
        let smooth_factor = max(1.0 - factor * factor, 0.0);
        attenuation = (smooth_factor * smooth_factor) / max(distance_squared, 0.0001); 
    } else if (light.light_type == 2.0) { // Spot
        let light_to_frag = light.position.xyz - fragment_pos;
        light_dir = normalize(light_to_frag);
        let cos_outer = cos(light.outer_angle);
        let spot_scale = 1.0 / max(cos(light.direction.w) - cos_outer, 0.0001);
        let spot_offset = -cos_outer * spot_scale;
        let cd = dot(normalize(-light_dir), normalize(light.direction.xyz));
        attenuation = clamp(cd * spot_scale + spot_offset, 0.0, 1.0);
        attenuation = attenuation * attenuation;
    }

    attenuation = clamp(attenuation, 0.0, 255.0);

    let halfway = normalize(light_dir + view_dir);

    let n_dot_v = max(dot(normal, view_dir), 0.0001);
    let n_dot_h = max(dot(normal, halfway), 0.0001);
    let l_dot_h = max(dot(light_dir, halfway), 0.0001);
    let v_dot_h = max(dot(view_dir, halfway), 0.0001);
    let n_dot_l = max(dot(normal, light_dir), 0.0001);

    let a = roughness * roughness;

    let lluminance = light.intensity * attenuation * n_dot_l * light.color.rgb;

    // specular reflectance at normal incidence angle for both dielectric and metallic materials
    var f0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + albedo * metallic;
    // account for clear coat interface
    let f0_clear_coat = clamp(f0 * (f0 * (0.941892 - 0.263008 * f0) + 0.346479) - 0.0285998, vec3<f32>(0.0), vec3<f32>(1.0));
    f0 = mix(f0, f0_clear_coat, clear_coat);
    
    let f90 = clamp(dot(f0, vec3<f32>(50.0 * 0.33)), 0.0, 1.0);

    let d = d_ggx(n_dot_h, roughness);
    let f = f_schlick_vec3(f0, f90, v_dot_h);
    let v = v_smith_ggx_height_correlated_fast(n_dot_v, n_dot_l, roughness);

    // specular BRDF
    let fr = (d * v) * f;

    let env_f = f_schlick_roughness(n_dot_v, f0, a);

    // shadow
    //let shadow_factor = select(0.0, calculate_csm_shadow(fragment_pos, normal, light_dir, shadow_map_index), light.b_csm_caster > 0u && shadow_map_index != -1);
    let shadow_factor = 0.0;

    // diffuse BRDF
    let diffuse_color = (1.0 - metallic) * albedo * irradiance * ao;
    let env_specular = prefiltered_color * (f0 * env_brdf.x + f90 * env_brdf.y);
    let fd = diffuse_color * (1.0 - shadow_factor) * fd_lambert() + env_specular * env_f;

    // remapping and linearization of clear coat roughness
    let clamped_clear_coat_roughness = clamp(clear_coat_roughness, 0.089, 1.0);
    let cc_roughness = clamped_clear_coat_roughness * clamped_clear_coat_roughness;

    // clear coat BRDG
    // TODO: clear coat should be using geometric normal instead of detail normal
    let dc = d_ggx(n_dot_h, cc_roughness);
    let vc = v_kelemen(l_dot_h);
    let fc = f_schlick_scalar(0.04, 1.0, v_dot_h) * clear_coat;
    let frc = (dc * vc) * fc;

    // account for energy loss in the base layer
    let brdf = ((fd + fr * (1.0 - fc)) * (1.0 - fc) + frc);
    return brdf * lluminance;
}
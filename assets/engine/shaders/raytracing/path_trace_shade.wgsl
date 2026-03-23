// =============================================================================
// Path Tracer - Shade Pass (Simple Monte Carlo)
// =============================================================================
// Unbiased Monte Carlo path tracing with BRDF importance sampling.
// - Evaluates direct lighting via Next Event Estimation (NEE)
// - Samples BRDF to generate next bounce direction
// - Accumulates path contribution over multiple frames
// =============================================================================

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "lighting_common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"

const MAX_RADIANCE_LUMINANCE = 10.0;
const MAX_NEE_LUMINANCE = 10.0;

// ─────────────────────────────────────────────────────────────────────────────
// Demodulation Helper - Safely divides radiance by albedo
// ─────────────────────────────────────────────────────────────────────────────
fn demodulate(radiance: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    // Prevent division by very small values while preserving color ratios
    let safe_albedo = max(albedo, vec3<f32>(0.001));
    return radiance / safe_albedo;
}

// ─────────────────────────────────────────────────────────────────────────────
// Path Tracer Parameters
// ─────────────────────────────────────────────────────────────────────────────
struct PathTracerParams {
    max_bounces: u32,
    reset_accum_flag: u32,
    use_gbuffer: u32,
    trace_rate: u32,
    frame_phase: u32,
    samples_per_pixel: u32,
    sample_index: u32,
    padding: u32,
};

// ─────────────────────────────────────────────────────────────────────────────
// Path State & Shade Structures
// ─────────────────────────────────────────────────────────────────────────────
struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal_section_index: vec4<f32>,
    state_u32: vec4<u32>,
    hit_attr0: vec4<f32>,
    hit_attr1: vec4<f32>,
    shadow_origin: vec4<f32>,
    shadow_direction: vec4<f32>,
    shadow_radiance: vec4<f32>,
    path_weight: vec4<f32>,
    rng_sample_count: vec4<f32>,
    accumulated_radiance: vec4<f32>,
    primary_albedo: vec4<f32>,
};

// ─────────────────────────────────────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────────────────────────────────────
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(2) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(3) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(4) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(5) var<storage, read> material_palette: array<u32>;
@group(1) @binding(6) var<storage, read> dense_lights_buffer: DenseLightsBuffer;
@group(1) @binding(7) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(8) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(9) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(15) var skybox_texture: texture_cube<f32>;
@group(1) @binding(16) var output_tex: texture_storage_2d<rgba16float, write>;

// =============================================================================
// Main Compute Shader
// =============================================================================
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    let pixel_coords = compute_phased_pixel_coords(gid.x, res, pt_params.trace_rate, pt_params.frame_phase);
    
    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }

    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;
    var info = path_state[pixel_index];
    let current_bounce = info.state_u32.x;

    // ─────────────────────────────────────────────────────────────────────────
    // Add visible shadow ray contribution (from previous bounce's NEE)
    // ─────────────────────────────────────────────────────────────────────────
    if (path_state[pixel_index].state_u32.z == 1u) {
        let shadow_contrib = safe_clamp_vec3_max(path_state[pixel_index].shadow_radiance.rgb, MAX_NEE_LUMINANCE);
        path_state[pixel_index].accumulated_radiance += vec4<f32>(shadow_contrib, 0.0);
        path_state[pixel_index].shadow_radiance = vec4<f32>(0.0);
        path_state[pixel_index].state_u32.z = 0u;
    }

    // Get sun direction for sky evaluation
    let light_view_index = u32(scene_lighting_data.view_index);
    let sun_dir = normalize(-view_buffer[light_view_index].view_direction.xyz);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Handle sky miss - ray escaped the scene
    // ─────────────────────────────────────────────────────────────────────────
    if (path_state[pixel_index].state_u32.w == 0xffffffffu && path_state[pixel_index].state_u32.y != 0u) {
        let ray_dir = normalize(path_state[pixel_index].direction_tmax.xyz);
        
        // Evaluate environment radiance
        let sky_radiance = evaluate_environment(
            ray_dir, 
            sun_dir, 
            scene_lighting_data,
            skybox_texture,
        );
        
        // Add sky contribution weighted by path throughput
        let sky_contrib = safe_clamp_vec3_max(sky_radiance, MAX_RADIANCE_LUMINANCE);
        path_state[pixel_index].accumulated_radiance += vec4<f32>(sky_contrib, 0.0);
        
        // Mark path as complete
        path_state[pixel_index].state_u32.y = 0u;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Shade valid surface hit
    // ─────────────────────────────────────────────────────────────────────────
    if (path_state[pixel_index].state_u32.w != 0xffffffffu && path_state[pixel_index].state_u32.y != 0u) {
        // =====================================================================
        // Material Property Extraction
        // =====================================================================
        var albedo: vec3<f32>;
        var roughness: f32;
        var metallic: f32;
        var emissive: f32;
        var reflectance: f32;
        
        let hit_pos = path_state[pixel_index].origin_tmin.xyz;
        let world_n = path_state[pixel_index].normal_section_index.xyz;
        var n = world_n;
        
        // Get material from texture sampling
        let prim_store = u32(path_state[pixel_index].direction_tmax.w);
        let entity_palette_base = material_table_offset[prim_store];
        let section_index = u32(path_state[pixel_index].normal_section_index.w);
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        let tiling = material.emission_roughness_metallic_tiling.w;
        
        // Reconstruct UVs from barycentrics if needed
        let has_deferred_uv = (path_state[pixel_index].hit_attr1.x == 0.0 && path_state[pixel_index].hit_attr1.y == 0.0 && path_state[pixel_index].hit_attr1.z == 0.0);
        var base_uv: vec2<f32>;
        if (has_deferred_uv) {
            let v0i = u32(path_state[pixel_index].hit_attr0.x);
            let v1i = u32(path_state[pixel_index].hit_attr0.y);
            let v2i = u32(path_state[pixel_index].hit_attr0.z);
            let u_bc = path_state[pixel_index].hit_attr0.w;
            let v_bc = path_state[pixel_index].hit_attr1.w;
            let w_bc = 1.0 - u_bc - v_bc;
            let uv0 = vertex_uv(vertex_buffer[v0i]);
            let uv1 = vertex_uv(vertex_buffer[v1i]);
            let uv2 = vertex_uv(vertex_buffer[v2i]);
            base_uv = uv0 * w_bc + uv1 * u_bc + uv2 * v_bc;
        } else {
            base_uv = vec2<f32>(path_state[pixel_index].hit_attr0.w, path_state[pixel_index].hit_attr1.w);
        }
        base_uv = base_uv * tiling;
        let lod = 0.0;

        // Sample material textures
        albedo = sample_texture_or_vec4_param_handle(
            u32(material.albedo_handle), base_uv, material.albedo,
            u32(material.texture_flags1.x), texture_pool_albedo, lod
        ).xyz;
        roughness = sample_texture_or_float_param_handle(
            u32(material.roughness_handle), base_uv,
            material.emission_roughness_metallic_tiling.y,
            u32(material.texture_flags1.z), texture_pool_roughness, lod
        );
        metallic = sample_texture_or_float_param_handle(
            u32(material.metallic_handle), base_uv,
            material.emission_roughness_metallic_tiling.z,
            u32(material.texture_flags1.w), texture_pool_metallic, lod
        );
        emissive = sample_texture_or_float_param_handle(
            u32(material.emission_handle), base_uv,
            material.emission_roughness_metallic_tiling.x,
            u32(material.texture_flags2.w), texture_pool_emission, lod
        );
        let specular = sample_texture_or_float_param_handle(
            u32(material.specular_handle), base_uv,
            material.ao_height_specular.z,
            u32(material.texture_flags2.z), texture_pool_specular, lod
        );
        reflectance = specular;

        // Normal mapping if TBN available
        let world_t = path_state[pixel_index].hit_attr0.xyz;
        let world_b = path_state[pixel_index].hit_attr1.xyz;
        let has_tbn = length(world_t) > 0.0001 && length(world_b) > 0.0001;
        if ((u32(material.texture_flags1.y) & 1u) != 0u && has_tbn) {
            let tbn = mat3x3<f32>(world_t, world_b, world_n);
            let nm = sample_handle_rgba(
                u32(material.normal_handle), base_uv,
                texture_pool_normal, lod
            ).xyz * 2.0 - 1.0;
            n = normalize(tbn * nm);
        }

        let clear_coat = 0.0;
        let clear_coat_roughness = 0.0;
        let clamped_roughness = clamp(roughness, 0.001, 1.0);
        let v_dir = -normalize(path_state[pixel_index].direction_tmax.xyz);
        let n_dot_v = max(dot(v_dir, n), 0.0001);
        
        // Check if this is bounce 0 (first hit in non-gbuffer mode) for demodulation
        let is_bounce_0 = current_bounce == 0u;
        
        // For bounce 0, store primary_albedo and demodulate; otherwise use existing
        if (is_bounce_0) {
            path_state[pixel_index].primary_albedo = vec4<f32>(albedo, 1.0);
        }
        let primary_albedo = path_state[pixel_index].primary_albedo.xyz;

        // =====================================================================
        // Emissive Contribution (Demodulated at bounce 0)
        // =====================================================================
        if (emissive > 0.0) {
            let emissive_radiance = emissive * albedo;
            var emissive_contribution = emissive_radiance * path_state[pixel_index].path_weight.xyz;
            
            // Demodulate at bounce 0: divide by primary_albedo (will be reapplied in output)
            if (is_bounce_0) {
                emissive_contribution = demodulate(emissive_contribution, primary_albedo);
            }
            
            path_state[pixel_index].accumulated_radiance += vec4<f32>(
                safe_clamp_vec3_max(emissive_contribution, MAX_NEE_LUMINANCE), 
                0.0
            );
        }
        
        // =====================================================================
        // RNG Setup
        // =====================================================================
        var rng = u32(path_state[pixel_index].rng_sample_count.x);
        if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
        else { rng = random_seed(rng); }

        // =====================================================================
        // BRDF Setup
        // =====================================================================
        let dielectric_f0 = 0.16 * reflectance * reflectance;
        let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
        
        // Fresnel at normal incidence for sampling probability
        let f = f_schlick_vec3(f0, 1.0, n_dot_v);
        let fresnel_luminance = luminance(f);
        //let fresnel_luminance = (f.x + f.y + f.z) / 3.0;
        
        // Probability of sampling specular vs diffuse
        let use_ggx = (roughness < 0.3) || (metallic > 0.5);
        let specular_prob_if_ggx = clamp(fresnel_luminance, 0.001, 0.99);
        let specular_prob = select(0.0, specular_prob_if_ggx, use_ggx);

        // =====================================================================
        // Direct Lighting via Next Event Estimation (NEE) - Demodulated at bounce 0
        // =====================================================================
        let num_lights = dense_lights_buffer.header.light_count;
        if (num_lights > 0u) {
            // Sample one random light
            rng = random_seed(rng);
            let light_idx = u32(rand_float(rng) * f32(num_lights)) % num_lights;
            let light = dense_lights_buffer.lights[light_idx];
            
            let light_dir = get_light_dir(light, hit_pos);
            let attenuation = get_light_attenuation(light, hit_pos);
            
            // Evaluate BRDF for this light direction
            let brdf = calculate_brdf_rt(
                n, v_dir, light_dir, albedo, roughness, metallic,
                reflectance, clear_coat, clear_coat_roughness
            );
            
            // Light contribution (multiply by num_lights for unbiased estimator)
            var raw_light_contrib = brdf * light.color.rgb * light.intensity * attenuation
                * path_state[pixel_index].path_weight.xyz * f32(num_lights);
            
            // Demodulate at bounce 0: divide by primary_albedo (will be reapplied in output)
            if (is_bounce_0) {
                raw_light_contrib = demodulate(raw_light_contrib, primary_albedo);
            }
            let light_contrib = safe_clamp_vec3_max(raw_light_contrib, MAX_NEE_LUMINANCE);
            
            // Setup shadow ray for visibility test
            let light_distance = select(1e30, length(light.position.xyz - hit_pos), light.light_type != 0.0);
            path_state[pixel_index].shadow_origin = vec4<f32>(hit_pos + n * 0.001, 0.0001);
            path_state[pixel_index].shadow_direction = vec4<f32>(light_dir, light_distance * 0.999);
            path_state[pixel_index].shadow_radiance = vec4<f32>(light_contrib, 1.0);
        }

        // =====================================================================
        // Sample BRDF for Next Bounce Direction
        // =====================================================================
        rng = random_seed(rng);
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);
        rng = random_seed(rng);
        let r3 = rand_float(rng);
        
        var next_dir: vec3<f32>;
        var pdf: f32;
        var brdf_value: vec3<f32>;
        
        let is_specular_lobe = r3 < specular_prob;
        if (is_specular_lobe) {
            // ─────────────────────────────────────────────────────────────────
            // GGX Specular Sampling
            // ─────────────────────────────────────────────────────────────────
            let h = sample_ggx(n, clamped_roughness, r1, r2);
            next_dir = normalize(reflect(-v_dir, h));
        } else {
            // ─────────────────────────────────────────────────────────────────
            // Cosine-Weighted Diffuse Sampling
            // ─────────────────────────────────────────────────────────────────
            next_dir = sample_uniform_hemisphere(n, r1, r2);
        }
        
        pdf = brdf_pdf(n, v_dir, next_dir, roughness, specular_prob);

        // Evaluate full BRDF for sampled direction
        brdf_value = calculate_brdf_rt(
            n, v_dir, next_dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        // =====================================================================
        // Update Path Throughput and Spawn Next Ray - Demodulated at bounce 0
        // =====================================================================
        let safe_pdf = max(pdf, 0.0001);
        
        // Monte Carlo estimator: (BRDF * cos(theta)) / PDF
        // Note: calculate_brdf_rt already includes the cosine term
        // At bounce 0: demodulate BRDF by primary_albedo so path_weight doesn't include it
        // This ensures all subsequent bounce contributions are also demodulated
        var throughput_brdf = brdf_value;
        if (is_bounce_0) {
            throughput_brdf = demodulate(brdf_value, primary_albedo);
        }
        let throughput_update = throughput_brdf / safe_pdf;
        let new_path_weight = path_state[pixel_index].path_weight.xyz * throughput_update;
        
        let reached_max_bounces = (path_state[pixel_index].state_u32.x + 1u) > pt_params.max_bounces;
        
        if (reached_max_bounces) {
            // Path terminates
            path_state[pixel_index].state_u32.y = 0u;
        } else {
            // Spawn next ray
            path_state[pixel_index].origin_tmin = vec4<f32>(hit_pos + n * 0.001, 0.0001);
            path_state[pixel_index].direction_tmax = vec4<f32>(next_dir, 1e30);
            path_state[pixel_index].state_u32.x = path_state[pixel_index].state_u32.x + 1u;
            path_state[pixel_index].state_u32.w = 0xffffffffu; // Mark as needing intersection test
        }

        path_state[pixel_index].path_weight = vec4<f32>(new_path_weight, 0.0);
        path_state[pixel_index].rng_sample_count.x = f32(rng);
    }
}

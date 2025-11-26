// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               PER-PIXEL PATH TRACING - RAY INITIALIZATION                 ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Initializes rays for per-pixel path tracing with:                        ║
// ║  • Tile-based stochastic pixel selection                                  ║
// ║  • BRDF-importance sampled ray directions                                 ║
// ║  • ReSTIR-based path sampling for high-quality convergence                ║
// ║  • Next Event Estimation (NEE) for direct lighting                        ║
// ║                                                                           ║
// ║  Each invocation corresponds to one tile (upscale_x × upscale_y pixels).  ║
// ║  A random pixel within the tile is selected for tracing this frame.       ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"
#include "raytracing/restir_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> gi_counters: GICounters;
@group(1) @binding(2) var<storage, read_write> pixel_path_state: array<PixelPathState>;
@group(1) @binding(3) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(4) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(5) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(6) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(7) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(8) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(10) var gbuffer_motion: texture_2d<f32>;

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ─────────────────────────────────────────────────────────────────────────
    // Calculate tile and ray indices
    // We dispatch one thread per tile, each tile samples one random pixel
    // ─────────────────────────────────────────────────────────────────────────
    let rays_per_tile = u32(gi_params.screen_ray_count);
    let resolution = vec2<u32>(u32(gi_params.resolution_x), u32(gi_params.resolution_y));
    let upscale = vec2<u32>(u32(gi_params.upscale_x), u32(gi_params.upscale_y));
    let tile_grid_dims = vec2<u32>(resolution.x / upscale.x, resolution.y / upscale.y);
    
    // Compute total tiles from resolution and upscale
    let total_tiles = tile_grid_dims.x * tile_grid_dims.y;
    let total_rays = total_tiles * rays_per_tile;
    
    if (gid.x >= total_rays) {
        return;
    }
    
    // Compute tile index and ray index within tile
    let tile_index = gid.x / rays_per_tile;
    let ray_index = gid.x % rays_per_tile;
    let frame_id = u32(gi_params.frame_index);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Initialize RNG with tile and frame for good temporal distribution
    // ─────────────────────────────────────────────────────────────────────────
    var rng = hash(gid.x ^ (frame_id * 0x9E3779B9u));
    
    // ─────────────────────────────────────────────────────────────────────────
    // Stochastic pixel selection within tile
    // ─────────────────────────────────────────────────────────────────────────
    let pixel_coords = tile_to_pixel_stochastic(
        tile_index,
        tile_grid_dims.x,
        upscale,
        resolution,
        &rng
    );
    let pixel_coord = vec2<i32>(i32(pixel_coords.x), i32(pixel_coords.y));
    
    // ─────────────────────────────────────────────────────────────────────────
    // Sample G-buffer at selected pixel location
    // ─────────────────────────────────────────────────────────────────────────
    let position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    
    // Skip sky pixels (no geometry)
    if (normal_length <= 0.0) {
        pixel_path_state[gid.x].state_u32 = vec4<u32>(0u, 0u, 0u, 0xffffffffu);
        pixel_path_state[gid.x].pixel_coords = vec4<f32>(f32(pixel_coords.x), f32(pixel_coords.y), 0.0, 0.0);
        return;
    }
    
    let albedo = textureLoad(gbuffer_albedo, pixel_coord, 0).rgb;
    let smra = textureLoad(gbuffer_smra, pixel_coord, 0);
    let motion_emissive = textureLoad(gbuffer_motion, pixel_coord, 0);

    let roughness = smra.g;
    let metallic = smra.b;
    let reflectance = smra.r * 0.0009765625; // Decode: 1.0 / 1024
    let emissive = motion_emissive.w;
    // Clear coat (not stored in G-buffer, assume none for now)
    let clear_coat = 0.0;
    let clear_coat_roughness = 0.0;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Get view direction
    // ─────────────────────────────────────────────────────────────────────────
    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    let v_dir = normalize(view.view_position.xyz - position);
    let n_dot_v = max(dot(v_dir, normal), 0.0001);
    
    // ═════════════════════════════════════════════════════════════════════════
    // BRDF-GUIDED IMPORTANCE SAMPLING WITH ReSTIR
    // ═════════════════════════════════════════════════════════════════════════
    
    let clamped_roughness = clamp(roughness, 0.0001, 1.0);
    let dielectric_f0 = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(dielectric_f0), albedo, metallic);
    
    let f = f_schlick_vec3(f0, 1.0, n_dot_v);
    let fresnel_luminance = (f.x + f.y + f.z) / 3.0;
    
    // Use GGX for glossy/metallic, cosine for rough diffuse
    let use_ggx = (clamped_roughness < 0.3) || (metallic > 0.5);
    let specular_prob_if_ggx = clamp(fresnel_luminance, 0.001, 0.99);
    let mis_specular_prob = select(0.0, specular_prob_if_ggx, use_ggx);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Generate BRDF sampling candidates
    // ─────────────────────────────────────────────────────────────────────────
    var candidate_samples: array<GISample, num_ris_samples>;
    
    for (var i = 0u; i < num_ris_samples; i = i + 1u) {
        rng = random_seed(rng);
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);
        rng = random_seed(rng);
        let r3 = rand_float(rng);
        
        var dir: vec3<f32>;
        if (use_ggx && r3 < specular_prob_if_ggx) {
            // GGX sampling for specular lobes
            let h = importance_sample_ggx(vec2<f32>(r1, r2), normal, clamped_roughness);
            dir = normalize(reflect(-v_dir, h));
        } else {
            // Cosine-weighted hemisphere sampling for diffuse
            let phi = 2.0 * PI * r1;
            let cos_theta = sqrt(1.0 - r2);
            let sin_theta = sqrt(r2);
            let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.999);
            let tangent = normalize(cross(up, normal));
            let bitangent = normalize(cross(normal, tangent));
            let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            dir = normalize(tangent * dir_local.x + bitangent * dir_local.y + normal * dir_local.z);
        }
        
        // Evaluate BRDF for this direction
        let brdf = calculate_brdf_rt(
            normal, v_dir, dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        let brdf_sample_pdf = brdf_pdf(normal, v_dir, dir, clamped_roughness, mis_specular_prob);
        let brdf_lum = max(0.0, brdf.x * 0.2126 + brdf.y * 0.7152 + brdf.z * 0.0722);
        
        candidate_samples[i].radiance_and_target_pdf = vec4f(brdf, brdf_lum);
        candidate_samples[i].direction_and_source_pdf = vec4f(dir, brdf_sample_pdf);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Perform RIS on candidates
    // ─────────────────────────────────────────────────────────────────────────
    var gi_reservoir = gi_reservoir_init();
    for (var i = 0u; i < num_ris_samples; i = i + 1u) {
        let sample = candidate_samples[i];
        let target_pdf = sample.radiance_and_target_pdf.w;
        let ris_weight = target_pdf / max(sample.direction_and_source_pdf.w, 0.0001);
        
        if (ris_weight > 0.0 && !isinf(ris_weight)) {
            gi_reservoir_update(&gi_reservoir, i, ris_weight, &rng);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Finalize reservoir and select best direction
    // ─────────────────────────────────────────────────────────────────────────
    var ray_dir: vec3<f32>;
    var path_weight = vec3<f32>(1.0, 1.0, 1.0);
    var ray_source_pdf = 0.0;
    var is_alive = 1u;
    
    if (gi_reservoir.m > 0u) {
        let selected_sample = candidate_samples[gi_reservoir.selected_index];
        let selected_dir = selected_sample.direction_and_source_pdf.xyz;
        let selected_brdf = selected_sample.radiance_and_target_pdf.xyz;
        let selected_target = selected_sample.radiance_and_target_pdf.w;
        gi_reservoir_finalize(&gi_reservoir, selected_target);
        
        ray_dir = selected_dir;
        ray_source_pdf = selected_sample.direction_and_source_pdf.w;
        
        // Update path weight with BRDF and reservoir weight
        let brdf_weight = selected_brdf * gi_reservoir.w;
        path_weight = brdf_weight * gi_params.indirect_boost;
        
        // Russian Roulette: Kill paths with very low throughput
        let weight_luminance = path_weight.x * 0.2126 + path_weight.y * 0.7152 + path_weight.z * 0.0722;
        let min_weight_threshold = 0.0001;
        
        if (weight_luminance < min_weight_threshold) {
            is_alive = 0u;
            ray_source_pdf = 0.0;
            pixel_path_state[gid.x].reservoir_radiance_m = vec4f(0.0);
            pixel_path_state[gid.x].reservoir_direction_w = vec4f(0.0);
        } else {
            // Store reservoir data for potential temporal reuse
            pixel_path_state[gid.x].reservoir_radiance_m = vec4f(selected_sample.radiance_and_target_pdf.xyz, f32(gi_reservoir.m));
            pixel_path_state[gid.x].reservoir_direction_w = vec4f(selected_dir, gi_reservoir.w);
        }
    } else {
        // Reservoir failed - generate fallback direction
        is_alive = 0u;
        pixel_path_state[gid.x].reservoir_radiance_m = vec4f(0.0);
        pixel_path_state[gid.x].reservoir_direction_w = vec4f(0.0);
        
        rng = random_seed(rng);
        let u1 = rand_float(rng);
        rng = random_seed(rng);
        let u2 = rand_float(rng);
        ray_dir = sample_cosine_hemisphere(u1, u2, normal);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // NEXT EVENT ESTIMATION (NEE) - DIRECT LIGHTING
    // Setup shadow rays for direct lighting contribution
    // ═════════════════════════════════════════════════════════════════════════
    
    let num_lights = light_count_buffer[0];
    if (num_lights > 0u) {
        rng = random_seed(rng);
        let light_idx = u32(rand_float(rng) * f32(num_lights)) % num_lights;
        let light = dense_lights_buffer[light_idx];
        
        let light_dir = get_light_dir(light, position);
        let attenuation = get_light_attenuation(light, position);
        
        let brdf = calculate_brdf_rt(
            normal, v_dir, light_dir, albedo, roughness, metallic,
            reflectance, clear_coat, clear_coat_roughness
        );
        
        let light_contrib = brdf * light.color.rgb * light.intensity * attenuation * f32(num_lights);
        
        // Setup shadow ray for visibility test
        let selected_distance = select(1e30, length(light.position.xyz - position), light.light_type != 0.0);
        pixel_path_state[gid.x].shadow_origin = vec4f(position + normal * 0.001, f32(light_idx));
        pixel_path_state[gid.x].shadow_direction = vec4f(light_dir, selected_distance * 0.999);
        pixel_path_state[gid.x].shadow_radiance = vec4f(light_contrib, 1.0);
    } else {
        pixel_path_state[gid.x].shadow_origin = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        pixel_path_state[gid.x].shadow_direction = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        pixel_path_state[gid.x].shadow_radiance = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // ═════════════════════════════════════════════════════════════════════════
    // STORE PATH STATE
    // ═════════════════════════════════════════════════════════════════════════
    pixel_path_state[gid.x].origin_tmin = vec4<f32>(position + normal * 0.001, 0.0001);
    pixel_path_state[gid.x].direction_tmax = vec4<f32>(ray_dir, 1e30);
    pixel_path_state[gid.x].normal_section_index = vec4<f32>(normal, 0.0);
    pixel_path_state[gid.x].state_u32 = vec4<u32>(0u, is_alive, 0u, 0xffffffffu);
    pixel_path_state[gid.x].hit_attr0 = vec4<f32>(0.0);
    pixel_path_state[gid.x].hit_attr1 = vec4<f32>(0.0);
    pixel_path_state[gid.x].rng_sample_count_frame_stamp = vec4<f32>(f32(rng), 0.0, f32(frame_id), 0.0);
    pixel_path_state[gid.x].path_weight = vec4<f32>(path_weight, ray_source_pdf);
    pixel_path_state[gid.x].throughput = vec4<f32>(emissive * albedo, 0.0);
    // Store pixel coordinates for update pass
    pixel_path_state[gid.x].pixel_coords = vec4<f32>(f32(pixel_coords.x), f32(pixel_coords.y), 0.0, 0.0);
}

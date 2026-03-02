// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               DDGI SPHERICAL HARMONICS PROBE SAMPLING                     ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Samples diffuse irradiance from SH-encoded probe grid.                   ║
// ║  Uses trilinear interpolation of SH coefficients for smooth results.      ║
// ║                                                                           ║
// ║  Advantages over octahedral sampling:                                     ║
// ║  • SH coefficients interpolate linearly (no octahedral edge artifacts)    ║
// ║  • More compact representation (12 floats vs 64 for 8x8 octahedral)       ║
// ║  • Natural low-frequency filtering (L1 captures dominant direction)       ║
// ║  • Efficient irradiance calculation via cosine lobe convolution           ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "postprocess_common.wgsl"
#include "sky_common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(2) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(3) var<storage, read> probe_depth_moments: array<u32>;
@group(1) @binding(4) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var output_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(8) var skybox_texture: texture_cube<f32>;

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ─────────────────────────────────────────────────────────────────────────
    // Bounds check
    // ─────────────────────────────────────────────────────────────────────────
    let res = textureDimensions(output_diffuse);
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Skip sky pixels (no geometry)
    // ─────────────────────────────────────────────────────────────────────────
    if (length(normal_data.xyz) <= 0.0) {
        textureStore(output_diffuse, pixel_coord, vec4f(0.0));
        return;
    }
    
    let normal = safe_normalize(normal_data.xyz);
    let cascade_count = ddgi_cascade_count(&ddgi_params);
    let highest_cascade_index = cascade_count - 1u;
    let is_inside_highest_cascade = ddgi_position_inside_cascade_bounds(
        &ddgi_params,
        highest_cascade_index,
        position
    );

    var irradiance = vec3<f32>(0.0);
    if (!is_inside_highest_cascade) {
        let light_view_index = u32(scene_lighting_data.view_index);
        let sun_dir = normalize(-view_buffer[light_view_index].view_direction.xyz);
        irradiance = evaluate_environment(
            normal,
            sun_dir,
            scene_lighting_data,
            skybox_texture
        );
    } else {
        // ─────────────────────────────────────────────────────────────────────────
        // Sample SH irradiance using shared helper (includes visibility weighting,
        // robust fallbacks, and indirect_boost).
        // ─────────────────────────────────────────────────────────────────────────
        irradiance = ddgi_sample_sh_irradiance_with_states(
            &ddgi_params,
            &sh_probes,
            &probe_states,
            &probe_depth_moments,
            position,
            normal
        );
    }

    textureStore(output_diffuse, pixel_coord, vec4f(irradiance, 1.0));
}


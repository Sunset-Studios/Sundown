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

// Sky pixels have zero or near-zero normal length; use squared length to avoid sqrt.
const DDGI_SAMPLE_SKY_NORMAL_LENGTH_SQ_EPS: f32 = 1e-10;

fn ddgi_sample_output_to_full_res_coord(
    sample_coord: vec2<u32>,
    sample_res: vec2<u32>,
    full_res: vec2<u32>
) -> vec2<i32> {
    let uv = (vec2<f32>(sample_coord) + vec2<f32>(0.5)) / vec2<f32>(f32(sample_res.x), f32(sample_res.y));
    let full_coord = min(
        vec2<u32>(uv * vec2<f32>(f32(full_res.x), f32(full_res.y))),
        full_res - vec2<u32>(1u)
    );
    return vec2<i32>(i32(full_coord.x), i32(full_coord.y));
}

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(2) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(3) var<storage, read> probe_depth_moments: array<u32>;
@group(1) @binding(4) var hzb_texture: texture_2d<f32>;
@group(1) @binding(5) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(6) var output_diffuse: texture_storage_2d<rgba16float, write>;
@group(1) @binding(7) var<uniform> scene_lighting_data: SceneLightingData;
@group(1) @binding(8) var skybox_texture: texture_cube<f32>;

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ─────────────────────────────────────────────────────────────────────────
    // Bounds check
    // ─────────────────────────────────────────────────────────────────────────
    let sample_res = textureDimensions(output_diffuse);
    if (gid.x >= sample_res.x || gid.y >= sample_res.y) {
        return;
    }
    
    let view_index = u32(frame_info.view_index);
    let sample_coord_u32 = gid.xy;
    let sample_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let full_res = textureDimensions(gbuffer_normal);
    let full_pixel_coord = ddgi_sample_output_to_full_res_coord(sample_coord_u32, sample_res, full_res);
    let uv =
        (vec2<f32>(f32(full_pixel_coord.x), f32(full_pixel_coord.y)) + vec2<f32>(0.5))
        / vec2<f32>(f32(full_res.x), f32(full_res.y));

    let depth = textureSampleLevel(hzb_texture, non_filtering_sampler, uv, 0.0).r;
    let position = reconstruct_world_position(uv, depth, view_index);
    let normal_data = textureLoad(gbuffer_normal, full_pixel_coord, 0);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Skip sky pixels (no geometry). Squared length avoids sqrt.
    // ─────────────────────────────────────────────────────────────────────────
    if (dot(normal_data.xyz, normal_data.xyz) <= DDGI_SAMPLE_SKY_NORMAL_LENGTH_SQ_EPS) {
        textureStore(output_diffuse, sample_coord, vec4<f32>(0.0));
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

    textureStore(output_diffuse, sample_coord, vec4<f32>(irradiance, 1.0));
}


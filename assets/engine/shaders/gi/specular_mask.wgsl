// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                       GI - SPECULAR MASK (GBUFFER)                        ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Builds a low-res (GI resolution) 0/1 mask indicating which pixels should ║
// ║  run the *per-pixel* GI pipeline. This is intended to gate expensive      ║
// ║  tracing + ReSTIR stages to specular-relevant pixels.                     ║
// ║                                                                           ║
// ║  Output format: r32uint                                                   ║
// ║    - 0u: skip                                                             ║
// ║    - 1u: run                                                              ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "gi/gi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(3) var out_specular_mask: texture_storage_2d<r32uint, write>;

// =============================================================================
// TUNABLES
// =============================================================================

const ROUGHNESS_THRESHOLD: f32 = 0.2;
const REFLECTANCE_THRESHOLD: f32 = 0.04;
const METALLIC_THRESHOLD: f32 = 0.05;

// =============================================================================
// MAIN
// =============================================================================

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gi_res = vec2<u32>(u32(gi_params.gi_resolution_x), u32(gi_params.gi_resolution_y));
    if (gid.x >= gi_res.x || gid.y >= gi_res.y) {
        return;
    }

    let full_res = vec2<u32>(u32(gi_params.full_resolution_x), u32(gi_params.full_resolution_y));
    let upscale_factor = u32(gi_params.upscale_factor);

    let full_pixel_coord = gi_pixel_to_full_res_pixel_coord(gid.xy, upscale_factor, full_res);

    let normal_data = textureLoad(gbuffer_normal, full_pixel_coord, 0u);
    // Sky pixels: always off.
    if (length(normal_data.xyz) <= 0.0) {
        textureStore(out_specular_mask, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    let smra = textureLoad(gbuffer_smra, full_pixel_coord, 0u);
    let roughness = smra.g;
    let metallic = smra.b;
    let reflectance = smra.r;

    let mask = select(0u, 1u, (roughness <= ROUGHNESS_THRESHOLD) && ((metallic >= METALLIC_THRESHOLD) || (reflectance >= REFLECTANCE_THRESHOLD)));
    textureStore(out_specular_mask, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<u32>(mask, 0u, 0u, 0u));
}



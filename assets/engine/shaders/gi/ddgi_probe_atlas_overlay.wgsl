// =============================================================================
// DDGI Probe Irradiance Atlas Overlay
// - Overlays the probe irradiance atlas on top of the scene so it can be
//   inspected directly.
// =============================================================================
#include "common.wgsl"
#include "gi/ddgi_common.wgsl"

@group(1) @binding(0) var probe_irradiance_atlas: texture_2d<f32>;
@group(1) @binding(1) var scene_color: texture_2d<f32>;
@group(1) @binding(2) var output_debug: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_dims = textureDimensions(output_debug);
    if (gid.x >= screen_dims.x || gid.y >= screen_dims.y) {
        return;
    }

    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let scene_rgb = textureLoad(scene_color, pixel_coord, 0).rgb;

    let atlas_dims_u32 = textureDimensions(probe_irradiance_atlas);
    let atlas_dims = vec2<f32>(f32(atlas_dims_u32.x), f32(atlas_dims_u32.y));
    let screen_dims_f = vec2<f32>(f32(screen_dims.x), f32(screen_dims.y));

    // -------------------------------------------------------------------------
    // Overlay rectangle (bottom-left) that preserves atlas aspect ratio
    // -------------------------------------------------------------------------
    let margin = 16.0;
    // Size cap for the on-screen overlay (bumped up for easier inspection).
    let max_overlay = vec2<f32>(screen_dims_f.x * 0.65, screen_dims_f.y * 0.65);
    let scale = min(max_overlay.x / max(atlas_dims.x, 1.0), max_overlay.y / max(atlas_dims.y, 1.0));
    let overlay_size = atlas_dims * scale;
    let overlay_origin = vec2<f32>(margin, screen_dims_f.y - margin - overlay_size.y);

    let p = vec2<f32>(f32(pixel_coord.x) + 0.5, f32(pixel_coord.y) + 0.5);
    let local = p - overlay_origin;

    let inside =
        local.x >= 0.0 && local.y >= 0.0 &&
        local.x < overlay_size.x && local.y < overlay_size.y;

    // Border (1px) for visibility.
    let border_px = 1.0;
    let on_border = inside && (
        local.x < border_px ||
        local.y < border_px ||
        local.x >= overlay_size.x - border_px ||
        local.y >= overlay_size.y - border_px
    );

    // Sample atlas if inside, otherwise show scene.
    var out_rgb = scene_rgb;
    let safe_overlay_size = max(overlay_size, vec2<f32>(1.0));
    let atlas_uv_unclamped = local / safe_overlay_size;
    let atlas_uv = clamp(atlas_uv_unclamped, vec2<f32>(0.0), vec2<f32>(1.0));
    let atlas_sample = textureSampleLevel(probe_irradiance_atlas, global_sampler, atlas_uv, 0.0).rgb;

    // Blend overlay over the scene.
    let overlay_alpha = 0.90;
    let overlay_rgb = mix(scene_rgb, atlas_sample, overlay_alpha);
    out_rgb = select(out_rgb, overlay_rgb, inside);

    out_rgb = select(out_rgb, vec3<f32>(1.0), on_border);
    textureStore(output_debug, pixel_coord, vec4<f32>(out_rgb, 1.0));
}



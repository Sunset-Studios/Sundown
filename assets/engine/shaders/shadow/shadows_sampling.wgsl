#include "shadow/shadows_common.wgsl"

// ------------------------------------------------------------------------------------
// Shadows
// ------------------------------------------------------------------------------------
#if SHADOWS_ENABLED

fn vsm_shadow_depth(
    world_pos: vec4<f32>,
    view_idx: u32,
    vsm_settings: ASVSMSettings,
) -> f32 {
  let camera_vp           = view_buffer[frame_info.view_index].view_projection_matrix;
  let light_vp            = view_buffer[view_idx].view_projection_matrix;
  let vtile_info          = vsm_world_to_virtual_tile(world_pos, camera_vp, light_vp, vsm_settings);

  let light_clip_pos      = vsm_calculate_render_clip_value_from_world_pos(
                                world_pos,
                                vtile_info.clipmap_index,
                                light_vp,
                                vsm_settings
                            );
  let depth_ndc           = light_clip_pos.z;

  return depth_ndc;
}

// Performs a 3×3 PCF (percentage-closer filter) from the shadow atlas for the fragment at
// world_pos. It uses the same virtual-→physical mapping logic as the regular
// sample helper and falls back to 1.0 when the page is not resident.
fn vsm_sample_shadow_bilinear(
    world_pos: vec4<f32>,
    view_idx: u32,
    shadow_idx: u32,
    page_table: texture_storage_2d_array<r32uint, read>,
    settings: ASVSMSettings,
) -> ShadowFilterResult {
    var out: ShadowFilterResult;

    // Build tile mappings
    let camera_vp   = view_buffer[frame_info.view_index].view_projection_matrix;
    let light_vp    = view_buffer[view_idx].view_projection_matrix;
    let vtile_info  = vsm_world_to_virtual_tile(world_pos, camera_vp, light_vp, settings);
    let ptile_info  = vsm_vtile_to_ptile(vtile_info, settings, shadow_idx, page_table);

    // 3×3 PCF gather inside the atlas
    // let phys_dim    = u32(settings.physical_dim);
    // let pool_stride = phys_dim * phys_dim;
    // let pool_idx    = ptile_info.memory_pool_index;

    // let pixel_f     = ptile_info.physical_pixel;
    // let base        = vec2<u32>(pixel_f);

    // let x0          = base.x;
    // let y0          = base.y;
    // let x1          = max(x0 - 1u, 0u);
    // let y1          = max(y0 - 1u, 0u);
    // let x2          = min(x0 + 1u, phys_dim - 1u);
    // let y2          = min(y0 + 1u, phys_dim - 1u);

    // let base_index  = pool_idx * pool_stride;
    // let idx00       = base_index + y0 * phys_dim + x0;
    // let idx10       = base_index + y0 * phys_dim + x1;
    // let idx01       = base_index + y1 * phys_dim + x0;
    // let idx11       = base_index + y1 * phys_dim + x1;
    // let idx02       = base_index + y2 * phys_dim + x0;
    // let idx12       = base_index + y2 * phys_dim + x1;
    // let idx20       = base_index + y0 * phys_dim + x2;
    // let idx21       = base_index + y1 * phys_dim + x2;
    // let idx22       = base_index + y2 * phys_dim + x2;

    // let d00         = unpack_depth(shadow_atlas_depth[idx00]);
    // let d10         = unpack_depth(shadow_atlas_depth[idx10]);
    // let d01         = unpack_depth(shadow_atlas_depth[idx01]);
    // let d11         = unpack_depth(shadow_atlas_depth[idx11]);
    // let d02         = unpack_depth(shadow_atlas_depth[idx02]);
    // let d12         = unpack_depth(shadow_atlas_depth[idx12]);
    // let d20         = unpack_depth(shadow_atlas_depth[idx20]);
    // let d21         = unpack_depth(shadow_atlas_depth[idx21]);
    // let d22         = unpack_depth(shadow_atlas_depth[idx22]);

    // // Average the nine sampled depths (simple PCF)
    // let filtered    = (d00 + d10 + d01 + d11 + d02 + d12 + d20 + d21 + d22) / 9.0;
    let filtered    = unpack_depth(shadow_atlas_depth[ptile_info.physical_id]);

    // Validate residency for this virtual tile
    let page_index  = vtile_info.clipmap_index + shadow_idx * u32(settings.max_lods);
    let entry       = textureLoad(page_table, vtile_info.tile_coords, page_index).r;
    let page_valid  = vsm_pte_is_valid(entry);

    out.depth = filtered;
    out.valid = page_valid;
    return out;
}

#endif
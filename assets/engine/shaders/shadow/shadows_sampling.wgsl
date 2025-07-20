#include "shadow/shadows_common.wgsl"

// ------------------------------------------------------------------------------------
// Shadows
// ------------------------------------------------------------------------------------
#if SHADOWS_ENABLED

const constant_bias = 1.5;
const slope_bias    = 1.75;

fn vsm_shadow_depth(
    world_pos: vec4<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    view_idx: u32,
    shadow_idx: u32,
    page_offset: texture_storage_2d_array<rgba32float, read>,
    vsm_settings: ASVSMSettings,
) -> f32 {
  let camera_vp           = view_buffer[u32(frame_info.view_index)].view_projection_matrix;
  let light_vp            = view_buffer[view_idx].view_projection_matrix;
  let vtile_info          = vsm_world_to_virtual_tile(world_pos, camera_vp, light_vp, vsm_settings);

  let page_index          = vtile_info.clipmap_index + shadow_idx * u32(vsm_settings.max_lods);
  let offset              = textureLoad(page_offset, vtile_info.tile_coords, page_index);

  let light_projection    = view_buffer[view_idx].projection_matrix;

  var adjusted_light_view = view_buffer[view_idx].view_matrix;
  adjusted_light_view[3]  = offset;

  let new_light_vp        = light_projection * adjusted_light_view;

  // ------------------------------------------------------------------
  // Normal-offset bias (see ‘normal offset shadows’, Holbert GDC 2011)
  // We move the receiver along its own normal by <normal_offset> texels.
  // ------------------------------------------------------------------

  // texel size in world units for current clip-map level
  let clip_extent  = f32(1u << vtile_info.clipmap_index) * vsm_settings.clip0_extent;
  let texel_world  = clip_extent / vsm_settings.virtual_dim;

  // ------------------------------------------------------------------
  // Hybrid constant-and-slope bias   (see Holbert 2011 + Epic notes)
  // ------------------------------------------------------------------
  let ndotl        = dot(normal, light_dir);
  let eps          = 0.04;
  let bias_texels  = constant_bias + slope_bias / max(abs(ndotl), eps);

  let normal_offset      = texel_world * bias_texels;

  let normal_shifted_pos = world_pos + vec4<f32>(normal * normal_offset, 0.0);

  let light_clip_pos      = vsm_calculate_render_clip_value_from_world_pos(
                                normal_shifted_pos,
                                vtile_info.clipmap_index,
                                new_light_vp,
                                vsm_settings
                            );
  let depth_ndc           = light_clip_pos.z / light_clip_pos.w;

  return depth_ndc;
}

// Performs a 3×3 PCF (percentage-closer filter) from the shadow atlas for the fragment at
// world_pos. It uses the same virtual-→physical mapping logic as the regular
// sample helper and falls back to 1.0 when the page is not resident.
fn vsm_sample_shadow(
    ref_depth: f32,
    world_pos: vec4<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    view_idx: u32,
    shadow_idx: u32,
    page_table: texture_storage_2d_array<r32uint, read>,
    settings: ASVSMSettings,
) -> ShadowFilterResult {
    var out: ShadowFilterResult;

    // Build tile mappings
    let camera_vp   = view_buffer[u32(frame_info.view_index)].view_projection_matrix;
    let light_vp    = view_buffer[view_idx].view_projection_matrix;
    let vtile_info  = vsm_world_to_virtual_tile(world_pos, camera_vp, light_vp, settings);
    let ptile_info  = vsm_vtile_to_ptile(vtile_info, settings, shadow_idx, page_table);

    // 3×3 PCF gather inside the atlas
    let mem_pool    = ptile_info.memory_pool_index;
    let phys_dim    = u32(settings.physical_dim);
    let tile_size   = u32(settings.tile_size);

    let pixel       = ptile_info.physical_pixel;

    let tile_origin_x = ptile_info.physical_xy.x * tile_size + 1u;
    let tile_origin_y = ptile_info.physical_xy.y * tile_size + 1u;

    let tile_final_x  = tile_origin_x + tile_size - 2u;
    let tile_final_y  = tile_origin_y + tile_size - 2u;

    let base_index   = mem_pool * phys_dim * phys_dim;
    let is_on_border = pixel.x <= tile_origin_x
        || pixel.x >= tile_final_x
        || pixel.y <= tile_origin_y
        || pixel.y >= tile_final_y;

    let x0 = clamp(pixel.x, tile_origin_x - 1u, tile_final_x + 2u);
    let y0 = clamp(pixel.y, tile_origin_y - 1u, tile_final_y + 2u);
    let x1 = select(clamp(pixel.x - 1u, tile_origin_x - 1u, tile_final_x + 2u), x0, is_on_border);
    let y1 = select(clamp(pixel.y - 1u, tile_origin_y - 1u, tile_final_y + 2u), y0, is_on_border);
    let x2 = select(clamp(pixel.x + 1u, tile_origin_x - 1u, tile_final_x + 2u), x1, is_on_border);
    let y2 = select(clamp(pixel.y + 1u, tile_origin_y - 1u, tile_final_y + 2u), y1, is_on_border);

    let idx00       = base_index + y0 * phys_dim + x0;
    let idx10       = base_index + y0 * phys_dim + x1;
    let idx01       = base_index + y1 * phys_dim + x0;
    let idx11       = base_index + y1 * phys_dim + x1;
    let idx02       = base_index + y2 * phys_dim + x0;
    let idx12       = base_index + y2 * phys_dim + x1;
    let idx20       = base_index + y0 * phys_dim + x2;
    let idx21       = base_index + y1 * phys_dim + x2;
    let idx22       = base_index + y2 * phys_dim + x2;

    let bias        = 0.000001;
    let d00         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx00]) + bias);
    let d10         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx10]) + bias);
    let d01         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx01]) + bias);
    let d11         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx11]) + bias);
    let d02         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx02]) + bias);
    let d12         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx12]) + bias);
    let d20         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx20]) + bias);
    let d21         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx21]) + bias);
    let d22         = select(0.0, 1.0, ref_depth >= unpack_depth(shadow_atlas_depth[idx22]) + bias);

    // Average the nine sampled depths (simple PCF)
    let filtered = (d00 + d10 + d01 + d11 + d02 + d12 + d20 + d21 + d22) / 9.0;

    if (!ptile_info.is_resident) {
        out.depth = 1.0;   // fully lit
        out.valid = false;
        return out;
    }

    out.depth = filtered;
    out.valid = true;

    return out;
}

#endif
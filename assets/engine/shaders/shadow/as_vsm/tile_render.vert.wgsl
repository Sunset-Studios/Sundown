// AS-VSM Stage B: Render Shadow Casters into shadow atlas (Vertex)
// Renders geometry into each requested tile viewport.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> visible_object_instances: array<i32>;
@group(1) @binding(3) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(4) var page_table: texture_storage_2d_array<r32uint, read>; // PTE format: Bit31=Valid, Bits30-27=LOD, Bits26-0=PhysID
@group(1) @binding(5) var<uniform> light_ub: ShadowCasterLight;
@group(1) @binding(6) var<storage, read> bitmask: array<u32>;
@group(1) @binding(7) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> light_shadow_idx_buffer: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) virtual_pixel: vec2<f32>,
  @location(1) @interpolate(flat) clipmap_index: u32,
  @location(2) @interpolate(flat) shadow_index: u32,
};

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32,
      @builtin(instance_index) instance_index: u32) -> VertexOutput {
  var out: VertexOutput;

#if SHADOWS_ENABLED
  let light_idx  = light_ub.light_index;
  let view_index = light_view_buffer[light_idx];
  let shadow_idx = light_shadow_idx_buffer[light_idx];

  if (view_index == 0xffffffffu || shadow_idx == 0xffffffffu) {
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.virtual_pixel = vec2<f32>(0.0);
    out.clipmap_index = 0u;
    out.shadow_index = 0u;
    return out;
  }

  // --------------------------- Model → World
  let object_instance_index = visible_object_instances[instance_index];
  let row_field             = object_instances[object_instance_index].row;
  let entity_row            = get_entity_row(row_field);

  let model_matrix          = entity_transforms[entity_row].transform;
  let local_pos             = vertex_buffer[vertex_index].position;
  var world_pos             = model_matrix * local_pos;
  world_pos.w = 1.0;

  // --------------------------- World → Physical Atlas
  let clipmap0_vp = view_buffer[view_index].view_projection_matrix;
  let camera_vp   = view_buffer[frame_info.view_index].view_projection_matrix;

  let vtile_info = vsm_world_to_virtual_tile(
    world_pos,
    camera_vp,
    clipmap0_vp,
    vsm_settings
  );

  // ────────────────────────────────────────────────────────────────
  // Clip the primitive so it only covers the requested *virtual tile*
  //   1. World → clip-space (full clip-map 0)
  //   2. Convert to virtual-texture UV, compute local UV inside tile
  //   3. Map local UV to clip-space and overwrite X/Y
  // ────────────────────────────────────────────────────────────────

  let word_and_mask = vsm_get_virtual_tile_word_and_mask(
    vtile_info.tile_coords,
    vtile_info.clipmap_index,
    shadow_idx,
    vsm_settings
  );

  let word       = word_and_mask.x;
  let bits       = bitmask[word];
  if (bits == 0u) {
    // Tile not requested, so push vertex outside clip-space so the primitive is clipped
    out.position    = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.virtual_pixel = vec2<f32>(0.0);
    out.clipmap_index = 0u;
    out.shadow_index = 0u;
    return out;
  }

  // ------------------------------------------------------------------
  // Fetch page-table entry for this shadow at the correct LOD slice
  //   Array layer = clip_map_index (per-light offset handled by caller).
  // ------------------------------------------------------------------
  let entry = textureLoad(
      page_table,
      vtile_info.tile_coords,
      vtile_info.clipmap_index + shadow_idx * u32(vsm_settings.max_lods)
  ).r;

  // light-space position for clipmap
  var clip_pos = vsm_calculate_render_clip_value_from_world_pos(
    world_pos,
    vtile_info.clipmap_index,
    clipmap0_vp
  );

  out.position = vsm_tile_transform(clip_pos, vtile_info.tile_xy_f, entry, vsm_settings);
  out.virtual_pixel = vtile_info.virtual_pixel;
  out.clipmap_index = vtile_info.clipmap_index;
  out.shadow_index = shadow_idx;

#else

  out.position = vec4<f32>(0.0);
  out.virtual_pixel = vec2<f32>(0.0);
  out.clipmap_index = 0u;
  out.shadow_index = 0u;

#endif

  return out;
}
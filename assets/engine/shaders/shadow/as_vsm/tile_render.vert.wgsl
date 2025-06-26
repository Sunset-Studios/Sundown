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
  @location(0) world_pos: vec4<f32>,
  @location(1) @interpolate(flat) shadow_index: u32,
  @location(2) @interpolate(flat) view_index: u32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32,
      @builtin(instance_index) ii: u32) -> VertexOutput {
  var out: VertexOutput;

#if SHADOWS_ENABLED
  let light_idx             = light_ub.light_index;
  let view_index            = light_view_buffer[light_idx];
  let shadow_idx            = light_shadow_idx_buffer[light_idx];

  let object_instance_index = visible_object_instances[ii];
  let row_field             = object_instances[object_instance_index].row;
  let entity_row            = get_entity_row(row_field);

  let model_matrix          = entity_transforms[entity_row].transform;
  let local_pos             = vertex_buffer[vi].position;
  let world_pos             = model_matrix * local_pos;

  let clipmap0_vp           = view_buffer[view_index].view_projection_matrix;
  let camera_vp             = view_buffer[frame_info.view_index].view_projection_matrix;

  let vtile_info = vsm_world_to_virtual_tile(
    world_pos,
    camera_vp,
    clipmap0_vp,
    vsm_settings
  );

  // let word_and_mask = vsm_get_virtual_tile_word_and_mask(
  //   vtile_info.tile_coords,
  //   vtile_info.clipmap_index,
  //   shadow_idx,
  //   vsm_settings
  // );

  // let word       = word_and_mask.x;
  // let mask       = word_and_mask.y;
  // let bits       = bitmask[word];
  // if ((bits & mask) == 0u) {
  //   out.position    = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  //   out.world_pos = vec4<f32>(0.0);
  //   out.shadow_index = 0u;
  //   out.view_index = 0u;
  //   return out;
  // }

  var clip_pos = vsm_calculate_render_clip_value_from_world_pos(
    world_pos,
    vtile_info.clipmap_index,
    clipmap0_vp,
    vsm_settings
  );

  out.position = clip_pos;
  out.world_pos = world_pos;
  out.shadow_index = shadow_idx;
  out.view_index = view_index;
#else
  out.position = vec4<f32>(0.0);
  out.world_pos = vec4<f32>(0.0);
  out.shadow_index = 0u;
  out.view_index = 0u;
#endif

  return out;
}
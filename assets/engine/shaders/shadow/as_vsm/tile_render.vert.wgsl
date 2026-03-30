// AS-VSM Stage B: Render Shadow Casters into shadow atlas (Vertex)
// Renders geometry into each requested tile viewport.
#include "common.wgsl"
#include "visibility/visibility_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> visible_meshlets: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(4) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(5) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(6) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(7) var<uniform> light_ub: ShadowCasterLight;
@group(1) @binding(8) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(9) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(10) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(11) var page_table: texture_storage_2d_array<r32uint, read>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) @interpolate(flat) shadow_index: u32,
  @location(2) @interpolate(flat) view_index: u32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32,
      @builtin(instance_index) ii: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4<f32>(-2.0, -2.0, 1.0, 1.0);
  out.world_pos = vec3<f32>(0.0);
  out.shadow_index = 0u;
  out.view_index = 0u;

#if SHADOWS_ENABLED
  let clip_index            = light_ub.clip_index;
  let light_idx             = light_ub.light_index;
  let visible_entry         = visible_meshlets[ii];
  let object_instance_index = meshlet_object_index(visible_entry);
  let global_meshlet_index  = meshlet_index(visible_entry);
  if (object_instance_index >= arrayLength(&object_instances) || global_meshlet_index >= arrayLength(&meshlets)) {
    return out;
  }

  let meshlet               = meshlets[global_meshlet_index];
  let triangle_index        = vi / 3u;
  let corner_index          = vi % 3u;
  if (triangle_index >= meshlet.triangle_count) {
    return out;
  }

  let local_triangle_index  = meshlet.triangle_offset + triangle_index * 3u + corner_index;
  let local_vertex_index    = meshlet_triangles[local_triangle_index];
  let global_vertex_index   = meshlet_vertices[meshlet.vertex_offset + local_vertex_index];
  let row_field             = object_instances[object_instance_index].row;

  let entity_lookup_row     = get_entity_row(row_field);
  if (entity_lookup_row >= arrayLength(&entity_index_lookup)) {
    return out;
  }

  let entity_row            = entity_index_lookup[entity_lookup_row];
  if (entity_row == INVALID_IDX || entity_row >= arrayLength(&entity_transforms)) {
    return out;
  }

  let view_index            = light_view_buffer[light_idx];
  let shadow_idx            = light_shadow_idx_buffer[light_idx];

  let model_matrix          = entity_transforms[entity_row].transform;
  let world_pos             = model_matrix * vertex_position4(vertex_buffer[global_vertex_index]);

  let clipmap0_vp           = view_buffer[view_index].view_projection_matrix;

  let vtile_info = vsm_world_to_virtual_tile_for_clip(
    world_pos,
    clipmap0_vp,
    vsm_settings,
    clip_index,
  );

  let clip_pos = vsm_calculate_render_clip_value_from_world_pos(
    world_pos,
    vtile_info.clipmap_index,
    clipmap0_vp,
    vsm_settings
  );

  out.position = clip_pos;
  out.world_pos = world_pos.xyz;
  out.shadow_index = shadow_idx;
  out.view_index = view_index;

#endif

  return out;
}
// AS-VSM Stage B: Render Shadow Casters into shadow atlas (Fragment)
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
@group(1) @binding(12) var<storage, read_write> shadow_atlas_depth: array<atomic<u32>>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) @interpolate(flat) shadow_index: u32,
  @location(2) @interpolate(flat) view_index: u32,
};

@fragment
fn fs(input: VertexOutput) {
#if SHADOWS_ENABLED
  let clip_index      = light_ub.clip_index;
  let clipmap0_vp     = view_buffer[input.view_index].view_projection_matrix;

  let vtile_info = vsm_world_to_virtual_tile_for_clip(
    vec4<f32>(input.world_pos, 1.0),
    clipmap0_vp,
    vsm_settings,
    clip_index,
  );
  let ptile_info = vsm_vtile_to_ptile(
    vtile_info,
    vsm_settings,
    input.shadow_index,
    page_table
  );

  if (ptile_info.is_dirty) {
    let depth_bits = pack_depth(input.position.z);
    atomicMax(&shadow_atlas_depth[ptile_info.physical_id], depth_bits);
  }
#endif
}
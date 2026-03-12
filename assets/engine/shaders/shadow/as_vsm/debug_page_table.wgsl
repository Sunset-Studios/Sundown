// Debug view for AS-VSM page table entries
#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(1) var<uniform> vsm_settings: ASVSMSettings;

@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
#if SHADOWS_ENABLED
  let dims = textureDimensions(page_table);
  let uv = in.uv * 0.5 + 0.5;
  let coord = vec2<i32>(
    i32(uv.x * f32(dims.x)),
    i32(uv.y * f32(dims.y))
  );

  let clipmap_index = 8u;

  let e = textureLoad(page_table, coord, clipmap_index).r;
  let resident = (e & pte_residency_mask) != 0u;
  let dirty    = (e & pte_dirty_mask) != 0u;
  let valid    = select(0.0, 1.0, resident && !dirty);

  // Visualise physical id (x & y) combined
  let phys_xy = vsm_pte_get_phys_xy(e);
  let phys_x = f32(phys_xy.x) / f32(vsm_settings.physical_tiles_per_row);
  let phys_y = f32(phys_xy.y) / f32(vsm_settings.physical_tiles_per_row);

  return vec4<f32>(phys_x, phys_y, f32(clipmap_index), valid);
#else
  return vec4<f32>(0.0);
#endif
} 
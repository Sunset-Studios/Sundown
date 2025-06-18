// Debug view for AS-VSM page table entries
#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(1) var<storage, read> settings: ASVSMSettings;

@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
#if SHADOWS_ENABLED
  let dims = textureDimensions(page_table);
  let coord = vec2<i32>(
    i32(in.uv.x * f32(dims.x)),
    i32(in.uv.y * f32(dims.y))
  );
  let e = textureLoad(page_table, coord, 0).r;
  let resident = (e & pte_residency_mask) != 0u;
  let dirty    = (e & pte_dirty_mask) != 0u;
  let valid    = select(0.0, 1.0, resident && !dirty);

  // LOD visualisation placeholder (no array layer info here)
  let lod = 0.0;

  // Visualise physical id (x & y) combined
  let phys_x = f32((e & pte_phys_x_mask) >> pte_phys_x_shift) / 128.0;
  let phys_y = f32((e & pte_phys_y_mask) >> pte_phys_y_shift) / 128.0;

  return vec4<f32>(phys_x, phys_y, lod, valid);
#else
  return vec4<f32>(0.0);
#endif
} 
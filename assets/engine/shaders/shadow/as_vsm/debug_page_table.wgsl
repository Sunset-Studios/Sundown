// Debug view for AS-VSM page table entries
#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2<precision_float>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read>;

@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
  let dims = textureDimensions(page_table);
  let coord = vec2<i32>(
    i32(in.uv.x * f32(dims.x)),
    i32(in.uv.y * f32(dims.y))
  );
  let e = textureLoad(page_table, coord, 0).r;
  let valid = f32((e >> 31u) & 1u);
  let lod   = f32((e >> 27u) & 0xFu) * 0.2;      // 4‐bit LOD
  let phys  = f32(e & 0x07FFFFFFu) / 2048.0;        // 27‐bit PhysID
  return vec4<f32>(phys, lod, 0.0, valid);
} 
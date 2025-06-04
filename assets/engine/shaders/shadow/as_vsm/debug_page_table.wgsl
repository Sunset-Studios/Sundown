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
  let lod = f32((e >> 31u) & 1u);
  let phys = f32(e & 0x7FFFFFFFu) / 2048.0; // normalize by max expected phys count
  return vec4<f32>(phys, lod, 0.0, 1.0);
} 
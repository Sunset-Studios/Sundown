#include "common.wgsl"

@group(1) @binding(0) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8,8,1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = textureDimensions(output_tex).x;
  let height = textureDimensions(output_tex).y;
  if gid.x >= width || gid.y >= height { return; }
  let color = vec4<f32>(0.0,0.0,0.0,1.0);
  // Placeholder ReSTIR logic: just clear to black
  textureStore(output_tex, vec2<i32>(gid.xy), color);
}
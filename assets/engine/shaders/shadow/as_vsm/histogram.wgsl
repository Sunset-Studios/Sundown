// AS-VSM Stage A: Depth Histogram
// Bins camera depth into a histogram for SDS-style split computation.
#include "common.wgsl"

@group(1) @binding(0) var camera_depth : texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> histogram : array<atomic<u32>>;

// Workgroup size: 16×16 threads
@compute @workgroup_size(16, 16)
fn cs(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(camera_depth);
  if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) {
    return;
  }

  // Get normalized view space depth based on depth texture
  let d = textureLoad(camera_depth, vec2<i32>(gid.xy), 0).r;
  let t = normalized_view_depth(d);

  // Compute bin index
  let bin_count = arrayLength(&histogram);
  let bin = u32(t * (f32(bin_count) - 1.0));

  // Accumulate histogram
  atomicAdd(&histogram[bin], 1u);
} 
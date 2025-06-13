// AS-VSM Stage B: Screen-space Feedback
// Categorises each pixel into a virtual tile & marks it in the bitmask.
#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var camera_depth: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> settings: ASVSMSettings;
@group(1) @binding(2) var<storage, read_write> bitmask: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
  // Early out if outside screen
  let dims = textureDimensions(camera_depth);
  if (id.x >= u32(dims.x) || id.y >= u32(dims.y)) {
    return;
  }

  // Reconstruct view-space depth from depth buffer
  let d = textureLoad(camera_depth, vec2<i32>(id.xy), 0).r;
  let t = normalized_view_depth(d);

  // Choose LOD based on split depth threshold
  let lod = select(0u, 1u, t > settings.split_depth);
  // Compute virtual tile coordinates at this LOD
  let tile_size = u32(settings.tile_size);
  let size = tile_size << lod;
  let tile_coord = id.xy / size;

  let virtual_tiles_per_row = u32(settings.virtual_tiles_per_row);
  let base_index         = lod * virtual_tiles_per_row * virtual_tiles_per_row;
  let tile_id            = base_index + tile_coord.y * virtual_tiles_per_row + tile_coord.x;
  let word_index         = tile_id >> 5u; // tile_id / 32
  let bit_index          = tile_id & 31u; // tile_id % 32
  let mask               = 1u << bit_index;
  atomicOr(&bitmask[word_index], mask);
} 
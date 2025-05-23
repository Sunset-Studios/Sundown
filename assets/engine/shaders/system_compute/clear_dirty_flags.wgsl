#include "common.wgsl"

@group(1) @binding(0) var<storage, read_write> flags_meta: array<atomic<u32>>;

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let entity_index = global_id.x;

  if (entity_index >= arrayLength(&flags_meta)) {
    return;
  }

  atomicAnd(&flags_meta[entity_index], ~EF_DIRTY);
} 
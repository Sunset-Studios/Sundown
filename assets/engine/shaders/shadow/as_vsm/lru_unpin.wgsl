#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var<storage, read_write> lru : array<atomic<u32>>;

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) id : vec3<u32>) {
  let idx = id.x + 1u;          // slot 0 == head pointer, skip it
  if (idx >= arrayLength(&lru)) {
    return;
  }
  // Clear the pinned bit
  atomicAnd(&lru[idx], ~lru_pinned_flag);
} 
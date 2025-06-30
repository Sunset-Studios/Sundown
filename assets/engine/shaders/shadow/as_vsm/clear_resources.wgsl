#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var<storage, read_write> got_shadow_feedback_buffer: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
#if SHADOWS_ENABLED
  let entity_id = id.x;
  let length = arrayLength(&got_shadow_feedback_buffer);
  if (entity_id >= length) {
    return;
  }
  atomicStore(&got_shadow_feedback_buffer[entity_id], 0u);
#endif
}
#include "common.wgsl"

// ==================================
// Data Structures 
// ==================================
struct Params {
  key_count: u32,
  radix_shift: u32,
  thread_blocks: u32,
  _padding: u32,
};

struct BufU32 {
  data: array<u32>,
};

struct BufA32 {
  data: array<atomic<u32>>,
};

// ==================================
// Bindings
// ==================================

@group(1) @binding(0) var<storage, read_write>  keys_buffer        : BufU32;
@group(1) @binding(1) var<storage, read_write>  error_count        : BufA32;   // atomic single counter
@group(1) @binding(2) var<uniform> params : Params;

// Validate (simple ascending check)
@compute @workgroup_size(256, 1, 1)
fn validate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let num_keys = params.key_count;
  let inc = 65536u;

  for (var i: u32 = id + 1u; i < num_keys; i = i + inc) {
    if (keys_buffer.data[i - 1u] > keys_buffer.data[i]) {
      atomicAdd(&error_count.data[0u], 1u);
    }
  }
}

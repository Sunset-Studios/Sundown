// AS-VSM Stage A.5: Prefix-sum to compute split depth
// Reads histogram and writes split_depth into settings buffer
#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var<storage, read> histogram: array<u32>;
@group(1) @binding(1) var<storage, read_write> settings: ASVSMSettings;

const BIN_COUNT: u32 = 64u;
const SPLIT_THRESHOLD: f32 = 0.85;

@compute @workgroup_size(1)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
  // Sum all histogram bins
  var total = 0u;
  for (var i = 0u; i < BIN_COUNT; i = i + 1u) {
    total = total + histogram[i];
  }

  // Compute target count
  let target_count = u32(f32(total) * SPLIT_THRESHOLD);
  // Find bin where cumulative count >= target
  var cummulative = 0u;
  var split_bin = BIN_COUNT - 1u;
  for (var i = 0u; i < BIN_COUNT; i = i + 1u) {
    cummulative = cummulative + histogram[i];
    if (cummulative >= target_count) {
      split_bin = i;
      break;
    }
  }

  // Compute normalized split depth in [0,1]
  let depth = f32(split_bin) / f32(BIN_COUNT - 1u);
  // Write split_depth
  settings.split_depth = depth;
} 
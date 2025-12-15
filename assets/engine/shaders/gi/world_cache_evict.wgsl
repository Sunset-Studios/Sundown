// =============================================================================
// GI-1.0 Screen Probe Update
// - Accumulates radiance from traced rays back to screen probes
// - Updates world cache with secondary bounce radiance
// - Performs temporal filtering with exponential moving average
// - Only updates probes marked active this frame
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

@group(1) @binding(0) var<storage, read_write> world_cache: array<WorldCacheCell>;

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    var life = world_cache[gid.x].position_frame.w;
    if (life > 0.0) {
        life = life - 1.0;
        world_cache[gid.x].position_frame.w = life;
        if (life <= 0.0) {
            world_cache[gid.x].position_frame = vec4<f32>(0.0);
            world_cache[gid.x].normal_rank = vec4<f32>(0.0);
            world_cache[gid.x].radiance_m = vec4<f32>(0.0);
            atomicStore(&world_cache[gid.x].fingerprint, WORLD_CACHE_CELL_EMPTY);
        }
    }
}
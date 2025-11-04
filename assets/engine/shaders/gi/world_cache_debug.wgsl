// =============================================================================
// GI-1.0 World Cache Debug Visualization
// - Visualizes the spatial hash-based radiance cache
// - Queries cache at each pixel's world position
// - Shows cached radiance intensity and coverage
// =============================================================================
#include "common.wgsl"
#include "gi/gi_common.wgsl"
#include "gi/world_cache_common.wgsl"

@group(1) @binding(0) var<uniform> gi_params: GIParams;
@group(1) @binding(1) var<storage, read_write> world_cache: array<WorldCacheCell>;
@group(1) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(4) var scene_color: texture_2d<f32>;
@group(1) @binding(5) var output_debug: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_debug);
    
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    
    // Read G-buffer
    let position = textureLoad(gbuffer_position, pixel_coord, 0).xyz;
    let normal_data = textureLoad(gbuffer_normal, pixel_coord, 0);
    let normal = safe_normalize(normal_data.xyz);
    let normal_length = length(normal_data.xyz);
    let scene = textureLoad(scene_color, pixel_coord, 0).rgb;
    
    // If no geometry, show scene color
    if (normal_length <= 0.001) {
        textureStore(output_debug, pixel_coord, vec4<f32>(scene, 1.0));
        return;
    }
    
    // Query world cache for entire grid cell (uniform color per cell)
    let cached_radiance = query_world_cache_cell(
        position,
        u32(gi_params.world_cache_size),
        gi_params.world_cache_cell_size
    );
    
    textureStore(output_debug, pixel_coord, vec4<f32>(cached_radiance, 1.0));
}




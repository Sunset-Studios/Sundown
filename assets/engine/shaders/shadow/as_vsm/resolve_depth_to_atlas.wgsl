#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var dummy_depth_image: texture_2d<f32>;
@group(1) @binding(1) var<storage, read_write> shadow_atlas_depth: array<u32>;
@group(1) @binding(2) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(3) var<uniform> light_ub: ShadowCasterLight;
@group(1) @binding(4) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(5) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(6) var<storage, read> dirty_tiles: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read> resolve_dispatch_args: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn cs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
#if SHADOWS_ENABLED
    let source_dims = textureDimensions(dummy_depth_image);
    let virtual_dim = u32(vsm_settings.virtual_dim);
    if (source_dims.x != virtual_dim || source_dims.y != virtual_dim) {
        return;
    }

    let light_idx = light_ub.light_index;
    if (light_idx >= arrayLength(&light_shadow_idx_buffer) || light_idx >= arrayLength(&light_view_buffer)) {
        return;
    }

    let shadow_idx = light_shadow_idx_buffer[light_idx];
    let max_lods = u32(vsm_settings.max_lods);
    let slice_index = shadow_idx * max_lods + light_ub.clip_index;
    let tiles_per_row = u32(vsm_settings.virtual_tiles_per_row);
    let tiles_per_slice = tiles_per_row * tiles_per_row;
    let tile_entry = dirty_tiles[slice_index * tiles_per_slice + workgroup_id.x];
    let tile_coords = vec2<u32>(
        tile_entry.x % tiles_per_row,
        tile_entry.x / tiles_per_row
    );

    let view_index = light_view_buffer[light_idx];
    let clipmap0_vp = view_buffer[view_index].view_projection_matrix;
    let snapped_translation = vsm_snapped_translation_for_lod(
        clipmap0_vp,
        light_ub.clip_index,
        vsm_settings
    );

    // The dummy target and virtual map have matching dimensions. The resolve
    // mapping is therefore a wrapped integer translation (and a Y flip).
    // Compute the virtual pixel reached by source pixel (0, 0), then invert it.
    let source_origin_uv = vec2<f32>(0.5, 0.5) / vec2<f32>(source_dims);
    let source_origin_clip = vec4<f32>(
        source_origin_uv.x * 2.0 - 1.0,
        (1.0 - source_origin_uv.y) * 2.0 - 1.0,
        0.0,
        1.0
    );
    let virtual_origin_uv = fract(
        (source_origin_clip - snapped_translation).xy * 0.5 + 0.5
    );
    let virtual_origin = vec2<u32>(virtual_origin_uv * f32(virtual_dim));

    let tile_size = u32(vsm_settings.tile_size);
    let tile_pixel_count = tile_size * tile_size;
    let physical_dim = u32(vsm_settings.physical_dim);
    let physical_xy = vsm_pte_get_phys_xy(tile_entry.y);
    let pool_index = vsm_pte_get_memory_pool_index(tile_entry.y);
    let pool_offset = pool_index * physical_dim * physical_dim;

    for (
        var tile_pixel_index = local_index;
        tile_pixel_index < tile_pixel_count;
        tile_pixel_index = tile_pixel_index + 256u
    ) {
        let local_pixel = vec2<u32>(
            tile_pixel_index % tile_size,
            tile_pixel_index / tile_size
        );
        let virtual_pixel = tile_coords * tile_size + local_pixel;
        let source_pixel = vec2<u32>(
            (virtual_pixel.x + virtual_dim - virtual_origin.x) % virtual_dim,
            (virtual_origin.y + virtual_dim - virtual_pixel.y) % virtual_dim
        );
        let depth_value = textureLoad(dummy_depth_image, vec2<i32>(source_pixel), 0).x;

        let physical_pixel = physical_xy * tile_size + local_pixel;
        let physical_id =
            pool_offset + physical_pixel.y * physical_dim + physical_pixel.x;
        shadow_atlas_depth[physical_id] = pack_depth(depth_value);
    }
#endif
}

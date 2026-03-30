#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(1) var dummy_depth_image: texture_2d<f32>;
@group(1) @binding(2) var<storage, read_write> shadow_atlas_depth: array<atomic<u32>>;
@group(1) @binding(3) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(4) var<uniform> light_ub: ShadowCasterLight;
@group(1) @binding(5) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(6) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(7) var<storage, read> dirty_slices: array<u32>;

@compute @workgroup_size(16, 16, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
#if SHADOWS_ENABLED
    let source_dims = textureDimensions(dummy_depth_image);
    if (gid.x >= source_dims.x || gid.y >= source_dims.y) {
        return;
    }

    let light_idx = light_ub.light_index;
    if (light_idx >= arrayLength(&light_shadow_idx_buffer) || light_idx >= arrayLength(&light_view_buffer)) {
        return;
    }

    let slice_index = light_idx * u32(vsm_settings.max_lods) + light_ub.clip_index;
    if (slice_index >= arrayLength(&dirty_slices) || dirty_slices[slice_index] != u32(frame_info.frame_index)) {
        return;
    }

    let shadow_idx = light_shadow_idx_buffer[light_idx];
    let view_index = light_view_buffer[light_idx];
    let clipmap0_vp = view_buffer[view_index].view_projection_matrix;
    let depth_value = textureLoad(dummy_depth_image, vec2<i32>(gid.xy), 0).x;
    let render_uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5, 0.5)) / vec2<f32>(source_dims);
    let render_clip = vec4<f32>(
        render_uv.x * 2.0 - 1.0,
        (1.0 - render_uv.y) * 2.0 - 1.0,
        depth_value,
        1.0
    );
    let sample_clip = render_clip - vsm_snapped_translation_for_lod(
        clipmap0_vp,
        light_ub.clip_index,
        vsm_settings
    );
    let virtual_uv = fract(sample_clip.xy * 0.5 + 0.5);
    let virtual_pixel = vec2<u32>(virtual_uv * vsm_settings.virtual_dim);

    var vtile_info: VirtualTileInfo;
    vtile_info.clipmap_index = light_ub.clip_index;
    vtile_info.local_pixel = virtual_pixel % u32(vsm_settings.tile_size);
    vtile_info.tile_coords = virtual_pixel / u32(vsm_settings.tile_size);

    let vtr = u32(vsm_settings.virtual_tiles_per_row);
    vtile_info.tile_id =
        vtile_info.clipmap_index * vtr * vtr
        + vtile_info.tile_coords.y * vtr
        + vtile_info.tile_coords.x;

    let ptile_info = vsm_vtile_to_ptile(vtile_info, vsm_settings, shadow_idx, page_table);
    if (ptile_info.is_dirty) {
        atomicMax(&shadow_atlas_depth[ptile_info.physical_id], pack_depth(depth_value));
    }
#endif
}

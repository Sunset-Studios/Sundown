#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(1) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(2) var<storage, read_write> dirty_slices: array<u32>;

// 8x8 tiles per workgroup; z-dimension is light index
@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
#if SHADOWS_ENABLED
	let vtr = u32(vsm_settings.virtual_tiles_per_row);
	let tx = gid.x;
	let ty = gid.y;
	let light_idx = gid.z;

	if (tx >= vtr || ty >= vtr) { return; }

	let max_lods = u32(vsm_settings.max_lods);
	for (var lod = 0u; lod < max_lods; lod = lod + 1u) {
		let slice = light_idx * max_lods + lod;
		let pte = textureLoad(page_table, vec2<u32>(tx, ty), slice).r;
		if (vsm_pte_is_dirty(pte)) {
			dirty_slices[slice] = u32(frame_info.frame_index);
		}
	}
#endif
}
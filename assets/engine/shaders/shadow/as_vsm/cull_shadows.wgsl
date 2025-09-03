#include "common.wgsl"
#include "shadow/shadows_common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct DrawCullData {
    draw_count: u32,
    view_index: u32,
    clipmap_index: u32,
}

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> visible_object_instances_no_occlusion: array<i32>;
@group(1) @binding(1) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(2) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(3) var<uniform> draw_cull_data: DrawCullData;
@group(1) @binding(4) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(5) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(6) var page_table: texture_storage_2d_array<r32uint, read_write>;
@group(1) @binding(7) var<storage, read_write> dirty_slices: array<u32>;

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
#if SHADOWS_ENABLED
    let g_id = global_id.x;
    if (g_id >= u32(draw_cull_data.draw_count)) {
        return;
    }
    
    // Early-out for invalid clip-level
    let clipmap_index = draw_cull_data.clipmap_index;
    if (clipmap_index >= u32(vsm_settings.max_lods)) {
        return;
    }

    // Resolve the visible instance
    let object_instance_index = visible_object_instances_no_occlusion[g_id];
    if (object_instance_index == -1) {
        return;
    }

    let light_shadow_idx = global_id.y;
    let object_instance  = object_instances[object_instance_index];

    // Setup some view variables
    let view = view_buffer[draw_cull_data.view_index];
    let vp_matrix = view.view_projection_matrix;
    let slice_idx = light_shadow_idx * u32(vsm_settings.max_lods) + clipmap_index;
    let vtr_i = i32(vsm_settings.virtual_tiles_per_row);

    let slice_frame = dirty_slices[slice_idx];
    if (slice_frame != u32(frame_info.frame_index)) {
        // Slice not dirtied at all this frame, so we can skip
        return;
    }

    // ────────────────────────────────────────────────────────────────
    // Append visible instance to the indirect draw buffer
    // ────────────────────────────────────────────────────────────────
    let batch_index    = object_instance.batch;
    let first_instance = draw_indirect_buffer[batch_index].first_instance;
    let count_index    = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
    let instance_index = first_instance + count_index;
    visible_object_instances[instance_index] = object_instance_index;
#endif
}
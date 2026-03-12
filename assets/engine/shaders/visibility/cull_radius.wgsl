#include "common.wgsl"
#include "acceleration_common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct DrawCullData {
    draw_count: u32,
    view_index: u32,
    clipmap_index: u32,
}

// Per-frame parameters required by the radial culler.
// Currently only stores the clip-map-0 extent in world space.
struct RadiusCullParams {
    clip0_extent: f32,
}

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var<storage, read> aabb_bounds: array<AABB>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(3) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(4) var<uniform> draw_cull_data: DrawCullData;
@group(1) @binding(5) var<uniform> radius_cull_params: RadiusCullParams;
@group(1) @binding(6) var<storage, read> entity_index_lookup: array<u32>;


// ------------------------------------------------------------------------------------
// Occlusion Helper Functions
// ------------------------------------------------------------------------------------ 

fn is_in_toroidal_ring(center: vec4<f32>, radius: f32, view: ptr<function, View>) -> u32 {
    // ------------------------------------------------------------------
    // Compute toroidal (ring-based) visibility for the given clip-map.
    // Clip-map 0 is treated as a simple sphere (inner ring radius == 0).
    // Each subsequent clip-map covers the annulus between radii:
    //   inner_radius  = clip0_extent * 2^(n-1)
    //   outer_radius  = clip0_extent * 2^n
    // ------------------------------------------------------------------

    // World-space distance from the object center to the camera position.
    let to_center      = center.xyz - (*view).view_position.xyz;
    let dist_sq        = dot(to_center, to_center);

    // Compute extents for the current clip-map level.
    let clip_level     = f32(draw_cull_data.clipmap_index);
    let clip0_extent   = radius_cull_params.clip0_extent;

    // outer = clip0_extent * 2^level
    let outer_extent   = clip0_extent * f32(1u << clip_level);
    // inner = outer / 2 (equivalent to clip0_extent * 2^(level-1))
    let inner_extent   = outer_extent * 0.5;

    // Include the object's own radius when comparing against bounds.
    let outer_limit_sq = (outer_extent + radius) * (outer_extent + radius);
    let inner_limit_sq = max((inner_extent - radius), 0.0) * max((inner_extent - radius), 0.0);

    // Determine whether the object lies within the annulus.
    let inside_outer   = dist_sq <= outer_limit_sq;
    let outside_inner  = dist_sq > inner_limit_sq;

    let ring_visible   = inside_outer && outside_inner;

    // For clip-map 0 we ignore the inner bound entirely.
    let sphere_visible = inside_outer;

    let visible_bool   = select(ring_visible, sphere_visible, draw_cull_data.clipmap_index == 0u);
    let visible        = u32(visible_bool);

    // Respect per-view culling enable flag.
    return visible * u32((*view).culling_enabled) + u32(1u - u32((*view).culling_enabled));
}

// ------------------------------------------------------------------------------------
// Compute Shader
// ------------------------------------------------------------------------------------ 

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let g_id = global_id.x;
    if (g_id >= u32(draw_cull_data.draw_count)) {
        return;
    }

    let row = object_instances[g_id].row;
    let entity_resolved = entity_index_lookup[get_entity_row(row)];

    let aabb_node = aabb_bounds[entity_resolved];
    let center = vec4<f32>((aabb_node.min.xyz + aabb_node.max.xyz) * 0.5, 1.0);
    var radius = length(aabb_node.max.xyz - aabb_node.min.xyz) * 0.5;
    radius *= 1.2; // Inflate bounds conservatively

    var view = view_buffer[draw_cull_data.view_index];
    let in_frustum = is_in_toroidal_ring(center, radius, &view);

    if (in_frustum == 0u) {
        return;
    }

    let batch_index = object_instances[g_id].batch;
    let first_instance = draw_indirect_buffer[batch_index].first_instance;
    let count_index = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
    let instance_index = first_instance + count_index;
    visible_object_instances[instance_index] = i32(g_id);
}

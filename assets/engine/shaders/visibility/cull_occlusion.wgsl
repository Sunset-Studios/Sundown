#include "common.wgsl"

// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct DrawCullData {
    draw_count: u32,
    hzb_width: u32,
    hzb_height: u32,
}

// ------------------------------------------------------------------------------------
// Buffers
// ------------------------------------------------------------------------------------ 

@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> aabb_bounds: array<AABBNodeBounds>;
@group(1) @binding(2) var<storage, read> visible_object_instances_no_occlusion: array<i32>;
@group(1) @binding(3) var<storage, read_write> visible_object_instances: array<i32>;
@group(1) @binding(4) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(5) var<storage, read> entity_aabb_node_indices: array<u32>;
@group(1) @binding(6) var<uniform> draw_cull_data: DrawCullData;
@group(1) @binding(7) var<storage, read_write> draw_indirect_buffer: array<DrawCommand>;
@group(1) @binding(8) var entity_id_texture: texture_2d<u32>;
@group(1) @binding(9) var<storage, read> occluder_buffer: array<u32>;


// ------------------------------------------------------------------------------------
// Occlusion Helper Functions
// ------------------------------------------------------------------------------------ 

fn aabb_project(
    min_pt: vec3<f32>, max_pt: vec3<f32>,
    out_uv_rect: ptr<function, vec4<f32>>,
    view: ptr<function, View>
) -> bool {
    // Build the 8 world‐space corners
    let corners = array<vec4<f32>, 8>(
        vec4f(min_pt.x, min_pt.y, min_pt.z, 1.0),
        vec4f(max_pt.x, min_pt.y, min_pt.z, 1.0),
        vec4f(min_pt.x, max_pt.y, min_pt.z, 1.0),
        vec4f(max_pt.x, max_pt.y, min_pt.z, 1.0),
        vec4f(min_pt.x, min_pt.y, max_pt.z, 1.0),
        vec4f(max_pt.x, min_pt.y, max_pt.z, 1.0),
        vec4f(min_pt.x, max_pt.y, max_pt.z, 1.0),
        vec4f(max_pt.x, max_pt.y, max_pt.z, 1.0)
    );

    // Initialize NDC min/max
    var ndc_min = vec2<f32>( 1.0,  1.0);
    var ndc_max = vec2<f32>(-1.0, -1.0);

    // Project each corner into NDC
    for (var i = 0u; i < 8u; i = i + 1u) {
        let cv = view.view_matrix * corners[i];
        // skip behind-camera points
        if (cv.z >= 0.0) {
            continue;
        }
        let clip = view.projection_matrix * cv;
        let ndc = clip.xy / clip.w;
        ndc_min = min(ndc_min, ndc);
        ndc_max = max(ndc_max, ndc);
    }

    // If completely off‐screen, no occlusion test
    if (ndc_max.x < -1.0 || ndc_min.x > 1.0 ||
        ndc_max.y < -1.0 || ndc_min.y > 1.0) {
        return false;
    }

    // Convert NDC box to [0..1] UV (0,0=top‐left)
    let u_min = clamp(ndc_min.x * 0.5 + 0.5, 0.0, 1.0);
    let u_max = clamp(ndc_max.x * 0.5 + 0.5, 0.0, 1.0);
    let v_min = clamp(-ndc_max.y * 0.5 + 0.5, 0.0, 1.0);
    let v_max = clamp(-ndc_min.y * 0.5 + 0.5, 0.0, 1.0);

    *out_uv_rect = vec4<f32>(u_min, v_min, u_max, v_max);
    return true;
}

fn is_occluded(aabb_node: ptr<function, AABBNodeBounds>, view: ptr<function, View>) -> u32 {
    if (view.occlusion_enabled == 0.0) {
        return 0u;
    }

    var uv_rect: vec4<f32>;
    if (!aabb_project(aabb_node.min_point.xyz, aabb_node.max_point.xyz, &uv_rect, view)) {
        // if the AABB is completely off-screen, skip occlusion
        return 0u;
    }

    // guard degenerate or inverted rects
    if (uv_rect.z <= uv_rect.x || uv_rect.w <= uv_rect.y) {
        return 0u;
    }

    let center = vec4f((aabb_node.min_point.xyz + aabb_node.max_point.xyz) * 0.5, 1.0);
    var radius = length(aabb_node.max_point.xyz - aabb_node.min_point.xyz) * 0.5;

    let width  = (uv_rect.z - uv_rect.x) * f32(draw_cull_data.hzb_width);
    let height = (uv_rect.w - uv_rect.y) * f32(draw_cull_data.hzb_height);
    let non_negative_size = max(width, height);

    let level_floor = floor(log2(non_negative_size));
    let max_level   = f32(textureNumLevels(input_texture) - 1u);
    let level       = clamp(level_floor, 0.0, max_level);

    // compute sphere depth in view-space and bias
    let center_view = view.view_matrix * center;
    let sphere_depth = -center_view.z - radius;
    let depth_bias: f32 = 1.0;
    let far_depth: f32 = 1.0;
    let entity_dims = vec2<f32>(textureDimensions(entity_id_texture));

    // subdivide the screen-space rect
    let u_min = uv_rect.x;
    let v_min = uv_rect.y;
    let u_max = uv_rect.z;
    let v_max = uv_rect.w;
    
    // sample a 3×3 grid (corners, edges, center)
    const sample_dim: u32 = 3u;
    var max_depth: f32 = 0.0;
    for (var ix: u32 = 0u; ix < sample_dim; ix = ix + 1u) {
        for (var iy: u32 = 0u; iy < sample_dim; iy = iy + 1u) {
            let uv = vec2<f32>(
                mix(u_min, u_max, f32(ix) / f32(sample_dim - 1u)),
                mix(v_min, v_max, f32(iy) / f32(sample_dim - 1u))
            );
            let raw_d: f32 = textureSampleLevel(input_texture, non_filtering_sampler, uv, level).r;
            let pixel = vec2<u32>(uv * entity_dims);
            let id = textureLoad(entity_id_texture, vec2<i32>(pixel), 0).x;
            let occ = occluder_buffer[id] != 0u;
            let d = select(far_depth, raw_d, occ);
            let lin_d = linearize_depth(d, view.near, view.far);
            max_depth = max(max_depth, lin_d);
        }
    }

    let visible = u32(sphere_depth < max_depth + depth_bias);

    // fully occluded
    return u32(view.culling_enabled) * (1u - visible);
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

    let object_instance_index = visible_object_instances_no_occlusion[g_id];
    if (object_instance_index == -1) {
        return;
    }

    let object_instance = object_instances[object_instance_index];
    let entity_resolved = get_entity_row(object_instance.row);

    let aabb_node_index = entity_aabb_node_indices[entity_resolved];
    var aabb_node = aabb_bounds[aabb_node_index];

    var view = view_buffer[frame_info.view_index];
    let occluded = is_occluded(&aabb_node, &view);

    if (occluded > 0u) {
        return;
    }

    let batch_index = object_instance.batch;
    let first_instance = draw_indirect_buffer[batch_index].first_instance;
    let count_index = atomicAdd(&draw_indirect_buffer[batch_index].instance_count, 1u);
    let instance_index = first_instance + count_index;
    visible_object_instances[instance_index] = object_instance_index;
}

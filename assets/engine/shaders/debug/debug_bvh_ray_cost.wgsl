// =============================================================================
// BVH Primary-Ray Cost Heatmap
// =============================================================================
// Traces one camera ray per output pixel and counts the intersection tests that
// traversal actually executes. The shared traversal instrumentation is compiled
// only for this shader variant, keeping production ray paths counter-free.
// =============================================================================

#define BVH_TRAVERSAL_COLLECT_STATS

#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "acceleration/bvh_ray_traversal.wgsl"

@group(1) @binding(0) var<storage, read> tlas_bvh2_bounds: array<AABB>;
@group(1) @binding(1) var<uniform> tlas_bvh_info: BVHInfo;
@group(1) @binding(2) var<storage, read> blas_bvh2_nodes: array<AABB>;
@group(1) @binding(3) var<storage, read> blas_directory: array<MeshDirectoryEntry>;
@group(1) @binding(4) var<storage, read> compact_transforms: array<RayInstanceTransform>;
@group(1) @binding(5) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(6) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(7) var output_texture: texture_storage_2d<rgba16float, write>;

fn bvh_cost_heatmap(value: f32) -> vec3<f32> {
    let color_0 = vec3<f32>(0.015, 0.020, 0.120);
    let color_1 = vec3<f32>(0.000, 0.450, 1.000);
    let color_2 = vec3<f32>(0.000, 0.900, 0.500);
    let color_3 = vec3<f32>(1.000, 0.900, 0.050);
    let color_4 = vec3<f32>(0.950, 0.050, 0.020);
    let scaled_value = clamp(value, 0.0, 1.0) * 4.0;
    let segment = min(u32(scaled_value), 3u);
    let segment_t = smoothstep(0.0, 1.0, clamp(scaled_value - f32(segment), 0.0, 1.0));

    if (segment == 0u) {
        return mix(color_0, color_1, segment_t);
    }
    if (segment == 1u) {
        return mix(color_1, color_2, segment_t);
    }
    if (segment == 2u) {
        return mix(color_2, color_3, segment_t);
    }
    return mix(color_3, color_4, segment_t);
}

@compute @workgroup_size(8, 8, 1)
fn cs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let dimensions = textureDimensions(output_texture);
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }

    bvh_stack_lane = local_idx;
    bvh_reset_traversal_stats();

    let view = view_buffer[u32(frame_info.view_index)];
    let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5);
    let uv = pixel_center / vec2<f32>(dimensions);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let tan_half_fov = tan(view.fov * 0.5);
    let sensor_x = ndc.x * view.aspect_ratio * tan_half_fov;
    let sensor_y = ndc.y * tan_half_fov;
    let forward = normalize(view.view_direction.xyz);
    let right = normalize(view.view_right.xyz);
    let up = normalize(cross(right, forward));
    let direction = normalize(forward + right * sensor_x + up * sensor_y);

    var ray: Ray;
    ray.origin_and_tmin = vec4<f32>(view.view_position.xyz + direction * 0.001, 0.0001);
    ray.direction_and_tmax = vec4<f32>(direction, 1e30);
    ray.inv_direction = vec4<f32>(
        1.0 / max(abs(direction.x), 1e-8) * select(1.0, -1.0, direction.x < 0.0),
        1.0 / max(abs(direction.y), 1e-8) * select(1.0, -1.0, direction.y < 0.0),
        1.0 / max(abs(direction.z), 1e-8) * select(1.0, -1.0, direction.z < 0.0),
        0.0
    );

    _ = trace_ray_closest(&ray);

    let primitive_test_count = bvh_traversal_stats.tlas_aabb_tests
        + bvh_traversal_stats.blas_aabb_tests
        + bvh_traversal_stats.triangle_tests;
    // Log scaling preserves contrast across typical traversal costs while reserving
    // the hottest color for rays that execute roughly 255 or more primitive tests.
    let heat = clamp(log2(f32(primitive_test_count) + 1.0) * 0.125, 0.0, 1.0);
    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(bvh_cost_heatmap(heat), 1.0));
}

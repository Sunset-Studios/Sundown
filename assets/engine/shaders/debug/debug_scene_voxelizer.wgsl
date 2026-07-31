#include "common.wgsl"

// ============================================================================
// Scene Voxelization Debug Trace
// ============================================================================
//
// Traces the packed HDDA hierarchy and shades the first occupied leaf voxel
// along each camera ray. This path runs only for the dedicated debug view and
// deliberately renders at half resolution on the CPU side.
// ============================================================================

@group(1) @binding(0) var<uniform> params: SceneVoxelClipmapParams;
@group(1) @binding(1) var<storage, read> voxel_grid: array<u32>;
@group(1) @binding(2) var<storage, read> voxel_hierarchy: array<u32>;
@group(1) @binding(3) var scene_color: texture_2d<f32>;
@group(1) @binding(4) var debug_output: texture_storage_2d<rgba16float, write>;

#include "acceleration/scene_voxel_hdda.wgsl"

fn trace_voxel_grid(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>
) -> vec4<f32> {
    let hit = scene_voxel_hdda_trace_first_occupied(
        ray_origin,
        ray_direction,
        0.0,
        SCENE_VOXEL_HDDA_MAX_FLOAT
    );
    if (hit.has_hit == 0u) {
        return vec4<f32>(0.0);
    }

    let voxel = vec3<u32>(hit.voxel_coord);
    let clip_params = params.levels[hit.clip_level];
    let linear_index =
        voxel.x + clip_params.resolution * (voxel.y + clip_params.resolution * voxel.z);
    let face_light = 0.35 + 0.65 * abs(dot(hit.entry_normal, -ray_direction));
    let checker = f32((voxel.x ^ voxel.y ^ voxel.z) & 1u) * 0.08;
    let identifier_tint = id_to_color(linear_index) * 0.12;
    let clip_color = vec3<f32>(0.2) + id_to_color(hit.clip_level + 17u) * 0.8;
    let base_color = clip_color + identifier_tint + checker;
    return vec4<f32>(base_color * face_light, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let resolution = textureDimensions(debug_output);
    if (gid.x >= resolution.x || gid.y >= resolution.y) {
        return;
    }

    let pixel = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(resolution);
    let clip_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let view_index = u32(frame_info.view_index);
    let inverse_view_projection = view_buffer[view_index].inverse_view_projection_matrix;
    var world_near = inverse_view_projection * vec4<f32>(clip_xy, -1.0, 1.0);
    var world_far = inverse_view_projection * vec4<f32>(clip_xy, 1.0, 1.0);
    world_near = world_near / world_near.w;
    world_far = world_far / world_far.w;

    let ray_origin = world_near.xyz;
    let ray_direction = normalize(world_far.xyz - world_near.xyz);
    let voxel_hit = trace_voxel_grid(ray_origin, ray_direction);
    let scene = textureSampleLevel(scene_color, global_sampler, uv, 0.0);
    let background = scene.rgb * 0.22;
    let color = select(background, voxel_hit.rgb, voxel_hit.a > 0.0);
    textureStore(debug_output, pixel, vec4<f32>(color, 1.0));
}

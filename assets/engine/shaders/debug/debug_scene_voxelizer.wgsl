#include "common.wgsl"

// ============================================================================
// Scene Voxelization Debug Trace
// ============================================================================
//
// Traces the dense occupancy bitset with a grid DDA and shades the first
// occupied voxel along each camera ray. This path runs only for the dedicated
// debug view and deliberately renders at half resolution on the CPU side.
// ============================================================================

struct SceneVoxelizationParams {
    grid_origin: vec3<f32>,
    voxel_size: f32,
    meshlet_count: u32,
    dispatch_width: u32,
    resolution: u32,
    _padding: u32,
};

@group(1) @binding(0) var<uniform> params: SceneVoxelizationParams;
@group(1) @binding(1) var<storage, read> voxel_grid: array<u32>;
@group(1) @binding(2) var scene_color: texture_2d<f32>;
@group(1) @binding(3) var debug_output: texture_storage_2d<rgba16float, write>;

const DEBUG_MAX_DDA_STEPS: u32 = 768u;
const DEBUG_RAY_EPSILON: f32 = 0.0001;
const DEBUG_DIRECTION_EPSILON: f32 = 0.000001;

fn voxel_is_occupied(coord: vec3<i32>) -> bool {
    if (
        any(coord < vec3<i32>(0)) ||
        any(coord >= vec3<i32>(i32(params.resolution)))
    ) {
        return false;
    }

    let voxel = vec3<u32>(coord);
    let linear_index =
        voxel.x + params.resolution * (voxel.y + params.resolution * voxel.z);
    let word = voxel_grid[linear_index >> 5u];
    return (word & (1u << (linear_index & 31u))) != 0u;
}

fn intersect_volume(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>
) -> vec2<f32> {
    let safe_direction = select(
        vec3<f32>(DEBUG_DIRECTION_EPSILON),
        ray_direction,
        abs(ray_direction) > vec3<f32>(DEBUG_DIRECTION_EPSILON)
    );
    let inverse_direction = vec3<f32>(1.0) / safe_direction;
    let volume_min = params.grid_origin;
    let volume_max =
        params.grid_origin + vec3<f32>(f32(params.resolution) * params.voxel_size);
    let plane0 = (volume_min - ray_origin) * inverse_direction;
    let plane1 = (volume_max - ray_origin) * inverse_direction;
    let near_plane = min(plane0, plane1);
    let far_plane = max(plane0, plane1);

    return vec2<f32>(
        max(near_plane.x, max(near_plane.y, near_plane.z)),
        min(far_plane.x, min(far_plane.y, far_plane.z))
    );
}

fn trace_voxel_grid(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>
) -> vec4<f32> {
    let volume_hit = intersect_volume(ray_origin, ray_direction);
    var current_t = max(volume_hit.x, 0.0) + DEBUG_RAY_EPSILON;
    if (volume_hit.y < current_t) {
        return vec4<f32>(0.0);
    }

    let volume_position =
        (ray_origin + ray_direction * current_t - params.grid_origin) /
        params.voxel_size;
    let grid_max = f32(params.resolution) - DEBUG_RAY_EPSILON;
    var voxel_coord = vec3<i32>(
        floor(clamp(volume_position, vec3<f32>(0.0), vec3<f32>(grid_max)))
    );
    let step = select(
        vec3<i32>(-1),
        vec3<i32>(1),
        ray_direction >= vec3<f32>(0.0)
    );

    let has_direction = abs(ray_direction) > vec3<f32>(DEBUG_DIRECTION_EPSILON);
    let safe_direction = select(
        vec3<f32>(DEBUG_DIRECTION_EPSILON),
        ray_direction,
        has_direction
    );
    let inverse_direction = vec3<f32>(1.0) / safe_direction;
    let next_coord = vec3<f32>(voxel_coord) + select(
        vec3<f32>(0.0),
        vec3<f32>(1.0),
        step > vec3<i32>(0)
    );
    let next_boundary = params.grid_origin + next_coord * params.voxel_size;
    var next_t = select(
        vec3<f32>(3.402823466e+38),
        (next_boundary - ray_origin) * inverse_direction,
        has_direction
    );
    let delta_t = select(
        vec3<f32>(3.402823466e+38),
        abs(vec3<f32>(params.voxel_size) * inverse_direction),
        has_direction
    );
    var hit_normal = -ray_direction;

    for (var iteration = 0u; iteration < DEBUG_MAX_DDA_STEPS; iteration = iteration + 1u) {
        if (voxel_is_occupied(voxel_coord)) {
            let voxel = vec3<u32>(voxel_coord);
            let linear_index =
                voxel.x + params.resolution * (voxel.y + params.resolution * voxel.z);
            let face_light = 0.35 + 0.65 * abs(dot(hit_normal, -ray_direction));
            let checker = f32((voxel.x ^ voxel.y ^ voxel.z) & 1u) * 0.08;
            let identifier_tint = id_to_color(linear_index) * 0.12;
            let base_color = vec3<f32>(0.08, 0.72, 1.0) + identifier_tint + checker;
            return vec4<f32>(base_color * face_light, 1.0);
        }

        var axis = 0u;
        if (next_t.y < next_t.x) {
            axis = 1u;
        }
        if (next_t.z < next_t[axis]) {
            axis = 2u;
        }

        current_t = next_t[axis];
        if (current_t > volume_hit.y) {
            break;
        }

        voxel_coord[axis] = voxel_coord[axis] + step[axis];
        next_t[axis] = next_t[axis] + delta_t[axis];
        hit_normal = vec3<f32>(0.0);
        hit_normal[axis] = -f32(step[axis]);
    }

    return vec4<f32>(0.0);
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

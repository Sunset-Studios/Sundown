#include "acceleration/scene_voxel_hierarchy_common.wgsl"

// ============================================================================
// Scene Voxel Clipmap HDDA
// ============================================================================
//
// Including shaders provide these bindings:
//
//   params                  SceneVoxelClipmapParams uniform
//   voxel_grid              packed 256^3 leaf occupancy per active clip
//   voxel_hierarchy         packed 32^3, 4^3, and 1^3 hierarchy per active clip
//
// Traversal selects the finest clip volume containing the current ray point.
// A clip segment ends when the ray exits that volume or enters a finer one,
// which makes overlapping clip volumes behave as nested shells. Coarse HDDA
// levels skip empty space within a clip; successful traces always report a
// leaf voxel and the exact interval that can bound a subsequent BVH trace.
// ============================================================================

struct SceneVoxelHDDAHit {
    voxel_coord: vec3<i32>,
    has_hit: u32,
    entry_normal: vec3<f32>,
    steps: u32,
    t_enter: f32,
    t_exit: f32,
    clip_level: u32,
    _padding: u32,
};

const SCENE_VOXEL_HDDA_MAX_STEPS: u32 = 4096u;
const SCENE_VOXEL_HDDA_MAX_CLIP_SEGMENTS: u32 = 16u;
const SCENE_VOXEL_HDDA_DIRECTION_EPSILON: f32 = 0.000001;
const SCENE_VOXEL_HDDA_MAX_FLOAT: f32 = 3.402823466e+38;

fn scene_voxel_hdda_miss(t_max: f32, steps: u32) -> SceneVoxelHDDAHit {
    var result: SceneVoxelHDDAHit;
    result.voxel_coord = vec3<i32>(-1);
    result.has_hit = 0u;
    result.entry_normal = vec3<f32>(0.0);
    result.steps = steps;
    result.t_enter = t_max;
    result.t_exit = t_max;
    result.clip_level = params.clip_level_count;
    result._padding = 0u;
    return result;
}

fn scene_voxel_hdda_occupancy(
    clip_level: u32,
    hdda_level: u32,
    coord: vec3<i32>
) -> bool {
    let level_resolution = scene_voxel_hdda_level_resolution(hdda_level);
    if (
        any(coord < vec3<i32>(0)) ||
        any(coord >= vec3<i32>(i32(level_resolution)))
    ) {
        return false;
    }

    let linear_index = scene_voxel_hdda_linear_index(
        vec3<u32>(coord),
        level_resolution
    );
    if (hdda_level == 0u) {
        let word_index =
            scene_voxel_clipmap_leaf_word_offset(clip_level) +
            (linear_index >> 5u);
        let word = voxel_grid[word_index];
        return (word & (1u << (linear_index & 31u))) != 0u;
    }

    let word_index =
        scene_voxel_clipmap_hierarchy_word_offset(clip_level) +
        scene_voxel_hdda_level_word_offset(hdda_level) +
        (linear_index >> 5u);
    let word = voxel_hierarchy[word_index];
    return (word & (1u << (linear_index & 31u))) != 0u;
}

fn scene_voxel_hdda_volume_interval(
    voxelization_params: SceneVoxelizationParams,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>
) -> vec2<f32> {
    let volume_min = voxelization_params.grid_origin;
    let volume_max = volume_min + vec3<f32>(
        f32(voxelization_params.resolution) * voxelization_params.voxel_size
    );
    var t_near = -SCENE_VOXEL_HDDA_MAX_FLOAT;
    var t_far = SCENE_VOXEL_HDDA_MAX_FLOAT;

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let direction = ray_direction[axis];
        if (abs(direction) <= SCENE_VOXEL_HDDA_DIRECTION_EPSILON) {
            if (ray_origin[axis] < volume_min[axis] || ray_origin[axis] >= volume_max[axis]) {
                return vec2<f32>(1.0, -1.0);
            }
            continue;
        }

        let inverse_direction = 1.0 / direction;
        let plane0 = (volume_min[axis] - ray_origin[axis]) * inverse_direction;
        let plane1 = (volume_max[axis] - ray_origin[axis]) * inverse_direction;
        t_near = max(t_near, min(plane0, plane1));
        t_far = min(t_far, max(plane0, plane1));
    }

    return vec2<f32>(t_near, t_far);
}

fn scene_voxel_hdda_point_inside(
    voxelization_params: SceneVoxelizationParams,
    world_position: vec3<f32>
) -> bool {
    let volume_min = voxelization_params.grid_origin;
    let volume_max = volume_min + vec3<f32>(
        f32(voxelization_params.resolution) * voxelization_params.voxel_size
    );
    return all(world_position >= volume_min) && all(world_position < volume_max);
}

fn scene_voxel_hdda_coord_at_t(
    voxelization_params: SceneVoxelizationParams,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    sample_t: f32,
    hdda_level: u32
) -> vec3<i32> {
    let grid_position = clamp(
        (ray_origin + ray_direction * sample_t - voxelization_params.grid_origin) /
            voxelization_params.voxel_size,
        vec3<f32>(0.0),
        vec3<f32>(f32(voxelization_params.resolution) - 0.0001)
    );
    return vec3<i32>(floor(
        grid_position / f32(scene_voxel_hdda_level_span(hdda_level))
    ));
}

fn scene_voxel_hdda_cell_exit(
    voxelization_params: SceneVoxelizationParams,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    coord: vec3<i32>,
    hdda_level: u32
) -> vec4<f32> {
    let cell_span = f32(scene_voxel_hdda_level_span(hdda_level));
    let cell_min =
        voxelization_params.grid_origin +
        vec3<f32>(coord) * cell_span * voxelization_params.voxel_size;
    let cell_max = cell_min + vec3<f32>(cell_span * voxelization_params.voxel_size);
    let positive_direction = ray_direction >= vec3<f32>(0.0);
    let exit_boundary = select(cell_min, cell_max, positive_direction);
    let active_axis = abs(ray_direction) > vec3<f32>(SCENE_VOXEL_HDDA_DIRECTION_EPSILON);
    let safe_direction = select(vec3<f32>(1.0), ray_direction, active_axis);
    let axis_t = select(
        vec3<f32>(SCENE_VOXEL_HDDA_MAX_FLOAT),
        (exit_boundary - ray_origin) / safe_direction,
        active_axis
    );

    var exit_axis = 0u;
    if (axis_t.y < axis_t.x) {
        exit_axis = 1u;
    }
    if (axis_t.z < axis_t[exit_axis]) {
        exit_axis = 2u;
    }

    var exit_normal = vec3<f32>(0.0);
    exit_normal[exit_axis] = select(1.0, -1.0, positive_direction[exit_axis]);
    return vec4<f32>(exit_normal, axis_t[exit_axis]);
}

fn scene_voxel_hdda_trace_clip_level(
    clip_level: u32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    ray_t_min: f32,
    ray_t_max: f32
) -> SceneVoxelHDDAHit {
    let voxelization_params = params.levels[clip_level];
    let traversal_epsilon = max(voxelization_params.voxel_size * 0.0001, 0.000001);
    var current_t = ray_t_min;
    var hdda_level = SCENE_VOXEL_HDDA_MAX_LEVEL;
    var entry_normal = -ray_direction;

    for (var iteration = 0u; iteration < SCENE_VOXEL_HDDA_MAX_STEPS; iteration = iteration + 1u) {
        if (current_t >= ray_t_max) {
            return scene_voxel_hdda_miss(ray_t_max, iteration);
        }

        let sample_t = min(current_t + traversal_epsilon, ray_t_max);
        let coord = scene_voxel_hdda_coord_at_t(
            voxelization_params,
            ray_origin,
            ray_direction,
            sample_t,
            hdda_level
        );

        if (scene_voxel_hdda_occupancy(clip_level, hdda_level, coord)) {
            if (hdda_level != 0u) {
                hdda_level = hdda_level - 1u;
                continue;
            }

            let cell_exit = scene_voxel_hdda_cell_exit(
                voxelization_params,
                ray_origin,
                ray_direction,
                coord,
                hdda_level
            );
            var result: SceneVoxelHDDAHit;
            result.voxel_coord = coord;
            result.has_hit = 1u;
            result.entry_normal = entry_normal;
            result.steps = iteration + 1u;
            result.t_enter = current_t;
            result.t_exit = min(cell_exit.w, ray_t_max);
            result.clip_level = clip_level;
            result._padding = 0u;
            return result;
        }

        let cell_exit = scene_voxel_hdda_cell_exit(
            voxelization_params,
            ray_origin,
            ray_direction,
            coord,
            hdda_level
        );
        entry_normal = cell_exit.xyz;
        current_t = max(cell_exit.w, current_t + traversal_epsilon);
        hdda_level = min(hdda_level + 1u, SCENE_VOXEL_HDDA_MAX_LEVEL);
    }

    return scene_voxel_hdda_miss(ray_t_max, SCENE_VOXEL_HDDA_MAX_STEPS);
}

fn scene_voxel_hdda_trace_first_occupied(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    ray_t_min: f32,
    ray_t_max: f32
) -> SceneVoxelHDDAHit {
    let outer_clip_level = params.clip_level_count - 1u;
    let outer_interval = scene_voxel_hdda_volume_interval(
        params.levels[outer_clip_level],
        ray_origin,
        ray_direction
    );
    let trace_begin = max(ray_t_min, outer_interval.x);
    let trace_end = min(ray_t_max, outer_interval.y);
    if (trace_begin > trace_end) {
        return scene_voxel_hdda_miss(ray_t_max, 0u);
    }

    let boundary_epsilon = max(params.levels[0].voxel_size * 0.0001, 0.000001);
    var current_t = trace_begin;
    var total_steps = 0u;

    for (
        var segment_index = 0u;
        segment_index < SCENE_VOXEL_HDDA_MAX_CLIP_SEGMENTS;
        segment_index = segment_index + 1u
    ) {
        if (current_t >= trace_end) {
            break;
        }

        let sample_t = min(current_t + boundary_epsilon, trace_end);
        let sample_position = ray_origin + ray_direction * sample_t;
        var active_clip_level = outer_clip_level;
        for (var clip_level = 0u; clip_level < params.clip_level_count; clip_level++) {
            if (scene_voxel_hdda_point_inside(params.levels[clip_level], sample_position)) {
                active_clip_level = clip_level;
                break;
            }
        }

        let active_interval = scene_voxel_hdda_volume_interval(
            params.levels[active_clip_level],
            ray_origin,
            ray_direction
        );
        var segment_end = min(trace_end, active_interval.y);

        // If the ray approaches the camera after starting in a coarse shell,
        // hand traversal to the first finer clip as soon as its bounds begin.
        for (var finer_level = 0u; finer_level < active_clip_level; finer_level++) {
            let finer_interval = scene_voxel_hdda_volume_interval(
                params.levels[finer_level],
                ray_origin,
                ray_direction
            );
            if (
                finer_interval.x > current_t + boundary_epsilon &&
                finer_interval.x < segment_end &&
                finer_interval.x <= finer_interval.y
            ) {
                segment_end = finer_interval.x;
            }
        }

        let segment_hit = scene_voxel_hdda_trace_clip_level(
            active_clip_level,
            ray_origin,
            ray_direction,
            current_t,
            segment_end
        );
        total_steps = total_steps + segment_hit.steps;
        if (segment_hit.has_hit != 0u) {
            var result = segment_hit;
            result.steps = total_steps;
            return result;
        }

        current_t = max(segment_end + boundary_epsilon, current_t + boundary_epsilon);
    }

    return scene_voxel_hdda_miss(ray_t_max, total_steps);
}

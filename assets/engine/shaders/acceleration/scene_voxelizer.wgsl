#include "common.wgsl"
#include "visibility/visibility_common.wgsl"

// ============================================================================
// Meshlet-Driven Scene Voxelization
// ============================================================================
//
// One workgroup owns one meshlet instance. Lane zero resolves the instance's
// current transform and rejects meshlets outside the volume; the remaining
// lanes transform the meshlet's unique vertices once into workgroup memory and
// batch triangle footprints into a shared prefix-summed cell stream. All lanes
// cooperatively consume that stream, so large projected triangles cannot
// serialize one lane. Each triangle is conservatively rasterized on its
// dominant plane and expanded through the triangle plane to cover intersected
// depth voxels. Atomic bit claims make overlap between triangles, meshlets, and
// objects safe without serialization.
//
// The input is a GPU-compacted list containing only meshlets overlapping dirty
// bricks. Dirty bricks are cleared first, then all intersecting geometry is
// restored so moving objects cannot erase overlapping static geometry.
// ============================================================================

struct SceneVoxelizationParams {
    grid_origin: vec3<f32>,
    voxel_size: f32,
    meshlet_count: u32,
    dispatch_width: u32,
    resolution: u32,
    flags: u32,
};

struct RasterTriangle {
    axis_interval0: vec4<f32>,
    axis_interval1: vec4<f32>,
    axis_interval2: vec4<f32>,
    plane: vec4<f32>,
    bounds: vec4<u32>,
    metadata: vec4<u32>,
};

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> meshlet_instances: array<MeshletInstance>;
@group(1) @binding(3) var<storage, read> entity_index_lookup: array<u32>;
@group(1) @binding(4) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(5) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(6) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(7) var<uniform> params: SceneVoxelizationParams;
@group(1) @binding(8) var<storage, read_write> voxel_grid: array<atomic<u32>>;
@group(1) @binding(9) var<storage, read_write> voxel_dispatch_count: atomic<u32>;

var<workgroup> active_meshlet: MeshletRecord;
var<workgroup> active_transform: mat4x4<f32>;
var<workgroup> active_meshlet_valid: u32;
var<workgroup> active_grid_vertices: array<vec4<f32>, 64>;
var<workgroup> active_raster_triangles: array<RasterTriangle, 64>;
var<workgroup> active_raster_prefix: array<u32, 64>;

const VOXELIZER_WORKGROUP_SIZE: u32 = 64u;
const VOXELIZER_MAX_MESHLET_VERTICES: u32 = 64u;
const VOXELIZER_RASTER_EPSILON: f32 = 0.00001;
const VOXELIZER_NORMAL_EPSILON_SQUARED: f32 = 0.0000000001;

fn projected_axis_separates_triangle_and_cell(
    axis: vec2<f32>,
    triangle_min: f32,
    triangle_max: f32,
    cell_center: vec2<f32>
) -> bool {
    if (dot(axis, axis) <= VOXELIZER_NORMAL_EPSILON_SQUARED) {
        return false;
    }

    let cell_projection = dot(cell_center, axis);
    let cell_radius = 0.5 * (abs(axis.x) + abs(axis.y));

    return triangle_min > cell_projection + cell_radius + VOXELIZER_RASTER_EPSILON ||
        triangle_max < cell_projection - cell_radius - VOXELIZER_RASTER_EPSILON;
}

fn dominant_axis(normal: vec3<f32>) -> u32 {
    let absolute_normal = abs(normal);
    if (absolute_normal.x >= absolute_normal.y && absolute_normal.x >= absolute_normal.z) {
        return 0u;
    }
    if (absolute_normal.y >= absolute_normal.z) {
        return 1u;
    }
    return 2u;
}

fn project_to_dominant_plane(value: vec3<f32>, axis: u32) -> vec2<f32> {
    switch axis {
        case 0u: {
            return value.yz;
        }
        case 1u: {
            return value.xz;
        }
        default: {
            return value.xy;
        }
    }
}

fn dominant_component(value: vec3<f32>, axis: u32) -> f32 {
    switch axis {
        case 0u: {
            return value.x;
        }
        case 1u: {
            return value.y;
        }
        default: {
            return value.z;
        }
    }
}

fn compose_voxel_coord(projected_coord: vec2<i32>, depth_coord: i32, axis: u32) -> vec3<i32> {
    switch axis {
        case 0u: {
            return vec3<i32>(depth_coord, projected_coord.x, projected_coord.y);
        }
        case 1u: {
            return vec3<i32>(projected_coord.x, depth_coord, projected_coord.y);
        }
        default: {
            return vec3<i32>(projected_coord, depth_coord);
        }
    }
}

fn meshlet_intersects_volume(meshlet: MeshletRecord, transform: mat4x4<f32>) -> bool {
    // Transforming center/extents is exact for affine AABBs and avoids eight matrix multiplies for
    // every meshlet that the volume rejects before any triangle work.
    let local_center = (meshlet.bounds_min.xyz + meshlet.bounds_max.xyz) * 0.5;
    let local_extent = (meshlet.bounds_max.xyz - meshlet.bounds_min.xyz) * 0.5;
    let world_center = (transform * vec4<f32>(local_center, 1.0)).xyz;
    let world_extent =
        abs(transform[0].xyz) * local_extent.x +
        abs(transform[1].xyz) * local_extent.y +
        abs(transform[2].xyz) * local_extent.z;
    let world_min = world_center - world_extent;
    let world_max = world_center + world_extent;

    let volume_min = params.grid_origin;
    let volume_max =
        params.grid_origin + vec3<f32>(f32(params.resolution) * params.voxel_size);
    return all(world_max >= volume_min) && all(world_min <= volume_max);
}

fn mark_voxel(voxel_coord: vec3<i32>) {
    let coord = vec3<u32>(voxel_coord);
    let linear_index =
        coord.x + params.resolution * (coord.y + params.resolution * coord.z);
    let word_index = linear_index >> 5u;
    let bit_mask = 1u << (linear_index & 31u);
    atomicOr(&voxel_grid[word_index], bit_mask);
}

fn empty_raster_triangle() -> RasterTriangle {
    var triangle: RasterTriangle;
    triangle.axis_interval0 = vec4<f32>(0.0);
    triangle.axis_interval1 = vec4<f32>(0.0);
    triangle.axis_interval2 = vec4<f32>(0.0);
    triangle.plane = vec4<f32>(0.0);
    triangle.bounds = vec4<u32>(0u);
    triangle.metadata = vec4<u32>(0u);
    return triangle;
}

fn prepare_raster_triangle(
    grid_triangle0: vec3<f32>,
    grid_triangle1: vec3<f32>,
    grid_triangle2: vec3<f32>
) -> RasterTriangle {
    var raster_triangle = empty_raster_triangle();
    let triangle_min = min(grid_triangle0, min(grid_triangle1, grid_triangle2));
    let triangle_max = max(grid_triangle0, max(grid_triangle1, grid_triangle2));
    let resolution_f = f32(params.resolution);

    if (any(triangle_max < vec3<f32>(0.0)) || any(triangle_min > vec3<f32>(resolution_f))) {
        return raster_triangle;
    }

    let edge0 = grid_triangle1 - grid_triangle0;
    let edge1 = grid_triangle2 - grid_triangle0;
    let triangle_normal = cross(edge0, edge1);
    if (dot(triangle_normal, triangle_normal) <= VOXELIZER_NORMAL_EPSILON_SQUARED) {
        return raster_triangle;
    }

    let projection_axis = dominant_axis(triangle_normal);
    let projected0 = project_to_dominant_plane(grid_triangle0, projection_axis);
    let projected1 = project_to_dominant_plane(grid_triangle1, projection_axis);
    let projected2 = project_to_dominant_plane(grid_triangle2, projection_axis);
    let projected_min = min(projected0, min(projected1, projected2));
    let projected_max = max(projected0, max(projected1, projected2));

    let max_coord = vec3<i32>(i32(params.resolution) - 1);
    let max_coord_f = f32(max_coord.x);
    let projected_coord_min = vec2<i32>(
        floor(
            clamp(
                projected_min - vec2<f32>(VOXELIZER_RASTER_EPSILON),
                vec2<f32>(0.0),
                vec2<f32>(max_coord_f)
            )
        )
    );
    let projected_coord_max = vec2<i32>(
        floor(
            clamp(
                projected_max + vec2<f32>(VOXELIZER_RASTER_EPSILON),
                vec2<f32>(0.0),
                vec2<f32>(max_coord_f)
            )
        )
    );

    let projected_edge0 = projected1 - projected0;
    let projected_edge1 = projected2 - projected1;
    let projected_edge2 = projected0 - projected2;
    let projected_axis0 = vec2<f32>(-projected_edge0.y, projected_edge0.x);
    let projected_axis1 = vec2<f32>(-projected_edge1.y, projected_edge1.x);
    let projected_axis2 = vec2<f32>(-projected_edge2.y, projected_edge2.x);

    let projected_interval0 = vec2<f32>(
        min(dot(projected0, projected_axis0), dot(projected2, projected_axis0)),
        max(dot(projected0, projected_axis0), dot(projected2, projected_axis0))
    );
    let projected_interval1 = vec2<f32>(
        min(dot(projected1, projected_axis1), dot(projected0, projected_axis1)),
        max(dot(projected1, projected_axis1), dot(projected0, projected_axis1))
    );
    let projected_interval2 = vec2<f32>(
        min(dot(projected2, projected_axis2), dot(projected1, projected_axis2)),
        max(dot(projected2, projected_axis2), dot(projected1, projected_axis2))
    );

    let projected_normal = project_to_dominant_plane(triangle_normal, projection_axis);
    let depth_normal = dominant_component(triangle_normal, projection_axis);
    let plane_distance = dot(triangle_normal, grid_triangle0);
    let raster_width = u32(projected_coord_max.x - projected_coord_min.x + 1);
    let raster_height = u32(projected_coord_max.y - projected_coord_min.y + 1);

    raster_triangle.axis_interval0 =
        vec4<f32>(projected_axis0, projected_interval0);
    raster_triangle.axis_interval1 =
        vec4<f32>(projected_axis1, projected_interval1);
    raster_triangle.axis_interval2 =
        vec4<f32>(projected_axis2, projected_interval2);
    raster_triangle.plane =
        vec4<f32>(projected_normal, depth_normal, plane_distance);
    raster_triangle.bounds = vec4<u32>(
        vec2<u32>(projected_coord_min),
        raster_width,
        raster_width * raster_height
    );
    raster_triangle.metadata = vec4<u32>(projection_axis, 0u, 0u, 0u);
    return raster_triangle;
}

fn rasterize_triangle_cell(triangle: RasterTriangle, cell_index: u32) {
    let raster_width = triangle.bounds.z;
    let projected_coord = vec2<i32>(
        i32(triangle.bounds.x + cell_index % raster_width),
        i32(triangle.bounds.y + cell_index / raster_width)
    );
    let cell_center = vec2<f32>(projected_coord) + vec2<f32>(0.5);

    if (
        projected_axis_separates_triangle_and_cell(
            triangle.axis_interval0.xy,
            triangle.axis_interval0.z,
            triangle.axis_interval0.w,
            cell_center
        ) ||
        projected_axis_separates_triangle_and_cell(
            triangle.axis_interval1.xy,
            triangle.axis_interval1.z,
            triangle.axis_interval1.w,
            cell_center
        ) ||
        projected_axis_separates_triangle_and_cell(
            triangle.axis_interval2.xy,
            triangle.axis_interval2.z,
            triangle.axis_interval2.w,
            cell_center
        )
    ) {
        return;
    }

    let projected_normal = triangle.plane.xy;
    let depth_normal = triangle.plane.z;
    let depth_center =
        (triangle.plane.w - dot(projected_normal, cell_center)) / depth_normal;
    let depth_radius =
        0.5 * (abs(projected_normal.x) + abs(projected_normal.y)) / abs(depth_normal);
    let unclamped_depth_min =
        floor(depth_center - depth_radius - VOXELIZER_RASTER_EPSILON);
    let unclamped_depth_max =
        floor(depth_center + depth_radius + VOXELIZER_RASTER_EPSILON);
    let max_coord_f = f32(params.resolution - 1u);
    if (unclamped_depth_max < 0.0 || unclamped_depth_min > max_coord_f) {
        return;
    }

    let depth_min = i32(clamp(unclamped_depth_min, 0.0, max_coord_f));
    let depth_max = i32(clamp(unclamped_depth_max, 0.0, max_coord_f));
    for (
        var depth_coord = depth_min;
        depth_coord <= depth_max;
        depth_coord = depth_coord + 1
    ) {
        mark_voxel(
            compose_voxel_coord(projected_coord, depth_coord, triangle.metadata.x)
        );
    }
}

@compute @workgroup_size(64)
fn cs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane_index: u32
) {
    let dispatch_width = min(atomicLoad(&voxel_dispatch_count), 65535u);
    let meshlet_instance_index =
        workgroup_id.x + workgroup_id.y * dispatch_width;

    if (lane_index == 0u) {
        active_meshlet_valid = 0u;

        if (
            meshlet_instance_index < atomicLoad(&voxel_dispatch_count) &&
            meshlet_instance_index < arrayLength(&meshlet_instances)
        ) {
            let meshlet_instance = meshlet_instances[meshlet_instance_index];
            let object_instance_index = meshlet_instance.object_instance_index;
            let meshlet_index_value = meshlet_instance.meshlet_index;

            if (
                object_instance_index < arrayLength(&object_instances) &&
                meshlet_index_value < arrayLength(&meshlets)
            ) {
                let entity_row = get_entity_row(object_instances[object_instance_index].row);
                if (entity_row < arrayLength(&entity_index_lookup)) {
                    let entity_index = entity_index_lookup[entity_row];
                    if (
                        entity_index != INVALID_IDX &&
                        entity_index < arrayLength(&entity_transforms)
                    ) {
                        let meshlet = meshlets[meshlet_index_value];
                        let transform = entity_transforms[entity_index].transform;
                        if (
                            meshlet.vertex_count <= VOXELIZER_MAX_MESHLET_VERTICES &&
                            meshlet_intersects_volume(meshlet, transform)
                        ) {
                            active_meshlet = meshlet;
                            active_transform = transform;
                            active_meshlet_valid = 1u;
                        }
                    }
                }
            }
        }
    }

    let meshlet_is_valid = workgroupUniformLoad(&active_meshlet_valid) != 0u;
    if (!meshlet_is_valid) {
        return;
    }

    active_grid_vertices[lane_index] = vec4<f32>(0.0);
    if (lane_index < active_meshlet.vertex_count) {
        let vertex_offset = active_meshlet.vertex_offset;
        let meshlet_vertex_stream_length = arrayLength(&meshlet_vertices);
        if (
            vertex_offset <= meshlet_vertex_stream_length &&
            meshlet_vertex_stream_length - vertex_offset > lane_index
        ) {
            let vertex_index = meshlet_vertices[vertex_offset + lane_index];
            if (vertex_index < arrayLength(&vertex_buffer)) {
                let world_position =
                    (active_transform * vertex_position4(vertex_buffer[vertex_index])).xyz;
                let grid_position =
                    (world_position - params.grid_origin) / params.voxel_size;
                active_grid_vertices[lane_index] = vec4<f32>(grid_position, 1.0);
            }
        }
    }

    let triangle_count = workgroupUniformLoad(&active_meshlet.triangle_count);
    let triangle_stream_length = arrayLength(&meshlet_triangles);

    for (
        var batch_start = 0u;
        batch_start < triangle_count;
        batch_start = batch_start + VOXELIZER_WORKGROUP_SIZE
    ) {
        let triangle_index = batch_start + lane_index;
        var raster_triangle = empty_raster_triangle();

        if (triangle_index < triangle_count) {
            let triangle_offset = active_meshlet.triangle_offset + triangle_index * 3u;
            if (
                triangle_offset <= triangle_stream_length &&
                triangle_stream_length - triangle_offset >= 3u
            ) {
                let local_index0 = meshlet_triangles[triangle_offset + 0u];
                let local_index1 = meshlet_triangles[triangle_offset + 1u];
                let local_index2 = meshlet_triangles[triangle_offset + 2u];
                if (
                    local_index0 < active_meshlet.vertex_count &&
                    local_index1 < active_meshlet.vertex_count &&
                    local_index2 < active_meshlet.vertex_count &&
                    active_grid_vertices[local_index0].w != 0.0 &&
                    active_grid_vertices[local_index1].w != 0.0 &&
                    active_grid_vertices[local_index2].w != 0.0
                ) {
                    raster_triangle = prepare_raster_triangle(
                        active_grid_vertices[local_index0].xyz,
                        active_grid_vertices[local_index1].xyz,
                        active_grid_vertices[local_index2].xyz
                    );
                }
            }
        }

        active_raster_triangles[lane_index] = raster_triangle;
        active_raster_prefix[lane_index] = raster_triangle.bounds.w;
        workgroupBarrier();

        // An inclusive workgroup scan turns the 64 per-triangle rectangles into one contiguous
        // cell stream. All lanes then share the aggregate work instead of waiting for the lane
        // that happened to receive the largest projected triangle.
        for (
            var scan_offset = 1u;
            scan_offset < VOXELIZER_WORKGROUP_SIZE;
            scan_offset = scan_offset << 1u
        ) {
            var prefix_add = 0u;
            if (lane_index >= scan_offset) {
                prefix_add = active_raster_prefix[lane_index - scan_offset];
            }
            workgroupBarrier();
            active_raster_prefix[lane_index] =
                active_raster_prefix[lane_index] + prefix_add;
            workgroupBarrier();
        }

        let batch_cell_count =
            active_raster_prefix[VOXELIZER_WORKGROUP_SIZE - 1u];
        for (
            var work_index = lane_index;
            work_index < batch_cell_count;
            work_index = work_index + VOXELIZER_WORKGROUP_SIZE
        ) {
            var lower_bound = 0u;
            var upper_bound = VOXELIZER_WORKGROUP_SIZE;
            for (
                var search_step = 0u;
                search_step < 7u;
                search_step = search_step + 1u
            ) {
                if (lower_bound < upper_bound) {
                    let midpoint = (lower_bound + upper_bound) >> 1u;
                    if (active_raster_prefix[midpoint] <= work_index) {
                        lower_bound = midpoint + 1u;
                    } else {
                        upper_bound = midpoint;
                    }
                }
            }

            let triangle_slot = lower_bound;
            var triangle_cell_offset = 0u;
            if (triangle_slot > 0u) {
                triangle_cell_offset = active_raster_prefix[triangle_slot - 1u];
            }
            rasterize_triangle_cell(
                active_raster_triangles[triangle_slot],
                work_index - triangle_cell_offset
            );
        }

        workgroupBarrier();
    }
}

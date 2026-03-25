import { vec3 } from "gl-matrix";
import { MeshoptClusterizer } from "meshoptimizer/clusterizer";

export function transform_position(position, world_matrix) {
  if (!world_matrix) {
    return [position[0], position[1], position[2]];
  }

  const transformed = vec3.transformMat4(
    vec3.create(),
    vec3.fromValues(position[0], position[1], position[2]),
    world_matrix
  );
  return [transformed[0], transformed[1], transformed[2]];
}

export function transform_direction(direction, normal_matrix) {
  if (!normal_matrix) {
    return [direction[0], direction[1], direction[2]];
  }

  const transformed = vec3.transformMat3(
    vec3.create(),
    vec3.fromValues(direction[0], direction[1], direction[2]),
    normal_matrix
  );
  if (vec3.length(transformed) > 1e-6) {
    vec3.normalize(transformed, transformed);
  }
  return [transformed[0], transformed[1], transformed[2]];
}

export function transform_bounds(bounds_min, bounds_max, world_matrix) {
  if (!world_matrix) {
    return {
      bounds_min: [...bounds_min],
      bounds_max: [...bounds_max],
    };
  }

  const transformed_min = [Infinity, Infinity, Infinity];
  const transformed_max = [-Infinity, -Infinity, -Infinity];

  for (let corner = 0; corner < 8; corner++) {
    const local_corner = [
      (corner & 1) !== 0 ? bounds_max[0] : bounds_min[0],
      (corner & 2) !== 0 ? bounds_max[1] : bounds_min[1],
      (corner & 4) !== 0 ? bounds_max[2] : bounds_min[2],
    ];
    const transformed_corner = transform_position(local_corner, world_matrix);
    transformed_min[0] = Math.min(transformed_min[0], transformed_corner[0]);
    transformed_min[1] = Math.min(transformed_min[1], transformed_corner[1]);
    transformed_min[2] = Math.min(transformed_min[2], transformed_corner[2]);
    transformed_max[0] = Math.max(transformed_max[0], transformed_corner[0]);
    transformed_max[1] = Math.max(transformed_max[1], transformed_corner[1]);
    transformed_max[2] = Math.max(transformed_max[2], transformed_corner[2]);
  }

  return {
    bounds_min: transformed_min,
    bounds_max: transformed_max,
  };
}

export function matrix_max_scale(world_matrix) {
  if (!world_matrix) {
    return 1.0;
  }

  return Math.max(
    Math.hypot(world_matrix[0], world_matrix[1], world_matrix[2]),
    Math.hypot(world_matrix[4], world_matrix[5], world_matrix[6]),
    Math.hypot(world_matrix[8], world_matrix[9], world_matrix[10])
  );
}

export function transform_meshlet_record(record, descriptor) {
  const world_matrix = descriptor.world_matrix ?? null;
  const normal_matrix = descriptor.normal_matrix ?? null;
  const transformed_bounds = transform_bounds(record.bounds_min, record.bounds_max, world_matrix);

  return {
    ...record,
    center: transform_position(record.center, world_matrix),
    radius: record.radius * matrix_max_scale(world_matrix),
    bounds_min: transformed_bounds.bounds_min,
    bounds_max: transformed_bounds.bounds_max,
    normal_cone_axis: transform_direction(record.normal_cone_axis, normal_matrix),
  };
}

export function transform_meshlet_group_record(record, descriptor) {
  const world_matrix = descriptor.world_matrix ?? null;
  const transformed_bounds = transform_bounds(record.bounds_min, record.bounds_max, world_matrix);

  return {
    ...record,
    center: transform_position(record.center, world_matrix),
    radius: record.radius * matrix_max_scale(world_matrix),
    bounds_min: transformed_bounds.bounds_min,
    bounds_max: transformed_bounds.bounds_max,
  };
}

export function build_empty_meshlet_sections(section_count) {
  return Array.from({ length: section_count }, () => ({
    meshlet_offset: 0,
    meshlet_count: 0,
    meshlet_group_offset: 0,
    meshlet_group_count: 0,
  }));
}

export function compute_position_bounds(positions, indices = null) {
  const bounds_min = [Infinity, Infinity, Infinity];
  const bounds_max = [-Infinity, -Infinity, -Infinity];

  if (indices && indices.length > 0) {
    for (let i = 0; i < indices.length; i++) {
      const base = (indices[i] ?? 0) * 3;
      bounds_min[0] = Math.min(bounds_min[0], positions[base + 0] ?? 0.0);
      bounds_min[1] = Math.min(bounds_min[1], positions[base + 1] ?? 0.0);
      bounds_min[2] = Math.min(bounds_min[2], positions[base + 2] ?? 0.0);
      bounds_max[0] = Math.max(bounds_max[0], positions[base + 0] ?? 0.0);
      bounds_max[1] = Math.max(bounds_max[1], positions[base + 1] ?? 0.0);
      bounds_max[2] = Math.max(bounds_max[2], positions[base + 2] ?? 0.0);
    }
  } else {
    for (let i = 0; i < positions.length; i += 3) {
      bounds_min[0] = Math.min(bounds_min[0], positions[i + 0] ?? 0.0);
      bounds_min[1] = Math.min(bounds_min[1], positions[i + 1] ?? 0.0);
      bounds_min[2] = Math.min(bounds_min[2], positions[i + 2] ?? 0.0);
      bounds_max[0] = Math.max(bounds_max[0], positions[i + 0] ?? 0.0);
      bounds_max[1] = Math.max(bounds_max[1], positions[i + 1] ?? 0.0);
      bounds_max[2] = Math.max(bounds_max[2], positions[i + 2] ?? 0.0);
    }
  }

  if (!Number.isFinite(bounds_min[0])) {
    return {
      min: [0, 0, 0],
      max: [0, 0, 0],
    };
  }

  return {
    min: bounds_min,
    max: bounds_max,
  };
}

export function get_position(positions, vertex_index) {
  const base = vertex_index * 3;
  return [positions[base + 0] ?? 0.0, positions[base + 1] ?? 0.0, positions[base + 2] ?? 0.0];
}

export function add_vec3(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub_vec3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale_vec3(v, scalar) {
  return [v[0] * scalar, v[1] * scalar, v[2] * scalar];
}

export function dot_vec3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function length_vec3(v) {
  return Math.sqrt(dot_vec3(v, v));
}

export function compute_sphere_from_bounds(bounds_min, bounds_max, vertices) {
  const center = scale_vec3(add_vec3(bounds_min, bounds_max), 0.5);

  let radius = 0.0;
  for (let i = 0; i < vertices.length; i++) {
    const delta = sub_vec3(vertices[i], center);
    radius = Math.max(radius, Math.sqrt(dot_vec3(delta, delta)));
  }

  return { center, radius };
}

export function expand_bits_10(value) {
  let x = value & 0x3ff;
  x = (x | (x << 16)) & 0x30000ff;
  x = (x | (x << 8)) & 0x300f00f;
  x = (x | (x << 4)) & 0x30c30c3;
  x = (x | (x << 2)) & 0x9249249;
  return x;
}

export function morton3d_10bit(x, y, z) {
  return expand_bits_10(x) | (expand_bits_10(y) << 1) | (expand_bits_10(z) << 2);
}

export function sort_meshlets_spatially(meshlets, primitive_bounds) {
  const extent = [
    primitive_bounds.max[0] - primitive_bounds.min[0],
    primitive_bounds.max[1] - primitive_bounds.min[1],
    primitive_bounds.max[2] - primitive_bounds.min[2],
  ];

  return meshlets
    .map((meshlet, original_index) => {
      const normalized_center = [0, 0, 0];
      for (let i = 0; i < 3; i++) {
        if (extent[i] <= 1e-8) {
          normalized_center[i] = 0.5;
        } else {
          normalized_center[i] =
            (meshlet.center[i] - primitive_bounds.min[i]) / extent[i];
        }
      }

      return {
        meshlet,
        original_index,
        morton: morton3d_10bit(
          Math.max(0, Math.min(1023, Math.floor(normalized_center[0] * 1023.0))),
          Math.max(0, Math.min(1023, Math.floor(normalized_center[1] * 1023.0))),
          Math.max(0, Math.min(1023, Math.floor(normalized_center[2] * 1023.0)))
        ),
      };
    })
    .sort((a, b) => {
      if (a.morton === b.morton) {
        return a.original_index - b.original_index;
      }
      return a.morton - b.morton;
    })
    .map((entry) => entry.meshlet);
}

export function compute_meshlet_aabb(global_vertices, positions) {
  const bounds_min = [Infinity, Infinity, Infinity];
  const bounds_max = [-Infinity, -Infinity, -Infinity];

  for (let i = 0; i < global_vertices.length; i++) {
    const vertex = get_position(positions, global_vertices[i]);
    bounds_min[0] = Math.min(bounds_min[0], vertex[0]);
    bounds_min[1] = Math.min(bounds_min[1], vertex[1]);
    bounds_min[2] = Math.min(bounds_min[2], vertex[2]);
    bounds_max[0] = Math.max(bounds_max[0], vertex[0]);
    bounds_max[1] = Math.max(bounds_max[1], vertex[1]);
    bounds_max[2] = Math.max(bounds_max[2], vertex[2]);
  }

  if (!Number.isFinite(bounds_min[0])) {
    return {
      min: [0, 0, 0],
      max: [0, 0, 0],
    };
  }

  return {
    min: bounds_min,
    max: bounds_max,
  };
}

export function build_meshlets(indices, positions, settings) {
  const triangle_count = Math.floor(indices.length / 3);
  const min_triangles = Math.max(1, Math.min(settings.min_triangles, triangle_count));
  const max_triangles = Math.max(min_triangles, settings.max_triangles);
  const buffers = MeshoptClusterizer.buildMeshletsSpatial(
    indices,
    positions,
    3,
    settings.max_vertices,
    min_triangles,
    max_triangles,
    settings.fill_weight
  );

  const computed_bounds = MeshoptClusterizer.computeMeshletBounds(buffers, positions, 3);
  const bounds_array = Array.isArray(computed_bounds) ? computed_bounds : [computed_bounds];
  const meshlets = [];

  for (let i = 0; i < buffers.meshletCount; i++) {
    const meshlet = MeshoptClusterizer.extractMeshlet(buffers, i);
    const global_vertices = Array.from(meshlet.vertices);
    const local_indices = Array.from(meshlet.triangles);
    const bounds = bounds_array[i];
    const aabb = compute_meshlet_aabb(global_vertices, positions);

    meshlets.push({
      global_vertices,
      local_indices,
      bounds_min: aabb.min,
      bounds_max: aabb.max,
      center: [bounds.centerX, bounds.centerY, bounds.centerZ],
      radius: bounds.radius,
      normal_cone_axis: [bounds.coneAxisX, bounds.coneAxisY, bounds.coneAxisZ],
      normal_cone_cutoff: bounds.coneCutoff,
    });
  }

  return meshlets;
}

export function build_meshlet_groups(meshlets, group_size) {
  const groups = [];

  for (let i = 0; i < meshlets.length; i += group_size) {
    const group_meshlets = meshlets.slice(i, i + group_size);
    const bounds_min = [Infinity, Infinity, Infinity];
    const bounds_max = [-Infinity, -Infinity, -Infinity];

    for (let j = 0; j < group_meshlets.length; j++) {
      const meshlet = group_meshlets[j];
      bounds_min[0] = Math.min(bounds_min[0], meshlet.bounds_min[0]);
      bounds_min[1] = Math.min(bounds_min[1], meshlet.bounds_min[1]);
      bounds_min[2] = Math.min(bounds_min[2], meshlet.bounds_min[2]);
      bounds_max[0] = Math.max(bounds_max[0], meshlet.bounds_max[0]);
      bounds_max[1] = Math.max(bounds_max[1], meshlet.bounds_max[1]);
      bounds_max[2] = Math.max(bounds_max[2], meshlet.bounds_max[2]);
    }

    const sphere = compute_sphere_from_bounds(
      bounds_min,
      bounds_max,
      group_meshlets.map((meshlet) => meshlet.center)
    );

    let radius = sphere.radius;
    for (let j = 0; j < group_meshlets.length; j++) {
      const meshlet = group_meshlets[j];
      const delta = sub_vec3(meshlet.center, sphere.center);
      radius = Math.max(radius, length_vec3(delta) + meshlet.radius);
    }

    groups.push({
      local_meshlet_offset: i,
      meshlet_count: group_meshlets.length,
      center: sphere.center,
      radius,
      bounds_min,
      bounds_max,
    });
  }

  return groups;
}

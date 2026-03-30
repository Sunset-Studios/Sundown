export const SBVH_BIN_COUNT = 16;
export const SBVH_MAX_REFERENCE_MULTIPLIER = 2;
export const SBVH_MAX_SPATIAL_DEPTH = 8;
const COST_EPSILON = 1e-5;
const BOUNDS_EPSILON = 1e-6;

function make_empty_bounds() {
  return {
    min_x: Infinity,
    min_y: Infinity,
    min_z: Infinity,
    max_x: -Infinity,
    max_y: -Infinity,
    max_z: -Infinity,
  };
}

function is_valid_bounds(bounds) {
  return (
    bounds.min_x <= bounds.max_x &&
    bounds.min_y <= bounds.max_y &&
    bounds.min_z <= bounds.max_z
  );
}

function clone_ref(ref) {
  return {
    tri_id: ref.tri_id,
    min_x: ref.min_x,
    min_y: ref.min_y,
    min_z: ref.min_z,
    max_x: ref.max_x,
    max_y: ref.max_y,
    max_z: ref.max_z,
  };
}

function ref_centroid(ref, axis) {
  switch (axis) {
    case 0:
      return (ref.min_x + ref.max_x) * 0.5;
    case 1:
      return (ref.min_y + ref.max_y) * 0.5;
    default:
      return (ref.min_z + ref.max_z) * 0.5;
  }
}

function bounds_extent(bounds, axis) {
  switch (axis) {
    case 0:
      return bounds.max_x - bounds.min_x;
    case 1:
      return bounds.max_y - bounds.min_y;
    default:
      return bounds.max_z - bounds.min_z;
  }
}

function surface_area(bounds) {
  const ex = Math.max(0.0, bounds.max_x - bounds.min_x);
  const ey = Math.max(0.0, bounds.max_y - bounds.min_y);
  const ez = Math.max(0.0, bounds.max_z - bounds.min_z);
  return 2.0 * (ex * ey + ex * ez + ey * ez);
}

function expand_bounds(bounds, ref) {
  bounds.min_x = Math.min(bounds.min_x, ref.min_x);
  bounds.min_y = Math.min(bounds.min_y, ref.min_y);
  bounds.min_z = Math.min(bounds.min_z, ref.min_z);
  bounds.max_x = Math.max(bounds.max_x, ref.max_x);
  bounds.max_y = Math.max(bounds.max_y, ref.max_y);
  bounds.max_z = Math.max(bounds.max_z, ref.max_z);
}

function merge_bounds(a, b) {
  if (!is_valid_bounds(a)) return { ...b };
  if (!is_valid_bounds(b)) return { ...a };
  return {
    min_x: Math.min(a.min_x, b.min_x),
    min_y: Math.min(a.min_y, b.min_y),
    min_z: Math.min(a.min_z, b.min_z),
    max_x: Math.max(a.max_x, b.max_x),
    max_y: Math.max(a.max_y, b.max_y),
    max_z: Math.max(a.max_z, b.max_z),
  };
}

function compute_bounds(refs) {
  const bounds = make_empty_bounds();
  for (let i = 0; i < refs.length; i++) {
    expand_bounds(bounds, refs[i]);
  }
  return bounds;
}

function compute_centroid_bounds(refs) {
  const bounds = make_empty_bounds();
  for (let i = 0; i < refs.length; i++) {
    const ref = refs[i];
    const cx = (ref.min_x + ref.max_x) * 0.5;
    const cy = (ref.min_y + ref.max_y) * 0.5;
    const cz = (ref.min_z + ref.max_z) * 0.5;
    bounds.min_x = Math.min(bounds.min_x, cx);
    bounds.min_y = Math.min(bounds.min_y, cy);
    bounds.min_z = Math.min(bounds.min_z, cz);
    bounds.max_x = Math.max(bounds.max_x, cx);
    bounds.max_y = Math.max(bounds.max_y, cy);
    bounds.max_z = Math.max(bounds.max_z, cz);
  }
  return bounds;
}

function clip_ref_to_bounds(ref, bounds) {
  const clipped = {
    tri_id: ref.tri_id,
    min_x: Math.max(ref.min_x, bounds.min_x),
    min_y: Math.max(ref.min_y, bounds.min_y),
    min_z: Math.max(ref.min_z, bounds.min_z),
    max_x: Math.min(ref.max_x, bounds.max_x),
    max_y: Math.min(ref.max_y, bounds.max_y),
    max_z: Math.min(ref.max_z, bounds.max_z),
  };
  return is_valid_bounds(clipped) ? clipped : null;
}

function clip_ref_to_bin(ref, axis, plane_min, plane_max) {
  const clipped = clone_ref(ref);
  if (axis === 0) {
    clipped.min_x = Math.max(clipped.min_x, plane_min);
    clipped.max_x = Math.min(clipped.max_x, plane_max);
  } else if (axis === 1) {
    clipped.min_y = Math.max(clipped.min_y, plane_min);
    clipped.max_y = Math.min(clipped.max_y, plane_max);
  } else {
    clipped.min_z = Math.max(clipped.min_z, plane_min);
    clipped.max_z = Math.min(clipped.max_z, plane_max);
  }
  return is_valid_bounds(clipped) ? clipped : null;
}

function make_bin() {
  return {
    enter: 0,
    exit: 0,
    count: 0,
    bounds: make_empty_bounds(),
  };
}

function bin_index(value, min_value, extent) {
  if (extent <= BOUNDS_EPSILON) {
    return 0;
  }
  const scaled = ((value - min_value) / extent) * SBVH_BIN_COUNT;
  const idx = Math.floor(scaled);
  return Math.max(0, Math.min(SBVH_BIN_COUNT - 1, idx));
}

function find_best_object_split(refs) {
  const centroid_bounds = compute_centroid_bounds(refs);
  let best = null;

  for (let axis = 0; axis < 3; axis++) {
    const extent = bounds_extent(centroid_bounds, axis);
    if (extent <= BOUNDS_EPSILON) {
      continue;
    }

    const axis_min =
      axis === 0 ? centroid_bounds.min_x : axis === 1 ? centroid_bounds.min_y : centroid_bounds.min_z;
    const bins = Array.from({ length: SBVH_BIN_COUNT }, make_bin);

    for (let i = 0; i < refs.length; i++) {
      const ref = refs[i];
      const idx = bin_index(ref_centroid(ref, axis), axis_min, extent);
      bins[idx].count++;
      expand_bounds(bins[idx].bounds, ref);
    }

    const left_counts = new Uint32Array(SBVH_BIN_COUNT);
    const right_counts = new Uint32Array(SBVH_BIN_COUNT);
    const left_bounds = Array.from({ length: SBVH_BIN_COUNT }, make_empty_bounds);
    const right_bounds = Array.from({ length: SBVH_BIN_COUNT }, make_empty_bounds);

    let running_count = 0;
    let running_bounds = make_empty_bounds();
    for (let i = 0; i < SBVH_BIN_COUNT; i++) {
      running_count += bins[i].count;
      running_bounds = merge_bounds(running_bounds, bins[i].bounds);
      left_counts[i] = running_count;
      left_bounds[i] = running_bounds;
    }

    running_count = 0;
    running_bounds = make_empty_bounds();
    for (let i = SBVH_BIN_COUNT - 1; i >= 0; i--) {
      running_count += bins[i].count;
      running_bounds = merge_bounds(running_bounds, bins[i].bounds);
      right_counts[i] = running_count;
      right_bounds[i] = running_bounds;
    }

    for (let i = 0; i < SBVH_BIN_COUNT - 1; i++) {
      const left_count = left_counts[i];
      const right_count = right_counts[i + 1];
      if (!left_count || !right_count) {
        continue;
      }

      const cost =
        surface_area(left_bounds[i]) * left_count + surface_area(right_bounds[i + 1]) * right_count;

      if (!best || cost < best.cost) {
        best = {
          type: "object",
          axis,
          split_index: i,
          plane: axis_min + (extent * (i + 1)) / SBVH_BIN_COUNT,
          cost,
        };
      }
    }
  }

  return best;
}

function find_best_spatial_split(refs, node_bounds) {
  let best = null;

  for (let axis = 0; axis < 3; axis++) {
    const extent = bounds_extent(node_bounds, axis);
    if (extent <= BOUNDS_EPSILON) {
      continue;
    }

    const axis_min =
      axis === 0 ? node_bounds.min_x : axis === 1 ? node_bounds.min_y : node_bounds.min_z;
    const bins = Array.from({ length: SBVH_BIN_COUNT }, make_bin);

    for (let i = 0; i < refs.length; i++) {
      const clipped = clip_ref_to_bounds(refs[i], node_bounds);
      if (!clipped) {
        continue;
      }

      const clipped_min =
        axis === 0 ? clipped.min_x : axis === 1 ? clipped.min_y : clipped.min_z;
      const clipped_max =
        axis === 0 ? clipped.max_x : axis === 1 ? clipped.max_y : clipped.max_z;

      const first_bin = bin_index(clipped_min, axis_min, extent);
      const last_bin = bin_index(clipped_max, axis_min, extent);

      bins[first_bin].enter++;
      bins[last_bin].exit++;

      for (let bin = first_bin; bin <= last_bin; bin++) {
        const plane_min = axis_min + (extent * bin) / SBVH_BIN_COUNT;
        const plane_max = axis_min + (extent * (bin + 1)) / SBVH_BIN_COUNT;
        const chopped = clip_ref_to_bin(clipped, axis, plane_min, plane_max);
        if (chopped) {
          expand_bounds(bins[bin].bounds, chopped);
        }
      }
    }

    const left_counts = new Uint32Array(SBVH_BIN_COUNT);
    const right_counts = new Uint32Array(SBVH_BIN_COUNT);
    const left_bounds = Array.from({ length: SBVH_BIN_COUNT }, make_empty_bounds);
    const right_bounds = Array.from({ length: SBVH_BIN_COUNT }, make_empty_bounds);

    let running_count = 0;
    let running_bounds = make_empty_bounds();
    for (let i = 0; i < SBVH_BIN_COUNT; i++) {
      running_count += bins[i].enter;
      running_bounds = merge_bounds(running_bounds, bins[i].bounds);
      left_counts[i] = running_count;
      left_bounds[i] = running_bounds;
    }

    running_count = 0;
    running_bounds = make_empty_bounds();
    for (let i = SBVH_BIN_COUNT - 1; i >= 0; i--) {
      running_count += bins[i].exit;
      running_bounds = merge_bounds(running_bounds, bins[i].bounds);
      right_counts[i] = running_count;
      right_bounds[i] = running_bounds;
    }

    for (let i = 0; i < SBVH_BIN_COUNT - 1; i++) {
      const left_count = left_counts[i];
      const right_count = right_counts[i + 1];
      if (!left_count || !right_count) {
        continue;
      }

      const cost =
        surface_area(left_bounds[i]) * left_count + surface_area(right_bounds[i + 1]) * right_count;

      if (!best || cost < best.cost) {
        best = {
          type: "spatial",
          axis,
          split_index: i,
          plane: axis_min + (extent * (i + 1)) / SBVH_BIN_COUNT,
          cost,
        };
      }
    }
  }

  return best;
}

function longest_axis(bounds) {
  const ex = bounds.max_x - bounds.min_x;
  const ey = bounds.max_y - bounds.min_y;
  const ez = bounds.max_z - bounds.min_z;
  if (ex >= ey && ex >= ez) return 0;
  if (ey >= ez) return 1;
  return 2;
}

function fallback_partition(refs, node_bounds) {
  const axis = longest_axis(node_bounds);
  const sorted = refs.slice().sort((a, b) => ref_centroid(a, axis) - ref_centroid(b, axis));
  const mid = Math.max(1, Math.floor(sorted.length * 0.5));
  return {
    left: sorted.slice(0, mid),
    right: sorted.slice(mid),
  };
}

function partition_object(refs, split) {
  const left = [];
  const right = [];

  for (let i = 0; i < refs.length; i++) {
    const ref = refs[i];
    if (ref_centroid(ref, split.axis) <= split.plane) {
      left.push(ref);
    } else {
      right.push(ref);
    }
  }

  if (!left.length || !right.length) {
    return null;
  }

  return { left, right };
}

function choose_single_side(ref, split) {
  return ref_centroid(ref, split.axis) <= split.plane ? "left" : "right";
}

function partition_spatial(refs, node_bounds, split, state) {
  const left = [];
  const right = [];

  for (let i = 0; i < refs.length; i++) {
    const clipped = clip_ref_to_bounds(refs[i], node_bounds);
    if (!clipped) {
      continue;
    }

    const min_v =
      split.axis === 0 ? clipped.min_x : split.axis === 1 ? clipped.min_y : clipped.min_z;
    const max_v =
      split.axis === 0 ? clipped.max_x : split.axis === 1 ? clipped.max_y : clipped.max_z;

    if (max_v <= split.plane + BOUNDS_EPSILON) {
      left.push(clipped);
      continue;
    }

    if (min_v >= split.plane - BOUNDS_EPSILON) {
      right.push(clipped);
      continue;
    }

    const left_ref = clone_ref(clipped);
    const right_ref = clone_ref(clipped);
    if (split.axis === 0) {
      left_ref.max_x = Math.min(left_ref.max_x, split.plane);
      right_ref.min_x = Math.max(right_ref.min_x, split.plane);
    } else if (split.axis === 1) {
      left_ref.max_y = Math.min(left_ref.max_y, split.plane);
      right_ref.min_y = Math.max(right_ref.min_y, split.plane);
    } else {
      left_ref.max_z = Math.min(left_ref.max_z, split.plane);
      right_ref.min_z = Math.max(right_ref.min_z, split.plane);
    }

    const left_valid = is_valid_bounds(left_ref);
    const right_valid = is_valid_bounds(right_ref);

    if (left_valid && right_valid && state.reference_count < state.max_reference_count) {
      left.push(left_ref);
      right.push(right_ref);
      state.reference_count++;
      continue;
    }

    const preferred_side = choose_single_side(clipped, split);
    if (preferred_side === "left" && left_valid) {
      left.push(left_ref);
    } else if (right_valid) {
      right.push(right_ref);
    } else if (left_valid) {
      left.push(left_ref);
    }
  }

  if (!left.length || !right.length) {
    return null;
  }

  return { left, right };
}

function build_node(refs, depth, state) {
  if (refs.length === 1) {
    return {
      leaf: true,
      tri_id: refs[0].tri_id,
      min_x: refs[0].min_x,
      min_y: refs[0].min_y,
      min_z: refs[0].min_z,
      max_x: refs[0].max_x,
      max_y: refs[0].max_y,
      max_z: refs[0].max_z,
    };
  }

  const node_bounds = compute_bounds(refs);
  const object_split = find_best_object_split(refs);
  const can_try_spatial =
    depth < SBVH_MAX_SPATIAL_DEPTH &&
    refs.length > 2 &&
    state.reference_count < state.max_reference_count;
  const spatial_split = can_try_spatial ? find_best_spatial_split(refs, node_bounds) : null;

  let partition = null;
  if (spatial_split && (!object_split || spatial_split.cost + COST_EPSILON < object_split.cost)) {
    partition = partition_spatial(refs, node_bounds, spatial_split, state);
  }

  if (!partition && object_split) {
    partition = partition_object(refs, object_split);
  }

  if (!partition) {
    partition = fallback_partition(refs, node_bounds);
  }

  const left_node = build_node(partition.left, depth + 1, state);
  const right_node = build_node(partition.right, depth + 1, state);

  return {
    leaf: false,
    left: left_node,
    right: right_node,
    min_x: node_bounds.min_x,
    min_y: node_bounds.min_y,
    min_z: node_bounds.min_z,
    max_x: node_bounds.max_x,
    max_y: node_bounds.max_y,
    max_z: node_bounds.max_z,
  };
}

function write_node(nodes, index, min_x, min_y, min_z, min_w, max_x, max_y, max_z, max_w) {
  const base = index * 8;
  nodes[base + 0] = min_x;
  nodes[base + 1] = min_y;
  nodes[base + 2] = min_z;
  nodes[base + 3] = min_w;
  nodes[base + 4] = max_x;
  nodes[base + 5] = max_y;
  nodes[base + 6] = max_z;
  nodes[base + 7] = max_w;
}

function flatten_tree(root, reference_count) {
  const node_count = Math.max(1, reference_count * 2 - 1);
  const nodes = new Float32Array(node_count * 8);
  let next_leaf = 0;
  let next_internal = reference_count;

  function visit(node) {
    if (node.leaf) {
      const index = next_leaf++;
      write_node(
        nodes,
        index,
        node.min_x,
        node.min_y,
        node.min_z,
        node.tri_id,
        node.max_x,
        node.max_y,
        node.max_z,
        -1.0
      );
      return index;
    }

    const left_index = visit(node.left);
    const right_index = visit(node.right);
    const index = next_internal++;
    write_node(
      nodes,
      index,
      node.min_x,
      node.min_y,
      node.min_z,
      left_index,
      node.max_x,
      node.max_y,
      node.max_z,
      right_index
    );
    return index;
  }

  const root_index = visit(root);
  if (root_index !== node_count - 1) {
    throw new Error(`[sbvh] expected root index ${node_count - 1}, got ${root_index}`);
  }

  return nodes;
}

export function patch_sbvh_child_indices(node_data, base_node_index, node_data_size) {
  const patched = node_data.slice();
  const node_count = Math.floor(patched.length / node_data_size);

  for (let node_index = 0; node_index < node_count; node_index++) {
    const base = node_index * node_data_size;
    const max_w = patched[base + 7];
    if (max_w < 0.0) {
      continue;
    }

    patched[base + 3] += base_node_index;
    patched[base + 7] += base_node_index;
  }

  return patched;
}

export function build_sbvh_from_positions_indices(positions, indices) {
  if (!(positions instanceof Float32Array) || !(indices instanceof Uint32Array) || indices.length < 3) {
    return null;
  }

  const primitive_count = Math.floor(indices.length / 3);
  if (primitive_count <= 0) {
    return null;
  }

  const refs = new Array(primitive_count);
  for (let tri_id = 0; tri_id < primitive_count; tri_id++) {
    const i0 = indices[tri_id * 3 + 0] ?? 0;
    const i1 = indices[tri_id * 3 + 1] ?? 0;
    const i2 = indices[tri_id * 3 + 2] ?? 0;
    const v0_base = i0 * 3;
    const v1_base = i1 * 3;
    const v2_base = i2 * 3;

    const v0x = positions[v0_base + 0] ?? 0.0;
    const v0y = positions[v0_base + 1] ?? 0.0;
    const v0z = positions[v0_base + 2] ?? 0.0;
    const v1x = positions[v1_base + 0] ?? 0.0;
    const v1y = positions[v1_base + 1] ?? 0.0;
    const v1z = positions[v1_base + 2] ?? 0.0;
    const v2x = positions[v2_base + 0] ?? 0.0;
    const v2y = positions[v2_base + 1] ?? 0.0;
    const v2z = positions[v2_base + 2] ?? 0.0;

    const min_x = Math.min(v0x, v1x, v2x);
    const min_y = Math.min(v0y, v1y, v2y);
    const min_z = Math.min(v0z, v1z, v2z);
    const max_x = Math.max(v0x, v1x, v2x);
    const max_y = Math.max(v0y, v1y, v2y);
    const max_z = Math.max(v0z, v1z, v2z);

    refs[tri_id] = {
      tri_id,
      min_x,
      min_y,
      min_z,
      max_x,
      max_y,
      max_z,
    };
  }

  const state = {
    reference_count: primitive_count,
    max_reference_count: Math.max(
      primitive_count,
      Math.ceil(primitive_count * SBVH_MAX_REFERENCE_MULTIPLIER)
    ),
  };

  const root = build_node(refs, 0, state);
  const node_data = flatten_tree(root, state.reference_count);

  return {
    primitive_count,
    reference_count: state.reference_count,
    node_count: Math.max(1, state.reference_count * 2 - 1),
    node_data,
  };
}

export function build_mesh_sbvh(mesh) {
  if (!mesh?.vertices || !mesh?.indices) {
    return null;
  }

  const positions = new Float32Array(mesh.vertices.length * 3);
  for (let i = 0; i < mesh.vertices.length; i++) {
    const position = mesh.vertices[i]?.position ?? [0.0, 0.0, 0.0];
    const base = i * 3;
    positions[base + 0] = position[0] ?? 0.0;
    positions[base + 1] = position[1] ?? 0.0;
    positions[base + 2] = position[2] ?? 0.0;
  }

  return build_sbvh_from_positions_indices(positions, mesh.indices);
}

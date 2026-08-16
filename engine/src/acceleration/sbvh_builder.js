// ═════════════════════════════════════════════════════════════════════════════════════════════════
//  SUNDOWN — CPU SPATIAL SPLIT BVH BUILDER
// ═════════════════════════════════════════════════════════════════════════════════════════════════
//
//  Builds a binary bottom-level acceleration structure from indexed triangle geometry. The builder
//  evaluates both centroid-based object splits and spatial splits using a binned surface-area
//  heuristic (SAH). Spatial splits may duplicate a triangle reference across both children, clipping
//  its exact convex polygon at the split plane to produce tight conservative fragment AABBs.
//
//  BUILD PIPELINE
//
//    Indexed geometry
//         │
//         ▼
//    Triangle AABB references ──► recursive object / spatial SAH partitioning
//                                             │
//                                             ▼
//                                   leaf-first flat BVH2 storage
//
//  NODE STORAGE — 8 × f32
//
//    ┌───────────────────────────────┬───────────────────────────────┐
//    │ min.xyz                       │ max.xyz                       │
//    ├───────────────────────────────┼───────────────────────────────┤
//    │ min.w                         │ max.w                         │
//    │ leaf: triangle id             │ leaf: -1                     │
//    │ branch: left child index      │ branch: right child index    │
//    └───────────────────────────────┴───────────────────────────────┘
//
//  Leaves occupy [0, reference_count); branches follow in post-order, which guarantees that the
//  root is the final node. A spatially split triangle can therefore appear in more than one leaf.
//
// ═════════════════════════════════════════════════════════════════════════════════════════════════

export const SBVH_BIN_COUNT = 16; // SAH resolution per axis.
export const SBVH_MAX_REFERENCE_MULTIPLIER = 2; // Maximum leaf growth from spatial duplication.
export const SBVH_MAX_SPATIAL_DEPTH = 8; // Restricts expensive spatial evaluation near the root.
export const SBVH_MAX_TREE_DEPTH = 24; // Must not exceed the traversal shader's node stack.
const COST_EPSILON = 1e-5; // Requires a meaningful spatial-SAH win over an object split.
const BOUNDS_EPSILON = 1e-6; // Treats near-zero extents and plane contacts as degenerate.
const float_rounding_values = new Float32Array(1);
const float_rounding_bits = new Uint32Array(float_rounding_values.buffer);

/**
 * Rounds a lower bound outward to the nearest representable f32.
 *
 * @param {number} value
 * @returns {number}
 */
function round_min_to_f32(value) {
  const rounded = Math.fround(value);
  if (!Number.isFinite(rounded) || rounded <= value) {
    return rounded;
  }

  float_rounding_values[0] = rounded;
  if (rounded === 0.0) {
    float_rounding_bits[0] = 0x80000001;
  } else if (rounded > 0.0) {
    float_rounding_bits[0]--;
  } else {
    float_rounding_bits[0]++;
  }
  return float_rounding_values[0];
}

/**
 * Rounds an upper bound outward to the nearest representable f32.
 *
 * @param {number} value
 * @returns {number}
 */
function round_max_to_f32(value) {
  const rounded = Math.fround(value);
  if (!Number.isFinite(rounded) || rounded >= value) {
    return rounded;
  }

  float_rounding_values[0] = rounded;
  if (rounded === 0.0) {
    float_rounding_bits[0] = 0x00000001;
  } else if (rounded > 0.0) {
    float_rounding_bits[0]++;
  } else {
    float_rounding_bits[0]--;
  }
  return float_rounding_values[0];
}

// ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                      BOUNDS PRIMITIVES                                        │
// └───────────────────────────────────────────────────────────────────────────────────────────────┘
//
// Bounds and triangle references intentionally share the same min/max field names. This keeps the
// bounds helpers polymorphic without creating temporary vector objects.

/**
 * Creates the identity value for AABB expansion.
 *
 * Inverted infinities let the first finite reference replace all six components naturally.
 *
 * @returns {object}
 */
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

/**
 * Tests whether every minimum lies on or below its matching maximum.
 *
 * @param {object} bounds
 * @returns {boolean}
 */
function is_valid_bounds(bounds) {
  return (
    bounds.min_x <= bounds.max_x &&
    bounds.min_y <= bounds.max_y &&
    bounds.min_z <= bounds.max_z
  );
}

/**
 * Builds a triangle reference from an exact convex polygon fragment.
 *
 * Spatial references retain their polygon so subsequent split candidates can clip actual geometry
 * instead of repeatedly chopping an increasingly loose conservative AABB.
 *
 * @param {number} tri_id
 * @param {number[]} polygon - Flat XYZ triples in winding order.
 * @returns {object|null}
 */
function make_ref_from_polygon(tri_id, polygon) {
  if (!polygon || polygon.length < 9) {
    return null;
  }

  let min_x = Infinity;
  let min_y = Infinity;
  let min_z = Infinity;
  let max_x = -Infinity;
  let max_y = -Infinity;
  let max_z = -Infinity;
  for (let i = 0; i < polygon.length; i += 3) {
    const x = polygon[i + 0];
    const y = polygon[i + 1];
    const z = polygon[i + 2];
    min_x = Math.min(min_x, x);
    min_y = Math.min(min_y, y);
    min_z = Math.min(min_z, z);
    max_x = Math.max(max_x, x);
    max_y = Math.max(max_y, y);
    max_z = Math.max(max_z, z);
  }

  const ref = {
    tri_id,
    polygon,
    min_x,
    min_y,
    min_z,
    max_x,
    max_y,
    max_z,
  };
  return is_valid_bounds(ref) ? ref : null;
}

/**
 * Clips a convex polygon against one axis-aligned half-space.
 *
 * @param {number[]} polygon - Flat XYZ triples in winding order.
 * @param {number} axis
 * @param {number} plane
 * @param {boolean} keep_less_equal
 * @returns {number[]|null}
 */
function clip_polygon_axis(polygon, axis, plane, keep_less_equal) {
  if (!polygon || polygon.length < 9) {
    return null;
  }

  const clipped = [];
  let previous_base = polygon.length - 3;
  let previous_coordinate = polygon[previous_base + axis];
  let previous_inside = keep_less_equal
    ? previous_coordinate <= plane
    : previous_coordinate >= plane;

  for (let current_base = 0; current_base < polygon.length; current_base += 3) {
    const current_coordinate = polygon[current_base + axis];
    const current_inside = keep_less_equal
      ? current_coordinate <= plane
      : current_coordinate >= plane;

    if (current_inside !== previous_inside) {
      const denominator = current_coordinate - previous_coordinate;
      if (denominator !== 0.0) {
        const t = (plane - previous_coordinate) / denominator;
        const intersection_x =
          polygon[previous_base + 0] + (polygon[current_base + 0] - polygon[previous_base + 0]) * t;
        const intersection_y =
          polygon[previous_base + 1] + (polygon[current_base + 1] - polygon[previous_base + 1]) * t;
        const intersection_z =
          polygon[previous_base + 2] + (polygon[current_base + 2] - polygon[previous_base + 2]) * t;
        clipped.push(intersection_x, intersection_y, intersection_z);
        clipped[clipped.length - 3 + axis] = plane;
      }
    }

    if (current_inside) {
      clipped.push(polygon[current_base + 0], polygon[current_base + 1], polygon[current_base + 2]);
    }

    previous_base = current_base;
    previous_coordinate = current_coordinate;
    previous_inside = current_inside;
  }

  return clipped.length >= 9 ? clipped : null;
}

/**
 * Returns the midpoint of a reference AABB along an axis.
 *
 * @param {object} ref
 * @param {number} axis - 0 = x, 1 = y, 2 = z.
 * @returns {number}
 */
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

/**
 * Returns an AABB extent along an axis.
 *
 * @param {object} bounds
 * @param {number} axis - 0 = x, 1 = y, 2 = z.
 * @returns {number}
 */
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

/**
 * Computes AABB surface area for SAH scoring.
 *
 * Negative extents are clamped so an empty accumulator contributes zero area.
 *
 * @param {object} bounds
 * @returns {number}
 */
function surface_area(bounds) {
  const ex = Math.max(0.0, bounds.max_x - bounds.min_x);
  const ey = Math.max(0.0, bounds.max_y - bounds.min_y);
  const ez = Math.max(0.0, bounds.max_z - bounds.min_z);
  return 2.0 * (ex * ey + ex * ez + ey * ez);
}

/**
 * Expands an AABB in place to include another bounds-like object.
 *
 * @param {object} bounds
 * @param {object} ref
 */
function expand_bounds(bounds, ref) {
  bounds.min_x = Math.min(bounds.min_x, ref.min_x);
  bounds.min_y = Math.min(bounds.min_y, ref.min_y);
  bounds.min_z = Math.min(bounds.min_z, ref.min_z);
  bounds.max_x = Math.max(bounds.max_x, ref.max_x);
  bounds.max_y = Math.max(bounds.max_y, ref.max_y);
  bounds.max_z = Math.max(bounds.max_z, ref.max_z);
}

/**
 * Returns the union of two AABBs while preserving the non-empty operand.
 *
 * @param {object} a
 * @param {object} b
 * @returns {object}
 */
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

/**
 * Computes the conservative bounds of a reference set.
 *
 * @param {object[]} refs
 * @returns {object}
 */
function compute_bounds(refs) {
  const bounds = make_empty_bounds();
  for (let i = 0; i < refs.length; i++) {
    expand_bounds(bounds, refs[i]);
  }
  return bounds;
}

/**
 * Computes bounds around reference centroids for object-split binning.
 *
 * @param {object[]} refs
 * @returns {object}
 */
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

/**
 * Intersects a reference polygon with the current node bounds when required.
 *
 * The normal fast path returns the reference unchanged because every descendant fragment is already
 * inside its node. Exact clipping remains as a numerical guard for inherited boundary drift.
 *
 * @param {object} ref
 * @param {object} bounds
 * @returns {object|null}
 */
function clip_ref_to_bounds(ref, bounds) {
  if (
    ref.min_x >= bounds.min_x &&
    ref.min_y >= bounds.min_y &&
    ref.min_z >= bounds.min_z &&
    ref.max_x <= bounds.max_x &&
    ref.max_y <= bounds.max_y &&
    ref.max_z <= bounds.max_z
  ) {
    return ref;
  }

  let polygon = clip_polygon_axis(ref.polygon, 0, bounds.min_x, false);
  polygon = polygon ? clip_polygon_axis(polygon, 0, bounds.max_x, true) : null;
  polygon = polygon ? clip_polygon_axis(polygon, 1, bounds.min_y, false) : null;
  polygon = polygon ? clip_polygon_axis(polygon, 1, bounds.max_y, true) : null;
  polygon = polygon ? clip_polygon_axis(polygon, 2, bounds.min_z, false) : null;
  polygon = polygon ? clip_polygon_axis(polygon, 2, bounds.max_z, true) : null;
  return polygon ? make_ref_from_polygon(ref.tri_id, polygon) : null;
}

/**
 * Intersects a reference with one bin slab along the selected axis.
 *
 * @param {object} ref
 * @param {number} axis
 * @param {number} plane_min
 * @param {number} plane_max
 * @returns {object|null}
 */
function clip_ref_to_bin(ref, axis, plane_min, plane_max) {
  let polygon = clip_polygon_axis(ref.polygon, axis, plane_min, false);
  polygon = polygon ? clip_polygon_axis(polygon, axis, plane_max, true) : null;
  return polygon ? make_ref_from_polygon(ref.tri_id, polygon) : null;
}

// ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                     BINNED SAH SEARCH                                         │
// └───────────────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Creates one bin accumulator.
 *
 * Object splits use `count`; spatial splits use `enter` and `exit` events. Both paths accumulate
 * clipped reference bounds in the shared `bounds` field.
 *
 * @returns {{enter: number, exit: number, count: number, bounds: object}}
 */
function make_bin() {
  return {
    enter: 0,
    exit: 0,
    count: 0,
    bounds: make_empty_bounds(),
  };
}

/**
 * Maps a scalar coordinate into a clamped bin index.
 *
 * @param {number} value
 * @param {number} min_value
 * @param {number} extent
 * @returns {number}
 */
function bin_index(value, min_value, extent) {
  if (extent <= BOUNDS_EPSILON) {
    return 0;
  }
  const scaled = ((value - min_value) / extent) * SBVH_BIN_COUNT;
  const idx = Math.floor(scaled);
  return Math.max(0, Math.min(SBVH_BIN_COUNT - 1, idx));
}

/**
 * Maps an inclusive maximum coordinate to its final occupied bin.
 *
 * A maximum exactly on a bin plane belongs to the bin on the plane's left, matching the spatial
 * partitioner's `max <= plane` ownership rule.
 *
 * @param {number} value
 * @param {number} min_value
 * @param {number} extent
 * @returns {number}
 */
function last_bin_index(value, min_value, extent) {
  if (extent <= BOUNDS_EPSILON) {
    return 0;
  }
  const scaled = ((value - min_value) / extent) * SBVH_BIN_COUNT;
  const idx = Math.ceil(scaled) - 1;
  return Math.max(0, Math.min(SBVH_BIN_COUNT - 1, idx));
}

/**
 * Finds the lowest-cost object split across all axes.
 *
 * Each reference belongs to exactly one centroid bin. Prefix and suffix scans then evaluate every
 * boundary in O(bin_count), using the unnormalized SAH cost:
 *
 *   cost = left_surface_area × left_count + right_surface_area × right_count
 *
 * The common parent-area and traversal terms are omitted because only relative split cost matters.
 *
 * @param {object[]} refs
 * @returns {{type: string, axis: number, split_index: number, plane: number, cost: number}|null}
 */
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

    // Prefix scan: all references on or before each candidate boundary.
    let running_count = 0;
    let running_bounds = make_empty_bounds();
    for (let i = 0; i < SBVH_BIN_COUNT; i++) {
      running_count += bins[i].count;
      running_bounds = merge_bounds(running_bounds, bins[i].bounds);
      left_counts[i] = running_count;
      left_bounds[i] = running_bounds;
    }

    // Suffix scan: all references after each candidate boundary.
    running_count = 0;
    running_bounds = make_empty_bounds();
    for (let i = SBVH_BIN_COUNT - 1; i >= 0; i--) {
      running_count += bins[i].count;
      running_bounds = merge_bounds(running_bounds, bins[i].bounds);
      right_counts[i] = running_count;
      right_bounds[i] = running_bounds;
    }

    // A split plane lies between bin i and bin i + 1; empty children are never valid.
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

/**
 * Finds the lowest-cost spatial split across all axes.
 *
 * A reference contributes an enter event to its first overlapping bin and an exit event to its
 * last. Prefix/suffix event scans count the reference once on each side of any plane it crosses,
 * modeling the duplicated leaves that partition_spatial() may emit. Per-bin bounds are clipped to
 * the bin slab so SAH evaluates the overlap reduction produced by the candidate plane.
 *
 * @param {object[]} refs
 * @param {object} node_bounds
 * @param {number} max_duplications
 * @returns {{type: string, axis: number, split_index: number, plane: number, cost: number}|null}
 */
function find_best_spatial_split(refs, node_bounds, max_duplications) {
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
      let last_bin = last_bin_index(clipped_max, axis_min, extent);
      if (last_bin < first_bin) {
        last_bin = first_bin;
      }

      bins[first_bin].enter++;
      bins[last_bin].exit++;

      // Accumulate only the portion of the conservative reference inside each overlapped slab.
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

    // References enter the left prefix once and remain active for all following boundaries.
    let running_count = 0;
    let running_bounds = make_empty_bounds();
    for (let i = 0; i < SBVH_BIN_COUNT; i++) {
      running_count += bins[i].enter;
      running_bounds = merge_bounds(running_bounds, bins[i].bounds);
      left_counts[i] = running_count;
      left_bounds[i] = running_bounds;
    }

    // References enter the right suffix from their last occupied bin and remain active backward.
    running_count = 0;
    running_bounds = make_empty_bounds();
    for (let i = SBVH_BIN_COUNT - 1; i >= 0; i--) {
      running_count += bins[i].exit;
      running_bounds = merge_bounds(running_bounds, bins[i].bounds);
      right_counts[i] = running_count;
      right_bounds[i] = running_bounds;
    }

    // Crossing references are intentionally counted in both children for spatial SAH.
    for (let i = 0; i < SBVH_BIN_COUNT - 1; i++) {
      const left_count = left_counts[i];
      const right_count = right_counts[i + 1];
      if (!left_count || !right_count) {
        continue;
      }

      const duplication_count = left_count + right_count - refs.length;
      if (duplication_count > max_duplications) {
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
          duplication_count,
        };
      }
    }
  }

  return best;
}

// ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                      PARTITION POLICY                                         │
// └───────────────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Selects the widest node axis for deterministic fallback partitioning.
 *
 * @param {object} bounds
 * @returns {number}
 */
function longest_axis(bounds) {
  const ex = bounds.max_x - bounds.min_x;
  const ey = bounds.max_y - bounds.min_y;
  const ez = bounds.max_z - bounds.min_z;
  if (ex >= ey && ex >= ez) return 0;
  if (ey >= ez) return 1;
  return 2;
}

/**
 * Guarantees progress when binned SAH cannot produce two non-empty children.
 *
 * Sorting by centroid and cutting at the median handles coincident centroids, zero-area geometry,
 * and other degenerate input without creating an empty recursive branch.
 *
 * @param {object[]} refs
 * @param {object} node_bounds
 * @returns {{left: object[], right: object[]}}
 */
function fallback_partition(refs, node_bounds) {
  const axis = longest_axis(node_bounds);
  const sorted = refs.slice().sort((a, b) => ref_centroid(a, axis) - ref_centroid(b, axis));
  const mid = Math.max(1, Math.floor(sorted.length * 0.5));
  return {
    left: sorted.slice(0, mid),
    right: sorted.slice(mid),
  };
}

/**
 * Applies a centroid object split without duplicating references.
 *
 * @param {object[]} refs
 * @param {object} split
 * @returns {{left: object[], right: object[]}|null}
 */
function partition_object(refs, split) {
  const left = [];
  const right = [];

  for (let i = 0; i < refs.length; i++) {
    const ref = refs[i];
    if (ref_centroid(ref, split.axis) < split.plane) {
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

/**
 * Applies a spatial split, clipping crossing references at the split plane.
 *
 * The partition is transactional: either every crossing reference receives both clipped fragments,
 * or the complete candidate is rejected. This guarantees that the union of emitted leaf bounds
 * continues to cover the original triangle and prevents failed candidates from consuming budget.
 *
 * @param {object[]} refs
 * @param {object} node_bounds
 * @param {object} split
 * @param {number} max_duplications
 * @returns {{left: object[], right: object[], duplication_count: number}|null}
 */
function partition_spatial(refs, node_bounds, split, max_duplications) {
  const left = [];
  const right = [];
  let duplication_count = 0;

  for (let i = 0; i < refs.length; i++) {
    const clipped = clip_ref_to_bounds(refs[i], node_bounds);
    if (!clipped) {
      continue;
    }

    const min_v =
      split.axis === 0 ? clipped.min_x : split.axis === 1 ? clipped.min_y : clipped.min_z;
    const max_v =
      split.axis === 0 ? clipped.max_x : split.axis === 1 ? clipped.max_y : clipped.max_z;

    if (max_v <= split.plane) {
      left.push(clipped);
      continue;
    }

    if (min_v >= split.plane) {
      right.push(clipped);
      continue;
    }

    // Split the exact convex triangle fragment so both output AABBs are tight on every axis.
    const left_polygon = clip_polygon_axis(clipped.polygon, split.axis, split.plane, true);
    const right_polygon = clip_polygon_axis(clipped.polygon, split.axis, split.plane, false);
    const left_ref = left_polygon ? make_ref_from_polygon(clipped.tri_id, left_polygon) : null;
    const right_ref = right_polygon ? make_ref_from_polygon(clipped.tri_id, right_polygon) : null;

    if (left_ref && right_ref) {
      left.push(left_ref);
      right.push(right_ref);
      duplication_count++;
      continue;
    }

    // Plane contacts can collapse one fragment. Keep the surviving exact fragment without
    // consuming duplication budget; the opposite side contains no triangle area.
    if (left_ref) {
      left.push(left_ref);
    } else if (right_ref) {
      right.push(right_ref);
    }
  }

  if (!left.length || !right.length || duplication_count > max_duplications) {
    return null;
  }

  return { left, right, duplication_count };
}

/**
 * Returns the minimum height of a binary subtree with one primitive per leaf.
 *
 * @param {number} reference_count
 * @returns {number}
 */
function minimum_subtree_depth(reference_count) {
  return Math.ceil(Math.log2(Math.max(1, reference_count)));
}

/**
 * Checks whether a partition can still be completed within the shader stack contract.
 *
 * @param {{left: object[], right: object[]}} partition
 * @param {number} child_depth
 * @returns {boolean}
 */
function partition_fits_depth(partition, child_depth) {
  return (
    child_depth + minimum_subtree_depth(partition.left.length) <= SBVH_MAX_TREE_DEPTH &&
    child_depth + minimum_subtree_depth(partition.right.length) <= SBVH_MAX_TREE_DEPTH
  );
}

// ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                    RECURSIVE CONSTRUCTION                                     │
// └───────────────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Recursively builds the pointer-based intermediate BVH.
 *
 * Spatial SAH is considered only while depth, reference-count, and node-size guards allow it. A
 * spatial candidate must beat the object candidate by COST_EPSILON; otherwise the cheaper and
 * topology-preserving object path wins. Median partitioning is the final progress guarantee.
 *
 * @param {object[]} refs
 * @param {number} depth
 * @param {{reference_count: number, max_reference_count: number}} state
 * @returns {object}
 */
function build_node(refs, depth, state) {
  if (depth + minimum_subtree_depth(refs.length) > SBVH_MAX_TREE_DEPTH) {
    throw new Error(
      `[sbvh] ${refs.length} references at depth ${depth} cannot fit within max depth ${SBVH_MAX_TREE_DEPTH}`
    );
  }

  if (refs.length === 1) {
    state.max_depth = Math.max(state.max_depth, depth);
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
  const remaining_duplications = state.max_reference_count - state.reference_count;
  const spatial_split = can_try_spatial
    ? find_best_spatial_split(refs, node_bounds, remaining_duplications)
    : null;

  let partition = null;
  let accepted_duplications = 0;
  if (spatial_split && (!object_split || spatial_split.cost + COST_EPSILON < object_split.cost)) {
    const spatial_partition = partition_spatial(
      refs,
      node_bounds,
      spatial_split,
      remaining_duplications
    );
    if (spatial_partition && partition_fits_depth(spatial_partition, depth + 1)) {
      partition = spatial_partition;
      accepted_duplications = spatial_partition.duplication_count;
    }
  }

  // A rejected or degenerate spatial partition falls back to the best object split.
  if (!partition && object_split) {
    const object_partition = partition_object(refs, object_split);
    if (object_partition && partition_fits_depth(object_partition, depth + 1)) {
      partition = object_partition;
    }
  }

  if (!partition) {
    partition = fallback_partition(refs, node_bounds);
  }

  if (!partition_fits_depth(partition, depth + 1)) {
    throw new Error(`[sbvh] fallback partition exceeded max depth ${SBVH_MAX_TREE_DEPTH}`);
  }

  state.reference_count += accepted_duplications;

  // Build children before emitting this branch so flattening can preserve post-order layout.
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

// ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                      GPU NODE ENCODING                                        │
// └───────────────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Writes one tightly packed 8-float node record.
 *
 * Metadata occupies the two w lanes so bounds and topology can be fetched together as two vec4s.
 *
 * @param {Float32Array} nodes
 * @param {number} index
 * @param {number} min_x
 * @param {number} min_y
 * @param {number} min_z
 * @param {number} min_w - Triangle id for leaves; left child index for branches.
 * @param {number} max_x
 * @param {number} max_y
 * @param {number} max_z
 * @param {number} max_w - -1 for leaves; right child index for branches.
 */
function write_node(nodes, index, min_x, min_y, min_z, min_w, max_x, max_y, max_z, max_w) {
  const base = index * 8;
  nodes[base + 0] = round_min_to_f32(min_x);
  nodes[base + 1] = round_min_to_f32(min_y);
  nodes[base + 2] = round_min_to_f32(min_z);
  nodes[base + 3] = min_w;
  nodes[base + 4] = round_max_to_f32(max_x);
  nodes[base + 5] = round_max_to_f32(max_y);
  nodes[base + 6] = round_max_to_f32(max_z);
  nodes[base + 7] = max_w;
}

/**
 * Flattens the intermediate tree into leaf-first, post-order BVH2 storage.
 *
 * There are R leaves and R - 1 branches for R final references. Separate cursors reserve the first
 * R slots for leaves while recursive post-order traversal places every parent after its children.
 * The resulting root index is therefore always node_count - 1.
 *
 * @param {object} root
 * @param {number} reference_count
 * @returns {Float32Array}
 */
function flatten_tree(root, reference_count) {
  const node_count = Math.max(1, reference_count * 2 - 1);
  const nodes = new Float32Array(node_count * 8);
  let next_leaf = 0;
  let next_internal = reference_count;

  /**
   * Emits a subtree and returns its flat node index.
   *
   * @param {object} node
   * @returns {number}
   */
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

// ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                         PUBLIC API                                            │
// └───────────────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Rebases branch child indices for insertion into a larger shared node buffer.
 *
 * The source array is copied so cached mesh-local SBVH data remains reusable. Leaves are identified
 * by their negative max.w sentinel and retain their triangle id in min.w.
 *
 * @param {Float32Array} node_data
 * @param {number} base_node_index
 * @param {number} node_data_size - Float components per node; expected to provide w lanes at 3/7.
 * @returns {Float32Array}
 */
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

/**
 * Builds an SBVH directly from tightly packed positions and triangle indices.
 *
 * Each indexed triangle becomes one conservative AABB reference. The spatial split budget may grow
 * the final reference count up to SBVH_MAX_REFERENCE_MULTIPLIER times the primitive count.
 *
 * @param {Float32Array} positions - XYZ triples indexed by `indices`.
 * @param {Uint32Array} indices - Triangle-list vertex indices.
 * @returns {{
 *   primitive_count: number,
 *   reference_count: number,
 *   node_count: number,
 *   max_depth: number,
 *   node_data: Float32Array
 * }|null}
 */
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
    // Missing components resolve to zero so malformed-but-addressable input remains finite.
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

    const polygon = [v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z];
    refs[tri_id] = make_ref_from_polygon(tri_id, polygon) ?? {
      tri_id,
      polygon,
      min_x,
      min_y,
      min_z,
      max_x,
      max_y,
      max_z,
    };
  }

  if (minimum_subtree_depth(primitive_count) > SBVH_MAX_TREE_DEPTH) {
    throw new Error(
      `[sbvh] ${primitive_count} primitives cannot fit within max depth ${SBVH_MAX_TREE_DEPTH}`
    );
  }

  // reference_count begins at one leaf per primitive and grows once per accepted duplication.
  const state = {
    reference_count: primitive_count,
    max_reference_count: Math.max(
      primitive_count,
      Math.ceil(primitive_count * SBVH_MAX_REFERENCE_MULTIPLIER)
    ),
    max_depth: 0,
  };

  const root = build_node(refs, 0, state);
  const node_data = flatten_tree(root, state.reference_count);

  return {
    primitive_count,
    reference_count: state.reference_count,
    node_count: Math.max(1, state.reference_count * 2 - 1),
    max_depth: state.max_depth,
    node_data,
  };
}

/**
 * Builds an SBVH from the engine mesh representation.
 *
 * CPU position data is used directly when retained by the mesh. Otherwise a temporary packed array
 * is assembled from vertex objects before dispatching to the typed-array builder.
 *
 * @param {object} mesh
 * @returns {{
 *   primitive_count: number,
 *   reference_count: number,
 *   node_count: number,
 *   max_depth: number,
 *   node_data: Float32Array
 * }|null}
 */
export function build_mesh_sbvh(mesh) {
  if (!mesh?.indices) {
    return null;
  }

  let positions = mesh.cpu_position_data;
  if (!(positions instanceof Float32Array)) {
    if (!mesh.vertices) {
      return null;
    }
    positions = new Float32Array(mesh.vertices.length * 3);
    for (let i = 0; i < mesh.vertices.length; i++) {
      const position = mesh.vertices[i]?.position ?? [0.0, 0.0, 0.0];
      const base = i * 3;
      positions[base + 0] = position[0] ?? 0.0;
      positions[base + 1] = position[1] ?? 0.0;
      positions[base + 2] = position[2] ?? 0.0;
    }
  }

  return build_sbvh_from_positions_indices(positions, mesh.indices);
}

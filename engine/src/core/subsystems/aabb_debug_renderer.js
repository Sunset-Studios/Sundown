import { EntityManager } from "../ecs/entity.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { SimulationLayer } from "../simulation_layer.js";
import { AABB, AABB_NODE_TYPE, AABB_NODE_FLAGS } from "../../acceleration/aabb.js";
import { profile_scope } from "../../utility/performance.js";
import { LineRenderer } from "../../renderer/line_renderer.js";
import { vec3 } from "gl-matrix";

// Constants for debug rendering
const LEAF_NODE_COLOR = [2.0, 0.0, 2.0, 1.0]; // Purple for leaf nodes
const INTERNAL_NODE_COLOR = [2.0, 0.5, 0.0, 1.0]; // Orange for internal nodes
const ROOT_NODE_COLOR = [0.0, 0.0, 2.0, 1.0]; // Blue for root node
const BOUNDS_COLOR = [0.0, 2.0, 0.0, 1.0]; // Green for bounds
const RAY_COLOR = [1.0, 1.0, 0.0, 1.0]; // Yellow for ray

export class AABBTreeDebugRenderer extends SimulationLayer {
  enabled = false;
  max_depth_to_render = Infinity;
  show_bounds = true;
  show_leaf_nodes = true;
  show_internal_nodes = true;
  ray_origin = null;
  ray_direction = null;
  ray_hit_point = null;
  show_ray = false;
  line_collection_id = null;
  should_recreate_line_collection = false;

  constructor() {
    super();
    this.name = "aabb_tree_debug_renderer";
  }

  update(delta_time) {
    this.should_recreate_line_collection =
      this.should_recreate_line_collection || AABB.data.modified;

    if (!this.enabled) {
      if (this.line_collection_id) {
        LineRenderer.clear_collection(this.line_collection_id);
        this.line_collection_id = null;
      }
      return;
    }

    profile_scope("aabb_tree_debug_renderer", () => {
      // Collect visible nodes
      this._collect_visible_nodes();

      // Add ray visualization if needed
      if (this.show_ray) {
        this._add_ray_visualization();
      }

      this.should_recreate_line_collection = false;
    });
  }

  _collect_visible_nodes() {
    if (!this.should_recreate_line_collection) {
        return;
    }

    const transforms = EntityManager.get_fragment_array(TransformFragment);
    if (!transforms) {
      return;
    }
    
    if (this.line_collection_id) {
      LineRenderer.clear_collection(this.line_collection_id);
    }

    this.line_collection_id = LineRenderer.start_collection();

    for (let i = 1; i < AABB.size; i++) {
      const is_root = i === AABB.root_node;

      // Get the node
      const node = AABB.get_node_data(i);
      const is_free = (node.flags & AABB_NODE_FLAGS.FREE) != 0;
      if (is_free) {
        continue;
      }
      
      const is_detached = !is_root && node.parent === 0;
      if (is_detached) {
        if (this.show_bounds && node.user_data) {
          const min_point = node.min_point;
          const max_point = node.max_point;
  
          LineRenderer.add_box(min_point, max_point, BOUNDS_COLOR);
  
        }
        continue;
      }

      // Add this node for visualization
      const is_leaf = node.node_type === AABB_NODE_TYPE.LEAF;

      if ((is_leaf && this.show_leaf_nodes) || (!is_leaf && this.show_internal_nodes)) {
        // Get node bounds
        const min_point = node.min_point;
        const max_point = node.max_point;

        // Determine color based on node type
        let color;
        if (is_root) {
          color = ROOT_NODE_COLOR;
        } else if (is_leaf) {
          color = LEAF_NODE_COLOR;
        } else {
          color = INTERNAL_NODE_COLOR;
        }

        // Add 12 lines to represent the box (3 edges from each of 4 corners)
        LineRenderer.add_box(min_point, max_point, color);
      }
    }

    LineRenderer.end_collection();
  }

  _add_ray_visualization() {
    if (!this.ray_origin || !this.ray_direction) {
      return;
    }

    // Normalize direction
    const dir = vec3.normalize(vec3.create(), this.ray_direction);

    // Determine ray length
    let ray_length = 1000; // Default length

    // If we have a hit point, use that to determine the length
    if (this.ray_hit_point) {
      const diff = vec3.sub(vec3.create(), this.ray_hit_point, this.ray_origin);
      ray_length = vec3.length(diff);
    }

    // Calculate end point
    const end_point = [
      this.ray_origin[0] + dir[0] * ray_length,
      this.ray_origin[1] + dir[1] * ray_length,
      this.ray_origin[2] + dir[2] * ray_length,
    ];

    LineRenderer.add_line(this.ray_origin, end_point, RAY_COLOR);

    // If we have a hit point, add a small cross to mark it
    if (this.ray_hit_point) {
      const hit_size = 0.1; // Size of the hit marker

      LineRenderer.add_axes(this.ray_hit_point, hit_size);
    }
  }

  toggle_visualization() {
    this.enabled = !this.enabled;
    this.should_recreate_line_collection = this.enabled;
    return this.enabled;
  }

  set_max_depth(depth) {
    this.max_depth_to_render = depth;
  }

  toggle_bounds() {
    this.show_bounds = !this.show_bounds;
    this.should_recreate_line_collection = true;
    return this.show_bounds;
  }

  toggle_leaf_nodes() {
    this.show_leaf_nodes = !this.show_leaf_nodes;
    this.should_recreate_line_collection = true;
    return this.show_leaf_nodes;
  }

  toggle_internal_nodes() {
    this.show_internal_nodes = !this.show_internal_nodes;
    this.should_recreate_line_collection = true;
    return this.show_internal_nodes;
  }

  set_ray(origin, direction, hit_point = null) {
    this.ray_origin = origin;
    this.ray_direction = direction;
    this.ray_hit_point = hit_point;
    this.show_ray = true;
    this.should_recreate_line_collection = true;
  }

  clear_ray() {
    this.show_ray = false;
    this.should_recreate_line_collection = true;
  }
}

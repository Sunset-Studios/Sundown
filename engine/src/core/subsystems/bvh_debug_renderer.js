import { SimulationLayer } from "../simulation_layer.js";
import { profile_scope } from "../../utility/performance.js";

export class BVHDebugRenderer extends SimulationLayer {
  enabled = false;
  max_depth_to_render = Infinity;
  show_bounds = true;
  show_leaf_nodes = true;
  show_internal_nodes = true;

  constructor() {
    super();
    this.name = "aabb_tree_debug_renderer";

    this._update_internal = this._update_internal.bind(this);
  }

  update(delta_time) {
    profile_scope("aabb_tree_debug_renderer", () => {
      this._update_internal();
    });
  }

  _update_internal() {
    if (!this.enabled) {
      this.clear_all();
      return;
    }

    // Collect visible nodes
    this._collect_visible_nodes();
  }

  _collect_visible_nodes() {
    // TODO: Dispatch a compute shader or debug viz. shader to show boxes for nearby node bounds
  }

  toggle_visualization() {
    this.enabled = !this.enabled;
    return this.enabled;
  }

  set_max_depth(depth) {
    this.max_depth_to_render = depth;
  }

  toggle_bounds() {
    this.show_bounds = !this.show_bounds;
    return this.show_bounds;
  }

  toggle_leaf_nodes() {
    this.show_leaf_nodes = !this.show_leaf_nodes;
    return this.show_leaf_nodes;
  }

  toggle_internal_nodes() {
    this.show_internal_nodes = !this.show_internal_nodes;
    return this.show_internal_nodes;
  }

  clear_all() {
    // TODO: Dispatch a compute shader or debug viz. shader to clear all debug rendering
  }
}

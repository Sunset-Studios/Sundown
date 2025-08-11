import { ReSTIRGI } from "./restir.js";

class GI {
  final_gi_texture = null;
  restir = null;
  
  constructor(params = {}) {
    this.restir = new ReSTIRGI();
  }

  /**
   * Adds global-illumination passes to the render graph and exposes the final
   * indirect-lighting texture via `final_gi_texture`.
   */
  add_passes(
    render_graph,
    width,
    height,
    position_texture,
    aabb_bounds,
    object_instances,
    entity_aabb_node_indices,
    force_recreate = false
  ) {
    this.restir.add_passes(render_graph, { width, height, force_recreate });
    this.final_gi_texture = this.restir.output_texture;
  }
}

export { GI };

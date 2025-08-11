import { SimulationLayer } from "../simulation_layer.js";
import { BVHProcessor } from "../../acceleration/bvh_processor.js";
import { BVHRaycast } from "../../acceleration/bvh_raycast.js";
import { profile_scope } from "../../utility/performance.js";
import { Renderer } from "../../renderer/renderer.js";

const ENTITY_ADAPTER_UPDATE_NAME = "bvh_entity_adapter.update";

/**
 * Adapter that connects the entity system to a GPU-based acceleration structure.
 * It collects bounds from all dynamic entities and uses an BVHGPUTreeProcessor
 * to build a Top-Level Acceleration Structure (TLAS) each frame.
 */
export class BVHEntityAdapter extends SimulationLayer {
  constructor() {
    super();
    this.name = "bvh_entity_adapter";
  }

  /**
   * Initialize the adapter.
   */
  init() {
    this.tlas_processor = new BVHProcessor();
    Renderer.get().on_post_render(this.on_post_render.bind(this));
  }

  update(delta_time) {
    profile_scope(ENTITY_ADAPTER_UPDATE_NAME, () => {
        this.tlas_processor.build();
    });
  }

  post_update(delta_time) {
    super.post_update(delta_time);
    BVHRaycast.flush();
  }

  async on_post_render() {
    await BVHRaycast.gather_results();
  }

  /**
   * Get statistics about the BVH
   * @returns {Object} - Tree statistics
   */
  get_stats() {
    // GPU BVH stats not yet implemented; return minimal placeholder
    return {
      allocated_nodes: 0,
      leaf_nodes: 0,
      internal_nodes: 0,
      max_depth: 0,
      dirty_nodes: 0,
    };
  }
}

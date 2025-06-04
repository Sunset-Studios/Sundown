import { Renderer } from "../../renderer/renderer.js";
import { Chunk } from "../ecs/solar/chunk.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { profile_scope } from "../../utility/performance.js";
import { TypedQueue } from "../../memory/container.js";

const entity_preprocessor_pre_update_key = "entity_preprocessor_pre_update";
const entity_preprocessor_post_update_key = "entity_preprocessor_post_update";

export class EntityPreprocessor extends SimulationLayer {
  _dirty_flag_retain_frames = 0;
  _deferred_dirty_clear_chunks = new TypedQueue(1024, 0, Uint32Array);

  init() {
    this._pre_update_internal = this._pre_update_internal.bind(this);
    this._post_update_internal = this._post_update_internal.bind(this);
    Renderer.get().on_post_render(this.on_post_render.bind(this));
  }

  pre_update(delta_time) {
    super.pre_update(delta_time);
    profile_scope(entity_preprocessor_pre_update_key, this._pre_update_internal);
  }

  _pre_update_internal() {
    EntityManager.process_pending_deletes();
  }
  
  post_update(delta_time) {
    super.post_update(delta_time);
    profile_scope(entity_preprocessor_post_update_key, this._post_update_internal);
  }
  
  _post_update_internal() {
    EntityManager.flush_gpu_buffers();
  }
  
  on_post_render() {
    EntityManager.sync_all_buffers();
    this.clear_all_dirty();
  }

  clear_all_dirty() {
    while (this._deferred_dirty_clear_chunks.length > 0) {
      const dirty_chunk_index = this._deferred_dirty_clear_chunks.pop();
      const dirty_chunk = Chunk.get(dirty_chunk_index);
      dirty_chunk.clear_entity_dirty_flags();
    }
    for (const dirty_chunk of Chunk.dirty) {
      dirty_chunk.clear_dirty();
      this._deferred_dirty_clear_chunks.push(dirty_chunk.chunk_index);
    }
  }
}

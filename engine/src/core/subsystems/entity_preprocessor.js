import { Renderer } from "../../renderer/renderer.js";
import { Chunk } from "../ecs/solar/chunk.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { profile_scope } from "../../utility/performance.js";

const entity_preprocessor_pre_update_key = "entity_preprocessor_pre_update";
const entity_preprocessor_post_update_key = "entity_preprocessor_post_update";
const copy_gpu_to_cpu_buffers_key = "copy_gpu_to_cpu_buffers";
const max_dirty_flag_retain_frames = 16;

export class EntityPreprocessor extends SimulationLayer {
  _dirty_flag_retain_frames = 0;

  init() {
    this._pre_update_internal = this._pre_update_internal.bind(this);
    this._post_update_internal = this._post_update_internal.bind(this);
    this._on_post_render_commands = this._on_post_render_commands.bind(this);
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
    
    Renderer.get().enqueue_post_commands(
      copy_gpu_to_cpu_buffers_key,
      this._on_post_render_commands
    );
  }
  
  _on_post_render_commands(graph, frame_data, encoder) {
    EntityManager.copy_gpu_to_cpu_buffers(encoder);
  }
  
  on_post_render() {
    EntityManager.sync_all_buffers();
    this.clear_all_dirty();
  }

  clear_all_dirty() {
    for (const dirty_chunk of Chunk.dirty) {
      dirty_chunk.clear_entity_dirty_flags();
      dirty_chunk.clear_dirty();
    }
  }
}

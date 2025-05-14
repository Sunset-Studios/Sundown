import { Renderer } from "../../renderer/renderer.js";
import { EntityFlags } from "../minimal.js";
import { SimulationLayer } from "../simulation_layer.js";
import { EntityManager } from "../ecs/entity.js";
import { profile_scope } from "../../utility/performance.js";

const entity_preprocessor_pre_update_key = "entity_preprocessor_pre_update";
const entity_preprocessor_post_update_key = "entity_preprocessor_post_update";
const copy_gpu_to_cpu_buffers_key = "copy_gpu_to_cpu_buffers";

const MAX_DIRTY_FLAG_RETAIN_FRAMES = 12;

export class EntityPreprocessor extends SimulationLayer {
  entity_query = null;
  dirty_flag_retain_frames = 0;

  init() {
    this._pre_update_internal = this._pre_update_internal.bind(this);
    this._post_update_internal = this._post_update_internal.bind(this);
    this._on_post_render_commands = this._on_post_render_commands.bind(this);
    Renderer.get().on_post_render(this.on_post_render.bind(this));
    this.entity_query = EntityManager.create_query();
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
    this.clear_dirty_flags();
  }

  on_post_render() {
    EntityManager.sync_all_buffers();
  }

  clear_dirty_flags() {
    ++this.dirty_flag_retain_frames;
    if (this.dirty_flag_retain_frames >= MAX_DIRTY_FLAG_RETAIN_FRAMES) {
      this.entity_query.for_each((chunk, slot, instance_count, archetype) => {
        let changed = false;
        for (let i = 0; i < instance_count; i++) {
          if ((chunk.flags_meta[slot + i] & EntityFlags.DIRTY) !== 0) {
            changed = true;
          }
          chunk.flags_meta[slot + i] &= ~EntityFlags.DIRTY;
        }
        if (changed) {
          chunk.mark_dirty();
        }
      });
      this.dirty_flag_retain_frames = 0;
    }
  }
}

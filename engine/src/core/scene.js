import { EntityManager } from "./ecs/entity.js";
import { SimulationLayer } from "./simulation_layer.js";
import { DevConsole } from "../tools/dev_console.js";

import { EntityPreprocessor } from "./subsystems/entity_preprocessor.js";
import { TextProcessor } from "./subsystems/text_processor.js";
import { StaticMeshProcessor } from "./subsystems/static_mesh_processor.js";
import { TransformProcessor } from "./subsystems/transform_processor.js";
import { AABBEntityAdapter } from "./subsystems/aabb_entity_adapter.js";
import { AABBTreeDebugRenderer } from "./subsystems/aabb_debug_renderer.js";

import { TransformFragment } from "./ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "./ecs/fragments/visibility_fragment.js";
import { StaticMeshFragment } from "./ecs/fragments/static_mesh_fragment.js";
import { TextFragment } from "./ecs/fragments/text_fragment.js";
import { SceneGraphFragment } from "./ecs/fragments/scene_graph_fragment.js";

import { cursor } from "../ui/2d/immediate.js";
import { FontCache } from "../ui/text/font_cache.js";
import { UI3DProcessor } from "./subsystems/ui_3d_processor.js";
import { UIProcessor } from "./subsystems/ui_processor.js";
import { SharedViewBuffer } from "./shared_data.js";
import { Renderer } from "../renderer/renderer.js";

export class Scene extends SimulationLayer {
  name = "";
  dev_cursor_enabled = true;
  dev_cursor_visible = false;

  constructor(name) {
    super();
    this.name = name;
  }

  init() {
    super.init();

    Renderer.get().set_scene_id(this.name);

    this.context.current_view = SharedViewBuffer.add_view_data();

    FontCache.auto_load_fonts();

    this.setup_default_fragments();
    this.setup_default_subsystems();
  }

  cleanup() {
    SharedViewBuffer.remove_view_data(this.context.current_view);
    this.context.current_view = null;

    this.teardown_default_subsystems();

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);
    this._update_dev_cursor();
  }

  setup_default_subsystems() {
    this.add_layer(EntityPreprocessor);
    this.add_layer(UIProcessor);
    this.add_layer(TextProcessor);
    this.add_layer(StaticMeshProcessor);
    this.add_layer(TransformProcessor);
    this.add_layer(AABBEntityAdapter);
    if (__DEV__) {
      this.add_layer(AABBTreeDebugRenderer);
    }
    
    const ui_3d_processor = this.add_layer(UI3DProcessor);
    ui_3d_processor.set_scene(this);
    
    if (__DEV__) {
      const dev_console = this.add_layer(DevConsole);
      dev_console.set_scene(this);
    }
  }

  teardown_default_subsystems() {
    if (__DEV__) {
      this.remove_layer(DevConsole);
      this.remove_layer(AABBTreeDebugRenderer);
    }
    this.remove_layer(UI3DProcessor);
    this.remove_layer(AABBEntityAdapter);
    this.remove_layer(EntityPreprocessor);
    this.remove_layer(UIProcessor);
    this.remove_layer(TextProcessor);
    this.remove_layer(StaticMeshProcessor);
    this.remove_layer(TransformProcessor);
  }

  setup_default_fragments() {
    EntityManager.preinit_fragments(
      TransformFragment,
      VisibilityFragment,
      StaticMeshFragment,
      TextFragment,
      SceneGraphFragment
    );
  }

  _update_dev_cursor() {
    if (!this.dev_cursor_enabled || !this.dev_cursor_visible) return;

    cursor({
      icon: "engine/sprites/cursor.png",
      width: 25,
      height: 25,
      background_color: "transparent",
    });
  }

  set_dev_cursor_enabled(enabled) {
    this.dev_cursor_enabled = enabled;
  }

  show_dev_cursor() {
    if (!this.dev_cursor_enabled) return;
    this.dev_cursor_visible = true;
  }

  hide_dev_cursor() {
    if (!this.dev_cursor_enabled) return;
    this.dev_cursor_visible = false;
  }
}

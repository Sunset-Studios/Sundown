import { EntityManager } from "./ecs/entity.js";
import { SimulationLayer } from "./simulation_layer.js";
import { DevConsole } from "../tools/dev_console.js";

import { EntityPreprocessor } from "./subsystems/entity_preprocessor.js";
import { TextProcessor } from "./subsystems/text_processor.js";
import { StaticMeshProcessor } from "./subsystems/static_mesh_processor.js";
import { TransformProcessor } from "./subsystems/transform_processor.js";

import { TransformFragment } from "./ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "./ecs/fragments/visibility_fragment.js";
import { StaticMeshFragment } from "./ecs/fragments/static_mesh_fragment.js";
import { TextFragment } from "./ecs/fragments/text_fragment.js";
import { SceneGraphFragment } from "./ecs/fragments/scene_graph_fragment.js";

import { Element } from "../ui/2d/element.js";
import { FontCache } from "../ui/text/font_cache.js";
import { UI3DProcessor } from "./subsystems/ui_3d_processor.js";
import { UIProcessor } from "./subsystems/ui_processor.js";
import { SharedViewBuffer } from "./shared_data.js";
import { Renderer } from "../renderer/renderer.js";
import { Cursor } from "../ui/2d/cursor.js";

export class Scene extends SimulationLayer {
  name = "";
  dev_cursor = null;
  dev_cursor_enabled = true;

  constructor(name) {
    super();
    this.name = name;
  }

  init() {
    super.init();

    Renderer.get().set_scene_id(this.name);

    this.context.current_view = SharedViewBuffer.add_view_data();
    Element.new_view_root(this.context.current_view);

    FontCache.auto_load_fonts();

    this.setup_default_fragments();
    this.setup_default_subsystems();
    this.setup_default_ui();
  }

  update(delta_time) {
    super.update(delta_time);
  }

  setup_default_subsystems() {
    if (__DEV__) {
      const dev_console = this.add_layer(DevConsole);
      dev_console.set_scene(this);
    }

    this.add_layer(EntityPreprocessor);
    this.add_layer(UIProcessor);
    this.add_layer(TextProcessor);
    this.add_layer(StaticMeshProcessor);
    this.add_layer(TransformProcessor);

    const ui_3d_processor = this.add_layer(UI3DProcessor);
    ui_3d_processor.set_scene(this);
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

  setup_default_ui() {
    this.dev_cursor = Cursor.create("dev_cursor", {
      icon: "engine/sprites/cursor.png",
      style: {
        position: "absolute",
        width: "25px",
        height: "25px",
        background: "transparent",
        border: "none",
      },
    });

    const view_root = Element.get_view_root();
    view_root.add_child(this.dev_cursor);
    this.dev_cursor.is_visible = false;
  }

  set_dev_cursor_enabled(enabled) {
    this.dev_cursor_enabled = enabled;
  }

  show_dev_cursor() {
    if (!this.dev_cursor_enabled) return;
    this.dev_cursor.is_visible = true;
  }

  hide_dev_cursor() {
    if (!this.dev_cursor_enabled) return;
    this.dev_cursor.is_visible = false;
  }
}

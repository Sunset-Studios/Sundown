import { EntityManager } from "./ecs/entity.js";
import { SimulationLayer } from "./simulation_layer.js";

import { EntityPreprocessor } from "./subsystems/entity_preprocessor.js";
import { TextProcessor } from "./subsystems/text_processor.js";
import { StaticMeshProcessor } from "./subsystems/static_mesh_processor.js";
import { TransformProcessor } from "./subsystems/transform_processor.js";

import { TransformFragment } from "./ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "./ecs/fragments/visibility_fragment.js";
import { StaticMeshFragment } from "./ecs/fragments/static_mesh_fragment.js";
import { TextFragment } from "./ecs/fragments/text_fragment.js";
import { SceneGraphFragment } from "./ecs/fragments/scene_graph_fragment.js";

import { FontCache } from "../ui/text/font_cache.js";
import { UI3DProcessor } from "./subsystems/ui_3d_processor.js";
import { UIProcessor } from "./subsystems/ui_processor.js";
import { SharedViewBuffer } from "./shared_data.js";
import { Renderer } from "../renderer/renderer.js";

export class Scene extends SimulationLayer {
  name = "";

  constructor(name) {
    super();
    this.name = name;
  }

  init() {
    super.init();

    this.context.current_view = SharedViewBuffer.add_view_data();

    FontCache.auto_load_fonts();

    this.setup_default_fragments();
    this.setup_default_subsystems();
  }

  update(delta_time) {
    super.update(delta_time);
  }

  setup_default_subsystems() {
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

  set_ui_root(ui_root) {
    this.get_layer(UIProcessor).set_ui_root(ui_root);
    Renderer.get().canvas.after(ui_root.dom);
  }
}

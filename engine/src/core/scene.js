import { SimulationLayer } from "./simulation_layer.js";
import { TextProcessor } from "./subsystems/text_processor.js";
import { StaticMeshProcessor } from "./subsystems/static_mesh_processor.js";
import { TransformProcessor } from "./subsystems/transform_processor.js";
import { FontCache } from "../ui/text/font_cache.js";
import { UI3DProcessor } from "./subsystems/ui_3d_processor.js";
import { UIProcessor } from "./subsystems/ui_processor.js";
import { SharedViewBuffer } from "./shared_data.js";
import { Renderer } from "../renderer/renderer.js";
import { profile_scope } from "../utility/performance.js";

export class Scene extends SimulationLayer {
  name = "";

  constructor(name) {
    super();
    this.name = name;
  }

  init() {
    super.init();

    this.context.current_view = SharedViewBuffer.get().add_view_data(
      Renderer.get().graphics_context
    );

    this.setup_default_subsystems();

    FontCache.auto_load_fonts();
  }

  update(delta_time) {
    super.update(delta_time);

    profile_scope("Scene.update", () => {
      this.context.entity_manager.process_query_changes();
    });
  }

  setup_default_subsystems() {
    this.add_layer(UIProcessor);
    this.add_layer(TextProcessor);
    this.add_layer(StaticMeshProcessor);
    this.add_layer(TransformProcessor);

    const ui_3d_processor = this.add_layer(UI3DProcessor);
    ui_3d_processor.set_scene(this);
  }

  get_entity_count() {
    return this.context.entity_manager.get_entity_count();
  }

  create_entity(refresh_entity_data = true) {
    return this.context.entity_manager.create_entity(refresh_entity_data);
  }

  delete_entity(entity, refresh_entity_data = true) {
    this.context.entity_manager.delete_entity(entity, refresh_entity_data);
  }

  duplicate_entity(entity, refresh_entity_data = true) {
    return this.context.entity_manager.duplicate_entity(entity, refresh_entity_data);
  }

  add_fragment(entity, FragmentType, refresh_entity_data = true) {
    return this.context.entity_manager.add_fragment(entity, FragmentType, refresh_entity_data);
  }

  remove_fragment(entity, FragmentType, refresh_entity_data = true) {
    this.context.entity_manager.remove_fragment(entity, FragmentType, refresh_entity_data);
  }

  add_tag(entity, Tag, refresh_entity_data = true) {
    this.context.entity_manager.add_tag(entity, Tag, refresh_entity_data);
  }

  remove_tag(entity, Tag, refresh_entity_data = true) {
    this.context.entity_manager.remove_tag(entity, Tag, refresh_entity_data);
  }

  get_fragment(entity, FragmentType) {
    return this.context.entity_manager.get_fragment(entity, FragmentType);
  }
  
  has_fragment(entity, FragmentType) {
    return this.context.entity_manager.has_fragment(entity, FragmentType);
  }

  refresh_entity_queries(params = {}) {
    this.context.entity_manager.update_queries(params);
  }

  set_ui_root(ui_root) {
    this.get_layer(UIProcessor).set_ui_root(ui_root);
    const canvas = Renderer.get().graphics_context.canvas;
    canvas.after(ui_root.dom);
  }
}

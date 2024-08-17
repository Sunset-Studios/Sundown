import { SimulationLayer } from "./simulation_layer.js";
import { StaticMeshProcessor } from "./subsystems/static_mesh_processor.js";
import { TransformProcessor } from "./subsystems/transform_processor.js";
import { SharedViewBuffer } from "./shared_data.js";
import { Renderer } from "../renderer/renderer.js";

export class Scene extends SimulationLayer {
  name = "";
  ui_root = null;

  constructor(name) {
    super();
    this.name = name;
  }

  async init() {
    super.init();

    this.context.current_view = SharedViewBuffer.get().add_view_data(
      Renderer.get().graphics_context
    );

    this.setup_default_subsystems();
  }

  update(delta_time) {
    super.update(delta_time);

    performance.mark("scene_update");

    this.context.entity_manager.process_query_changes();

    if (this.ui_root) {
      this.ui_root.update(delta_time);
    }
  }

  setup_default_subsystems() {
    this.add_layer(StaticMeshProcessor);
    this.add_layer(TransformProcessor);
  }

  create_entity(refresh_entity_data = true) {
    return this.context.entity_manager.create_entity(refresh_entity_data);
  }

  delete_entity(entity, refresh_entity_data = true) {
    this.context.entity_manager.delete_entity(entity, refresh_entity_data);
  }

  add_fragment(entity, FragmentType, data, refresh_entity_data = true) {
    this.context.entity_manager.add_fragment(entity, FragmentType, data, refresh_entity_data);
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

  update_fragment(entity, FragmentType, data, refresh_entity_data = true) {
    this.context.entity_manager.update_fragment(entity, FragmentType, data, refresh_entity_data);
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
    this.ui_root = ui_root;
    const canvas = Renderer.get().graphics_context.canvas;
    canvas.after(ui_root.dom);
  }
}

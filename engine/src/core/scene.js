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

  async init(parent_context) {
    super.init(this.context);

    this.context.current_view = SharedViewBuffer.get().add_view_data(
      Renderer.get().graphics_context
    );

    this.setup_default_subsystems();
  }

  update(delta_time, parent_context) {
    super.update(delta_time, this.context);

    performance.mark("scene_update");
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

  update_fragment(entity, FragmentType, data, refresh_entity_data = true) {
    this.context.entity_manager.update_fragment(entity, FragmentType, data, refresh_entity_data);
  }

  get_fragment(entity, FragmentType) {
    return this.context.entity_manager.get_fragment(entity, FragmentType);
  }
  
  has_fragment(entity, FragmentType) {
    return this.context.entity_manager.has_fragment(entity, FragmentType);
  }

  refresh_entity_queries() {
    this.context.entity_manager.update_queries();
  }

  set_ui_root(ui_root) {
    this.ui_root = ui_root;
    const canvas = Renderer.get().graphics_context.canvas;
    canvas.after(ui_root.dom);
  }
}

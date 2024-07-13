import { SimulationLayer } from "./simulation_layer.js";
import { FreeformArcballControlProcessor } from "./subsystems/freeform_arcball_control_processor.js";
import { StaticMeshProcessor } from "./subsystems/static_mesh_processor.js";
import { TransformProcessor } from "./subsystems/transform_processor.js";
import { SharedViewBuffer } from "./shared_data.js";
import { Renderer } from "../renderer/renderer.js";

export class Scene extends SimulationLayer {
  name = "";
  sphere_mesh = null;

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
    this.add_layer(FreeformArcballControlProcessor);
    this.add_layer(StaticMeshProcessor);
    this.add_layer(TransformProcessor);
  }

  create_entity() {
    return this.context.entity_manager.create_entity();
  }

  delete_entity(entity) {
    this.context.entity_manager.delete_entity(entity);
  }

  add_fragment(entity, FragmentType, data) {
    this.context.entity_manager.add_fragment(entity, FragmentType, data);
  }

  remove_fragment(entity, FragmentType) {
    this.context.entity_manager.remove_fragment(entity, FragmentType);
  }

  update_fragment(entity, FragmentType, data) {
    this.context.entity_manager.update_fragment(entity, FragmentType, data);
  }

  get_fragment(entity, FragmentType) {
    return this.context.entity_manager.get_fragment(entity, FragmentType);
  }
}

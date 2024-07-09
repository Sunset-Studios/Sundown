import { SimulationLayer } from "@/core/simulation_layer";
import { FreeformViewControlProcessor } from "@/core/subsystems/freeform_view_control_processor";
import { StaticMeshProcessor } from "@/core/subsystems/static_mesh_processor";
import { TransformProcessor } from "@/core/subsystems/transform_processor";
import { SharedViewBuffer } from "@/core/shared_data";
import { Renderer } from "@/renderer/renderer";

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
    this.add_layer(FreeformViewControlProcessor);
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

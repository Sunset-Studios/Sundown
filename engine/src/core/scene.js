import { EntityManager } from "./ecs/entity.js";
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

    this.context.current_view = SharedViewBuffer.add_view_data();

    this.setup_default_subsystems();

    FontCache.auto_load_fonts();
  }

  update(delta_time) {
    super.update(delta_time);

    profile_scope("Scene.update", () => {
      EntityManager.process_query_changes();
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
    return EntityManager.get_entity_count();
  }

  get_entity_instance_count(entity) {
    return EntityManager.get_entity_instance_count(entity);
  }

  change_entity_instance_count(entity, instance_count) {
    return EntityManager.change_entity_instance_count(entity, instance_count);
  }

  reserve_entities(size) {
    EntityManager.reserve_entities(size);
  }

  create_entity(refresh_entity_data = true) {
    return EntityManager.create_entity(refresh_entity_data);
  }

  delete_entity(entity, refresh_entity_data = true) {
    EntityManager.delete_entity(entity, refresh_entity_data);
  }

  duplicate_entity(entity, refresh_entity_data = true, instance = 0) {
    return EntityManager.duplicate_entity(entity, refresh_entity_data, instance);
  }

  add_fragment(entity, FragmentType, refresh_entity_data = true) {
    return EntityManager.add_fragment(entity, FragmentType, refresh_entity_data);
  }

  remove_fragment(entity, FragmentType, refresh_entity_data = true) {
    EntityManager.remove_fragment(entity, FragmentType, refresh_entity_data);
  }

  add_tag(entity, Tag, refresh_entity_data = true) {
    EntityManager.add_tag(entity, Tag, refresh_entity_data);
  }

  remove_tag(entity, Tag, refresh_entity_data = true) {
    EntityManager.remove_tag(entity, Tag, refresh_entity_data);
  }

  get_fragment(entity, FragmentType, instance = 0) {
    return EntityManager.get_fragment(entity, FragmentType, instance);
  }
  
  has_fragment(entity, FragmentType) {
    return EntityManager.has_fragment(entity, FragmentType);
  }

  refresh_entities() {
    EntityManager.refresh_entities();
  }

  set_ui_root(ui_root) {
    this.get_layer(UIProcessor).set_ui_root(ui_root);
    Renderer.get().canvas.after(ui_root.dom);
  }
}

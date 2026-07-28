import { SimulationLayer } from "./simulation_layer.js";
import { DevConsole } from "../tools/dev_console.js";
import { GLTFSceneLoader } from "../renderer/gltf_scene_loader.js";

import { LightViewProcessor } from "./subsystems/light_view_processor.js";
import { EntityPreprocessor } from "./subsystems/entity_preprocessor.js";
import { StaticMeshProcessor } from "./subsystems/static_mesh_processor.js";
import { TransformProcessor } from "./subsystems/transform_processor.js";
import { BVHEntityAdapter } from "./subsystems/bvh_entity_adapter.js";
import { BVHDebugRenderer } from "./subsystems/bvh_debug_renderer.js";
import { MeshBLASProcessor } from "../acceleration/mesh_blas_processor.js";

import { cursor } from "../ui/2d/immediate.js";
import { FontCache } from "../ui/text/font_cache.js";
import { ViewProcessor } from "./subsystems/view_processor.js";
import { UI3DRenderProcessor } from "./subsystems/ui_3d_render_processor.js";
import { UIProcessor } from "./subsystems/ui_processor.js";
import { SharedViewBuffer, SharedFrameInfoBuffer } from "./shared_data.js";
import { Renderer } from "../renderer/renderer.js";
import { ProjectContext } from "./project_context.js";
import { SceneDataLoader } from "../streaming/scene_data_loader.js";
import { error } from "../utility/logging.js";

export class Scene extends SimulationLayer {
  name = "";
  scene_data_loader = null;
  scene_data_ready = Promise.resolve(null);
  dev_cursor_enabled = true;
  dev_cursor_visible = false;

  constructor(name) {
    super();
    this.name = String(name || this.constructor.name);
  }

  init() {
    super.init();

    Renderer.get().set_scene_id(this.name);
    this.scene_data_loader = new SceneDataLoader(this.name, {
      project: ProjectContext.get_active(),
    });
    this.scene_data_loader.connect_handlers(Renderer.get().get_scene_data_handlers());
    this.scene_data_ready = this.scene_data_loader.load({
      optional: true,
    });
    void this.scene_data_ready.catch((scene_data_error) => {
      error(`Failed to load scene data for '${this.name}':`, scene_data_error);
    });

    const view = SharedViewBuffer.add_view_data();
    view.renderable_state = 1;

    this.context.current_view = view.get_index();
    SharedFrameInfoBuffer.set_view_index(this.context.current_view);

    FontCache.auto_load_fonts();

    this.setup_default_subsystems();
  }

  cleanup() {
    this.scene_data_loader?.dispose();
    this.scene_data_loader = null;
    this.scene_data_ready = Promise.resolve(null);

    SharedViewBuffer.remove_view_data(this.context.current_view);
    this.context.current_view = null;

    this.teardown_default_subsystems();

    super.cleanup();
  }

  when_scene_data_ready() {
    return this.scene_data_ready;
  }

  register_scene_data_handler(handler) {
    if (!this.scene_data_loader) {
      throw new Error(
        `Scene data handlers for '${this.name}' can only be registered while the scene is active.`
      );
    }
    return this.scene_data_loader.register_handler(handler);
  }

  async save_scene_data(options = {}) {
    const scene_data_loader = this.scene_data_loader;
    if (!scene_data_loader) {
      throw new Error(`Scene data for '${this.name}' is unavailable before init or after cleanup.`);
    }
    await this.scene_data_ready;
    if (scene_data_loader !== this.scene_data_loader) {
      throw new Error(`Scene '${this.name}' was unloaded before its data could be saved.`);
    }
    return await scene_data_loader.save_to_disk(options);
  }

  update(delta_time) {
    super.update(delta_time);
    this._update_dev_cursor();
  }

  setup_default_subsystems() {
    const view_processor = this.add_layer(ViewProcessor);
    view_processor.set_scene(this);

    this.add_layer(LightViewProcessor);
    this.add_layer(UIProcessor);
    this.add_layer(StaticMeshProcessor);
    this.add_layer(TransformProcessor);
    this.add_layer(BVHEntityAdapter);
    this.add_layer(MeshBLASProcessor);
    if (__DEV__) {
      this.add_layer(BVHDebugRenderer);
    }

    const ui_3d_render_processor = this.add_layer(UI3DRenderProcessor);
    ui_3d_render_processor.set_scene(this);

    if (__DEV__) {
      const dev_console = this.add_layer(DevConsole);
      dev_console.set_scene(this);
    }

    this.add_layer(EntityPreprocessor);
  }

  teardown_default_subsystems() {
    if (__DEV__) {
      this.remove_layer(DevConsole);
      this.remove_layer(MeshBLASProcessor);
    }
    this.remove_layer(BVHDebugRenderer);
    this.remove_layer(BVHEntityAdapter);
    this.remove_layer(EntityPreprocessor);
    this.remove_layer(UI3DRenderProcessor);
    this.remove_layer(UIProcessor);
    this.remove_layer(StaticMeshProcessor);
    this.remove_layer(TransformProcessor);
    this.remove_layer(LightViewProcessor);
    this.remove_layer(ViewProcessor);
  }

  _update_dev_cursor() {
    if (!this.dev_cursor_enabled || !this.dev_cursor_visible) return;

    cursor({
      icon: "engine/sprites/cursor.png",
      width: 25,
      height: 25,
      background_color: "transparent",
      z_order: 1,
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

  load_gltf_scene(
    gltf_path,
    position = [0, 0, 0],
    rotation = [0, 0, 0, 1],
    scale = [1, 1, 1],
    scene_index = null,
    parent_entity = null,
    callback = null,
    options = {}
  ) {
    return GLTFSceneLoader.load_scene(
      gltf_path,
      position,
      rotation,
      scale,
      scene_index,
      parent_entity,
      callback,
      options
    );
  }
}

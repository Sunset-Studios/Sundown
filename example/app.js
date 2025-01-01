import { Renderer } from "../engine/src/renderer/renderer.js";
import { DeferredShadingStrategy } from "../engine/src/renderer/strategies/deferred_shading.js";
import { Material } from "../engine/src/renderer/material.js";
import SimulationCore from "../engine/src/core/simulation_core.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { Scene } from "../engine/src/core/scene.js";
import application_state from "../engine/src/core/application_state.js";
import { ComputeTaskQueue } from "../engine/src/renderer/compute_task_queue.js";
import { StaticMeshFragment } from "../engine/src/core/ecs/fragments/static_mesh_fragment.js";
import { TransformFragment } from "../engine/src/core/ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "../engine/src/core/ecs/fragments/visibility_fragment.js";
import { SceneGraphFragment } from "../engine/src/core/ecs/fragments/scene_graph_fragment.js";
import { FreeformArcballControlProcessor } from "../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { TextFragment } from "../engine/src/core/ecs/fragments/text_fragment.js";
import { LightType } from "../engine/src/core/minimal.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { Name } from "../engine/src/utility/names.js";
import { SharedEnvironmentMapData } from "../engine/src/core/shared_data.js";
import { profile_scope } from "../engine/src/utility/performance.js";
import { frame_runner } from "../engine/src/utility/frame_runner.js";
import { spawn_mesh_entity } from "../engine/src/core/ecs/entity_utils.js";
import { quat } from "gl-matrix";
import { FontCache } from "../engine/src/ui/text/font_cache.js";

export class TestScene extends Scene {
  async init(parent_context) {
    super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skybox for this scene.
    await SharedEnvironmentMapData.get().add_skybox(
      Renderer.get().graphics_context,
      "default_scene_skybox",
      [
        "engine/textures/gradientbox/px.png",
        "engine/textures/gradientbox/nx.png",
        "engine/textures/gradientbox/ny.png",
        "engine/textures/gradientbox/py.png",
        "engine/textures/gradientbox/pz.png",
        "engine/textures/gradientbox/nz.png",
      ]
    );

    // Create a light and add it to the scene
    const light_entity = this.create_entity();

    // Add a light fragment to the light entity
    const light_fragment_view = this.add_fragment(light_entity, LightFragment, false);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color.r = 1;
    light_fragment_view.color.g = 1;
    light_fragment_view.color.b = 1;
    light_fragment_view.intensity = 5;
    light_fragment_view.position.x = 50;
    light_fragment_view.position.y = 100;
    light_fragment_view.position.z = 50;
    light_fragment_view.active = true;

    // Create some text
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    const text_entity = spawn_mesh_entity(
      this,
      {x: 0, y: 15.0, z: 0},
      {x: 0, y: 0, z: 0, w: 1},
      {x: 1, y: 1, z: 1},
      Mesh.quad(Renderer.get().graphics_context),
      font_object.material
    );
    const text_fragment_view = this.add_fragment(text_entity, TextFragment, false);
    text_fragment_view.font = font_id;
    text_fragment_view.text = "Hello, World!";
    text_fragment_view.font_size = 24;

    // Create a sphere mesh and add it to the scene
    const mesh = await Mesh.from_gltf(
      Renderer.get().graphics_context,
      "engine/models/sphere/sphere.gltf"
    );

    // Create a default material
    const default_material_id = Material.create("MyMaterial", "StandardMaterial");

    // Create a 3D grid of sphere entities
    const grid_size = 100; // 100x100x10 grid
    const grid_layers = 10;
    const spacing = 2; // 2 units apart

    for (let x = 0; x < grid_size; x++) {
      for (let z = 0; z < grid_size; z++) {
        for (let y = 0; y < grid_layers; y++) {
          let entity = this.create_entity(false /* refresh_entity_queries */);

          // Add a static mesh fragment to the sphere entity
          const static_mesh_fragment_view = this.add_fragment(entity, StaticMeshFragment, false);
          static_mesh_fragment_view.mesh = BigInt(Name.from(mesh.name));
          static_mesh_fragment_view.material_slots = [default_material_id];
          static_mesh_fragment_view.instance_count = BigInt(1);

          const rotation = quat.fromValues(0, 0, 0, 1);

          // Add a transform fragment to the sphere entity
          const transform_fragment_view = this.add_fragment(entity, TransformFragment, false);
          transform_fragment_view.position = [
            (x - Math.floor(grid_size / 2)) * spacing,
            (y - Math.floor(grid_layers / 2)) * spacing,
            (z - Math.floor(grid_size / 2)) * spacing,
          ];
          transform_fragment_view.rotation = rotation;
          transform_fragment_view.scale = [0.5, 0.5, 0.5];

          // Add a visibility fragment to the sphere entity
          const visibility_fragment_view = this.add_fragment(entity, VisibilityFragment, false);
          visibility_fragment_view.visible = 1;

          // Add a scene graph fragment to the sphere entity
          const scene_graph_fragment_view = this.add_fragment(entity, SceneGraphFragment, false);
          scene_graph_fragment_view.parent = null;
        }
      }
    }

    this.refresh_entity_queries();
  }

  update(delta_time) {
    super.update(delta_time);

    const transforms = EntityManager.get().get_fragment_array(TransformFragment);

    ComputeTaskQueue.get().new_task(
      "ripples",
      "effects/transform_ripples.wgsl",
      [transforms.position_buffer, transforms.dirty_flags_buffer],
      [transforms.position_buffer, transforms.dirty_flags_buffer],
      Math.ceil(transforms.dirty.length / 256)
    );
  }
}

async function init() {
  application_state.is_running = true;

  // Initialize input provider
  const input_provider = InputProvider.get();
  await SimulationCore.get().register_simulation_layer(input_provider);
  input_provider.push_context(InputProvider.default_context());

  // Initialize renderer with document canvas
  const canvas = document.getElementById("gpu-canvas");
  await Renderer.get().setup(canvas, DeferredShadingStrategy);

  // Create a test scene and register it with the simulation system
  const scene = new TestScene("TestScene");
  await SimulationCore.get().register_simulation_layer(scene);
}

function simulate(delta_time) {
  if (application_state.is_running) {
    profile_scope("frame_loop", () => {
      SimulationCore.get().update(delta_time);
      Renderer.get().render(delta_time);
    });
  }
}

function run() {
  frame_runner(simulate, 60);
}

(async () => {
  await init();
  run();
})();

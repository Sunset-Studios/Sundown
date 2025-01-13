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
import { FontCache } from "../engine/src/ui/text/font_cache.js";

export class TestScene extends Scene {
  async init(parent_context) {
    super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skybox for this scene.
    await SharedEnvironmentMapData.add_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.png",
      "engine/textures/gradientbox/nx.png",
      "engine/textures/gradientbox/ny.png",
      "engine/textures/gradientbox/py.png",
      "engine/textures/gradientbox/pz.png",
      "engine/textures/gradientbox/nz.png",
    ]);

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity();

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.add_fragment(light_entity, LightFragment, false);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color.r = 1;
    light_fragment_view.color.g = 1;
    light_fragment_view.color.b = 1;
    light_fragment_view.intensity = 5;
    light_fragment_view.position.x = 50;
    light_fragment_view.position.y = 0;
    light_fragment_view.position.z = 50;
    light_fragment_view.active = true;

    // Create a sphere mesh and add it to the scene
    const mesh = await Mesh.from_gltf(
      "engine/models/sphere/sphere.gltf"
    );

    // Create a default material
    const default_material_id = Material.create("MyMaterial", "StandardMaterial");

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    // Create a 3D grid of sphere entities
    const grid_size = 100; // 100x100x10 grid
    const grid_layers = 10;
    const spacing = 5; // 2 units apart

    EntityManager.reserve_entities(grid_size * grid_size * grid_layers + 2);

    for (let x = 0; x < grid_size; x++) {
      for (let z = 0; z < grid_size; z++) {
        for (let y = 0; y < grid_layers; y++) {
          const sphere = spawn_mesh_entity(
            {
              x: (x - Math.floor(grid_size / 2)) * spacing,
              y: (y - Math.floor(grid_layers / 2)) * spacing,
              z: (z - Math.floor(grid_size / 2)) * spacing,
            },
            { x: 0, y: 0, z: 0, w: 1 },
            { x: 1.0, y: 1.0, z: 1.0 },
            mesh,
            default_material_id,
            null /* parent */,
            [] /* children */,
            true /* start_visible */,
            false /* refresh_entities */
          );
        }
      }
    }

    const text_entity = spawn_mesh_entity(
      {
        x: 0,
        y: 25,
        z: 0,
      },
      { x: 0, y: 0, z: 0, w: 1 },
      { x: 0.5, y: 0.5, z: 0.5 },
      Mesh.quad(),
      font_object.material,
      null /* parent */,
      [] /* children */,
      true /* start_visible */,
      false /* refresh_entities */
    );
    const text_fragment_view = EntityManager.add_fragment(
      text_entity,
      TextFragment,
      false /* refresh_entities */
    );
    text_fragment_view.font = font_id;
    text_fragment_view.text = "Sundown Engine";
    text_fragment_view.font_size = 32;

    EntityManager.refresh_entities();
  }

  update(delta_time) {
    super.update(delta_time);

    const transforms = EntityManager.get_fragment_array(TransformFragment);

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
  await Renderer.create(canvas, DeferredShadingStrategy, { pointer_lock: true });

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

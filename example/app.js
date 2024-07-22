import { Renderer } from "../engine/src/renderer/renderer.js";
import { Material, MaterialTemplate } from "../engine/src/renderer/material.js";
import { Texture } from "../engine/src/renderer/texture.js";
import SimulationCore from "../engine/src/core/simulation_core.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { Scene } from "../engine/src/core/scene.js";
import application_state from "../engine/src/core/application_state.js";
import { StaticMeshFragment } from "../engine/src/core/ecs/fragments/static_mesh_fragment.js";
import { TransformFragment } from "../engine/src/core/ecs/fragments/transform_fragment.js";
import {
  LightFragment,
  LightType,
} from "../engine/src/core/ecs/fragments/light_fragment.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { Name } from "../engine/src/utility/names.js";
import { SharedEnvironmentMapData } from "../engine/src/core/shared_data.js";
import { profile_scope } from "../engine/src/utility/performance.js";

async function init() {
  application_state.is_running = true;

  // Initialize input provider
  const input_provider = InputProvider.get();
  await SimulationCore.get().register_simulation_layer(input_provider);
  input_provider.push_context(InputProvider.default_context());

  // Initialize renderer with document canvas
  const canvas = document.getElementById("gpu-canvas");
  await Renderer.get().setup(canvas);

  // Initialize scene
  {
    // Create a test scene and register it with the simulation system
    const scene = new Scene("TestScene");
    await SimulationCore.get().register_simulation_layer(scene);

    // Set the skybox for this scene.
    await SharedEnvironmentMapData.get().add_skybox(
      Renderer.get().graphics_context,
      "test_scene_skybox",
      [
        "engine/textures/spacebox/px.png",
        "engine/textures/spacebox/nx.png",
        "engine/textures/spacebox/ny.png",
        "engine/textures/spacebox/py.png",
        "engine/textures/spacebox/pz.png",
        "engine/textures/spacebox/nz.png",
      ]
    );

    // Create a light and add it to the scene
    const light_entity = scene.create_entity();

    // Add a light fragment to the light entity
    scene.add_fragment(light_entity, LightFragment, {
      type: LightType.DIRECTIONAL,
      color: { r: 1, g: 1, b: 1 },
      intensity: 3,
      position: { x: 50, y: 100, z: 50 },
    });

    // Create a sphere mesh and add it to the scene
    const mesh = await Mesh.from_gltf(
      Renderer.get().graphics_context,
      "engine/models/sphere/sphere.gltf"
    );

    // Create a default material
    const default_material = Material.create("StandardMaterial");
    const default_material_id = default_material.get_state_hash();

    // Create a grid of sphere entities
    const grid_size = 100; // 5x5 grid
    const spacing = 2; // 2 units apart

    for (let x = 0; x < grid_size; x++) {
      for (let z = 0; z < grid_size; z++) {
        // Create a sphere entity and add it to the scene
        const sphere_entity = scene.create_entity(
          false /* refresh_entity_queries */
        );

        // Add a static mesh fragment to the sphere entity
        scene.add_fragment(
          sphere_entity,
          StaticMeshFragment,
          {
            mesh: BigInt(Name.from(mesh.name)),
            material_slots: [default_material_id],
          },
          false /* refresh_entity_queries */
        );

        // Add a transform fragment to the sphere entity
        scene.add_fragment(
          sphere_entity,
          TransformFragment,
          {
            position: {
              x: (x - Math.floor(grid_size / 2)) * spacing,
              y: 0,
              z: (z - Math.floor(grid_size / 2)) * spacing,
            },
            rotation: { x: -90, y: 90, z: 0 },
            scale: { x: 0.5, y: 0.5, z: 0.5 },
          },
          false /* refresh_entity_queries */
        );
      }
    }

    scene.refresh_entity_queries();
  }
}

function run() {
  function simulate() {
    if (application_state.is_running) {
      profile_scope("frame_loop", () => {
        SimulationCore.get().update();
        Renderer.get().render();
      });
      requestAnimationFrame(simulate);
    }
  }
  requestAnimationFrame(simulate);
}

(async () => {
  await init();
  run();
})();

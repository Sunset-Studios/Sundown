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
import { FreeformArcballControlProcessor } from "../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import {
  LightFragment,
} from "../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../engine/src/core/minimal.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { Name } from "../engine/src/utility/names.js";
import { SharedEnvironmentMapData } from "../engine/src/core/shared_data.js";
import { profile_scope } from "../engine/src/utility/performance.js";
import { frame_runner } from "../engine/src/utility/frame_runner.js";
import { quat } from "gl-matrix";

export class TestScene extends Scene {
  async init(parent_context) {
    await super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(
      FreeformArcballControlProcessor
    );
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
    this.add_fragment(light_entity, LightFragment, {
      type: LightType.DIRECTIONAL,
      color: { r: 1, g: 1, b: 1 },
      intensity: 5,
      position: { x: 50, y: 100, z: 50 },
    });

    // Create a sphere mesh and add it to the scene
    const mesh = await Mesh.from_gltf(
      Renderer.get().graphics_context,
      "engine/models/cube/cube.gltf"
    );

    // Create a default material
    const default_material_id = Material.create(
      "MyMaterial",
      "StandardMaterial"
    );

    // Create a 3D grid of sphere entities
    const grid_size = 100; // 100x100x10 grid
    const grid_layers = 10;
    const spacing = 2; // 2 units apart

    for (let x = 0; x < grid_size; x++) {
      for (let z = 0; z < grid_size; z++) {
        for (let y = 0; y < grid_layers; y++) {
          let entity = this.create_entity(false /* refresh_entity_queries */);

          // Add a static mesh fragment to the sphere entity
          this.add_fragment(
            entity,
            StaticMeshFragment,
            {
              mesh: BigInt(Name.from(mesh.name)),
              material_slots: [default_material_id],
              instance_count: BigInt(1),
            },
            false /* refresh_entity_queries */
          );

          const rotation = quat.fromValues(0, 0, 0, 1);

          // Add a transform fragment to the sphere entity
          this.add_fragment(
            entity,
            TransformFragment,
            {
              position: {
                x: (x - Math.floor(grid_size / 2)) * spacing,
                y: (y - Math.floor(grid_layers / 2)) * spacing,
                z: (z - Math.floor(grid_size / 2)) * spacing,
              },
              rotation: { x: rotation[0], y: rotation[1], z: rotation[2], w: rotation[3] },
              scale: { x: 0.5, y: 0.5, z: 0.5 },
            },
            false /* refresh_entity_queries */
          );
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

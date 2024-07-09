import { Renderer } from "@/renderer/renderer.js";
import SimulationCore from "@/core/simulation_core.js";
import { InputProvider } from "@/input/input_provider.js";
import { Scene } from "@/core/scene.js";
import application_state from "@/core/application_state.js";
import { StaticMeshFragment } from "@/core/ecs/fragments/static_mesh_fragment.js";
import { TransformFragment } from "@/core/ecs/fragments/transform_fragment.js";
import { Mesh } from "@/renderer/mesh.js";
import { Name } from "@/utility/names.js";
import { vec3, quat } from "gl-matrix";

async function init() {
  application_state.is_running = true;

  const input_provider = InputProvider.get();
  await SimulationCore.get().register_simulation_layer(input_provider);
  input_provider.push_context(InputProvider.default_context());

  const canvas = document.getElementById("gpu-canvas");
  await Renderer.get().setup(canvas);

  {
    const scene = new Scene("TestScene");
    await SimulationCore.get().register_simulation_layer(scene);
  
    const sphere_mesh = await Mesh.from_gltf(
      Renderer.get().graphics_context,
      'models/sphere/sphere.gltf',
    );
    const sphere_entity = scene.create_entity();
    scene.add_fragment(sphere_entity, StaticMeshFragment, {
      mesh: BigInt(Name.from('models/sphere/sphere.gltf')),
    });
    scene.add_fragment(sphere_entity, TransformFragment, {
      position: { x: 0, y: 0, z: 0 },
      rotation: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1, z: 1 },
    });
  }
}

function run() {
  function simulate() {
    if (application_state.is_running) {
      performance.mark("frame_start");

      SimulationCore.get().update();

      Renderer.get().render();

      requestAnimationFrame(simulate);
    }
  }
  requestAnimationFrame(simulate);
}

await init();
run();

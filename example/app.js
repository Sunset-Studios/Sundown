import SimulationCore from "../engine/src/core/simulation_core.js";
import { SimulationLayer } from "../engine/src/core/simulation_layer.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { InputKey } from '../engine/src/input/input_types.js'
import { PostProcessStack } from "../engine/src/renderer/post_process_stack.js";
import { Material } from "../engine/src/renderer/material.js";
import { Simulator } from "../engine/src/core/simulator.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { Scene } from "../engine/src/core/scene.js";
import { ComputeTaskQueue } from "../engine/src/renderer/compute_task_queue.js";
import { TransformFragment } from "../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { TextFragment } from "../engine/src/core/ecs/fragments/text_fragment.js";
import { LightType } from "../engine/src/core/minimal.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { Name } from "../engine/src/utility/names.js";
import { SharedEnvironmentMapData, SharedViewBuffer } from "../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../engine/src/core/ecs/entity_utils.js";
import { FontCache } from "../engine/src/ui/text/font_cache.js";

import { MasterMind } from "../engine/src/ml/mastermind.js";
import { NeuralModel } from "../engine/src/ml/neural_model.js";
import { FullyConnected } from "../engine/src/ml/layers/fully_connected.js";
import { ReLU } from "../engine/src/ml/layers/relu.js";
import { Tanh } from "../engine/src/ml/layers/tanh.js";
import { MSELoss } from "../engine/src/ml/layers/mse_loss.js";
import { Tensor, TensorInitializer } from "../engine/src/ml/tensor.js";
import { Adam } from "../engine/src/ml/optimizers/adam.js";

import { profile_scope } from "../engine/src/utility/performance.js";

export class RenderingScene extends Scene {
  scene_entities = [];

  init(parent_context) {
    super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    SharedViewBuffer.set_view_data(0, {
      position: [-1.0, 22.0, 26.0],
      rotation: [-0.00061309, 0.9948077, -0.10095515, -0.00604141],
    });

    // Set the skybox for this scene.
    SharedEnvironmentMapData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.png",
      "engine/textures/gradientbox/nx.png",
      "engine/textures/gradientbox/ny.png",
      "engine/textures/gradientbox/py.png",
      "engine/textures/gradientbox/pz.png",
      "engine/textures/gradientbox/nz.png",
    ]);

    // Set the skybox color to white.
    SharedEnvironmentMapData.set_skybox_color([1, 1, 1, 1]);

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity();
    this.scene_entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.add_fragment(light_entity, LightFragment, false);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color.r = 1;
    light_fragment_view.color.g = 1;
    light_fragment_view.color.b = 1;
    light_fragment_view.intensity = 3;
    light_fragment_view.position.x = 50;
    light_fragment_view.position.y = 0;
    light_fragment_view.position.z = 50;
    light_fragment_view.active = true;

    // Create a sphere mesh and add it to the scene
    const mesh = Mesh.from_gltf("engine/models/sphere/sphere.gltf");

    // Create a default material
    const default_material_id = Material.create("MyMaterial", "StandardMaterial");

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    // Create a 3D grid of sphere entities
    const grid_size = 100; // 100x100x10 grid
    const grid_layers = 10;
    const spacing = 5; // 2 units apart

    EntityManager.reserve_entities(grid_size * grid_size * grid_layers);

    for (let x = 0; x < grid_size; x++) {
      for (let z = 0; z < grid_size; z++) {
        for (let y = 0; y < grid_layers; y++) {
          const sphere = spawn_mesh_entity(
            [
              (x - Math.floor(grid_size / 2)) * spacing,
              (y - Math.floor(grid_layers / 2)) * spacing,
              (z - Math.floor(grid_size / 2)) * spacing,
            ],
            [0, 0, 0, 1],
            [0.5, 0.5, 0.5],
            mesh,
            default_material_id,
            null /* parent */,
            [] /* children */,
            true /* start_visible */
          );
          this.scene_entities.push(sphere);
        }
      }
    }

    const text_entity = spawn_mesh_entity(
      [0, 25, -5],
      [0, 0, 0, 1],
      [0.5, 0.5, 0.5],
      Mesh.quad(),
      font_object.material,
    );
    const text_fragment_view = EntityManager.add_fragment(text_entity, TextFragment);
    text_fragment_view.font = font_id;
    text_fragment_view.text = "Sundown Engine";
    text_fragment_view.font_size = 32;
    text_fragment_view.color.r = 1;
    text_fragment_view.color.g = 1;
    text_fragment_view.color.b = 1;
    text_fragment_view.color.a = 1;
    text_fragment_view.emissive = 1;
    this.scene_entities.push(text_entity);

    PostProcessStack.register_pass(0, "outline", "effects/outline_post.wgsl", {
      outline_thickness: 1.0,
      depth_threshold: 0.1,
      normal_threshold: 1.0,
      depth_scale: 2000.0,
      outline_color: [0.2, 0.3, 0.8, 1.0],
    });
  }

  cleanup() {
    PostProcessStack.reset();

    for (const entity of this.scene_entities) {
      delete_entity(entity);
    }

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);

    const transforms = EntityManager.get_fragment_array(TransformFragment);

    ComputeTaskQueue.get().new_task(
      "ripples",
      "effects/transform_ripples.wgsl",
      [transforms.position_buffer, transforms.flags_buffer],
      [transforms.position_buffer, transforms.flags_buffer],
      Math.ceil(transforms.flags.length / 256)
    );
  }
}

export class MLScene extends Scene {
  mastermind = null;
  sine_model = null;
  xor_model = null;

  scene_entities = [];

  init(parent_context) {
    super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skybox for this scene.
    SharedEnvironmentMapData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.png",
      "engine/textures/gradientbox/nx.png",
      "engine/textures/gradientbox/ny.png",
      "engine/textures/gradientbox/py.png",
      "engine/textures/gradientbox/pz.png",
      "engine/textures/gradientbox/nz.png",
    ]);

    // Set the skybox color to white.
    SharedEnvironmentMapData.set_skybox_color([1, 1, 1, 1]);

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity();

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.add_fragment(light_entity, LightFragment, false);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color.r = 1;
    light_fragment_view.color.g = 1;
    light_fragment_view.color.b = 1;
    light_fragment_view.intensity = 3;
    light_fragment_view.position.x = 50;
    light_fragment_view.position.y = 0;
    light_fragment_view.position.z = 0;
    light_fragment_view.active = true;

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    EntityManager.reserve_entities(1);

    const text_entity = spawn_mesh_entity(
      [0, 25, -50],
      [0, 0, 0, 1],
      [0.5, 0.5, 0.5],
      Mesh.quad(),
      font_object.material,
      null /* parent */,
      [] /* children */,
      true /* start_visible */
    );
    const text_fragment_view = EntityManager.add_fragment(text_entity, TextFragment);
    text_fragment_view.font = font_id;
    text_fragment_view.text = "ML Test";
    text_fragment_view.font_size = 32;
    text_fragment_view.color.r = 1;
    text_fragment_view.color.g = 1;
    text_fragment_view.color.b = 1;
    text_fragment_view.color.a = 1;
    text_fragment_view.emissive = 1;

    this.scene_entities.push(light_entity);
    this.scene_entities.push(text_entity);

    this.setup_ml_test();
  }

  cleanup() {
    this.mastermind.destroy();

    this.sine_model = null;
    this.xor_model = null;

    for (const entity of this.scene_entities) {
      delete_entity(entity);
    }

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);

    profile_scope("ml_training_test.update", () => {
      for (let i = 0; i < 4; i++) {
        // Generate and enqueue a training batch for the sine model.
        const sine_batch = this.create_sine_batch();
        this.mastermind.add_training_batch(this.sine_model, sine_batch.input, sine_batch.target);

        // Generate and enqueue a training batch for the XOR model.
        const xor_batch = this.create_xor_batch();
        this.mastermind.add_training_batch(this.xor_model, xor_batch.input, xor_batch.target);
      }

      this.mastermind.tick(delta_time);
    });
  }

  setup_ml_test() {
    // Create a MasterMind instance with weight sharing enabled.
    this.mastermind = MasterMind.create({
      enable_weight_sharing: false,
      weight_sharing_interval: 0.5, // seconds between weight sharing updates
      mini_batch_size: 4,
    });

    // ---------------------------------------------------------------------------
    // Model A: Sine Function Approximator
    // Task: Given an input x, predict sin(x).
    // Architecture: [1] -> FullyConnectedLayer (1 -> 10) -> Tanh ->
    //               FullyConnectedLayer (10 -> 1) -> MSELoss
    // ---------------------------------------------------------------------------
    const sine_model = new NeuralModel("sine_approximator", {
      learning_rate: 0.01,
      optimizer: new Adam(),
    });
    sine_model.add(new FullyConnected(1, 10, { initializer: TensorInitializer.GLOROT }));
    sine_model.add(new Tanh());
    sine_model.add(new FullyConnected(10, 1, { initializer: TensorInitializer.GLOROT }));
    sine_model.add(new Tanh());
    sine_model.add(new MSELoss(false /* enabled_logging */, "sine_approximator"));

    this.sine_model = this.mastermind.register_model(sine_model);

    // ---------------------------------------------------------------------------
    // Model B: XOR Classifier
    // Task: Given two binary inputs, predict the XOR (0 or 1).
    // Architecture: [2] -> FullyConnectedLayer (2 -> 4) -> ReLu ->
    //               FullyConnectedLayer (4 -> 1) -> MSELoss
    // ---------------------------------------------------------------------------
    const xor_model = new NeuralModel("xor_classifier", {
      learning_rate: 0.01,
      optimizer: new Adam(),
    });
    xor_model.add(new FullyConnected(2, 4, { initializer: TensorInitializer.GLOROT }));
    xor_model.add(new ReLU());
    xor_model.add(new FullyConnected(4, 1, { initializer: TensorInitializer.GLOROT }));
    xor_model.add(new MSELoss(false /* enabled_logging */, "xor_classifier"));

    this.xor_model = this.mastermind.register_model(xor_model);
  }

  // Helper function: Create a training batch for the sine approximator.
  create_sine_batch() {
    // Random x in range [-π, π]
    const x = Math.random() * (2 * Math.PI) - Math.PI;
    const y = Math.sin(x);
    return {
      input: Tensor.create(new Float32Array([x]), [1, 1]),
      target: Tensor.create(new Float32Array([y]), [1, 1]),
    };
  }

  // Helper function: Create a training batch for the XOR classifier.
  create_xor_batch() {
    // The XOR truth table
    const xor_data = [
      { input: [0, 0], target: 0 },
      { input: [0, 1], target: 1 },
      { input: [1, 0], target: 1 },
      { input: [1, 1], target: 0 },
    ];
    const sample = xor_data[Math.floor(Math.random() * xor_data.length)];
    return {
      input: Tensor.create(new Float32Array(sample.input), [1, 2]),
      target: Tensor.create(new Float32Array([sample.target]), [1, 1]),
    };
  }
}

export class SceneSwitcher extends SimulationLayer {
  current_scene_index = null;
  scenes = [];

  constructor(name) {
    super();
    this.name = name;
  }

  async update(delta_time) {
    super.update(delta_time);

    if (InputProvider.get_action(InputKey.K_Return)) {
      if (this.current_scene_index !== null) {
        SimulationCore.unregister_simulation_layer(this.scenes[this.current_scene_index]);
      }

      this.current_scene_index = (this.current_scene_index + 1) % this.scenes.length;

      await SimulationCore.register_simulation_layer(this.scenes[this.current_scene_index]);
    }
  }

  async add_scene(scene) {
    if (this.current_scene_index === null) {
      this.current_scene_index = 0;
      await SimulationCore.register_simulation_layer(scene);
    }
    this.scenes.push(scene);
  }
}

(async () => {
  const simulator = await Simulator.create("gpu-canvas", "ui-canvas");

  // Create a test scene and register it with the simulation system
  const ml_scene = new MLScene("MLScene");
  const rendering_scene = new RenderingScene("RenderingScene");

  const scene_switcher = new SceneSwitcher("SceneSwitcher");
  await scene_switcher.add_scene(rendering_scene);
  await scene_switcher.add_scene(ml_scene);
  await simulator.add_sim_layer(scene_switcher);

  simulator.run();
})();

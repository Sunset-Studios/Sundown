// ═══════════════════════════════════════════════════════════════════════════════
// MLScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { SharedEnvironmentData } from "../../engine/src/core/shared_data.js";
import { delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { profile_scope } from "../../engine/src/utility/performance.js";
import { Layer, TrainingContext } from "../../engine/src/ml/layer.js";
import { LayerType } from "../../engine/src/ml/ml_types.js";
import { Input } from "../../engine/src/ml/layers/input.js";
import { MasterMind } from "../../engine/src/ml/mastermind.js";
import { Tensor, TensorInitializer } from "../../engine/src/ml/math/tensor.js";
import { Adam } from "../../engine/src/ml/optimizers/adam.js";
import { world_label } from "../../engine/src/ui/3d/common.js";

export class MLScene extends Scene {
  mastermind = null;
  sine_model = null;
  xor_model = null;
  xor_model_input = null;

  scene_entities = [];

  init(parent_context) {
    super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skybox for this scene.
    SharedEnvironmentData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.ktx2",
      "engine/textures/gradientbox/nx.ktx2",
      "engine/textures/gradientbox/ny.ktx2",
      "engine/textures/gradientbox/py.ktx2",
      "engine/textures/gradientbox/pz.ktx2",
      "engine/textures/gradientbox/nz.ktx2",
    ]);

    // Set the skybox color to white.
    SharedEnvironmentData.set_skybox_color([1, 1, 1, 1]);

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 3;
    light_fragment_view.position = [50, 0, 0];
    light_fragment_view.active = true;

    this.scene_entities.push(light_entity);

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

    world_label("ML Test", [0, 25, -50], {
      width: 40.0,
      height: 10.0,
    });

    profile_scope("ml_training_test.update", () => {
      for (let i = 0; i < 4; i++) {
        // Generate and enqueue a training batch for the sine model. This example shows how to use the mastermind to add training data.
        const sine_batch = this.create_sine_batch();
        this.mastermind.add_training_batch(this.sine_model, sine_batch.input, sine_batch.target);

        // Generate and enqueue a training batch for the XOR model. This example shows how to use the Input layer to add training data.
        const xor_batch = this.create_xor_batch();
        const xor_input_layer = Layer.get(this.xor_model_input);
        Input.add_sample_batch(xor_input_layer, xor_batch.input, xor_batch.target);
      }

      this.mastermind.tick(delta_time);
    });
  }

  setup_ml_test() {
    // Create a MasterMind instance with weight sharing enabled.
    this.mastermind = MasterMind.create({
      enable_weight_sharing: false,
      weight_sharing_interval: 0.5, // seconds between weight sharing updates
      mini_batch_size: 16,
    });

    // Demonstrates two possible APIs:
    // 1. The "store" API, which is a high-level API that allows for more control over subnets.
    //    It also allows external observers to observe changes in the store's state. Useful for applying views over the subnet data.
    //    The mastermind creates a default store, but you can create your own using MLOps.new_ops_store().
    // 2. The "layers" API, which is a high-level API that does not use observers. The store API relies on the layers API internally.
    //    It hides the details of the model from the user and provides a more intuitive API for training and inference via
    //    simple function calls and layer chaining.

    // ---------------------------------------------------------------------------
    // Model A: Sine Function Approximator (store API)
    // ---------------------------------------------------------------------------
    // Task: Given an input x, predict sin(x).
    // Architecture: [1] -> FullyConnectedLayer (1 -> 10) -> Tanh ->
    //               FullyConnectedLayer (10 -> 1) -> MSELoss
    // ---------------------------------------------------------------------------
    {
      const root = this.mastermind.store.add_input(1000, 16);

      const hidden1 = this.mastermind.store.add_layer(LayerType.FULLY_CONNECTED, 10, root, {
        initializer: TensorInitializer.GLOROT,
      });

      const sig = this.mastermind.store.add_activation(LayerType.SIGMOID, hidden1);

      const hidden2 = this.mastermind.store.add_layer(LayerType.FULLY_CONNECTED, 1, sig, {
        initializer: TensorInitializer.GLOROT,
      });

      this.mastermind.store.add_loss(
        LayerType.MSE,
        false /* enabled_logging */,
        "sine_approximator",
        hidden2
      );

      this.mastermind.store.set_subnet_context(root, {
        name: "sine_approximator",
        learning_rate: 0.01,
        weight_decay: 0.001,
        optimizer: new Adam(),
      });

      this.sine_model = this.mastermind.get_registered_subnet_id(root);
    }

    // ---------------------------------------------------------------------------
    // Model B: XOR Classifier (layers API)
    // Task: Given two binary inputs, predict the XOR (0 or 1).
    // Architecture: InputLayer [2] -> FullyConnectedLayer (2 -> 8) -> ReLu ->
    //               FullyConnectedLayer (8 -> 4) -> ReLu ->
    //               FullyConnectedLayer (4 -> 1) -> Sigmoid -> MSELoss
    // ---------------------------------------------------------------------------
    {
      const root = Layer.create(LayerType.INPUT, {
        capacity: 1000,
        batch_size: 16,
      });

      const hidden1 = Layer.create(
        LayerType.FULLY_CONNECTED,
        {
          output_size: 8,
          initializer: TensorInitializer.GLOROT,
        },
        root
      );

      const relu1 = Layer.create(LayerType.RELU, {}, hidden1);

      const hidden2 = Layer.create(
        LayerType.FULLY_CONNECTED,
        { output_size: 4, initializer: TensorInitializer.GLOROT },
        relu1
      );

      const relu2 = Layer.create(LayerType.RELU, {}, hidden2);

      const hidden3 = Layer.create(
        LayerType.FULLY_CONNECTED,
        { output_size: 1, initializer: TensorInitializer.GLOROT },
        relu2
      );

      const sigmoid = Layer.create(LayerType.SIGMOID, {}, hidden3);

      Layer.create(LayerType.MSE, { enable_logging: false, name: "xor_classifier" }, sigmoid);

      Layer.set_subnet_context(
        root,
        new TrainingContext({
          name: "xor_classifier",
          learning_rate: 0.01,
          weight_decay: 0.0001,
          optimizer: new Adam(),
        })
      );

      this.xor_model_input = root;
      this.xor_model = this.mastermind.register_subnet(root);
    }
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

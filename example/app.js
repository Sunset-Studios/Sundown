import { Simulator } from "../engine/src/core/simulator.js";
import SimulationCore from "../engine/src/core/simulation_core.js";
import { SimulationLayer } from "../engine/src/core/simulation_layer.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { Scene } from "../engine/src/core/scene.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { InputKey } from "../engine/src/input/input_types.js";
import { PostProcessStack } from "../engine/src/renderer/post_process_stack.js";
import { AABBTreeDebugRenderer } from "../engine/src/core/subsystems/aabb_debug_renderer.js";
import { AABBRaycast, Ray, RaycastHit } from "../engine/src/acceleration/aabb_raycast.js";
import { AABBGPURaycast } from "../engine/src/acceleration/aabb_gpu_raycast.js";
import { ComputeTaskQueue } from "../engine/src/renderer/compute_task_queue.js";
import { TransformFragment } from "../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { TextFragment } from "../engine/src/core/ecs/fragments/text_fragment.js";
import { StaticMeshFragment } from "../engine/src/core/ecs/fragments/static_mesh_fragment.js";
import { LightType, EntityTransformFlags } from "../engine/src/core/minimal.js";
import { Material } from "../engine/src/renderer/material.js";
import { Buffer } from "../engine/src/renderer/buffer.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { LineRenderer } from "../engine/src/renderer/line_renderer.js";
import { SharedEnvironmentMapData, SharedViewBuffer } from "../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../engine/src/core/ecs/entity_utils.js";
import { FontCache } from "../engine/src/ui/text/font_cache.js";
import { Name } from "../engine/src/utility/names.js";
import { profile_scope } from "../engine/src/utility/performance.js";
import { vec4, quat } from "gl-matrix";

import * as UI from "../engine/src/ui/2d/immediate.js";

import { Layer, TrainingContext } from "../engine/src/ml/layer.js";
import { LayerType } from "../engine/src/ml/ml_types.js";
import { MasterMind } from "../engine/src/ml/mastermind.js";
import { FullyConnected } from "../engine/src/ml/layers/fully_connected.js";
import { MSELoss } from "../engine/src/ml/layers/mse_loss.js";
import { ReLU } from "../engine/src/ml/layers/relu.js";
import { Tensor, TensorInitializer } from "../engine/src/ml/math/tensor.js";
import { Adam } from "../engine/src/ml/optimizers/adam.js";

// ------------------------------------------------------------------------------------
// =============================== Rendering Scene ===============================
// ------------------------------------------------------------------------------------

export class RenderingScene extends Scene {
  entities = [];

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
    this.entities.push(light_entity);

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
    // Configure the materials using combined uniform buffers
    const default_material = Material.get(default_material_id);
    // Create a combined uniform buffer for the default material
    // Contains: color (vec4) and emission (float, aligned to vec4)
    const default_material_data = new Float32Array([
      // color: vec4 (RGBA)
      0.7, 0.7, 0.7, 1.0,
      // emission: float (followed by padding to maintain alignment)
      0.0, 0.0, 0.0, 0.0,
    ]);
    const default_material_buffer = Buffer.create({
      name: "default_material_buffer",
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      raw_data: default_material_data,
    });
    // Set the uniform buffer for the material
    default_material.set_uniform_data("material_params", default_material_buffer);

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
            true /* start_visible */,
            EntityTransformFlags.NO_AABB_UPDATE | EntityTransformFlags.IGNORE_PARENT_SCALE
          );
          this.entities.push(sphere);
        }
      }
    }

    const text_entity = spawn_mesh_entity(
      [0, 25, -5],
      [0, 0, 0, 1],
      [0.5, 0.5, 0.5],
      Mesh.quad(),
      font_object.material,
      null /* parent */,
      [] /* children */,
      true /* start_visible */,
      EntityTransformFlags.NO_AABB_UPDATE | EntityTransformFlags.IGNORE_PARENT_SCALE
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
    this.entities.push(text_entity);

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

    for (const entity of this.entities) {
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
      [transforms.position_buffer, transforms.flags_buffer, transforms.dirty_buffer],
      [transforms.position_buffer, transforms.flags_buffer, transforms.dirty_buffer],
      Math.ceil(transforms.flags.length / 256)
    );
  }
}

// ------------------------------------------------------------------------------------
// =============================== ML Scene =========================================
// ------------------------------------------------------------------------------------

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
      const root = this.mastermind.store.add_layer(LayerType.FULLY_CONNECTED, 1, 10, null, {
        initializer: TensorInitializer.GLOROT,
      });

      const tanh = this.mastermind.store.add_activation(LayerType.TANH, root);

      const hidden1 = this.mastermind.store.add_layer(LayerType.FULLY_CONNECTED, 10, 1, tanh, {
        initializer: TensorInitializer.GLOROT,
      });

      const output = this.mastermind.store.add_loss(
        LayerType.MSE,
        false /* enabled_logging */,
        "sine_approximator",
        hidden1
      );

      const context = this.mastermind.store.set_subnet_context(root, {
        name: "sine_approximator",
        learning_rate: 0.01,
        weight_decay: 0.001,
        optimizer: new Adam(),
      });

      this.sine_model = this.mastermind.register_subnet(root);
    }

    // ---------------------------------------------------------------------------
    // Model B: XOR Classifier (model API)
    // Task: Given two binary inputs, predict the XOR (0 or 1).
    // Architecture: [2] -> FullyConnectedLayer (2 -> 8) -> ReLu ->
    //               FullyConnectedLayer (8 -> 4) -> ReLu ->
    //               FullyConnectedLayer (4 -> 1) -> Sigmoid -> MSELoss
    // ---------------------------------------------------------------------------
    {
      const root = Layer.create(LayerType.FULLY_CONNECTED, {
        input_size: 2,
        output_size: 8,
        initializer: TensorInitializer.GLOROT,
      });

      const relu1 = Layer.create(LayerType.RELU, {}, root);

      const hidden1 = Layer.create(
        LayerType.FULLY_CONNECTED,
        { input_size: 8, output_size: 4, initializer: TensorInitializer.GLOROT },
        relu1
      );

      const relu2 = Layer.create(LayerType.RELU, {}, hidden1);

      const hidden2 = Layer.create( 
        LayerType.FULLY_CONNECTED,
        { input_size: 4, output_size: 1, initializer: TensorInitializer.GLOROT },
        relu2
      );

      const sigmoid = Layer.create(LayerType.SIGMOID, {}, hidden2);

      const loss = Layer.create(
        LayerType.MSE,
        { enable_logging: false, name: "xor_classifier" },
        sigmoid
      );

      Layer.set_subnet_context(root, new TrainingContext({
        name: "xor_classifier",
        learning_rate: 0.01,
        weight_decay: 0.0001,
        optimizer: new Adam(),
      }));

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

// ------------------------------------------------------------------------------------
// =============================== AABB Scene =======================================
// ------------------------------------------------------------------------------------

const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: 400,
  x: 25,
  anchor_x: "right",
  anchor_y: "bottom",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 600,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const stats_label_config = {
  text_color: "#fff",
  x: 0,
  y: 0,
  wrap: true,
  font: "16px monospace",
  width: "100%",
  height: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

const button_config = {
  y: 0,
  x: 0,
  width: "fit-content",
  font: "bold 16px monospace",
  height: 30,
  background_color: "#FFA500",
  text_color: "#111111",
  corner_radius: 5,
  text_padding: 10,
};

export class AABBScene extends Scene {
  name = "AABBScene";
  show_ui = false;
  entities = [];
  selected_entity = null;
  use_gpu_raycast = false;
  ray_hits = [];
  last_ray_origin = null;
  last_ray_direction = null;
  ray_line_collection = null;

  init(parent_context) {
    super.init(parent_context);

    // Set the skybox for this scene.
    SharedEnvironmentMapData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.png",
      "engine/textures/gradientbox/nx.png",
      "engine/textures/gradientbox/ny.png",
      "engine/textures/gradientbox/py.png",
      "engine/textures/gradientbox/pz.png",
      "engine/textures/gradientbox/nz.png",
    ]);

    // Set the skybox color to a subtle blue
    SharedEnvironmentMapData.set_skybox_color([0.7, 0.8, 1.0, 1]);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Reset view to a good position for the BVH scene
    SharedViewBuffer.set_view_data(0, {
      position: [47.0751, 55.28902, 106.885414],
      rotation: [-0.023805, 0.97379, -0.190533, -0.121665],
    });

    this.aabb_tree_debug_renderer = this.get_layer(AABBTreeDebugRenderer);

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity();
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.add_fragment(light_entity, LightFragment, false);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color.r = 1;
    light_fragment_view.color.g = 1;
    light_fragment_view.color.b = 1;
    light_fragment_view.intensity = 3;
    light_fragment_view.position.x = 50;
    light_fragment_view.position.y = 20;
    light_fragment_view.position.z = 50;
    light_fragment_view.active = true;

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    // Add a title text entity
    const text_entity = spawn_mesh_entity(
      [25, 65, 25],
      [0, 0, 0, 1],
      [0.5, 0.5, 0.5],
      Mesh.quad(),
      font_object.material
    );
    const text_fragment_view = EntityManager.add_fragment(text_entity, TextFragment);
    text_fragment_view.font = font_id;
    text_fragment_view.text = "BVH Test Scene";
    text_fragment_view.font_size = 32;
    text_fragment_view.color.r = 1;
    text_fragment_view.color.g = 1;
    text_fragment_view.color.b = 1;
    text_fragment_view.color.a = 1;
    text_fragment_view.emissive = 1;
    this.entities.push(text_entity);

    // Create a default material
    this.default_material_id = Material.create("AABBTreeDefaultMaterial", "StandardMaterial");
    // Create a default material for the selected entity
    this.selected_entity_material_id = Material.create(
      "AABBTreeSelectedEntityMaterial",
      "StandardMaterial"
    );

    {
      // Configure the materials using combined uniform buffers
      const default_material = Material.get(this.default_material_id);
      // Create a combined uniform buffer for the default material
      // Contains: color (vec4) and emission (float, aligned to vec4)
      this.default_material_data = new Float32Array([
        // color: vec4 (RGBA)
        0.7, 0.7, 0.7, 1.0,
        // emission: float (followed by padding to maintain alignment)
        0.2, 0.0, 0.0, 0.0,
      ]);

      this.default_material_buffer = Buffer.create({
        name: "default_material_buffer",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        raw_data: this.default_material_data,
      });

      // Set the uniform buffer for the material
      default_material.set_uniform_data("material_params", this.default_material_buffer);
    }

    {
      // Create a combined uniform buffer for the selected entity material
      const selected_entity_material = Material.get(this.selected_entity_material_id);
      // Create a combined uniform buffer for the selected entity material
      // Contains: color (vec4) and emission (float, aligned to vec4)
      this.selected_entity_material_data = new Float32Array([
        // color: vec4 (RGBA)
        1.0, 0.3, 0.3, 1.0,
        // emission: float (followed by padding to maintain alignment)
        1.0, 0.0, 0.0, 0.0,
      ]);

      this.selected_entity_material_buffer = Buffer.create({
        name: "selected_entity_material_buffer",
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        raw_data: this.selected_entity_material_data,
      });

      // Set the uniform buffer for the material
      selected_entity_material.set_uniform_data(
        "material_params",
        this.selected_entity_material_buffer
      );
    }

    // Create a sphere mesh
    this.sphere_mesh = Mesh.from_gltf("engine/models/sphere/sphere.gltf");

    // Create a cube mesh
    this.cube_mesh = Mesh.cube();

    // Setup a grid of entities
    this.setup_entity_grid();
  }

  cleanup() {
    for (let i = 0; i < this.entities.length; i++) {
      delete_entity(this.entities[i]);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    AABBGPURaycast.cleanup();

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);

    this.handle_input();

    // Run a raycast every frame from mouse position
    if (this.entities.length > 0) {
      this.run_raycast();
    }

    this.render_ui();
  }

  setup_entity_grid() {
    // Create a grid of entities for testing
    const grid_size = 20;
    const spacing = 3.0;

    EntityManager.reserve_entities(grid_size * grid_size * grid_size);

    for (let x = 0; x < grid_size; x++) {
      for (let z = 0; z < grid_size; z++) {
        for (let y = 0; y < grid_size; y++) {
          // Alternate between cubes and spheres
          const mesh = (x + z) % 2 === 0 ? this.cube_mesh : this.sphere_mesh;

          // Random position with small offset
          const position = [x * spacing, y * spacing, z * spacing];

          // Random scale
          const scale = [
            0.5 /*+ Math.random() * 0.5*/, 0.5 /*+ Math.random() * 0.5*/,
            0.5 /*+ Math.random() * 0.5*/,
          ];

          // Create entity
          const entity = spawn_mesh_entity(
            position,
            quat.fromEuler(quat.create(), 0, 0, 0),
            scale,
            mesh,
            this.default_material_id
          );

          this.entities.push(entity);
        }
      }
    }
  }

  handle_input() {
    // Toggle GPU raycast mode
    if (InputProvider.get_action(InputKey.K_g)) {
      this.use_gpu_raycast = !this.use_gpu_raycast;
    }

    // Toggle raycast UI
    if (InputProvider.get_action(InputKey.K_u)) {
      this.show_ui = !this.show_ui;
      if (this.show_ui) {
        this.show_dev_cursor();
      } else {
        this.hide_dev_cursor();
      }
    }

    // Add a new entity at the previous hit point
    if (InputProvider.get_action(InputKey.K_Space) && this.ray_hits.length > 0) {
      const hit = this.ray_hits[0];

      // Create entity at hit point
      const entity = spawn_mesh_entity(
        hit.point,
        [0, 0, 0, 1],
        [0.5, 0.5, 0.5],
        Math.random() > 0.5 ? this.cube_mesh : this.sphere_mesh,
        this.default_material_id,
        null,
        [],
        true,
        EntityTransformFlags.IGNORE_PARENT_SCALE | EntityTransformFlags.NO_AABB_UPDATE
      );

      this.entities.push(entity);
    }

    // Delete the selected entity
    if (InputProvider.get_action(InputKey.K_Backspace) && this.selected_entity) {
      delete_entity(this.selected_entity);

      // Remove from entities array
      const index = this.entities.indexOf(this.selected_entity);
      if (index >= 0) {
        this.entities.splice(index, 1);
      }

      this.selected_entity = null;
    }
  }

  run_raycast() {
    // Get camera position and direction
    const view_data = SharedViewBuffer.get_view_data(0);
    if (!view_data) return;

    // Get cursor world position (this is a point on the far plane)
    const cursor_world_position = UI.UIContext.input_state.world_position;
    if (!cursor_world_position) return;

    // Use camera position as ray origin
    this.last_ray_origin = cursor_world_position;

    // Calculate ray direction from camera to cursor world position
    this.last_ray_direction = vec4.sub(vec4.create(), cursor_world_position, view_data.position);
    // Normalize the direction vector
    const length = Math.sqrt(
      this.last_ray_direction[0] * this.last_ray_direction[0] +
        this.last_ray_direction[1] * this.last_ray_direction[1] +
        this.last_ray_direction[2] * this.last_ray_direction[2]
    );

    this.last_ray_direction[0] /= length;
    this.last_ray_direction[1] /= length;
    this.last_ray_direction[2] /= length;

    // Create ray for raycasting
    const ray = new Ray(this.last_ray_origin, this.last_ray_direction);

    // Perform raycast based on current mode
    if (this.use_gpu_raycast) {
      AABBGPURaycast.raycast(ray, { first_hit_only: true }, (hits) => {
        this.process_raycast_results(hits);
      });
    } else {
      AABBRaycast.raycast(ray, { first_hit_only: true }, (hits) => {
        this.process_raycast_results(hits);
      });
    }
  }

  process_raycast_results(hits) {
    // Process the hits (if any)
    let new_ray_hits = [];
    if (hits) {
      if (Array.isArray(hits)) {
        new_ray_hits = hits;
      } else {
        new_ray_hits = [hits]; // Single hit, wrap in array
      }
    } else {
      new_ray_hits = [];
    }

    // Early out if same hits as last frame
    if (
      this.ray_hits.length === new_ray_hits.length &&
      this.ray_hits.every((hit, index) => hit === new_ray_hits[index])
    ) {
      return;
    }

    this.ray_hits = new_ray_hits;

    // Update selected entity highlighting
    if (this.selected_entity !== null) {
      // Reset previous selection
      const static_mesh_fragment = EntityManager.get_fragment(
        this.selected_entity,
        StaticMeshFragment
      );
      if (static_mesh_fragment) {
        static_mesh_fragment.material_slots = [this.default_material_id];
      }

      this.selected_entity = null;
    }

    // Select new entity if we hit something
    if (this.ray_hits.length > 0) {
      const hit = this.ray_hits[0];

      // Update the line visualization
      if (this.ray_line_collection) {
        LineRenderer.clear_collection(this.ray_line_collection);
      }

      this.ray_line_collection = LineRenderer.start_collection();

      // Draw the ray from camera to end point
      LineRenderer.add_line(this.last_ray_origin, hit.point, [1, 0, 0, 1]);

      LineRenderer.end_collection();

      // Highlight the selected entity by writing to the material buffer
      this.selected_entity = hit.user_data;
      const static_mesh_fragment = EntityManager.get_fragment(
        this.selected_entity,
        StaticMeshFragment
      );
      if (static_mesh_fragment) {
        static_mesh_fragment.material_slots = [this.selected_entity_material_id];
      }
    }

    this.default_material_buffer.write(this.default_material_data);
  }

  render_ui() {
    if (!this.show_ui) return;

    // Use immediate mode UI panel instead of window
    UI.panel(stats_panel_config, () => {
      // Add buttons for common actions
      UI.begin_container({
        layout: "row",
        x: 0,
        gap: 10,
        height: 40,
        padding_top: 10,
      });

      // Use a button to toggle the raycast mode
      const raycast_mode_text = `Raycast Mode: ${this.use_gpu_raycast ? "GPU" : "CPU"}`;
      const raycast_mode_button = UI.button(raycast_mode_text, button_config);
      if (raycast_mode_button.clicked) {
        this.use_gpu_raycast = !this.use_gpu_raycast;
      }

      UI.end_container();

      // Add some spacing
      UI.begin_container({ height: 10, width: "100%" });
      UI.end_container();

      UI.label("Raycast Results:", stats_label_config);

      if (this.ray_hits.length > 0) {
        const hit = this.ray_hits[0];
        UI.label(`Hit Entity: ${hit.user_data}`, stats_label_config);
        UI.label(`Distance: ${hit.distance.toFixed(2)}`, stats_label_config);

        const pos_text = `Position: [${hit.point[0].toFixed(2)}, ${hit.point[1].toFixed(2)}, ${hit.point[2].toFixed(2)}]`;
        UI.label(pos_text, stats_label_config);

        const normal_text = `Normal: [${hit.normal[0].toFixed(2)}, ${hit.normal[1].toFixed(2)}, ${hit.normal[2].toFixed(2)}]`;
        UI.label(normal_text, stats_label_config);
      } else {
        UI.label("No hit", stats_label_config);
      }

      // Add some spacing
      UI.begin_container({ height: 10, width: "100%" });
      UI.end_container();

      // Instructions section
      UI.label("Controls:", stats_label_config);

      const control_labels = [
        "G: Toggle CPU/GPU raycasting",
        "Space: Add object at hit point",
        "Delete: Remove selected object",
      ];

      for (const text of control_labels) {
        UI.label(text, stats_label_config);
      }

      // Add buttons for common actions
      UI.begin_container({
        layout: "row",
        x: 0,
        gap: 10,
        height: 40,
        padding_top: 10,
      });

      const add_one_button = UI.button("Add Object", button_config);
      if (add_one_button.clicked) {
        if (this.ray_hits.length > 0) {
          const hit = this.ray_hits[0];
          const entity = spawn_mesh_entity(
            hit.point,
            quat.fromEuler(quat.create(), 0, 0, 0),
            [0.5, 0.5, 0.5],
            Math.random() > 0.5 ? this.cube_mesh : this.sphere_mesh,
            this.default_material_id
          );
          this.entities.push(entity);
        }
      }

      const delete_button = UI.button("Delete Selected", button_config);
      if (delete_button.clicked) {
        if (this.selected_entity) {
          delete_entity(this.selected_entity);
          const index = this.entities.indexOf(this.selected_entity);
          if (index >= 0) {
            this.entities.splice(index, 1);
          }
          this.selected_entity = null;
        }
      }

      // Add button for stress test
      const add_button = UI.button("Add 1000 Objects", button_config);
      if (add_button.clicked) {
        this.add_random_objects(1000);
      }

      UI.end_container();
    });
  }

  add_random_objects(count) {
    // Create random objects in a sphere around the camera
    const view_data = SharedViewBuffer.get_view_data(0);
    if (!view_data) return;

    const center = view_data.position;
    const radius = 50.0;

    for (let i = 0; i < count; i++) {
      // Random point in sphere
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = radius * Math.cbrt(Math.random()); // Cube root for uniform distribution

      const x = center[0] + r * Math.sin(phi) * Math.cos(theta);
      const y = center[1] + r * Math.sin(phi) * Math.sin(theta);
      const z = center[2] + r * Math.cos(phi);

      // Random size
      const scale = 0.2 + Math.random() * 0.8;

      // Create entity
      const entity = spawn_mesh_entity(
        [x, y, z],
        [0, 0, 0, 1],
        [scale, scale, scale],
        Math.random() > 0.5 ? this.cube_mesh : this.sphere_mesh,
        this.default_material_id,
        null,
        [],
        true
      );

      this.entities.push(entity);
    }
  }
}

// ------------------------------------------------------------------------------------
// =============================== Scene Switcher ====================================
// ------------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------------
// =============================== Main ==============================================
// ------------------------------------------------------------------------------------

(async () => {
  const simulator = await Simulator.create("gpu-canvas", "ui-canvas");

  // Create scenes and register them with the simulation system
  const aabb_scene = new AABBScene("AABBScene");
  const rendering_scene = new RenderingScene("RenderingScene");
  const ml_scene = new MLScene("MLScene");

  const scene_switcher = new SceneSwitcher("SceneSwitcher");
  //await scene_switcher.add_scene(aabb_scene);
  //await scene_switcher.add_scene(rendering_scene);
  await scene_switcher.add_scene(ml_scene);
  await simulator.add_sim_layer(scene_switcher);

  simulator.run();
})();

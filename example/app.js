import { Simulator } from "../engine/src/core/simulator.js";
import SimulationCore from "../engine/src/core/simulation_core.js";
import { FragmentGpuBuffer } from "../engine/src/core/ecs/solar/memory.js";
import { SimulationLayer } from "../engine/src/core/simulation_layer.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { Scene } from "../engine/src/core/scene.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { InputKey } from "../engine/src/input/input_types.js";
import { PostProcessStack } from "../engine/src/renderer/post_process_stack.js";
import { AABBTreeDebugRenderer } from "../engine/src/core/subsystems/aabb_debug_renderer.js";
import { AABBRaycast, Ray } from "../engine/src/acceleration/aabb_raycast.js";
import { AABBGPURaycast } from "../engine/src/acceleration/aabb_gpu_raycast.js";
import { ComputeTaskQueue } from "../engine/src/renderer/compute_task_queue.js";
import { TransformFragment } from "../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { TextFragment } from "../engine/src/core/ecs/fragments/text_fragment.js";
import { StaticMeshFragment } from "../engine/src/core/ecs/fragments/static_mesh_fragment.js";
import { VisibilityFragment } from "../engine/src/core/ecs/fragments/visibility_fragment.js";
import { LightType, EntityFlags } from "../engine/src/core/minimal.js";
import { StandardMaterial } from "../engine/src/renderer/material.js";
import { Texture } from "../engine/src/renderer/texture.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { SharedEnvironmentMapData, SharedViewBuffer } from "../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../engine/src/core/ecs/entity_utils.js";
import { FontCache } from "../engine/src/ui/text/font_cache.js";
import { Name } from "../engine/src/utility/names.js";
import { profile_scope } from "../engine/src/utility/performance.js";
import { log } from "../engine/src/utility/logging.js";
import { radians } from "../engine/src/utility/math.js";
import { vec3, vec4, quat } from "gl-matrix";

import * as UI from "../engine/src/ui/2d/immediate.js";

import { Layer, TrainingContext } from "../engine/src/ml/layer.js";
import { LayerType } from "../engine/src/ml/ml_types.js";
import { Input } from "../engine/src/ml/layers/input.js";
import { MasterMind } from "../engine/src/ml/mastermind.js";
import { Tensor, TensorInitializer } from "../engine/src/ml/math/tensor.js";
import { Adam } from "../engine/src/ml/optimizers/adam.js";
import { MaterialFamilyType } from "../engine/src/renderer/renderer_types.js";

// ------------------------------------------------------------------------------------
// =============================== Rendering Scene ===============================
// ------------------------------------------------------------------------------------

const positions_name = "position";
const ripples_name = "ripples";
const ripples_shader = "effects/transform_ripples.wgsl";

export class RenderingScene extends Scene {
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [-1.0, 22.0, 26.0];
    view_data.view_rotation = [-0.00061309, 0.9948077, -0.10095515, -0.00604141];

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
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1, 1];
    light_fragment_view.intensity = 3;
    light_fragment_view.position = [50, 0, 50, 1];
    light_fragment_view.active = true;

    // Create a sphere mesh and add it to the scene
    const mesh = Mesh.from_gltf("engine/models/cube/cube.gltf");

    // Create a default material
    const default_material = StandardMaterial.create("MyMaterial");
    const default_material_id = default_material.material_id;

    {
      let dirt_albedo = Texture.load(["engine/textures/voxel/dirt_albedo.jpg"], {
        name: "dirt_albedo",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "dirt_albedo",
      });
      let dirt_roughness = Texture.load(["engine/textures/voxel/dirt_roughness.jpg"], {
        name: "dirt_roughness",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "dirt_roughness",
      });

      default_material.set_albedo([0.7, 0.7, 0.7, 1.0], dirt_albedo);
      default_material.set_roughness(0.5, dirt_roughness);
      default_material.set_emission(0.2);
      default_material.set_tiling(2.0, 2.0);
    }

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    // Create a 3D grid of sphere entities
    const grid_size = 100; // 100x100x10 grid
    const grid_layers = 100;
    const spacing = 5; // 2 units apart

    const sphere = spawn_mesh_entity(
      [0, 0, 0],
      [0, 0, 0],
      [0.5, 0.5, 0.5],
      mesh,
      default_material_id,
      null /* parent */,
      [] /* children */,
      true /* start_visible */,
      EntityFlags.NO_AABB_UPDATE | EntityFlags.IGNORE_PARENT_SCALE
    );
    EntityManager.set_entity_instance_count(sphere, grid_size * grid_size * grid_layers);

    this.entities.push(sphere);

    let sphere_count = 0;
    for (let x = 0; x < grid_size; x++) {
      for (let z = 0; z < grid_size; z++) {
        for (let y = 0; y < grid_layers; y++) {
          const pos = [
            (x - Math.floor(grid_size / 2)) * spacing,
            (y - Math.floor(grid_layers / 2)) * spacing,
            (z - Math.floor(grid_size / 2)) * spacing,
          ];
          const view = EntityManager.get_fragment(sphere, TransformFragment, sphere_count);
          view.position = pos;
          ++sphere_count;
        }
      }
    }

    PostProcessStack.register_pass(0, "vhs", "effects/vhs_post.wgsl", {
      noise_intensity: 0.25,
      scanline_intensity: 0.35,
      color_bleeding: 0.25,
      distortion_frequency: 0.75,
      distortion_amplitude: 0.15,
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

    const flags = FragmentGpuBuffer.entity_flags_buffer;
    const positions = EntityManager.get_fragment_gpu_buffer(TransformFragment, positions_name);

    const total_transforms = EntityManager.get_total_subscribed(TransformFragment);

    ComputeTaskQueue.new_task(
      ripples_name,
      ripples_shader,
      [positions.buffer, flags.buffer],
      [positions.buffer, flags.buffer],
      Math.ceil(total_transforms / 256)
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
  xor_model_input = null;

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
    const light_entity = EntityManager.create_entity([LightFragment]);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 3;
    light_fragment_view.position = [50, 0, 0];
    light_fragment_view.active = true;

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

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
    text_fragment_view.font_size = 32;
    text_fragment_view.text_color = [1, 1, 1, 1];
    text_fragment_view.text_emissive = 1;
    text_fragment_view.text = "ML Test";

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
          input_size: 2,
          output_size: 8,
          initializer: TensorInitializer.GLOROT,
        },
        root
      );

      const relu1 = Layer.create(LayerType.RELU, {}, hidden1);

      const hidden2 = Layer.create(
        LayerType.FULLY_CONNECTED,
        { input_size: 8, output_size: 4, initializer: TensorInitializer.GLOROT },
        relu1
      );

      const relu2 = Layer.create(LayerType.RELU, {}, hidden2);

      const hidden3 = Layer.create(
        LayerType.FULLY_CONNECTED,
        { input_size: 4, output_size: 1, initializer: TensorInitializer.GLOROT },
        relu2
      );

      const sigmoid = Layer.create(LayerType.SIGMOID, {}, hidden3);

      const loss = Layer.create(
        LayerType.MSE,
        { enable_logging: false, name: "xor_classifier" },
        sigmoid
      );

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

// ------------------------------------------------------------------------------------
// =============================== Textures Scene =======================================
// ------------------------------------------------------------------------------------

export class TexturesScene extends Scene {
  name = "TexturesScene";
  entities = [];

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
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [39.198, 14.0851, 78.60858];
    view_data.view_rotation = [-0.0203683, 0.9771718, -0.179953, -0.110603];

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1, 1];
    light_fragment_view.intensity = 2;
    light_fragment_view.position = [50, 20, -10];
    light_fragment_view.active = true;

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    // Add a title text entity
    const text_entity = spawn_mesh_entity(
      [0, 20, 0],
      [0, 0, 0, 1],
      [0.5, 0.5, 0.5],
      Mesh.quad(),
      font_object.material
    );
    const text_fragment_view = EntityManager.add_fragment(text_entity, TextFragment);
    text_fragment_view.font = font_id;
    text_fragment_view.font_size = 32;
    text_fragment_view.text_color = [1, 1, 1, 1];
    text_fragment_view.text_emissive = 1;
    text_fragment_view.text = "Textures Test Scene";
    this.entities.push(text_entity);

    // Load metal plane material
    {
      let worn_panel_albedo = Texture.load(["engine/textures/worn_panel/worn_panel_albedo.png"], {
        name: "worn_panel_albedo",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "worn_panel_albedo",
      });
      let worn_panel_normal = Texture.load(["engine/textures/worn_panel/worn_panel_normal.png"], {
        name: "worn_panel_normal",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "worn_panel_normal",
      });
      let worn_panel_roughness = Texture.load(
        ["engine/textures/worn_panel/worn_panel_roughness.png"],
        {
          name: "worn_panel_roughness",
          format: "rgba8unorm",
          dimension: "2d",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          material_notifier: "worn_panel_roughness",
        }
      );
      let worn_panel_metallic = Texture.load(
        ["engine/textures/worn_panel/worn_panel_metallic.png"],
        {
          name: "worn_panel_metallic",
          format: "rgba8unorm",
          dimension: "2d",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          material_notifier: "worn_panel_metallic",
        }
      );
      let worn_panel_ao = Texture.load(["engine/textures/worn_panel/worn_panel_ao.png"], {
        name: "worn_panel_ao",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "worn_panel_ao",
      });

      // Create a default material
      const default_plane_material = StandardMaterial.create("TexturesDefaultMaterial");
      this.default_plane_material_id = default_plane_material.material_id;
      default_plane_material.set_albedo([0.5, 0.5, 0.5, 1], worn_panel_albedo);
      default_plane_material.set_normal([0, 1, 0, 1], worn_panel_normal);
      default_plane_material.set_roughness(0.5, worn_panel_roughness);
      default_plane_material.set_metallic(0.5, worn_panel_metallic);
      default_plane_material.set_ao(0.5, worn_panel_ao);
      default_plane_material.set_emission(0.1);
      default_plane_material.set_tiling(30.0);
    }

    // Load wall material
    {
      let wall_albedo = Texture.load(["engine/textures/wall/wall_albedo.png"], {
        name: "wall_albedo",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_albedo",
      });
      let wall_normal = Texture.load(["engine/textures/wall/wall_normal.png"], {
        name: "wall_normal",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_normal",
      });
      let wall_roughness = Texture.load(["engine/textures/wall/wall_roughness.png"], {
        name: "wall_roughness",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_roughness",
      });
      let wall_metallic = Texture.load(["engine/textures/wall/wall_metallic.png"], {
        name: "wall_metallic",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_metallic",
      });
      let wall_ao = Texture.load(["engine/textures/wall/wall_ao.png"], {
        name: "wall_ao",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_ao",
      });

      // Create a default material
      const wall_material = StandardMaterial.create("TexturesWallMaterial");
      this.wall_material_id = wall_material.material_id;
      wall_material.set_albedo([0.5, 0.5, 0.5, 1], wall_albedo);
      wall_material.set_normal([0, 1, 0, 1], wall_normal);
      wall_material.set_roughness(0.5, wall_roughness);
      wall_material.set_metallic(0.5, wall_metallic);
      wall_material.set_ao(0.5, wall_ao);
      wall_material.set_emission(0.1);
      wall_material.set_tiling(2.0);
    }

    // Create a sphere mesh
    this.sphere_mesh = Mesh.from_gltf("engine/models/sphere/sphere.gltf");

    // Create a cube mesh
    this.cube_mesh = Mesh.cube();

    // Setup the world plane
    this.setup_world_plane();

    // Setup sphere entity
    this.setup_sphere_entity();
  }

  cleanup() {
    for (let i = 0; i < this.entities.length; i++) {
      delete_entity(this.entities[i]);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);
  }

  setup_world_plane() {
    // Create a plane entity
    const plane = spawn_mesh_entity(
      [0, 0, 0],
      quat.fromEuler(quat.create(), 0, 0, 0),
      [500, 0.1, 500],
      Mesh.cube(),
      this.default_plane_material_id
    );
    this.entities.push(plane);
  }

  setup_sphere_entity() {
    // Create a sphere entity
    const sphere = spawn_mesh_entity(
      [0, 10, 10],
      quat.fromEuler(quat.create(), 0, 0, 0),
      [5, 5, 5],
      this.sphere_mesh,
      this.wall_material_id
    );
    this.entities.push(sphere);
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
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [47.0751, 55.28902, 106.885414];
    view_data.view_rotation = [-0.023805, 0.97379, -0.190533, -0.121665];

    this.aabb_tree_debug_renderer = this.get_layer(AABBTreeDebugRenderer);

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 3;
    light_fragment_view.position = [50, 20, 50];
    light_fragment_view.active = true;
    this.entities.push(light_entity);

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    // Add a title text entity
    const text_entity = spawn_mesh_entity(
      [10, 65, 25],
      [0, 0, 0, 1],
      [0.5, 0.5, 0.5],
      Mesh.quad(),
      font_object.material
    );
    const text_fragment_view = EntityManager.add_fragment(text_entity, TextFragment);
    text_fragment_view.font = font_id;
    text_fragment_view.font_size = 32;
    text_fragment_view.text_color = [1, 1, 1, 1];
    text_fragment_view.text_emissive = 1;
    text_fragment_view.text = "BVH Test Scene";
    this.entities.push(text_entity);

    // Create a default material
    const default_material = StandardMaterial.create("AABBTreeDefaultMaterial");
    this.default_material_id = default_material.material_id;
    default_material.set_albedo([0.5, 0.5, 0.5, 1]);
    default_material.set_normal([0, 1, 0, 1]);
    default_material.set_roughness(0.5);
    default_material.set_metallic(0.5);
    default_material.set_ao(0.5);
    default_material.set_emission(0.1);

    // Create a default material for the selected entity
    const selected_entity_material = StandardMaterial.create("AABBTreeSelectedEntityMaterial");
    this.selected_entity_material_id = selected_entity_material.material_id;
    selected_entity_material.set_albedo([1.0, 0.3, 0.3, 1]);
    selected_entity_material.set_emission(1.0);

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
            this.default_material_id,
            null,
            [],
            true
          );

          this.entities.push(entity);
        }
      }
    }

    log(`[AABB] Spawned ${this.entities.length} entities`);
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
        EntityFlags.IGNORE_PARENT_SCALE | EntityFlags.NO_AABB_UPDATE
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
    this.last_ray_origin = view_data.view_position;

    // Calculate ray direction from camera to cursor world position
    this.last_ray_direction = vec4.sub(
      vec4.create(),
      cursor_world_position,
      view_data.view_position
    );
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
    const previous_selected_entity = this.selected_entity;

    // Select new entity if we hit something
    if (this.ray_hits.length > 0) {
      const hit = this.ray_hits[0];

      // Highlight the selected entity by writing to the material buffer
      this.selected_entity = EntityManager.get_entity_from_id(hit.user_data);
    }

    if (previous_selected_entity === this.selected_entity) {
      return;
    }

    if (previous_selected_entity !== null) {
      // Reset previous selection
      const static_mesh_fragment = EntityManager.get_fragment(
        previous_selected_entity,
        StaticMeshFragment
      );
      if (static_mesh_fragment) {
        static_mesh_fragment.material_slots = [BigInt(this.default_material_id)];
      }
    }

    if (this.selected_entity !== null) {
      const static_mesh_fragment = EntityManager.get_fragment(
        this.selected_entity,
        StaticMeshFragment
      );
      if (static_mesh_fragment) {
        static_mesh_fragment.material_slots = [BigInt(this.selected_entity_material_id)];
      }
    }
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

    const center = view_data.view_position;
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
// =============================== Solar ECS Test Scene ==============================
// ------------------------------------------------------------------------------------

export class SolarECSTestScene extends Scene {
  name = "SolarECSTestScene";
  entities = []; // Stores all entities in the scene
  text_update_timer = 0; // Timer for text updates
  text_update_interval = 0.2; // Time in seconds between text updates

  grid_entity_counts = { x: 20, y: 20, z: 20 }; // Number of entities per dimension
  grid_spacing = { x: 4.0, y: 4.0, z: 4.0 }; // Explicit spacing between entities

  grid_text_entities = []; // Stores entities that are part of the text grid
  num_entities_to_update_per_cycle = 250; // Number of entities to update each cycle

  combined_text_presets = [
    // Morse code
    "...",
    ".-",
    "..",
    "---",
    ".-.",
    ".-..",
    "--",
    "-.",
    "..-",
    ".-.-",
    "---.",
    "....-",
    ".--",
    "-",
    "..-",
    ".-.-",
    "---.- .-",
    "....",
  ];
  random_texts = []; // Will be assigned in init

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

    // Set the skybox color to a subtle green
    SharedEnvironmentMapData.set_skybox_color([0.7, 1.0, 0.8, 1]);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set initial camera view
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [10.0, 10.0, 15.0];
    view_data.view_rotation = quat.fromEuler(quat.create(), 0, -180, 0); // Example rotation

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 2.5;
    light_fragment_view.position = [10, 30, 10];
    light_fragment_view.active = true;

    // Get Exo-Medium font for potential text elements
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    // Assign the combined presets to random_texts so the title can use them too
    this.random_texts = this.combined_text_presets;

    // Create the grid of text entities
    const counts_x = this.grid_entity_counts.x;
    const counts_y = this.grid_entity_counts.y;
    const counts_z = this.grid_entity_counts.z;

    const center_offset_x = (counts_x - 1) * 0.5;
    const center_offset_y = (counts_y - 1) * 0.5;
    const center_offset_z = (counts_z - 1) * 0.5;

    for (let ix = 0; ix < counts_x; ix++) {
      for (let iy = 0; iy < counts_y; iy++) {
        for (let iz = 0; iz < counts_z; iz++) {
          const pos = [
            (ix - center_offset_x) * this.grid_spacing.x,
            (iy - center_offset_y) * this.grid_spacing.y,
            (iz - center_offset_z) * this.grid_spacing.z,
          ];

          const grid_entity = spawn_mesh_entity(
            pos,
            [0, 0, 0],
            [1.0, 1.0, 1.0],
            Mesh.quad(),
            font_object.material,
            null,
            [],
            true,
            EntityFlags.NO_AABB_UPDATE | EntityFlags.IGNORE_PARENT_SCALE
          );

          const text_frag = EntityManager.add_fragment(grid_entity, TextFragment);
          text_frag.font = font_id;
          text_frag.font_size = 10; // Smaller font size for grid
          text_frag.text_color = [
            Math.random() * 0.5 + 0.5,
            Math.random() * 0.5 + 0.5,
            Math.random() * 0.5 + 0.5,
            1.0,
          ]; // Brighter random colors
          text_frag.text_emissive = 0.7;
          text_frag.text = this.random_texts[Math.floor(Math.random() * this.random_texts.length)]; // Initial random text

          this.entities.push(grid_entity);
          this.grid_text_entities.push(grid_entity);
        }
      }
    }

    log(`[${this.name}] Initialized with ${this.entities.length} entities.`);
  }

  cleanup() {
    log(`[${this.name}] Cleaning up...`);
    for (let i = 0; i < this.entities.length; i++) {
      delete_entity(this.entities[i]);
    }
    this.entities.length = 0;
    this.grid_text_entities = []; // Clear the specific list too

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
    log(`[${this.name}] Cleanup complete.`);
  }

  pre_update(delta_time) {
    super.pre_update(delta_time);

    // Timer-based update for the grid entities
    this.text_update_timer += delta_time;
    if (this.text_update_timer >= this.text_update_interval) {
      this.text_update_timer -= this.text_update_interval; // Carry over excess time

      if (this.grid_text_entities && this.grid_text_entities.length > 0) {
        const num_to_update = Math.min(
          this.num_entities_to_update_per_cycle,
          this.grid_text_entities.length
        );

        // Create a Set of indices to update to ensure we update unique entities if num_to_update < total
        const indices_to_update = new Set();
        while (
          indices_to_update.size < num_to_update &&
          indices_to_update.size < this.grid_text_entities.length
        ) {
          indices_to_update.add(Math.floor(Math.random() * this.grid_text_entities.length));
        }

        for (const entity_index of indices_to_update) {
          const entity_to_update = this.grid_text_entities[entity_index];

          const random_text_index = Math.floor(Math.random() * this.random_texts.length);
          const new_text = this.random_texts[random_text_index];

          const tfv = EntityManager.get_fragment(entity_to_update, TextFragment);
          if (tfv && new_text.length > 0) {
            tfv.text = new_text;
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------
// =============================== Voxel Terrain Scene ===============================
// ------------------------------------------------------------------------------------

export class VoxelTerrainScene extends Scene {
  name = "VoxelTerrainScene";
  entities = [];
  terrain_material_id = null;
  cube_mesh = null;

  init(parent_context) {
    super.init(parent_context);

    const freeform_arcball = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball.set_scene(this);

    SharedEnvironmentMapData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.png",
      "engine/textures/gradientbox/nx.png",
      "engine/textures/gradientbox/ny.png",
      "engine/textures/gradientbox/py.png",
      "engine/textures/gradientbox/pz.png",
      "engine/textures/gradientbox/nz.png",
    ]);
    SharedEnvironmentMapData.set_skybox_color([0.5, 0.7, 0.5, 1]);

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [20.7373, 54.0735, 68.58896];
    view_data.view_rotation = [-0.036352589, 0.94788336, -0.25605953, -0.13457019];

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 2.5;
    light_fragment_view.position = [10, 30, 70];
    light_fragment_view.active = true;

    // Create terrain material
    const terrain_material = StandardMaterial.create("TerrainMaterial");
    this.terrain_material_id = terrain_material.material_id;

    {
      let dirt_albedo = Texture.load(["engine/textures/voxel/dirt_albedo.jpg"], {
        name: "dirt_albedo",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "dirt_albedo",
      });
      let dirt_roughness = Texture.load(["engine/textures/voxel/dirt_roughness.jpg"], {
        name: "dirt_roughness",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "dirt_roughness",
      });

      terrain_material.set_albedo([0.7, 0.7, 0.7, 1.0], dirt_albedo);
      terrain_material.set_roughness(0.5, dirt_roughness);
      terrain_material.set_emission(0.2);
      terrain_material.set_tiling(2.0, 2.0);
    }

    // Create cube mesh for voxels
    this.cube_mesh = Mesh.cube();

    // Terrain parameters - Perlin-based fractal noise
    const grid_width = 150;
    const grid_depth = 150;
    const block_size = 1.0;
    const base_frequency = 0.05;
    const height_scale = 20.0;
    const height_offset = 10.0;
    const octaves = 5;
    const persistence = 0.5;

    // Build permutation table for Perlin noise
    const perlin_perm = new Array(512);
    {
      const p = new Array(256);
      for (let i = 0; i < 256; i++) p[i] = i;
      for (let i = 255; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [p[i], p[j]] = [p[j], p[i]];
      }
      for (let i = 0; i < 512; i++) perlin_perm[i] = p[i & 255];
    }

    const fade_function = (t) => t * t * t * (t * (t * 6 - 15) + 10);
    const lerp_function = (a, b, t) => a + t * (b - a);
    const grad_function = (hash, x, y) => {
      switch (hash & 3) {
        case 0:
          return x + y;
        case 1:
          return -x + y;
        case 2:
          return x - y;
        case 3:
          return -x - y;
      }
    };

    function perlin_noise(x, y) {
      const xi = Math.floor(x) & 255;
      const yi = Math.floor(y) & 255;
      const xf = x - Math.floor(x);
      const yf = y - Math.floor(y);
      const u = fade_function(xf);
      const v = fade_function(yf);

      const aa = perlin_perm[xi + perlin_perm[yi]];
      const ab = perlin_perm[xi + perlin_perm[yi + 1]];
      const ba = perlin_perm[xi + 1 + perlin_perm[yi]];
      const bb = perlin_perm[xi + 1 + perlin_perm[yi + 1]];

      const x1 = lerp_function(grad_function(aa, xf, yf), grad_function(ba, xf - 1, yf), u);
      const x2 = lerp_function(grad_function(ab, xf, yf - 1), grad_function(bb, xf - 1, yf - 1), u);
      return lerp_function(x1, x2, v);
    }

    function fractal_noise(x, y) {
      let total = 0;
      let freq = base_frequency;
      let amp = 1;
      let max = 0;
      for (let o = 0; o < octaves; o++) {
        total += perlin_noise(x * freq, y * freq) * amp;
        max += amp;
        amp *= persistence;
        freq *= 2;
      }
      return total / max;
    }

    // Precompute heights
    const heights = new Array(grid_width);
    let total_blocks = 0;
    for (let xi = 0; xi < grid_width; xi++) {
      heights[xi] = new Array(grid_depth);
      for (let zi = 0; zi < grid_depth; zi++) {
        const noise_val = fractal_noise(xi, zi) * height_scale + height_offset;
        const height = Math.floor(noise_val);
        heights[xi][zi] = height;
        total_blocks += height;
      }
    }

    // Spawn instanced cube entity for terrain
    const terrain_entity = spawn_mesh_entity(
      [0, 0, 0],
      [0, 0, 0, 1],
      [block_size, block_size, block_size],
      this.cube_mesh,
      this.terrain_material_id,
      null,
      [],
      true,
      EntityFlags.NO_AABB_UPDATE | EntityFlags.IGNORE_PARENT_SCALE
    );
    EntityManager.set_entity_instance_count(terrain_entity, total_blocks);
    this.entities.push(terrain_entity);

    // Assign transforms for each voxel
    let block_index = 0;
    for (let xi = 0; xi < grid_width; xi++) {
      for (let zi = 0; zi < grid_depth; zi++) {
        for (let yi = 0; yi < heights[xi][zi]; yi++) {
          if (block_index >= total_blocks) {
            break;
          }

          const pos = [
            (xi - grid_width / 2) * block_size * 2.0,
            (yi) * block_size * 2.0,
            (zi - grid_depth / 2) * block_size * 2.0,
          ];
          const view = EntityManager.get_fragment(terrain_entity, TransformFragment, block_index);
          view.position = pos;
          block_index++;
        }
      }
    }

    // Get Exo-Medium font
    const font_id = Name.from("Exo-Medium");
    const font_object = FontCache.get_font_object(font_id);

    // Add a title text entity
    const text_entity = spawn_mesh_entity(
      [-10, 55, 0],
      [0, 0, 0, 1],
      [0.5, 0.5, 0.5],
      Mesh.quad(),
      font_object.material,
      null,
      [],
      true,
      EntityFlags.NO_AABB_UPDATE | EntityFlags.IGNORE_PARENT_SCALE
    );
    const text_fragment_view = EntityManager.add_fragment(text_entity, TextFragment);
    text_fragment_view.font = font_id;
    text_fragment_view.font_size = 32;
    text_fragment_view.text_color = [1, 1, 1, 1];
    text_fragment_view.text_emissive = 1;
    text_fragment_view.text = "Voxel Terrain Scene";
    this.entities.push(text_entity);

    log(`[${this.name}] Initialized with ${total_blocks} blocks.`);
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.remove_layer(FreeformArcballControlProcessor);
    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);
  }
}

// ------------------------------------------------------------------------------------
// =============================== ObjectPaintingScene ====================================
// ------------------------------------------------------------------------------------
const info_panel_config = {
  layout: "column",
  gap: 4,
  y: 100,
  anchor_y: "bottom",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 600,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const info_label_config = {
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

export class ObjectPaintingScene extends Scene {
  entities = [];
  sphere_mesh = null;
  brush_material_id = null;
  brush_entity = null;

  // --- Configurable parameters ---
  brush_radius = 2.0; // radius of the sphere brush
  brush_emit_intensity = 1.0; // material emission
  paint_rate = 0.25; // seconds between paint ticks
  spawn_count = 256; // objects per tick
  spawn_radius = 5.0; // radius of random paint sphere
  // -------------------------------

  last_paint_timer = 0;

  init(parent_context) {
    super.init(parent_context);

    // Arcball camera control
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Skybox + view
    SharedEnvironmentMapData.set_skybox("default_scene_skybox", [
      "engine/textures/simple_skybox/px.png",
      "engine/textures/simple_skybox/nx.png",
      "engine/textures/simple_skybox/ny.png",
      "engine/textures/simple_skybox/py.png",
      "engine/textures/simple_skybox/pz.png",
      "engine/textures/simple_skybox/nz.png",
    ]);
    SharedEnvironmentMapData.set_skybox_color([1, 1, 1, 1]);

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0, 0, 10];
    view_data.view_rotation = [0, 0, 0, 1];

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 2.5;
    light_fragment_view.position = [10, 30, 10];
    light_fragment_view.active = true;

    // Load sphere mesh & create transparent/emissive brush material
    this.sphere_mesh = Mesh.from_gltf("engine/models/sphere/sphere.gltf");

    const object_material1 = StandardMaterial.create("ObjectPaintingObjectMaterial");
    this.object_material1_id = object_material1.material_id;
    const object_material2 = StandardMaterial.create("ObjectPaintingObjectMaterial2");
    this.object_material2_id = object_material2.material_id;
    const object_material3 = StandardMaterial.create("ObjectPaintingObjectMaterial3");
    this.object_material3_id = object_material3.material_id;

    object_material1.set_albedo([0.1, 0.1, 0.3, 1]);
    object_material1.set_emission(0.3);
    object_material1.set_roughness(0.5);
    object_material1.set_tiling(2.0, 2.0);

    object_material2.set_albedo([0.3, 0.0, 0.0, 1]);
    object_material2.set_emission(0.3);
    object_material2.set_roughness(0.5);
    object_material2.set_tiling(2.0, 2.0);

    object_material3.set_albedo([0.0, 0.3, 0.0, 1]);
    object_material3.set_emission(0.3);
    object_material3.set_roughness(0.5);
    object_material3.set_tiling(2.0, 2.0);
  }

  update(delta_time) {
    super.update(delta_time);

    // Move brush to follow mouse
    const world_pos = UI.UIContext.input_state.world_position;
    const view = SharedViewBuffer.get_view_data(0);
    const view_dir = view.forward;
    const paint_pos = vec3.scaleAndAdd(vec3.create(), world_pos, view_dir, 50);

    // While Space key is held, paint objects every paint_rate seconds
    if (InputProvider.get_state(InputKey.K_Space)) {
      this.last_paint_timer += delta_time;
      if (this.last_paint_timer >= this.paint_rate) {
        this.last_paint_timer -= this.paint_rate;
        for (let i = 0; i < this.spawn_count; i++) {
          // uniform random point in sphere
          const r = this.spawn_radius * Math.cbrt(Math.random());
          const theta = 2 * Math.PI * Math.random();
          const phi = Math.acos(2 * Math.random() - 1);
          const x = paint_pos[0] + r * Math.sin(phi) * Math.cos(theta);
          const y = paint_pos[1] + r * Math.sin(phi) * Math.sin(theta);
          const z = paint_pos[2] + r * Math.cos(phi);

          // spawn a sphere instance
          const entity = spawn_mesh_entity(
            [x, y, z],
            quat.create(),
            [0.1 + Math.random() * 0.3, 0.1 + Math.random() * 0.3, 0.1 + Math.random() * 0.3],
            this.sphere_mesh,
            [this.object_material1_id, this.object_material2_id, this.object_material3_id][
              Math.floor(Math.random() * 3)
            ],
            null,
            [],
            true,
            EntityFlags.NO_AABB_UPDATE | EntityFlags.IGNORE_PARENT_SCALE
          );
          this.entities.push(entity);
        }
      }
    } else {
      // reset timer so we can paint immediately on next hold
      this.last_paint_timer = this.paint_rate;
    }

    // show fixed center cursor
    this.render_ui();
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.remove_layer(FreeformArcballControlProcessor);
    super.cleanup();
  }

  // draw a fixed, 2D crosshair at screen center
  render_ui() {
    const { width, height } = UI.UIContext.canvas_size;
    const cx = width * 0.5;
    const cy = height * 0.5;

    // vertical line
    UI.panel(
      {
        x: cx - 1,
        y: cy - 10,
        width: 2,
        height: 20,
        background_color: "#FFFFFF",
        dont_consume_cursor_events: true,
      },
      () => {}
    );

    // horizontal line
    UI.panel(
      {
        x: cx - 10,
        y: cy - 1,
        width: 20,
        height: 2,
        background_color: "#FFFFFF",
        dont_consume_cursor_events: true,
      },
      () => {}
    );

    // help text overlay
    UI.panel(info_panel_config, () => {
      UI.label("Hold [Space] to Paint objects", info_label_config);
    });
  }
}

// Simple Test Gym Scene - Cornell-box blockout for DDGI testing
export class GITestScene extends Scene {
  name = "GITestScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    const room_size = 10.0;
    const wall_thickness = 0.1;
    const ambient_emissive = 0.03;

    // camera arcball
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // white skybox
    SharedEnvironmentMapData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.png",
      "engine/textures/gradientbox/nx.png",
      "engine/textures/gradientbox/ny.png",
      "engine/textures/gradientbox/py.png",
      "engine/textures/gradientbox/pz.png",
      "engine/textures/gradientbox/nz.png",
    ]);
    SharedEnvironmentMapData.set_skybox_color([1, 1, 1, 1]);

    // camera
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0, 13, 40];
    view_data.view_rotation = [0.0005166, 0.9986818, -0.027326133, 0.0188794];

    // directional light
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 0.5;
    light_fragment_view.position = [25, 45, 15];
    light_fragment_view.active = true;

    // materials
    const wall_material = StandardMaterial.create("testgym_wall_material");
    const wall_material_id = wall_material.material_id;
    wall_material.set_albedo([1, 1, 1, 1]);
    wall_material.set_emission(ambient_emissive);
    wall_material.set_roughness(1.0);

    const red_material = StandardMaterial.create("testgym_red_material");
    const red_material_id = red_material.material_id;
    red_material.set_albedo([1, 0.2, 0.2, 1]);
    red_material.set_emission(ambient_emissive);

    const blue_material = StandardMaterial.create("testgym_blue_material");
    const blue_material_id = blue_material.material_id;
    blue_material.set_albedo([0.2, 0.2, 1, 1]);
    blue_material.set_emission(ambient_emissive);

    const gray_material = StandardMaterial.create("testgym_gray_material");
    const gray_material_id = gray_material.material_id;
    gray_material.set_albedo([0.5, 0.5, 0.5, 1]);
    gray_material.set_emission(ambient_emissive);

    // meshes
    const cube_mesh = Mesh.cube();
    const sphere_mesh = Mesh.from_gltf("engine/models/sphere/sphere.gltf");

    // Cornell-box walls (tight box)
    {
      // floor top at y=0
      const floor = spawn_mesh_entity(
        [0, -wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(floor);

      // ceiling bottom at y=room_size
      const ceiling = spawn_mesh_entity(
        [0, room_size * 2.0 + wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling);

      // back wall inner surface at z=-room_size/2
      const back_wall = spawn_mesh_entity(
        [0, room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall);

      // left wall inner surface at x=-room_size/2
      const left_wall = spawn_mesh_entity(
        [-room_size - wall_thickness, room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        red_material_id
      );
      this.entities.push(left_wall);

      // right wall inner surface at x=+room_size/2
      const right_wall = spawn_mesh_entity(
        [room_size + wall_thickness, room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        blue_material_id
      );
      this.entities.push(right_wall);

      // blockout "buildings"
      const building_data = [
        { mesh: cube_mesh, position: [-3, 2, -3], scale: [1, 2, 1], material_id: gray_material_id },
        { mesh: cube_mesh, position: [2, 3, -2], scale: [1, 3, 1], material_id: red_material_id },
        {
          mesh: sphere_mesh,
          position: [-1, 1.5, 2],
          scale: [1.5, 1.5, 1.5],
          material_id: blue_material_id,
        },
      ];
      for (const item of building_data) {
        const b = spawn_mesh_entity(
          item.position,
          [0, 0, 0, 1],
          item.scale,
          item.mesh,
          item.material_id
        );
        this.entities.push(b);
      }
    }
    // Additional Cornell box with different wall colors at x = -30
    {
      const offset_x = -30.0;

      const left_wall_material_second = StandardMaterial.create("testgym_left_material_second");
      const left_wall_material_second_id = left_wall_material_second.material_id;
      left_wall_material_second.set_albedo([0, 1, 0, 1]);
      left_wall_material_second.set_emission(ambient_emissive);

      const right_wall_material_second = StandardMaterial.create("testgym_right_material_second");
      const right_wall_material_second_id = right_wall_material_second.material_id;
      right_wall_material_second.set_albedo([1, 0, 1, 1]);
      right_wall_material_second.set_emission(ambient_emissive);

      // spawn elements for second box
      const floor_second = spawn_mesh_entity(
        [offset_x, -wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id 
      );
      this.entities.push(floor_second);

      const ceiling_second = spawn_mesh_entity(
        [offset_x, room_size * 2.0 + wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling_second);

      const back_wall_second = spawn_mesh_entity(
        [offset_x, room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall_second);

      const left_wall_second = spawn_mesh_entity(
        [offset_x - (room_size + wall_thickness), room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        left_wall_material_second_id
      );
      this.entities.push(left_wall_second);

      const right_wall_second = spawn_mesh_entity(
        [offset_x + room_size + wall_thickness, room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        right_wall_material_second_id
      );
      this.entities.push(right_wall_second);

      const building_data_second = [
        {
          mesh: cube_mesh,
          position: [offset_x - 3, 2, -3],
          scale: [1, 2, 1],
          material_id: left_wall_material_second_id,
        },
        {
          mesh: sphere_mesh,
          position: [offset_x + 2, 1.5, 2],
          scale: [1.5, 1.5, 1.5],
          material_id: right_wall_material_second_id,
        },
      ];
      for (const item of building_data_second) {
        const b = spawn_mesh_entity(
          item.position,
          [0, 0, 0, 1],
          item.scale,
          item.mesh,
          item.material_id
        );
        this.entities.push(b);
      }
    }
    // Additional Cornell box with different wall colors at x = 30
    {
      const offset_x = 30.0;

      const left_wall_material_third = StandardMaterial.create("testgym_left_material_third");
      const left_wall_material_third_id = left_wall_material_third.material_id;
      left_wall_material_third.set_albedo([1, 0.5, 0, 1]);
      left_wall_material_third.set_emission(ambient_emissive);

      const right_wall_material_third = StandardMaterial.create("testgym_right_material_third", {}, { family: MaterialFamilyType.Transparent });
      const right_wall_material_third_id = right_wall_material_third.material_id;
      right_wall_material_third.set_albedo([0.5, 0, 0.5, 0.3]);
      right_wall_material_third.set_emission(ambient_emissive);

      const floor_third = spawn_mesh_entity(
        [offset_x, -wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(floor_third);

      const ceiling_third = spawn_mesh_entity(
        [offset_x, room_size * 2.0 + wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling_third);

      const back_wall_third = spawn_mesh_entity(
        [offset_x, room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall_third);

      const left_wall_third = spawn_mesh_entity(
        [offset_x - (room_size + wall_thickness), room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        left_wall_material_third_id
      );
      this.entities.push(left_wall_third);

      const right_wall_third = spawn_mesh_entity(
        [offset_x + room_size + wall_thickness, room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        right_wall_material_third_id
      );
      this.entities.push(right_wall_third);

      const building_data_third = [
        {
          mesh: sphere_mesh,
          position: [offset_x - 2, 1.5, -2],
          scale: [1.5, 1.5, 1.5],
          material_id: left_wall_material_third_id,
        },
        {
          mesh: cube_mesh,
          position: [offset_x + 3, 2, 3],
          scale: [2, 1, 2],
          material_id: right_wall_material_third_id,
        },
      ];
      for (const item of building_data_third) {
        const b = spawn_mesh_entity(
          item.position,
          [0, 0, 0, 1],
          item.scale,
          item.mesh,
          item.material_id
        );
        this.entities.push(b);
      }
    }
  }

  cleanup() {
    for (const e of this.entities) {
      delete_entity(e);
    }
    this.remove_layer(FreeformArcballControlProcessor);
    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);
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
// =============================== Shadow Test Scene ==================================
// ------------------------------------------------------------------------------------

export class ShadowTestScene extends Scene {
  name = "ShadowTestScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    // Add arcball camera control
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.move_speed = 75.0;
    freeform_arcball_control_processor.set_scene(this);

    // Configure skybox
    SharedEnvironmentMapData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.png",
      "engine/textures/gradientbox/nx.png",
      "engine/textures/gradientbox/ny.png",
      "engine/textures/gradientbox/py.png",
      "engine/textures/gradientbox/pz.png",
      "engine/textures/gradientbox/nz.png",
    ]);
    SharedEnvironmentMapData.set_skybox_color([1, 1, 1, 1]);

    // Position the camera high above the city
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [47, 352, 770];
    view_data.view_rotation = [0.0, 0.967463, -0.23873913, 0.0];
    view_data.far = 4000.0;

    // Create a sun-like directional light
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 5.0;
    light_fragment_view.position = [100, 300, 100];
    light_fragment_view.active = true;

    // Ground material
    const ground_material = StandardMaterial.create("shadow_ground_material");
    const ground_material_id = ground_material.material_id;
    ground_material.set_albedo([0.5, 0.5, 0.5, 1]);
    ground_material.set_roughness(1.0);

    // Building material
    const building_material = StandardMaterial.create("shadow_building_material");
    const building_material_id = building_material.material_id;
    building_material.set_albedo([0.35, 0.35, 0.35, 1]);
    building_material.set_roughness(0.8);

    // Shared cube mesh
    const cube_mesh = Mesh.cube();
    const quad_mesh = Mesh.quad();

    // Create an expansive ground plane
    const ground_plane_size = 1000.0;
    const ground_entity = spawn_mesh_entity(
      [0.0, 0.0, 0.0],
      quat.fromEuler(quat.create(), -90.0, 0.0, 0.0),
      [ground_plane_size, 1.0, ground_plane_size],
      quad_mesh,
      ground_material_id
    );
    this.entities.push(ground_entity);

    // Procedurally generate a dense grid of buildings
    const grid_size = 80;           // 80 × 80 buildings
    const building_spacing = 20.0;  // distance between building centres
    const building_base_size = 6.0; // footprint of each building

    const building_entity = spawn_mesh_entity(
      [0.0, 0.0, 0.0, 1.0],
      quat.fromEuler(quat.create(), 0.0, 0.0, 0.0),
      [0.0, 0.0, 0.0],
      cube_mesh,
      building_material_id
    );
    EntityManager.set_entity_instance_count(building_entity, grid_size * grid_size);
    this.entities.push(building_entity);

    const half_grid = (grid_size - 1) * building_spacing * 0.5;

    let instance_index = 0;
    for (let gx = 0; gx < grid_size; gx++) {
      for (let gz = 0; gz < grid_size; gz++) {
        // Randomised height to create varied skyline
        const height = 10.0 + Math.random() * 90.0; // between 10 and 100 units

        const position = [
          gx * building_spacing - half_grid,
          height * 0.5,
          gz * building_spacing - half_grid,
        ];
        const scale = [building_base_size, height, building_base_size];

        const transform_fragment = EntityManager.get_fragment(building_entity, TransformFragment, instance_index);
        transform_fragment.position = position;
        transform_fragment.scale = scale;

        const visibility_fragment = EntityManager.get_fragment(building_entity, VisibilityFragment, instance_index);
        visibility_fragment.occluder = 0;

        instance_index++;
      }
    }

    log(`[${this.name}] Spawned ${this.entities.length} entities.`);
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.remove_layer(FreeformArcballControlProcessor);
    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);
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
  const textures_scene = new TexturesScene("TexturesScene");
  const solar_ecs_scene = new SolarECSTestScene("SolarECSTestScene");
  const voxel_terrain_scene = new VoxelTerrainScene("VoxelTerrainScene");
  const object_painting_scene = new ObjectPaintingScene("ObjectPaintingScene");
  const gi_test_scene = new GITestScene("GITestScene");
  const shadow_test_scene = new ShadowTestScene("ShadowTestScene");

  const scene_switcher = new SceneSwitcher("SceneSwitcher");
  //await scene_switcher.add_scene(solar_ecs_scene);
  //await scene_switcher.add_scene(textures_scene);
  //await scene_switcher.add_scene(aabb_scene);
  //await scene_switcher.add_scene(rendering_scene);
  //await scene_switcher.add_scene(ml_scene);
  //await scene_switcher.add_scene(voxel_terrain_scene);
  //await scene_switcher.add_scene(object_painting_scene);
  await scene_switcher.add_scene(gi_test_scene);
  //await scene_switcher.add_scene(shadow_test_scene);

  await simulator.add_sim_layer(scene_switcher);

  simulator.run();
})();

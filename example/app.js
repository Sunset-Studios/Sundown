import { Simulator } from "../engine/src/core/simulator.js";
import SimulationCore from "../engine/src/core/simulation_core.js";
import { FragmentGpuBuffer } from "../engine/src/core/ecs/solar/memory.js";
import { SimulationLayer } from "../engine/src/core/simulation_layer.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { Scene } from "../engine/src/core/scene.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { InputKey } from "../engine/src/input/input_types.js";
import { PostProcessStack } from "../engine/src/renderer/post_process_stack.js";
import { BVHDebugRenderer } from "../engine/src/core/subsystems/bvh_debug_renderer.js";
import { BVHRaycast } from "../engine/src/acceleration/bvh_raycast.js";
import { ComputeTaskQueue } from "../engine/src/renderer/compute_task_queue.js";
import { TransformFragment } from "../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { StaticMeshFragment } from "../engine/src/core/ecs/fragments/static_mesh_fragment.js";
import { VisibilityFragment } from "../engine/src/core/ecs/fragments/visibility_fragment.js";
import { LightType } from "../engine/src/core/minimal.js";
import { StandardMaterial } from "../engine/src/renderer/material.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../engine/src/core/ecs/entity_utils.js";
import { TextureChannel } from "../engine/src/renderer/renderer_types.js";
import { profile_scope } from "../engine/src/utility/performance.js";
import { log } from "../engine/src/utility/logging.js";
import { vec3, vec4, quat } from "gl-matrix";

import * as UI from "../engine/src/ui/2d/immediate.js";
import * as UI3D from "../engine/src/ui/3d/immediate.js";

import { Layer, TrainingContext } from "../engine/src/ml/layer.js";
import { LayerType } from "../engine/src/ml/ml_types.js";
import { Input } from "../engine/src/ml/layers/input.js";
import { MasterMind } from "../engine/src/ml/mastermind.js";
import { Tensor, TensorInitializer } from "../engine/src/ml/math/tensor.js";
import { Adam } from "../engine/src/ml/optimizers/adam.js";
import example_cvar_config from "./config/cvars.js";

function world_label(text, position, config = {}) {

  UI3D.panel(
    {
      position,
      billboard: config.billboard ?? true,
      width: config.width ?? 1.0,
      height: config.height ?? 1.0,
      unit_scale: Number(config.unit_scale ?? 1.0),
      pivot: config.pivot ?? [0.5, 0.5],
      padding: 0,
      background_color: config.background_color ?? [0, 0, 0, 0],
      z_order: config.z_order ?? 1,
    },
    () => {
      UI3D.label(text, {
        width: "100%",
        height: "100%",
        font: config.font ?? "Exo-Medium",
        text_color: config.text_color ?? [1, 1, 1, 1],
        text_align: config.text_align ?? "center",
        text_valign: config.text_valign ?? "middle",
        text_emissive: Number(config.text_emissive ?? 1.0),
      });
    }
  );
}

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
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1, 1];
    light_fragment_view.intensity = 30.0;
    light_fragment_view.position = [50, 15, 50, 1];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;
    light_fragment_view.shadow_casting = 0;

    // Create a sphere mesh and add it to the scene
    const mesh = Mesh.from_gltf("engine/models/cube/cube.gltf");

    // Create a default material
    const default_material = StandardMaterial.create("MyMaterial");
    const default_material_id = default_material.material_id;

    {
      let dirt_albedo = {
        name: "dirt_albedo",
        paths: ["engine/textures/voxel/dirt_albedo.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "dirt_albedo",
      };
      let dirt_roughness = {
        name: "dirt_roughness",
        paths: ["engine/textures/voxel/dirt_roughness.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "dirt_roughness",
      };

      default_material.sample_albedo(dirt_albedo);
      default_material.sample_roughness(dirt_roughness);
      default_material.set_tiling(2.0, 2.0);
    }

    // Create a 3D grid of sphere entities
    const grid_size = 105; // 100x100x10 grid
    const grid_layers = 105;
    const spacing = 5; // 2 units apart

    const sphere = spawn_mesh_entity(
      [0, 0, 0],
      [0, 0, 0],
      [0.5, 0.5, 0.5],
      mesh,
      default_material_id
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

    log(`[${this.name}] Initialized with ${sphere_count} cubes.`);

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
    SharedEnvironmentData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.jpg",
      "engine/textures/gradientbox/nx.jpg",
      "engine/textures/gradientbox/ny.jpg",
      "engine/textures/gradientbox/py.jpg",
      "engine/textures/gradientbox/pz.jpg",
      "engine/textures/gradientbox/nz.jpg",
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

      Layer.create(
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
  swaying_cube_entity = null;
  swaying_cube_base_pos = [0, 10, 50];

  init(parent_context) {
    super.init(parent_context);

    // Set the skydome for this scene.
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Reset view to a good position for the BVH scene
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [39.198, 14.0851, 78.60858];
    view_data.view_rotation = [-0.0203683, 0.9771718, -0.179953, -0.110603];
    view_data.far = 10000.0;

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1, 1];
    light_fragment_view.intensity = 7;
    light_fragment_view.position = [50, 20, 30];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Load metal plane material
    {
      let floor_albedo = {
        name: "floor_albedo",
        paths: ["engine/textures/rubber_floor/Diffuse.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "floor_albedo",
      };
      let floor_normal = {
        name: "floor_normal",
        paths: ["engine/textures/rubber_floor/Normal.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "floor_normal",
      };
      let floor_roughness = {
        name: "floor_roughness",
        paths: ["engine/textures/rubber_floor/ARM.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "floor_roughness",
      };

      // Create a default material
      const default_plane_material = StandardMaterial.create("TexturesPlaneMaterial");
      this.default_plane_material_id = default_plane_material.material_id;
      default_plane_material.sample_albedo(floor_albedo);
      default_plane_material.sample_normal(floor_normal);
      default_plane_material.sample_roughness(floor_roughness, TextureChannel.G);
      default_plane_material.set_metallic(0.2);
      default_plane_material.set_emission(0.1);
      default_plane_material.set_tiling(150.0);
    }

    // Load wall material
    {
      let wall_albedo = {
        name: "wall_albedo",
        paths: ["engine/textures/wall/wall_albedo.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_albedo",
      };
      let wall_normal = {
        name: "wall_normal",
        paths: ["engine/textures/wall/wall_normal.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_normal",
      };
      let wall_roughness = {
        name: "wall_roughness",
        paths: ["engine/textures/wall/wall_roughness.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_roughness",
      };
      let wall_ao = {
        name: "wall_ao",
        paths: ["engine/textures/wall/wall_ao.jpg"],
        name: "wall_ao",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_ao",
      };

      // Create a default material
      const wall_material = StandardMaterial.create("TexturesWallMaterial");
      this.wall_material_id = wall_material.material_id;
      wall_material.sample_albedo(wall_albedo);
      wall_material.sample_normal(wall_normal);
      wall_material.sample_roughness(wall_roughness);
      wall_material.sample_ao(wall_ao);
      wall_material.set_metallic(0.9);
      wall_material.set_emission(0.1);
      wall_material.set_tiling(2.0);
    }

    // Create a sphere mesh
    this.sphere_mesh = Mesh.sphere();

    // Create a cube mesh
    this.cube_mesh = Mesh.cube();

    // Setup the world plane
    this.setup_world_plane();

    // Setup sphere entity
    this.setup_sphere_entity();

    // Setup the barrels
    this.setup_barrels();
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

    world_label("Textures Test Scene", [0, 20, 0], {
      width: 40.0,
      height: 10.0,
      billboard: false
    });

    // Animate the swaying cube
    if (this.swaying_cube_entity) {
      const sway_amplitude = 12; // units left/right
      const sway_frequency = 0.25; // Hz
      const t = performance.now() * 0.001; // seconds
      const x =
        this.swaying_cube_base_pos[0] + Math.sin(t * Math.PI * 2 * sway_frequency) * sway_amplitude;
      const y = this.swaying_cube_base_pos[1];
      const z = this.swaying_cube_base_pos[2];
      const tf = EntityManager.get_fragment(this.swaying_cube_entity, TransformFragment);
      if (tf) {
        tf.position = [x, y, z];
      }
    }
  }

  setup_world_plane() {
    // Create a plane entity
    const plane = spawn_mesh_entity(
      [0, 0, 0],
      quat.fromEuler(quat.create(), 0.0, 0, 0),
      [1000, 5.0, 1000],
      this.cube_mesh,
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

  setup_barrels() {
    const barrel_mesh = Mesh.from_gltf("engine/models/barrel/Barrel.gltf");

    const num_barrels = 500;
    const barrel_spawn_range = 1000; // spread barrels across the entire ground plane
    const barrel_entity = spawn_mesh_entity(
      [0, 0, 0],
      [0, 0, 0, 1],
      [1.0, 1.0, 1.0],
      barrel_mesh,
      0 // GLTF sets the material id
    );
    this.entities.push(barrel_entity);

    // Set the instance count
    EntityManager.set_entity_instance_count(barrel_entity, num_barrels);

    // Set the instance data
    for (let i = 0; i < num_barrels; i++) {
      const x = (Math.random() - 0.5) * barrel_spawn_range;
      const z = (Math.random() - 0.5) * barrel_spawn_range;
      const y = 5.0; // Place on top of the plane
      const scale = 1.0 + Math.random() * 10.0;
      const view = EntityManager.get_fragment(barrel_entity, TransformFragment, i);
      view.position = [x, y, z];
      view.scale = [scale, scale, scale];
      view.rotation = quat.fromEuler(quat.create(), 0, Math.random() * 360, 0);
    }
  }
}

// ------------------------------------------------------------------------------------
// =============================== BVH Scene =======================================
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

export class BVHScene extends Scene {
  name = "BVHScene";
  show_ui = false;
  entities = [];
  selected_entity = null;
  last_ray_origin = null;
  last_ray_direction = null;
  last_ray = null;

  init(parent_context) {
    super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skydome
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // Reset view to a good position for the BVH scene
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [47.0751, 55.28902, 106.885414];
    view_data.view_rotation = [-0.023805, 0.97379, -0.190533, -0.121665];

    this.aabb_tree_debug_renderer = this.get_layer(BVHDebugRenderer);

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 10;
    light_fragment_view.position = [50, 20, 50];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = true;
    this.entities.push(light_entity);

    // Create a default material
    const default_material = StandardMaterial.create("BVHDefaultMaterial");
    this.default_material_id = default_material.material_id;
    default_material.set_albedo([0.5, 0.5, 0.5, 1]);
    default_material.set_normal([0, 1, 0, 1]);
    default_material.set_roughness(0.5);
    default_material.set_metallic(0.5);
    default_material.set_emission(0.1);

    // Create a default material for the selected entity
    const selected_entity_material = StandardMaterial.create("BVHSelectedEntityMaterial");
    this.selected_entity_material_id = selected_entity_material.material_id;
    selected_entity_material.set_albedo([1.0, 0.3, 0.3, 1]);
    selected_entity_material.set_emission(1.0);

    // Create a sphere mesh
    this.sphere_mesh = Mesh.sphere();

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

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);

    world_label("BVH Test Scene", [10, 65, 25], {
      width: 40.0,
      height: 10.0,
      billboard: false,
    });

    this.handle_input();

    // Run a raycast every frame from mouse position
    if (this.entities.length > 0) {
      this.run_raycast();
    }

    this.render_ui();
  }

  setup_entity_grid() {
    // Create a grid of entities for testing
    const grid_size = 5;
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
            this.default_material_id
          );

          this.entities.push(entity);
        }
      }
    }

    log(`[BVH] Spawned ${this.entities.length} entities`);
  }

  handle_input() {
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
    if (InputProvider.get_action(InputKey.K_Space) && this.last_ray) {
      const hit = BVHRaycast.get_hit_result(this.last_ray);

      // Create entity at hit point
      const entity = spawn_mesh_entity(
        hit.position,
        [0, 0, 0, 1],
        [0.5, 0.5, 0.5],
        Math.random() > 0.5 ? this.cube_mesh : this.sphere_mesh,
        this.default_material_id
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

    if (this.last_ray) {
      const hit = BVHRaycast.get_hit_result(this.last_ray);
      this.process_raycast_results(hit);
    }

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

    // Perform raycast based on current mode by requesting a ray from the BVHRaycast class
    this.last_ray = BVHRaycast.request_ray();
    this.last_ray.setup(this.last_ray_origin, this.last_ray_direction);
  }

  process_raycast_results(hit) {
    // Update selected entity highlighting
    const previous_selected_entity = this.selected_entity;

    // Highlight the selected entity by writing to the material buffer
    this.selected_entity = EntityManager.get_entity_from_id(hit.user_data);

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
      UI.label("Raycast Results:", stats_label_config);

      if (this.last_ray) {
        const hit = BVHRaycast.get_hit_result(this.last_ray);
        UI.label(`Hit Entity: ${hit.user_data}`, stats_label_config);
        UI.label(`Distance: ${hit.distance.toFixed(2)}`, stats_label_config);

        const pos_text = `Position: [${hit.position[0].toFixed(2)}, ${hit.position[1].toFixed(2)}, ${hit.position[2].toFixed(2)}]`;
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
        "U: Toggle UI",
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
        if (this.last_ray) {
          const hit = BVHRaycast.get_hit_result(this.last_ray);
          const entity = spawn_mesh_entity(
            hit.position,
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
        this.default_material_id
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
  instance_count_timer = 0; // Timer for instance count updates
  instance_count_update_interval = 0.5; // Time in seconds between instance count changes

  grid_entity_counts = { x: 20, y: 20, z: 20 }; // Number of entities per dimension
  grid_spacing = { x: 4.0, y: 4.0, z: 4.0 }; // Explicit spacing between entities

  grid_mesh_entities = [];
  cloud_max_instance_count = 8;
  cloud_radius = 0.9;
  cloud_entities_to_update_per_frame = 250;
  cloud_update_pass_active = false;
  cloud_updates_remaining = 0;

  init(parent_context) {
    super.init(parent_context);

    // Set the skybox for this scene.
    SharedEnvironmentData.set_skybox("default_scene_skybox", [
      "engine/textures/gradientbox/px.jpg",
      "engine/textures/gradientbox/nx.jpg",
      "engine/textures/gradientbox/ny.jpg",
      "engine/textures/gradientbox/py.jpg",
      "engine/textures/gradientbox/pz.jpg",
      "engine/textures/gradientbox/nz.jpg",
    ]);

    // Set the skybox color to a subtle green
    SharedEnvironmentData.set_skybox_color([0.7, 1.0, 0.8, 1]);

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

    // Create a uniform grid of mesh entities. Each entity owns a small instance cloud.
    const counts_x = this.grid_entity_counts.x;
    const counts_y = this.grid_entity_counts.y;
    const counts_z = this.grid_entity_counts.z;

    const mesh = Mesh.from_gltf("engine/models/cube/cube.gltf");
    const materials = [
      StandardMaterial.create("SolarECSGridMaterial_Mint", {
        albedo: [0.55, 1.0, 0.72, 1.0],
        emission: 0.05,
        roughness: 0.7,
        metallic: 0.0,
      }),
      StandardMaterial.create("SolarECSGridMaterial_Coral", {
        albedo: [1.0, 0.48, 0.36, 1.0],
        emission: 0.03,
        roughness: 0.5,
        metallic: 0.0,
      }),
      StandardMaterial.create("SolarECSGridMaterial_Ice", {
        albedo: [0.45, 0.74, 1.0, 1.0],
        emission: 0.08,
        roughness: 0.28,
        metallic: 0.05,
      }),
      StandardMaterial.create("SolarECSGridMaterial_Gold", {
        albedo: [1.0, 0.82, 0.32, 1.0],
        emission: 0.02,
        roughness: 0.36,
        metallic: 0.45,
      }),
      StandardMaterial.create("SolarECSGridMaterial_GlassGreen", {
        albedo: [0.28, 0.9, 0.64, 1.0],
        emission: 0.12,
        roughness: 0.18,
        metallic: 0.12,
      }),
    ];

    const center_offset_x = (counts_x - 1) * 0.5;
    const center_offset_y = (counts_y - 1) * 0.5;
    const center_offset_z = (counts_z - 1) * 0.5;

    for (let ix = 0; ix < counts_x; ix++) {
      for (let iy = 0; iy < counts_y; iy++) {
        for (let iz = 0; iz < counts_z; iz++) {
          const center = [
            (ix - center_offset_x) * this.grid_spacing.x,
            (iy - center_offset_y) * this.grid_spacing.y,
            (iz - center_offset_z) * this.grid_spacing.z,
          ];

          const entity = spawn_mesh_entity(
            center,
            [0, 0, 0],
            [0.18, 0.18, 0.18],
            mesh,
            materials[Math.floor(Math.random() * materials.length)].material_id
          );
          this.entities.push(entity);
          this.grid_mesh_entities.push({
            entity,
            center,
          });

          EntityManager.set_entity_instance_count(entity, this.cloud_max_instance_count);
          this.#randomize_cloud_instances(entity, center, this.cloud_max_instance_count);
        }
      }
    }

    log(
      `[${this.name}] Initialized with ${this.grid_mesh_entities.length} mesh entities and ${this.cloud_max_instance_count} max instances per entity.`
    );
  }

  cleanup() {
    log(`[${this.name}] Cleaning up...`);
    for (let i = 0; i < this.entities.length; i++) {
      delete_entity(this.entities[i]);
    }
    this.entities.length = 0;
    this.grid_mesh_entities.length = 0;
    this.instance_count_timer = 0;
    this.cloud_update_pass_active = false;
    this.cloud_updates_remaining = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
    log(`[${this.name}] Cleanup complete.`);
  }

  pre_update(delta_time) {
    super.pre_update(delta_time);

    this.instance_count_timer += delta_time;
    if (
      !this.cloud_update_pass_active &&
      this.instance_count_timer >= this.instance_count_update_interval
    ) {
      this.instance_count_timer -= this.instance_count_update_interval;
      this.cloud_updates_remaining = this.grid_mesh_entities.length;
      this.cloud_update_pass_active = true;
    }

    this.#update_cloud_entity_batch();
  }

  #randomize_cloud_instances(entity, center, instance_count) {
    for (let i = 0; i < instance_count; i++) {
      const transform = EntityManager.get_fragment(entity, TransformFragment, i);

      if (i === 0) {
        transform.position = center;
        transform.scale = [0.24, 0.24, 0.24];
        continue;
      }

      const offset = this.#random_cloud_offset();
      const scale = 0.08 + Math.random() * 0.18;
      transform.position = [
        center[0] + offset[0],
        center[1] + offset[1],
        center[2] + offset[2],
      ];
      transform.scale = [scale, scale, scale];
    }
  }

  #random_cloud_offset() {
    const theta = Math.random() * Math.PI * 2;
    const z = Math.random() * 2 - 1;
    const radial = Math.sqrt(Math.max(0, 1 - z * z));
    const radius = this.cloud_radius * Math.cbrt(Math.random());

    return [
      Math.cos(theta) * radial * radius,
      z * radius,
      Math.sin(theta) * radial * radius,
    ];
  }

  #update_cloud_entity_batch() {
    if (!this.cloud_update_pass_active || this.grid_mesh_entities.length === 0) {
      return;
    }

    const update_count = Math.min(
      this.cloud_entities_to_update_per_frame,
      this.cloud_updates_remaining
    );

    for (let i = 0; i < update_count; i++) {
      const item =
        this.grid_mesh_entities[Math.floor(Math.random() * this.grid_mesh_entities.length)];
      const instance_count = this.#random_cloud_instance_count();
      EntityManager.set_entity_instance_count(item.entity, instance_count);
      this.#randomize_cloud_instances(item.entity, item.center, instance_count);
    }

    this.cloud_updates_remaining -= update_count;
    if (this.cloud_updates_remaining <= 0) {
      this.cloud_update_pass_active = false;
      this.cloud_updates_remaining = 0;
    }
  }

  #random_cloud_instance_count() {
    const max_count = Math.max(1, this.cloud_max_instance_count);
    return 1 + Math.floor(Math.random() * max_count);
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

    SharedEnvironmentData.set_skydome("default_scene_skydome");

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [20.7373, 54.0735, 68.58896];
    view_data.view_rotation = [-0.036352589, 0.94788336, -0.25605953, -0.13457019];

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 3.0;
    light_fragment_view.position = [-15, 20, 5];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Create terrain material
    const terrain_material = StandardMaterial.create("TerrainMaterial");
    this.terrain_material_id = terrain_material.material_id;

    {
      let dirt_albedo = {
        name: "dirt_albedo",
        paths: ["engine/textures/voxel/dirt_albedo.jpg"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "dirt_albedo",
      };

      terrain_material.sample_albedo(dirt_albedo);
      terrain_material.set_roughness(0.8);
      terrain_material.set_emission(0.0);
      terrain_material.set_metallic(0.1);
      terrain_material.set_specular(0.5);
    }

    // Create cube mesh for voxels
    this.cube_mesh = Mesh.cube();

    // Terrain parameters - Perlin-based fractal noise
    // Modified: Much more dramatic height variation for mountainous terrain
    const grid_width = 300;
    const grid_depth = 300;
    const block_size = 1.0;
    const base_frequency = 0.03;
    const height_scale = 25.0;
    const height_offset = 5.0;
    const octaves = 6;
    const persistence = 0.55;

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
      quat.fromEuler(quat.create(), 0, 0, 0),
      [block_size, block_size, block_size],
      this.cube_mesh,
      this.terrain_material_id
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
            yi * block_size * 2.0,
            (zi - grid_depth / 2) * block_size * 2.0,
          ];
          const view = EntityManager.get_fragment(terrain_entity, TransformFragment, block_index);
          view.position = pos;
          block_index++;
        }
      }
    }

    // Create sandy ground plane material
    const sandy_material = StandardMaterial.create("SandyGroundMaterial");
    sandy_material.set_albedo([0.94, 0.87, 0.69, 1.0]); // Sandy beige color
    sandy_material.set_roughness(0.9);
    sandy_material.set_emission(0.0);
    sandy_material.set_metallic(0.1);
    sandy_material.set_specular(0.5);

    // Create large ground plane entity
    const plane_entity = spawn_mesh_entity(
      [0, -10, 0], // Position slightly below terrain base
      quat.fromEuler(quat.create(), 0, 0, 0), // Rotate to be horizontal
      [800, 5, 800], // Large scale to cover the terrain area
      Mesh.cube(),
      sandy_material.material_id
    );
    this.entities.push(plane_entity);

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
    SharedEnvironmentData.set_skybox("default_scene_skybox", [
      "engine/textures/simple_skybox/px.jpg",
      "engine/textures/simple_skybox/nx.jpg",
      "engine/textures/simple_skybox/ny.jpg",
      "engine/textures/simple_skybox/py.jpg",
      "engine/textures/simple_skybox/pz.jpg",
      "engine/textures/simple_skybox/nz.jpg",
    ]);
    SharedEnvironmentData.set_skybox_color([1, 1, 1, 1]);

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
    light_fragment_view.intensity = 10.0;
    light_fragment_view.position = [10, 30, 20];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Load sphere mesh & create transparent/emissive brush material
    this.sphere_mesh = Mesh.sphere();

    const object_material1 = StandardMaterial.create("ObjectPaintingObjectMaterial");
    this.object_material1_id = object_material1.material_id;
    const object_material2 = StandardMaterial.create("ObjectPaintingObjectMaterial2");
    this.object_material2_id = object_material2.material_id;
    const object_material3 = StandardMaterial.create("ObjectPaintingObjectMaterial3");
    this.object_material3_id = object_material3.material_id;

    object_material1.set_albedo([1.0, 1.0, 1.0, 1]);
    object_material1.set_roughness(0.01);
    object_material1.set_metallic(0.99);

    object_material2.set_albedo([0.3, 0.0, 0.0, 1]);
    object_material2.set_roughness(0.9);
    object_material2.set_metallic(0.1);

    object_material3.set_albedo([0.0, 0.3, 0.0, 1]);
    object_material3.set_roughness(0.9);
    object_material3.set_metallic(0.1);
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
            [0, 0, 0, 1],
            [0.3 + Math.random() * 0.3, 0.3 + Math.random() * 0.3, 0.3 + Math.random() * 0.3],
            this.sphere_mesh,
            [this.object_material1_id, this.object_material2_id, this.object_material3_id][
            Math.floor(Math.random() * 3)
            ]
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
      () => { }
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
      () => { }
    );

    // help text overlay
    UI.panel(info_panel_config, () => {
      UI.label("Hold [Space] to Paint objects", info_label_config);
      UI.label("Total entities: " + EntityManager.get_entity_count(), info_label_config);
    });
  }
}

// Simple Test Gym Scene - Cornell-box blockout for DDGI testing
export class GITestScene extends Scene {
  name = "GITestScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    // Mesh.cube() spans [-1, 1], so final dimensions are 2 * scale.
    // Keep Cornell boxes around 3.5m interior size (target: 3-4m).
    const cornell_box_size_m = 3.5;
    const room_size = cornell_box_size_m * 0.5;
    const wall_thickness = 0.05;
    const box_spacing = 12.0;
    const scene_y_offset = -3.5;
    const ambient_emissive = 0.0;

    // camera arcball
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skydome for this scene.
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // camera
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0, 8 + scene_y_offset, 24];
    view_data.view_rotation = [0.0005166, 0.9986818, -0.027326133, 0.0188794];

    // directional light
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 0.5;
    light_fragment_view.position = [5, 5 + scene_y_offset, 5];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Load worn panel textures for metallic floor
    let floor_albedo = {
      name: "floor_albedo",
      paths: ["engine/textures/rubber_floor/Diffuse.jpg"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_albedo",
    };
    let floor_normal = {
      name: "floor_normal",
      paths: ["engine/textures/rubber_floor/Normal.jpg"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_normal",
    };
    let floor_arm_metallic = {
      name: "floor_metallic",
      paths: ["engine/textures/rubber_floor/ARM.jpg"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_metallic",
    };
    let floor_arm_roughness = {
      name: "floor_roughness",
      paths: ["engine/textures/rubber_floor/ARM.jpg"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_roughness",
    };
    let floor_arm_ao = {
      name: "floor_ao",
      paths: ["engine/textures/rubber_floor/ARM.jpg"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_ao",
    };

    // Create metallic floor material
    const metallic_floor_material = StandardMaterial.create("testgym_metallic_floor_material");
    const metallic_floor_material_id = metallic_floor_material.material_id;
    metallic_floor_material.sample_albedo(floor_albedo);
    metallic_floor_material.sample_normal(floor_normal);
    metallic_floor_material.sample_ao(floor_arm_ao, TextureChannel.R);
    metallic_floor_material.sample_roughness(floor_arm_roughness, TextureChannel.G);
    metallic_floor_material.sample_metallic(floor_arm_metallic, TextureChannel.B);
    metallic_floor_material.set_tiling(250.0);

    // materials
    const wall_material = StandardMaterial.create("testgym_wall_material");
    const wall_material_id = wall_material.material_id;
    wall_material.set_albedo([1, 1, 1, 1]);
    wall_material.set_roughness(0.8);
    wall_material.set_metallic(0.01);

    const red_material = StandardMaterial.create("testgym_red_material");
    const red_material_id = red_material.material_id;
    red_material.set_albedo([1, 0.2, 0.2, 1]);
    red_material.set_metallic(0.01);
    red_material.set_roughness(0.8);

    const blue_material = StandardMaterial.create("testgym_blue_material");
    const blue_material_id = blue_material.material_id;
    blue_material.set_albedo([0.2, 0.2, 1, 1]);
    blue_material.set_metallic(0.01);
    blue_material.set_roughness(0.8);

    const gray_material = StandardMaterial.create("testgym_gray_material");
    const gray_material_id = gray_material.material_id;
    gray_material.set_albedo([0.5, 0.5, 0.5, 1]);
    gray_material.set_metallic(0.01);
    gray_material.set_roughness(0.8);

    // White emissive material for ceiling lights
    const emissive_white_material = StandardMaterial.create("testgym_emissive_white_material");
    const emissive_white_material_id = emissive_white_material.material_id;
    emissive_white_material.set_albedo([0.9, 0.9, 0.5, 1]);
    emissive_white_material.set_emission(100.0);
    emissive_white_material.set_metallic(0.01);
    emissive_white_material.set_roughness(0.9);

    // meshes
    const cube_mesh = Mesh.cube();
    const sphere_mesh = Mesh.sphere();

    // Create large metallic floor plane
    const floor_plane = spawn_mesh_entity(
      [0, -5, 0],
      [0, 0, 0, 1],
      [200, 1, 200],
      cube_mesh,
      metallic_floor_material_id
    );
    this.entities.push(floor_plane);

    // Cornell-box walls (tight box)
    {
      // floor top at y=0
      const floor = spawn_mesh_entity(
        [0, scene_y_offset - wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(floor);

      // ceiling bottom at y=room_size
      const ceiling = spawn_mesh_entity(
        [0, scene_y_offset + room_size * 2.0 + wall_thickness, 0.0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling);

      // White emissive rectangle on ceiling (light)
      const emissive_light = spawn_mesh_entity(
        [0, scene_y_offset + room_size * 2.0 - 0.1, 0.0],
        [0, 0, 0, 1],
        [0.5, 0.1, 0.5],
        cube_mesh,
        emissive_white_material_id
      );
      this.entities.push(emissive_light);

      // back wall inner surface at z=-room_size/2
      const back_wall = spawn_mesh_entity(
        [0, scene_y_offset + room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall);

      // left wall inner surface at x=-room_size/2
      const left_wall = spawn_mesh_entity(
        [-room_size - wall_thickness, scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        red_material_id
      );
      this.entities.push(left_wall);

      // right wall inner surface at x=+room_size/2
      const right_wall = spawn_mesh_entity(
        [room_size + wall_thickness, scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        blue_material_id
      );
      this.entities.push(right_wall);

      // blockout "buildings"
      const building_data = [
        { mesh: cube_mesh, position: [-0.9, scene_y_offset + 0.65, -0.9], scale: [0.45, 0.65, 0.45], material_id: gray_material_id },
        { mesh: cube_mesh, position: [0.85, scene_y_offset + 0.95, -0.45], scale: [0.35, 0.95, 0.35], material_id: red_material_id },
        {
          mesh: sphere_mesh,
          position: [-0.2, scene_y_offset + 0.35, 0.85],
          scale: [0.35, 0.35, 0.35],
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
    // Additional Cornell box with different wall colors at x = -box_spacing
    {
      const offset_x = -box_spacing;

      const left_wall_material_second = StandardMaterial.create("testgym_left_material_second");
      const left_wall_material_second_id = left_wall_material_second.material_id;
      left_wall_material_second.set_albedo([0, 1, 0, 1]);
      left_wall_material_second.set_emission(ambient_emissive);
      left_wall_material_second.set_metallic(0.9);

      const right_wall_material_second = StandardMaterial.create("testgym_right_material_second");
      const right_wall_material_second_id = right_wall_material_second.material_id;
      right_wall_material_second.set_albedo([1, 0, 1, 1]);
      right_wall_material_second.set_emission(ambient_emissive);
      right_wall_material_second.set_metallic(0.9);

      // spawn elements for second box
      const floor_second = spawn_mesh_entity(
        [offset_x, scene_y_offset - wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(floor_second);

      const ceiling_second = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size * 2.0 + wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling_second);

      // White emissive rectangle on ceiling (light)
      const emissive_light_second = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size * 2.0 - 0.1, 0.0],
        [0, 0, 0, 1],
        [0.5, 0.1, 0.5],
        cube_mesh,
        emissive_white_material_id
      );
      this.entities.push(emissive_light_second);

      const back_wall_second = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall_second);

      const left_wall_second = spawn_mesh_entity(
        [offset_x - (room_size + wall_thickness), scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        left_wall_material_second_id
      );
      this.entities.push(left_wall_second);

      const right_wall_second = spawn_mesh_entity(
        [offset_x + room_size + wall_thickness, scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        right_wall_material_second_id
      );
      this.entities.push(right_wall_second);

      const building_data_second = [
        {
          mesh: cube_mesh,
          position: [offset_x - 0.7, scene_y_offset + 0.55, -0.7],
          scale: [0.4, 0.55, 0.4],
          material_id: left_wall_material_second_id,
        },
        {
          mesh: sphere_mesh,
          position: [offset_x + 0.55, scene_y_offset + 0.3, 0.75],
          scale: [0.3, 0.3, 0.3],
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
    // Additional Cornell box with different wall colors at x = +box_spacing
    {
      const offset_x = box_spacing;

      const left_wall_material_third = StandardMaterial.create("testgym_left_material_third");
      const left_wall_material_third_id = left_wall_material_third.material_id;
      left_wall_material_third.set_albedo([1, 0.5, 0, 1]);
      left_wall_material_third.set_emission(ambient_emissive);
      left_wall_material_third.set_metallic(0.9);

      const right_wall_material_third = StandardMaterial.create("testgym_right_material_third");
      const right_wall_material_third_id = right_wall_material_third.material_id;
      right_wall_material_third.set_albedo([0.5, 0, 0.5, 0.3]);
      right_wall_material_third.set_emission(ambient_emissive);
      right_wall_material_third.set_metallic(0.9);

      const floor_third = spawn_mesh_entity(
        [offset_x, scene_y_offset - wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(floor_third);

      const ceiling_third = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size * 2.0 + wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling_third);

      // White emissive rectangle on ceiling (light)
      const emissive_light_third = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size * 2.0 - 0.1, 0.0],
        [0, 0, 0, 1],
        [0.5, 0.1, 0.5],
        cube_mesh,
        emissive_white_material_id
      );
      this.entities.push(emissive_light_third);

      const back_wall_third = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall_third);

      const left_wall_third = spawn_mesh_entity(
        [offset_x - (room_size + wall_thickness), scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        left_wall_material_third_id
      );
      this.entities.push(left_wall_third);

      const right_wall_third = spawn_mesh_entity(
        [offset_x + room_size + wall_thickness, scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        right_wall_material_third_id
      );
      this.entities.push(right_wall_third);

      const building_data_third = [
        {
          mesh: sphere_mesh,
          position: [offset_x - 0.6, scene_y_offset + 0.4, -0.6],
          scale: [0.4, 0.4, 0.4],
          material_id: left_wall_material_third_id,
        },
        {
          mesh: cube_mesh,
          position: [offset_x + 0.7, scene_y_offset + 0.45, 0.7],
          scale: [0.45, 0.45, 0.45],
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

    // Load and place the station behind the three Cornell boxes.
    let station_root = this.load_gltf_scene(
      "engine/models/station/station.gltf",
      [0, scene_y_offset, -45],
      [0, 0, 0, 1],
      [1.5, 1.5, 1.5]
    );
    this.entities.push(station_root);
  }

  cleanup() {
    for (const e of this.entities) {
      delete_entity(e);
    }
    this.remove_layer(FreeformArcballControlProcessor);
    super.cleanup();
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
// =============================== Scene Settings ====================================
// ------------------------------------------------------------------------------------

const scene_settings_panel_config = {
  layout: "column",
  gap: 10,
  x: 20,
  y: 20,
  anchor_x: "right",
  width: 392,
  padding: 12,
  background_color: "rgba(7, 12, 18, 0.96)",
  border: "1px solid rgba(105, 205, 255, 0.22)",
  corner_radius: 12,
  box_shadow: "0 16px 40px rgba(0,0,0,0.52)",
};

const scene_settings_header_config = {
  width: "100%",
  height: 44,
  x: 0,
  y: 0,
  background_color: "rgba(23, 34, 46, 0.9)",
  hover_color: "rgba(35, 53, 69, 0.96)",
  border: "1px solid rgba(105, 205, 255, 0.16)",
  corner_radius: 8,
  font: "600 13px monospace",
  text_color: "#f1f8fc",
  text_align: "left",
  text_padding: 14,
};

const scene_settings_section_header_config = {
  ...scene_settings_header_config,
  height: 36,
  background_color: "rgba(93, 196, 255, 0.08)",
  hover_color: "rgba(93, 196, 255, 0.14)",
  border: "1px solid rgba(93, 196, 255, 0.14)",
  corner_radius: 7,
  font: "600 12px monospace",
  text_color: "#8fd9ff",
  text_padding: 12,
};

const scene_settings_section_config = {
  layout: "column",
  x: 0,
  y: 0,
  gap: 10,
  width: "100%",
  height: 180,
  padding: 10,
  background_color: "rgba(255, 255, 255, 0.022)",
  border: "1px solid rgba(255, 255, 255, 0.065)",
  corner_radius: 8,
};

const scene_settings_group_label_config = {
  width: "100%",
  height: 18,
  x: 0,
  y: 0,
  font: "600 10px monospace",
  text_color: "#7f94a6",
  text_align: "left",
  text_valign: "middle",
  text_padding: 2,
};

const scene_settings_slider_config = {
  width: 88,
  height: 30,
  x: 0,
  y: 0,
  mode: "numeric",
  background_color: "rgba(255, 255, 255, 0.045)",
  hover_color: "rgba(92, 190, 255, 0.14)",
  active_color: "rgba(92, 190, 255, 0.23)",
  text_color: "#eef8ff",
  font: "11px monospace",
  corner_radius: 5,
};

const scene_settings_vector_field_config = {
  group_config: {
    height: 52,
  },
  label_config: scene_settings_group_label_config,
  row_config: {
    gap: 6,
    height: 30,
  },
  component_config: {
    width: 112,
    height: 30,
    background_color: "rgba(255, 255, 255, 0.035)",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    corner_radius: 6,
  },
  component_label_config: {
    width: 24,
    height: 30,
    font: "700 11px monospace",
  },
  component_label_styles: [
    { text_color: "#ff7c8f", background_color: "rgba(255, 93, 117, 0.09)" },
    { text_color: "#74dfa1", background_color: "rgba(87, 214, 139, 0.09)" },
    { text_color: "#7daeff", background_color: "rgba(99, 155, 255, 0.1)" },
  ],
  slider_config: scene_settings_slider_config,
};

function scene_settings_numeric_slider(id, value, range, config, on_change) {
  const result = UI.slider(value, {
    ...config,
    ...range,
    id: `scene_settings.${id}`,
    scrub_speed: range.step,
  });
  if (result.active && result.changed) {
    on_change(result.value);
  }
}

function scene_settings_intensity_field(value, on_change) {
  UI.panel(
    {
      layout: "row",
      gap: 8,
      width: "100%",
      height: 32,
      x: 0,
      y: 0,
    },
    () => {
      UI.label("INTENSITY", {
        ...scene_settings_group_label_config,
        width: 84,
        height: 32,
      });
      scene_settings_numeric_slider(
        "intensity",
        value,
        { min: 0, max: 100, step: 0.1, precision: 1 },
        {
          ...scene_settings_slider_config,
          width: 256,
          height: 32,
          background_color: "rgba(93, 196, 255, 0.075)",
          border: "1px solid rgba(93, 196, 255, 0.12)",
        },
        on_change
      );
    }
  );
}

export class SceneSettingsPanel extends SimulationLayer {
  scene_switcher = null;
  is_open = true;
  directional_light_section_open = true;

  constructor(scene_switcher) {
    super();
    this.name = "SceneSettingsPanel";
    this.scene_switcher = scene_switcher;
  }

  update(delta_time) {
    super.update(delta_time);

    const active_scene = this.scene_switcher.scenes[this.scene_switcher.current_scene_index];
    active_scene?.show_dev_cursor?.();

    const directional_light = this.find_directional_light(active_scene);
    const panel_height = !this.is_open
      ? 68
      : this.directional_light_section_open
        ? 304
        : 114;

    UI.panel(
      {
        ...scene_settings_panel_config,
        height: panel_height,
      },
      () => {
        const panel_header = UI.button(
          `${this.is_open ? "v" : ">"}  SCENE SETTINGS`,
          scene_settings_header_config
        );
        if (panel_header.clicked) {
          this.is_open = !this.is_open;
        }

        if (!this.is_open) {
          return;
        }

        const section_header = UI.button(
          `${this.directional_light_section_open ? "v" : ">"}  DIRECTIONAL LIGHT`,
          scene_settings_section_header_config
        );
        if (section_header.clicked) {
          this.directional_light_section_open = !this.directional_light_section_open;
        }

        if (!this.directional_light_section_open) {
          return;
        }

        UI.panel(scene_settings_section_config, () => {
          if (!directional_light) {
            UI.label("No directional light in this scene", {
              ...scene_settings_group_label_config,
              width: "100%",
              height: 32,
            });
            return;
          }

          this.render_directional_light_fields(directional_light);
        });
      }
    );
  }

  find_directional_light(scene) {
    const entity_lists = [scene?.entities, scene?.scene_entities];
    let fallback = null;

    for (const entities of entity_lists) {
      if (!Array.isArray(entities)) continue;

      for (const entity of entities) {
        if (
          !EntityManager.entity_exists(entity) ||
          !EntityManager.has_fragment(entity, LightFragment)
        ) {
          continue;
        }

        const light = EntityManager.get_fragment(entity, LightFragment);
        if (light.type !== LightType.DIRECTIONAL || !light.active) continue;
        if (light.is_primary_sun) return light;
        fallback ??= light;
      }
    }

    return fallback;
  }

  render_directional_light_fields(light) {
    const position = light.position;
    const direction_length = Math.hypot(position[0], position[1], position[2]);
    let direction = [position[0], position[1], position[2]];

    // Existing scenes use an arbitrary-length position vector for directional lights.
    // Bring legacy values into the editor's -1..1 component range once, then preserve
    // the edited components exactly. Lighting and shadow code normalize this vector.
    if (direction_length < 0.0001) {
      direction = [0, -1, 0];
      light.position = [...direction, position[3]];
      light.shadows_dirty = 1;
    } else if (direction.some((component) => Math.abs(component) > 1)) {
      direction = direction.map((component) => component / direction_length);
      light.position = [...direction, position[3]];
      light.shadows_dirty = 1;
    }

    const direction_range = { min: -1, max: 1, step: 0.01, precision: 2 };
    const direction_result = UI.vector_field(direction, {
      ...scene_settings_vector_field_config,
      id: "scene_settings.direction",
      label: "DIRECTION",
      components: ["X", "Y", "Z"],
      slider_config: {
        ...scene_settings_vector_field_config.slider_config,
        ...direction_range,
      },
    });
    if (
      direction_result.changed &&
      Math.hypot(
        direction_result.value[0],
        direction_result.value[1],
        direction_result.value[2]
      ) >= 0.0001
    ) {
      light.position = [...direction_result.value, position[3]];
      light.shadows_dirty = 1;
    }

    const color = light.color;
    const color_range = { min: 0, max: 1, step: 0.01, precision: 2 };
    const color_result = UI.vector_field(color, {
      ...scene_settings_vector_field_config,
      id: "scene_settings.color",
      label: "COLOR",
      components: ["R", "G", "B"],
      slider_config: {
        ...scene_settings_vector_field_config.slider_config,
        ...color_range,
      },
    });
    if (color_result.changed) {
      light.color = color_result.value;
    }

    scene_settings_intensity_field(light.intensity, (value) => {
      light.intensity = value;
      light.shadows_dirty = 1;
    });
  }
}

// ------------------------------------------------------------------------------------
// =============================== Shadow Test Scene ==================================
// ------------------------------------------------------------------------------------

export class ShadowTestScene extends Scene {
  name = "ShadowTestScene";
  entities = [];
  swaying_balls = [];
  swaying_ball_base_positions = [];

  init(parent_context) {
    super.init(parent_context);

    // Add arcball camera control
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.move_speed = 75.0;
    freeform_arcball_control_processor.set_scene(this);

    // Configure skybox
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // Position the camera high above the city
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [205.515, 40.4267, 273.94];
    view_data.view_rotation = [0.04745, 0.57238, 0.03344, -0.81212];
    view_data.far = 100000.0;

    // Create a sun-like directional light
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 0.9];
    light_fragment_view.intensity = 0.2;
    light_fragment_view.position = [30, 55, 40];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // // Create a point light
    const point_light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(point_light_entity);

    const point_light_fragment_view = EntityManager.get_fragment(point_light_entity, LightFragment);
    point_light_fragment_view.type = LightType.POINT;
    point_light_fragment_view.color = [1, 1, 0];
    point_light_fragment_view.intensity = 5.0;
    point_light_fragment_view.position = [0, 20, 0];
    point_light_fragment_view.radius = 20.0;
    point_light_fragment_view.active = true;
    point_light_fragment_view.shadow_casting = false;

    const num_point_lights = 12;
    EntityManager.set_entity_instance_count(point_light_entity, num_point_lights);

    // Ground material
    const ground_material = StandardMaterial.create("shadow_ground_material");
    const ground_material_id = ground_material.material_id;
    ground_material.set_albedo([0.5, 0.5, 0.5, 1]);
    ground_material.set_roughness(0.9);
    ground_material.set_metallic(0.0);

    // Building material
    const building_material = StandardMaterial.create("shadow_building_material");
    const building_material_id = building_material.material_id;
    building_material.set_albedo([0.35, 0.35, 0.35, 1]);
    building_material.set_roughness(0.9);
    building_material.set_metallic(0.0);

    // Shared cube mesh
    const cube_mesh = Mesh.cube();
    const sphere_mesh = Mesh.sphere();

    const grid_size = 80; // 80 × 80 buildings
    const building_spacing = 35.0; // distance between building centres (was 20.0)
    const building_base_size = 6.0; // footprint of each building
    const half_grid = (grid_size - 1) * building_spacing * 0.5;

    // Create an expansive ground plane
    const ground_plane_size = 3000.0;
    this.ground_entity = spawn_mesh_entity(
      [0.0, 0.0, 0.0],
      [0, 0, 0, 1],
      [ground_plane_size, 1.0, ground_plane_size],
      cube_mesh,
      ground_material_id
    );
    this.entities.push(this.ground_entity);

    // Compute corner positions using half_grid
    const offset = half_grid * 0.9; // Slightly inset from absolute corners to avoid clipping buildings

    this.swaying_ball_base_positions = [
      [offset, 350, offset],
      [offset, 350, -offset],
      [-offset, 350, offset],
      [-offset, 350, -offset],
    ];

    // Create ball material
    const ball_material = StandardMaterial.create("swaying_ball_material");
    ball_material.set_albedo([0.9, 0.9, 0.2, 1]);
    ball_material.set_emission(0.3);
    ball_material.set_roughness(0.2);
    ball_material.set_metallic(0.9);

    // Create spheres
    for (const base_pos of this.swaying_ball_base_positions) {
      const ball_position = [...base_pos];
      const ball_scale = [40, 40, 40];
      const entity = spawn_mesh_entity(
        ball_position,
        [0, 0, 0, 1],
        ball_scale,
        sphere_mesh,
        ball_material.material_id
      );
      this.swaying_balls.push(entity);
      this.entities.push(entity);
    }

    const building_entity = spawn_mesh_entity(
      [0.0, 0.0, 0.0, 1.0],
      quat.fromEuler(quat.create(), 0.0, 0.0, 0.0),
      [0.0, 0.0, 0.0],
      cube_mesh,
      building_material_id
    );
    EntityManager.set_entity_instance_count(building_entity, grid_size * grid_size);
    this.entities.push(building_entity);

    let instance_index = 0;
    for (let gx = 0; gx < grid_size; gx++) {
      for (let gz = 0; gz < grid_size; gz++) {
        // More randomized height: most buildings short, few very tall
        const base = 8.0;
        const max = 180.0;
        const exponent = 4.0; // Higher = more short buildings
        const height = base + Math.pow(Math.random(), exponent) * (max - base);

        const position = [
          gx * building_spacing - half_grid,
          height * 0.5,
          gz * building_spacing - half_grid,
        ];
        const scale = [building_base_size, height, building_base_size];

        const transform_fragment = EntityManager.get_fragment(
          building_entity,
          TransformFragment,
          instance_index
        );
        transform_fragment.position = position;
        transform_fragment.scale = scale;

        const visibility_fragment = EntityManager.get_fragment(
          building_entity,
          VisibilityFragment,
          instance_index
        );
        visibility_fragment.occluder = 0;

        instance_index++;
      }
    }

    log(`[${this.name}] Spawned ${this.entities.length} entities.`);

    // --- Neon Signs ---------------------------------------------------------
    // Create several vibrant emissive materials for different neon colours.
    const neon_colours = [
      { name: "neon_cyan", color: [0.0, 1.0, 1.0, 1.0] },
      { name: "neon_magenta", color: [1.0, 0.0, 1.0, 1.0] },
      { name: "neon_yellow", color: [1.0, 1.0, 0.0, 1.0] },
      { name: "neon_orange", color: [1.0, 0.5, 0.0, 1.0] },
      { name: "neon_green", color: [0.0, 1.0, 0.0, 1.0] },
    ];

    const neon_material_ids = neon_colours.map((c) => {
      const m = StandardMaterial.create(`shadow_${c.name}`);
      m.set_albedo(c.color);
      m.set_emission(100.0); // extremely bright
      m.set_roughness(0.1);
      m.set_metallic(0.9);
      return m.material_id;
    });

    const neon_sign_transforms = [];
    const neon_sign_material_indices = [];

    // Decide which buildings receive signs and where they go.
    for (let i = 0; i < instance_index; i++) {
      // Roughly 30 % of buildings receive a sign.
      if (Math.random() < 0.3) {
        const building_tf = EntityManager.get_fragment(building_entity, TransformFragment, i);
        if (!building_tf) continue;

        const bp = building_tf.position;
        const bs = building_tf.scale;

        // Choose a random face: 0 = +z front, 1 = -z back, 2 = -x left, 3 = +x right
        const face = Math.floor(Math.random() * 4);

        // Sign parameters
        const sign_depth = 0.25;
        const sign_height = 2.0;

        let sign_pos;
        let sign_scale;

        const vertical_offset = 0.2 + Math.random() * 0.6; // between 20 % and 80 % up the wall

        switch (face) {
          case 0: // +z (front)
            sign_scale = [bs[0] * 0.8, sign_height, sign_depth];
            sign_pos = [
              bp[0],
              bp[1] + bs[1] * vertical_offset,
              bp[2] + bs[2] + sign_depth * 0.5 + 0.05,
            ];
            break;
          case 1: // -z (back)
            sign_scale = [bs[0] * 0.8, sign_height, sign_depth];
            sign_pos = [
              bp[0],
              bp[1] + bs[1] * vertical_offset,
              bp[2] - bs[2] - sign_depth * 0.5 - 0.05,
            ];
            break;
          case 2: // -x (left)
            sign_scale = [sign_depth, sign_height, bs[2] * 0.8];
            sign_pos = [
              bp[0] - bs[0] - sign_depth * 0.5 - 0.05,
              bp[1] + bs[1] * vertical_offset,
              bp[2],
            ];
            break;
          case 3: // +x (right)
          default:
            sign_scale = [sign_depth, sign_height, bs[2] * 0.8];
            sign_pos = [
              bp[0] + bs[0] + sign_depth * 0.5 + 0.05,
              bp[1] + bs[1] * vertical_offset,
              bp[2],
            ];
            break;
        }

        neon_sign_transforms.push({ position: sign_pos, scale: sign_scale });
        neon_sign_material_indices.push(Math.floor(Math.random() * neon_material_ids.length));
      }
    }

    // Spawn instanced entities grouped by material for efficiency.
    if (neon_sign_transforms.length > 0) {
      // Group transforms by selected material.
      const grouped = new Map();
      neon_sign_transforms.forEach((t, idx) => {
        const mat_idx = neon_sign_material_indices[idx];
        if (!grouped.has(mat_idx)) grouped.set(mat_idx, []);
        grouped.get(mat_idx).push(t);
      });

      // Create one entity per material.
      grouped.forEach((transforms, mat_idx) => {
        const neon_entity = spawn_mesh_entity(
          [0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 0],
          cube_mesh,
          neon_material_ids[mat_idx]
        );

        EntityManager.set_entity_instance_count(neon_entity, transforms.length);

        transforms.forEach((tr, i) => {
          const tf = EntityManager.get_fragment(neon_entity, TransformFragment, i);
          tf.position = tr.position;
          tf.scale = tr.scale;
        });

        this.entities.push(neon_entity);
      });
    }

    // ------------------------------------------------------------------
    // Scatter 100 point-lights around the generated city.
    // Each light is attached to a random building, either on a façade or
    // on the rooftop, with some colour variation for extra visual flavour.
    // ------------------------------------------------------------------
    const point_light_colours = [
      [1.0, 0.8, 0.6], // warm
      [0.6, 0.8, 1.0], // cool
      [1.0, 1.0, 0.8], // neutral
    ];

    // Create a uniform grid of point lights with better distribution
    const light_grid_size = Math.ceil(Math.sqrt(num_point_lights));
    const grid_spacing = 300.0;
    const grid_offset = (light_grid_size - 1) * grid_spacing * 0.5;

    for (let i = 0; i < num_point_lights; i++) {
      const p_view = EntityManager.get_fragment(point_light_entity, LightFragment, i);

      // Configure each light instance
      p_view.type = LightType.POINT;
      p_view.active = true;
      p_view.shadow_casting = false;

      // Calculate grid position for uniform distribution
      const grid_x = i % light_grid_size;
      const grid_z = Math.floor(i / light_grid_size);

      // Position lights in a centered grid pattern
      const x = grid_x * grid_spacing - grid_offset;
      const z = grid_z * grid_spacing - grid_offset;
      const y = 60.0 + Math.sin(i * 0.5) * 20.0; // Vary height slightly for visual interest

      p_view.position = [x, y, z];
      p_view.radius = 100.0;
      p_view.intensity = 15.0 + Math.sin(i * 0.3) * 1.0; // Vary intensity slightly
      p_view.color = point_light_colours[i % point_light_colours.length];
    }
    // ------------------------------------------------------------------------
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

    // Animate the swaying balls
    for (let i = 0; i < this.swaying_balls.length; i++) {
      const entity = this.swaying_balls[i];
      const base_pos = this.swaying_ball_base_positions[i];

      const sway_amplitude = 1200;
      const sway_frequency = 0.15;
      const t = performance.now() * 0.001;
      const phase = i * Math.PI * 0.5; // Offset phase for variety

      const x = base_pos[0] + Math.sin(t * Math.PI * 2 * sway_frequency + phase) * sway_amplitude;
      const y = base_pos[1];
      const z = base_pos[2];

      const tf = EntityManager.get_fragment(entity, TransformFragment);
      if (tf) {
        tf.position = [x, y, z];
      }
    }
  }
}

// ------------------------------------------------------------------------------------
// =============================== GLTF Model Scene ==================================
// ------------------------------------------------------------------------------------

export class GLTFModelScene extends Scene {
  name = "GLTFModelScene";
  entities = [];
  barrel_entities = [];
  barrel_offsets = [];

  init(parent_context) {
    super.init(parent_context);

    // Add arcball camera control
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set skybox
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // Camera setup
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [21, 2, 30];
    view_data.view_rotation = quat.fromEuler(quat.create(), 0, 180, 0);

    // Add a directional light
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 5.0;
    light_fragment_view.position = [10, 20, 10];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Load a GLTF model (e.g., barrel)
    const model_mesh = Mesh.from_gltf("engine/models/barrel/Barrel.gltf");

    // Spawn three barrels at positions offset by 15 units on x-axis
    const barrel_positions = [
      [5, -10, 0],
      [20, -10, 0],
      [35, -10, 0],
    ];
    this.barrel_entities = [];
    this.barrel_offsets = [];
    for (let i = 0; i < barrel_positions.length; i++) {
      const pos = barrel_positions[i];
      const entity = spawn_mesh_entity(
        pos,
        [0, 0, 0, 1],
        [10.0, 10.0, 10.0],
        model_mesh,
        0 // GLTF sets the material id
      );
      this.barrel_entities.push(entity);
      this.entities.push(entity);
      this.barrel_offsets.push(Math.random() * Math.PI * 2);
    }
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

    world_label("GLTF Model Scene", [0, 10, 0], {
      width: 40.0,
      height: 10.0,
    });

    const t = performance.now() * 0.01;
    for (let i = 0; i < this.barrel_entities.length; i++) {
      const entity = this.barrel_entities[i];
      const tf = EntityManager.get_fragment(entity, TransformFragment);
      if (tf) {
        tf.rotation = quat.fromEuler(quat.create(), 0, t + this.barrel_offsets[i], 0);
      }
    }
  }
}

// ------------------------------------------------------------------------------------
// =============================== Sponza Scene ======================================
// ------------------------------------------------------------------------------------

export class SponzaScene extends Scene {
  name = "SponzaScene";
  entities = [];
  sway_light_enabled = false;
  sway_period_sec = 35.0;
  sway_angle_deg = 120.0;
  sun_light_entity = null;
  sun_light_base_dir = [0, 0, 0];
  sun_light_intensity_on = 30.0;
  time_elapsed_sec = 0;

  init(parent_context) {
    super.init(parent_context);

    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [8.360948, 7.528844, -1.364703];
    view_data.view_rotation = [-0.075029, 0.645789, -0.0024352, -0.68824034];

    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [0.9, 0.9, 1.0];
    light_fragment_view.intensity = this.sun_light_intensity_on;
    light_fragment_view.position = [5.0, 20, 2.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    this.sun_light_entity = light_entity;
    this.sun_light_base_dir = [
      light_fragment_view.position[0],
      light_fragment_view.position[1],
      light_fragment_view.position[2],
    ];

    const ground_material = StandardMaterial.create("sponza_ground_material");
    const ground_material_id = ground_material.material_id;
    ground_material.set_albedo([0.75, 0.75, 0.75, 1.0]);
    ground_material.set_roughness(0.9);
    ground_material.set_metallic(0.8);

    const cube_mesh = Mesh.cube();
    const ground_entity = spawn_mesh_entity(
      [0, 0, 0],
      quat.fromEuler(quat.create(), 0, 0, 0),
      [500, 1.0, 500],
      cube_mesh,
      ground_material_id
    );
    this.entities.push(ground_entity);

    const root_entity = this.load_gltf_scene(
      "engine/models/sponza/Sponza.gltf",
      [0, 2.5, 0],
      [0, 0, 0, 1],
      [2.0, 2.0, 2.0]
    );
    this.entities.push(root_entity);
  }

  update(delta_time) {
    super.update(delta_time);
    this.time_elapsed_sec += delta_time;

    // Toggle light sway with 'T'
    if (InputProvider.get_action(InputKey.K_t)) {
      this.sway_light_enabled = !this.sway_light_enabled;
    }

    // Toggle directional light on/off with 'L' (intensity 0 = off)
    if (InputProvider.get_action(InputKey.K_l) && this.sun_light_entity) {
      const light_fragment_view = EntityManager.get_fragment(this.sun_light_entity, LightFragment);
      if (light_fragment_view) {
        const is_off = light_fragment_view.intensity <= 0.0;
        light_fragment_view.intensity = is_off ? this.sun_light_intensity_on : 0.0;
        light_fragment_view.shadows_dirty = 1;
      }
    }

    if (this.sway_light_enabled && this.sun_light_entity) {
      const base_x = this.sun_light_base_dir[0];
      const base_y = this.sun_light_base_dir[1];
      const base_z = this.sun_light_base_dir[2];

      const two_pi = Math.PI * 2.0;
      const phase = (this.time_elapsed_sec / this.sway_period_sec) * two_pi;
      const angle_rad = Math.sin(phase) * ((this.sway_angle_deg * Math.PI) / 180.0);

      const cos_a = Math.cos(angle_rad);
      const sin_a = Math.sin(angle_rad);
      const rot_x = base_x * cos_a + base_z * sin_a;
      const rot_z = -base_x * sin_a + base_z * cos_a;

      const light_fragment_view = EntityManager.get_fragment(this.sun_light_entity, LightFragment);
      if (light_fragment_view) {
        light_fragment_view.position = [rot_x, base_y, rot_z];
        light_fragment_view.shadows_dirty = 1;
      }
    }

  }

  cleanup() {
    for (const e of this.entities) {
      delete_entity(e);
    }
    this.entities.length = 0;
  }
}

// ------------------------------------------------------------------------------------
// =============================== Living Room Scene ==============================
// ------------------------------------------------------------------------------------

export class LivingRoomScene extends Scene {
  name = "LivingRoomScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    // ─────────────────────────────────────────────────────────────────────────
    // Camera Controls
    // ─────────────────────────────────────────────────────────────────────────
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skydome for ambient lighting
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // ─────────────────────────────────────────────────────────────────────────
    // Camera Setup - positioned to view the living room interior
    // ─────────────────────────────────────────────────────────────────────────
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [10.61615, 18.35985, 60.53034];
    view_data.view_rotation = [-0.023007927, 0.97692966, -0.15258932, -0.1472981];

    // ─────────────────────────────────────────────────────────────────────────
    // Primary Directional Light (simulating window light)
    // ─────────────────────────────────────────────────────────────────────────
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 1.0, 1.0];  // Warm daylight tint
    light_fragment_view.intensity = 0.0;
    light_fragment_view.position = [0.0, 5.0, -10.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    const ground_material = StandardMaterial.create("sponza_ground_material");
    const ground_material_id = ground_material.material_id;
    ground_material.set_albedo([0.75, 0.75, 0.75, 1.0]);
    ground_material.set_roughness(0.9);
    ground_material.set_metallic(0.8);

    const cube_mesh = Mesh.cube();
    const ground_entity = spawn_mesh_entity(
      [0, 0, 0],
      quat.fromEuler(quat.create(), 0, 0, 0),
      [2000, 1.0, 2000],
      cube_mesh,
      ground_material_id
    );
    this.entities.push(ground_entity);

    // ─────────────────────────────────────────────────────────────────────────
    // Load Living Room GLTF Model - Instanced 100x in a 10x10 grid
    // ─────────────────────────────────────────────────────────────────────────
    const living_room_entity = this.load_gltf_scene(
      "engine/models/living_room/living_room.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [1, 1, 1],
    );
    this.entities.push(living_room_entity);

    const grid_size = 1;
    const num_instances = grid_size * grid_size;
    const spacing = 120.0; // Distance between living room instances
    const half_grid = (grid_size - 1) * spacing * 0.5;

    // Set instance count for the living room entity
    EntityManager.set_entity_instance_count(living_room_entity, num_instances);

    // Position each instance in a uniform grid along the floor
    let instance_index = 0;
    for (let gx = 0; gx < grid_size; gx++) {
      for (let gz = 0; gz < grid_size; gz++) {
        const x = gx * spacing - half_grid;
        const y = 1.0; // Slightly above the ground
        const z = gz * spacing - half_grid;

        const transform_view = EntityManager.get_fragment(
          living_room_entity,
          TransformFragment,
          instance_index
        );
        transform_view.position = [x, y, z];
        transform_view.scale = [2, 2, 2];

        instance_index++;
      }
    }

    log(`[${this.name}] Living room scene initialized with ${num_instances} instances.`);
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }
}

// ------------------------------------------------------------------------------------
// =============================== City Scene =========================================
// ------------------------------------------------------------------------------------

export class CityScene extends Scene {
  name = "CityScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    // ─────────────────────────────────────────────────────────────────────────
    // Camera Controls
    // ─────────────────────────────────────────────────────────────────────────
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.move_speed = 50.0;
    freeform_arcball_control_processor.set_scene(this);

    // Set the skydome for ambient lighting
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // ─────────────────────────────────────────────────────────────────────────
    // Camera Setup - positioned to overlook the city
    // ─────────────────────────────────────────────────────────────────────────
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [108.015, 9.81021, 33.1744];
    view_data.view_rotation = [-0.01606280, 0.78540349, -0.023472044, -0.613837182];
    view_data.far = 10000.0;

    // ─────────────────────────────────────────────────────────────────────────
    // Primary Directional Light (sun)
    // ─────────────────────────────────────────────────────────────────────────
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.95, 0.85];  // Warm sunlight tint
    light_fragment_view.intensity = 5.0;
    light_fragment_view.position = [30.0, 50.0, 20.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;
    light_fragment_view.shadow_clipmaps = 12;

    // ─────────────────────────────────────────────────────────────────────────
    // Load City GLTF Model
    // ─────────────────────────────────────────────────────────────────────────
    let city_root = this.load_gltf_scene("engine/models/city/City.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [2, 2, 2],
    );
    this.entities.push(city_root);

    log(`[${this.name}] City scene initialized.`);
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }
}

// ------------------------------------------------------------------------------------
// =============================== SciFi City Scene =========================================
// ------------------------------------------------------------------------------------

export class SciFiCityScene extends Scene {
  name = "SciFiCityScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    // ─────────────────────────────────────────────────────────────────────────
    // Camera Controls
    // ─────────────────────────────────────────────────────────────────────────
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.move_speed = 50.0;
    freeform_arcball_control_processor.set_scene(this);

    // Set the skydome for ambient lighting
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // ─────────────────────────────────────────────────────────────────────────
    // Camera Setup - positioned to overlook the city
    // ─────────────────────────────────────────────────────────────────────────
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [76.99639, 51.54750, -46.67527];
    view_data.view_rotation = [-0.144709, 0.641239, -0.125076, -0.7408550];
    view_data.far = 10000.0;

    // ─────────────────────────────────────────────────────────────────────────
    // Primary Directional Light (sun)
    // ─────────────────────────────────────────────────────────────────────────
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.95, 0.85];  // Warm sunlight tint
    light_fragment_view.intensity = 0.0;
    light_fragment_view.position = [30.0, -50.0, 20.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;
    light_fragment_view.shadow_clipmaps = 12;

    // ─────────────────────────────────────────────────────────────────────────
    // Emissive Area Light Plane
    // ─────────────────────────────────────────────────────────────────────────
    const area_light_material = StandardMaterial.create("scifi_area_light_material");
    const area_light_material_id = area_light_material.material_id;
    area_light_material.set_albedo([0.9, 0.95, 1.0, 1.0]);  // Slightly cool white
    area_light_material.set_emission(2000.0);
    area_light_material.set_metallic(0.0);
    area_light_material.set_roughness(1.0);

    const cube_mesh = Mesh.cube();
    const area_light_plane = spawn_mesh_entity(
      [75.0, 24.0, -3.0],  // Positioned off to the side and elevated
      quat.fromEuler(quat.create(), 0, -90, 10),  // Angled slightly towards scene
      [20.0, 10.0, 1.0],  // Large flat plane
      cube_mesh,
      area_light_material_id
    );
    this.entities.push(area_light_plane);

    const area_light_plane2 = spawn_mesh_entity(
      [30.0, 20.0, 60.0],  // Positioned off to the side and elevated
      quat.fromEuler(quat.create(), 0, 45, 10),  // Angled slightly towards scene
      [20.0, 10.0, 1.0],  // Large flat plane
      cube_mesh,
      area_light_material_id
    );
    this.entities.push(area_light_plane2);

    // ─────────────────────────────────────────────────────────────────────────
    // Load SciFi City GLTF Model
    // ─────────────────────────────────────────────────────────────────────────
    let scifi_city_root = this.load_gltf_scene("engine/models/scifi-city/CityScene.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [2, 2, 2],
    );
    this.entities.push(scifi_city_root);

    log(`[${this.name}] SciFi City scene initialized.`);
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }
}

// ------------------------------------------------------------------------------------
// =============================== Bistro Test Scene ==================================
// ------------------------------------------------------------------------------------

export class BistroTestScene extends Scene {
  name = "BistroTestScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.move_speed = 30.0;
    freeform_arcball_control_processor.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [35.0, 15.0, 40.0];
    view_data.view_rotation = quat.fromEuler(quat.create(), -10.0, 135.0, 0.0);
    view_data.far = 10000.0;

    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.98, 0.95];
    light_fragment_view.intensity = 10.0;
    light_fragment_view.position = [40.0, 70.0, 30.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    const bistro_root = this.load_gltf_scene(
      "engine/models/bistro/bistro_exterior.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [1.0, 1.0, 1.0]
    );
    this.entities.push(bistro_root);

    log(`[${this.name}] Bistro scene initialized.`);
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }
}

// ------------------------------------------------------------------------------------
// =============================== 3D UI Test Scene ====================================
// ------------------------------------------------------------------------------------

export class UI3DTestScene extends Scene {
  name = "UI3DTestScene";
  entities = [];
  pulse = 0;
  counter = 0;
  selected_mode = "Layout";
  marker_position = [0.0, 0.0, 1.5];
  marker_rotation = quat.create();
  marker_half_extent = 1.03;

  init(parent_context) {
    super.init(parent_context);

    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.move_speed = 10.0;
    freeform_arcball_control_processor.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");
    this.show_dev_cursor();

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0.0, 2.4, -8.0];
    view_data.view_rotation = [0.0, 0.0, 0.0, 1.0];
    view_data.far = 1000.0;

    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.96, 0.9, 1.0];
    light_fragment_view.intensity = 5.0;
    light_fragment_view.position = [4.0, 6.0, -3.0, 1.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    const marker_material = StandardMaterial.create("UI3D_Marker");
    marker_material.set_albedo([0.08, 0.16, 0.2, 1.0]);
    marker_material.set_emission(0.15);

    this.marker_rotation = quat.fromEuler(quat.create(), 0, 45, 0);
    const marker = spawn_mesh_entity(
      this.marker_position,
      this.marker_rotation,
      [1.0, 1.0, 1.0],
      Mesh.cube(),
      marker_material.material_id
    );
    this.entities.push(marker);
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);

    this.pulse += delta_time;
    const pulse = 0.5 + Math.sin(this.pulse * 2.25) * 0.5;
    const scan_progress = 0.18 + pulse * 0.72;
    const load_progress = 0.54 + Math.sin(this.pulse * 1.15 + 0.8) * 0.18;
    const accent = [0.16 + pulse * 0.18, 0.72, 0.86, 1.0];

    this.render_hud_panel({
      position: [0.0, 3.7, 5.0],
      billboard: true,
      width: 4.8,
      height: 3.1,
      title: "HUD TEST",
      subtitle: "WORLD-SPACE UI",
      badge: "LIVE",
      pulse,
      scan_progress,
      load_progress,
      accent,
      interactive: true,
    });

    this.render_cube_hud_faces(pulse, scan_progress, load_progress, accent);
  }

  render_cube_hud_faces(pulse, scan_progress, load_progress, accent) {
    const face_panels = [
      { title: "FRONT", normal: [0, 0, -1], right: [-1, 0, 0], up: [0, 1, 0] },
      { title: "BACK", normal: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0] },
      { title: "LEFT", normal: [-1, 0, 0], right: [0, 0, 1], up: [0, 1, 0] },
      { title: "RIGHT", normal: [1, 0, 0], right: [0, 0, -1], up: [0, 1, 0] },
      { title: "TOP", normal: [0, 1, 0], right: [1, 0, 0], up: [0, 0, -1] },
      { title: "BOTTOM", normal: [0, -1, 0], right: [1, 0, 0], up: [0, 0, 1] },
    ];

    for (let i = 0; i < face_panels.length; i++) {
      const face = face_panels[i];
      const normal = vec3.transformQuat(vec3.create(), face.normal, this.marker_rotation);
      const right = vec3.transformQuat(vec3.create(), face.right, this.marker_rotation);
      const up = vec3.transformQuat(vec3.create(), face.up, this.marker_rotation);
      const position = vec3.scaleAndAdd(
        vec3.create(),
        this.marker_position,
        normal,
        this.marker_half_extent
      );

      this.render_hud_panel({
        position,
        right,
        up,
        width: 1.72,
        height: 1.18,
        title: face.title,
        subtitle: "FACE HUD",
        badge: `${i + 1}/6`,
        pulse,
        scan_progress: Math.max(0.08, Math.min(0.96, scan_progress - i * 0.04)),
        load_progress: Math.max(0.08, Math.min(0.96, load_progress + i * 0.035)),
        accent,
        compact: true,
        interactive: false,
      });
    }
  }

  render_hud_panel({
    position,
    billboard = false,
    right,
    up,
    width,
    height,
    title,
    subtitle,
    badge,
    pulse,
    scan_progress,
    load_progress,
    accent,
    compact = false,
    interactive = false,
  }) {
    const scale = compact ? 0.36 : 1.0;

    UI3D.panel({
      position,
      billboard,
      right,
      up,
      width,
      height,
      layout: "column",
      gap: 0.12 * scale,
      padding: 0.18 * scale,
      background_color: [0.018, 0.024, 0.032, 0.9],
      border: { width: 0.025 * scale, color: [0.32, 0.78, 0.86, 0.45] },
      corner_radius: 0.12 * scale,
      z_order: 2,
    }, () => {
      UI3D.begin_container({
        x: 0,
        y: 0,
        width: "100%",
        height: 0.46 * scale,
        layout: "row",
        gap: 0.12 * scale,
      });

      UI3D.label(title, {
        x: 0,
        y: 0,
        width: 2.25 * scale,
        height: "100%",
        text_color: [0.9, 0.98, 1.0, 1.0],
        text_align: "left",
        text_valign: "middle",
      });
      UI3D.label(subtitle, {
        x: 0,
        y: 0,
        width: 1.25 * scale,
        height: "100%",
        text_color: [0.48, 0.82, 0.88, 0.95],
        text_align: "center",
        text_valign: "middle",
        background_color: [0.05, 0.13, 0.16, 0.74],
        border: { width: 0.012 * scale, color: [0.3, 0.75, 0.82, 0.35] },
        corner_radius: 0.08 * scale,
      });
      UI3D.label(badge, {
        x: 0,
        y: 0,
        width: 0.68 * scale,
        height: "100%",
        text_color: [0.96, 0.68, 0.28, 1.0],
        text_align: "center",
        text_valign: "middle",
        background_color: [0.15, 0.08, 0.025, 0.82],
        corner_radius: 0.08 * scale,
      });

      UI3D.end_container();

      UI3D.panel({
        x: 0,
        y: 0,
        width: "100%",
        height: 0.74 * scale,
        layout: "row",
        gap: 0.12 * scale,
        padding: 0.12 * scale,
        background_color: [0.035, 0.048, 0.06, 0.82],
        border: { width: 0.01 * scale, color: [0.4, 0.72, 0.78, 0.22] },
        corner_radius: 0.1 * scale,
      }, () => {
        this.stat_tile("Signal", `${Math.round(82 + pulse * 12)}%`, [0.5, 0.9, 0.76, 1.0], scale);
        this.stat_tile("Latency", `${(1.8 + (1 - pulse) * 0.8).toFixed(1)}ms`, [0.98, 0.76, 0.38, 1.0], scale);
        this.stat_tile("Events", String(this.counter), [0.65, 0.78, 1.0, 1.0], scale);
      });

      this.progress_row("Scene Sync", scan_progress, accent, scale);
      this.progress_row("Batch Load", load_progress, [0.66, 0.55, 1.0, 1.0], scale);

      if (!interactive) {
        return;
      }

      UI3D.begin_container({
        x: 0,
        y: 0,
        width: "100%",
        height: 0.46 * scale,
        layout: "row",
        gap: 0.1 * scale,
      });

      const button_base = {
        x: 0,
        y: 0,
        width: 0.96 * scale,
        height: "100%",
        text_color: [0.92, 0.98, 1.0, 1.0],
        background_color: [0.055, 0.095, 0.12, 0.94],
        hover_color: [0.09, 0.26, 0.31, 0.98],
        active_color: [0.12, 0.44, 0.52, 1.0],
        border: { width: 0.012 * scale, color: [0.36, 0.78, 0.86, 0.42] },
        corner_radius: 0.08 * scale,
      };

      if (UI3D.button("Orbit", button_base).clicked) {
        this.selected_mode = "Orbit";
      }
      if (UI3D.button("Inspect", button_base).clicked) {
        this.selected_mode = "Inspect";
      }
      if (UI3D.button("Pulse +", { ...button_base, width: 1.08 * scale }).clicked) {
        this.counter++;
      }
      UI3D.label(this.selected_mode.toUpperCase(), {
        x: 0,
        y: 0,
        width: 1.0 * scale,
        height: "100%",
        text_color: [0.94, 0.72, 0.34, 1.0],
        text_align: "center",
        text_valign: "middle",
        background_color: [0.13, 0.08, 0.025, 0.7],
        corner_radius: 0.08 * scale,
      });

      UI3D.end_container();
    });
  }

  stat_tile(label_text, value_text, color, scale = 1.0) {
    UI3D.panel({
      x: 0,
      y: 0,
      width: 1.31 * scale,
      height: "100%",
      layout: "column",
      gap: 0.035 * scale,
      padding: 0.075 * scale,
      background_color: [0.018, 0.026, 0.034, 0.9],
      border: { width: 0.008 * scale, color: [color[0], color[1], color[2], 0.26] },
      corner_radius: 0.08 * scale,
    }, () => {
      UI3D.label(label_text, {
        x: 0,
        y: 0,
        width: "100%",
        height: 0.2 * scale,
        text_color: [0.48, 0.6, 0.64, 1.0],
        text_align: "left",
        text_valign: "middle",
      });
      UI3D.label(value_text, {
        x: 0,
        y: 0,
        width: "100%",
        height: 0.32 * scale,
        text_color: color,
        text_align: "left",
        text_valign: "middle",
      });
    });
  }

  progress_row(label_text, value, color, scale = 1.0) {
    const clamped_value = Math.max(0, Math.min(1, value));
    UI3D.panel({
      x: 0,
      y: 0,
      width: "100%",
      height: 0.38 * scale,
      layout: "row",
      gap: 0.12 * scale,
      padding: 0.07 * scale,
      background_color: [0.025, 0.036, 0.046, 0.72],
      corner_radius: 0.08 * scale,
    }, () => {
      UI3D.label(label_text, {
        x: 0,
        y: 0,
        width: 1.15 * scale,
        height: "100%",
        text_color: [0.62, 0.76, 0.78, 1.0],
        text_align: "left",
        text_valign: "middle",
      });
      UI3D.panel({
        x: 0,
        y: 0,
        width: 2.33 * scale,
        height: "100%",
        background_color: [0.07, 0.08, 0.09, 0.92],
        border: { width: 0.008 * scale, color: [0.25, 0.32, 0.35, 0.55] },
        corner_radius: 0.06 * scale,
      }, () => {
        UI3D.rect({
          x: 0,
          y: 0,
          width: `${Math.round(clamped_value * 100)}%`,
          height: "100%",
          background_color: [color[0], color[1], color[2], 0.86],
          corner_radius: 0.06 * scale,
        });
      });
      UI3D.label(`${Math.round(clamped_value * 100)}%`, {
        x: 0,
        y: 0,
        width: 0.55 * scale,
        height: "100%",
        text_color: color,
        text_align: "right",
        text_valign: "middle",
      });
    });
  }
}


// ------------------------------------------------------------------------------------
// =============================== Main ==============================================
// ------------------------------------------------------------------------------------

(async () => {
  const simulator = await Simulator.create("gpu-canvas", "ui-canvas", {
    project: {
      name: "example",
      root: "example",
      cvar_config: example_cvar_config,
    },
  });

  // Scene names are durable IDs. For example, SponzaScene automatically reads
  // and writes assets/example/scenes/SponzaScene.scene.bin.
  const bvh_scene = new BVHScene("BVHScene");
  const rendering_scene = new RenderingScene("RenderingScene");
  const ml_scene = new MLScene("MLScene");
  const solar_ecs_scene = new SolarECSTestScene("SolarECSTestScene");
  const voxel_terrain_scene = new VoxelTerrainScene("VoxelTerrainScene");
  const object_painting_scene = new ObjectPaintingScene("ObjectPaintingScene");
  const gltf_model_scene = new GLTFModelScene("GLTFModelScene");
  const textures_scene = new TexturesScene("TexturesScene");
  const gi_test_scene = new GITestScene("GITestScene");
  const shadow_test_scene = new ShadowTestScene("ShadowTestScene");
  const sponza_scene = new SponzaScene("SponzaScene");
  const living_room_scene = new LivingRoomScene("LivingRoomScene");
  const city_scene = new CityScene("CityScene");
  const scifi_city_scene = new SciFiCityScene("SciFiCityScene");
  const bistro_test_scene = new BistroTestScene("BistroTestScene");
  const ui_3d_scene = new UI3DTestScene("UI3DTestScene");

  const scene_switcher = new SceneSwitcher("SceneSwitcher");
  //await scene_switcher.add_scene(solar_ecs_scene);
  //await scene_switcher.add_scene(rendering_scene);
  //await scene_switcher.add_scene(bvh_scene);
  //await scene_switcher.add_scene(ml_scene);
  //await scene_switcher.add_scene(voxel_terrain_scene);
  //await scene_switcher.add_scene(object_painting_scene);
  //await scene_switcher.add_scene(gltf_model_scene);
  //await scene_switcher.add_scene(textures_scene);
  //await scene_switcher.add_scene(gi_test_scene);
  //await scene_switcher.add_scene(shadow_test_scene);
  //await scene_switcher.add_scene(ui_3d_scene);
  await scene_switcher.add_scene(sponza_scene);
  //await scene_switcher.add_scene(living_room_scene);
  //await scene_switcher.add_scene(city_scene);
  //await scene_switcher.add_scene(scifi_city_scene);
  //await scene_switcher.add_scene(bistro_test_scene);

  simulator.add_sim_layer(scene_switcher);
  simulator.add_sim_layer(new SceneSettingsPanel(scene_switcher));

  simulator.run();
})();

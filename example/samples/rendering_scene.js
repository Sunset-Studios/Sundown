// ═══════════════════════════════════════════════════════════════════════════════
// RenderingScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { FragmentGpuBuffer } from "../../engine/src/core/ecs/solar/memory.js";
import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { PostProcessStack } from "../../engine/src/renderer/post_process/post_process_stack.js";
import { ComputeTaskQueue } from "../../engine/src/renderer/task_queues/compute_task_queue.js";
import { TransformFragment } from "../../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { StandardMaterial } from "../../engine/src/renderer/material.js";
import { Mesh } from "../../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { log } from "../../engine/src/utility/logging.js";

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
        paths: ["example/textures/voxel/dirt_albedo.ktx2"],
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
        paths: ["example/textures/voxel/dirt_roughness.ktx2"],
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

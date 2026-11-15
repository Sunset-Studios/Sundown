// ═══════════════════════════════════════════════════════════════════════════════
// SolarECSTestScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { TransformFragment } from "../../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { StandardMaterial } from "../../engine/src/renderer/material.js";
import { Mesh } from "../../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { log } from "../../engine/src/utility/logging.js";
import { quat } from "gl-matrix";

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
      "engine/textures/gradientbox/px.ktx2",
      "engine/textures/gradientbox/nx.ktx2",
      "engine/textures/gradientbox/ny.ktx2",
      "engine/textures/gradientbox/py.ktx2",
      "engine/textures/gradientbox/pz.ktx2",
      "engine/textures/gradientbox/nz.ktx2",
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
      transform.position = [center[0] + offset[0], center[1] + offset[1], center[2] + offset[2]];
      transform.scale = [scale, scale, scale];
    }
  }

  #random_cloud_offset() {
    const theta = Math.random() * Math.PI * 2;
    const z = Math.random() * 2 - 1;
    const radial = Math.sqrt(Math.max(0, 1 - z * z));
    const radius = this.cloud_radius * Math.cbrt(Math.random());

    return [Math.cos(theta) * radial * radius, z * radius, Math.sin(theta) * radial * radius];
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

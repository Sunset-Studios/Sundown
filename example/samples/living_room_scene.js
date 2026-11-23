// ═══════════════════════════════════════════════════════════════════════════════
// LivingRoomScene — Sundown samples
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
    light_fragment_view.color = [1.0, 1.0, 1.0]; // Warm daylight tint
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
      [50, 1.0, 50],
      cube_mesh,
      ground_material_id
    );
    this.entities.push(ground_entity);

    // ─────────────────────────────────────────────────────────────────────────
    // Load Living Room GLTF Model - Instanced 100x in a 10x10 grid
    // ─────────────────────────────────────────────────────────────────────────
    const living_room_entity = this.load_gltf_scene(
      "example/models/living_room/living_room.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [1, 1, 1]
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

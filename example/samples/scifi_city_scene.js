// ═══════════════════════════════════════════════════════════════════════════════
// SciFiCityScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { StandardMaterial } from "../../engine/src/renderer/material.js";
import { Mesh } from "../../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { log } from "../../engine/src/utility/logging.js";
import { quat } from "gl-matrix";

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
    view_data.view_position = [76.99639, 51.5475, -46.67527];
    view_data.view_rotation = [-0.144709, 0.641239, -0.125076, -0.740855];
    view_data.far = 10000.0;

    // ─────────────────────────────────────────────────────────────────────────
    // Primary Directional Light (sun)
    // ─────────────────────────────────────────────────────────────────────────
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.95, 0.85]; // Warm sunlight tint
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
    area_light_material.set_albedo([0.9, 0.95, 1.0, 1.0]); // Slightly cool white
    area_light_material.set_emission(2000.0);
    area_light_material.set_metallic(0.0);
    area_light_material.set_roughness(1.0);

    const cube_mesh = Mesh.cube();
    const area_light_plane = spawn_mesh_entity(
      [75.0, 24.0, -3.0], // Positioned off to the side and elevated
      quat.fromEuler(quat.create(), 0, -90, 10), // Angled slightly towards scene
      [20.0, 10.0, 1.0], // Large flat plane
      cube_mesh,
      area_light_material_id
    );
    this.entities.push(area_light_plane);

    const area_light_plane2 = spawn_mesh_entity(
      [30.0, 20.0, 60.0], // Positioned off to the side and elevated
      quat.fromEuler(quat.create(), 0, 45, 10), // Angled slightly towards scene
      [20.0, 10.0, 1.0], // Large flat plane
      cube_mesh,
      area_light_material_id
    );
    this.entities.push(area_light_plane2);

    // ─────────────────────────────────────────────────────────────────────────
    // Load SciFi City GLTF Model
    // ─────────────────────────────────────────────────────────────────────────
    let scifi_city_root = this.load_gltf_scene(
      "engine/models/scifi-city/CityScene.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [2, 2, 2]
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

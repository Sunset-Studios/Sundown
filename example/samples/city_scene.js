// ═══════════════════════════════════════════════════════════════════════════════
// CityScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { log } from "../../engine/src/utility/logging.js";

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
    view_data.view_rotation = [-0.0160628, 0.78540349, -0.023472044, -0.613837182];
    view_data.far = 10000.0;

    // ─────────────────────────────────────────────────────────────────────────
    // Primary Directional Light (sun)
    // ─────────────────────────────────────────────────────────────────────────
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.95, 0.85]; // Warm sunlight tint
    light_fragment_view.intensity = 5.0;
    light_fragment_view.position = [30.0, 50.0, 20.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;
    light_fragment_view.shadow_clipmaps = 12;

    // ─────────────────────────────────────────────────────────────────────────
    // Load City GLTF Model
    // ─────────────────────────────────────────────────────────────────────────
    let city_root = this.load_gltf_scene(
      "example/models/city/City.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [2, 2, 2]
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

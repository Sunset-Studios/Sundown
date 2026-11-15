// ═══════════════════════════════════════════════════════════════════════════════
// BistroTestScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { log } from "../../engine/src/utility/logging.js";
import { quat } from "gl-matrix";

export class BistroTestScene extends Scene {
  name = "BistroTestScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    const camera_control = this.add_layer(FreeformArcballControlProcessor);
    camera_control.move_speed = 20.0;
    camera_control.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [11.0, 5.5, 24.0];
    view_data.view_rotation = quat.fromEuler(quat.create(), 0, 180, 0);
    view_data.near = 0.1;
    view_data.far = 1000.0;

    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.94, 0.82, 1.0];
    light_fragment_view.intensity = 8.0;
    light_fragment_view.position = [0.35, 0.8, 0.25, 0.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;
    light_fragment_view.shadow_clipmaps = 12;

    const bistro_root = this.load_gltf_scene(
      "engine/models/bistro/bistro_exterior.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [1, 1, 1],
      null,
      null,
      null,
      { blend_material_mode: "MASK" }
    );
    this.entities.push(bistro_root);

    log(`[${this.name}] Bistro exterior scene initialized.`);
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

// ═══════════════════════════════════════════════════════════════════════════════
// OfficeTestScene — Sundown samples
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

export class OfficeTestScene extends Scene {
  name = "OfficeTestScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    const camera_control = this.add_layer(FreeformArcballControlProcessor);
    camera_control.move_speed = 4.0;
    camera_control.min_move_speed = 1.0;
    camera_control.max_move_speed = 12.0;
    camera_control.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0.0, 1.65, 7.5];
    view_data.view_rotation = quat.fromEuler(quat.create(), 0, 180, 0);
    view_data.near = 0.05;
    view_data.far = 250.0;

    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.96, 0.9, 1.0];
    light_fragment_view.intensity = 3.0;
    light_fragment_view.position = [-0.35, 0.8, 0.25, 0.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;
    light_fragment_view.shadow_clipmaps = 8;

    const office_root = this.load_gltf_scene(
      "engine/models/office/office.gltf",
      [0, 0, 0],
      [0, 0, 0, 1],
      [1, 1, 1]
    );
    this.entities.push(office_root);

    log(`[${this.name}] Office scene initialized.`);
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

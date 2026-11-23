// ═══════════════════════════════════════════════════════════════════════════════
// SponzaScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { InputProvider } from "../../engine/src/input/input_provider.js";
import { InputKey } from "../../engine/src/input/input_types.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { StandardMaterial } from "../../engine/src/renderer/material.js";
import { Mesh } from "../../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { quat } from "gl-matrix";

export class SponzaScene extends Scene {
  name = "SponzaScene";
  entities = [];
  sway_light_enabled = false;
  sway_period_sec = 35.0;
  sway_angle_deg = 120.0;
  sun_light_entity = null;
  sun_light_base_dir = [0, 0, 0];
  sun_light_intensity_on = 10.0;
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
    light_fragment_view.position = [0.24, 0.24, 0.1];
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
      [50, 1.0, 50],
      cube_mesh,
      ground_material_id
    );
    this.entities.push(ground_entity);

    const root_entity = this.load_gltf_scene(
      "example/models/sponza/Sponza.gltf",
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

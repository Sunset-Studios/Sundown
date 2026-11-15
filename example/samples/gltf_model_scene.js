// ═══════════════════════════════════════════════════════════════════════════════
// GLTFModelScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { TransformFragment } from "../../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { Mesh } from "../../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { quat } from "gl-matrix";
import { world_label } from "../../engine/src/ui/3d/common.js";

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

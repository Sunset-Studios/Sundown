import { Scene } from "../engine/src/core/scene.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { spawn_mesh_entity, delete_entity } from "../engine/src/core/ecs/entity_utils.js";
import { LightType } from "../engine/src/core/minimal.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../engine/src/core/shared_data.js";
import { StandardMaterial } from "../engine/src/renderer/material.js";
import { Mesh } from "../engine/src/renderer/mesh.js";

export class MainScene extends Scene {
  entities = [];

  init() {
    super.init();

    const view = SharedViewBuffer.get_view_data(this.context.current_view);
    view.view_position = [-1, 22, 26];
    view.view_rotation = [-0.00061309, 0.9948077, -0.10095515, -0.00604141];
    SharedEnvironmentData.set_skydome("app_skydome");

    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);
    const light = EntityManager.get_fragment(light_entity, LightFragment);
    light.type = LightType.DIRECTIONAL;
    light.color = [1, 1, 1, 1];
    light.intensity = 30;
    light.position = [50, 15, 50, 1];
    light.active = true;
    light.is_primary_sun = 1;
    light.shadow_casting = 0;

    const mesh = Mesh.from_gltf("engine/models/cube/cube.gltf");
    const material = StandardMaterial.create("app_material");
    this.entities.push(
      spawn_mesh_entity([0, 0, 0], [0, 0, 0], [4, 4, 4], mesh, material.material_id)
    );
  }

  cleanup() {
    for (const entity of this.entities) delete_entity(entity);
    this.entities.length = 0;
    super.cleanup();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ObjectPaintingScene — Sundown samples
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
import { vec3 } from "gl-matrix";
import * as UI from "../../engine/src/ui/2d/immediate.js";

const info_panel_config = {
  layout: "column",
  gap: 4,
  y: 100,
  anchor_y: "bottom",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 600,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const info_label_config = {
  text_color: "#fff",
  x: 0,
  y: 0,
  wrap: true,
  font: "16px monospace",
  width: "100%",
  height: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

export class ObjectPaintingScene extends Scene {
  entities = [];
  sphere_mesh = null;
  brush_material_id = null;
  brush_entity = null;

  // --- Configurable parameters ---
  brush_radius = 2.0; // radius of the sphere brush
  brush_emit_intensity = 1.0; // material emission
  paint_rate = 0.25; // seconds between paint ticks
  spawn_count = 256; // objects per tick
  spawn_radius = 5.0; // radius of random paint sphere
  // -------------------------------

  last_paint_timer = 0;

  init(parent_context) {
    super.init(parent_context);

    // Arcball camera control
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Skybox + view
    SharedEnvironmentData.set_skybox("default_scene_skybox", [
      "engine/textures/simple_skybox/px.ktx2",
      "engine/textures/simple_skybox/nx.ktx2",
      "engine/textures/simple_skybox/ny.ktx2",
      "engine/textures/simple_skybox/py.ktx2",
      "engine/textures/simple_skybox/pz.ktx2",
      "engine/textures/simple_skybox/nz.ktx2",
    ]);
    SharedEnvironmentData.set_skybox_color([1, 1, 1, 1]);

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0, 0, 10];
    view_data.view_rotation = [0, 0, 0, 1];

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 10.0;
    light_fragment_view.position = [10, 30, 20];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Load sphere mesh & create transparent/emissive brush material
    this.sphere_mesh = Mesh.sphere();

    const object_material1 = StandardMaterial.create("ObjectPaintingObjectMaterial");
    this.object_material1_id = object_material1.material_id;
    const object_material2 = StandardMaterial.create("ObjectPaintingObjectMaterial2");
    this.object_material2_id = object_material2.material_id;
    const object_material3 = StandardMaterial.create("ObjectPaintingObjectMaterial3");
    this.object_material3_id = object_material3.material_id;

    object_material1.set_albedo([1.0, 1.0, 1.0, 1]);
    object_material1.set_roughness(0.01);
    object_material1.set_metallic(0.99);

    object_material2.set_albedo([0.3, 0.0, 0.0, 1]);
    object_material2.set_roughness(0.9);
    object_material2.set_metallic(0.1);

    object_material3.set_albedo([0.0, 0.3, 0.0, 1]);
    object_material3.set_roughness(0.9);
    object_material3.set_metallic(0.1);
  }

  update(delta_time) {
    super.update(delta_time);

    // Move brush to follow mouse
    const world_pos = UI.UIContext.input_state.world_position;
    const view = SharedViewBuffer.get_view_data(0);
    const view_dir = view.forward;
    const paint_pos = vec3.scaleAndAdd(vec3.create(), world_pos, view_dir, 50);

    // While Space key is held, paint objects every paint_rate seconds
    if (InputProvider.get_state(InputKey.K_Space)) {
      this.last_paint_timer += delta_time;
      if (this.last_paint_timer >= this.paint_rate) {
        this.last_paint_timer -= this.paint_rate;
        for (let i = 0; i < this.spawn_count; i++) {
          // uniform random point in sphere
          const r = this.spawn_radius * Math.cbrt(Math.random());
          const theta = 2 * Math.PI * Math.random();
          const phi = Math.acos(2 * Math.random() - 1);
          const x = paint_pos[0] + r * Math.sin(phi) * Math.cos(theta);
          const y = paint_pos[1] + r * Math.sin(phi) * Math.sin(theta);
          const z = paint_pos[2] + r * Math.cos(phi);

          // spawn a sphere instance
          const entity = spawn_mesh_entity(
            [x, y, z],
            [0, 0, 0, 1],
            [0.3 + Math.random() * 0.3, 0.3 + Math.random() * 0.3, 0.3 + Math.random() * 0.3],
            this.sphere_mesh,
            [this.object_material1_id, this.object_material2_id, this.object_material3_id][
              Math.floor(Math.random() * 3)
            ]
          );
          this.entities.push(entity);
        }
      }
    } else {
      // reset timer so we can paint immediately on next hold
      this.last_paint_timer = this.paint_rate;
    }

    // show fixed center cursor
    this.render_ui();
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.remove_layer(FreeformArcballControlProcessor);
    super.cleanup();
  }

  // draw a fixed, 2D crosshair at screen center
  render_ui() {
    const { width, height } = UI.UIContext.canvas_size;
    const cx = width * 0.5;
    const cy = height * 0.5;

    // vertical line
    UI.panel(
      {
        x: cx - 1,
        y: cy - 10,
        width: 2,
        height: 20,
        background_color: "#FFFFFF",
        dont_consume_cursor_events: true,
      },
      () => {}
    );

    // horizontal line
    UI.panel(
      {
        x: cx - 10,
        y: cy - 1,
        width: 20,
        height: 2,
        background_color: "#FFFFFF",
        dont_consume_cursor_events: true,
      },
      () => {}
    );

    // help text overlay
    UI.panel(info_panel_config, () => {
      UI.label("Hold [Space] to Paint objects", info_label_config);
      UI.label("Total entities: " + EntityManager.get_entity_count(), info_label_config);
    });
  }
}

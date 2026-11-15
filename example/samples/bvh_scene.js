// ═══════════════════════════════════════════════════════════════════════════════
// BVHScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { InputProvider } from "../../engine/src/input/input_provider.js";
import { InputKey } from "../../engine/src/input/input_types.js";
import { BVHDebugRenderer } from "../../engine/src/core/subsystems/bvh_debug_renderer.js";
import { BVHRaycast } from "../../engine/src/acceleration/bvh_raycast.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { StaticMeshFragment } from "../../engine/src/core/ecs/fragments/static_mesh_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { StandardMaterial } from "../../engine/src/renderer/material.js";
import { Mesh } from "../../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { log } from "../../engine/src/utility/logging.js";
import { vec4, quat } from "gl-matrix";
import * as UI from "../../engine/src/ui/2d/immediate.js";
import { world_label } from "../../engine/src/ui/3d/common.js";

const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: 400,
  x: 25,
  anchor_x: "right",
  anchor_y: "bottom",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 600,
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
};

const stats_label_config = {
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

const button_config = {
  y: 0,
  x: 0,
  width: "fit-content",
  font: "bold 16px monospace",
  height: 30,
  background_color: "#FFA500",
  text_color: "#111111",
  corner_radius: 5,
  text_padding: 10,
};

export class BVHScene extends Scene {
  name = "BVHScene";
  show_ui = false;
  entities = [];
  selected_entity = null;
  last_ray_origin = null;
  last_ray_direction = null;
  last_ray = null;

  init(parent_context) {
    super.init(parent_context);

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skydome
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // Reset view to a good position for the BVH scene
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [47.0751, 55.28902, 106.885414];
    view_data.view_rotation = [-0.023805, 0.97379, -0.190533, -0.121665];

    this.aabb_tree_debug_renderer = this.get_layer(BVHDebugRenderer);

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 10;
    light_fragment_view.position = [50, 20, 50];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = true;
    this.entities.push(light_entity);

    // Create a default material
    const default_material = StandardMaterial.create("BVHDefaultMaterial");
    this.default_material_id = default_material.material_id;
    default_material.set_albedo([0.5, 0.5, 0.5, 1]);
    default_material.set_normal([0, 1, 0, 1]);
    default_material.set_roughness(0.5);
    default_material.set_metallic(0.5);
    default_material.set_emission(0.1);

    // Create a default material for the selected entity
    const selected_entity_material = StandardMaterial.create("BVHSelectedEntityMaterial");
    this.selected_entity_material_id = selected_entity_material.material_id;
    selected_entity_material.set_albedo([1.0, 0.3, 0.3, 1]);
    selected_entity_material.set_emission(1.0);

    // Create a sphere mesh
    this.sphere_mesh = Mesh.sphere();

    // Create a cube mesh
    this.cube_mesh = Mesh.cube();

    // Setup a grid of entities
    this.setup_entity_grid();
  }

  cleanup() {
    for (let i = 0; i < this.entities.length; i++) {
      delete_entity(this.entities[i]);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);

    world_label("BVH Test Scene", [10, 65, 25], {
      width: 40.0,
      height: 10.0,
      billboard: false,
    });

    this.handle_input();

    // Run a raycast every frame from mouse position
    if (this.entities.length > 0) {
      this.run_raycast();
    }

    this.render_ui();
  }

  setup_entity_grid() {
    // Create a grid of entities for testing
    const grid_size = 5;
    const spacing = 3.0;

    for (let x = 0; x < grid_size; x++) {
      for (let z = 0; z < grid_size; z++) {
        for (let y = 0; y < grid_size; y++) {
          // Alternate between cubes and spheres
          const mesh = (x + z) % 2 === 0 ? this.cube_mesh : this.sphere_mesh;

          // Random position with small offset
          const position = [x * spacing, y * spacing, z * spacing];

          // Random scale
          const scale = [
            0.5 /*+ Math.random() * 0.5*/, 0.5 /*+ Math.random() * 0.5*/,
            0.5 /*+ Math.random() * 0.5*/,
          ];

          // Create entity
          const entity = spawn_mesh_entity(
            position,
            quat.fromEuler(quat.create(), 0, 0, 0),
            scale,
            mesh,
            this.default_material_id
          );

          this.entities.push(entity);
        }
      }
    }

    log(`[BVH] Spawned ${this.entities.length} entities`);
  }

  handle_input() {
    // Toggle raycast UI
    if (InputProvider.get_action(InputKey.K_u)) {
      this.show_ui = !this.show_ui;
      if (this.show_ui) {
        this.show_dev_cursor();
      } else {
        this.hide_dev_cursor();
      }
    }

    // Add a new entity at the previous hit point
    if (InputProvider.get_action(InputKey.K_Space) && this.last_ray) {
      const hit = BVHRaycast.get_hit_result(this.last_ray);

      // Create entity at hit point
      const entity = spawn_mesh_entity(
        hit.position,
        [0, 0, 0, 1],
        [0.5, 0.5, 0.5],
        Math.random() > 0.5 ? this.cube_mesh : this.sphere_mesh,
        this.default_material_id
      );

      this.entities.push(entity);
    }

    // Delete the selected entity
    if (InputProvider.get_action(InputKey.K_Backspace) && this.selected_entity) {
      delete_entity(this.selected_entity);

      // Remove from entities array
      const index = this.entities.indexOf(this.selected_entity);
      if (index >= 0) {
        this.entities.splice(index, 1);
      }

      this.selected_entity = null;
    }
  }

  run_raycast() {
    // Get camera position and direction
    const view_data = SharedViewBuffer.get_view_data(0);
    if (!view_data) return;

    // Get cursor world position (this is a point on the far plane)
    const cursor_world_position = UI.UIContext.input_state.world_position;
    if (!cursor_world_position) return;

    if (this.last_ray) {
      const hit = BVHRaycast.get_hit_result(this.last_ray);
      this.process_raycast_results(hit);
    }

    // Use camera position as ray origin
    this.last_ray_origin = view_data.view_position;

    // Calculate ray direction from camera to cursor world position
    this.last_ray_direction = vec4.sub(
      vec4.create(),
      cursor_world_position,
      view_data.view_position
    );
    // Normalize the direction vector
    const length = Math.sqrt(
      this.last_ray_direction[0] * this.last_ray_direction[0] +
        this.last_ray_direction[1] * this.last_ray_direction[1] +
        this.last_ray_direction[2] * this.last_ray_direction[2]
    );

    this.last_ray_direction[0] /= length;
    this.last_ray_direction[1] /= length;
    this.last_ray_direction[2] /= length;

    // Perform raycast based on current mode by requesting a ray from the BVHRaycast class
    this.last_ray = BVHRaycast.request_ray();
    this.last_ray.setup(this.last_ray_origin, this.last_ray_direction);
  }

  process_raycast_results(hit) {
    // Update selected entity highlighting
    const previous_selected_entity = this.selected_entity;

    // Highlight the selected entity by writing to the material buffer
    this.selected_entity = EntityManager.get_entity_from_id(hit.user_data);

    if (previous_selected_entity === this.selected_entity) {
      return;
    }

    if (previous_selected_entity !== null) {
      // Reset previous selection
      const static_mesh_fragment = EntityManager.get_fragment(
        previous_selected_entity,
        StaticMeshFragment
      );
      if (static_mesh_fragment) {
        static_mesh_fragment.material_slots = [BigInt(this.default_material_id)];
      }
    }

    if (this.selected_entity !== null) {
      const static_mesh_fragment = EntityManager.get_fragment(
        this.selected_entity,
        StaticMeshFragment
      );
      if (static_mesh_fragment) {
        static_mesh_fragment.material_slots = [BigInt(this.selected_entity_material_id)];
      }
    }
  }

  render_ui() {
    if (!this.show_ui) return;

    // Use immediate mode UI panel instead of window
    UI.panel(stats_panel_config, () => {
      UI.label("Raycast Results:", stats_label_config);

      if (this.last_ray) {
        const hit = BVHRaycast.get_hit_result(this.last_ray);
        UI.label(`Hit Entity: ${hit.user_data}`, stats_label_config);
        UI.label(`Distance: ${hit.distance.toFixed(2)}`, stats_label_config);

        const pos_text = `Position: [${hit.position[0].toFixed(2)}, ${hit.position[1].toFixed(2)}, ${hit.position[2].toFixed(2)}]`;
        UI.label(pos_text, stats_label_config);

        const normal_text = `Normal: [${hit.normal[0].toFixed(2)}, ${hit.normal[1].toFixed(2)}, ${hit.normal[2].toFixed(2)}]`;
        UI.label(normal_text, stats_label_config);
      } else {
        UI.label("No hit", stats_label_config);
      }

      // Add some spacing
      UI.begin_container({ height: 10, width: "100%" });
      UI.end_container();

      // Instructions section
      UI.label("Controls:", stats_label_config);

      const control_labels = [
        "U: Toggle UI",
        "Space: Add object at hit point",
        "Delete: Remove selected object",
      ];

      for (const text of control_labels) {
        UI.label(text, stats_label_config);
      }

      // Add buttons for common actions
      UI.begin_container({
        layout: "row",
        x: 0,
        gap: 10,
        height: 40,
        padding_top: 10,
      });

      const add_one_button = UI.button("Add Object", button_config);
      if (add_one_button.clicked) {
        if (this.last_ray) {
          const hit = BVHRaycast.get_hit_result(this.last_ray);
          const entity = spawn_mesh_entity(
            hit.position,
            quat.fromEuler(quat.create(), 0, 0, 0),
            [0.5, 0.5, 0.5],
            Math.random() > 0.5 ? this.cube_mesh : this.sphere_mesh,
            this.default_material_id
          );
          this.entities.push(entity);
        }
      }

      const delete_button = UI.button("Delete Selected", button_config);
      if (delete_button.clicked) {
        if (this.selected_entity) {
          delete_entity(this.selected_entity);
          const index = this.entities.indexOf(this.selected_entity);
          if (index >= 0) {
            this.entities.splice(index, 1);
          }
          this.selected_entity = null;
        }
      }

      // Add button for stress test
      const add_button = UI.button("Add 1000 Objects", button_config);
      if (add_button.clicked) {
        this.add_random_objects(1000);
      }

      UI.end_container();
    });
  }

  add_random_objects(count) {
    // Create random objects in a sphere around the camera
    const view_data = SharedViewBuffer.get_view_data(0);
    if (!view_data) return;

    const center = view_data.view_position;
    const radius = 50.0;

    for (let i = 0; i < count; i++) {
      // Random point in sphere
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = radius * Math.cbrt(Math.random()); // Cube root for uniform distribution

      const x = center[0] + r * Math.sin(phi) * Math.cos(theta);
      const y = center[1] + r * Math.sin(phi) * Math.sin(theta);
      const z = center[2] + r * Math.cos(phi);

      // Random size
      const scale = 0.2 + Math.random() * 0.8;

      // Create entity
      const entity = spawn_mesh_entity(
        [x, y, z],
        [0, 0, 0, 1],
        [scale, scale, scale],
        Math.random() > 0.5 ? this.cube_mesh : this.sphere_mesh,
        this.default_material_id
      );

      this.entities.push(entity);
    }
  }
}

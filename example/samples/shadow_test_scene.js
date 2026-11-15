// ═══════════════════════════════════════════════════════════════════════════════
// ShadowTestScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { TransformFragment } from "../../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { VisibilityFragment } from "../../engine/src/core/ecs/fragments/visibility_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { StandardMaterial } from "../../engine/src/renderer/material.js";
import { Mesh } from "../../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { log } from "../../engine/src/utility/logging.js";
import { quat } from "gl-matrix";

export class ShadowTestScene extends Scene {
  name = "ShadowTestScene";
  entities = [];
  swaying_balls = [];
  swaying_ball_base_positions = [];

  init(parent_context) {
    super.init(parent_context);

    // Add arcball camera control
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.move_speed = 75.0;
    freeform_arcball_control_processor.set_scene(this);

    // Configure skybox
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // Position the camera high above the city
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [205.515, 40.4267, 273.94];
    view_data.view_rotation = [0.04745, 0.57238, 0.03344, -0.81212];
    view_data.far = 100000.0;

    // Create a sun-like directional light
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 0.9];
    light_fragment_view.intensity = 0.2;
    light_fragment_view.position = [30, 55, 40];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // // Create a point light
    const point_light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(point_light_entity);

    const point_light_fragment_view = EntityManager.get_fragment(point_light_entity, LightFragment);
    point_light_fragment_view.type = LightType.POINT;
    point_light_fragment_view.color = [1, 1, 0];
    point_light_fragment_view.intensity = 5.0;
    point_light_fragment_view.position = [0, 20, 0];
    point_light_fragment_view.radius = 20.0;
    point_light_fragment_view.active = true;
    point_light_fragment_view.shadow_casting = false;

    const num_point_lights = 12;
    EntityManager.set_entity_instance_count(point_light_entity, num_point_lights);

    // Ground material
    const ground_material = StandardMaterial.create("shadow_ground_material");
    const ground_material_id = ground_material.material_id;
    ground_material.set_albedo([0.5, 0.5, 0.5, 1]);
    ground_material.set_roughness(0.9);
    ground_material.set_metallic(0.0);

    // Building material
    const building_material = StandardMaterial.create("shadow_building_material");
    const building_material_id = building_material.material_id;
    building_material.set_albedo([0.35, 0.35, 0.35, 1]);
    building_material.set_roughness(0.9);
    building_material.set_metallic(0.0);

    // Shared cube mesh
    const cube_mesh = Mesh.cube();
    const sphere_mesh = Mesh.sphere();

    const grid_size = 80; // 80 × 80 buildings
    const building_spacing = 35.0; // distance between building centres (was 20.0)
    const building_base_size = 6.0; // footprint of each building
    const half_grid = (grid_size - 1) * building_spacing * 0.5;

    // Create an expansive ground plane
    const ground_plane_size = 3000.0;
    this.ground_entity = spawn_mesh_entity(
      [0.0, 0.0, 0.0],
      [0, 0, 0, 1],
      [ground_plane_size, 1.0, ground_plane_size],
      cube_mesh,
      ground_material_id
    );
    this.entities.push(this.ground_entity);

    // Compute corner positions using half_grid
    const offset = half_grid * 0.9; // Slightly inset from absolute corners to avoid clipping buildings

    this.swaying_ball_base_positions = [
      [offset, 350, offset],
      [offset, 350, -offset],
      [-offset, 350, offset],
      [-offset, 350, -offset],
    ];

    // Create ball material
    const ball_material = StandardMaterial.create("swaying_ball_material");
    ball_material.set_albedo([0.9, 0.9, 0.2, 1]);
    ball_material.set_emission(0.3);
    ball_material.set_roughness(0.2);
    ball_material.set_metallic(0.9);

    // Create spheres
    for (const base_pos of this.swaying_ball_base_positions) {
      const ball_position = [...base_pos];
      const ball_scale = [40, 40, 40];
      const entity = spawn_mesh_entity(
        ball_position,
        [0, 0, 0, 1],
        ball_scale,
        sphere_mesh,
        ball_material.material_id
      );
      this.swaying_balls.push(entity);
      this.entities.push(entity);
    }

    const building_entity = spawn_mesh_entity(
      [0.0, 0.0, 0.0, 1.0],
      quat.fromEuler(quat.create(), 0.0, 0.0, 0.0),
      [0.0, 0.0, 0.0],
      cube_mesh,
      building_material_id
    );
    EntityManager.set_entity_instance_count(building_entity, grid_size * grid_size);
    this.entities.push(building_entity);

    let instance_index = 0;
    for (let gx = 0; gx < grid_size; gx++) {
      for (let gz = 0; gz < grid_size; gz++) {
        // More randomized height: most buildings short, few very tall
        const base = 8.0;
        const max = 180.0;
        const exponent = 4.0; // Higher = more short buildings
        const height = base + Math.pow(Math.random(), exponent) * (max - base);

        const position = [
          gx * building_spacing - half_grid,
          height * 0.5,
          gz * building_spacing - half_grid,
        ];
        const scale = [building_base_size, height, building_base_size];

        const transform_fragment = EntityManager.get_fragment(
          building_entity,
          TransformFragment,
          instance_index
        );
        transform_fragment.position = position;
        transform_fragment.scale = scale;

        const visibility_fragment = EntityManager.get_fragment(
          building_entity,
          VisibilityFragment,
          instance_index
        );
        visibility_fragment.occluder = 0;

        instance_index++;
      }
    }

    log(`[${this.name}] Spawned ${this.entities.length} entities.`);

    // --- Neon Signs ---------------------------------------------------------
    // Create several vibrant emissive materials for different neon colours.
    const neon_colours = [
      { name: "neon_cyan", color: [0.0, 1.0, 1.0, 1.0] },
      { name: "neon_magenta", color: [1.0, 0.0, 1.0, 1.0] },
      { name: "neon_yellow", color: [1.0, 1.0, 0.0, 1.0] },
      { name: "neon_orange", color: [1.0, 0.5, 0.0, 1.0] },
      { name: "neon_green", color: [0.0, 1.0, 0.0, 1.0] },
    ];

    const neon_material_ids = neon_colours.map((c) => {
      const m = StandardMaterial.create(`shadow_${c.name}`);
      m.set_albedo(c.color);
      m.set_emission(100.0); // extremely bright
      m.set_roughness(0.1);
      m.set_metallic(0.9);
      return m.material_id;
    });

    const neon_sign_transforms = [];
    const neon_sign_material_indices = [];

    // Decide which buildings receive signs and where they go.
    for (let i = 0; i < instance_index; i++) {
      // Roughly 30 % of buildings receive a sign.
      if (Math.random() < 0.3) {
        const building_tf = EntityManager.get_fragment(building_entity, TransformFragment, i);
        if (!building_tf) continue;

        const bp = building_tf.position;
        const bs = building_tf.scale;

        // Choose a random face: 0 = +z front, 1 = -z back, 2 = -x left, 3 = +x right
        const face = Math.floor(Math.random() * 4);

        // Sign parameters
        const sign_depth = 0.25;
        const sign_height = 2.0;

        let sign_pos;
        let sign_scale;

        const vertical_offset = 0.2 + Math.random() * 0.6; // between 20 % and 80 % up the wall

        switch (face) {
          case 0: // +z (front)
            sign_scale = [bs[0] * 0.8, sign_height, sign_depth];
            sign_pos = [
              bp[0],
              bp[1] + bs[1] * vertical_offset,
              bp[2] + bs[2] + sign_depth * 0.5 + 0.05,
            ];
            break;
          case 1: // -z (back)
            sign_scale = [bs[0] * 0.8, sign_height, sign_depth];
            sign_pos = [
              bp[0],
              bp[1] + bs[1] * vertical_offset,
              bp[2] - bs[2] - sign_depth * 0.5 - 0.05,
            ];
            break;
          case 2: // -x (left)
            sign_scale = [sign_depth, sign_height, bs[2] * 0.8];
            sign_pos = [
              bp[0] - bs[0] - sign_depth * 0.5 - 0.05,
              bp[1] + bs[1] * vertical_offset,
              bp[2],
            ];
            break;
          case 3: // +x (right)
          default:
            sign_scale = [sign_depth, sign_height, bs[2] * 0.8];
            sign_pos = [
              bp[0] + bs[0] + sign_depth * 0.5 + 0.05,
              bp[1] + bs[1] * vertical_offset,
              bp[2],
            ];
            break;
        }

        neon_sign_transforms.push({ position: sign_pos, scale: sign_scale });
        neon_sign_material_indices.push(Math.floor(Math.random() * neon_material_ids.length));
      }
    }

    // Spawn instanced entities grouped by material for efficiency.
    if (neon_sign_transforms.length > 0) {
      // Group transforms by selected material.
      const grouped = new Map();
      neon_sign_transforms.forEach((t, idx) => {
        const mat_idx = neon_sign_material_indices[idx];
        if (!grouped.has(mat_idx)) grouped.set(mat_idx, []);
        grouped.get(mat_idx).push(t);
      });

      // Create one entity per material.
      grouped.forEach((transforms, mat_idx) => {
        const neon_entity = spawn_mesh_entity(
          [0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 0],
          cube_mesh,
          neon_material_ids[mat_idx]
        );

        EntityManager.set_entity_instance_count(neon_entity, transforms.length);

        transforms.forEach((tr, i) => {
          const tf = EntityManager.get_fragment(neon_entity, TransformFragment, i);
          tf.position = tr.position;
          tf.scale = tr.scale;
        });

        this.entities.push(neon_entity);
      });
    }

    // ------------------------------------------------------------------
    // Scatter 100 point-lights around the generated city.
    // Each light is attached to a random building, either on a façade or
    // on the rooftop, with some colour variation for extra visual flavour.
    // ------------------------------------------------------------------
    const point_light_colours = [
      [1.0, 0.8, 0.6], // warm
      [0.6, 0.8, 1.0], // cool
      [1.0, 1.0, 0.8], // neutral
    ];

    // Create a uniform grid of point lights with better distribution
    const light_grid_size = Math.ceil(Math.sqrt(num_point_lights));
    const grid_spacing = 300.0;
    const grid_offset = (light_grid_size - 1) * grid_spacing * 0.5;

    for (let i = 0; i < num_point_lights; i++) {
      const p_view = EntityManager.get_fragment(point_light_entity, LightFragment, i);

      // Configure each light instance
      p_view.type = LightType.POINT;
      p_view.active = true;
      p_view.shadow_casting = false;

      // Calculate grid position for uniform distribution
      const grid_x = i % light_grid_size;
      const grid_z = Math.floor(i / light_grid_size);

      // Position lights in a centered grid pattern
      const x = grid_x * grid_spacing - grid_offset;
      const z = grid_z * grid_spacing - grid_offset;
      const y = 60.0 + Math.sin(i * 0.5) * 20.0; // Vary height slightly for visual interest

      p_view.position = [x, y, z];
      p_view.radius = 100.0;
      p_view.intensity = 15.0 + Math.sin(i * 0.3) * 1.0; // Vary intensity slightly
      p_view.color = point_light_colours[i % point_light_colours.length];
    }
    // ------------------------------------------------------------------------
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

    // Animate the swaying balls
    for (let i = 0; i < this.swaying_balls.length; i++) {
      const entity = this.swaying_balls[i];
      const base_pos = this.swaying_ball_base_positions[i];

      const sway_amplitude = 1200;
      const sway_frequency = 0.15;
      const t = performance.now() * 0.001;
      const phase = i * Math.PI * 0.5; // Offset phase for variety

      const x = base_pos[0] + Math.sin(t * Math.PI * 2 * sway_frequency + phase) * sway_amplitude;
      const y = base_pos[1];
      const z = base_pos[2];

      const tf = EntityManager.get_fragment(entity, TransformFragment);
      if (tf) {
        tf.position = [x, y, z];
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GITestScene — Sundown samples
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
import { TextureChannel } from "../../engine/src/renderer/renderer_types.js";

export class GITestScene extends Scene {
  name = "GITestScene";
  entities = [];

  init(parent_context) {
    super.init(parent_context);

    // Mesh.cube() spans [-1, 1], so final dimensions are 2 * scale.
    // Keep Cornell boxes around 3.5m interior size (target: 3-4m).
    const cornell_box_size_m = 3.5;
    const room_size = cornell_box_size_m * 0.5;
    const wall_thickness = 0.05;
    const box_spacing = 12.0;
    const scene_y_offset = -3.5;
    const ambient_emissive = 0.0;

    // camera arcball
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Set the skydome for this scene.
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // camera
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0, 8 + scene_y_offset, 24];
    view_data.view_rotation = [0.0005166, 0.9986818, -0.027326133, 0.0188794];

    // directional light
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 0.5;
    light_fragment_view.position = [5, 5 + scene_y_offset, 5];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Load worn panel textures for metallic floor
    let floor_albedo = {
      name: "floor_albedo",
      paths: ["example/textures/rubber_floor/Diffuse.ktx2"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_albedo",
    };
    let floor_normal = {
      name: "floor_normal",
      paths: ["example/textures/rubber_floor/Normal.ktx2"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_normal",
    };
    let floor_arm_metallic = {
      name: "floor_metallic",
      paths: ["example/textures/rubber_floor/ARM.ktx2"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_metallic",
    };
    let floor_arm_roughness = {
      name: "floor_roughness",
      paths: ["example/textures/rubber_floor/ARM.ktx2"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_roughness",
    };
    let floor_arm_ao = {
      name: "floor_ao",
      paths: ["example/textures/rubber_floor/ARM.ktx2"],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: "floor_ao",
    };

    // Create metallic floor material
    const metallic_floor_material = StandardMaterial.create("testgym_metallic_floor_material");
    const metallic_floor_material_id = metallic_floor_material.material_id;
    metallic_floor_material.sample_albedo(floor_albedo);
    metallic_floor_material.sample_normal(floor_normal);
    metallic_floor_material.sample_ao(floor_arm_ao, TextureChannel.R);
    metallic_floor_material.sample_roughness(floor_arm_roughness, TextureChannel.G);
    metallic_floor_material.sample_metallic(floor_arm_metallic, TextureChannel.B);
    metallic_floor_material.set_tiling(250.0);

    // materials
    const wall_material = StandardMaterial.create("testgym_wall_material");
    const wall_material_id = wall_material.material_id;
    wall_material.set_albedo([1, 1, 1, 1]);
    wall_material.set_roughness(0.8);
    wall_material.set_metallic(0.01);

    const red_material = StandardMaterial.create("testgym_red_material");
    const red_material_id = red_material.material_id;
    red_material.set_albedo([1, 0.2, 0.2, 1]);
    red_material.set_metallic(0.01);
    red_material.set_roughness(0.8);

    const blue_material = StandardMaterial.create("testgym_blue_material");
    const blue_material_id = blue_material.material_id;
    blue_material.set_albedo([0.2, 0.2, 1, 1]);
    blue_material.set_metallic(0.01);
    blue_material.set_roughness(0.8);

    const gray_material = StandardMaterial.create("testgym_gray_material");
    const gray_material_id = gray_material.material_id;
    gray_material.set_albedo([0.5, 0.5, 0.5, 1]);
    gray_material.set_metallic(0.01);
    gray_material.set_roughness(0.8);

    // White emissive material for ceiling lights
    const emissive_white_material = StandardMaterial.create("testgym_emissive_white_material");
    const emissive_white_material_id = emissive_white_material.material_id;
    emissive_white_material.set_albedo([0.9, 0.9, 0.5, 1]);
    emissive_white_material.set_emission(100.0);
    emissive_white_material.set_metallic(0.01);
    emissive_white_material.set_roughness(0.9);

    // meshes
    const cube_mesh = Mesh.cube();
    const sphere_mesh = Mesh.sphere();

    // Create large metallic floor plane
    const floor_plane = spawn_mesh_entity(
      [0, -5, 0],
      [0, 0, 0, 1],
      [200, 1, 200],
      cube_mesh,
      metallic_floor_material_id
    );
    this.entities.push(floor_plane);

    // Cornell-box walls (tight box)
    {
      // floor top at y=0
      const floor = spawn_mesh_entity(
        [0, scene_y_offset - wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(floor);

      // ceiling bottom at y=room_size
      const ceiling = spawn_mesh_entity(
        [0, scene_y_offset + room_size * 2.0 + wall_thickness, 0.0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling);

      // White emissive rectangle on ceiling (light)
      const emissive_light = spawn_mesh_entity(
        [0, scene_y_offset + room_size * 2.0 - 0.1, 0.0],
        [0, 0, 0, 1],
        [0.5, 0.1, 0.5],
        cube_mesh,
        emissive_white_material_id
      );
      this.entities.push(emissive_light);

      // back wall inner surface at z=-room_size/2
      const back_wall = spawn_mesh_entity(
        [0, scene_y_offset + room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall);

      // left wall inner surface at x=-room_size/2
      const left_wall = spawn_mesh_entity(
        [-room_size - wall_thickness, scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        red_material_id
      );
      this.entities.push(left_wall);

      // right wall inner surface at x=+room_size/2
      const right_wall = spawn_mesh_entity(
        [room_size + wall_thickness, scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        blue_material_id
      );
      this.entities.push(right_wall);

      // blockout "buildings"
      const building_data = [
        {
          mesh: cube_mesh,
          position: [-0.9, scene_y_offset + 0.65, -0.9],
          scale: [0.45, 0.65, 0.45],
          material_id: gray_material_id,
        },
        {
          mesh: cube_mesh,
          position: [0.85, scene_y_offset + 0.95, -0.45],
          scale: [0.35, 0.95, 0.35],
          material_id: red_material_id,
        },
        {
          mesh: sphere_mesh,
          position: [-0.2, scene_y_offset + 0.35, 0.85],
          scale: [0.35, 0.35, 0.35],
          material_id: blue_material_id,
        },
      ];
      for (const item of building_data) {
        const b = spawn_mesh_entity(
          item.position,
          [0, 0, 0, 1],
          item.scale,
          item.mesh,
          item.material_id
        );
        this.entities.push(b);
      }
    }
    // Additional Cornell box with different wall colors at x = -box_spacing
    {
      const offset_x = -box_spacing;

      const left_wall_material_second = StandardMaterial.create("testgym_left_material_second");
      const left_wall_material_second_id = left_wall_material_second.material_id;
      left_wall_material_second.set_albedo([0, 1, 0, 1]);
      left_wall_material_second.set_emission(ambient_emissive);
      left_wall_material_second.set_metallic(0.9);

      const right_wall_material_second = StandardMaterial.create("testgym_right_material_second");
      const right_wall_material_second_id = right_wall_material_second.material_id;
      right_wall_material_second.set_albedo([1, 0, 1, 1]);
      right_wall_material_second.set_emission(ambient_emissive);
      right_wall_material_second.set_metallic(0.9);

      // spawn elements for second box
      const floor_second = spawn_mesh_entity(
        [offset_x, scene_y_offset - wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(floor_second);

      const ceiling_second = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size * 2.0 + wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling_second);

      // White emissive rectangle on ceiling (light)
      const emissive_light_second = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size * 2.0 - 0.1, 0.0],
        [0, 0, 0, 1],
        [0.5, 0.1, 0.5],
        cube_mesh,
        emissive_white_material_id
      );
      this.entities.push(emissive_light_second);

      const back_wall_second = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall_second);

      const left_wall_second = spawn_mesh_entity(
        [offset_x - (room_size + wall_thickness), scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        left_wall_material_second_id
      );
      this.entities.push(left_wall_second);

      const right_wall_second = spawn_mesh_entity(
        [offset_x + room_size + wall_thickness, scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        right_wall_material_second_id
      );
      this.entities.push(right_wall_second);

      const building_data_second = [
        {
          mesh: cube_mesh,
          position: [offset_x - 0.7, scene_y_offset + 0.55, -0.7],
          scale: [0.4, 0.55, 0.4],
          material_id: left_wall_material_second_id,
        },
        {
          mesh: sphere_mesh,
          position: [offset_x + 0.55, scene_y_offset + 0.3, 0.75],
          scale: [0.3, 0.3, 0.3],
          material_id: right_wall_material_second_id,
        },
      ];
      for (const item of building_data_second) {
        const b = spawn_mesh_entity(
          item.position,
          [0, 0, 0, 1],
          item.scale,
          item.mesh,
          item.material_id
        );
        this.entities.push(b);
      }
    }
    // Additional Cornell box with different wall colors at x = +box_spacing
    {
      const offset_x = box_spacing;

      const left_wall_material_third = StandardMaterial.create("testgym_left_material_third");
      const left_wall_material_third_id = left_wall_material_third.material_id;
      left_wall_material_third.set_albedo([1, 0.5, 0, 1]);
      left_wall_material_third.set_emission(ambient_emissive);
      left_wall_material_third.set_metallic(0.9);

      const right_wall_material_third = StandardMaterial.create("testgym_right_material_third");
      const right_wall_material_third_id = right_wall_material_third.material_id;
      right_wall_material_third.set_albedo([0.5, 0, 0.5, 0.3]);
      right_wall_material_third.set_emission(ambient_emissive);
      right_wall_material_third.set_metallic(0.9);

      const floor_third = spawn_mesh_entity(
        [offset_x, scene_y_offset - wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(floor_third);

      const ceiling_third = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size * 2.0 + wall_thickness, 0],
        [0, 0, 0, 1],
        [room_size, wall_thickness, room_size],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(ceiling_third);

      // White emissive rectangle on ceiling (light)
      const emissive_light_third = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size * 2.0 - 0.1, 0.0],
        [0, 0, 0, 1],
        [0.5, 0.1, 0.5],
        cube_mesh,
        emissive_white_material_id
      );
      this.entities.push(emissive_light_third);

      const back_wall_third = spawn_mesh_entity(
        [offset_x, scene_y_offset + room_size, -room_size - wall_thickness],
        [0, 0, 0, 1],
        [room_size, room_size, wall_thickness],
        cube_mesh,
        wall_material_id
      );
      this.entities.push(back_wall_third);

      const left_wall_third = spawn_mesh_entity(
        [offset_x - (room_size + wall_thickness), scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        left_wall_material_third_id
      );
      this.entities.push(left_wall_third);

      const right_wall_third = spawn_mesh_entity(
        [offset_x + room_size + wall_thickness, scene_y_offset + room_size, 0],
        [0, 0, 0, 1],
        [wall_thickness, room_size, room_size],
        cube_mesh,
        right_wall_material_third_id
      );
      this.entities.push(right_wall_third);

      const building_data_third = [
        {
          mesh: sphere_mesh,
          position: [offset_x - 0.6, scene_y_offset + 0.4, -0.6],
          scale: [0.4, 0.4, 0.4],
          material_id: left_wall_material_third_id,
        },
        {
          mesh: cube_mesh,
          position: [offset_x + 0.7, scene_y_offset + 0.45, 0.7],
          scale: [0.45, 0.45, 0.45],
          material_id: right_wall_material_third_id,
        },
      ];
      for (const item of building_data_third) {
        const b = spawn_mesh_entity(
          item.position,
          [0, 0, 0, 1],
          item.scale,
          item.mesh,
          item.material_id
        );
        this.entities.push(b);
      }
    }

    // Load and place the station behind the three Cornell boxes.
    let station_root = this.load_gltf_scene(
      "example/models/station/station.gltf",
      [0, scene_y_offset, -45],
      [0, 0, 0, 1],
      [1.5, 1.5, 1.5]
    );
    this.entities.push(station_root);
  }

  cleanup() {
    for (const e of this.entities) {
      delete_entity(e);
    }
    this.remove_layer(FreeformArcballControlProcessor);
    super.cleanup();
  }
}

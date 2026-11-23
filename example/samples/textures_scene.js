// ═══════════════════════════════════════════════════════════════════════════════
// TexturesScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { TransformFragment } from "../../engine/src/core/ecs/fragments/transform_fragment.js";
import { FreeformArcballControlProcessor } from "../../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { LightFragment } from "../../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../../engine/src/core/minimal.js";
import { StandardMaterial } from "../../engine/src/renderer/material.js";
import { Mesh } from "../../engine/src/renderer/mesh.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../../engine/src/core/shared_data.js";
import { spawn_mesh_entity, delete_entity } from "../../engine/src/core/ecs/entity_utils.js";
import { TextureChannel } from "../../engine/src/renderer/renderer_types.js";
import { quat } from "gl-matrix";
import { world_label } from "../../engine/src/ui/3d/common.js";

export class TexturesScene extends Scene {
  name = "TexturesScene";
  entities = [];
  swaying_cube_entity = null;
  swaying_cube_base_pos = [0, 10, 50];

  init(parent_context) {
    super.init(parent_context);

    // Set the skydome for this scene.
    SharedEnvironmentData.set_skydome("default_scene_skydome");

    // Add the freeform arcball control processor to the scene
    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.set_scene(this);

    // Reset view to a good position for the BVH scene
    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [39.198, 14.0851, 78.60858];
    view_data.view_rotation = [-0.0203683, 0.9771718, -0.179953, -0.110603];
    view_data.far = 10000.0;

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    // Add a light fragment to the light entity
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1, 1];
    light_fragment_view.intensity = 7;
    light_fragment_view.position = [50, 20, 30];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Load metal plane material
    {
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
      let floor_roughness = {
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

      // Create a default material
      const default_plane_material = StandardMaterial.create("TexturesPlaneMaterial");
      this.default_plane_material_id = default_plane_material.material_id;
      default_plane_material.sample_albedo(floor_albedo);
      default_plane_material.sample_normal(floor_normal);
      default_plane_material.sample_roughness(floor_roughness, TextureChannel.G);
      default_plane_material.set_metallic(0.2);
      default_plane_material.set_emission(0.1);
      default_plane_material.set_tiling(150.0);
    }

    // Load wall material
    {
      let wall_albedo = {
        name: "wall_albedo",
        paths: ["example/textures/wall/wall_albedo.ktx2"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_albedo",
      };
      let wall_normal = {
        name: "wall_normal",
        paths: ["example/textures/wall/wall_normal.ktx2"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_normal",
      };
      let wall_roughness = {
        name: "wall_roughness",
        paths: ["example/textures/wall/wall_roughness.ktx2"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_roughness",
      };
      let wall_ao = {
        name: "wall_ao",
        paths: ["example/textures/wall/wall_ao.ktx2"],
        name: "wall_ao",
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "wall_ao",
      };

      // Create a default material
      const wall_material = StandardMaterial.create("TexturesWallMaterial");
      this.wall_material_id = wall_material.material_id;
      wall_material.sample_albedo(wall_albedo);
      wall_material.sample_normal(wall_normal);
      wall_material.sample_roughness(wall_roughness);
      wall_material.sample_ao(wall_ao);
      wall_material.set_metallic(0.9);
      wall_material.set_emission(0.1);
      wall_material.set_tiling(2.0);
    }

    // Create a sphere mesh
    this.sphere_mesh = Mesh.sphere();

    // Create a cube mesh
    this.cube_mesh = Mesh.cube();

    // Setup the world plane
    this.setup_world_plane();

    // Setup sphere entity
    this.setup_sphere_entity();

    // Setup the barrels
    this.setup_barrels();
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

    world_label("Textures Test Scene", [0, 20, 0], {
      width: 40.0,
      height: 10.0,
      billboard: false,
    });

    // Animate the swaying cube
    if (this.swaying_cube_entity) {
      const sway_amplitude = 12; // units left/right
      const sway_frequency = 0.25; // Hz
      const t = performance.now() * 0.001; // seconds
      const x =
        this.swaying_cube_base_pos[0] + Math.sin(t * Math.PI * 2 * sway_frequency) * sway_amplitude;
      const y = this.swaying_cube_base_pos[1];
      const z = this.swaying_cube_base_pos[2];
      const tf = EntityManager.get_fragment(this.swaying_cube_entity, TransformFragment);
      if (tf) {
        tf.position = [x, y, z];
      }
    }
  }

  setup_world_plane() {
    // Create a plane entity
    const plane = spawn_mesh_entity(
      [0, 0, 0],
      quat.fromEuler(quat.create(), 0.0, 0, 0),
      [1000, 5.0, 1000],
      this.cube_mesh,
      this.default_plane_material_id
    );
    this.entities.push(plane);
  }

  setup_sphere_entity() {
    // Create a sphere entity
    const sphere = spawn_mesh_entity(
      [0, 10, 10],
      quat.fromEuler(quat.create(), 0, 0, 0),
      [5, 5, 5],
      this.sphere_mesh,
      this.wall_material_id
    );
    this.entities.push(sphere);
  }

  setup_barrels() {
    const barrel_mesh = Mesh.from_gltf("example/models/barrel/Barrel.gltf");

    const num_barrels = 500;
    const barrel_spawn_range = 1000; // spread barrels across the entire ground plane
    const barrel_entity = spawn_mesh_entity(
      [0, 0, 0],
      [0, 0, 0, 1],
      [1.0, 1.0, 1.0],
      barrel_mesh,
      0 // GLTF sets the material id
    );
    this.entities.push(barrel_entity);

    // Set the instance count
    EntityManager.set_entity_instance_count(barrel_entity, num_barrels);

    // Set the instance data
    for (let i = 0; i < num_barrels; i++) {
      const x = (Math.random() - 0.5) * barrel_spawn_range;
      const z = (Math.random() - 0.5) * barrel_spawn_range;
      const y = 5.0; // Place on top of the plane
      const scale = 1.0 + Math.random() * 10.0;
      const view = EntityManager.get_fragment(barrel_entity, TransformFragment, i);
      view.position = [x, y, z];
      view.scale = [scale, scale, scale];
      view.rotation = quat.fromEuler(quat.create(), 0, Math.random() * 360, 0);
    }
  }
}

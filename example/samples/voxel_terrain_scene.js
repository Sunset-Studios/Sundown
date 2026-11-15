// ═══════════════════════════════════════════════════════════════════════════════
// VoxelTerrainScene — Sundown samples
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
import { log } from "../../engine/src/utility/logging.js";
import { quat } from "gl-matrix";

export class VoxelTerrainScene extends Scene {
  name = "VoxelTerrainScene";
  entities = [];
  terrain_material_id = null;
  cube_mesh = null;

  init(parent_context) {
    super.init(parent_context);

    const freeform_arcball = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [20.7373, 54.0735, 68.58896];
    view_data.view_rotation = [-0.036352589, 0.94788336, -0.25605953, -0.13457019];

    // Create a light and add it to the scene
    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);
    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1, 1, 1];
    light_fragment_view.intensity = 3.0;
    light_fragment_view.position = [-15, 20, 5];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    // Create terrain material
    const terrain_material = StandardMaterial.create("TerrainMaterial");
    this.terrain_material_id = terrain_material.material_id;

    {
      let dirt_albedo = {
        name: "dirt_albedo",
        paths: ["engine/textures/voxel/dirt_albedo.ktx2"],
        format: "rgba8unorm",
        dimension: "2d",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        material_notifier: "dirt_albedo",
      };

      terrain_material.sample_albedo(dirt_albedo);
      terrain_material.set_roughness(0.8);
      terrain_material.set_emission(0.0);
      terrain_material.set_metallic(0.1);
      terrain_material.set_specular(0.5);
    }

    // Create cube mesh for voxels
    this.cube_mesh = Mesh.cube();

    // Terrain parameters - Perlin-based fractal noise
    // Modified: Much more dramatic height variation for mountainous terrain
    const grid_width = 300;
    const grid_depth = 300;
    const block_size = 1.0;
    const base_frequency = 0.03;
    const height_scale = 25.0;
    const height_offset = 5.0;
    const octaves = 6;
    const persistence = 0.55;

    // Build permutation table for Perlin noise
    const perlin_perm = new Array(512);
    {
      const p = new Array(256);
      for (let i = 0; i < 256; i++) p[i] = i;
      for (let i = 255; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [p[i], p[j]] = [p[j], p[i]];
      }
      for (let i = 0; i < 512; i++) perlin_perm[i] = p[i & 255];
    }

    const fade_function = (t) => t * t * t * (t * (t * 6 - 15) + 10);
    const lerp_function = (a, b, t) => a + t * (b - a);
    const grad_function = (hash, x, y) => {
      switch (hash & 3) {
        case 0:
          return x + y;
        case 1:
          return -x + y;
        case 2:
          return x - y;
        case 3:
          return -x - y;
      }
    };

    function perlin_noise(x, y) {
      const xi = Math.floor(x) & 255;
      const yi = Math.floor(y) & 255;
      const xf = x - Math.floor(x);
      const yf = y - Math.floor(y);
      const u = fade_function(xf);
      const v = fade_function(yf);

      const aa = perlin_perm[xi + perlin_perm[yi]];
      const ab = perlin_perm[xi + perlin_perm[yi + 1]];
      const ba = perlin_perm[xi + 1 + perlin_perm[yi]];
      const bb = perlin_perm[xi + 1 + perlin_perm[yi + 1]];

      const x1 = lerp_function(grad_function(aa, xf, yf), grad_function(ba, xf - 1, yf), u);
      const x2 = lerp_function(grad_function(ab, xf, yf - 1), grad_function(bb, xf - 1, yf - 1), u);
      return lerp_function(x1, x2, v);
    }

    function fractal_noise(x, y) {
      let total = 0;
      let freq = base_frequency;
      let amp = 1;
      let max = 0;
      for (let o = 0; o < octaves; o++) {
        total += perlin_noise(x * freq, y * freq) * amp;
        max += amp;
        amp *= persistence;
        freq *= 2;
      }
      return total / max;
    }

    // Precompute heights
    const heights = new Array(grid_width);
    let total_blocks = 0;
    for (let xi = 0; xi < grid_width; xi++) {
      heights[xi] = new Array(grid_depth);
      for (let zi = 0; zi < grid_depth; zi++) {
        const noise_val = fractal_noise(xi, zi) * height_scale + height_offset;
        const height = Math.floor(noise_val);
        heights[xi][zi] = height;
        total_blocks += height;
      }
    }

    // Spawn instanced cube entity for terrain
    const terrain_entity = spawn_mesh_entity(
      [0, 0, 0],
      quat.fromEuler(quat.create(), 0, 0, 0),
      [block_size, block_size, block_size],
      this.cube_mesh,
      this.terrain_material_id
    );
    EntityManager.set_entity_instance_count(terrain_entity, total_blocks);
    this.entities.push(terrain_entity);

    // Assign transforms for each voxel
    let block_index = 0;
    for (let xi = 0; xi < grid_width; xi++) {
      for (let zi = 0; zi < grid_depth; zi++) {
        for (let yi = 0; yi < heights[xi][zi]; yi++) {
          if (block_index >= total_blocks) {
            break;
          }

          const pos = [
            (xi - grid_width / 2) * block_size * 2.0,
            yi * block_size * 2.0,
            (zi - grid_depth / 2) * block_size * 2.0,
          ];
          const view = EntityManager.get_fragment(terrain_entity, TransformFragment, block_index);
          view.position = pos;
          block_index++;
        }
      }
    }

    // Create sandy ground plane material
    const sandy_material = StandardMaterial.create("SandyGroundMaterial");
    sandy_material.set_albedo([0.94, 0.87, 0.69, 1.0]); // Sandy beige color
    sandy_material.set_roughness(0.9);
    sandy_material.set_emission(0.0);
    sandy_material.set_metallic(0.1);
    sandy_material.set_specular(0.5);

    // Create large ground plane entity
    const plane_entity = spawn_mesh_entity(
      [0, -10, 0], // Position slightly below terrain base
      quat.fromEuler(quat.create(), 0, 0, 0), // Rotate to be horizontal
      [800, 5, 800], // Large scale to cover the terrain area
      Mesh.cube(),
      sandy_material.material_id
    );
    this.entities.push(plane_entity);

    log(`[${this.name}] Initialized with ${total_blocks} blocks.`);
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
  }
}

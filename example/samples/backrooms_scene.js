// ═══════════════════════════════════════════════════════════════════════════════
// BackroomsScene — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { EntityManager } from "../../engine/src/core/ecs/entity.js";
import { Scene } from "../../engine/src/core/scene.js";
import { PostProcessStack } from "../../engine/src/renderer/post_process/post_process_stack.js";
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

export class BackroomsScene extends Scene {
  name = "BackroomsScene";
  entities = [];
  geometry_batches = new Map();
  cube_mesh = null;
  flicker_material = null;

  init(parent_context) {
    super.init(parent_context);

    this.entities.length = 0;
    this.geometry_batches.clear();

    const camera_control = this.add_layer(FreeformArcballControlProcessor);
    camera_control.move_speed = 7.0;
    camera_control.min_move_speed = 2.0;
    camera_control.max_move_speed = 24.0;
    camera_control.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0.0, 1.72, 6.5];
    view_data.view_rotation = quat.fromEuler(quat.create(), 0, 180, 0);
    view_data.near = 0.05;
    view_data.far = 180.0;

    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 1.0, 1.0]; // Warm daylight tint
    light_fragment_view.intensity = 0.0;
    light_fragment_view.position = [0.0, 5.0, -10.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    this.cube_mesh = Mesh.cube();
    const materials = this.create_materials();

    this.create_room_network(materials);
    this.create_liminal_details(materials);
    this.flush_geometry_batches();

    const geometry_count = [...this.geometry_batches.values()].reduce(
      (count, batch) => count + batch.length,
      0
    );
    log(
      `[${this.name}] Initialized ${geometry_count} pieces in ` +
        `${this.geometry_batches.size} material batches.`
    );
  }

  create_texture_config(name, path) {
    return {
      name,
      paths: [path],
      format: "rgba8unorm",
      dimension: "2d",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      material_notifier: name,
    };
  }

  create_materials() {
    const wallpaper = StandardMaterial.create("backrooms_wallpaper");
    wallpaper.sample_albedo(
      this.create_texture_config(
        "backrooms_wallpaper_albedo",
        "example/textures/backrooms/wallpaper_albedo.ktx2"
      )
    );
    wallpaper.set_roughness(0.88);
    wallpaper.set_metallic(0.0);
    wallpaper.set_specular(0.12);
    wallpaper.set_ao(0.82);
    wallpaper.set_tiling(1.0);

    const carpet = StandardMaterial.create("backrooms_damp_carpet");
    carpet.sample_albedo(
      this.create_texture_config(
        "backrooms_carpet_albedo",
        "example/textures/backrooms/carpet_albedo.ktx2"
      )
    );
    carpet.set_roughness(0.98);
    carpet.set_metallic(0.0);
    carpet.set_specular(0.04);
    carpet.set_ao(0.72);
    carpet.set_tiling(3.0);

    const ceiling = StandardMaterial.create("backrooms_acoustic_ceiling");
    ceiling.sample_albedo(
      this.create_texture_config(
        "backrooms_ceiling_albedo",
        "example/textures/backrooms/ceiling_tile_albedo.ktx2"
      )
    );
    ceiling.set_roughness(0.96);
    ceiling.set_metallic(0.0);
    ceiling.set_specular(0.05);
    ceiling.set_tiling(4.0);

    const baseboard = StandardMaterial.create("backrooms_baseboard");
    baseboard.set_albedo([0.285, 0.25, 0.11, 1.0]);
    baseboard.set_roughness(0.76);
    baseboard.set_metallic(0.02);

    const ceiling_grid = StandardMaterial.create("backrooms_ceiling_grid");
    ceiling_grid.set_albedo([0.47, 0.43, 0.25, 1.0]);
    ceiling_grid.set_roughness(0.64);
    ceiling_grid.set_metallic(0.18);

    const fluorescent = StandardMaterial.create("backrooms_fluorescent");
    fluorescent.set_albedo([1.0, 0.96, 0.68, 1.0]);
    fluorescent.set_emission(38.0);
    fluorescent.set_roughness(0.94);
    fluorescent.set_metallic(0.0);

    this.flicker_material = StandardMaterial.create("backrooms_flickering_fluorescent");
    this.flicker_material.set_albedo([0.92, 0.93, 0.64, 1.0]);
    this.flicker_material.set_emission(34.0);
    this.flicker_material.set_roughness(0.95);
    this.flicker_material.set_metallic(0.0);

    const dark_void = StandardMaterial.create("backrooms_unlit_void");
    dark_void.set_albedo([0.006, 0.007, 0.004, 1.0]);
    dark_void.set_roughness(0.99);
    dark_void.set_metallic(0.0);

    const vinyl = StandardMaterial.create("backrooms_aged_vinyl");
    vinyl.set_albedo([0.42, 0.255, 0.055, 1.0]);
    vinyl.set_roughness(0.58);
    vinyl.set_metallic(0.0);
    vinyl.set_specular(0.26);

    const metal = StandardMaterial.create("backrooms_tarnished_metal");
    metal.set_albedo([0.24, 0.235, 0.19, 1.0]);
    metal.set_roughness(0.62);
    metal.set_metallic(0.74);

    const plastic = StandardMaterial.create("backrooms_brown_plastic");
    plastic.set_albedo([0.105, 0.075, 0.035, 1.0]);
    plastic.set_roughness(0.72);
    plastic.set_metallic(0.0);

    const screen = StandardMaterial.create("backrooms_crt_screen");
    screen.set_albedo([0.055, 0.17, 0.09, 1.0]);
    screen.set_emission(4.2);
    screen.set_roughness(0.2);
    screen.set_metallic(0.05);

    const clock_face = StandardMaterial.create("backrooms_clock_face");
    clock_face.set_albedo([0.76, 0.71, 0.49, 1.0]);
    clock_face.set_roughness(0.86);
    clock_face.set_metallic(0.0);

    const damp_stain = StandardMaterial.create("backrooms_damp_stain");
    damp_stain.set_albedo([0.22, 0.17, 0.055, 1.0]);
    damp_stain.set_roughness(0.9);
    damp_stain.set_metallic(0.0);
    damp_stain.set_specular(0.12);

    const puddle = StandardMaterial.create("backrooms_standing_water");
    puddle.set_albedo([0.11, 0.095, 0.035, 1.0]);
    puddle.set_roughness(0.28);
    puddle.set_metallic(0.0);
    puddle.set_specular(0.72);

    const exit_sign = StandardMaterial.create("backrooms_exit_sign");
    exit_sign.set_albedo([0.09, 0.34, 0.12, 1.0]);
    exit_sign.set_emission(5.5);
    exit_sign.set_roughness(0.7);
    exit_sign.set_metallic(0.0);

    const false_daylight = StandardMaterial.create("backrooms_false_daylight");
    false_daylight.set_albedo([0.41, 0.65, 0.63, 1.0]);
    false_daylight.set_emission(2.4);
    false_daylight.set_roughness(0.88);
    false_daylight.set_metallic(0.0);

    return {
      wallpaper: wallpaper.material_id,
      carpet: carpet.material_id,
      ceiling: ceiling.material_id,
      baseboard: baseboard.material_id,
      ceiling_grid: ceiling_grid.material_id,
      fluorescent: fluorescent.material_id,
      flicker: this.flicker_material.material_id,
      dark_void: dark_void.material_id,
      vinyl: vinyl.material_id,
      metal: metal.material_id,
      plastic: plastic.material_id,
      screen: screen.material_id,
      clock_face: clock_face.material_id,
      damp_stain: damp_stain.material_id,
      puddle: puddle.material_id,
      exit_sign: exit_sign.material_id,
      false_daylight: false_daylight.material_id,
    };
  }

  queue_box(position, scale, material_id, rotation = [0, 0, 0, 1]) {
    let batch = this.geometry_batches.get(material_id);
    if (!batch) {
      batch = [];
      this.geometry_batches.set(material_id, batch);
    }
    batch.push({ position, rotation, scale });
  }

  flush_geometry_batches() {
    for (const [material_id, transforms] of this.geometry_batches) {
      if (transforms.length === 0) continue;

      const first = transforms[0];
      const entity = spawn_mesh_entity(
        first.position,
        first.rotation,
        first.scale,
        this.cube_mesh,
        material_id
      );
      EntityManager.set_entity_instance_count(entity, transforms.length);

      for (let i = 0; i < transforms.length; i++) {
        const transform = transforms[i];
        const transform_view = EntityManager.get_fragment(entity, TransformFragment, i);
        transform_view.position = transform.position;
        transform_view.rotation = transform.rotation;
        transform_view.scale = transform.scale;
      }

      this.entities.push(entity);
    }
  }

  create_room_network(materials) {
    const rooms = [
      {
        center: [0, 0],
        size: [22, 18],
        height: 7.2,
        openings: {
          north: [{ center: 0, width: 5 }],
          east: [{ center: 0, width: 5 }],
          west: [{ center: 0, width: 4.4 }],
        },
        fixtures: [
          [-5, 0, "lit"],
          [5, 0, "flicker"],
          [0, -6, "lit"],
        ],
      },
      {
        center: [19, 0],
        size: [16, 6],
        height: 6.6,
        sides: { west: false },
        openings: { east: [{ center: 0, width: 5 }] },
        fixtures: [
          [15, 0, "lit"],
          [23, 0, "dead"],
        ],
      },
      {
        center: [39, 0],
        size: [24, 22],
        height: 8.4,
        sides: { west: false },
        openings: { north: [{ center: 44, width: 6 }] },
        fixtures: [
          [32, -6, "lit"],
          [44, -6, "flicker"],
          [32, 6, "lit"],
          [44, 6, "dead"],
        ],
      },
      {
        center: [0, -18],
        size: [6, 18],
        height: 7.0,
        sides: { south: false },
        openings: { north: [{ center: 0, width: 5 }] },
        fixtures: [
          [0, -14, "lit"],
          [0, -23, "flicker"],
        ],
      },
      {
        center: [0, -36],
        size: [28, 18],
        height: 5.8,
        sides: { south: false },
        openings: {
          west: [{ center: -36, width: 6 }],
          east: [{ center: -36, width: 6 }],
        },
        fixtures: [
          [-8, -32, "lit"],
          [8, -32, "lit"],
          [-8, -41, "dead"],
          [8, -41, "flicker"],
        ],
      },
      {
        center: [-25, -36],
        size: [22, 14],
        height: 7.0,
        sides: { east: false },
        fixtures: [
          [-30, -36, "lit"],
          [-20, -36, "dead"],
        ],
      },
      {
        center: [24, -36],
        size: [20, 6],
        height: 4.8,
        sides: { west: false },
        openings: { east: [{ center: -36, width: 6 }] },
        fixtures: [
          [19, -36, "flicker"],
          [29, -36, "lit"],
        ],
      },
      {
        center: [44, -36],
        size: [20, 22],
        height: 9.0,
        sides: { west: false },
        openings: { south: [{ center: 44, width: 6 }] },
        fixtures: [
          [39, -31, "lit"],
          [49, -31, "dead"],
          [39, -42, "flicker"],
          [49, -42, "lit"],
        ],
      },
      {
        center: [44, -18],
        size: [6, 14],
        height: 6.0,
        sides: { north: false, south: false },
        fixtures: [
          [44, -15, "lit"],
          [44, -22, "dead"],
        ],
      },
      {
        center: [-22, 0],
        size: [22, 14],
        height: 6.5,
        sides: { east: false },
        fixtures: [
          [-27, 0, "flicker"],
          [-17, 0, "dead"],
        ],
      },
    ];

    for (const room of rooms) {
      this.add_room(room, materials);
    }

    const portals = [
      ["x", 11, 0, 5, 7.2, 5.35],
      ["x", 27, 0, 5, 8.4, 5.1],
      ["z", -9, 0, 5, 7.2, 5.35],
      ["z", -27, 0, 5, 7.0, 4.8],
      ["x", -14, -36, 6, 7.0, 4.75],
      ["x", 14, -36, 6, 5.8, 4.1],
      ["x", 34, -36, 6, 9.0, 4.0],
      ["z", -25, 44, 6, 9.0, 4.9],
      ["z", -11, 44, 6, 8.4, 4.9],
      ["x", -11, 0, 4.4, 7.2, 5.0],
    ];

    for (const portal of portals) {
      this.add_portal(...portal, materials);
    }
  }

  add_room(room, materials) {
    const [center_x, center_z] = room.center;
    const [size_x, size_z] = room.size;
    const half_x = size_x * 0.5;
    const half_z = size_z * 0.5;
    const min_x = center_x - half_x;
    const max_x = center_x + half_x;
    const min_z = center_z - half_z;
    const max_z = center_z + half_z;
    const sides = room.sides ?? {};
    const openings = room.openings ?? {};

    this.queue_box([center_x, -0.15, center_z], [half_x, 0.15, half_z], materials.carpet);
    this.queue_box(
      [center_x, room.height + 0.15, center_z],
      [half_x, 0.15, half_z],
      materials.ceiling
    );

    if (sides.north !== false) {
      this.add_boundary_wall("z", min_z, min_x, max_x, openings.north, room.height, materials);
    }
    if (sides.south !== false) {
      this.add_boundary_wall("z", max_z, min_x, max_x, openings.south, room.height, materials);
    }
    if (sides.west !== false) {
      this.add_boundary_wall("x", min_x, min_z, max_z, openings.west, room.height, materials);
    }
    if (sides.east !== false) {
      this.add_boundary_wall("x", max_x, min_z, max_z, openings.east, room.height, materials);
    }

    const grid_spacing = 4.0;
    for (let x = Math.ceil(min_x / grid_spacing) * grid_spacing; x < max_x; x += grid_spacing) {
      this.queue_box(
        [x, room.height - 0.025, center_z],
        [0.026, 0.025, Math.max(0.1, half_z - 0.08)],
        materials.ceiling_grid
      );
    }
    for (let z = Math.ceil(min_z / grid_spacing) * grid_spacing; z < max_z; z += grid_spacing) {
      this.queue_box(
        [center_x, room.height - 0.025, z],
        [Math.max(0.1, half_x - 0.08), 0.025, 0.026],
        materials.ceiling_grid
      );
    }

    for (const fixture of room.fixtures ?? []) {
      this.add_fluorescent_fixture(fixture[0], fixture[1], room.height, fixture[2], materials);
    }
  }

  add_boundary_wall(axis, fixed, start, end, openings = [], height, materials) {
    const sorted_openings = [...openings].sort((a, b) => a.center - b.center);
    let cursor = start;

    for (const opening of sorted_openings) {
      const opening_start = Math.max(start, opening.center - opening.width * 0.5);
      const opening_end = Math.min(end, opening.center + opening.width * 0.5);
      if (opening_start > cursor) {
        this.add_wall_segment(axis, fixed, cursor, opening_start, height, materials);
      }
      cursor = Math.max(cursor, opening_end);
    }

    if (cursor < end) {
      this.add_wall_segment(axis, fixed, cursor, end, height, materials);
    }
  }

  add_wall_segment(axis, fixed, start, end, height, materials) {
    const length = end - start;
    if (length <= 0.02) return;

    const center = (start + end) * 0.5;
    if (axis === "x") {
      this.queue_box(
        [fixed, height * 0.5, center],
        [0.12, height * 0.5, length * 0.5],
        materials.wallpaper
      );
      this.queue_box([fixed, 0.13, center], [0.16, 0.13, length * 0.5], materials.baseboard);
    } else {
      this.queue_box(
        [center, height * 0.5, fixed],
        [length * 0.5, height * 0.5, 0.12],
        materials.wallpaper
      );
      this.queue_box([center, 0.13, fixed], [length * 0.5, 0.13, 0.16], materials.baseboard);
    }
  }

  add_portal(axis, fixed, center, width, height, door_height, materials) {
    const header_height = Math.max(0.12, height - door_height);
    if (axis === "x") {
      this.queue_box(
        [fixed, door_height + header_height * 0.5, center],
        [0.14, header_height * 0.5, width * 0.5],
        materials.wallpaper
      );
      this.queue_box(
        [fixed, door_height * 0.5, center - width * 0.5],
        [0.18, door_height * 0.5, 0.1],
        materials.baseboard
      );
      this.queue_box(
        [fixed, door_height * 0.5, center + width * 0.5],
        [0.18, door_height * 0.5, 0.1],
        materials.baseboard
      );
      this.queue_box([fixed, door_height, center], [0.18, 0.1, width * 0.5], materials.baseboard);
    } else {
      this.queue_box(
        [center, door_height + header_height * 0.5, fixed],
        [width * 0.5, header_height * 0.5, 0.14],
        materials.wallpaper
      );
      this.queue_box(
        [center - width * 0.5, door_height * 0.5, fixed],
        [0.1, door_height * 0.5, 0.18],
        materials.baseboard
      );
      this.queue_box(
        [center + width * 0.5, door_height * 0.5, fixed],
        [0.1, door_height * 0.5, 0.18],
        materials.baseboard
      );
      this.queue_box([center, door_height, fixed], [width * 0.5, 0.1, 0.18], materials.baseboard);
    }
  }

  add_fluorescent_fixture(x, z, height, fixture_type, materials) {
    this.queue_box([x, height - 0.045, z], [1.72, 0.045, 0.56], materials.ceiling_grid);

    let panel_material = materials.fluorescent;
    if (fixture_type === "flicker") panel_material = materials.flicker;
    if (fixture_type === "dead") panel_material = materials.dark_void;

    this.queue_box([x, height - 0.095, z], [1.52, 0.025, 0.42], panel_material);
  }

  create_liminal_details(materials) {
    // A wall that almost bisects the first room forces an immediate, inexplicable detour.
    this.add_wall_segment("x", -4.2, -5.2, 1.2, 5.65, materials);
    this.queue_box([-4.2, 5.9, -2.0], [0.55, 0.24, 3.35], materials.wallpaper);

    // A door-shaped absence is visible from the starting position, but has no frame or purpose.
    this.queue_box([6.4, 2.45, -8.86], [1.25, 2.45, 0.035], materials.dark_void);
    this.queue_box([6.4, 5.0, -8.82], [1.48, 0.11, 0.11], materials.baseboard);
    this.queue_box([5.04, 2.5, -8.82], [0.11, 2.5, 0.11], materials.baseboard);
    this.queue_box([7.76, 2.5, -8.82], [0.11, 2.5, 0.11], materials.baseboard);

    // Damp carpet and standing water break up the otherwise uniform floor response.
    this.queue_box(
      [4.45, 0.012, 3.15],
      [1.38, 0.012, 0.48],
      materials.damp_stain,
      quat.fromEuler(quat.create(), 0, 7, 0)
    );
    this.queue_box(
      [5.65, 0.013, 2.86],
      [0.72, 0.012, 0.32],
      materials.damp_stain,
      quat.fromEuler(quat.create(), 0, -14, 0)
    );
    this.queue_box(
      [4.45, 0.027, 3.15],
      [0.92, 0.008, 0.27],
      materials.puddle,
      quat.fromEuler(quat.create(), 0, 11, 0)
    );
    this.queue_box(
      [5.18, 0.028, 2.92],
      [0.56, 0.008, 0.2],
      materials.puddle,
      quat.fromEuler(quat.create(), 0, -17, 0)
    );
    this.queue_box(
      [-7.2, 0.012, 5.2],
      [1.2, 0.012, 0.45],
      materials.damp_stain,
      quat.fromEuler(quat.create(), 0, -11, 0)
    );

    // Repeated threshold frames make the south corridor feel longer than its footprint.
    for (let z = -11.5; z >= -25.5; z -= 3.5) {
      this.queue_box([-2.66, 2.75, z], [0.16, 2.75, 0.18], materials.baseboard);
      this.queue_box([2.66, 2.75, z], [0.16, 2.75, 0.18], materials.baseboard);
      this.queue_box([0, 5.38, z], [2.82, 0.16, 0.18], materials.baseboard);
    }

    // Several ceiling tiles are missing or hanging loose, revealing only featureless blackness.
    this.add_ceiling_void(-5.8, 4.4, 7.2, materials);
    this.add_ceiling_void(22.8, 0.0, 6.6, materials);
    this.add_ceiling_void(-8.0, -40.8, 5.8, materials);
    this.add_ceiling_void(39.0, -31.2, 9.0, materials);
    this.queue_box(
      [41.0, 7.85, -41.0],
      [1.75, 0.045, 0.58],
      materials.ceiling,
      quat.fromEuler(quat.create(), 8, 0, -5)
    );

    // The tall room uses a too-regular forest of columns, interrupted by one swollen column.
    for (let x = 32; x <= 46; x += 7) {
      for (let z = -6; z <= 6; z += 6) {
        const is_wrong_column = x === 39 && z === 0;
        const width = is_wrong_column ? 1.15 : 0.52;
        this.queue_box([x, 4.2, z], [width, 4.2, width], materials.wallpaper);
        this.queue_box([x, 0.14, z], [width + 0.1, 0.14, width + 0.1], materials.baseboard);
      }
    }
    this.queue_box([39, 7.82, 0], [12, 0.42, 0.3], materials.wallpaper);

    // Low cross-beams turn the short connector into a subtly compressed threshold.
    for (let x = 18; x <= 31; x += 3.25) {
      this.queue_box([x, 4.25, -36], [0.26, 0.55, 3.0], materials.wallpaper);
      this.queue_box([x, 3.72, -36], [0.31, 0.06, 3.0], materials.baseboard);
    }

    // Repeated empty institutional seating gives the western room a stranded waiting-area memory.
    this.add_chair(-31, -39.5, materials);
    this.add_chair(-27.2, -39.5, materials);
    this.add_chair(-23.4, -39.5, materials);
    this.add_chair(-19.6, -39.5, materials);
    this.add_wall_clock(-25.0, -42.78, materials);
    this.queue_box(
      [-29.5, 0.012, -35.15],
      [2.1, 0.012, 0.52],
      materials.damp_stain,
      quat.fromEuler(quat.create(), 0, 6, 0)
    );
    this.queue_box(
      [-27.4, 0.013, -35.45],
      [1.15, 0.012, 0.34],
      materials.damp_stain,
      quat.fromEuler(quat.create(), 0, -12, 0)
    );
    this.queue_box(
      [-30.2, 0.027, -35.0],
      [1.15, 0.008, 0.26],
      materials.puddle,
      quat.fromEuler(quat.create(), 0, -8, 0)
    );
    this.queue_box(
      [-28.85, 0.028, -35.25],
      [0.62, 0.008, 0.2],
      materials.puddle,
      quat.fromEuler(quat.create(), 0, 14, 0)
    );

    // Cubicle fragments stop short of enclosing anything useful.
    this.queue_box([-25.0, 1.45, -1.8], [0.09, 1.45, 4.0], materials.wallpaper);
    this.queue_box([-19.0, 1.45, 1.9], [3.0, 1.45, 0.09], materials.wallpaper);
    this.queue_box([-28.5, 1.45, 2.6], [2.4, 1.45, 0.09], materials.wallpaper);
    this.add_wall_phone(-30.2, -6.76, materials);

    // An interior "window" emits cold, immobile daylight into an otherwise sealed office.
    this.queue_box([-32.86, 3.25, 2.0], [0.035, 1.35, 2.1], materials.false_daylight);
    this.queue_box([-32.78, 3.25, -0.2], [0.11, 1.55, 0.12], materials.baseboard);
    this.queue_box([-32.78, 3.25, 4.2], [0.11, 1.55, 0.12], materials.baseboard);
    this.queue_box([-32.78, 1.82, 2.0], [0.11, 0.12, 2.32], materials.baseboard);
    this.queue_box([-32.78, 4.68, 2.0], [0.11, 0.12, 2.32], materials.baseboard);

    // A glowing EXIT marker points deeper into the building instead of toward an exit.
    this.queue_box([13.86, 3.9, -36.0], [0.035, 0.38, 0.92], materials.exit_sign);
    this.queue_box([13.8, 3.9, -36.0], [0.08, 0.48, 1.05], materials.plastic);
    this.queue_box([13.71, 3.9, -36.0], [0.035, 0.34, 0.86], materials.exit_sign);

    // A lone powered CRT faces the large final room, with no cable or apparent source.
    this.add_crt_cart(38.5, -43.0, materials);

    // The stairs climb into a sealed, black doorway that sits too high to be plausible.
    for (let i = 0; i < 8; i++) {
      const step_height = 0.3 + i * 0.34;
      this.queue_box(
        [48.5, step_height * 0.5, -38.7 - i * 0.88],
        [2.15, step_height * 0.5, 0.48],
        materials.carpet
      );
    }
    this.queue_box([48.5, 5.25, -46.76], [2.35, 2.25, 0.1], materials.dark_void);
    this.queue_box([48.5, 7.53, -46.72], [2.62, 0.12, 0.15], materials.baseboard);
    this.queue_box([46.05, 5.25, -46.72], [0.12, 2.4, 0.15], materials.baseboard);
    this.queue_box([50.95, 5.25, -46.72], [0.12, 2.4, 0.15], materials.baseboard);

    // Ceiling ducts in the service loop make the alternate route feel mechanically occupied.
    this.queue_box([42.4, 5.35, -18], [0.32, 0.32, 6.8], materials.metal);
    this.queue_box([45.6, 5.05, -18], [0.18, 0.18, 6.8], materials.metal);
    for (let z = -23; z <= -13; z += 2.5) {
      this.queue_box([44, 5.35, z], [1.62, 0.07, 0.07], materials.metal);
    }

    // A leaking service pipe leaves a dark trail beneath the final dead fixture.
    this.queue_box([46.1, 7.55, -28.4], [0.2, 0.2, 2.6], materials.metal);
    this.queue_box(
      [46.0, 0.012, -30.45],
      [0.82, 0.012, 1.15],
      materials.damp_stain,
      quat.fromEuler(quat.create(), 0, -8, 0)
    );
    this.queue_box(
      [46.45, 0.013, -29.15],
      [0.48, 0.012, 0.72],
      materials.damp_stain,
      quat.fromEuler(quat.create(), 0, 15, 0)
    );
    this.queue_box(
      [45.85, 0.027, -30.65],
      [0.42, 0.008, 0.78],
      materials.puddle,
      quat.fromEuler(quat.create(), 0, -9, 0)
    );
    this.queue_box(
      [46.35, 0.028, -29.75],
      [0.28, 0.008, 0.5],
      materials.puddle,
      quat.fromEuler(quat.create(), 0, 13, 0)
    );
  }

  add_ceiling_void(x, z, height, materials) {
    this.queue_box([x, height - 0.175, z], [1.74, 0.035, 0.58], materials.dark_void);
    this.queue_box([x - 1.76, height - 0.205, z], [0.035, 0.13, 0.62], materials.ceiling_grid);
    this.queue_box([x + 1.76, height - 0.205, z], [0.035, 0.13, 0.62], materials.ceiling_grid);
  }

  add_chair(x, z, materials) {
    this.queue_box([x, 0.88, z], [1.3, 0.18, 0.68], materials.vinyl);
    this.queue_box([x, 1.68, z - 0.56], [1.3, 0.72, 0.16], materials.vinyl);
    this.queue_box([x - 1.05, 0.42, z], [0.09, 0.42, 0.09], materials.metal);
    this.queue_box([x + 1.05, 0.42, z], [0.09, 0.42, 0.09], materials.metal);
    this.queue_box([x - 1.05, 0.78, z - 0.54], [0.09, 0.78, 0.09], materials.metal);
    this.queue_box([x + 1.05, 0.78, z - 0.54], [0.09, 0.78, 0.09], materials.metal);
  }

  add_wall_clock(x, z, materials) {
    this.queue_box([x, 4.35, z], [0.95, 0.68, 0.08], materials.plastic);
    this.queue_box([x, 4.35, z + 0.09], [0.82, 0.55, 0.025], materials.clock_face);
    this.queue_box([x, 4.42, z + 0.13], [0.035, 0.32, 0.025], materials.plastic);
    this.queue_box(
      [x + 0.15, 4.25, z + 0.13],
      [0.035, 0.27, 0.025],
      materials.plastic,
      quat.fromEuler(quat.create(), 0, 0, 54)
    );
  }

  add_wall_phone(x, z, materials) {
    this.queue_box([x, 2.35, z], [0.52, 0.82, 0.13], materials.plastic);
    this.queue_box([x - 0.46, 2.4, z + 0.16], [0.16, 0.72, 0.13], materials.vinyl);
    this.queue_box([x + 0.14, 2.15, z + 0.16], [0.2, 0.22, 0.08], materials.clock_face);
  }

  add_crt_cart(x, z, materials) {
    this.queue_box([x, 2.25, z], [1.45, 1.05, 0.9], materials.plastic);
    this.queue_box([x, 2.3, z + 0.92], [1.12, 0.76, 0.035], materials.screen);
    this.queue_box([x, 1.0, z], [1.65, 0.12, 1.05], materials.metal);
    this.queue_box([x - 1.35, 0.5, z - 0.72], [0.09, 0.5, 0.09], materials.metal);
    this.queue_box([x + 1.35, 0.5, z - 0.72], [0.09, 0.5, 0.09], materials.metal);
    this.queue_box([x - 1.35, 0.5, z + 0.72], [0.09, 0.5, 0.09], materials.metal);
    this.queue_box([x + 1.35, 0.5, z + 0.72], [0.09, 0.5, 0.09], materials.metal);
  }

  cleanup() {
    PostProcessStack.clear_view(0);

    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.entities.length = 0;
    this.geometry_batches.clear();
    this.flicker_material = null;
    this.cube_mesh = null;

    this.remove_layer(FreeformArcballControlProcessor);
    super.cleanup();
  }
}

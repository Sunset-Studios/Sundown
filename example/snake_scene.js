import { quat } from "gl-matrix";

import { Scene } from "../engine/src/core/scene.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { StaticMeshFragment } from "../engine/src/core/ecs/fragments/static_mesh_fragment.js";
import { TransformFragment } from "../engine/src/core/ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "../engine/src/core/ecs/fragments/visibility_fragment.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../engine/src/core/shared_data.js";
import { LightType } from "../engine/src/core/minimal.js";
import { MAX_CLIPMAP_LEVELS } from "../engine/src/renderer/shadows/shadow_utils.js";
import { spawn_mesh_entity, delete_entity } from "../engine/src/core/ecs/entity_utils.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { InputKey } from "../engine/src/input/input_types.js";
import { StandardMaterial } from "../engine/src/renderer/material.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { Renderer } from "../engine/src/renderer/renderer.js";
import { button, label, panel } from "../engine/src/ui/2d/immediate.js";

import {
  DIRECTIONS,
  advanceGame,
  createInitialState,
  queueDirection,
  togglePause,
} from "./snake_game.js";
import {
  advanceRippleBursts,
  createRippleBurst,
  getRippleScaleMultiplier,
} from "./snake_ripple.js";

const BOARD_SIZE = 14;
const CELL_SPACING = 2.35;
const FLOOR_Y = -0.9;
const TILE_Y = -0.25;
const PIECE_Y = 0.75;
const BOARD_HALF_EXTENT = ((BOARD_SIZE - 1) * CELL_SPACING) * 0.5;
const SEGMENT_SCALE = [0.6, 0.6, 0.6];
const HEAD_SCALE = [0.7, 0.7, 0.7];
const FOOD_SCALE = [0.5, 0.5, 0.5];
const TILE_SCALE = [0.95, 0.12, 0.95];
const WALL_THICKNESS = 0.45;
const WALL_HEIGHT = 0.7;
const FLOOR_SCALE = [BOARD_HALF_EXTENT + 2.5, 0.5, BOARD_HALF_EXTENT + 2.5];
const HIDDEN_POSITION = [0, -50, 0];
const CAMERA_FAR = 160.0;
const CAMERA_FRAME_PADDING = 2.5;
const CAMERA_FOV = Math.PI / 3.4;
const SNAKE_SPEED = 5;
const RIPPLE_PULSE_DURATION = 0.28;
const RIPPLE_SEGMENT_DELAY = 0.06;
const RIPPLE_AMPLITUDE = 0.28;
const RIPPLE_MAX_MULTIPLIER = 1.45;

const HUD_PANEL = {
  layout: "column",
  gap: 6,
  x: 40,
  y: 40,
  width: 240,
  anchor_x: "right",
  padding: 12,
  background_color: "rgba(8, 11, 16, 0.62)",
  border: "1px solid rgba(144, 156, 176, 0.26)",
  corner_radius: 12,
  box_shadow: "0 12 28 rgba(0, 0, 0, 0.18)",
};

const HUD_TITLE = {
  width: "100%",
  height: "fit-content",
  font: "700 22px monospace",
  text_color: "#f3f6fb",
};

const HUD_TEXT = {
  width: "100%",
  height: "fit-content",
  font: "14px monospace",
  text_color: "#cbd5e3",
};

const CONTROLS_PANEL = {
  width: 250,
  height: 150,
  x: 40,
  y: 40,
  anchor_x: "right",
  anchor_y: "bottom",
  padding: 10,
  background_color: "transparent",
};

const CONTROL_BUTTON = {
  width: 75,
  height: 45,
  font: "13px monospace",
  text_color: "#eef3fb",
  background_color: "rgba(26, 35, 47, 0.84)",
  border: "1px solid rgba(132, 148, 172, 0.34)",
  corner_radius: 9,
};

const ACTION_BUTTON = {
  width: 82,
  height: 36,
  font: "13px monospace",
  text_color: "#eef3fb",
  background_color: "rgba(26, 35, 47, 0.84)",
  border: "1px solid rgba(132, 148, 172, 0.34)",
  corner_radius: 9,
};

export class SnakeScene extends Scene {
  entities = [];
  boardTiles = [];
  segmentEntities = [];
  foodEntity = null;
  materials = {};
  state = null;
  rippleBursts = [];
  lastAspectRatio = 0;

  init() {
    super.init();

    this.configure_view();
    this.create_materials();
    this.create_light();
    this.create_board();
    this.create_snake_pool();
    this.create_food_entity();
    this.restart_game();
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }

    this.entities.length = 0;
    this.boardTiles.length = 0;
    this.segmentEntities.length = 0;
    this.foodEntity = null;
    this.rippleBursts.length = 0;

    super.cleanup();
  }

  configure_view() {
    SharedEnvironmentData.set_skydome("snake_skydome");
    SharedEnvironmentData.set_skybox_color([0.03, 0.04, 0.06, 1.0]);

    const view = SharedViewBuffer.get_view_data(this.context.current_view);
    view.view_position = [0, 36, 0.001];
    view.view_rotation = quat.fromEuler(quat.create(), 89.9, 180, 0);
    view.fov = CAMERA_FOV;
    view.near = 0.1;
    view.far = CAMERA_FAR;

    this.update_camera_framing(true);
  }

  update_camera_framing(force = false) {
    const aspectRatio = Renderer.get().aspect_ratio || 1;

    if (!force && Math.abs(aspectRatio - this.lastAspectRatio) < 0.001) {
      return;
    }

    const view = SharedViewBuffer.get_view_data(this.context.current_view);
    const boardHalfSpan = BOARD_HALF_EXTENT + CELL_SPACING * 0.5 + CAMERA_FRAME_PADDING;
    const tanHalfFov = Math.tan(CAMERA_FOV * 0.5);
    const verticalDistance = boardHalfSpan / tanHalfFov;
    const horizontalDistance = boardHalfSpan / (tanHalfFov * Math.max(aspectRatio, 0.001));
    const requiredDistance = Math.max(verticalDistance, horizontalDistance);

    view.view_position = [0, requiredDistance, 0.001];
    this.lastAspectRatio = aspectRatio;
  }

  create_materials() {
    this.materials.floor = StandardMaterial.create("snake_floor", {
      albedo: [0.06, 0.08, 0.12, 1.0],
      roughness: 0.98,
      metallic: 0.0,
      specular: 0.05,
    }).material_id;

    this.materials.tileDark = StandardMaterial.create("snake_tile_dark", {
      albedo: [0.14, 0.18, 0.24, 1.0],
      roughness: 0.95,
      metallic: 0.0,
      specular: 0.05,
    }).material_id;

    this.materials.tileLight = StandardMaterial.create("snake_tile_light", {
      albedo: [0.18, 0.23, 0.3, 1.0],
      roughness: 0.94,
      metallic: 0.0,
      specular: 0.06,
    }).material_id;

    this.materials.wall = StandardMaterial.create("snake_wall", {
      albedo: [0.3, 0.34, 0.42, 1.0],
      roughness: 0.1,
      metallic: 0.0,
      specular: 0.08,
    }).material_id;

    this.materials.segment = StandardMaterial.create("snake_segment", {
      albedo: [0.35, 0.78, 0.34, 1.0],
      roughness: 0.7,
      metallic: 0.0,
      specular: 0.08,
    }).material_id;

    this.materials.head = StandardMaterial.create("snake_head", {
      albedo: [0.77, 0.95, 0.43, 1.0],
      roughness: 0.6,
      metallic: 0.0,
      specular: 0.1,
    }).material_id;

    this.materials.food = StandardMaterial.create("snake_food", {
      albedo: [0.98, 0.41, 0.29, 1.0],
      roughness: 0.5,
      metallic: 0.0,
      emission: 25.0,
      specular: 0.1,
    }).material_id;
  }

  create_light() {
    const lightEntity = EntityManager.create_entity([LightFragment]);
    const light = EntityManager.get_fragment(lightEntity, LightFragment);

    light.type = LightType.DIRECTIONAL;
    light.color = [1.0, 0.97, 0.92, 1.0];
    light.intensity = 18.0;
    light.position = [40.0, 60.0, 20.0, 1.0];
    light.active = true;
    light.is_primary_sun = 1;
    light.shadow_clipmaps = MAX_CLIPMAP_LEVELS;

    this.entities.push(lightEntity);
  }

  create_board() {
    const cube = Mesh.cube();

    const floor = spawn_mesh_entity(
      [0, FLOOR_Y, 0],
      [0, 0, 0, 1],
      FLOOR_SCALE,
      cube,
      this.materials.floor
    );
    this.entities.push(floor);

    for (let z = 0; z < BOARD_SIZE; z += 1) {
      for (let x = 0; x < BOARD_SIZE; x += 1) {
        const material = (x + z) % 2 === 0 ? this.materials.tileDark : this.materials.tileLight;
        const tile = spawn_mesh_entity(
          this.grid_to_world({ x, y: z }, TILE_Y),
          [0, 0, 0, 1],
          TILE_SCALE,
          cube,
          material
        );

        this.boardTiles.push(tile);
        this.entities.push(tile);
      }
    }

    const wallScaleHorizontal = [BOARD_HALF_EXTENT + CELL_SPACING * 0.5, WALL_HEIGHT, WALL_THICKNESS];
    const wallScaleVertical = [WALL_THICKNESS, WALL_HEIGHT, BOARD_HALF_EXTENT + CELL_SPACING * 0.5];
    const outer = BOARD_HALF_EXTENT + CELL_SPACING * 0.5;

    const walls = [
      { position: [0, 0.1, outer], scale: wallScaleHorizontal },
      { position: [0, 0.1, -outer], scale: wallScaleHorizontal },
      { position: [outer, 0.1, 0], scale: wallScaleVertical },
      { position: [-outer, 0.1, 0], scale: wallScaleVertical },
    ];

    for (const wall of walls) {
      const entity = spawn_mesh_entity(
        wall.position,
        [0, 0, 0, 1],
        wall.scale,
        cube,
        this.materials.wall
      );
      this.entities.push(entity);
    }
  }

  create_snake_pool() {
    const cube = Mesh.cube();
    const poolSize = BOARD_SIZE * BOARD_SIZE;

    for (let index = 0; index < poolSize; index += 1) {
      const entity = spawn_mesh_entity(
        HIDDEN_POSITION,
        [0, 0, 0, 1],
        SEGMENT_SCALE,
        cube,
        this.materials.segment
      );

      const visibility = EntityManager.get_fragment(entity, VisibilityFragment);
      visibility.visible = 0;

      this.segmentEntities.push(entity);
      this.entities.push(entity);
    }
  }

  create_food_entity() {
    this.foodEntity = spawn_mesh_entity(
      HIDDEN_POSITION,
      [0, 0, 0, 1],
      FOOD_SCALE,
      Mesh.sphere(),
      this.materials.food
    );

    const visibility = EntityManager.get_fragment(this.foodEntity, VisibilityFragment);
    visibility.visible = 0;

    this.entities.push(this.foodEntity);
  }

  grid_to_world(point, y = PIECE_Y) {
    return [
      (point.x - (BOARD_SIZE - 1) * 0.5) * CELL_SPACING,
      y,
      (point.y - (BOARD_SIZE - 1) * 0.5) * CELL_SPACING,
    ];
  }

  restart_game() {
    this.rippleBursts.length = 0;
    this.state = createInitialState({
      gridSize: BOARD_SIZE,
      speed: SNAKE_SPEED,
    });
    this.sync_entities_to_state();
  }

  queue_direction(direction) {
    this.state = queueDirection(this.state, direction);

    if (this.state.paused) {
      this.state = togglePause(this.state);
    }
  }

  toggle_pause() {
    this.state = togglePause(this.state);
  }

  queue_eat_ripple(count = 1) {
    const total = Math.max(0, count);

    for (let index = 0; index < total; index += 1) {
      this.rippleBursts.push(
        createRippleBurst({
          elapsed: -index * RIPPLE_PULSE_DURATION * 0.45,
          pulseDuration: RIPPLE_PULSE_DURATION,
          segmentDelay: RIPPLE_SEGMENT_DELAY,
          amplitude: RIPPLE_AMPLITUDE,
          maxMultiplier: RIPPLE_MAX_MULTIPLIER,
        })
      );
    }
  }

  get_segment_scale(index) {
    const baseScale = index === 0 ? HEAD_SCALE : SEGMENT_SCALE;
    const multiplier = getRippleScaleMultiplier(index, this.rippleBursts);

    return baseScale.map((value) => value * multiplier);
  }

  sync_entities_to_state() {
    for (let index = 0; index < this.segmentEntities.length; index += 1) {
      const entity = this.segmentEntities[index];
      const visibility = EntityManager.get_fragment(entity, VisibilityFragment);
      const transform = EntityManager.get_fragment(entity, TransformFragment);

      if (index >= this.state.snake.length) {
        transform.position = HIDDEN_POSITION;
        visibility.visible = 0;
        continue;
      }

      const segment = this.state.snake[index];
      const mesh = EntityManager.get_fragment(entity, StaticMeshFragment);

      transform.position = this.grid_to_world(segment);
      transform.scale = this.get_segment_scale(index);
      mesh.material_slots = [BigInt(index === 0 ? this.materials.head : this.materials.segment)];
      visibility.visible = 1;
    }

    const foodVisibility = EntityManager.get_fragment(this.foodEntity, VisibilityFragment);
    if (this.state.food) {
      const transform = EntityManager.get_fragment(this.foodEntity, TransformFragment);
      transform.position = this.grid_to_world(this.state.food);
      transform.scale = FOOD_SCALE;
      foodVisibility.visible = 1;
    } else {
      const transform = EntityManager.get_fragment(this.foodEntity, TransformFragment);
      transform.position = HIDDEN_POSITION;
      foodVisibility.visible = 0;
    }
  }

  handle_keyboard() {
    if (this.any_action([InputKey.K_Up, InputKey.K_w])) {
      this.queue_direction(DIRECTIONS.UP);
    } else if (this.any_action([InputKey.K_Down, InputKey.K_s])) {
      this.queue_direction(DIRECTIONS.DOWN);
    } else if (this.any_action([InputKey.K_Left, InputKey.K_a])) {
      this.queue_direction(DIRECTIONS.LEFT);
    } else if (this.any_action([InputKey.K_Right, InputKey.K_d])) {
      this.queue_direction(DIRECTIONS.RIGHT);
    }

    if (this.any_action([InputKey.K_Space, InputKey.K_p])) {
      this.toggle_pause();
    }

    if (this.any_action([InputKey.K_r, InputKey.K_Return])) {
      this.restart_game();
    }
  }

  any_action(keys) {
    return keys.some((key) => InputProvider.get_action(key));
  }

  update(deltaTime) {
    this.update_camera_framing();
    this.handle_keyboard();
    let shouldSyncEntities = false;

    if (!this.state.paused && this.rippleBursts.length > 0) {
      this.rippleBursts = advanceRippleBursts(
        this.rippleBursts,
        deltaTime,
        this.state.snake.length
      );
      shouldSyncEntities = true;
    }

    if (!this.state.paused && !this.state.gameOver && !this.state.won) {
      const previousState = this.state;
      const nextState = advanceGame(this.state, deltaTime);
      const scoreIncrease = Math.max(0, nextState.score - previousState.score);

      this.state = nextState;
      shouldSyncEntities = shouldSyncEntities || nextState !== previousState;

      if (scoreIncrease > 0) {
        this.queue_eat_ripple(scoreIncrease);
        shouldSyncEntities = true;
      }
    }

    if (shouldSyncEntities) {
      this.sync_entities_to_state();
    }

    super.update(deltaTime);
    this.render_ui();
  }

  render_ui() {
    panel(HUD_PANEL, () => {
      label("Snake 3D", HUD_TITLE);
      label(`Score: ${this.state.score}`, HUD_TEXT);
      label("Pause: Space / P", HUD_TEXT);
      label("Restart: R / Enter", HUD_TEXT);
    });

    panel(CONTROLS_PANEL, () => {
      if (button("Up", { ...CONTROL_BUTTON, x: 70, y: 0 }).clicked) {
        this.queue_direction(DIRECTIONS.UP);
      }

      if (button("Left", { ...CONTROL_BUTTON, x: 0, y: 45 }).clicked) {
        this.queue_direction(DIRECTIONS.LEFT);
      }

      if (button("Down", { ...CONTROL_BUTTON, x: 70, y: 45 }).clicked) {
        this.queue_direction(DIRECTIONS.DOWN);
      }

      if (button("Right", { ...CONTROL_BUTTON, x: 140, y: 45 }).clicked) {
        this.queue_direction(DIRECTIONS.RIGHT);
      }

      if (
        button(this.state.paused ? "Resume" : "Pause", {
          ...ACTION_BUTTON,
          x: 0,
          y: 94,
        }).clicked
      ) {
        this.toggle_pause();
      }

      if (
        button("Restart", {
          ...ACTION_BUTTON,
          x: 140,
          y: 94,
        }).clicked
      ) {
        this.restart_game();
      }
    });
  }
}

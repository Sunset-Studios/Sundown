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
const UI_FONT_FAMILY = '"ExoDisplay", "Poppins", sans-serif';

const HUD_PANEL = {
  layout: "column",
  gap: 10,
  x: 18,
  y: 18,
  width: 248,
  padding: 14,
  background_color: "rgba(5, 10, 18, 0.68)",
  border: "1px solid rgba(95, 164, 214, 0.22)",
  corner_radius: 18,
  box_shadow: "0 14 28 rgba(0,0,0,0.22)",
};

const HUD_BADGE = {
  x: 0,
  width: 92,
  height: 24,
  text_align: "center",
  text_valign: "middle",
  font: `700 10px ${UI_FONT_FAMILY}`,
  text_color: "#d8f6ff",
  background_color: "rgba(67, 184, 255, 0.18)",
  border: "1px solid rgba(92, 196, 255, 0.34)",
  corner_radius: 999,
};

const HUD_TITLE = {
  x: 0,
  width: "100%",
  height: "fit-content",
  font: `700 26px ${UI_FONT_FAMILY}`,
  text_color: "#f7fbff",
  text_align: "left",
};

const STATUS_PANEL = {
  x: 0,
  width: "100%",
  layout: "column",
  gap: 4,
  padding: 12,
  background_color: "rgba(11, 18, 30, 0.92)",
  border: "1px solid rgba(95, 164, 214, 0.18)",
  corner_radius: 16,
};

const STATUS_BADGE = {
  x: 0,
  width: "fit-content",
  height: 24,
  text_padding: 10,
  text_align: "center",
  text_valign: "middle",
  font: `700 10px ${UI_FONT_FAMILY}`,
  corner_radius: 999,
};

const STATUS_TITLE = {
  x: 0,
  width: "100%",
  height: "fit-content",
  font: `700 18px ${UI_FONT_FAMILY}`,
  text_color: "#f7fbff",
  text_align: "left",
};

const STATUS_TEXT = {
  x: 0,
  width: "100%",
  height: "fit-content",
  wrap: true,
  text_padding: 0,
  font: `500 12px ${UI_FONT_FAMILY}`,
  text_color: "#9bb0c6",
  text_align: "left",
  text_valign: "top",
};

const HINTS_STRIP = {
  x: 0,
  width: "100%",
  layout: "row",
  gap: 8,
};

const HINT_CHIP = {
  x: 0,
  width: 106,
  height: 28,
  text_align: "center",
  text_valign: "middle",
  font: `600 11px ${UI_FONT_FAMILY}`,
  text_color: "#c7d7e8",
  background_color: "rgba(16, 26, 40, 0.76)",
  border: "1px solid rgba(89, 113, 139, 0.28)",
  corner_radius: 999,
};

const CONTROLS_PANEL = {
  layout: "column",
  gap: 15,
  width: 244,
  height: 210,
  x: 18,
  y: 18,
  anchor_y: "bottom",
  anchor_x: "right",
  padding: 15,
  background_color: "rgba(5, 10, 18, 0.68)",
  border: "1px solid rgba(95, 164, 214, 0.22)",
  corner_radius: 18,
  box_shadow: "0 14 28 rgba(0,0,0,0.22)",
};

const CONTROLS_TITLE = {
  x: 0,
  y: 0,
  width: "100%",
  height: "fit-content",
  font: `700 18px ${UI_FONT_FAMILY}`,
  text_color: "#f7fbff",
  text_align: "center",
};

const D_PAD_PANEL = {
  x: 0,
  y: 0,
  width: "100%",
  layout: "column",
  gap: 8,
  background_color: "rgba(12, 20, 32, 0.88)",
  border: "1px solid rgba(95, 164, 214, 0.18)",
  corner_radius: 16,
};

const CONTROL_ROW = {
  x: 0,
  y: 0,
  width: "100%",
  height: 42,
  layout: "row",
  gap: 12,
};

const CONTROL_BUTTON = {
  x: 0,
  y: 0,
  width: 64,
  height: 42,
  font: `700 12px ${UI_FONT_FAMILY}`,
  text_color: "#f5faff",
  background_color: "rgba(24, 39, 58, 0.92)",
  border: "1px solid rgba(103, 124, 153, 0.34)",
  corner_radius: 12,
};

const ACTION_BUTTON = {
  x: 0,
  y: 0,
  width: 102,
  height: 38,
  font: `700 12px ${UI_FONT_FAMILY}`,
  text_color: "#f5faff",
  background_color: "rgba(20, 32, 47, 0.94)",
  border: "1px solid rgba(103, 124, 153, 0.32)",
  corner_radius: 12,
};

const ACTION_ROW = {
  x: 0,
  y: 0,
  width: "100%",
  height: 38,
  layout: "row",
  gap: 12,
};

const TOP_ROW_SPACER = {
  x: 0,
  y: 0,
  width: 64,
  height: 42,
};

const OVERLAY_PANEL = {
  width: 420,
  layout: "column",
  gap: 10,
  padding: 24,
  background_color: "rgba(5, 10, 18, 0.9)",
  border: "1px solid rgba(95, 164, 214, 0.24)",
  corner_radius: 26,
  box_shadow: "0 24 44 rgba(0,0,0,0.32)",
};

const OVERLAY_TAG = {
  x: 0,
  y: 0,
  width: "fit-content",
  height: 30,
  text_padding: 14,
  text_align: "center",
  text_valign: "middle",
  font: `700 11px ${UI_FONT_FAMILY}`,
  corner_radius: 999,
};

const OVERLAY_TITLE = {
  x: 0,
  y: 0,
  width: "100%",
  height: "fit-content",
  font: `700 34px ${UI_FONT_FAMILY}`,
  text_color: "#f7fbff",
  text_align: "left",
};

const OVERLAY_TEXT = {
  x: 0,
  y: 0,
  width: "100%",
  height: "fit-content",
  wrap: true,
  text_padding: 0,
  font: `500 14px ${UI_FONT_FAMILY}`,
  text_color: "#a5bad0",
  text_align: "left",
  text_valign: "top",
};

const OVERLAY_HINT = {
  x: 0,
  y: 0,
  width: "100%",
  height: "fit-content",
  font: `500 12px ${UI_FONT_FAMILY}`,
  text_color: "#7f95aa",
  text_align: "left",
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

  get_heading_label(direction) {
    switch (direction) {
      case DIRECTIONS.UP:
        return "Northbound";
      case DIRECTIONS.DOWN:
        return "Southbound";
      case DIRECTIONS.LEFT:
        return "Westbound";
      case DIRECTIONS.RIGHT:
      default:
        return "Eastbound";
    }
  }

  get_status_meta() {
    if (this.state.won) {
      return {
        tag: "BOARD CLEARED",
        title: "Perfect line",
        detail: "Every tile is covered. Restart to chase another clean sweep.",
        badge_background: "rgba(80, 220, 159, 0.18)",
        badge_border: "1px solid rgba(122, 242, 190, 0.34)",
        badge_text: "#c6ffea",
        accent_background: "rgba(25, 54, 43, 0.84)",
        accent_border: "1px solid rgba(96, 212, 163, 0.24)",
      };
    }

    if (this.state.gameOver) {
      return {
        tag: "RUN ENDED",
        title: "Collision detected",
        detail: "The board is still warm. Tap Restart or press R to jump right back in.",
        badge_background: "rgba(255, 115, 115, 0.18)",
        badge_border: "1px solid rgba(255, 148, 148, 0.34)",
        badge_text: "#ffd9d9",
        accent_background: "rgba(59, 27, 31, 0.88)",
        accent_border: "1px solid rgba(255, 120, 132, 0.22)",
      };
    }

    if (this.state.paused) {
      return {
        tag: "PAUSED",
        title: "Hold the line",
        detail: "Resume with Space, P, or the button below whenever you are ready.",
        badge_background: "rgba(255, 196, 92, 0.18)",
        badge_border: "1px solid rgba(255, 212, 126, 0.34)",
        badge_text: "#ffe8b5",
        accent_background: "rgba(57, 43, 20, 0.88)",
        accent_border: "1px solid rgba(255, 196, 92, 0.22)",
      };
    }

    return {
      tag: "LIVE",
      title: "Smooth run",
      detail: "Keep threading the board. Eat cleanly and let the snake evolve.",
      badge_background: "rgba(77, 206, 255, 0.18)",
      badge_border: "1px solid rgba(109, 219, 255, 0.34)",
      badge_text: "#d6f7ff",
      accent_background: "rgba(18, 37, 53, 0.9)",
      accent_border: "1px solid rgba(77, 206, 255, 0.2)",
    };
  }

  get_direction_button_style(isActive) {
    if (isActive) {
      return {
        background_color: "rgba(61, 197, 255, 0.26)",
        border: "1px solid rgba(110, 223, 255, 0.5)",
        text_color: "#f9fdff",
      };
    }

    return {};
  }

  get_action_button_style(isPrimary) {
    if (isPrimary) {
      return {
        background_color: "rgba(65, 195, 255, 0.22)",
        border: "1px solid rgba(110, 223, 255, 0.4)",
      };
    }

    return {};
  }

  render_overlay(status) {
    panel(OVERLAY_PANEL, () => {
      label(status.tag, {
        ...OVERLAY_TAG,
        background_color: status.badge_background,
        border: status.badge_border,
        text_color: status.badge_text,
      });
      label(status.title, OVERLAY_TITLE);
      label(status.detail, OVERLAY_TEXT);
      label("Space / P to pause   |   R / Enter to restart", OVERLAY_HINT);
    });
  }

  update(deltaTime) {
    this.update_camera_framing();
    this.handle_keyboard();

    if (!this.state.paused && this.rippleBursts.length > 0) {
      this.rippleBursts = advanceRippleBursts(
        this.rippleBursts,
        deltaTime,
        this.state.snake.length
      );
    }

    if (!this.state.paused && !this.state.gameOver && !this.state.won) {
      const previousState = this.state;
      const nextState = advanceGame(this.state, deltaTime);
      const scoreIncrease = Math.max(0, nextState.score - previousState.score);

      this.state = nextState;

      if (scoreIncrease > 0) {
        this.queue_eat_ripple(scoreIncrease);
      }
    }

    this.sync_entities_to_state();

    super.update(deltaTime);
    this.render_ui();
  }

  render_ui() {
    const heading = this.state.nextDirection ?? this.state.direction;
    const headingLabel = this.get_heading_label(heading);
    const status = this.get_status_meta();

    panel(HUD_PANEL, () => {
      label("SUNDOWN", HUD_BADGE);
      label("Snake 3D", HUD_TITLE);

      panel(
        {
          ...STATUS_PANEL,
          background_color: status.accent_background,
          border: status.accent_border,
        },
        () => {
          label(status.tag, {
            ...STATUS_BADGE,
            background_color: status.badge_background,
            border: status.badge_border,
            text_color: status.badge_text,
          });
          label(status.title, STATUS_TITLE);
          label(status.detail, STATUS_TEXT);
        }
      );

      panel(HINTS_STRIP, () => {
        label(headingLabel, HINT_CHIP);
        label("WASD / Arrows", HINT_CHIP);
      });
    });

    panel(CONTROLS_PANEL, () => {
      label("Controls", CONTROLS_TITLE);

      panel(D_PAD_PANEL, () => {
        panel(CONTROL_ROW, () => {
          panel(TOP_ROW_SPACER, () => {});
          if (
            button("UP", {
              ...CONTROL_BUTTON,
              ...this.get_direction_button_style(heading === DIRECTIONS.UP),
            }).clicked
          ) {
            this.queue_direction(DIRECTIONS.UP);
          }
          panel(TOP_ROW_SPACER, () => {});
        });

        panel(CONTROL_ROW, () => {
          if (
            button("LEFT", {
              ...CONTROL_BUTTON,
              ...this.get_direction_button_style(heading === DIRECTIONS.LEFT),
            }).clicked
          ) {
            this.queue_direction(DIRECTIONS.LEFT);
          }

          if (
            button("DOWN", {
              ...CONTROL_BUTTON,
              ...this.get_direction_button_style(heading === DIRECTIONS.DOWN),
            }).clicked
          ) {
            this.queue_direction(DIRECTIONS.DOWN);
          }

          if (
            button("RIGHT", {
              ...CONTROL_BUTTON,
              ...this.get_direction_button_style(heading === DIRECTIONS.RIGHT),
            }).clicked
          ) {
            this.queue_direction(DIRECTIONS.RIGHT);
          }
        });
      });

      panel(ACTION_ROW, () => {
        if (
          button(this.state.paused ? "Resume" : "Pause", {
            ...ACTION_BUTTON,
            ...this.get_action_button_style(this.state.paused),
          }).clicked
        ) {
          this.toggle_pause();
        }

        if (
          button("Restart", {
            ...ACTION_BUTTON,
            ...this.get_action_button_style(this.state.gameOver || this.state.won),
          }).clicked
        ) {
          this.restart_game();
        }
      });
    });

    if (this.state.paused || this.state.gameOver || this.state.won) {
      this.render_overlay(status);
    }
  }
}

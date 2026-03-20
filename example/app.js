import { vec3, vec4, quat } from "gl-matrix";

import { Simulator } from "../engine/src/core/simulator.js";
import { Scene } from "../engine/src/core/scene.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../engine/src/core/shared_data.js";
import { LightType, WORLD_UP } from "../engine/src/core/minimal.js";
import { delete_entity, spawn_mesh_entity } from "../engine/src/core/ecs/entity_utils.js";
import { FreeformArcballControlProcessor } from "../engine/src/core/subsystems/freeform_arcball_control_processor.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { StandardMaterial } from "../engine/src/renderer/material.js";
import { MAX_CLIPMAP_LEVELS } from "../engine/src/renderer/shadows/shadow_utils.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { InputKey } from "../engine/src/input/input_types.js";
import { Renderer } from "../engine/src/renderer/renderer.js";

import * as UI from "../engine/src/ui/2d/immediate.js";

const RAD_TO_DEG = 180.0 / Math.PI;
const WORLD_FORWARD_VEC3 = vec3.fromValues(0, 0, 1);
const WORLD_UP_VEC3 = vec3.fromValues(WORLD_UP[0], WORLD_UP[1], WORLD_UP[2]);

const OVERLAY_COLORS = {
  surface: "rgba(14, 18, 24, 0.78)",
  surfaceStrong: "rgba(18, 24, 32, 0.9)",
  border: "rgba(255, 255, 255, 0.12)",
  text: "#f5f7fb",
  muted: "rgba(245, 247, 251, 0.72)",
  accent: "#8fd7ff",
  accentStrong: "rgba(68, 185, 255, 0.22)",
  accentBorder: "rgba(143, 215, 255, 0.4)",
  idleButton: "rgba(255, 255, 255, 0.08)",
  surfaceInner: "rgba(255, 255, 255, 0.04)",
  stickGlow: "rgba(143, 215, 255, 0.16)",
  stickCore: "rgba(68, 185, 255, 0.22)",
  stickHighlight: "rgba(255, 255, 255, 0.16)",
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function pointInRect(x, y, rect) {
  return x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height;
}

function clientToCanvasPosition(clientX, clientY) {
  const canvas = Renderer.get().canvas;
  const rect = canvas.getBoundingClientRect();

  if (rect.width <= 0 || rect.height <= 0) {
    return { x: 0, y: 0 };
  }

  return {
    x: clamp(((clientX - rect.left) / rect.width) * canvas.width, 0, canvas.width),
    y: clamp(((clientY - rect.top) / rect.height) * canvas.height, 0, canvas.height),
  };
}

class SponzaScene extends Scene {
  name = "SponzaScene";
  entities = [];
  sunLightEntity = null;
  sunLightEnabled = true;
  swayLightEnabled = false;
  swayPeriodSec = 35.0;
  swayAngleDeg = 120.0;
  sunLightIntensityOn = 30.0;
  sunLightBasePosition = [5.0, 20.0, 2.0];
  timeElapsedSec = 0.0;
  moveInput = [0, 0];
  lookInput = [0, 0];
  cameraYaw = 0.0;
  cameraPitch = 0.0;
  cameraMoveSpeed = 12.0;
  cameraLookSpeed = 2.35;
  maxCameraPitch = Math.PI * 0.48;
  overlayLayout = null;
  activePadPointers = new Map();
  movePadPointerId = null;
  lookPadPointerId = null;

  init() {
    super.init();

    const freeformArcball = this.add_layer(FreeformArcballControlProcessor);
    freeformArcball.move_speed = 12.0;
    freeformArcball.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");

    const viewData = SharedViewBuffer.get_view_data(this.context.current_view);
    viewData.view_position = vec4.fromValues(8.360948, 7.528844, -1.364703, 1.0);
    viewData.view_rotation = [-0.075029, 0.645789, -0.0024352, -0.68824034];
    this.syncCameraAnglesFromView(viewData);

    const lightEntity = EntityManager.create_entity([LightFragment]);
    this.entities.push(lightEntity);

    const lightFragmentView = EntityManager.get_fragment(lightEntity, LightFragment);
    lightFragmentView.type = LightType.DIRECTIONAL;
    lightFragmentView.color = [0.9, 0.9, 1.0];
    lightFragmentView.intensity = this.sunLightIntensityOn;
    lightFragmentView.position = [...this.sunLightBasePosition];
    lightFragmentView.active = true;
    lightFragmentView.is_primary_sun = 1;
    lightFragmentView.shadow_clipmaps = MAX_CLIPMAP_LEVELS;

    this.sunLightEntity = lightEntity;

    const groundMaterial = StandardMaterial.create("sponza_ground_material");
    groundMaterial.set_albedo([0.75, 0.75, 0.75, 1.0]);
    groundMaterial.set_roughness(0.9);
    groundMaterial.set_metallic(0.8);

    const groundEntity = spawn_mesh_entity(
      [0, 0, 0],
      quat.create(),
      [2000, 1.0, 2000],
      Mesh.cube(),
      groundMaterial.material_id
    );
    this.entities.push(groundEntity);

    const rootEntity = this.load_gltf_scene(
      "engine/models/sponza/Sponza.gltf",
      [0, 2.5, 0],
      [0, 0, 0, 1],
      [1.0, 1.0, 1.0]
    );
    this.entities.push(rootEntity);

    this.overlayLayout = this.computeOverlayLayout(UI.UIContext.canvas_size.width, UI.UIContext.canvas_size.height);

    this.handlePointerDown = this.onPointerDown.bind(this);
    this.handlePointerMove = this.onPointerMove.bind(this);
    this.handlePointerUp = this.onPointerUp.bind(this);

    window.addEventListener("pointerdown", this.handlePointerDown, { passive: false });
    window.addEventListener("pointermove", this.handlePointerMove, { passive: false });
    window.addEventListener("pointerup", this.handlePointerUp, { passive: false });
    window.addEventListener("pointercancel", this.handlePointerUp, { passive: false });

    this.show_dev_cursor();
    SharedViewBuffer.update_transforms([this.context.current_view]);
  }

  cleanup() {
    window.removeEventListener("pointerdown", this.handlePointerDown);
    window.removeEventListener("pointermove", this.handlePointerMove);
    window.removeEventListener("pointerup", this.handlePointerUp);
    window.removeEventListener("pointercancel", this.handlePointerUp);

    this.resetPadValue("move");
    this.resetPadValue("look");
    this.hide_dev_cursor();

    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.entities.length = 0;

    super.cleanup();
  }

  update(deltaTime) {
    super.update(deltaTime);
    this.timeElapsedSec += deltaTime;
    this.overlayLayout = this.computeOverlayLayout(UI.UIContext.canvas_size.width, UI.UIContext.canvas_size.height);

    if (InputProvider.get_action(InputKey.K_l)) {
      this.toggleDirectionalLight();
    }

    if (InputProvider.get_action(InputKey.K_t)) {
      this.toggleLightSway();
    }

    this.applyTouchCameraControls(deltaTime);
    this.updateDirectionalLight();
    SharedViewBuffer.update_transforms([this.context.current_view]);
    this.renderImmediateOverlay();
  }

  onPointerDown(event) {
    if (event.button !== undefined && event.button !== 0) {
      return;
    }

    if (!this.overlayLayout) {
      return;
    }

    const point = clientToCanvasPosition(event.clientX, event.clientY);
    if (pointInRect(point.x, point.y, this.overlayLayout.movePad) && this.movePadPointerId === null) {
      this.movePadPointerId = event.pointerId;
      this.activePadPointers.set(event.pointerId, "move");
      this.updatePadValue("move", point.x, point.y);
      this.preventPointerDefault(event);
      return;
    }

    if (pointInRect(point.x, point.y, this.overlayLayout.lookPad) && this.lookPadPointerId === null) {
      this.lookPadPointerId = event.pointerId;
      this.activePadPointers.set(event.pointerId, "look");
      this.updatePadValue("look", point.x, point.y);
      this.preventPointerDefault(event);
    }
  }

  onPointerMove(event) {
    const padName = this.activePadPointers.get(event.pointerId);
    if (!padName) {
      return;
    }

    const point = clientToCanvasPosition(event.clientX, event.clientY);
    this.updatePadValue(padName, point.x, point.y);
    this.preventPointerDefault(event);
  }

  onPointerUp(event) {
    const padName = this.activePadPointers.get(event.pointerId);
    if (!padName) {
      return;
    }

    this.activePadPointers.delete(event.pointerId);
    if (padName === "move") {
      this.movePadPointerId = null;
    } else {
      this.lookPadPointerId = null;
    }

    this.resetPadValue(padName);
    this.preventPointerDefault(event);
  }

  preventPointerDefault(event) {
    if (event.cancelable) {
      event.preventDefault();
    }
  }

  computeOverlayLayout(canvasWidth, canvasHeight) {
    const width = Math.max(canvasWidth || Renderer.get().canvas.width, 1);
    const height = Math.max(canvasHeight || Renderer.get().canvas.height, 1);
    const compact = width <= 900;
    const small = width <= 600;
    const padding = compact ? 14 : 18;
    const panelWidth = Math.min(440, Math.max(240, width - padding * 2));
    const panelHeight = small ? 88 : 96;
    const buttonHeight = 48;
    const buttonGap = 10;
    const buttonWidth = compact ? width - padding * 2 : 206;
    const topGroupHeight = compact
      ? panelHeight + buttonGap + buttonHeight * 2 + buttonGap
      : Math.max(panelHeight, buttonHeight * 2 + buttonGap);

    const preferredPadSize = small ? width * 0.4 : width * 0.34;
    const minPadSize = small ? 132 : 148;
    const maxPadSize = small ? 180 : 220;
    const maxHorizontalPadSize = (width - padding * 2 - 12) * 0.5;
    const padSize = clamp(preferredPadSize, minPadSize, Math.min(maxPadSize, maxHorizontalPadSize));
    const padY = height - padding - padSize;

    const infoCard = {
      x: padding,
      y: padding,
      width: Math.min(panelWidth, width - padding * 2),
      height: panelHeight,
    };

    const lightButton = compact
      ? {
          x: padding,
          y: padding + panelHeight + buttonGap,
          width: buttonWidth,
          height: buttonHeight,
        }
      : {
          x: width - padding - buttonWidth,
          y: padding,
          width: buttonWidth,
          height: buttonHeight,
        };

    const swayButton = compact
      ? {
          x: padding,
          y: lightButton.y + buttonHeight + buttonGap,
          width: buttonWidth,
          height: buttonHeight,
        }
      : {
          x: width - padding - buttonWidth,
          y: padding + buttonHeight + buttonGap,
          width: buttonWidth,
          height: buttonHeight,
        };

    return {
      width,
      height,
      compact,
      small,
      padding,
      topGroupHeight,
      infoCard,
      lightButton,
      swayButton,
      movePad: {
        x: padding,
        y: padY,
        width: padSize,
        height: padSize,
      },
      lookPad: {
        x: width - padding - padSize,
        y: padY,
        width: padSize,
        height: padSize,
      },
    };
  }

  renderImmediateOverlay() {
    const layout = this.overlayLayout;
    if (!layout) {
      return;
    }

    this.renderInfoCard(layout.infoCard, layout.small);
    this.renderToggleButton(
      layout.lightButton,
      this.sunLightEnabled ? "Directional Light: On" : "Directional Light: Off",
      this.sunLightEnabled,
      () => this.toggleDirectionalLight()
    );
    this.renderToggleButton(
      layout.swayButton,
      this.swayLightEnabled ? "Light Sway: On" : "Light Sway: Off",
      this.swayLightEnabled,
      () => this.toggleLightSway()
    );

    this.renderTrackpad(layout.movePad, "Move", this.moveInput, this.movePadPointerId !== null);
    this.renderTrackpad(layout.lookPad, "Look", this.lookInput, this.lookPadPointerId !== null);
  }

  renderInfoCard(rect, compactText) {
    UI.panel(
      {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        layout: "absolute",
        background_color: OVERLAY_COLORS.surface,
        border: `1px solid ${OVERLAY_COLORS.border}`,
        corner_radius: 18,
      },
      () => {
        UI.label("Sponza Demo", {
          x: 16,
          y: 14,
          width: "fit-content",
          height: "fit-content",
          font: "700 12px Poppins, sans-serif",
          text_color: OVERLAY_COLORS.accent,
        });

        UI.label("Mobile camera pads + light controls", {
          x: 16,
          y: 34,
          width: rect.width - 32,
          height: "fit-content",
          font: compactText ? "600 15px Poppins, sans-serif" : "600 16px Poppins, sans-serif",
          text_color: OVERLAY_COLORS.text,
        });

        UI.label("Left pad moves. Right pad looks around.", {
          x: 16,
          y: 60,
          width: rect.width - 32,
          height: "fit-content",
          font: compactText ? "500 12px Poppins, sans-serif" : "500 14px Poppins, sans-serif",
          text_color: OVERLAY_COLORS.muted,
        });
      }
    );
  }

  renderToggleButton(rect, text, isActive, onClick) {
    const hovered = pointInRect(UI.UIContext.input_state.x, UI.UIContext.input_state.y, rect);
    const background = isActive
      ? OVERLAY_COLORS.accentStrong
      : hovered
        ? "rgba(255, 255, 255, 0.14)"
        : OVERLAY_COLORS.idleButton;
    const borderColor = isActive ? OVERLAY_COLORS.accentBorder : OVERLAY_COLORS.border;

    const state = UI.button(text, {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
      background_color: background,
      border: `1px solid ${borderColor}`,
      corner_radius: 14,
      font: "600 15px Poppins, sans-serif",
      text_color: OVERLAY_COLORS.text,
      text_align: "left",
      text_padding: 14,
    });

    if (state.clicked) {
      onClick();
    }
  }

  renderTrackpad(rect, text, value, isActive) {
    const radius = Math.min(rect.width, rect.height) * 0.32;
    const knobSize = rect.width * 0.34;
    const innerInset = rect.width * 0.14;
    const knobX = rect.x + rect.width * 0.5 - knobSize * 0.5 + value[0] * radius;
    const knobY = rect.y + rect.height * 0.5 - knobSize * 0.5 + value[1] * radius;
    const glowSize = knobSize * 1.6;

    UI.panel(
      {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        layout: "absolute",
        background_color: OVERLAY_COLORS.surface,
        border: `1px solid ${OVERLAY_COLORS.border}`,
        corner_radius: rect.width * 0.13,
      },
      () => {
        UI.label(text, {
          x: 16,
          y: 14,
          width: "fit-content",
          height: "fit-content",
          font: "700 13px Poppins, sans-serif",
          text_color: OVERLAY_COLORS.muted,
        });

        UI.panel({
          x: innerInset,
          y: innerInset,
          width: rect.width - innerInset * 2,
          height: rect.height - innerInset * 2,
          background_color: isActive ? "rgba(143, 215, 255, 0.14)" : OVERLAY_COLORS.surfaceInner,
          border: "1px solid rgba(255, 255, 255, 0.1)",
          corner_radius: (rect.width - innerInset * 2) * 0.5,
        });

        UI.panel({
          x: rect.width * 0.5 - glowSize * 0.5 + value[0] * radius,
          y: rect.height * 0.5 - glowSize * 0.5 + value[1] * radius,
          width: glowSize,
          height: glowSize,
          background_color: OVERLAY_COLORS.stickGlow,
          corner_radius: glowSize * 0.5,
        });
      }
    );

    UI.panel({
      x: knobX,
      y: knobY,
      width: knobSize,
      height: knobSize,
      background_color: OVERLAY_COLORS.stickCore,
      border: `1px solid ${OVERLAY_COLORS.stickHighlight}`,
      corner_radius: knobSize * 0.5,
    });
  }

  updatePadValue(padName, canvasX, canvasY) {
    const rect = padName === "move" ? this.overlayLayout?.movePad : this.overlayLayout?.lookPad;
    if (!rect) {
      return;
    }

    const radius = Math.max(1, Math.min(rect.width, rect.height) * 0.32);
    const centerX = rect.x + rect.width * 0.5;
    const centerY = rect.y + rect.height * 0.5;

    let x = (canvasX - centerX) / radius;
    let y = (canvasY - centerY) / radius;
    const length = Math.hypot(x, y);

    if (length > 1.0) {
      x /= length;
      y /= length;
    }

    if (padName === "move") {
      this.moveInput[0] = x;
      this.moveInput[1] = y;
    } else {
      this.lookInput[0] = x;
      this.lookInput[1] = y;
    }
  }

  resetPadValue(padName) {
    if (padName === "move") {
      this.moveInput[0] = 0;
      this.moveInput[1] = 0;
    } else {
      this.lookInput[0] = 0;
      this.lookInput[1] = 0;
    }
  }

  toggleDirectionalLight() {
    if (!this.sunLightEntity) {
      return;
    }

    this.sunLightEnabled = !this.sunLightEnabled;

    const lightFragmentView = EntityManager.get_fragment(this.sunLightEntity, LightFragment);
    if (!lightFragmentView) {
      return;
    }

    lightFragmentView.intensity = this.sunLightEnabled ? this.sunLightIntensityOn : 0.0;
    lightFragmentView.shadows_dirty = 1;
  }

  toggleLightSway() {
    this.swayLightEnabled = !this.swayLightEnabled;
  }

  syncCameraAnglesFromView(viewData) {
    const forward = vec3.transformQuat(vec3.create(), WORLD_FORWARD_VEC3, viewData.view_rotation);
    vec3.normalize(forward, forward);

    this.cameraYaw = Math.atan2(forward[0], forward[2]);
    this.cameraPitch = Math.asin(clamp(forward[1], -1.0, 1.0));
  }

  applyTouchCameraControls(deltaTime) {
    const hasLookInput = Math.hypot(this.lookInput[0], this.lookInput[1]) > 0.001;
    const hasMoveInput = Math.hypot(this.moveInput[0], this.moveInput[1]) > 0.001;

    if (!hasLookInput && !hasMoveInput) {
      return;
    }

    const viewData = SharedViewBuffer.get_view_data(this.context.current_view);
    const viewPosition = vec4.clone(viewData.view_position);

    if (hasLookInput) {
      this.cameraYaw += this.lookInput[0] * this.cameraLookSpeed * deltaTime;
      this.cameraPitch = clamp(
        this.cameraPitch - this.lookInput[1] * this.cameraLookSpeed * deltaTime,
        -this.maxCameraPitch,
        this.maxCameraPitch
      );
    }

    const rotation = quat.fromEuler(
      quat.create(),
      this.cameraPitch * RAD_TO_DEG,
      this.cameraYaw * RAD_TO_DEG,
      0.0
    );

    if (hasMoveInput) {
      const forward = vec3.transformQuat(vec3.create(), WORLD_FORWARD_VEC3, rotation);
      forward[1] = 0.0;

      if (vec3.length(forward) > 0.0001) {
        vec3.normalize(forward, forward);
      } else {
        vec3.copy(forward, WORLD_FORWARD_VEC3);
      }

      const right = vec3.cross(vec3.create(), forward, WORLD_UP_VEC3);
      vec3.normalize(right, right);

      const forwardAmount = -this.moveInput[1] * this.cameraMoveSpeed * deltaTime;
      const strafeAmount = this.moveInput[0] * this.cameraMoveSpeed * deltaTime;

      vec4.scaleAndAdd(
        viewPosition,
        viewPosition,
        vec4.fromValues(forward[0], 0, forward[2], 0),
        forwardAmount
      );
      vec4.scaleAndAdd(
        viewPosition,
        viewPosition,
        vec4.fromValues(right[0], 0, right[2], 0),
        strafeAmount
      );
    }

    viewData.view_rotation = rotation;
    viewData.view_position = viewPosition;
  }

  updateDirectionalLight() {
    if (!this.sunLightEntity) {
      return;
    }

    const lightFragmentView = EntityManager.get_fragment(this.sunLightEntity, LightFragment);
    if (!lightFragmentView) {
      return;
    }

    if (!this.swayLightEnabled) {
      return;
    }

    const phase = (this.timeElapsedSec / this.swayPeriodSec) * Math.PI * 2.0;
    const angleRad = Math.sin(phase) * ((this.swayAngleDeg * Math.PI) / 180.0);

    const cosAngle = Math.cos(angleRad);
    const sinAngle = Math.sin(angleRad);

    const baseX = this.sunLightBasePosition[0];
    const baseY = this.sunLightBasePosition[1];
    const baseZ = this.sunLightBasePosition[2];

    const rotatedX = baseX * cosAngle + baseZ * sinAngle;
    const rotatedZ = -baseX * sinAngle + baseZ * cosAngle;

    lightFragmentView.position = [rotatedX, baseY, rotatedZ];
    lightFragmentView.shadows_dirty = 1;
  }
}

(async () => {
  const simulator = await Simulator.create("gpu-canvas", "ui-canvas");
  simulator.add_sim_layer(new SponzaScene("SponzaScene"));
  simulator.run();
})();

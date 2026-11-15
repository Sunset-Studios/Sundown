// ═══════════════════════════════════════════════════════════════════════════════
// UI3DTestScene — Sundown samples
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
import { vec3, quat } from "gl-matrix";
import * as UI3D from "../../engine/src/ui/3d/immediate.js";

export class UI3DTestScene extends Scene {
  name = "UI3DTestScene";
  entities = [];
  pulse = 0;
  counter = 0;
  selected_mode = "Layout";
  marker_position = [0.0, 0.0, 1.5];
  marker_rotation = quat.create();
  marker_half_extent = 1.03;

  init(parent_context) {
    super.init(parent_context);

    const freeform_arcball_control_processor = this.add_layer(FreeformArcballControlProcessor);
    freeform_arcball_control_processor.move_speed = 10.0;
    freeform_arcball_control_processor.set_scene(this);

    SharedEnvironmentData.set_skydome("default_scene_skydome");
    this.show_dev_cursor();

    const view_data = SharedViewBuffer.get_view_data(0);
    view_data.view_position = [0.0, 2.4, -8.0];
    view_data.view_rotation = [0.0, 0.0, 0.0, 1.0];
    view_data.far = 1000.0;

    const light_entity = EntityManager.create_entity([LightFragment]);
    this.entities.push(light_entity);

    const light_fragment_view = EntityManager.get_fragment(light_entity, LightFragment);
    light_fragment_view.type = LightType.DIRECTIONAL;
    light_fragment_view.color = [1.0, 0.96, 0.9, 1.0];
    light_fragment_view.intensity = 5.0;
    light_fragment_view.position = [4.0, 6.0, -3.0, 1.0];
    light_fragment_view.active = true;
    light_fragment_view.is_primary_sun = 1;

    const marker_material = StandardMaterial.create("UI3D_Marker");
    marker_material.set_albedo([0.08, 0.16, 0.2, 1.0]);
    marker_material.set_emission(0.15);

    this.marker_rotation = quat.fromEuler(quat.create(), 0, 45, 0);
    const marker = spawn_mesh_entity(
      this.marker_position,
      this.marker_rotation,
      [1.0, 1.0, 1.0],
      Mesh.cube(),
      marker_material.material_id
    );
    this.entities.push(marker);
  }

  cleanup() {
    for (const entity of this.entities) {
      delete_entity(entity);
    }
    this.entities.length = 0;

    this.remove_layer(FreeformArcballControlProcessor);

    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);

    this.pulse += delta_time;
    const pulse = 0.5 + Math.sin(this.pulse * 2.25) * 0.5;
    const scan_progress = 0.18 + pulse * 0.72;
    const load_progress = 0.54 + Math.sin(this.pulse * 1.15 + 0.8) * 0.18;
    const accent = [0.16 + pulse * 0.18, 0.72, 0.86, 1.0];

    this.render_hud_panel({
      position: [0.0, 3.7, 5.0],
      billboard: true,
      width: 4.8,
      height: 3.1,
      title: "HUD TEST",
      subtitle: "WORLD-SPACE UI",
      badge: "LIVE",
      pulse,
      scan_progress,
      load_progress,
      accent,
      interactive: true,
    });

    this.render_cube_hud_faces(pulse, scan_progress, load_progress, accent);
  }

  render_cube_hud_faces(pulse, scan_progress, load_progress, accent) {
    const face_panels = [
      { title: "FRONT", normal: [0, 0, -1], right: [-1, 0, 0], up: [0, 1, 0] },
      { title: "BACK", normal: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0] },
      { title: "LEFT", normal: [-1, 0, 0], right: [0, 0, 1], up: [0, 1, 0] },
      { title: "RIGHT", normal: [1, 0, 0], right: [0, 0, -1], up: [0, 1, 0] },
      { title: "TOP", normal: [0, 1, 0], right: [1, 0, 0], up: [0, 0, -1] },
      { title: "BOTTOM", normal: [0, -1, 0], right: [1, 0, 0], up: [0, 0, 1] },
    ];

    for (let i = 0; i < face_panels.length; i++) {
      const face = face_panels[i];
      const normal = vec3.transformQuat(vec3.create(), face.normal, this.marker_rotation);
      const right = vec3.transformQuat(vec3.create(), face.right, this.marker_rotation);
      const up = vec3.transformQuat(vec3.create(), face.up, this.marker_rotation);
      const position = vec3.scaleAndAdd(
        vec3.create(),
        this.marker_position,
        normal,
        this.marker_half_extent
      );

      this.render_hud_panel({
        position,
        right,
        up,
        width: 1.72,
        height: 1.18,
        title: face.title,
        subtitle: "FACE HUD",
        badge: `${i + 1}/6`,
        pulse,
        scan_progress: Math.max(0.08, Math.min(0.96, scan_progress - i * 0.04)),
        load_progress: Math.max(0.08, Math.min(0.96, load_progress + i * 0.035)),
        accent,
        compact: true,
        interactive: false,
      });
    }
  }

  render_hud_panel({
    position,
    billboard = false,
    right,
    up,
    width,
    height,
    title,
    subtitle,
    badge,
    pulse,
    scan_progress,
    load_progress,
    accent,
    compact = false,
    interactive = false,
  }) {
    const scale = compact ? 0.36 : 1.0;

    UI3D.panel(
      {
        position,
        billboard,
        right,
        up,
        width,
        height,
        layout: "column",
        gap: 0.12 * scale,
        padding: 0.18 * scale,
        background_color: [0.018, 0.024, 0.032, 0.9],
        border: { width: 0.025 * scale, color: [0.32, 0.78, 0.86, 0.45] },
        corner_radius: 0.12 * scale,
        z_order: 2,
      },
      () => {
        UI3D.begin_container({
          x: 0,
          y: 0,
          width: "100%",
          height: 0.46 * scale,
          layout: "row",
          gap: 0.12 * scale,
        });

        UI3D.label(title, {
          x: 0,
          y: 0,
          width: 2.25 * scale,
          height: "100%",
          text_color: [0.9, 0.98, 1.0, 1.0],
          text_align: "left",
          text_valign: "middle",
        });
        UI3D.label(subtitle, {
          x: 0,
          y: 0,
          width: 1.25 * scale,
          height: "100%",
          text_color: [0.48, 0.82, 0.88, 0.95],
          text_align: "center",
          text_valign: "middle",
          background_color: [0.05, 0.13, 0.16, 0.74],
          border: { width: 0.012 * scale, color: [0.3, 0.75, 0.82, 0.35] },
          corner_radius: 0.08 * scale,
        });
        UI3D.label(badge, {
          x: 0,
          y: 0,
          width: 0.68 * scale,
          height: "100%",
          text_color: [0.96, 0.68, 0.28, 1.0],
          text_align: "center",
          text_valign: "middle",
          background_color: [0.15, 0.08, 0.025, 0.82],
          corner_radius: 0.08 * scale,
        });

        UI3D.end_container();

        UI3D.panel(
          {
            x: 0,
            y: 0,
            width: "100%",
            height: 0.74 * scale,
            layout: "row",
            gap: 0.12 * scale,
            padding: 0.12 * scale,
            background_color: [0.035, 0.048, 0.06, 0.82],
            border: { width: 0.01 * scale, color: [0.4, 0.72, 0.78, 0.22] },
            corner_radius: 0.1 * scale,
          },
          () => {
            this.stat_tile(
              "Signal",
              `${Math.round(82 + pulse * 12)}%`,
              [0.5, 0.9, 0.76, 1.0],
              scale
            );
            this.stat_tile(
              "Latency",
              `${(1.8 + (1 - pulse) * 0.8).toFixed(1)}ms`,
              [0.98, 0.76, 0.38, 1.0],
              scale
            );
            this.stat_tile("Events", String(this.counter), [0.65, 0.78, 1.0, 1.0], scale);
          }
        );

        this.progress_row("Scene Sync", scan_progress, accent, scale);
        this.progress_row("Batch Load", load_progress, [0.66, 0.55, 1.0, 1.0], scale);

        if (!interactive) {
          return;
        }

        UI3D.begin_container({
          x: 0,
          y: 0,
          width: "100%",
          height: 0.46 * scale,
          layout: "row",
          gap: 0.1 * scale,
        });

        const button_base = {
          x: 0,
          y: 0,
          width: 0.96 * scale,
          height: "100%",
          text_color: [0.92, 0.98, 1.0, 1.0],
          background_color: [0.055, 0.095, 0.12, 0.94],
          hover_color: [0.09, 0.26, 0.31, 0.98],
          active_color: [0.12, 0.44, 0.52, 1.0],
          border: { width: 0.012 * scale, color: [0.36, 0.78, 0.86, 0.42] },
          corner_radius: 0.08 * scale,
        };

        if (UI3D.button("Orbit", button_base).clicked) {
          this.selected_mode = "Orbit";
        }
        if (UI3D.button("Inspect", button_base).clicked) {
          this.selected_mode = "Inspect";
        }
        if (UI3D.button("Pulse +", { ...button_base, width: 1.08 * scale }).clicked) {
          this.counter++;
        }
        UI3D.label(this.selected_mode.toUpperCase(), {
          x: 0,
          y: 0,
          width: 1.0 * scale,
          height: "100%",
          text_color: [0.94, 0.72, 0.34, 1.0],
          text_align: "center",
          text_valign: "middle",
          background_color: [0.13, 0.08, 0.025, 0.7],
          corner_radius: 0.08 * scale,
        });

        UI3D.end_container();
      }
    );
  }

  stat_tile(label_text, value_text, color, scale = 1.0) {
    UI3D.panel(
      {
        x: 0,
        y: 0,
        width: 1.31 * scale,
        height: "100%",
        layout: "column",
        gap: 0.035 * scale,
        padding: 0.075 * scale,
        background_color: [0.018, 0.026, 0.034, 0.9],
        border: { width: 0.008 * scale, color: [color[0], color[1], color[2], 0.26] },
        corner_radius: 0.08 * scale,
      },
      () => {
        UI3D.label(label_text, {
          x: 0,
          y: 0,
          width: "100%",
          height: 0.2 * scale,
          text_color: [0.48, 0.6, 0.64, 1.0],
          text_align: "left",
          text_valign: "middle",
        });
        UI3D.label(value_text, {
          x: 0,
          y: 0,
          width: "100%",
          height: 0.32 * scale,
          text_color: color,
          text_align: "left",
          text_valign: "middle",
        });
      }
    );
  }

  progress_row(label_text, value, color, scale = 1.0) {
    const clamped_value = Math.max(0, Math.min(1, value));
    UI3D.panel(
      {
        x: 0,
        y: 0,
        width: "100%",
        height: 0.38 * scale,
        layout: "row",
        gap: 0.12 * scale,
        padding: 0.07 * scale,
        background_color: [0.025, 0.036, 0.046, 0.72],
        corner_radius: 0.08 * scale,
      },
      () => {
        UI3D.label(label_text, {
          x: 0,
          y: 0,
          width: 1.15 * scale,
          height: "100%",
          text_color: [0.62, 0.76, 0.78, 1.0],
          text_align: "left",
          text_valign: "middle",
        });
        UI3D.panel(
          {
            x: 0,
            y: 0,
            width: 2.33 * scale,
            height: "100%",
            background_color: [0.07, 0.08, 0.09, 0.92],
            border: { width: 0.008 * scale, color: [0.25, 0.32, 0.35, 0.55] },
            corner_radius: 0.06 * scale,
          },
          () => {
            UI3D.rect({
              x: 0,
              y: 0,
              width: `${Math.round(clamped_value * 100)}%`,
              height: "100%",
              background_color: [color[0], color[1], color[2], 0.86],
              corner_radius: 0.06 * scale,
            });
          }
        );
        UI3D.label(`${Math.round(clamped_value * 100)}%`, {
          x: 0,
          y: 0,
          width: 0.55 * scale,
          height: "100%",
          text_color: color,
          text_align: "right",
          text_valign: "middle",
        });
      }
    );
  }
}

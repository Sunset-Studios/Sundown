// ═══════════════════════════════════════════════════════════════════════════════
// SceneSettingsPanel — Sundown samples
// ═══════════════════════════════════════════════════════════════════════════════

import { SimulationLayer } from "../engine/src/core/simulation_layer.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { LightType } from "../engine/src/core/minimal.js";
import * as UI from "../engine/src/ui/2d/immediate.js";

const scene_settings_panel_config = {
  layout: "column",
  gap: 10,
  x: 20,
  y: 20,
  anchor_x: "right",
  width: 392,
  padding: 12,
  background_color: "rgba(7, 12, 18, 0.96)",
  border: "1px solid rgba(105, 205, 255, 0.22)",
  corner_radius: 12,
  box_shadow: "0 16px 40px rgba(0,0,0,0.52)",
};

const scene_settings_header_config = {
  width: "100%",
  height: 44,
  x: 0,
  y: 0,
  background_color: "rgba(23, 34, 46, 0.9)",
  hover_color: "rgba(35, 53, 69, 0.96)",
  border: "1px solid rgba(105, 205, 255, 0.16)",
  corner_radius: 8,
  font: "600 13px monospace",
  text_color: "#f1f8fc",
  text_align: "left",
  text_padding: 14,
};

const scene_settings_section_header_config = {
  ...scene_settings_header_config,
  height: 36,
  background_color: "rgba(93, 196, 255, 0.08)",
  hover_color: "rgba(93, 196, 255, 0.14)",
  border: "1px solid rgba(93, 196, 255, 0.14)",
  corner_radius: 7,
  font: "600 12px monospace",
  text_color: "#8fd9ff",
  text_padding: 12,
};

const scene_settings_section_config = {
  layout: "column",
  x: 0,
  y: 0,
  gap: 10,
  width: "100%",
  height: 180,
  padding: 10,
  background_color: "rgba(255, 255, 255, 0.022)",
  border: "1px solid rgba(255, 255, 255, 0.065)",
  corner_radius: 8,
};

const scene_settings_group_label_config = {
  width: "100%",
  height: 18,
  x: 0,
  y: 0,
  font: "600 10px monospace",
  text_color: "#7f94a6",
  text_align: "left",
  text_valign: "middle",
  text_padding: 2,
};

const scene_settings_slider_config = {
  width: 88,
  height: 30,
  x: 0,
  y: 0,
  mode: "numeric",
  background_color: "rgba(255, 255, 255, 0.045)",
  hover_color: "rgba(92, 190, 255, 0.14)",
  active_color: "rgba(92, 190, 255, 0.23)",
  text_color: "#eef8ff",
  font: "11px monospace",
  corner_radius: 5,
};

const scene_settings_vector_field_config = {
  group_config: {
    height: 52,
  },
  label_config: scene_settings_group_label_config,
  row_config: {
    gap: 6,
    height: 30,
  },
  component_config: {
    width: 112,
    height: 30,
    background_color: "rgba(255, 255, 255, 0.035)",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    corner_radius: 6,
  },
  component_label_config: {
    width: 24,
    height: 30,
    font: "700 11px monospace",
  },
  component_label_styles: [
    { text_color: "#ff7c8f", background_color: "rgba(255, 93, 117, 0.09)" },
    { text_color: "#74dfa1", background_color: "rgba(87, 214, 139, 0.09)" },
    { text_color: "#7daeff", background_color: "rgba(99, 155, 255, 0.1)" },
  ],
  slider_config: scene_settings_slider_config,
};

function scene_settings_numeric_slider(id, value, range, config, on_change) {
  const result = UI.slider(value, {
    ...config,
    ...range,
    id: `scene_settings.${id}`,
    scrub_speed: range.step,
  });
  if (result.active && result.changed) {
    on_change(result.value);
  }
}

function scene_settings_intensity_field(value, on_change) {
  UI.panel(
    {
      layout: "row",
      gap: 8,
      width: "100%",
      height: 32,
      x: 0,
      y: 0,
    },
    () => {
      UI.label("INTENSITY", {
        ...scene_settings_group_label_config,
        width: 84,
        height: 32,
      });
      scene_settings_numeric_slider(
        "intensity",
        value,
        { min: 0, max: 100, step: 0.1, precision: 1 },
        {
          ...scene_settings_slider_config,
          width: 256,
          height: 32,
          background_color: "rgba(93, 196, 255, 0.075)",
          border: "1px solid rgba(93, 196, 255, 0.12)",
        },
        on_change
      );
    }
  );
}

export class SceneSettingsPanel extends SimulationLayer {
  scene_switcher = null;
  is_open = true;
  directional_light_section_open = true;

  constructor(scene_switcher) {
    super();
    this.name = "SceneSettingsPanel";
    this.scene_switcher = scene_switcher;
  }

  update(delta_time) {
    super.update(delta_time);

    const active_scene = this.scene_switcher.scenes[this.scene_switcher.current_scene_index];
    active_scene?.show_dev_cursor?.();

    const directional_light = this.find_directional_light(active_scene);
    const panel_height = !this.is_open ? 68 : this.directional_light_section_open ? 304 : 114;

    UI.panel(
      {
        ...scene_settings_panel_config,
        height: panel_height,
      },
      () => {
        const panel_header = UI.button(
          `${this.is_open ? "v" : ">"}  SCENE SETTINGS`,
          scene_settings_header_config
        );
        if (panel_header.clicked) {
          this.is_open = !this.is_open;
        }

        if (!this.is_open) {
          return;
        }

        const section_header = UI.button(
          `${this.directional_light_section_open ? "v" : ">"}  DIRECTIONAL LIGHT`,
          scene_settings_section_header_config
        );
        if (section_header.clicked) {
          this.directional_light_section_open = !this.directional_light_section_open;
        }

        if (!this.directional_light_section_open) {
          return;
        }

        UI.panel(scene_settings_section_config, () => {
          if (!directional_light) {
            UI.label("No directional light in this scene", {
              ...scene_settings_group_label_config,
              width: "100%",
              height: 32,
            });
            return;
          }

          this.render_directional_light_fields(directional_light);
        });
      }
    );
  }

  find_directional_light(scene) {
    const entity_lists = [scene?.entities, scene?.scene_entities];
    let fallback = null;

    for (const entities of entity_lists) {
      if (!Array.isArray(entities)) continue;

      for (const entity of entities) {
        if (
          !EntityManager.entity_exists(entity) ||
          !EntityManager.has_fragment(entity, LightFragment)
        ) {
          continue;
        }

        const light = EntityManager.get_fragment(entity, LightFragment);
        if (light.type !== LightType.DIRECTIONAL || !light.active) continue;
        if (light.is_primary_sun) return light;
        fallback ??= light;
      }
    }

    return fallback;
  }

  render_directional_light_fields(light) {
    const position = light.position;
    const direction_length = Math.hypot(position[0], position[1], position[2]);
    let direction = [position[0], position[1], position[2]];

    // Existing scenes use an arbitrary-length position vector for directional lights.
    // Bring legacy values into the editor's -1..1 component range once, then preserve
    // the edited components exactly. Lighting and shadow code normalize this vector.
    if (direction_length < 0.0001) {
      direction = [0, -1, 0];
      light.position = [...direction, position[3]];
      light.shadows_dirty = 1;
    } else if (direction.some((component) => Math.abs(component) > 1)) {
      direction = direction.map((component) => component / direction_length);
      light.position = [...direction, position[3]];
      light.shadows_dirty = 1;
    }

    const direction_range = { min: -1, max: 1, step: 0.01, precision: 2 };
    const direction_result = UI.vector_field(direction, {
      ...scene_settings_vector_field_config,
      id: "scene_settings.direction",
      label: "DIRECTION",
      components: ["X", "Y", "Z"],
      slider_config: {
        ...scene_settings_vector_field_config.slider_config,
        ...direction_range,
      },
    });
    if (
      direction_result.changed &&
      Math.hypot(direction_result.value[0], direction_result.value[1], direction_result.value[2]) >=
        0.0001
    ) {
      light.position = [...direction_result.value, position[3]];
      light.shadows_dirty = 1;
    }

    const color = light.color;
    const color_range = { min: 0, max: 1, step: 0.01, precision: 2 };
    const color_result = UI.vector_field(color, {
      ...scene_settings_vector_field_config,
      id: "scene_settings.color",
      label: "COLOR",
      components: ["R", "G", "B"],
      slider_config: {
        ...scene_settings_vector_field_config.slider_config,
        ...color_range,
      },
    });
    if (color_result.changed) {
      light.color = color_result.value;
    }

    scene_settings_intensity_field(light.intensity, (value) => {
      light.intensity = value;
      light.shadows_dirty = 1;
    });
  }
}

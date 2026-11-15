// ═══════════════════════════════════════════════════════════════════════════════
// Common world-space UI utilities built on the immediate-mode API.
// ═══════════════════════════════════════════════════════════════════════════════

import { begin_container, end_container, label } from "./immediate.js";

// Reuse descriptors for labels emitted every frame rather than constructing
// temporary options and callbacks. Command/layout data belongs to immediate UI.
const world_label_default_config = {
  pivot: [0.5, 0.5],
  background_color: [0, 0, 0, 0],
  text_color: [1, 1, 1, 1],
};
const world_label_panel_config = {
  widget_id: 0,
  position: null,
  billboard: true,
  width: 1,
  height: 1,
  unit_scale: 1,
  pivot: world_label_default_config.pivot,
  padding: 0,
  background_color: world_label_default_config.background_color,
  z_order: 1,
};
const world_label_text_config = {
  widget_id: 0,
  width: "100%",
  height: "100%",
  font: "Exo-Medium",
  text_color: world_label_default_config.text_color,
  text_align: "center",
  text_valign: "middle",
  text_emissive: 1,
};

/** Draw a centered world-space label, billboarded toward the camera by default. */
export function world_label(text, position, config = world_label_default_config) {
  world_label_panel_config.position = position;
  world_label_panel_config.billboard = config.billboard ?? true;
  world_label_panel_config.width = config.width ?? 1;
  world_label_panel_config.height = config.height ?? 1;
  world_label_panel_config.unit_scale = Number(config.unit_scale ?? 1);
  world_label_panel_config.pivot = config.pivot ?? world_label_default_config.pivot;
  world_label_panel_config.background_color = config.background_color ?? world_label_default_config.background_color;
  world_label_panel_config.z_order = config.z_order ?? 1;
  world_label_text_config.font = config.font ?? "Exo-Medium";
  world_label_text_config.text_color = config.text_color ?? world_label_default_config.text_color;
  world_label_text_config.text_align = config.text_align ?? "center";
  world_label_text_config.text_valign = config.text_valign ?? "middle";
  world_label_text_config.text_emissive = Number(config.text_emissive ?? 1);

  begin_container(world_label_panel_config);
  label(text, world_label_text_config);
  end_container();
}

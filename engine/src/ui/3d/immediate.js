import { InputKey, InputRange } from "../../input/input_types.js";
import { InputProvider } from "../../input/input_provider.js";
import { EntityFlags } from "../../core/minimal.js";
import { FrameAllocator, FrameStackAllocator } from "../../memory/allocator.js";
import { SharedViewBuffer } from "../../core/shared_data.js";
import { Renderer } from "../../renderer/renderer.js";
import { FontCache } from "../text/font_cache.js";
import { Name } from "../../utility/names.js";
import { world_pos_to_screen_pos } from "../../utility/camera.js";
import { profile_scope } from "../../utility/performance.js";
import {
  get_current_world_transform,
  resolve_parent_transform,
} from "../../utility/transform_utils.js";
import { mat4, quat, vec3 } from "gl-matrix";
import { UIContext } from "../2d/immediate.js";

const left = "left";
const right = "right";
const top = "top";
const bottom = "bottom";
const center = "center";
const middle = "middle";
const absolute = "absolute";
const row = "row";
const column = "column";
const row_reversed = "row_reversed";
const column_reversed = "column_reversed";
const width_name = "width";
const height_name = "height";
const fit_content = "fit-content";

export const UI3DCommandType = Object.freeze({
  Quad: "quad",
  Text: "text",
});

class Layout3DContainer {
  x = 0;
  y = 0;
  width = 0;
  height = 0;
  layout = absolute;
  gap = 0;
  padding_left = 0;
  padding_top = 0;
  padding_right = 0;
  padding_bottom = 0;
  cursor = { x: 0, y: 0 };
  config = {};
  auto_width = false;
  auto_height = false;
  content_max_x = 0;
  content_max_y = 0;
  z_order = 0;
  depth = 0;
  background_depth = 0;
  root = null;
}

class UI3DRoot {
  position = vec3.create();
  right = vec3.fromValues(1, 0, 0);
  up = vec3.fromValues(0, 1, 0);
  normal = vec3.fromValues(0, 0, 1);
  unit_scale = 1;
  layer_depth = 0.002;
  parent_entity = null;
}

const default_root = new UI3DRoot();
const temp_vec3_a = vec3.create();
const temp_vec3_b = vec3.create();
const temp_vec3_c = vec3.create();
const temp_vec3_d = vec3.create();
const temp_parent_world_transform = mat4.create();
const temp_resolved_parent_world_transform = mat4.create();

export const UI3DContext = {
  commands: [],
  command_order: 0,
  layout_allocator: new FrameAllocator(128, Layout3DContainer),
  layout_stack: new FrameStackAllocator(64, 0),
  id_counter: 0,
  scroll_state: {},
  input_state: {},
  font_cache: new Map(),

  get_unique_id() {
    return this.id_counter++;
  },

  get_default_font_id() {
    let font_id = this.font_cache.get("default");
    if (font_id !== undefined) {
      return font_id;
    }

    font_id = Name.from("Exo-Medium");
    this.font_cache.set("default", font_id);
    return font_id;
  },
};

function parse_dimension(value, base = 0) {
  if (typeof value === "string") {
    if (value.endsWith("%")) {
      return parseFloat(value) * 0.01 * base;
    }
    return parseFloat(value);
  }
  return Number(value) || 0;
}

function text_size_from_rect(height, text_padding = 0) {
  return Math.max(0, height - text_padding * 2);
}

function color_to_vec4(color, fallback = [1, 1, 1, 1]) {
  if (!color) return fallback;

  if (typeof color === "string") {
    if (color === "transparent") {
      return [0, 0, 0, 0];
    }

    if (color.startsWith("#")) {
      const hex = color.slice(1);
      const expand = hex.length === 3 || hex.length === 4;
      const r = expand ? hex[0] + hex[0] : hex.slice(0, 2);
      const g = expand ? hex[1] + hex[1] : hex.slice(2, 4);
      const b = expand ? hex[2] + hex[2] : hex.slice(4, 6);
      const a = expand ? hex[3] + hex[3] : hex.slice(6, 8);
      return [
        parseInt(r || "ff", 16) / 255,
        parseInt(g || "ff", 16) / 255,
        parseInt(b || "ff", 16) / 255,
        a ? parseInt(a, 16) / 255 : 1,
      ];
    }

    const rgba_match = color.match(/rgba?\(([^)]+)\)/);
    if (rgba_match) {
      const parts = rgba_match[1].split(",").map((part) => Number(part.trim()));
      return [
        (parts[0] ?? fallback[0] * 255) / 255,
        (parts[1] ?? fallback[1] * 255) / 255,
        (parts[2] ?? fallback[2] * 255) / 255,
        parts[3] ?? fallback[3],
      ];
    }
  }

  if (Array.isArray(color) || ArrayBuffer.isView(color)) {
    return [
      Number(color[0] ?? fallback[0]),
      Number(color[1] ?? fallback[1]),
      Number(color[2] ?? fallback[2]),
      Number(color[3] ?? fallback[3]),
    ];
  }

  if (typeof color === "object") {
    return [
      Number(color.r ?? fallback[0]),
      Number(color.g ?? fallback[1]),
      Number(color.b ?? fallback[2]),
      Number(color.a ?? fallback[3]),
    ];
  }

  return fallback;
}

function parse_border(border) {
  if (!border) {
    return { width: 0, color: [0, 0, 0, 0] };
  }

  if (typeof border === "number") {
    return { width: border, color: [1, 1, 1, 1] };
  }

  if (typeof border === "object" && !Array.isArray(border)) {
    return {
      width: Number(border.width ?? 1),
      color: color_to_vec4(border.color, [1, 1, 1, 1]),
    };
  }

  return { width: 0, color: [0, 0, 0, 0] };
}

function resolve_z_order(config = {}, inherited_z_order = 0) {
  const configured_z_order = config.z_order ?? config.z_index ?? inherited_z_order;
  const z_order = Number(configured_z_order);
  return Number.isFinite(z_order) ? z_order : inherited_z_order;
}

function get_current_container() {
  const container_index = UI3DContext.layout_stack.peek();
  if (container_index === null) {
    return {
      x: 0,
      y: 0,
      cursor: { x: 0, y: 0 },
      width: 0,
      height: 0,
      z_order: 0,
      depth: 0,
      root: default_root,
    };
  }

  return UI3DContext.layout_allocator.get(container_index.value);
}

function get_current_z_order() {
  const container = get_current_container();
  return container?.z_order ?? 0;
}

function push_command(command, config = {}, inherited_z_order = get_current_z_order()) {
  command.z_order = resolve_z_order(config, inherited_z_order);
  command.order = UI3DContext.command_order++;
  command.batch_key ??= config.batch_key ?? config.ui_batch_key;
  command.parent_entity ??= config.parent_entity ?? config.parent ?? get_current_container()?.root?.parent_entity ?? null;
  UI3DContext.commands.push(command);
  return UI3DContext.commands.length - 1;
}

function set_command(index, command, config = {}, inherited_z_order = get_current_z_order()) {
  const existing = UI3DContext.commands[index];
  command.z_order = existing?.z_order ?? resolve_z_order(config, inherited_z_order);
  command.order = existing?.order ?? UI3DContext.command_order++;
  command.batch_key ??= existing?.batch_key ?? config.batch_key ?? config.ui_batch_key;
  command.parent_entity ??= existing?.parent_entity ?? config.parent_entity ?? config.parent ?? get_current_container()?.root?.parent_entity ?? null;
  UI3DContext.commands[index] = command;
}

function material_command_config(config = {}) {
  return {
    material_id: config.material_id,
    material_template: config.material_template,
    material_shader: config.material_shader,
    material_family: config.material_family,
    material_key: config.material_key,
    material_name: config.material_name,
    storage_bindings: config.storage_bindings ?? config.buffers,
    texture_bindings: config.texture_bindings ?? config.textures,
    uniform_bindings: config.uniform_bindings ?? config.uniforms,
  };
}

function basis_from_config(config = {}) {
  const root = new UI3DRoot();
  vec3.copy(root.position, config.position ?? default_root.position);
  root.parent_entity = config.parent_entity ?? config.parent ?? null;

  if (config.right) {
    vec3.normalize(root.right, config.right);
  }
  if (config.up) {
    vec3.normalize(root.up, config.up);
  }

  if (!config.right && !config.up && config.rotation) {
    vec3.transformQuat(root.right, vec3.fromValues(1, 0, 0), config.rotation);
    vec3.transformQuat(root.up, vec3.fromValues(0, 1, 0), config.rotation);
    vec3.normalize(root.right, root.right);
    vec3.normalize(root.up, root.up);
  }

  if (!config.right && !config.up && config.billboard) {
    const view = SharedViewBuffer.get_view_data(0);
    vec3.copy(root.right, view.right);
    vec3.normalize(root.right, root.right);
    vec3.cross(root.up, root.right, view.forward);
    vec3.normalize(root.up, root.up);
  }

  root.unit_scale = Number(config.unit_scale ?? config.world_units_per_px ?? default_root.unit_scale);
  root.layer_depth = Number(config.layer_depth ?? config.depth_step ?? default_root.layer_depth);
  vec3.cross(root.normal, root.right, root.up);
  vec3.normalize(root.normal, root.normal);
  return root;
}

function clone_root_with_origin(config, width, height) {
  const root = basis_from_config(config);
  const pivot_x = Number(config.pivot_x ?? config.pivot?.[0] ?? 0.5);
  const pivot_y = Number(config.pivot_y ?? config.pivot?.[1] ?? 0.5);
  const origin = vec3.clone(root.position);

  vec3.scaleAndAdd(origin, origin, root.right, -width * root.unit_scale * pivot_x);
  vec3.scaleAndAdd(origin, origin, root.up, height * root.unit_scale * pivot_y);
  root.position = origin;

  return root;
}

function local_to_world(root, x, y, out = vec3.create(), depth = 0) {
  vec3.copy(out, root.position);
  vec3.scaleAndAdd(out, out, root.right, x * root.unit_scale);
  vec3.scaleAndAdd(out, out, root.up, -y * root.unit_scale);
  vec3.scaleAndAdd(out, out, root.normal, depth);
  return out;
}

function local_rect_to_world(root, x, y, width, height, depth = 0) {
  const origin = local_to_world(root, x, y, vec3.create(), depth);
  const x_axis = vec3.scale(vec3.create(), root.right, width * root.unit_scale);
  const y_axis = vec3.scale(vec3.create(), root.up, -height * root.unit_scale);
  return { origin, x_axis, y_axis };
}

function project_rect_to_screen(root, x, y, width, height) {
  const renderer = Renderer.get();
  if (!renderer) return null;

  const view = SharedViewBuffer.get_view_data(0);
  const canvas_width = renderer.canvas.width;
  const canvas_height = renderer.canvas.height;

  const p0 = local_to_world(root, x, y, temp_vec3_a);
  const p1 = local_to_world(root, x + width, y, temp_vec3_b);
  const p2 = local_to_world(root, x + width, y + height, temp_vec3_c);
  const p3 = local_to_world(root, x, y + height, temp_vec3_d);

  if (root.parent_entity) {
    const parent_transform = resolve_parent_transform(
      get_current_world_transform(
        root.parent_entity,
        0,
        temp_parent_world_transform
      ),
      EntityFlags.IGNORE_PARENT_SCALE,
      temp_resolved_parent_world_transform
    );
    vec3.transformMat4(p0, p0, parent_transform);
    vec3.transformMat4(p1, p1, parent_transform);
    vec3.transformMat4(p2, p2, parent_transform);
    vec3.transformMat4(p3, p3, parent_transform);
  }

  const projected = [
    world_pos_to_screen_pos(view, p0, canvas_width, canvas_height),
    world_pos_to_screen_pos(view, p1, canvas_width, canvas_height),
    world_pos_to_screen_pos(view, p2, canvas_width, canvas_height),
    world_pos_to_screen_pos(view, p3, canvas_width, canvas_height),
  ];

  if (projected.some((point) => !point || !point.is_visible)) {
    return null;
  }

  let min_x = Number.POSITIVE_INFINITY;
  let min_y = Number.POSITIVE_INFINITY;
  let max_x = Number.NEGATIVE_INFINITY;
  let max_y = Number.NEGATIVE_INFINITY;

  for (const point of projected) {
    min_x = Math.min(min_x, point.x);
    min_y = Math.min(min_y, point.y);
    max_x = Math.max(max_x, point.x);
    max_y = Math.max(max_y, point.y);
  }

  return {
    x: min_x,
    y: min_y,
    width: max_x - min_x,
    height: max_y - min_y,
  };
}

function is_input_within(bounds) {
  if (!bounds) return false;

  return (
    UIContext.input_state.x >= bounds.x &&
    UIContext.input_state.x <= bounds.x + bounds.width &&
    UIContext.input_state.y >= bounds.y &&
    UIContext.input_state.y <= bounds.y + bounds.height
  );
}

function element_handle_input(root, x, y, width, height, config) {
  const bounds = project_rect_to_screen(root, x, y, width, height);
  const hovered = is_input_within(bounds);
  const clicked = hovered && UIContext.input_state.clicked;
  const pressed = hovered && UIContext.input_state.pressed;
  let dragged = false;

  if (!UIContext.drag_state.active && (clicked || pressed)) {
    InputProvider.consume_action(InputKey.B_mouse_left);
  }

  if (config.draggable) {
    if (!UIContext.drag_state.active && hovered && UIContext.input_state.pressed) {
      UIContext.drag_state.active = true;
      UIContext.drag_state.widget_id = config.widget_id;
    }

    if (
      UIContext.drag_state.active &&
      !UIContext.drag_state.started &&
      UIContext.drag_state.timer > (config.drag_delay || 0) &&
      UIContext.drag_state.widget_id === config.widget_id
    ) {
      config.on_drag_start?.();
      UIContext.drag_state.started = true;
    }

    if (UIContext.drag_state.active && UIContext.drag_state.started) {
      if (UIContext.drag_state.widget_id === config.widget_id) {
        const offset_x = UIContext.input_state.prev_x - UIContext.input_state.x;
        const offset_y = UIContext.input_state.y - UIContext.input_state.prev_y;
        if (UIContext.input_state.pressed) {
          config.on_drag?.(offset_x, offset_y);
          dragged = true;
        } else {
          config.on_drop?.(x, y);
        }
        InputProvider.consume_action(InputKey.B_mouse_left);
      } else if (UIContext.input_state.pressed && hovered) {
        config.on_drag_over?.(x, y, width, height);
      }
    }
  }

  if (config.scrollable && hovered && UIContext.input_state.wheel !== 0) {
    const scroll_delta = -UIContext.input_state.wheel * (config.scroll_speed || 20);
    const scroll_id = config.widget_id;
    const current_scroll = UI3DContext.scroll_state[scroll_id] || 0;
    UI3DContext.scroll_state[scroll_id] = current_scroll + scroll_delta;
    InputProvider.consume_range(InputRange.M_wheel);
  }

  return { hovered, clicked, pressed, dragged, widget_id: config.widget_id, bounds };
}

function child_container_layout_update(container, x, y, width, height) {
  if (container.layout !== absolute) {
    if (container.layout === row) {
      container.cursor.x += width + container.gap;
    } else if (container.layout === row_reversed) {
      container.cursor.x -= width + container.gap;
    } else if (container.layout === column) {
      container.cursor.y += height + container.gap;
    } else if (container.layout === column_reversed) {
      container.cursor.y -= height + container.gap;
    }

    if (container.auto_width) {
      if (container.layout === row_reversed) {
        container.content_max_x = Math.max(
          container.content_max_x,
          container.x + container.width + container.padding_right - container.cursor.x
        );
      } else {
        container.content_max_x = Math.max(
          container.content_max_x,
          container.cursor.x - (container.x + container.padding_left + container.gap)
        );
      }
    }
    if (container.auto_height) {
      if (container.layout === column_reversed) {
        container.content_max_y = Math.max(
          container.content_max_y,
          container.y + container.height + container.padding_bottom - container.cursor.y
        );
      } else {
        container.content_max_y = Math.max(
          container.content_max_y,
          container.cursor.y - (container.y + container.padding_top + container.gap)
        );
      }
    }
  } else {
    if (container.auto_width) {
      const rel_x = x - (container.x + container.padding_left);
      container.content_max_x = Math.max(container.content_max_x, rel_x + width);
    }
    if (container.auto_height) {
      const rel_y = y - (container.y + container.padding_top);
      container.content_max_y = Math.max(container.content_max_y, rel_y + height);
    }
  }
}

function resolve_widget_rect(config, container, label_text = "") {
  const text_padding = Number(config.text_padding ?? config.padding ?? 0);
  let width;
  let height;

  if (config.height === fit_content) {
    height = container.height > 0 ? container.height : 1 + text_padding * 2;
  } else {
    height = parse_dimension(config.height, container.height);
  }

  if (config.width === fit_content) {
    width =
      measure_text_width(label_text, {
        ...config,
        _resolved_font_size: text_size_from_rect(height, text_padding),
      }) +
      text_padding * 2;
  } else {
    width = parse_dimension(config.width, container.width);
  }

  let offset_x;
  if (config.x !== undefined) {
    offset_x = parse_dimension(config.x, container.width);
  } else {
    offset_x = container.auto_width ? 0 : (container.width - width) / 2;
  }

  let offset_y;
  if (config.y !== undefined) {
    offset_y = parse_dimension(config.y, container.height);
  } else {
    offset_y = container.auto_height ? 0 : (container.height - height) / 2;
  }

  let x;
  if (config.anchor_x === right) {
    x = container.x + container.width - offset_x - width;
  } else {
    x = container.cursor.x + offset_x;
  }

  let y;
  if (config.anchor_y === bottom) {
    y = container.y + container.height - offset_y - height;
  } else {
    y = container.cursor.y + offset_y;
  }

  return { x, y, width, height };
}

function push_quad(
  root,
  x,
  y,
  width,
  height,
  config = {},
  inherited_z_order = get_current_z_order(),
  depth = 0
) {
  if (width <= 0 || height <= 0) {
    return -1;
  }

  const border = parse_border(config.border);
  const fill_color = color_to_vec4(
    config.background_color ?? config.color ?? config.fill_color,
    [0, 0, 0, 0]
  );
  const border_color = color_to_vec4(config.border_color, border.color);

  const world = local_rect_to_world(root, x, y, width, height, depth);
  return push_command(
    {
      type: UI3DCommandType.Quad,
      origin: world.origin,
      x_axis: world.x_axis,
      y_axis: world.y_axis,
      fill_color,
      border_color,
      width,
      height,
      corner_radius: Number(config.corner_radius ?? config.radius ?? 0),
      border_width: border.width,
      emissive: Number(config.emissive ?? 0),
      ...material_command_config(config),
    },
    config,
    inherited_z_order
  );
}

function font_from_config(config = {}) {
  let font_id =
    typeof config.font === "number"
      ? config.font
      : Name.from(config.font_name ?? config.font ?? "Exo-Medium");
  let font = FontCache.get_font_object(font_id);
  if (!font) {
    font_id = UI3DContext.get_default_font_id();
    font = FontCache.get_font_object(font_id);
  }
  return { font_id, font };
}

function measure_text_width(text, config = {}) {
  const { font } = font_from_config(config);
  if (!font || !text) return 0;

  const layout = layout_text_glyphs(text, font, config);
  return layout.width;
}

export function layout_text_glyphs(text, font, config = {}) {
  const string_value = String(text ?? "");
  const font_size = Number(config._resolved_font_size ?? 1);
  const scale = font_size / Math.max(1, font?.line_height || font?.texture_height || font_size);
  const glyphs = [];
  let width = 0;
  let previous_code_point = null;

  if (!font || !string_value.length) {
    return { glyphs, width, height: font_size, scale };
  }

  for (const char of string_value) {
    const code_point = char.codePointAt(0);
    const glyph_index =
      font.code_point_index_map.get(code_point) ??
      font.code_point_index_map.get(" ".codePointAt(0)) ??
      0;

    if (previous_code_point !== null) {
      const kern = font.kerning_matrix?.get_adjacent_value(previous_code_point, code_point) ?? 0;
      width += kern * scale;
    }

    glyphs.push({
      glyph_index,
      code_point,
      x: width + (font.x_offset[glyph_index] ?? 0) * scale,
      y: (font.y_offset[glyph_index] ?? 0) * scale,
      width: (font.width[glyph_index] ?? 0) * scale,
      height: (font.height[glyph_index] ?? 0) * scale,
      page: font.page[glyph_index] ?? 0,
    });

    width += (font.x_advance[glyph_index] || font.width[glyph_index] || font_size) * scale;
    previous_code_point = code_point;
  }

  return { glyphs, width, height: font_size, scale };
}

function push_text(
  text,
  root,
  x,
  y,
  width,
  height,
  config = {},
  inherited_z_order = get_current_z_order(),
  depth = 0
) {
  const string_value = String(text ?? "");
  if (!string_value.length) {
    return;
  }

  const { font_id, font } = font_from_config(config);
  if (!font) {
    return;
  }

  const text_padding = Number(config.text_padding ?? 0);
  const content_width = Math.max(0, width - text_padding * 2);
  const content_height = Math.max(0, height - text_padding * 2);
  let font_size = content_height;
  let layout = layout_text_glyphs(string_value, font, {
    ...config,
    _resolved_font_size: font_size,
  });

  if (layout.width > 0 && layout.height > 0) {
    const width_scale = content_width > 0 ? content_width / layout.width : 1;
    const height_scale = content_height > 0 ? content_height / layout.height : 1;
    const fit_scale = Math.min(width_scale, height_scale);
    if (Number.isFinite(fit_scale) && fit_scale > 0 && fit_scale !== 1) {
      font_size *= fit_scale;
      layout = layout_text_glyphs(string_value, font, {
        ...config,
        _resolved_font_size: font_size,
      });
    }
  }

  const text_width = layout.width;
  const text_height = layout.height;
  const text_align = config.text_align ?? left;
  const text_valign = config.text_valign ?? top;

  let cursor_x = x + text_padding;
  if (text_align === center) {
    cursor_x = x + (width - text_width) * 0.5;
  } else if (text_align === right) {
    cursor_x = x + width - text_width - text_padding;
  }

  let baseline_y = y + text_padding;
  if (text_valign === middle || text_valign === center) {
    baseline_y = y + (height - text_height) * 0.5;
  } else if (text_valign === bottom) {
    baseline_y = y + height - text_height - text_padding;
  }

  const color = color_to_vec4(config.text_color ?? config.color, [1, 1, 1, 1]);
  const emissive = Number(config.text_emissive ?? config.emissive ?? 1);
  const batch_key = config.batch_key ?? config.ui_batch_key ?? `text_${UI3DContext.command_order}`;

  for (let i = 0; i < layout.glyphs.length; i++) {
    const glyph = layout.glyphs[i];
    const glyph_x = cursor_x + glyph.x;
    const glyph_y = baseline_y + glyph.y;

    if (glyph.width > 0 && glyph.height > 0) {
      const world = local_rect_to_world(root, glyph_x, glyph_y, glyph.width, glyph.height, depth);

      push_command(
        {
          type: UI3DCommandType.Text,
          origin: world.origin,
          x_axis: world.x_axis,
          y_axis: world.y_axis,
          color,
          font_id,
          glyph_index: glyph.glyph_index,
          glyph_width: font.width[glyph.glyph_index],
          glyph_height: font.height[glyph.glyph_index],
          glyph_x: font.x[glyph.glyph_index],
          glyph_y: font.y[glyph.glyph_index],
          page_texture_size: [font.texture_width, font.texture_height],
          font_texture: Name.string(Number(font.page_textures?.[glyph.page] ?? 0)),
          emissive,
          batch_key,
          ...material_command_config(config),
        },
        config,
        inherited_z_order + 0.001
      );
    }
  }
}

/**
 * Clears all immediate 3D UI commands and layout state. Call once at frame start.
 */
export function reset_ui_3d() {
  UI3DContext.commands.length = 0;
  UI3DContext.command_order = 0;
  UI3DContext.layout_allocator.reset();
  UI3DContext.layout_stack.reset();
  UI3DContext.id_counter = 0;
}

export function get_ui_3d_commands() {
  return UI3DContext.commands;
}

export function begin_container(config = {}) {
  config.widget_id = UI3DContext.get_unique_id();

  const parent_index = UI3DContext.layout_stack.peek();
  const parent =
    parent_index !== null
      ? UI3DContext.layout_allocator.get(parent_index.value)
      : {
          x: 0,
          y: 0,
          cursor: { x: 0, y: 0 },
          width: parse_dimension(config.width, 0),
          height: parse_dimension(config.height, 0),
          z_order: 0,
          depth: 0,
          root: null,
        };

  const auto_width = !(width_name in config);
  const auto_height = !(height_name in config);
  const width = auto_width ? 0 : parse_dimension(config.width, parent.width);
  const height = auto_height ? 0 : parse_dimension(config.height, parent.height);

  let offset_x;
  if (config.x !== undefined) {
    offset_x = parse_dimension(config.x, parent.width);
  } else {
    offset_x = parent.auto_width ? 0 : (parent.width - width) / 2;
  }

  let offset_y;
  if (config.y !== undefined) {
    offset_y = parse_dimension(config.y, parent.height);
  } else {
    offset_y = parent.auto_height ? 0 : (parent.height - height) / 2;
  }

  let x;
  if (config.anchor_x === right) {
    x = parent.x + parent.width - offset_x - width;
  } else {
    x = parent.cursor.x + offset_x;
  }

  let y;
  if (config.anchor_y === bottom) {
    y = parent.y + parent.height - offset_y - height;
  } else {
    y = parent.cursor.y + offset_y;
  }

  const root = parent.root ?? clone_root_with_origin(config, width, height);
  const background_depth = parent.root ? parent.depth + root.layer_depth : 0;
  const content_depth = background_depth + root.layer_depth;
  const gap = parse_dimension(config.gap || 0, parent.width);
  const layout = config.layout || absolute;
  const padding_left = config.padding_left || config.padding || 0;
  const padding_top = config.padding_top || config.padding || 0;
  const padding_right = config.padding_right || config.padding || 0;
  const padding_bottom = config.padding_bottom || config.padding || 0;

  const container_index = UI3DContext.layout_stack.push();
  container_index.value = UI3DContext.layout_allocator.length;

  const container = UI3DContext.layout_allocator.allocate();
  container.x = x;
  container.y = y;
  container.width = auto_width ? 0 : width - padding_left - padding_right;
  container.height = auto_height ? 0 : height - padding_top - padding_bottom;
  container.layout = layout;
  container.gap = gap;
  container.padding_left = padding_left;
  container.padding_top = padding_top;
  container.padding_right = padding_right;
  container.padding_bottom = padding_bottom;
  container.root = root;
  container.config = config;
  container.z_order = resolve_z_order(config, parent.z_order ?? 0);
  container.background_depth = background_depth;
  container.depth = content_depth;
  container.auto_width = auto_width;
  container.auto_height = auto_height;
  container.content_max_x = 0;
  container.content_max_y = 0;

  if (layout === row_reversed) {
    container.cursor = { x: x + width - padding_right, y: y + padding_top };
  } else if (layout === column_reversed) {
    container.cursor = { x: x + padding_left, y: y + padding_bottom };
  } else {
    container.cursor = { x: x + padding_left, y: y + padding_top };
  }

  container._base_command_index = UI3DContext.commands.length;
  push_command({ type: UI3DCommandType.Quad, deferred: true }, config, container.z_order);
}

export function end_container() {
  const container_index = UI3DContext.layout_stack.peek();
  const container = UI3DContext.layout_allocator.get(container_index.value);

  if (container.auto_width) {
    container.width = container.content_max_x;
  }
  if (container.auto_height) {
    container.height = container.content_max_y;
  }

  const total_width = container.width + container.padding_left + container.padding_right;
  const total_height = container.height + container.padding_top + container.padding_bottom;
  const border = parse_border(container.config.border);
  const fill_color = color_to_vec4(
    container.config.background_color ?? container.config.color ?? container.config.fill_color,
    [0, 0, 0, 0]
  );
  if (fill_color[3] > 0 || border.width > 0) {
    const world = local_rect_to_world(
      container.root,
      container.x,
      container.y,
      total_width,
      total_height,
      container.background_depth
    );
    set_command(
      container._base_command_index,
      {
        type: UI3DCommandType.Quad,
        origin: world.origin,
        x_axis: world.x_axis,
        y_axis: world.y_axis,
        fill_color,
        border_color: color_to_vec4(container.config.border_color, border.color),
        width: total_width,
        height: total_height,
        corner_radius: Number(container.config.corner_radius ?? container.config.radius ?? 0),
        border_width: border.width,
        emissive: Number(container.config.emissive ?? 0),
      },
      container.config,
      container.z_order
    );
  } else {
    UI3DContext.commands.splice(container._base_command_index, 1);
  }

  UI3DContext.layout_stack.pop();

  const parent_index = UI3DContext.layout_stack.peek();
  if (parent_index !== null) {
    const parent = UI3DContext.layout_allocator.get(parent_index.value);
    child_container_layout_update(parent, container.x, container.y, total_width, total_height);
  }

  return element_handle_input(
    container.root,
    container.x,
    container.y,
    total_width,
    total_height,
    container.config
  );
}

export function panel(config, callback) {
  begin_container(config);
  callback?.();
  return end_container();
}

export function button(label, config = {}) {
  config.widget_id = UI3DContext.get_unique_id();

  const container = get_current_container();
  const rect = resolve_widget_rect(config, container, label);
  const state = element_handle_input(container.root, rect.x, rect.y, rect.width, rect.height, config);

  const draw_config = {
    ...config,
    background_color:
      state.pressed && config.active_color
        ? config.active_color
        : state.hovered && config.hover_color
          ? config.hover_color
          : config.background_color,
  };

  push_quad(
    container.root,
    rect.x,
    rect.y,
    rect.width,
    rect.height,
    draw_config,
    container.z_order,
    container.depth
  );
  push_text(config.text ?? label, container.root, rect.x, rect.y, rect.width, rect.height, {
    text_align: center,
    text_valign: middle,
    ...config,
  }, container.z_order, container.depth + container.root.layer_depth);

  child_container_layout_update(container, rect.x, rect.y, rect.width, rect.height);

  return state;
}

export function label(text, config = {}) {
  config.widget_id = UI3DContext.get_unique_id();

  const container = get_current_container();
  const rect = resolve_widget_rect(config, container, text);

  push_quad(
    container.root,
    rect.x,
    rect.y,
    rect.width,
    rect.height,
    config,
    container.z_order,
    container.depth
  );
  push_text(
    text,
    container.root,
    rect.x,
    rect.y,
    rect.width,
    rect.height,
    config,
    container.z_order,
    container.depth + container.root.layer_depth
  );

  child_container_layout_update(container, rect.x, rect.y, rect.width, rect.height);

  return element_handle_input(container.root, rect.x, rect.y, rect.width, rect.height, config);
}

export function rect(config = {}) {
  config.widget_id = UI3DContext.get_unique_id();

  const container = get_current_container();
  const rect_data = resolve_widget_rect(config, container);

  push_quad(
    container.root,
    rect_data.x,
    rect_data.y,
    rect_data.width,
    rect_data.height,
    config,
    container.z_order,
    container.depth
  );
  child_container_layout_update(container, rect_data.x, rect_data.y, rect_data.width, rect_data.height);

  return element_handle_input(
    container.root,
    rect_data.x,
    rect_data.y,
    rect_data.width,
    rect_data.height,
    config
  );
}

export function input(name, config = {}) {
  if (!UI3DContext.input_state[name]) {
    UI3DContext.input_state[name] = {
      value: config.value || "",
      is_focused: false,
    };
  }

  const field_state = UI3DContext.input_state[name];
  const result = button(field_state.value || config.placeholder || "", {
    text_align: left,
    text_padding: 8,
    ...config,
  });

  if (result.clicked) {
    field_state.is_focused = true;
  }

  return field_state;
}

export class Immediate3DUIUpdater {
  static all_updaters = [];

  constructor() {
    Immediate3DUIUpdater.all_updaters.push(this);
  }

  update(delta_time) {
    throw new Error("Not implemented");
  }

  static update_all(delta_time) {
    profile_scope("Immediate3DUIUpdater.update_all", () => {
      for (let i = 0; i < Immediate3DUIUpdater.all_updaters.length; i++) {
        Immediate3DUIUpdater.all_updaters[i].update(delta_time);
      }
    });
  }

  static remove_of_type(type) {
    for (let i = 0; i < Immediate3DUIUpdater.all_updaters.length; i++) {
      if (Immediate3DUIUpdater.all_updaters[i] instanceof type) {
        Immediate3DUIUpdater.all_updaters.splice(i, 1);
      }
    }
  }
}

export function rotation_from_euler(out, x, y, z) {
  return quat.fromEuler(out, x, y, z);
}

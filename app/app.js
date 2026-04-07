import { quat, vec3 } from "gl-matrix";

import { Scene } from "../engine/src/core/scene.js";
import { Simulator } from "../engine/src/core/simulator.js";
import { LightType } from "../engine/src/core/minimal.js";
import { SharedEnvironmentData, SharedViewBuffer } from "../engine/src/core/shared_data.js";
import { EntityManager } from "../engine/src/core/ecs/entity.js";
import { delete_entity, spawn_mesh_entity } from "../engine/src/core/ecs/entity_utils.js";
import { LightFragment } from "../engine/src/core/ecs/fragments/light_fragment.js";
import { StaticMeshFragment } from "../engine/src/core/ecs/fragments/static_mesh_fragment.js";
import { TransformFragment } from "../engine/src/core/ecs/fragments/transform_fragment.js";
import { MouseCameraControlProcessor } from "./subsystems/mouse_camera_control_processor.js";
import { InputProvider } from "../engine/src/input/input_provider.js";
import { InputKey } from "../engine/src/input/input_types.js";
import { StandardMaterial } from "../engine/src/renderer/material.js";
import { Mesh } from "../engine/src/renderer/mesh.js";
import { MAX_CLIPMAP_LEVELS } from "../engine/src/renderer/shadows/shadow_utils.js";
import * as UI from "../engine/src/ui/2d/immediate.js";
import { world_pos_to_screen_pos } from "../engine/src/utility/camera.js";
import { radians } from "../engine/src/utility/math.js";

import cvar_config from "./config/cvars.js";

const file = (name, size_kb, summary, modified, extra = {}) => ({
  kind: "file",
  name,
  sizeBytes: size_kb * 1024,
  summary,
  modifiedMs: Date.parse(modified),
  path: name,
  extension: name.includes(".") ? name.split(".").pop().toLowerCase() : "file",
  ...extra,
});

const directory = (name, summary, modified, children, extra = {}) => ({
  kind: "directory",
  name,
  summary,
  modifiedMs: Date.parse(modified),
  path: name,
  children,
  ...extra,
});

const FILE_SYSTEM_SANDBOX = directory("Chronicle", "Fallback spatial browser dataset.", "2026-04-06T00:58:00", [
  directory("Projects", "Playable worlds, design spikes, and active prototypes.", "2026-04-06T00:42:00", [
    file("filesystem-vision.md", 18, "Experience pillars for the 3D browser.", "2026-04-05T22:14:00"),
    file("sundown-world-layout.json", 32, "Scene composition notes and layout anchors.", "2026-04-05T21:36:00"),
    directory("Sprint-05", "Tasks, blockers, and interaction targets.", "2026-04-06T00:12:00", [
      file("checklist.txt", 4, "Short operational checklist for this sprint.", "2026-04-05T23:57:00"),
      file("camera-feel.md", 9, "Notes on movement, orbiting, and focus.", "2026-04-05T23:18:00"),
    ]),
  ]),
  directory("Media", "Captured references, previews, and motion snippets.", "2026-04-05T19:17:00", [
    file("launch-teaser.mp4", 164800, "Concept teaser showing folders as glowing towers.", "2026-04-05T18:40:00"),
    file("hover-state.gif", 860, "Selection pulse and hover timing tests.", "2026-04-05T18:18:00"),
  ]),
  file("README.md", 12, "Overview of the project, goals, and controls.", "2026-04-06T00:36:00"),
  file("backlog.csv", 44, "Feature backlog ordered by player value.", "2026-04-05T22:41:00"),
]);

function get_tauri_invoke() {
  const tauri = globalThis.window?.__TAURI__;
  if (tauri?.core?.invoke) return (command, args = {}) => tauri.core.invoke(command, args);
  if (tauri?.invoke) return (command, args = {}) => tauri.invoke(command, args);
  if (globalThis.window?.__TAURI_INTERNALS__?.invoke) {
    return (command, args = {}) => globalThis.window.__TAURI_INTERNALS__.invoke(command, args);
  }
  return null;
}

function format_bytes(bytes) {
  const value = Math.max(0, bytes ?? 0);
  if (value >= 1024 * 1024 * 1024) return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  if (value >= 1024) return `${Math.max(1, Math.round(value / 1024))} KB`;
  return `${value} B`;
}

function format_timestamp(modified_ms) {
  if (!modified_ms) return "Unknown";
  try {
    return new Date(modified_ms).toLocaleString();
  } catch {
    return "Unknown";
  }
}

function normalize_entry(entry) {
  return {
    kind: entry.kind,
    name: entry.name,
    path: entry.path,
    sizeBytes: entry.sizeBytes ?? entry.size_bytes ?? 0,
    modifiedMs: entry.modifiedMs ?? entry.modified_ms ?? null,
    extension: entry.extension ?? (entry.kind === "file" ? "file" : null),
    childCount: entry.childCount ?? entry.child_count ?? null,
    summary: entry.summary ?? null,
    children: entry.children ?? [],
  };
}

function normalize_directory_listing(listing) {
  return {
    kind: listing.kind ?? "directory",
    name: listing.name ?? "Computer",
    path: listing.path ?? null,
    parentPath: listing.parentPath ?? listing.parent_path ?? null,
    modifiedMs: listing.modifiedMs ?? listing.modified_ms ?? null,
    virtualRoot: listing.virtualRoot ?? listing.virtual_root ?? false,
    children: (listing.children ?? []).map(normalize_entry),
  };
}

function annotate_tree(node, parent_path = null) {
  const path = parent_path ? `${parent_path}/${node.name}` : node.name;
  if (node.kind === "directory") {
    return {
      ...node,
      path,
      childCount: node.children.length,
      children: node.children.map((child) => annotate_tree(child, path)),
    };
  }
  return { ...node, path, childCount: 0, children: [] };
}

function path_label(directory) {
  if (!directory || directory.virtualRoot || !directory.path) return "Computer";
  return directory.path;
}

function hash_string(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i++) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

class SandboxFilesystemProvider {
  mode = "demo";
  isLive = false;

  constructor(root) {
    this.root = annotate_tree(root);
    this.index = new Map();
    this.index_tree(this.root);
  }

  index_tree(node) {
    this.index.set(node.path, node);
    for (const child of node.children ?? []) this.index_tree(child);
  }

  async listDirectory(path) {
    if (!path) {
      return normalize_directory_listing({
        kind: "directory",
        name: "Computer",
        path: null,
        parentPath: null,
        virtualRoot: true,
        children: [{ kind: "directory", name: this.root.name, path: this.root.path, childCount: this.root.childCount }],
      });
    }

    const node = this.index.get(path);
    if (!node || node.kind !== "directory") throw new Error(`Demo directory not found: ${path}`);

    return normalize_directory_listing({
      kind: "directory",
      name: node.name,
      path: node.path,
      parentPath: node.path.includes("/") ? node.path.split("/").slice(0, -1).join("/") : null,
      virtualRoot: false,
      modifiedMs: node.modifiedMs,
      children: node.children,
    });
  }

  async openPath() {
    return false;
  }
}

class DesktopFilesystemProvider {
  mode = "desktop";
  isLive = true;

  constructor(invoke) {
    this.invoke = invoke;
  }

  async listDirectory(path) {
    return normalize_directory_listing(await this.invoke("filesystem_list_directory", { path }));
  }

  async openPath(path) {
    await this.invoke("filesystem_open_path", { path });
  }
}

function create_filesystem_provider() {
  const invoke = get_tauri_invoke();
  return invoke ? new DesktopFilesystemProvider(invoke) : new SandboxFilesystemProvider(FILE_SYSTEM_SANDBOX);
}

class FileSystemScene extends Scene {
  name = "FileSystemScene";
  provider = create_filesystem_provider();
  scene_entities = [];
  entry_slots = [];
  current_directory = null;
  focused_index = -1;
  loading = false;
  move_mode = false;
  load_token = 0;
  last_error = null;
  status_message = "Fly through the node field and hover a file or folder.";
  layout_cache = new Map();

  init() {
    super.init();

    const controls = this.add_layer(MouseCameraControlProcessor);
    controls.pivot_distance = 18.0;
    controls.set_scene(this);

    this.cube_mesh = Mesh.cube();
    this.materials = this.create_materials();

    const view_data = SharedViewBuffer.get_view_data(this.context.current_view);
    view_data.view_position = [0, 2, 28, 1];
    view_data.view_rotation = quat.fromEuler(quat.create(), 0, 180, 0);
    view_data.far = 1000.0;
    view_data.fov = radians(15);

    SharedEnvironmentData.set_skybox("default_scene_skybox", [
      "engine/textures/simple_skybox/px.jpg",
      "engine/textures/simple_skybox/nx.jpg",
      "engine/textures/simple_skybox/ny.jpg",
      "engine/textures/simple_skybox/py.jpg",
      "engine/textures/simple_skybox/pz.jpg",
      "engine/textures/simple_skybox/nz.jpg",
    ]);
    SharedEnvironmentData.set_skybox_color([0.50, 0.67, 0.93, 1.0]);

    this.spawn_lights();

    this.load_directory(null);
  }

  cleanup() {
    this.clear_directory_entities();
    for (const entity of this.scene_entities) delete_entity(entity);
    this.scene_entities.length = 0;
    this.remove_layer(MouseCameraControlProcessor);
    super.cleanup();
  }

  update(delta_time) {
    super.update(delta_time);

    this.update_focus_from_cursor();
    this.handle_input();
    this.render_ui();
  }

  create_materials() {
    return {
      folder: StandardMaterial.create("chronicle_folder_node", {
        albedo: [0.24, 0.72, 0.92, 1],
        roughness: 0.28,
        metallic: 0.35,
        emission: 0.6,
      }).material_id,
      folder_focus: StandardMaterial.create("chronicle_folder_focus", {
        albedo: [0.74, 0.96, 1.0, 1],
        roughness: 0.12,
        metallic: 0.55,
        emission: 4.4,
      }).material_id,
      file: StandardMaterial.create("chronicle_file_node", {
        albedo: [0.90, 0.54, 0.92, 1],
        roughness: 0.24,
        metallic: 0.18,
        emission: 0.35,
      }).material_id,
      file_focus: StandardMaterial.create("chronicle_file_focus", {
        albedo: [1.0, 0.84, 1.0, 1],
        roughness: 0.14,
        metallic: 0.28,
        emission: 3.8,
      }).material_id,
      mover: StandardMaterial.create("chronicle_move_focus", {
        albedo: [1.0, 0.78, 0.46, 1],
        roughness: 0.12,
        metallic: 0.48,
        emission: 6.0,
      }).material_id,
    };
  }

  spawn_lights() {
    const sun = EntityManager.create_entity([LightFragment]);
    const sun_view = EntityManager.get_fragment(sun, LightFragment);
    sun_view.type = LightType.DIRECTIONAL;
    sun_view.color = [1.0, 0.97, 0.93, 1.0];
    sun_view.intensity = 5.5;
    sun_view.position = [35.0, 60.0, 15.0, 1.0];
    sun_view.active = true;
    sun_view.is_primary_sun = 1;
    sun_view.shadow_clipmaps = MAX_CLIPMAP_LEVELS;
    this.scene_entities.push(sun);

    const fill = EntityManager.create_entity([LightFragment]);
    const fill_view = EntityManager.get_fragment(fill, LightFragment);
    fill_view.type = LightType.DIRECTIONAL;
    fill_view.color = [0.40, 0.54, 0.90, 1.0];
    fill_view.intensity = 1.4;
    fill_view.position = [-24.0, 26.0, -14.0, 1.0];
    fill_view.active = true;
    fill_view.shadow_casting = 0;
    this.scene_entities.push(fill);
  }

  current_directory_key() {
    return this.current_directory?.path ?? "__computer__";
  }

  get_layout_bucket() {
    const key = this.current_directory_key();
    if (!this.layout_cache.has(key)) this.layout_cache.set(key, new Map());
    return this.layout_cache.get(key);
  }

  ensure_layout_positions() {
    const bucket = this.get_layout_bucket();
    const items = this.current_directory?.children ?? [];

    for (let index = 0; index < items.length; index++) {
      const item = items[index];
      if (bucket.has(item.path)) continue;

      const hash = hash_string(item.path ?? item.name);
      const angle = index * 1.618 + (hash % 360) * 0.01745;
      const radius = 9 + (index % 4) * 3.5 + (hash % 7) * 0.35;
      const height = ((hash % 11) - 5) * 1.9 + (item.kind === "directory" ? 1.5 : 0);
      bucket.set(item.path, [Math.cos(angle) * radius, height, Math.sin(angle) * radius]);
    }
  }

  clear_directory_entities() {
    for (let i = 0; i < this.entry_slots.length; i++) {
      const slot = this.entry_slots[i];
      delete_entity(slot.entity.id);
    }
    this.entry_slots.length = 0;
    this.focused_index = -1;
  }

  rebuild_directory_entities() {
    this.clear_directory_entities();
    
    if (!this.current_directory) return;

    this.ensure_layout_positions();
    const bucket = this.get_layout_bucket();

    for (const item of this.current_directory.children) {
      const layout_position = bucket.get(item.path);
      const size_seed = Math.max(0.9, Math.min(3.6, Math.log2((item.sizeBytes ?? 0) / 1024 + 2)));
      const base_scale =
        item.kind === "directory"
          ? [1.8, 1.9 + (item.childCount ?? 0) * 0.06, 1.8]
          : [1.1, 0.9 + size_seed * 0.22, 1.1];

      const entity = spawn_mesh_entity(layout_position, quat.create(), base_scale, this.cube_mesh, this.materials.file);
      this.entry_slots.push({ item, entity, base_position: [...layout_position], base_scale });
    }
  }

  refresh_focus_materials() {
    for (let index = 0; index < this.entry_slots.length; index++) {
      const slot = this.entry_slots[index];
      const mesh = EntityManager.get_fragment(slot.entity, StaticMeshFragment);
      const focused = index === this.focused_index;
      const active_material = this.move_mode && focused
        ? this.materials.mover
        : slot.item.kind === "directory"
          ? focused ? this.materials.folder_focus : this.materials.folder
          : focused ? this.materials.file_focus : this.materials.file;
      mesh.material_slots = [BigInt(active_material)];
    }
  }

  update_focus_from_cursor() {
    if (!this.current_directory || this.entry_slots.length === 0 || this.move_mode) return;

    const view_data = SharedViewBuffer.get_view_data(this.context.current_view);
    const camera_position = vec3.fromValues(view_data.view_position[0], view_data.view_position[1], view_data.view_position[2]);
    const { width, height } = UI.UIContext.canvas_size;
    const cursor_x = UI.UIContext.input_state.x;
    const cursor_y = UI.UIContext.input_state.y;
    const max_cursor_distance = Math.max(48, Math.min(width, height) * 0.075);
    const max_cursor_distance_sq = max_cursor_distance * max_cursor_distance;

    let best_index = -1;
    let best_distance_sq = Infinity;
    let best_depth = Infinity;

    for (let index = 0; index < this.entry_slots.length; index++) {
      const slot = this.entry_slots[index];
      const transform = EntityManager.get_fragment(slot.entity, TransformFragment);
      const screen_position = world_pos_to_screen_pos(view_data, transform.position, width, height);
      if (!screen_position?.is_visible) continue;

      const dx = screen_position.x - cursor_x;
      const dy = screen_position.y - cursor_y;
      const distance_sq = dx * dx + dy * dy;
      if (distance_sq > max_cursor_distance_sq) continue;

      const depth = vec3.distance(camera_position, transform.position);
      if (
        distance_sq < best_distance_sq ||
        (Math.abs(distance_sq - best_distance_sq) < 1.0 && depth < best_depth)
      ) {
        best_distance_sq = distance_sq;
        best_depth = depth;
        best_index = index;
      }
    }

    if (best_index !== this.focused_index) {
      this.focused_index = best_index;
      this.refresh_focus_materials();
    }
  }

  get_focused_slot() {
    return this.focused_index >= 0 ? this.entry_slots[this.focused_index] : null;
  }

  async load_directory(path, preferred_focus_path = null) {
    const token = ++this.load_token;
    this.loading = true;
    this.move_mode = false;
    this.last_error = null;
    this.status_message = path ? `Scanning ${path}...` : "Scanning the root directory hub...";

    try {
      const listing = await this.provider.listDirectory(path);
      if (token !== this.load_token) return;

      this.current_directory = listing;
      this.rebuild_directory_entities();

      if (preferred_focus_path) {
        this.focused_index = this.entry_slots.findIndex((slot) => slot.item.path === preferred_focus_path);
      } else {
        this.focused_index = this.entry_slots.length > 0 ? 0 : -1;
      }

      this.refresh_focus_materials();
      this.status_message = listing.virtualRoot
        ? "Root hub loaded. Hover a node and press Enter."
        : `${listing.name} opened into free space.`;
    } catch (error) {
      if (token !== this.load_token) return;
      this.current_directory = normalize_directory_listing({ name: "Unavailable", path, children: [], virtualRoot: !path });
      this.clear_directory_entities();
      this.last_error = String(error);
      this.status_message = "Directory scan failed.";
    } finally {
      if (token === this.load_token) this.loading = false;
    }
  }

  async activate_focused_item() {
    const slot = this.get_focused_slot();
    if (!slot) return;

    if (slot.item.kind === "directory") {
      await this.load_directory(slot.item.path);
    } else {
      try {
        await this.provider.openPath(slot.item.path);
        this.status_message = `${slot.item.name} opened with the operating system.`;
      } catch (error) {
        this.last_error = String(error);
        this.status_message = `Open failed for ${slot.item.name}.`;
      }
    }
  }

  async go_up() {
    if (!this.current_directory) return;
    if (this.current_directory.parentPath) {
      await this.load_directory(this.current_directory.parentPath, this.current_directory.path);
      return;
    }
    if (!this.current_directory.virtualRoot) {
      await this.load_directory(null, this.current_directory.path);
      return;
    }
    this.status_message = "You are already at the root directory hub.";
  }

  apply_move_controls(delta_scale = 0.4) {
    if (!this.move_mode) return;
    const slot = this.get_focused_slot();
    if (!slot) return;

    let moved = false;
    const step = delta_scale;
    if (InputProvider.get_state(InputKey.K_j)) {
      slot.base_position[0] -= step;
      moved = true;
    }
    if (InputProvider.get_state(InputKey.K_l)) {
      slot.base_position[0] += step;
      moved = true;
    }
    if (InputProvider.get_state(InputKey.K_i)) {
      slot.base_position[2] -= step;
      moved = true;
    }
    if (InputProvider.get_state(InputKey.K_k)) {
      slot.base_position[2] += step;
      moved = true;
    }
    if (InputProvider.get_state(InputKey.K_u)) {
      slot.base_position[1] += step;
      moved = true;
    }
    if (InputProvider.get_state(InputKey.K_o)) {
      slot.base_position[1] -= step;
      moved = true;
    }

    if (moved) {
      this.get_layout_bucket().set(slot.item.path, [...slot.base_position]);
    }
  }

  handle_input() {
    if (this.loading) return;

    this.apply_move_controls();

    if (InputProvider.get_action(InputKey.B_mouse_left)) {
      this.activate_focused_item();
    }
    if (InputProvider.get_action(InputKey.K_Backspace)) {
      this.go_up();
    }
    if (InputProvider.get_action(InputKey.K_h)) {
      this.load_directory(null);
    }
    if (InputProvider.get_action(InputKey.K_r)) {
      this.load_directory(this.current_directory?.path ?? null);
    }
    if (InputProvider.get_action(InputKey.K_g)) {
      this.move_mode = !this.move_mode && this.focused_index >= 0;
      this.status_message = this.move_mode
        ? "Move mode active. Use I/J/K/L/U/O to reposition the focused node."
        : "Move mode released.";
      this.refresh_focus_materials();
    }
  }

  render_ui() {
    const focused = this.get_focused_slot();
    const focused_item = focused?.item ?? null;

    const { width, height } = UI.UIContext.canvas_size;

    UI.panel(
      {
        x: 18,
        y: 18,
        width: 360,
        padding: 14,
        layout: "column",
        gap: 6,
        background_color: "rgba(7, 11, 20, 0.38)",
        border: "1px solid rgba(255, 255, 255, 0.14)",
        corner_radius: 14,
      },
      () => {
        UI.label(path_label(this.current_directory), { font: '600 15px "Trebuchet MS"', text_color: "#f2f6ff", width: "100%", height: "fit-content", wrap: true });
        UI.label(
          focused_item
            ? `${focused_item.name} | ${focused_item.kind === "directory" ? "Directory" : format_bytes(focused_item.sizeBytes)} | ${format_timestamp(focused_item.modifiedMs)}`
            : "Hover a node to focus it.",
          { font: '13px "Trebuchet MS"', text_color: "#dce9ff", width: "100%", height: "fit-content", wrap: true }
        );
        UI.label(this.status_message, {
          font: '12px "Trebuchet MS"',
          text_color: this.last_error ? "#ffd6a3" : "#a5ffd1",
          width: "100%",
          height: "fit-content",
          wrap: true,
        });
      }
    );

    UI.panel(
      {
        x: 18,
        y: height - 50,
        width: Math.min(980, width - 36),
        height: 34,
        padding_left: 14,
        padding_top: 9,
        background_color: "rgba(7, 11, 20, 0.34)",
        border: "1px solid rgba(255, 255, 255, 0.14)",
        corner_radius: 999,
      },
      () => {
        UI.label(
          "RMB drag: camera XZ pan | LMB drag: camera XY pan | Shift+LMB drag: orbit | Wheel zoom | Hover to focus | Enter open | Backspace up | G move mode | I/J/K/L/U/O reposition | H root | R refresh",
          { font: '12px "Trebuchet MS"', text_color: "#eef4ff", width: "100%", height: "fit-content" }
        );
      }
    );
  }
}

(async () => {
  const simulator = await Simulator.create("gpu-canvas", "ui-canvas", {
    pointer_lock: false,
    project: {
      name: "Chronicle",
      root: "app",
      cvar_config,
    },
  });

  simulator.add_sim_layer(new FileSystemScene("FileSystemScene"));

  simulator.run();
})();

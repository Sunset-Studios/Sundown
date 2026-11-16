// ═══════════════════════════════════════════════════════════════════════════════
// Application-owned smoke catalogs — Node-safe IDs, independent of scene modules.
// ═══════════════════════════════════════════════════════════════════════════════

import path from "node:path";
import { pathToFileURL } from "node:url";

export function select_scenes(scenes, filters = []) {
  const names = filters.flatMap((filter) => filter.split(",")).map((name) => name.trim());
  if (!names.length) return scenes.slice();

  const selected = new Set();
  for (const name of names) {
    const normalized = name.replace(/[_-]/g, "").toLowerCase();
    const scene = scenes.find(
      (entry) => entry.id.replace(/[_-]/g, "").toLowerCase() === normalized
    );
    if (!scene)
      throw new Error(
        `Unknown scene '${name}'. Use --list with --catalog to see available scenes.`
      );
    selected.add(scene);
  }
  return scenes.filter((scene) => selected.has(scene));
}

export async function load_scene_catalog(catalog_path, filters = []) {
  let scenes;
  if (catalog_path) {
    const catalog_url = pathToFileURL(path.resolve(catalog_path));
    const catalog = await import(catalog_url.href);
    scenes = catalog.default;
  } else {
    const names = filters.flatMap((filter) => filter.split(",")).map((name) => name.trim());
    scenes = [...new Set(names)].map((id) => ({ id }));
  }

  if (!Array.isArray(scenes) || !scenes.length) {
    throw new Error(
      "Provide --catalog <module> exporting a nonempty default array of scene IDs, or --scene <id>."
    );
  }
  const ids = new Set();
  for (const scene of scenes) {
    if (
      typeof scene?.id !== "string" ||
      !scene.id.trim() ||
      scene.id !== scene.id.trim() ||
      scene.id.includes(",")
    ) {
      throw new Error(
        "Smoke catalog entries must have nonempty, trimmed string IDs without commas."
      );
    }
    const normalized = scene.id.replace(/[_-]/g, "").toLowerCase();
    if (ids.has(normalized)) throw new Error(`Duplicate smoke scene ID '${scene.id}'.`);
    ids.add(normalized);
  }
  return scenes;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared sample catalog — safe to import in browser launchers and Node tools.
// Scene modules are loaded only when selected; durable IDs retain saved data.
// ═══════════════════════════════════════════════════════════════════════════════

export const sample_scenes = [
  { id: "SolarECSTestScene", load: () => import("./solar_ecs_test_scene.js") },
  { id: "RenderingScene", load: () => import("./rendering_scene.js") },
  { id: "BVHScene", load: () => import("./bvh_scene.js") },
  { id: "MLScene", load: () => import("./ml_scene.js") },
  { id: "VoxelTerrainScene", load: () => import("./voxel_terrain_scene.js") },
  { id: "ObjectPaintingScene", load: () => import("./object_painting_scene.js") },
  { id: "GLTFModelScene", load: () => import("./gltf_model_scene.js") },
  { id: "TexturesScene", load: () => import("./textures_scene.js") },
  { id: "GITestScene", load: () => import("./gi_test_scene.js") },
  { id: "ShadowTestScene", load: () => import("./shadow_test_scene.js") },
  { id: "UI3DTestScene", load: () => import("./ui_3d_test_scene.js") },
  { id: "SponzaScene", load: () => import("./sponza_scene.js") },
  { id: "OfficeTestScene", load: () => import("./office_test_scene.js") },
  { id: "LivingRoomScene", load: () => import("./living_room_scene.js") },
  { id: "BackroomsScene", load: () => import("./backrooms_scene.js") },
  { id: "CityScene", load: () => import("./city_scene.js") },
  { id: "SciFiCityScene", load: () => import("./scifi_city_scene.js") },
];

export function select_sample_scenes(filters = []) {
  const names = filters.flatMap((filter) => filter.split(",")).map((name) => name.trim());
  if (!names.length) return sample_scenes.slice();
  const selected = new Set();
  for (const name of names) {
    const normalized = name.replace(/[_-]/g, "").toLowerCase();
    const sample = sample_scenes.find((entry) => entry.id.toLowerCase() === normalized);
    if (!sample)
      throw new Error(`Unknown sample scene '${name}'. Use --list to see available scenes.`);
    selected.add(sample);
  }
  return sample_scenes.filter((sample) => selected.has(sample));
}

export async function create_sample_scene(sample) {
  const module = await sample.load();
  const scene_class = module[sample.id];
  if (typeof scene_class !== "function")
    throw new Error(`Sample '${sample.id}' does not export its scene class`);
  return new scene_class(sample.id);
}

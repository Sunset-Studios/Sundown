import { MaterialAllocationTable } from "./material_allocation_table.js";
import { SharedEnvironmentData } from "../core/shared_data.js";
import { EntityManager } from "../core/ecs/entity.js";
import { StaticMeshFragment } from "../core/ecs/fragments/static_mesh_fragment.js";
import { Name } from "../utility/names.js";
import { ResourceCache } from "./resource_cache.js";
import { CacheTypes } from "./renderer_types.js";
import { Texture } from "./texture.js";

const material_offsets_name = "material_table_offset";

const default_texture_pool_names = {
  albedo: Name.from("texture_pool_albedo"),
  normal: Name.from("texture_pool_normal"),
  roughness: Name.from("texture_pool_roughness"),
  metallic: Name.from("texture_pool_metallic"),
  ao: Name.from("texture_pool_ao"),
  height: Name.from("texture_pool_height"),
  specular: Name.from("texture_pool_specular"),
  emission: Name.from("texture_pool_emission"),
};

const material_rg_buffers = {
  params_gpu_buffer: null,
  material_palette_buffer: null,
  material_offsets_buffer: null
}

const scene_lighting_data = {
  scene_lighting_buffer: null,
  skybox_image: null
}

export function register_material_buffers(render_graph) {
  material_rg_buffers.params_gpu_buffer = render_graph.register_buffer(
    MaterialAllocationTable.params_buffer.config.name
  );
  material_rg_buffers.material_palette_buffer = render_graph.register_buffer(
    MaterialAllocationTable.palette_buffer.config.name
  );
  material_rg_buffers.material_offsets = EntityManager.get_fragment_gpu_buffer(
    StaticMeshFragment,
    material_offsets_name
  );
  material_rg_buffers.material_offsets_buffer = render_graph.register_buffer(
    material_rg_buffers.material_offsets.buffer.config.name
  );

  return material_rg_buffers;
}

export function register_texture_pools(render_graph) {
  const default_texture = render_graph.register_image(Texture.default_array().config.name);
  return Object.fromEntries(
    Object.entries(default_texture_pool_names).map(([key, name]) => {
      const texture = ResourceCache.get().fetch(CacheTypes.IMAGE, name);
      return [key, texture ? render_graph.register_image(texture.config.name) : default_texture];
    })
  );
}

export function register_scene_lighting_data(render_graph) {
  scene_lighting_data.scene_lighting_buffer = render_graph.register_buffer(
    SharedEnvironmentData.get_skydome_data().config.name
  );
  scene_lighting_data.skybox_image = render_graph.register_image(
    SharedEnvironmentData.get_skybox().config.name
  );

  return scene_lighting_data;
}
import { Renderer } from "./renderer.js";
import { Shader } from "./shader.js";
import { EntityManager } from "../core/ecs/entity.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { BindGroup } from "./bind_group.js";
import { PipelineState } from "./pipeline_state.js";
import { ResourceCache } from "./resource_cache.js";
import { profile_scope } from "../utility/performance.js";
import { hash_data_map, hash_value } from "../utility/hashing.js";
import { Name } from "../utility/names.js";
import { Texture } from "./texture.js";
import { UserInterfaceFragment } from "../core/ecs/fragments/user_interface_fragment.js";
import { StaticMeshFragment } from "../core/ecs/fragments/static_mesh_fragment.js";
import { global_dispatcher } from "../core/dispatcher.js";
import {
  ShaderResourceType,
  MaterialFamilyType,
  MaterialPassType,
  CacheTypes,
  BindGroupType,
  TextureChannel,
} from "./renderer_types.js";
import { MaterialAllocationTable, MATERIAL_PARAMS_SIZE } from "./material_allocation_table.js";

const material_offsets_name = "material_table_offset";

function reflect_resources(reflection) {
  const result = [];
  const groups = reflection ? reflection.get_bind_groups() : [];
  if (BindGroupType.Material < groups.length) {
    const material_group = groups[BindGroupType.Material];
    for (let i = 0; i < material_group.length; i++) {
      const binding = material_group[i];
      const binding_type = Shader.resource_type_from_reflection_type(binding.resourceType);
      result.push({
        type: binding_type,
        name: binding.name,
        binding: i,
        is_array: binding.type.name.includes("array"),
      });
    }
  }
  return result;
};

export class MaterialTemplate {
  static templates = new Map();

  name = null;
  shader = null;
  depth_shader = null;
  resolve_shader = null;
  pipeline_state_config = null;
  resources = [];
  depth_resources = [];
  resolve_resources = [];
  parent = null;
  family = null;
  pipeline_state = null;
  depth_pipeline_state = null;
  resolve_pipeline_state = null;

  constructor(
    name,
    shader,
    depth_shader = null,
    resolve_shader = null,
    family = MaterialFamilyType.Opaque,
    pipeline_state_config = {},
    parent = null
  ) {
    this.name = name;
    this.shader = shader;
    this.depth_shader = depth_shader;
    this.resolve_shader = resolve_shader;
    this.pipeline_state_config = pipeline_state_config;
    this.parent = parent;
    this.family = family;
  }

  add_resource(resource) {
    this.resources.push(resource);
  }

  get base_reflection() {
    return this.shader?.reflection ?? null;
  }

  get depth_reflection() {
    return this.depth_shader?.reflection ?? null;
  }

  get resolve_reflection() {
    return this.resolve_shader?.reflection ?? null;
  }

  static create(
    name,
    shader_path,
    family = MaterialFamilyType.Opaque,
    pipeline_state_config = {},
    parent_name = null,
    defines = {}
  ) {
    const key = `${name}-${family}`;
    if (this.templates.has(key)) {
      return this.templates.get(key);
    }

    let parent = null;
    let shader = null;
    let depth_shader = null;
    let resolve_shader = null;

    if (parent_name) {
      parent = this.get_template(parent_name);
      if (!parent) {
        throw new Error(`Parent template '${parent_name}' not found`);
      }
      shader = parent.shader;
      depth_shader = parent.depth_shader;
      resolve_shader = parent.resolve_shader;
    }

    if (family === MaterialFamilyType.Transparent) {
      defines["TRANSPARENT"] = true;
    }

    if (shader_path) {
      shader = ResourceCache.get().fetch(
        CacheTypes.SHADER,
        Shader.create(shader_path, { ...defines, MESHLET_RASTER_PASS: true })
      );
      depth_shader = ResourceCache.get().fetch(
        CacheTypes.SHADER,
        Shader.create(shader_path, { ...defines, MESHLET_DEPTH_PASS: true })
      );
      resolve_shader = ResourceCache.get().fetch(
        CacheTypes.SHADER,
        Shader.create(shader_path, { ...defines, MESHLET_RESOLVE_PASS: true })
      );
    }

    const template = new MaterialTemplate(
      key,
      shader,
      depth_shader,
      resolve_shader,
      family,
      pipeline_state_config,
      parent
    );

    if (parent) {
      template.resources = [...parent.resources];
      template.depth_resources = [...parent.depth_resources];
      template.resolve_resources = [...parent.resolve_resources];
    }

    template.resources.push(...reflect_resources(template.base_reflection));
    template.depth_resources.push(...reflect_resources(template.depth_reflection));
    template.resolve_resources.push(...reflect_resources(template.resolve_reflection));

    this.templates.set(key, template);

    return template;
  }

  create_pipeline_state(
    bind_group_layouts,
    output_targets = [],
    depth_stencil_options = {},
    pass_type = MaterialPassType.Raster
  ) {
    const pass_suffix = pass_type === MaterialPassType.Depth
      ? "_depth"
      : pass_type === MaterialPassType.Resolve
        ? "_resolve"
        : "_raster";
    const pipeline_name = `${this.name}${pass_suffix}`;
    const shader_module = pass_type === MaterialPassType.Depth
      ? this.depth_shader.module
      : pass_type === MaterialPassType.Resolve
        ? this.resolve_shader.module
        : this.shader.module;
    let all_bind_group_layouts = [...bind_group_layouts];
    let ref = pass_type === MaterialPassType.Depth
      ? this.depth_reflection
      : pass_type === MaterialPassType.Resolve
        ? this.resolve_reflection
        : this.base_reflection;

    // Set material binding group inputs for groups not already covered by provided layouts
    const groups = ref.get_bind_groups();
    if (all_bind_group_layouts.length < groups.length) {
      for (let i = all_bind_group_layouts.length; i < groups.length; i++) {
        const bind_group = groups[i];

        all_bind_group_layouts.push(
          BindGroup.create_layout(
            `${pipeline_name}_group_${i}`,
            bind_group.map((binding) => {
              let binding_obj = {};
              const binding_type = Shader.resource_type_from_reflection_type(binding.resourceType);
              switch (binding_type) {
                case ShaderResourceType.Uniform:
                  binding_obj = {
                    buffer: {
                      type: "uniform",
                    },
                  };
                  break;
                case ShaderResourceType.Storage:
                  binding_obj = {
                    buffer: {
                      type: binding.access === "read" ? "read-only-storage" : "storage",
                    },
                  };
                  break;
                case ShaderResourceType.Texture:
                  binding_obj = {
                    texture: {
                      viewDimension: Texture.dimension_from_type_name(binding.type.name),
                      sampleType: Texture.filter_type_from_binding_format(binding.type.format.name),
                    },
                  };
                  break;
                case ShaderResourceType.StorageTexture:
                  binding_obj = {
                    storageTexture: {
                      access:
                        binding.type.access === "write"
                          ? "write-only"
                          : binding.type.access === "read"
                            ? "read-only"
                            : "read-write",
                      viewDimension: Texture.dimension_from_type_name(binding.type.name),
                      sampleType: "float",
                      format: Shader.get_optimal_texture_format(binding.type.name),
                    },
                  };
                  break;
                case ShaderResourceType.Sampler:
                  binding_obj = {
                    sampler: {},
                  };
                  break;
              }

              return {
                binding: binding.binding,
                visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
                ...binding_obj,
              };
            }),
            true /* force */
          )
        );
      }
    }

    // Set material shader fragment output targets
    const fragment_outputs = ref.entry.fragment[0]?.outputs ?? [];
    const non_depth_attachments = output_targets.filter((target) => target.config.type !== "depth");
    const output_attachments = fragment_outputs.length > 0 && non_depth_attachments.length > fragment_outputs.length
      ? non_depth_attachments.slice(-fragment_outputs.length)
      : non_depth_attachments;

    const targets = output_attachments.map((target) => {
      let t = {
        name: target.config.name,
        format: target.config.format,
      };
      if (target.config.blend) {
        t.blend = target.config.blend;
      }
      return t;
    });

    fragment_outputs.forEach((output, i) => {
      if (targets[i]) {
        return;
      }

      const target = {
        format: Shader.get_optimal_texture_format(
          output.type.format ? output.type.format.name : output.type.name
        ),
      };

      if (this.pipeline_state_config.targets && i < this.pipeline_state_config.targets.length) {
        if (this.pipeline_state_config.targets[i].format) {
          target.format = this.pipeline_state_config.targets[i].format;
        }
        if (this.pipeline_state_config.targets[i].blend) {
          target.blend = this.pipeline_state_config.targets[i].blend;
        }
      }

      targets.push(target);
    });

    // Create pipeline state based on shader, pipeline state config and targets
    let pipeline_descriptor = {
      label: pipeline_name,
      bind_layouts: all_bind_group_layouts.filter((layout) => layout !== null),
      vertex: {
        module: shader_module,
        entryPoint:
          (this.pipeline_state_config.vertex && this.pipeline_state_config.vertex.entry_point) ||
          "vs",
        buffers: [],
      },
      fragment: {
        module: shader_module,
        entryPoint:
          (this.pipeline_state_config.fragment &&
            this.pipeline_state_config.fragment.entry_point) ||
          "fs",
        targets: targets,
      },
      primitive: {
        topology: this.pipeline_state_config.primitive_topology_type || "triangle-list",
        cullMode: this.pipeline_state_config.rasterizer_state?.cull_mode || "back",
      },
    };

    const depth_target = output_targets.find((target) => target.config.type === "depth");
    if (this.pipeline_state_config.depth_stencil_target) {
      pipeline_descriptor.depthStencil = {
        format: this.pipeline_state_config.depth_stencil_target.format ?? "depth32float",
        depthWriteEnabled:
          this.pipeline_state_config.depth_stencil_target.depth_write_enabled ??
          depth_stencil_options.depth_write_enabled ??
          false,
        depthCompare:
          this.pipeline_state_config.depth_stencil_target.depth_compare ??
          depth_stencil_options.depth_compare ??
          "less-equal",
        depthBias: this.pipeline_state_config.depth_stencil_target.depth_bias ?? 0.0,
        depthBiasClamp: this.pipeline_state_config.depth_stencil_target.depth_bias_clamp ?? 0.0,
        depthBiasSlopeScale:
          this.pipeline_state_config.depth_stencil_target.depth_slope_scale ?? 0.0,
      };
    } else if (depth_target && depth_stencil_options) {
      pipeline_descriptor.depthStencil = {
        format: depth_target.config.format,
        depthWriteEnabled: depth_stencil_options.depth_write_enabled ?? false,
        depthCompare: depth_stencil_options.depth_compare ?? "less-equal",
        depthBias: depth_stencil_options.depth_bias ?? 0.0,
        depthBiasClamp: depth_stencil_options.depth_bias_clamp ?? 0.0,
        depthBiasSlopeScale: depth_stencil_options.depth_slope_scale ?? 0.0,
      };
    }

    return PipelineState.create_render(pipeline_name, pipeline_descriptor);
  }

  static get_template(name, family = MaterialFamilyType.Opaque) {
    const key = `${name}-${family}`;
    return this.templates.get(key);
  }

  get_all_resources() {
    if (this.parent) {
      return [...this.parent.get_all_resources(), ...this.resources];
    }
    return this.resources;
  }

  get_resources_for_pass(pass_type) {
    if (pass_type === MaterialPassType.Depth) {
      return this.depth_resources.length > 0 ? this.depth_resources : this.resources;
    }
    if (pass_type === MaterialPassType.Resolve) {
      return this.resolve_resources.length > 0 ? this.resolve_resources : this.resources;
    }
    return this.resources;
  }
}

export class Material {
  static materials = new Map();

  constructor(name, template, parent_id = null) {
    this.name = name;
    this.template = template;
    this.pipeline_state = null;
    this.depth_pipeline_state = null;
    this.resolve_pipeline_state = null;
    this.bind_group = null;
    this.depth_bind_group = null;
    this.resolve_bind_group = null;
    this.parent = parent_id;
    this.uniform_data = new Map();
    this.storage_data = new Map();
    this.texture_data = new Map();
    this.sampler_data = new Map();
    this.data_listeners = new Set();
    this.bind_group_update_flags = 0;
    // Depending on behaviors based on the family, it might be useful to have it exposed like this for derived materials.
    // Otherwise, TODO so we only use the family from the template.
    this.family = this.template.family;
    this.set_uniform_data.bind(this);
    this.set_storage_data.bind(this);
    this.set_texture_data.bind(this);
    this.set_sampler_data.bind(this);
  }

  set needs_bind_group_update(value) {
    this.bind_group_update_flags = value ? 7 : 0;
  }

  _build_bind_group_entries(pass_type) {
    return this.template
      .get_resources_for_pass(pass_type)
      .map((resource) => {
        switch (resource.type) {
          case ShaderResourceType.Uniform:
            return {
              binding: resource.binding,
              resource: { buffer: this.uniform_data.get(resource.name).buffer },
            };
          case ShaderResourceType.Storage:
            return {
              binding: resource.binding,
              resource: { buffer: this.storage_data.get(resource.name).buffer },
            };
          case ShaderResourceType.Texture:
            let texture = this.texture_data.get(resource.name);
            if (!texture) {
              texture = resource.is_array ? Texture.default_array() : Texture.default();
            }
            let view = texture.view;
            return {
              binding: resource.binding,
              resource: view,
            };
          case ShaderResourceType.Sampler:
            return {
              binding: resource.binding,
              resource: this.sampler_data.get(resource.name),
            };
        }
      })
      .filter((entry) => entry !== null);
  }

  _refresh_bind_group(pass_type = MaterialPassType.Raster) {
    if (this.bind_group_update_flags === 0) {
      return;
    }

    if (pass_type === MaterialPassType.Depth && (this.bind_group_update_flags & 1) !== 0 && this.depth_pipeline_state) {
      const entries = this._build_bind_group_entries(MaterialPassType.Depth);
      this.depth_bind_group = BindGroup.create(
        `${this.name}_depth`,
        this.depth_pipeline_state,
        BindGroupType.Material,
        entries,
        true /* force */
      );
      this.bind_group_update_flags = this.bind_group_update_flags & ~1;
    }

    if (pass_type === MaterialPassType.Raster && (this.bind_group_update_flags & 2) !== 0 && this.pipeline_state) {
      const entries = this._build_bind_group_entries(MaterialPassType.Raster);
      this.bind_group = BindGroup.create(
        this.name,
        this.pipeline_state,
        BindGroupType.Material,
        entries,
        true /* force */
      );
      this.bind_group_update_flags = this.bind_group_update_flags & ~2;
    }

    if (pass_type === MaterialPassType.Resolve && (this.bind_group_update_flags & 4) !== 0 && this.resolve_pipeline_state) {
      const entries = this._build_bind_group_entries(MaterialPassType.Resolve);
      this.resolve_bind_group = BindGroup.create(
        `${this.name}_resolve`,
        this.resolve_pipeline_state,
        BindGroupType.Material,
        entries,
        true /* force */
      );
      this.bind_group_update_flags = this.bind_group_update_flags & ~4;
    }
  }

  update_pipeline_state(bind_groups, output_targets = [], pass_type = MaterialPassType.Raster) {
    const renderer = Renderer.get();
    const depth_prepass_enabled = renderer.is_depth_prepass_enabled();

    const layouts = bind_groups.filter((bg) => bg !== null).map((bg) => bg.layout);
    if (pass_type === MaterialPassType.Depth) {
      this.depth_pipeline_state = this.template.create_pipeline_state(
        layouts,
        output_targets,
        { depth_write_enabled: true, depth_compare: "less" },
        MaterialPassType.Depth
      );
    } else if (pass_type === MaterialPassType.Resolve) {
      this.resolve_pipeline_state = this.template.create_pipeline_state(
        layouts,
        output_targets,
        null,
        MaterialPassType.Resolve
      );
    } else {
      this.pipeline_state = this.template.create_pipeline_state(
        layouts,
        output_targets,
        {
          depth_write_enabled: depth_prepass_enabled
            ? false
            : this.family === MaterialFamilyType.Opaque,
          depth_compare: "less-equal",
        },
        MaterialPassType.Raster
      );
    }
  }

  set_uniform_data(name, data) {
    this.uniform_data.set(name, data);
    this.needs_bind_group_update = true;
  }

  listen_for_uniform_data(name, cb = null) {
    if (!this.data_listeners.has(name)) {
      this.data_listeners.add(name);
      global_dispatcher.on(name, (data) => {
        if (data) {
          this.set_uniform_data(name, data);
          if (cb) {
            cb(name, data);
          }
        }
      });
    }
  }

  set_storage_data(name, data) {
    this.storage_data.set(name, data);
    this.needs_bind_group_update = true;
  }

  listen_for_storage_data(name, cb = null) {
    if (!this.data_listeners.has(name)) {
      this.data_listeners.add(name);
      global_dispatcher.on(name, (data) => {
        if (data) {
          this.set_storage_data(name, data);
          if (cb) {
            cb(name, data);
          }
        }
      });
    }
  }

  set_texture_data(name, texture) {
    this.texture_data.set(name, texture);
    this.needs_bind_group_update = true;
  }

  listen_for_texture_data(name, cb = null) {
    if (!this.data_listeners.has(name)) {
      this.data_listeners.add(name);
      global_dispatcher.on(name, (data) => {
        if (data) {
          this.set_texture_data(name, data);
          if (cb) {
            cb(name, data);
          }
        }
      });
    }
  }

  set_sampler_data(name, sampler) {
    this.sampler_data.set(name, sampler);
    this.needs_bind_group_update = true;
  }

  listen_for_sampler_data(name, cb = null) {
    if (!this.data_listeners.has(name)) {
      this.data_listeners.add(name);
      global_dispatcher.on(name, (data) => {
        if (data) {
          this.set_sampler_data(name, data);
          if (cb) {
            cb(name, data);
          }
        }
      });
    }
  }

  _get_pipeline_state_for_pass(pass_type) {
    if (pass_type === MaterialPassType.Depth) return this.depth_pipeline_state;
    if (pass_type === MaterialPassType.Resolve) return this.resolve_pipeline_state;
    return this.pipeline_state;
  }

  _get_bind_group_for_pass(pass_type) {
    if (pass_type === MaterialPassType.Depth) return this.depth_bind_group;
    if (pass_type === MaterialPassType.Resolve) return this.resolve_bind_group;
    return this.bind_group;
  }

  bind(render_pass, bind_groups = [], output_targets = [], pass_type = MaterialPassType.Raster) {
    let pso = this._get_pipeline_state_for_pass(pass_type);
    if (!pso && !this.parent && bind_groups.length > 0 && output_targets.length > 0) {
      this.update_pipeline_state(bind_groups, output_targets, pass_type);
      for (let i = 0; i < bind_groups.length; i++) {
        if (bind_groups[i]) {
          bind_groups[i].bind(render_pass);
        }
      }
    }

    pso = this._get_pipeline_state_for_pass(pass_type);
    const parent_material = Material.get(this.parent);
    if (parent_material) {
      pso = parent_material._get_pipeline_state_for_pass(pass_type);
    }

    this._refresh_bind_group(pass_type);

    if (pso) {
      render_pass.set_pipeline(pso);
    }

    const bind_group = this._get_bind_group_for_pass(pass_type);
    if (bind_group) {
      bind_group.bind(render_pass);
    }
  }

  new_instance(instance_name) {
    return Material.create(instance_name, this.template.name, {}, this.parent);
  }

  static create(name, template_name, options = {}, parent_id = null) {
    const template = MaterialTemplate.get_template(
      template_name,
      options.family || MaterialFamilyType.Opaque
    );
    if (!template) {
      throw new Error(`Material template '${template_name}' not found`);
    }

    const material_id = Name.from(name);
    let material = ResourceCache.get().fetch(CacheTypes.MATERIAL, material_id);

    if (options.force_new && material) {
      Material.materials.delete(material_id);
      ResourceCache.get().remove(CacheTypes.MATERIAL, material_id);
      material = null;
    }

    if (!material) {
      material = new Material(name, template, parent_id);
      material.family = template.family;
      ResourceCache.get().store(CacheTypes.MATERIAL, material_id, material);
      Material.materials.set(material_id, material);
    }

    return material_id;
  }

  static #default_material = null;
  static default_material() {
    if (!this.#default_material) {
      MaterialTemplate.create(
        "DefaultMaterial",
        "visibility/visibility_draw_standard.wgsl",
        MaterialFamilyType.Opaque
      );
      this.#default_material = Material.create("DefaultMaterial", "DefaultMaterial", {
        family: MaterialFamilyType.Opaque,
      });
    }

    return this.#default_material;
  }

  static #default_ui_material = null;
  static default_ui_material() {
    if (!this.#default_ui_material) {
      MaterialTemplate.create(
        "DefaultUIMaterial",
        "ui_standard_material.wgsl",
        MaterialFamilyType.Transparent,
        {
          rasterizer_state: {
            cull_mode: "none",
          },
        }
      );
      this.#default_ui_material = Material.create("DefaultUIMaterial", "DefaultUIMaterial", {
        family: MaterialFamilyType.Transparent,
      }); 

      const default_ui_material_object = Material.get(this.#default_ui_material);
      const element_data_buffer = FragmentGpuBuffer.get_buffer_name(
        UserInterfaceFragment,
        "element_data"
      );
      default_ui_material_object.listen_for_storage_data(element_data_buffer);
    }
    return this.#default_ui_material;
  }

  static get(material_id) {
    return Material.materials.get(material_id);
  }
}

/**
 * Standard material is a material helper class that has a color, normal, roughness, metallic, and emission.
 * It is the default material for the engine.
 */
const float_params_offset = 0;
const texture_flags1_offset = 16;
const texture_flags2_offset = 20;
const texture_handles_offset = 24;
const standard_texture_pool_keys = [
  "albedo",
  "normal",
  "roughness",
  "metallic",
  "ao",
  "height",
  "specular",
  "emission",
];

export class StandardMaterial {
  material_id = null;
  material_allocation_index = null;

  static create(name, params = {}, options = {}, template = null) {
    if (!template) {
      const family = options.family !== undefined ? options.family : MaterialFamilyType.Opaque;

      // TODO: Need a better way to handle these kinds of template permutations.
      if (options.raster_state?.cull_mode == "none") {
        MaterialTemplate.create("StandardMaterial_NoCull", "visibility/visibility_draw_standard.wgsl", family, {
          rasterizer_state: {
            cull_mode: "none",
          },
        });
        template = `StandardMaterial_NoCull`;
      } else {
        MaterialTemplate.create("StandardMaterial", "visibility/visibility_draw_standard.wgsl", family);
        template = `StandardMaterial`;
      }
    }

    let standard_material = new StandardMaterial();

    // Create the material
    standard_material.material_id = Material.create(name, template, options);

    // Get the material
    const material = Material.get(standard_material.material_id);

    // Allocate the material params in the allocation table
    standard_material.material_allocation_index = MaterialAllocationTable.find_or_allocate(
      standard_material.material_id
    );
    const offset = standard_material.material_allocation_index * MATERIAL_PARAMS_SIZE;
    const params_buffer = MaterialAllocationTable.params_data;
    const params_gpu_buffer = MaterialAllocationTable.params_buffer;
    const material_palette_buffer = MaterialAllocationTable.palette_buffer;

    const material_palette_offsets_buffer = EntityManager.get_fragment_gpu_buffer(
      StaticMeshFragment,
      material_offsets_name
    );

    // Set the material storage buffers for the material
    material.listen_for_storage_data("material_params");
    material.listen_for_storage_data("material_table_offset");
    material.listen_for_storage_data("material_palette");
    material.set_storage_data("material_params", params_gpu_buffer);
    material.set_storage_data("material_table_offset", material_palette_offsets_buffer.buffer);
    material.set_storage_data("material_palette", material_palette_buffer);

    // Visibility resolve uses shared texture pools for all standard materials in a bucket.
    // The representative material therefore needs every pool bound, not only the ones this
    // specific material instance samples directly.
    for (let i = 0; i < standard_texture_pool_keys.length; i++) {
      const pool_key = standard_texture_pool_keys[i];
      const pool_name = `texture_pool_${pool_key}`;
      material.listen_for_texture_data(pool_name);

      const existing_pool = ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(pool_name));
      if (existing_pool) {
        material.set_texture_data(pool_name, existing_pool);
      }
    }

    // Set initial values
    params_buffer.set(
      [
        // albedo: vec4
        0.5, 0.5, 0.5, 1.0,
        // normal: vec4
        0.0, 0.0, 1.0, 1.0,
        // emission_roughness_metallic_tiling: vec4
        0.0, 0.7, 0.3, 1.0,
        // ao_height_specular: vec4 (ao, height, specular, padding)
        1.0, 0.0, 0.1, 0.0,
        // texture flags 1: vec4 (albedo, normal, roughness, metallic)
        0, 0, 0, 0,
        // texture flags 2: vec4 (ao, height, specular, emission)
        0, 0, 0, 0,
        // albedo_handle: u32
        0,
        // normal_handle: u32
        0,
        // roughness_handle: u32
        0,
        // metallic_handle: u32
        0,
        // ao_handle: u32
        0,
        // height_handle: u32
        0,
        // specular_handle: u32
        0,
        // emission_handle: u32
        0,
      ],
      offset
    );

    // Set default parameter values
    if (params.albedo_texture) {
      standard_material.sample_albedo(params.albedo_texture);
    } else {
      standard_material.set_albedo(params.albedo || [1, 1, 1, 1]);
    }

    if (params.normal_texture) {
      standard_material.sample_normal(params.normal_texture);
    } else {
      standard_material.set_normal(params.normal || [0, 0, 1, 1]);
    }

    if (params.roughness_texture) {
      standard_material.sample_roughness(
        params.roughness_texture,
        params.roughness_channel ?? TextureChannel.R
      );
    } else {
      standard_material.set_roughness(params.roughness || 0.7);
    }

    if (params.metallic_texture) {
      standard_material.sample_metallic(
        params.metallic_texture,
        params.metallic_channel ?? TextureChannel.R
      );
    } else {
      standard_material.set_metallic(params.metallic || 0.3);
    }

    if (params.emission_texture) {
      standard_material.sample_emission(
        params.emission_texture,
        params.emission_channel ?? TextureChannel.R
      );
    } else {
      standard_material.set_emission(params.emission || 0.0);
    }

    if (params.ao_texture) {
      standard_material.sample_ao(params.ao_texture, params.ao_channel ?? TextureChannel.R);
    } else {
      standard_material.set_ao(params.ao || 1.0);
    }

    if (params.height_texture) {
      standard_material.sample_height(
        params.height_texture,
        params.height_channel ?? TextureChannel.R
      );
    } else {
      standard_material.set_height(params.height || 0.0);
    }

    if (params.specular_texture) {
      standard_material.sample_specular(
        params.specular_texture,
        params.specular_channel ?? TextureChannel.R
      );
    } else {
      standard_material.set_specular(params.specular || 0.1);
    }

    standard_material._update_albedo_bindless_handle =
      standard_material._update_albedo_bindless_handle.bind(standard_material);
    standard_material._update_normal_bindless_handle =
      standard_material._update_normal_bindless_handle.bind(standard_material);
    standard_material._update_roughness_bindless_handle =
      standard_material._update_roughness_bindless_handle.bind(standard_material);
    standard_material._update_metallic_bindless_handle =
      standard_material._update_metallic_bindless_handle.bind(standard_material);
    standard_material._update_ao_bindless_handle =
      standard_material._update_ao_bindless_handle.bind(standard_material);
    standard_material._update_height_bindless_handle =
      standard_material._update_height_bindless_handle.bind(standard_material);
    standard_material._update_specular_bindless_handle =
      standard_material._update_specular_bindless_handle.bind(standard_material);
    standard_material._update_emission_bindless_handle =
      standard_material._update_emission_bindless_handle.bind(standard_material);

    return standard_material;
  }

  set_albedo(color) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[float_params_offset + offset + 0] = color[0];
    params_buffer[float_params_offset + offset + 1] = color[1];
    params_buffer[float_params_offset + offset + 2] = color[2];
    params_buffer[float_params_offset + offset + 3] = color[3];
    params_buffer[texture_flags1_offset + offset + 0] = 0;
    this.mark_params_dirty();
  }

  sample_albedo(texture_config) {
    if (!texture_config) return;

    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;

    // Standard materials only support bindless-style texture sampling which is why we
    // set this pool key and create the material internally (so we can catch relevant texture events
    // in order to get the proper bindless handle)
    texture_config.pool_key = "albedo";

    const material = Material.get(this.material_id);

    const is_external_load = texture_config.paths && texture_config.paths.length > 0;
    const texture = is_external_load
      ? Texture.load(texture_config)
      : Texture.create(texture_config);

    material.set_texture_data("albedo", texture);
    if (texture.config.material_notifier) {
      material.listen_for_texture_data(
        texture.config.material_notifier,
        this._update_albedo_bindless_handle
      );
    }
    material.listen_for_texture_data(`texture_pool_${texture_config.pool_key}`);
    params_buffer[offset + texture_handles_offset + 0] = texture.bindless_handle;

    params_buffer[offset + texture_flags1_offset + 0] = 1;

    this.mark_params_dirty();
  }

  _update_albedo_bindless_handle(name, texture) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + texture_handles_offset + 0] = texture.bindless_handle;
    params_buffer[offset + texture_flags1_offset + 0] |= 1;
    this.mark_params_dirty();
  }

  set_normal(normal) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 4] = normal[0];
    params_buffer[offset + float_params_offset + 5] = normal[1];
    params_buffer[offset + float_params_offset + 6] = normal[2];
    params_buffer[offset + float_params_offset + 7] = normal[3];
    params_buffer[offset + texture_flags1_offset + 1] = 0;
    this.mark_params_dirty();
  }

  sample_normal(texture_config) {
    if (!texture_config) return;

    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;

    // Standard materials only support bindless-style texture sampling which is why we
    // set this pool key and create the material internally (so we can catch relevant texture events
    // in order to get the proper bindless handle)
    texture_config.pool_key = "normal";

    const material = Material.get(this.material_id);

    const is_external_load = texture_config.paths && texture_config.paths.length > 0;
    const texture = is_external_load
      ? Texture.load(texture_config)
      : Texture.create(texture_config);

    material.set_texture_data("normal", texture);
    if (texture.config.material_notifier) {
      material.listen_for_texture_data(
        texture.config.material_notifier,
        this._update_normal_bindless_handle
      );
    }
    material.listen_for_texture_data(`texture_pool_${texture_config.pool_key}`);
    params_buffer[offset + texture_handles_offset + 1] = texture.bindless_handle;

    const flip_y = texture_config.flip_y !== undefined ? texture_config.flip_y : true;
    const invert_y = texture_config.invert_y !== undefined ? texture_config.invert_y : flip_y;
    params_buffer[offset + texture_flags1_offset + 1] = 1 | (invert_y ? 2 : 0);

    this.mark_params_dirty();
  }

  _update_normal_bindless_handle(name, texture) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + texture_handles_offset + 1] = texture.bindless_handle;
    params_buffer[offset + texture_flags1_offset + 1] |= 1;
    this.mark_params_dirty();
  }

  set_roughness(roughness) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 9] = roughness;
    params_buffer[offset + texture_flags1_offset + 2] = 0;
    this.mark_params_dirty();
  }

  sample_roughness(texture_config, channel = TextureChannel.R) {
    if (!texture_config) return;

    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;

    // Standard materials only support bindless-style texture sampling which is why we
    // set this pool key and create the material internally (so we can catch relevant texture events
    // in order to get the proper bindless handle)
    texture_config.pool_key = "roughness";

    const material = Material.get(this.material_id);

    const is_external_load = texture_config.paths && texture_config.paths.length > 0;
    const texture = is_external_load
      ? Texture.load(texture_config)
      : Texture.create(texture_config);

    material.set_texture_data("roughness", texture);
    if (texture.config.material_notifier) {
      material.listen_for_texture_data(
        texture.config.material_notifier,
        this._update_roughness_bindless_handle
      );
    }
    material.listen_for_texture_data(`texture_pool_${texture_config.pool_key}`);
    params_buffer[offset + texture_handles_offset + 2] = texture.bindless_handle;

    let flag = 1;
    if (channel >= 0 && channel <= 3) {
      flag |= channel << 1;
    }
    params_buffer[offset + texture_flags1_offset + 2] = flag;

    this.mark_params_dirty();
  }

  _update_roughness_bindless_handle(name, texture) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + texture_handles_offset + 2] = texture.bindless_handle;
    params_buffer[offset + texture_flags1_offset + 2] |= 1;
    this.mark_params_dirty();
  }

  set_metallic(metallic) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 10] = metallic;
    params_buffer[offset + texture_flags1_offset + 3] = 0;
    this.mark_params_dirty();
  }

  sample_metallic(texture_config, channel = TextureChannel.R) {
    if (!texture_config) return;

    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;

    // Standard materials only support bindless-style texture sampling which is why we
    // set this pool key and create the material internally (so we can catch relevant texture events
    // in order to get the proper bindless handle)
    texture_config.pool_key = "metallic";

    const material = Material.get(this.material_id);

    const is_external_load = texture_config.paths && texture_config.paths.length > 0;
    const texture = is_external_load
      ? Texture.load(texture_config)
      : Texture.create(texture_config);

    material.set_texture_data("metallic", texture);
    if (texture.config.material_notifier) {
      material.listen_for_texture_data(
        texture.config.material_notifier,
        this._update_metallic_bindless_handle
      );
    }
    material.listen_for_texture_data(`texture_pool_${texture_config.pool_key}`);
    params_buffer[offset + texture_handles_offset + 3] = texture.bindless_handle;

    let flag = 1;
    if (channel >= 0 && channel <= 3) {
      flag |= channel << 1;
    }
    params_buffer[offset + texture_flags1_offset + 3] = flag;

    this.mark_params_dirty();
  }

  _update_metallic_bindless_handle(name, texture) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + texture_handles_offset + 3] = texture.bindless_handle;
    params_buffer[offset + texture_flags1_offset + 3] |= 1;
    this.mark_params_dirty();
  }

  set_ao(ao) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 12] = ao;
    params_buffer[offset + texture_flags2_offset + 0] = 0;
    this.mark_params_dirty();
  }

  sample_ao(texture_config, channel = TextureChannel.R) {
    if (!texture_config) return;

    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;

    // Standard materials only support bindless-style texture sampling which is why we
    // set this flag and create the material internally (so we can catch relevatn texture events
    // in order to get the proper bindless handle)
    texture_config.pool_key = "ao";

    const material = Material.get(this.material_id);

    const is_external_load = texture_config.paths && texture_config.paths.length > 0;
    const texture = is_external_load
      ? Texture.load(texture_config)
      : Texture.create(texture_config);

    material.set_texture_data("ao", texture);
    if (texture.config.material_notifier) {
      material.listen_for_texture_data(
        texture.config.material_notifier,
        this._update_ao_bindless_handle
      );
    }
    material.listen_for_texture_data(`texture_pool_${texture_config.pool_key}`);
    params_buffer[offset + texture_handles_offset + 4] = texture.bindless_handle;

    let flag = 1;
    if (channel >= 0 && channel <= 3) {
      flag |= channel << 1;
    }
    params_buffer[offset + texture_flags2_offset + 0] = flag;

    this.mark_params_dirty();
  }

  _update_ao_bindless_handle(name, texture) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + texture_handles_offset + 4] = texture.bindless_handle;
    params_buffer[offset + texture_flags2_offset + 0] |= 1;
    this.mark_params_dirty();
  }

  set_height(height) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 13] = height;
    params_buffer[offset + texture_flags2_offset + 1] = 0;
    this.mark_params_dirty();
  }

  sample_height(texture_config, channel = TextureChannel.R) {
    if (!texture_config) return;

    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;

    // Standard materials only support bindless-style texture sampling which is why we
    // set this pool key and create the material internally (so we can catch relevant texture events
    // in order to get the proper bindless handle)
    texture_config.pool_key = "height";

    const material = Material.get(this.material_id);

    const is_external_load = texture_config.paths && texture_config.paths.length > 0;
    const texture = is_external_load
      ? Texture.load(texture_config)
      : Texture.create(texture_config);

    material.set_texture_data("height", texture);
    if (texture.config.material_notifier) {
      material.listen_for_texture_data(
        texture.config.material_notifier,
        this._update_height_bindless_handle
      );
    }
    material.listen_for_texture_data(`texture_pool_${texture_config.pool_key}`);
    params_buffer[offset + texture_handles_offset + 5] = texture.bindless_handle;

    let flag = 1;
    if (channel >= 0 && channel <= 3) {
      flag |= channel << 1;
    }
    params_buffer[offset + texture_flags2_offset + 1] = flag;

    this.mark_params_dirty();
  }

  _update_height_bindless_handle(name, texture) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + texture_handles_offset + 5] = texture.bindless_handle;
    params_buffer[offset + texture_flags2_offset + 1] |= 1;
    this.mark_params_dirty();
  }

  set_specular(specular) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 14] = specular;
    params_buffer[offset + texture_flags2_offset + 2] = 0;
    this.mark_params_dirty();
  }

  sample_specular(texture_config, channel = TextureChannel.R) {
    if (!texture_config) return;

    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;

    // Standard materials only support bindless-style texture sampling which is why we
    // set this pool key and create the material internally (so we can catch relevant texture events
    // in order to get the proper bindless handle)
    texture_config.pool_key = "specular";

    const material = Material.get(this.material_id);

    const is_external_load = texture_config.paths && texture_config.paths.length > 0;
    const texture = is_external_load
      ? Texture.load(texture_config)
      : Texture.create(texture_config);

    material.set_texture_data("specular", texture);
    if (texture.config.material_notifier) {
      material.listen_for_texture_data(
        texture.config.material_notifier,
        this._update_specular_bindless_handle
      );
    }
    material.listen_for_texture_data(`texture_pool_${texture_config.pool_key}`);
    params_buffer[offset + texture_handles_offset + 6] = texture.bindless_handle;

    let flag = 1;
    if (channel >= 0 && channel <= 3) {
      flag |= channel << 1;
    }
    params_buffer[offset + texture_flags2_offset + 2] = flag;

    this.mark_params_dirty();
  }

  _update_specular_bindless_handle(name, texture) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + texture_handles_offset + 6] = texture.bindless_handle;
    params_buffer[offset + texture_flags2_offset + 2] |= 1;
    this.mark_params_dirty();
  }

  set_emission(emission) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 8] = emission;
    params_buffer[offset + texture_flags2_offset + 3] = 0;
    this.mark_params_dirty();
  }

  sample_emission(texture_config, channel = TextureChannel.R) {
    if (!texture_config) return;

    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;

    // Standard materials only support bindless-style texture sampling which is why we
    // set this pool key and create the material internally (so we can catch relevant texture events
    // in order to get the proper bindless handle)
    texture_config.pool_key = "emission";

    const material = Material.get(this.material_id);

    const is_external_load = texture_config.paths && texture_config.paths.length > 0;
    const texture = is_external_load
      ? Texture.load(texture_config)
      : Texture.create(texture_config);

    material.set_texture_data("emission", texture);
    if (texture.config.material_notifier) {
      material.listen_for_texture_data(
        texture.config.material_notifier,
        this._update_emission_bindless_handle
      );
    }
    material.listen_for_texture_data(`texture_pool_${texture_config.pool_key}`);
    params_buffer[offset + texture_handles_offset + 7] = texture.bindless_handle;

    let flag = 1;
    if (channel >= 0 && channel <= 3) {
      flag |= channel << 1;
    }
    params_buffer[offset + texture_flags2_offset + 3] = flag;

    this.mark_params_dirty();
  }

  _update_emission_bindless_handle(name, texture) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + texture_handles_offset + 7] = texture.bindless_handle;
    params_buffer[offset + texture_flags2_offset + 3] |= 1;
    this.mark_params_dirty();
  }

  set_tiling(tiling) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 11] = tiling;
    this.mark_params_dirty();
  }

  set_alpha_cutoff(alpha_cutoff) {
    const params_buffer = MaterialAllocationTable.params_data;
    const offset = this.material_allocation_index * MATERIAL_PARAMS_SIZE;
    params_buffer[offset + float_params_offset + 15] = Math.max(0.0, alpha_cutoff);
    this.mark_params_dirty();
  }

  mark_params_dirty() {
    MaterialAllocationTable.mark_params_dirty(this.material_allocation_index);
  }
}

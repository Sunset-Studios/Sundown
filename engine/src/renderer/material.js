import { Shader } from "./shader.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { BindGroup } from "./bind_group.js";
import { PipelineState } from "./pipeline_state.js";
import { ResourceCache } from "./resource_cache.js";
import { profile_scope } from "../utility/performance.js";
import { hash_data, hash_value } from "../utility/hashing.js";
import { Name } from "../utility/names.js";
import { Texture } from "./texture.js";
import { Buffer } from "./buffer.js";
import { UserInterfaceFragment } from "../core/ecs/fragments/user_interface_fragment.js";
import { global_dispatcher } from "../core/dispatcher.js";
import {
  ShaderResourceType,
  MaterialFamilyType,
  CacheTypes,
  BindGroupType,
} from "./renderer_types.js";

export class MaterialTemplate {
  static templates = new Map();

  constructor(
    name,
    shader,
    family = MaterialFamilyType.Opaque,
    pipeline_state_config = {},
    parent = null
  ) {
    this.name = name;
    this.shader = shader;
    this.pipeline_state_config = pipeline_state_config;
    this.resources = [];
    this.parent = parent;
    this.family = family;
  }

  add_resource(resource) {
    this.resources.push(resource);
  }

  get reflection() {
    return this.shader.reflection;
  }

  static create(
    name,
    shader_path,
    family = MaterialFamilyType.Opaque,
    pipeline_state_config = {},
    parent_name = null,
    defines = {}
  ) {
    if (this.templates.has(name)) {
      return this.templates.get(name);
    }

    let parent = null;
    let shader = null;

    if (parent_name) {
      parent = this.get_template(parent_name);
      if (!parent) {
        throw new Error(`Parent template '${parent_name}' not found`);
      }
      shader = parent.shader;
    }

    if (shader_path) {
      shader = Shader.create(shader_path, defines);
    }
    if (shader.defines["TRANSPARENT"]) {
      family = MaterialFamilyType.Transparent;
    }

    const template = new MaterialTemplate(name, shader, family, pipeline_state_config, parent);

    if (parent) {
      template.resources = [...parent.resources];
    }

    // Reflect on shader and add resources
    const groups = template.reflection ? template.reflection.getBindGroups() : [];
    if (BindGroupType.Material < groups.length) {
      const material_group = groups[BindGroupType.Material];
      for (let i = 0; i < material_group.length; i++) {
        const binding = material_group[i];
        const binding_type = Shader.resource_type_from_reflection_type(binding.resourceType);
        template.add_resource({
          type: binding_type,
          name: binding.name,
          binding: i,
        });
      }
    }

    this.templates.set(name, template);

    return template;
  }

  create_pipeline_state(bind_group_layouts, output_targets = [], depth_stencil_options = {}) {
    let all_bind_group_layouts = [bind_group_layouts[0]];

    // Set material binding group inputs
    const groups = this.reflection.getBindGroups();
    if (BindGroupType.Pass < groups.length) {
      for (let i = BindGroupType.Pass; i < groups.length; i++) {
        const bind_group = groups[i];

        all_bind_group_layouts.push(
          BindGroup.create_layout(
            this.name,
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
                      sampleType: binding.type.name.includes("depth") ? "depth" : "float",
                    },
                  };
                  break;
                case ShaderResourceType.StorageTexture:
                  binding_obj = {
                    storageTexture: {
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
    const targets = output_targets
      .filter((target) => target.config.type !== "depth")
      .map((target) => {
        let t = {
          name: target.config.name,
          format: target.config.format,
        };
        if (target.config.blend) {
          t.blend = target.config.blend;
        }
        return t;
      });

    const fragment_outputs = this.reflection.entry.fragment[0].outputs;
    fragment_outputs.forEach((output, i) => {
      if (targets[i]) {
        return;
      }

      const target = {
        format: Shader.get_optimal_texture_format(output.type.name),
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
      label: this.name,
      bind_layouts: all_bind_group_layouts.filter((layout) => layout !== null),
      vertex: {
        module: this.shader.module,
        entryPoint:
          (this.pipeline_state_config.vertex && this.pipeline_state_config.vertex.entry_point) ||
          "vs",
        buffers: [],
      },
      fragment: {
        module: this.shader.module,
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
          true,
        depthCompare:
          this.pipeline_state_config.depth_stencil_target.depth_compare ??
          depth_stencil_options.depth_compare ??
          "less",
        depthBias: this.pipeline_state_config.depth_stencil_target.depth_bias ?? 0,
        depthBiasClamp: this.pipeline_state_config.depth_stencil_target.depth_bias_clamp ?? 0,
        depthBiasSlopeScale: this.pipeline_state_config.depth_stencil_target.depth_slope_scale ?? 0,
      };
    } else if (depth_target) {
      pipeline_descriptor.depthStencil = {
        format: depth_target.config.format,
        depthWriteEnabled: depth_stencil_options.depth_write_enabled ?? true,
        depthCompare: depth_stencil_options.depth_compare ?? "less",
        depthBias: depth_stencil_options.depth_bias ?? 0,
        depthBiasClamp: depth_stencil_options.depth_bias_clamp ?? 0,
        depthBiasSlopeScale: depth_stencil_options.depth_slope_scale ?? 0,
      };
    }

    return PipelineState.create_render(this.name, pipeline_descriptor);
  }

  static get_template(name) {
    return this.templates.get(name);
  }

  get_all_resources() {
    if (this.parent) {
      return [...this.parent.get_all_resources(), ...this.resources];
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
    this.bind_group = null;
    this.parent = parent_id;
    this.uniform_data = new Map();
    this.storage_data = new Map();
    this.texture_data = new Map();
    this.sampler_data = new Map();
    this.data_listeners = new Set();
    this.state_hash = 0;
    this.needs_bind_group_update = false;
    // Depending on behaviors based on the family, it might be useful to have it exposed like this for derived materials.
    // Otherwise, TODO so we only use the family from the template.
    this.family = this.template.family;
    this.writes_entity_id = true;
    this.set_uniform_data.bind(this);
    this.set_storage_data.bind(this);
    this.set_texture_data.bind(this);
    this.set_sampler_data.bind(this);
    this._update_state_hash();
  }

  _update_state_hash() {
    profile_scope("Material._update_state_hash", () => {
      let hash = hash_value(this.template.name);
      hash = hash_data(this.uniform_data, hash);
      hash = hash_data(this.storage_data, hash);
      hash = hash_data(this.texture_data, hash);
      hash = hash_data(this.sampler_data, hash);
      this.state_hash = hash;
    });
  }

  _refresh_bind_group() {
    const entries = this.template.get_all_resources().map((resource) => {
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
          const texture = this.texture_data.get(resource.name) ?? Texture.default();
          return {
            binding: resource.binding,
            resource: texture.view,
          };
        case ShaderResourceType.Sampler:
          return {
            binding: resource.binding,
            resource: this.sampler_data.get(resource.name),
          };
      }
    });

    for (let i = 0; i < entries.length; i++) {
      if (!entries[i].resource) {
        throw new Error(`Binding ${this.template.name} has an invalid resource at index ${i}`);
      }
    }

    this.bind_group = BindGroup.create(
      this.name,
      this.pipeline_state,
      BindGroupType.Material,
      entries,
      true /* force */
    );
  }

  update_pipeline_state(bind_groups, output_targets = []) {
    this.pipeline_state = this.template.create_pipeline_state(
      bind_groups
        .filter((bind_group) => bind_group !== null)
        .map((bind_group) => bind_group.layout),
      output_targets,
      {
        depth_write_enabled: this.family === MaterialFamilyType.Opaque,
      }
    );
  }

  set_uniform_data(name, data) {
    this.uniform_data.set(name, data);
    this._update_state_hash();
    this.needs_bind_group_update = true;
  }

  listen_for_uniform_data(name) {
    if (!this.data_listeners.has(name)) {
      this.data_listeners.add(name);
      global_dispatcher.on(name, (data) => {
        if (data) {
          this.set_uniform_data(name, data);
        }
      });
    }
  }

  set_storage_data(name, data) {
    this.storage_data.set(name, data);
    this._update_state_hash();
    this.needs_bind_group_update = true;
  }

  listen_for_storage_data(name) {
    if (!this.data_listeners.has(name)) {
      this.data_listeners.add(name);
      global_dispatcher.on(name, (data) => {
        if (data) {
          this.set_storage_data(name, data);
        }
      });
    }
  }

  set_texture_data(name, texture) {
    this.texture_data.set(name, texture);
    this._update_state_hash();
    this.needs_bind_group_update = true;
  }

  listen_for_texture_data(name) {
    if (!this.data_listeners.has(name)) {
      this.data_listeners.add(name);
      global_dispatcher.on(name, (data) => {
        if (data) {
          this.set_texture_data(name, data);
        }
      });
    }
  }

  set_sampler_data(name, sampler) {
    this.sampler_data.set(name, sampler);
    this._update_state_hash();
    this.needs_bind_group_update = true;
  }

  listen_for_sampler_data(name) {
    if (!this.data_listeners.has(name)) {
      this.data_listeners.add(name);
      global_dispatcher.on(name, (data) => {
        if (data) {
          this.set_sampler_data(name, data);
        }
      });
    }
  }

  bind(render_pass, bind_groups = [], output_targets = []) {
    if (
      !this.pipeline_state &&
      !this.parent &&
      bind_groups.length > 0 &&
      output_targets.length > 0
    ) {
      this.update_pipeline_state(bind_groups, output_targets);
      for (let i = 0; i < bind_groups.length; i++) {
        if (bind_groups[i]) {
          bind_groups[i].bind(render_pass);
        }
      }
    }

    let pso = this.pipeline_state;
    const parent_material = Material.get(this.parent);
    if (parent_material) {
      pso = parent_material.pipeline_state;
    }

    if (this.needs_bind_group_update) {
      this._refresh_bind_group();
      this.needs_bind_group_update = false;
    }

    if (pso) {
      render_pass.set_pipeline(pso);
    }

    if (this.bind_group) {
      this.bind_group.bind(render_pass);
    }
  }

  get_state_hash() {
    return this.state_hash;
  }

  new_instance(instance_name) {
    return Material.create(instance_name, this.template.name, {}, this.parent);
  }

  static create(name, template_name, options = {}, parent_id = null) {
    const template = MaterialTemplate.get_template(template_name);
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
      if (options.family) {
        material.family = options.family;
      }
      if (options.writes_entity_id !== undefined) {
        material.writes_entity_id = options.writes_entity_id;
      }
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
        "standard_material.wgsl",
        MaterialFamilyType.Opaque,
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
        },
      );
      this.#default_ui_material = Material.create("DefaultUIMaterial", "DefaultUIMaterial", {
        family: MaterialFamilyType.Transparent,
      });

      const default_ui_material_object = Material.get(this.#default_ui_material);
      const element_data_buffer = FragmentGpuBuffer.get_buffer_name(UserInterfaceFragment, "element_data");
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
export class StandardMaterial {
  material_id = null;
  material_params_data = null;
  material_params_buffer = null;

  static create(name, params = {}, options = {}, template = null) {
    if (!template) {
      MaterialTemplate.create(
        "StandardMaterial",
        "standard_material.wgsl",
        MaterialFamilyType.Opaque,
      );
      template = "StandardMaterial";
    }

    let standard_material = new StandardMaterial();

    // Create the material
    standard_material.material_id = Material.create(name, template, options);

    // Get the material
    const material = Material.get(standard_material.material_id);

    // Create a combined uniform buffer for the default material
    // Contains: color (vec4) and emission (float, aligned to vec4)
    standard_material.material_params_data = new Float32Array([
      // color: vec4 (RGBA)
      0.5, 0.5, 0.5, 1.0,
      // normal: vec4 (RGBA)
      0.0, 0.0, 1.0, 1.0,
      // emission_roughness_metallic_tiling
      0.2, 0.7, 0.3, 1.0,
      // ao_height_specular_padding
      0.1, 0.0, 0.1, 0.0,
      // texture flags 1: vec4 (albedo, normal, roughness, metallic)
      0.0, 0.0, 0.0, 0.0,
      // texture flags 2: vec4 (ao, height, specular, emission)
      0.0, 0.0, 0.0, 0.0,
    ]);

    standard_material.material_params_buffer = Buffer.create({
      name: `${name}_material_params_buffer`,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      raw_data: standard_material.material_params_data,
    });

    // Set the uniform buffer for the material
    material.set_uniform_data("material_params", standard_material.material_params_buffer);

    // Set default parameter values
    standard_material.set_albedo(params.albedo || [1, 1, 1, 1], params.albedo_texture || null);
    standard_material.set_normal(params.normal || [0, 0, 1, 1], params.normal_texture || null);
    standard_material.set_roughness(
      params.roughness !== undefined ? params.roughness : 0.7,
      params.roughness_texture || null
    );
    standard_material.set_metallic(
      params.metallic !== undefined ? params.metallic : 0.3,
      params.metallic_texture || null
    );
    standard_material.set_emission(
      params.emission !== undefined ? params.emission : 0.2,
      params.emission_texture || null
    );
    standard_material.set_ao(params.ao !== undefined ? params.ao : 0.1, params.ao_texture || null);
    standard_material.set_height(
      params.height !== undefined ? params.height : 0.0,
      params.height_texture || null
    );
    standard_material.set_specular(
      params.specular !== undefined ? params.specular : 0.1,
      params.specular_texture || null
    );

    return standard_material;
  }

  set_albedo(color, texture = null) {
    const material = Material.get(this.material_id);

    if (texture) {
      material.set_texture_data("albedo", texture);
      if (texture.config.material_notifier) {
        material.listen_for_texture_data(texture.config.material_notifier);
      }
    }

    this.material_params_data[0] = color[0];
    this.material_params_data[1] = color[1];
    this.material_params_data[2] = color[2];
    this.material_params_data[3] = color[3];
    this.material_params_data[16] = texture ? 1 : 0;

    this.update_texture_flags();
  }

  set_normal(normal, texture = null) {
    const material = Material.get(this.material_id);

    if (texture) {
      material.set_texture_data("normal", texture);
      if (texture.config.material_notifier) {
        material.listen_for_texture_data(texture.config.material_notifier);
      }
    }

    this.material_params_data[4] = normal[0];
    this.material_params_data[5] = normal[1];
    this.material_params_data[6] = normal[2];
    this.material_params_data[7] = normal[3];
    this.material_params_data[17] = texture ? 1 : 0;

    this.update_texture_flags();
  }

  set_roughness(roughness, texture = null) {
    const material = Material.get(this.material_id);

    if (texture) {
      material.set_texture_data("roughness", texture);
      if (texture.config.material_notifier) {
        material.listen_for_texture_data(texture.config.material_notifier);
      }
    }

    this.material_params_data[9] = roughness;
    this.material_params_data[18] = texture ? 1 : 0;

    this.update_texture_flags();
  }

  set_metallic(metallic, texture = null) {
    const material = Material.get(this.material_id);

    if (texture) {
      material.set_texture_data("metallic", texture);
      if (texture.config.material_notifier) {
        material.listen_for_texture_data(texture.config.material_notifier);
      }
    }

    this.material_params_data[10] = metallic;
    this.material_params_data[19] = texture ? 1 : 0;

    this.update_texture_flags();
  }

  set_ao(ao, texture = null) {
    const material = Material.get(this.material_id);

    if (texture) {
      material.set_texture_data("ao", texture);
      if (texture.config.material_notifier) {
        material.listen_for_texture_data(texture.config.material_notifier);
      }
    }

    this.material_params_data[12] = ao;
    this.material_params_data[20] = texture ? 1 : 0;

    this.update_texture_flags();
  }

  set_height(height, texture = null) {
    const material = Material.get(this.material_id);

    if (texture) {
      material.set_texture_data("height", texture);
      if (texture.config.material_notifier) {
        material.listen_for_texture_data(texture.config.material_notifier);
      }
    }

    this.material_params_data[13] = height;
    this.material_params_data[21] = texture ? 1 : 0;

    this.update_texture_flags();
  }

  set_specular(specular, texture = null) {
    const material = Material.get(this.material_id);

    if (texture) {
      material.set_texture_data("specular", texture);
      if (texture.config.material_notifier) {
        material.listen_for_texture_data(texture.config.material_notifier);
      }
    }

    this.material_params_data[14] = specular;
    this.material_params_data[22] = texture ? 1 : 0;

    this.update_texture_flags();
  }

  set_emission(emission, texture = null) {
    const material = Material.get(this.material_id);

    if (texture) {
      material.set_texture_data("emission", texture);
      if (texture.config.material_notifier) {
        material.listen_for_texture_data(texture.config.material_notifier);
      }
    }

    this.material_params_data[8] = emission;
    this.material_params_data[23] = texture ? 1 : 0;

    this.update_texture_flags();
  }

  set_tiling(tiling) {
    this.material_params_data[11] = tiling;

    this.update_texture_flags();
  }

  update_texture_flags() {
    this.material_params_buffer.write_raw(this.material_params_data);
  }
}

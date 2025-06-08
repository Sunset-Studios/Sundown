import { Renderer } from "./renderer.js";
import { Shader } from "./shader.js";
import { FragmentGpuBuffer } from "../core/ecs/solar/memory.js";
import { BindGroup } from "./bind_group.js";
import { PipelineState } from "./pipeline_state.js";
import { ResourceCache } from "./resource_cache.js";
import { profile_scope } from "../utility/performance.js";
import { hash_data_map, hash_value } from "../utility/hashing.js";
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

  name = null;
  shader = null;
  depth_shader = null;
  pipeline_state_config = null;
  resources = [];
  parent = null;
  family = null;
  pipeline_state = null;
  depth_pipeline_state = null;

  constructor(
    name,
    shader,
    depth_shader = null,
    family = MaterialFamilyType.Opaque,
    pipeline_state_config = {},
    parent = null
  ) {
    this.name = name;
    this.shader = shader;
    this.depth_shader = depth_shader;
    this.pipeline_state_config = pipeline_state_config;
    this.parent = parent;
    this.family = family;
  }

  add_resource(resource) {
    this.resources.push(resource);
  }

  get base_reflection() {
    return this.shader.reflection;
  }

  get depth_reflection() {
    return this.depth_shader.reflection;
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

    if (parent_name) {
      parent = this.get_template(parent_name);
      if (!parent) {
        throw new Error(`Parent template '${parent_name}' not found`);
      }
      shader = parent.shader;
    }

    if (family === MaterialFamilyType.Transparent) {
      defines["TRANSPARENT"] = true;
    }
    if (shader_path) {
      shader = Shader.create(shader_path, defines);
      depth_shader = Shader.create(shader_path, { ...defines, DEPTH_ONLY: true });
    }

    const template = new MaterialTemplate(
      key,
      shader,
      depth_shader,
      family,
      pipeline_state_config,
      parent
    );

    if (parent) {
      template.resources = [...parent.resources];
    }

    // Reflect on shader and add resources
    const groups = template.base_reflection ? template.base_reflection.getBindGroups() : [];
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

    this.templates.set(key, template);

    return template;
  }

  create_pipeline_state(
    bind_group_layouts,
    output_targets = [],
    depth_stencil_options = {},
    is_depth_only = false
  ) {
    const pipeline_name = is_depth_only ? `${this.name}_depth` : this.name;
    const shader_module = is_depth_only ? this.depth_shader.module : this.shader.module;
    let all_bind_group_layouts = [bind_group_layouts[0]];
    let ref = is_depth_only ? this.depth_reflection : this.base_reflection;

    // Set material binding group inputs
    const groups = ref.getBindGroups();
    if (BindGroupType.Pass < groups.length) {
      for (let i = BindGroupType.Pass; i < groups.length; i++) {
        const bind_group = groups[i];

        all_bind_group_layouts.push(
          BindGroup.create_layout(
            pipeline_name,
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

    const fragment_outputs = ref.entry.fragment[0].outputs;
    fragment_outputs.forEach((output, i) => {
      if (targets[i]) {
        return;
      }

      const target = {
        format: Shader.get_optimal_texture_format(output.type.format.name),
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
    } else if (depth_target) {
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
}

export class Material {
  static materials = new Map();

  constructor(name, template, parent_id = null) {
    this.name = name;
    this.template = template;
    this.pipeline_state = null;
    this.depth_pipeline_state = null;
    this.bind_group = null;
    this.depth_bind_group = null;
    this.parent = parent_id;
    this.uniform_data = new Map();
    this.storage_data = new Map();
    this.texture_data = new Map();
    this.sampler_data = new Map();
    this.data_listeners = new Set();
    this.state_hash = 0;
    this.bind_group_update_flags = 0;
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

  set needs_bind_group_update(value) {
    this.bind_group_update_flags = value ? 3 : 0;
  }

  _update_state_hash() {
    profile_scope("Material._update_state_hash", () => {
      let hash = hash_value(this.template.name);
      hash = hash_data_map(this.uniform_data, hash);
      hash = hash_data_map(this.storage_data, hash);
      hash = hash_data_map(this.texture_data, hash);
      hash = hash_data_map(this.sampler_data, hash);
      this.state_hash = hash;
    });
  }

  _refresh_bind_group(is_depth_only = false) {
    if (this.bind_group_update_flags === 0) {
      return;
    }

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

    if (is_depth_only && (this.bind_group_update_flags & 1) !== 0 && this.depth_pipeline_state) {
      this.depth_bind_group = BindGroup.create(
        `${this.name}_depth`,
        this.depth_pipeline_state,
        BindGroupType.Material,
        entries,
        true /* force */
      );
      this.bind_group_update_flags = this.bind_group_update_flags & ~1;
    }

    if (!is_depth_only && (this.bind_group_update_flags & 2) !== 0 && this.pipeline_state) {
      this.bind_group = BindGroup.create(
        this.name,
        this.pipeline_state,
        BindGroupType.Material,
        entries,
        true /* force */
      );
      this.bind_group_update_flags = this.bind_group_update_flags & ~2;
    }
  }

  update_pipeline_state(bind_groups, output_targets = [], is_depth_only = false) {
    const renderer = Renderer.get();
    const depth_prepass_enabled = renderer.is_depth_prepass_enabled();

    // Build the main pipeline state
    const layouts = bind_groups.filter((bg) => bg !== null).map((bg) => bg.layout);
    if (is_depth_only) {
      this.depth_pipeline_state = this.template.create_pipeline_state(
        layouts,
        output_targets,
        { depth_write_enabled: true, depth_compare: "less" },
        true /* is_depth_only */
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
        false /* is_depth_only */
      );
    }
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

  bind(render_pass, bind_groups = [], output_targets = [], is_depth_only = false) {
    let pso = is_depth_only ? this.depth_pipeline_state : this.pipeline_state;
    if (!pso && !this.parent && bind_groups.length > 0 && output_targets.length > 0) {
      this.update_pipeline_state(bind_groups, output_targets, is_depth_only);
      for (let i = 0; i < bind_groups.length; i++) {
        if (bind_groups[i]) {
          bind_groups[i].bind(render_pass);
        }
      }
    }

    pso = is_depth_only ? this.depth_pipeline_state : this.pipeline_state;
    const parent_material = Material.get(this.parent);
    if (parent_material) {
      pso = is_depth_only ? parent_material.depth_pipeline_state : parent_material.pipeline_state;
    }

    this._refresh_bind_group(is_depth_only);

    if (pso) {
      render_pass.set_pipeline(pso);
    }

    const bind_group = is_depth_only ? this.depth_bind_group : this.bind_group;
    if (bind_group) {
      bind_group.bind(render_pass);
    }
  }

  get_state_hash() {
    return this.state_hash;
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
      if (options.writes_entity_id !== undefined && options.writes_entity_id !== null) {
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
      this.#default_ui_material = Material.create("DefaultUIMaterial", "DefaultUIMaterial");

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
export class StandardMaterial {
  material_id = null;
  material_params_data = null;
  material_params_buffer = null;

  static create(name, params = {}, options = {}, template = null) {
    if (!template) {
      const family = options.family !== undefined ? options.family : MaterialFamilyType.Opaque;
      MaterialTemplate.create("StandardMaterial", "standard_material.wgsl", family);
      template = `StandardMaterial`;
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

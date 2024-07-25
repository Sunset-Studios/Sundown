import { Shader, ShaderResourceType } from "./shader.js";
import { BindGroup, BindGroupType } from "./bind_group.js";
import { PipelineState } from "./pipeline_state.js";
import { ResourceCache, CacheTypes } from "./resource_cache.js";
import { profile_scope } from "../utility/performance.js";
import { hash_data, hash_value } from "../utility/hashing.js";
import { Texture } from "./texture.js";

export class MaterialTemplate {
  static templates = new Map();

  constructor(
    context,
    name,
    shader,
    pipeline_state_config = {},
    parent = null
  ) {
    this.context = context;
    this.name = name;
    this.shader = shader;
    this.pipeline_state_config = pipeline_state_config;
    this.resources = [];
    this.parent = parent;
  }

  add_resource(resource) {
    this.resources.push(resource);
  }

  get reflection() {
    return this.shader.reflection;
  }

  static create(
    context,
    name,
    shader_path,
    pipeline_state_config = {},
    parent_name = null
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
      shader = Shader.create(context, shader_path);
    }

    const template = new MaterialTemplate(
      context,
      name,
      shader,
      pipeline_state_config,
      parent
    );

    if (parent) {
      template.resources = [...parent.resources];
    }

    // Reflect on shader and add resources
    const groups = template.reflection
      ? template.reflection.getBindGroups()
      : [];
    if (BindGroupType.Material < groups.length) {
      const material_group = groups[BindGroupType.Material];
      for (let i = 0; i < material_group.length; i++) {
        const binding = material_group[i];
        const binding_type = Shader.resource_type_from_reflection_type(
          binding.resourceType
        );
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

  create_pipeline_state(context, bind_group_layouts, output_targets = []) {
    let all_bind_group_layouts = [...bind_group_layouts];

    // Set material binding group inputs
    const groups = this.reflection.getBindGroups();
    if (BindGroupType.Material < groups.length) {
      const material_group = groups[BindGroupType.Material];

      all_bind_group_layouts.push(
        BindGroup.create_layout(
          context,
          this.name,
          material_group.map((binding) => {
            let binding_obj = {};
            const binding_type = Shader.resource_type_from_reflection_type(
              binding.resourceType
            );
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
                    type: "read-only-storage",
                  },
                };
                break;
              case ShaderResourceType.Texture:
                binding_obj = {
                  texture: {
                    viewDimension: Texture.dimension_from_type_name(
                      binding.type.name
                    ),
                    sampleType: binding.type.name.includes("depth")
                      ? "depth"
                      : "float",
                  },
                };
                break;
              case ShaderResourceType.StorageTexture:
                binding_obj = {
                  storageTexture: {
                    viewDimension: Texture.dimension_from_type_name(
                      binding.type.name
                    ),
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

    // Set material shader fragment output targets
    const targets = output_targets
      .filter((target) => target.config.type !== "depth")
      .map((target) => ({
        name: target.config.name,
        format: target.config.format,
      }));

    const fragment_outputs = this.reflection.entry.fragment[0].outputs;
    fragment_outputs.forEach((output, i) => {
      if (targets[i]) {
        return;
      }

      const target = {
        format: Shader.get_optimal_texture_format(output.type.name),
      };

      if (
        this.pipeline_state_config.targets &&
        i < this.pipeline_state_config.targets.length
      ) {
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
          (this.pipeline_state_config.vertex &&
            this.pipeline_state_config.vertex.entry_point) ||
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
        topology:
          this.pipeline_state_config.primitive_topology_type || "triangle-list",
        cullMode:
          this.pipeline_state_config.rasterizer_state?.cull_mode || "back",
      },
    };

    const depth_target = output_targets.find(target => target.config.type === "depth")
    if (this.pipeline_state_config.depth_stencil_target) {
      pipeline_descriptor.depthStencil = {
        format:
          this.pipeline_state_config.depth_stencil_target.format ??
          "depth32float",
        depthWriteEnabled:
          this.pipeline_state_config.depth_stencil_target.depth_write_enabled ??
          true,
        depthCompare:
          this.pipeline_state_config.depth_stencil_target.depth_compare ??
          "less",
      };
    } else if (depth_target) {
      pipeline_descriptor.depthStencil = {
        format: depth_target.config.format,
        depthWriteEnabled: true,
        depthCompare: "less",
      };
    }

    return PipelineState.create_render(context, this.name, pipeline_descriptor);
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

  constructor(template) {
    this.template = template;
    this.pipeline_state = null;
    this.bind_group = null;
    this.uniform_data = new Map();
    this.storage_data = new Map();
    this.texture_data = new Map();
    this.sampler_data = new Map();
    this.state_hash = 0;
    this.needs_bind_group_update = false;
    this._update_state_hash();
  }

  _update_state_hash() {
    profile_scope("Material._update_state_hash", () => {
      Material.materials.delete(this.state_hash);
      ResourceCache.get().remove(CacheTypes.MATERIAL, this.state_hash);

      let hash = hash_value(this.template.name);
      hash = hash_data(this.uniform_data, hash);
      hash = hash_data(this.storage_data, hash);
      hash = hash_data(this.texture_data, hash);
      hash = hash_data(this.sampler_data, hash);
      this.state_hash = hash;

      ResourceCache.get().store(CacheTypes.MATERIAL, this.state_hash, this);
      Material.materials.set(this.state_hash, this);
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
          return {
            binding: resource.binding,
            resource: this.texture_data.get(resource.name).view,
          };
        case ShaderResourceType.Sampler:
          return {
            binding: resource.binding,
            resource: this.sampler_data.get(resource.name),
          };
      }
    });

    this.bind_group = BindGroup.create(
      this.template.context,
      this.template.name,
      this.pipeline_state,
      BindGroupType.Material,
      entries,
      true /* force */
    );
  }

  update_pipeline_state(bind_groups, output_targets = []) {
    this.pipeline_state = this.template.create_pipeline_state(
      this.template.context,
      bind_groups
        .filter((bind_group) => bind_group !== null)
        .map((bind_group) => bind_group.layout),
      output_targets
    );
  }

  set_uniform_data(name, data) {
    this.uniform_data.set(name, data);
    this._update_state_hash();
    this.needs_bind_group_update = true;
  }

  set_storage_data(name, data) {
    this.storage_data.set(name, data);
    this._update_state_hash();
    this.needs_bind_group_update = true;
  }

  set_texture_data(name, texture) {
    this.texture_data.set(name, texture);
    this._update_state_hash();
    this.needs_bind_group_update = true;
  }

  set_sampler_data(name, sampler) {
    this.sampler_data.set(name, sampler);
    this._update_state_hash();
    this.needs_bind_group_update = true;
  }

  bind(render_pass, bind_groups = [], output_targets = []) {
    if (
      !this.pipeline_state &&
      bind_groups.length > 0 &&
      output_targets.length > 0
    ) {
      this.update_pipeline_state(bind_groups, output_targets);
      bind_groups.forEach((bind_group) => {
        if (bind_group) {
          bind_group.bind(render_pass);
        }
      });
    }

    if (this.needs_bind_group_update) {
      this._refresh_bind_group();
      this.needs_bind_group_update = false;
    }

    if (this.pipeline_state) {
      render_pass.set_pipeline(this.pipeline_state);
    }

    if (this.bind_group) {
      this.bind_group.bind(render_pass);
    }
  }

  get_state_hash() {
    return this.state_hash;
  }

  static create(template_name) {
    const template = MaterialTemplate.get_template(template_name);
    if (!template) {
      throw new Error(`Material template '${template_name}' not found`);
    }
    return new Material(template);
  }

  static get(material_id) {
    return Material.materials.get(material_id);
  }
}

// Usage example
// const context = Renderer.get().graphics_context;

// // Create a material template
// const shader_path = "standard.wgsl";
// const template = MaterialTemplate.create(
//   context,
//   "StandardMaterial",
//   shader_path
// );

// // Add resources to the template
// template.add_resource({
//   type: ShaderResourceType.Uniform,
//   name: "model_view_projection",
//   binding: 0,
// });
// template.add_resource({
//   type: ShaderResourceType.Texture,
//   name: "albedo_texture",
//   binding: 1,
// });
// template.add_resource({
//   type: ShaderResourceType.Sampler,
//   name: "texture_sampler",
//   binding: 2,
// });

// // Create a material instance
// const material = Material.create("StandardMaterial");

// // Set material instance data
// material.set_uniform_data("model_view_projection", new Buffer(/* ... */));
// material.set_texture_data("albedo_texture", new Texture(/* ... */));
// material.set_sampler_data("texture_sampler", device.create_sampler(/* ... */));

// // Use the material in rendering
// function render(render_pass) {
//   // ... other rendering setup ...
//   material.bind(render_pass);
//   // ... draw calls ...
// }

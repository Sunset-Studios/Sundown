import { Shader, ShaderResourceType, ShaderResource } from "./shader.js";
import { BindGroup } from "./bind_group.js";
import { PipelineState } from "./pipeline_state.js";
import { Shader } from "./shader.js";
import { Texture } from "./texture.js";
import { Buffer } from "./buffer.js";

export class MaterialTemplate {
  static templates = new Map();

  constructor(context, name, shader, pipeline_state) {
    this.context = context;
    this.name = name;
    this.shader = shader;
    this.pipeline_state = pipeline_state;
    this.resources = [];
  }

  add_resource(resource) {
    this.resources.push(resource);
  }

  static create(context, name, shader_path, pipeline_state_config) {
    if (this.templates.has(name)) {
      return this.templates.get(name);
    }
    const pipeline_state = PipelineState.create(
      context,
      name,
      pipeline_state_config
    );
    const shader = Shader.create(context, shader_path);
    const template = new MaterialTemplate(
      context,
      name,
      shader,
      pipeline_state
    );

    this.templates.set(name, template);

    return template;
  }

  static get_template(name) {
    return this.templates.get(name);
  }
}

export class Material {
  constructor(template) {
    this.template = template;
    this.uniform_data = new Map();
    this.storage_data = new Map();
    this.texture_data = new Map();
    this.sampler_data = new Map();
    this.state_hash = 0;
    this._update_state_hash();
  }

  _update_state_hash() {
    let hash = this._hash_value(this.template.name);
    hash = this._hash_data(this.uniform_data, hash);
    hash = this._hash_data(this.storage_data, hash);
    hash = this._hash_data(this.texture_data, hash);
    hash = this._hash_data(this.sampler_data, hash);
    this.state_hash = hash;
  }

  _hash_data(data_map, initial_hash) {
    let hash = initial_hash;
    for (const [key, value] of data_map) {
      hash = ((hash << 5) - hash) + this._hash_value(key);
      hash = ((hash << 5) - hash) + this._hash_value(value);
      hash |= 0; // Convert to 32-bit integer
    }
    return hash;
  }

  _hash_value(value) {
    if (typeof value === 'number') {
      return value;
    } else if (typeof value === 'string') {
      return value.split('').reduce((acc, char) => {
        return ((acc << 5) - acc) + char.charCodeAt(0);
      }, 0);
    } else if (value instanceof Buffer || value instanceof Texture) {
      return value.config.name.split('').reduce((acc, char) => {
        return ((acc << 5) - acc) + char.charCodeAt(0);
      }, 0);
    } else {
      return 0;
    }
  }

  set_uniform_data(name, data) {
    this.uniform_data.set(name, data);
    this._update_state_hash();
  }

  set_storage_data(name, data) {
    this.storage_data.set(name, data);
    this._update_state_hash();
  }

  set_texture_data(name, texture) {
    this.texture_data.set(name, texture);
    this._update_state_hash();
  }

  set_sampler_data(name, sampler) {
    this.sampler_data.set(name, sampler);
    this._update_state_hash();
  }

  bind(render_pass, bind_group) {
    const entries = this.template.resources.map((resource) => {
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

    const bind_group = BindGroup.create(
      this.template.context,
      this.template.name,
      this.template.pipeline_state,
      bind_group,
      entries
    );
    bind_group.bind(render_pass);
  }

  create(template_name) {
    const template = MaterialTemplate.get_template(template_name);
    if (!template) {
      throw new Error(`Material template '${template_name}' not found`);
    }
    return new Material(template);
  }

  get_state_hash() {
    return this.state_hash;
  }
}

// Usage example
// const context = Renderer.get().graphics_context;

// // Create a material template
// const shader_path = "standard.wgsl";
// const pipeline_state_config = {
//   /* ... */
// };
// const template = MaterialTemplate.create(
//   context,
//   "StandardMaterial",
//   shader_path,
//   pipeline_state_config
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
//   material.bind(render_pass, 0);
//   // ... draw calls ...
// }
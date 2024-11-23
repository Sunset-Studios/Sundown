// USAGE EXAMPLE
//
// 1. Define your fragment configuration
// const mesh_fragment_config = {
//     name: 'Mesh',
//     constants: {
//         some_constant: 64,
//     },
//     fields: {
//         position: {
//             type: DataType.FLOAT32,
//             vector: { x: true, y: true, z: true },
//             stride: 1
//         }
//     },
//     buffers: {
//         vertices: {
//             type: DataType.FLOAT32,
//             usage: BufferType.VERTEX,
//             stride: 3
//         }
//     },
//     overrides: {
//         // Override with pre/post hooks
//         initialize: {
//             pre: `console.log('Before initialization');`,
//             post: `
//                 // Additional initialization
//                 this.data.customCache = new Map();
//                 console.log('After initialization');
//             `
//         },
//         // Complete override
//         resize: {
//             skipDefault: true,
//             pre: `
//                 console.log('Custom resize implementation');
//                 if (!this.data) this.initialize();
//                 super.resize(new_size);

//                 // Custom resize logic
//                 this.data.customCache.clear();
//                 this.rebuild_buffers(Renderer.get().graphics_context);
//             `
//         },
//         // Extend default behavior
//         update_entity_data: {
//             post: `
//                 // Additional entity update logic
//                 if (data.position) {
//                     this.data.customCache.set(entity, {
//                         lastUpdated: Date.now(),
//                         position: { ...data.position }
//                     });
//                 }
//             `
//         }
//     },
//     custom_methods: {
//         get_last_update: {
//             params: 'entity',
//             body: `
//                 return this.data.customCache.get(entity)?.lastUpdated || null;
//             `
//         }
//     },
//     hooks: {
//         on_post_render: {
//             params: 'context',
//             body: `
//                 console.log('Post render hook');
//             `
//         }
//     }
// };
//
// // 2. Generate the fragment
// const fragment_code = FragmentGenerator.generate(mesh_fragment_config);
//
// // 3. Either save to file or eval (for development)
//
// // 4. Use the generated fragment
// const mesh_fragment = new MeshFragment();

import { BufferType } from "./fragment_generator_types.js";

export class FragmentGenerator {
  static generate(config, relative_config_path = '') {
    const {
      name,
      imports = {},
      constants = {},
      fields = {},
      members = {},
      buffers = {},
      custom_methods = {},
      hooks = {},
      overrides = {},
    } = config;

    const all_imports = {
      EntityLinearDataContainer: "../entity_utils.js",
      Fragment: "../fragment.js",
      Renderer: "../../../renderer/renderer.js",
      Buffer: "../../../renderer/buffer.js",
      global_dispatcher: "../../../core/dispatcher.js",
    };

    const implementations = {
      initialize: this.generate_initialize(fields, members, buffers, hooks, overrides.initialize),
      resize: this.generate_resize(fields, buffers, overrides.resize),
      add_entity: this.generate_add_entity(overrides.add_entity),
      remove_entity: this.generate_remove_entity(fields, overrides.remove_entity),
      get_entity_data: this.generate_get_entity_data(fields, overrides.get_entity_data),
      duplicate_entity_data: this.generate_duplicate_entity_data(
        fields,
        overrides.duplicate_entity_data
      ),
      update_entity_data: this.generate_update_entity_data(fields, overrides.update_entity_data),
      to_gpu_data: this.generate_to_gpu_data(buffers, overrides.to_gpu_data),
      rebuild_buffers: this.generate_rebuild_buffers(buffers, overrides.rebuild_buffers),
      sync_buffers: this.generate_sync_buffers(buffers, overrides.sync_buffers),
    };

    let adjusted_imports = Object.entries(all_imports).reduce((acc, [name, path]) => {
      const adjusted_path = relative_config_path ? relative_config_path + '/' + path : path;
      let final_path = adjusted_path.startsWith('.') ? adjusted_path : '.' + adjusted_path;
      final_path = final_path.replaceAll('\\', '/');
      acc[name] = final_path;
      return acc;
    }, {});

    adjusted_imports = { ...adjusted_imports, ...imports };

    const import_statements = Object.entries(adjusted_imports)
      .map(([name, path]) => `import { ${name} } from "${path}";`)
      .join("\n");

    return `
${import_statements}

${Object.entries(buffers)
      .map(([key, value]) => `
        const ${key}_buffer_name = "${key}_buffer";
        const ${key}_cpu_buffer_name = "${key}_cpu_buffer";
        const ${key}_event = "${key}";
        const ${key}_update_event = "${key}_update";
      `)
      .join("\n")}

export class ${name}Fragment extends Fragment {
    ${Object.entries(constants)
      .map(([key, value]) => `static ${key} = ${value};`)
      .join("\n")}
    ${implementations.initialize}
    ${implementations.resize}
    ${implementations.add_entity}
    ${implementations.remove_entity}
    ${implementations.get_entity_data}
    ${implementations.duplicate_entity_data}
    ${implementations.update_entity_data}
    ${implementations.to_gpu_data}
    ${implementations.rebuild_buffers}
    ${implementations.sync_buffers}
    ${this.generate_custom_methods(custom_methods)}
    ${this.generate_hooks(hooks)}
}`;
  }

  static generate_initialize(fields, members, buffers, hooks, override) {
    if (override) {
      return `
    static initialize() {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_initialize(fields, members, buffers, hooks) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static initialize() {
        ${this.get_default_initialize(fields, members, buffers, hooks)}
    }`;
  }

  static get_default_initialize(fields, members, buffers, hooks) {
    const field_inits = Object.entries(fields).map(([key, field]) => {
      if (field.is_container) {
        return `${key}: new EntityLinearDataContainer(${field.type.array.name || "Uint32Array"})`;
      }
      if (field.vector) {
        return `${key}: {
                ${Object.keys(field.vector)
                  .map((axis) => `${axis}: new ${field.type.array.name}(1)`)
                  .join(",\n")}
            }`;
      }
      return `${key}: new ${field.type.array.name}(${field.stride || 1})`;
    });

    const member_inits = Object.entries(members).map(([key, member]) => `${key}: ${member}`);

    const buffer_inits = Object.keys(buffers).map(
      (key) => `${key}_buffer: null${buffers[key].cpu_buffer ? `,\n${key}_cpu_buffer: null` : ""}`
    );

    return `
        this.data = {
            ${[...field_inits, ...member_inits, ...buffer_inits].join(",\n            ")},
            gpu_data_dirty: true
        };
        ${
          hooks.on_post_render
            ? `Renderer.get().on_post_render(this.on_post_render.bind(this));`
            : ""
        }
        ${
          Object.keys(buffers).length > 0
            ? `this.rebuild_buffers(Renderer.get().graphics_context);`
            : ""
        }
    `;
  }

  static generate_resize(fields, buffers, override) {
    if (override) {
      return `
    static resize(new_size) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_resize(fields, buffers) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static resize(new_size) {
        ${this.get_default_resize(fields, buffers)}
    }`;
  }

  static get_default_resize(fields, buffers) {
    return `
        if (!this.data) this.initialize();
        super.resize(new_size);

        ${Object.entries(fields)
          .map(([key, field]) => {
            if (field.vector) {
              return `Object.keys(this.data.${key}).forEach(axis => {
                    Fragment.resize_array(this.data.${key}, axis, new_size, ${field.type.array.name});
                });`;
            }
            if (!field.is_container) {
              return `Fragment.resize_array(this.data, "${key}", new_size, ${
                field.type.array.name
              }, ${field.stride || 1});`;
            }
          })
          .join("\n        ")}
        
        ${Object.keys(buffers).length === 0 ? "" : `this.rebuild_buffers(Renderer.get().graphics_context);`}
    `;
  }

  static generate_add_entity(override) {
    if (override) {
      return `
    static add_entity(entity, data) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_add_entity() : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static add_entity(entity, data) {
        ${this.get_default_add_entity()}
    }`;
  }

  static get_default_add_entity() {
    return `super.add_entity(entity, data);`;
  }

  static generate_remove_entity(fields, override) {
    if (override) {
      return `
    static remove_entity(entity) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_remove_entity(fields) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static remove_entity(entity) {
        ${this.get_default_remove_entity(fields)}
    }`;
  }

  static get_default_remove_entity(fields) {
    const field_resets = Object.entries(fields)
      .map(([key, field]) => {
        if (field.is_container) {
          return `// ${key} will be handled separately`;
        }
        if (field.vector) {
          return `${key}: {
                    ${Object.keys(field.vector)
                      .map(
                        (axis) =>
                          `${axis}: ${field.stride > 1 ? `Array(${field.stride}).fill(${field.default ? field.default[axis] : 0})` : field.default ? field.default[axis] : 0}`
                      )
                      .join(",\n")}
                }`;
        }
        return `${key}: ${field.stride > 1 ? `Array(${field.stride}).fill(${field.default || 0})` : field.default || 0}`;
      })
      .filter((reset) => !reset.startsWith("//"));

    return `
        super.remove_entity(entity);
        this.update_entity_data(entity, {
            ${field_resets.join(",\n            ")}
        });
        ${Object.entries(fields)
          .filter(([_, field]) => field.is_container)
          .map(([key, _]) => `this.data.${key}.remove(entity);`)
          .join("\n        ")}
    `;
  }

  static generate_get_entity_data(fields, override) {
    if (override) {
      return `
    static get_entity_data(entity) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_get_entity_data(fields) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static get_entity_data(entity) {
        ${this.get_default_get_entity_data(fields)}
    }`;
  }

  static get_default_get_entity_data(fields) {
    return `return super.get_entity_data(entity);`;
  }

  static generate_duplicate_entity_data(fields, override) {
    if (override) {
      return `
    static duplicate_entity_data(entity) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_duplicate_entity_data(fields) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static duplicate_entity_data(entity) {
        ${this.get_default_duplicate_entity_data(fields)}
    }`;
  }

  static get_default_duplicate_entity_data(fields) {
    return `
        const data = {};
        ${Object.entries(fields)
          .map(([key, field]) => {
            if (field.is_container) {
              return `data.${key} = this.data.${key}.get_data_for_entity(entity);`;
            }
            if (field.vector) {
              return `data.${key} = {
                        ${Object.keys(field.vector)
                          .map((axis) => `${axis}: this.data.${key}.${axis}[entity]`)
                          .join(",\n        ")}
                    };`;
            }
            if (field.stride > 1) {
              return `data.${key} = Array(${field.stride}).fill(${field.default || 0});
                    for (let i = 0; i < ${field.stride}; i++) {
                        data.${key}[i] = this.data.${key}[entity * ${field.stride} + i];
                    }`;
            }
            return `data.${key} = this.data.${key}[entity];`;
          })
          .join("\n        ")}
        return data;
    `;
  }

  static generate_update_entity_data(fields, override) {
    if (override) {
      return `
    static update_entity_data(entity, data) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_update_entity_data(fields) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static update_entity_data(entity, data) {
        ${this.get_default_update_entity_data(fields)}
    }`;
  }

  static get_default_update_entity_data(fields) {
    const container_keys = Object.entries(fields)
      .filter(([_, field]) => field.is_container)
      .map(([key, _]) => `${key}`);

    const field_updates = `
      ${container_keys.length > 0 ? `const { ${container_keys.join(",")}, ...rest_of_data } = data;` : ""}
      super.update_entity_data(entity, ${container_keys.length > 0 ? "rest_of_data" : "data"});

      ${container_keys
        .map(
          (key) => `
        if (${key}) {
            this.data.${key}.update(entity, ${key});
        }
      `
        )
        .join("\n")}
    `;

    return `
        if (!this.data) {
            this.initialize();
        }
        ${field_updates}
        if (this.data.dirty) {
            this.data.dirty[entity] = 1;
        }
        this.data.gpu_data_dirty = true;
    `;
  }

  static generate_rebuild_buffers(buffers, override) {
    if (Object.keys(buffers).length === 0) return "";

    if (override) {
      return `
    static rebuild_buffers(context) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.generate_default_rebuild_buffers(buffers) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static rebuild_buffers(context) {
        ${this.generate_default_rebuild_buffers(buffers)}
    }`;
  }

  static generate_default_rebuild_buffers(buffers) {
    return `
        ${Object.entries(buffers)
          .map(
            ([key, buffer]) => `
        {
            ${
              buffer.gpu_data
                ? `${buffer.gpu_data}`
                : `const gpu_data = this.data.${key} ? this.data.${key} : new ${buffer.type.array.name}(this.size * ${buffer.stride || 1});`
            }
            if (!this.data.${key}_buffer || this.data.${key}_buffer.config.size < gpu_data.byteLength) {
                this.data.${key}_buffer = Buffer.create(context, {
                name: ${key}_buffer_name,
                usage: ${buffer.usage},
                raw_data: gpu_data,
                force: true
            });
            ${
              buffer.cpu_buffer
                ? `
                this.data.${key}_cpu_buffer = Buffer.create(context, {
                    name: ${key}_cpu_buffer_name,
                    usage: ${BufferType.CPU_READ},
                    raw_data: gpu_data,
                    force: true
                });`
                : ""
            }
            Renderer.get().mark_bind_groups_dirty(true);
            ${buffer.no_dispatch ? "" : `global_dispatcher.dispatch(${key}_event, this.data.${key}_buffer);`}
          } else {
            this.data.${key}_buffer.write(context, gpu_data);
          }

          ${buffer.no_dispatch ? "" : `global_dispatcher.dispatch(${key}_update_event);`}
        }
        `
          )
          .join("\n")}

        this.data.gpu_data_dirty = false;
    `;
  }

  static generate_sync_buffers(buffers, override) {
    if (Object.keys(buffers).length === 0) return "";

    if (override) {
      return `
    static async sync_buffers(context) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.generate_default_sync_buffers(buffers) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static async sync_buffers(context) {
        ${this.generate_default_sync_buffers(buffers)}
    }`;
  }

  static generate_default_sync_buffers(buffers) {
    return `
        ${Object.entries(buffers)
          .filter(([_, buf]) => buf.cpu_buffer)
          .map(
            ([key, _]) => `
        if (this.data.${key}_cpu_buffer?.buffer.mapState === "unmapped") {
            await this.data.${key}_cpu_buffer.read(
                context,
                this.data.${key},
                this.data.${key}.byteLength,
                0,
                0,
                ${buffers[key].type.array.name}
            );
        }`
          )
          .join("\n")}
    `;
  }

  static generate_to_gpu_data(buffers, override) {
    if (Object.keys(buffers).length === 0) return "";

    if (override) {
      return `
    static to_gpu_data(context) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.generate_default_to_gpu_data(buffers) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static to_gpu_data(context) {
        ${this.generate_default_to_gpu_data(buffers)}
    }`;
  }

  static generate_default_to_gpu_data(buffers) {
    return `
      if (!this.data) this.initialize();

      if (!this.data.gpu_data_dirty) {
          return {
                ${Object.keys(buffers)
                  .map((key) => `${key}_buffer: this.data.${key}_buffer`)
                  .join(",\n            ")}
            };
        }

      this.rebuild_buffers(context);

      return {
            ${Object.keys(buffers)
              .map((key) => `${key}_buffer: this.data.${key}_buffer`)
              .join(",\n            ")}
        };
    `;
  }

  static generate_custom_methods(methods) {
    return Object.entries(methods)
      .map(
        ([name, impl]) => `
    static ${name}(${impl.params || ""}) {
        ${impl.body}
    }`
      )
      .join("\n");
  }

  static generate_hooks(hooks) {
    return Object.entries(hooks)
      .map(
        ([hook, impl]) => `
    static async ${hook}(${impl.params || ""}) {
        ${impl.body}
    }`
      )
      .join("\n");
  }
}

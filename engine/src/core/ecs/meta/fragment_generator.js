// USAGE EXAMPLE
//
// 1. Define your fragment configuration
// const mesh_fragment_config = {
//     name: 'Mesh',
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
  static generate(config) {
    const {
      name,
      fields = {},
      buffers = {},
      custom_methods = {},
      hooks = {},
      overrides = {},
    } = config;

    const implementations = {
      initialize: this.generate_initialize(fields, buffers, hooks, overrides.initialize),
      resize: this.generate_resize(fields, buffers, overrides.resize),
      add_entity: this.generate_add_entity(overrides.add_entity),
      remove_entity: this.generate_remove_entity(fields, overrides.remove_entity),
      duplicate_entity_data: this.generate_duplicate_entity_data(
        fields,
        overrides.duplicate_entity_data
      ),
      update_entity_data: this.generate_update_entity_data(fields, overrides.update_entity_data),
      to_gpu_data: this.generate_to_gpu_data(buffers, overrides.to_gpu_data),
      buffer_management: this.generate_buffer_management(buffers, overrides.buffer_management),
    };

    return `
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";

export class ${name}Fragment extends Fragment {
    ${implementations.initialize}
    ${implementations.resize}
    ${implementations.add_entity}
    ${implementations.remove_entity}
    ${implementations.duplicate_entity_data}
    ${implementations.update_entity_data}
    ${implementations.to_gpu_data}
    ${implementations.buffer_management}
    ${this.generate_custom_methods(custom_methods)}
    ${this.generate_hooks(hooks)}
}`;
  }

  static generate_initialize(fields, buffers, hooks, override) {
    if (override) {
      return `
    static initialize() {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_initialize(fields, buffers, hooks) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static initialize() {
        ${this.get_default_initialize(fields, buffers, hooks)}
    }`;
  }

  static get_default_initialize(fields, buffers, hooks) {
    const field_inits = Object.entries(fields).map(([key, field]) => {
      if (field.vector) {
        return `${key}: {
                    ${Object.keys(field.vector)
                      .map((axis) => `${axis}: new ${field.type.array.name}(1)`)
                      .join(",\n")}
                }`;
      }
      return `${key}: new ${field.type.array.name}(${field.stride || 1})`;
    });

    const buffer_inits = Object.keys(buffers).map(
      (key) => `${key}_buffer: null${buffers[key].cpu_buffer ? `,\n${key}_cpu_buffer: null` : ""}`
    );

    return `
        this.data = {
            ${[...field_inits, ...buffer_inits].join(",\n            ")},
            dirty: new Uint8Array(1),
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
            return `Fragment.resize_array(this.data, "${key}", new_size, ${
              field.type.array.name
            }, ${field.stride || 1});`;
          })
          .join("\n        ")}
        
        Fragment.resize_array(this.data, "dirty", new_size, Uint8Array);
        
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
    return `
        super.remove_entity(entity);
        this.update_entity_data(entity, {
            ${Object.entries(fields)
              .map(([key, field]) => {
                if (field.vector) {
                  return `${key}: {
                    ${Object.keys(field.vector)
                      .map((axis) => `${axis}: ${field.default || 0}`)
                      .join(",\n")}
                }`;
                }
                return `${key}: ${field.default || 0}`;
              })
              .join(",\n        ")}
        });
    `;
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
        const data = this.get_entity_data(entity);
        return {
            ${Object.entries(fields)
              .map(([key, field]) => {
                if (field.vector) {
                  return `${key}: {
                    ${Object.keys(field.vector)
                      .map((axis) => `${axis}: data.${key}.${axis}`)
                      .join(",\n")}
                }`;
                }
                return `${key}: data.${key}`;
              })
              .join(",\n        ")}
        };
    `;
  }

  static generate_update_entity_data(override) {
    if (override) {
      return `
    static update_entity_data(entity, data) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_update_entity_data() : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static update_entity_data(entity, data) {
        ${this.get_default_update_entity_data()}
    }`;
  }

  static get_default_update_entity_data() {
    return `
        if (!this.data) {
            this.initialize();
        }
        super.update_entity_data(entity, data);
        this.data.dirty[entity] = 1;
        this.data.gpu_data_dirty = true;
    `;
  }

  static generate_buffer_management(buffers) {
    if (Object.keys(buffers).length === 0) return "";

    return `
    static rebuild_buffers(context) {
        ${Object.entries(buffers)
          .map(
            ([key, buffer]) => `
        const ${key}_size = this.size * ${buffer.stride || 1} * ${buffer.type.byte_size};
        if (!this.data.${key}_buffer || this.data.${key}_buffer.config.size < ${key}_size) {
            this.data.${key}_buffer = Buffer.create(context, {
                name: "${key}_buffer",
                usage: ${buffer.usage},
                raw_data: this.data.${key},
                force: true
            });
            ${
              buffer.cpu_buffer
                ? `
            this.data.${key}_cpu_buffer = Buffer.create(context, {
                name: "${key}_cpu_buffer",
                usage: ${BufferType.CPU_READ},
                raw_data: this.data.${key},
                force: true
            });`
                : ""
            }
            Renderer.get().mark_bind_groups_dirty(true);
        }`
          )
          .join("\n")}
    }

    static async sync_buffers(context) {
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
    }`;
  }

  static generate_to_gpu_data(buffers) {
    if (Object.keys(buffers).length === 0) return "";

    return `
    static to_gpu_data(context) {
        if (!this.data) this.initialize();
        return {
            ${Object.keys(buffers)
              .map((key) => `${key}_buffer: this.data.${key}_buffer`)
              .join(",\n            ")}
        };
    }`;
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
    static ${hook}(${impl.params || ""}) {
        ${impl.body}
    }`
      )
      .join("\n");
  }
}

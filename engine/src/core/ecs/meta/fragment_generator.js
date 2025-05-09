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
//             pre: `log('Before initialization');`,
//             post: `
//                 // Additional initialization
//                 this.data.customCache = new Map();
//                 log('After initialization');
//             `
//         },
//         // Complete override
//         resize: {
//             skipDefault: true,
//             pre: `
//                 log('Custom resize implementation');
//                 if (!this.data) this.initialize();

//                 // Custom resize logic
//                 this.data.customCache.clear();
//                 this.rebuild_buffers();
//             `
//         },
//         // Extend default behavior
//         duplicate_entity_data: {
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
//                 log('Post render hook');
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
  static generate(config, relative_config_path = "") {
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

    const fragment_name = name + "Fragment";

    const all_imports = {
      EntityLinearDataContainer: "../entity_utils.js",
      Fragment: "../fragment.js",
      Renderer: "../../../renderer/renderer.js",
      Buffer: "../../../renderer/buffer.js",
      global_dispatcher: "../../../core/dispatcher.js",
      RingBufferAllocator: "../../../memory/allocator.js",
      EntityID: "../entity.js",
      EntityManager: "../entity.js",
      MAX_BUFFERED_FRAMES: "../../../core/minimal.js",
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
      to_gpu_data: this.generate_to_gpu_data(buffers, overrides.to_gpu_data),
      rebuild_buffers: this.generate_rebuild_buffers(buffers, overrides.rebuild_buffers),
      sync_buffers: this.generate_sync_buffers(buffers, overrides.sync_buffers),
      copy_entity_instance: this.generate_copy_entity_instance(
        fields,
        overrides.copy_entity_instance
      ),
    };

    let adjusted_imports = Object.entries(all_imports).reduce((acc, [name, path]) => {
      const adjusted_path = relative_config_path ? relative_config_path + "/" + path : path;
      let final_path = adjusted_path.startsWith(".") ? adjusted_path : "." + adjusted_path;
      final_path = final_path.replaceAll("\\", "/");
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
  .map(
    ([key, value]) => `
        const ${key}_buffer_name = "${key}_buffer";
        const ${key}_cpu_buffer_name = "${key}_cpu_buffer";
        const ${key}_event = "${key}";
        const ${key}_update_event = "${key}_update";
      `
  )
  .join("\n")}

${Object.entries(fields)
  .filter(([_, field]) => field.vector)
  .map(
    ([key, field]) => `
    class ${key[0].toUpperCase() + key.slice(1)}DataView {
        constructor() {
            this.current_entity = -1n;
            this.absolute_entity = -1n;
        }

        ${Object.keys(field.vector)
          .map(
            (axis) => `
            get ${axis}() {
                return ${fragment_name}.data.${key}.${axis}[${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`}];
            }

            set ${axis}(value) {
                ${fragment_name}.data.${key}.${axis}[${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`}] = value;
                if (${fragment_name}.data.dirty) {
                    ${fragment_name}.data.dirty[${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`}] = 1;
                }
                ${fragment_name}.data.gpu_data_dirty = true;
            }
        `
          )
          .join("\n")}

        view_entity(entity, instance = 0) {
            this.current_entity = entity;
            this.absolute_entity = EntityID.get_absolute_index(entity) + instance;
            return this;
        }
    }
`
  )
  .join("\n")}

class ${name}DataView {
    current_entity = -1n;
    absolute_entity = -1n;

    constructor() {
      ${Object.entries(fields)
        .filter(([_, field]) => field.vector)
        .map(
          ([key, _]) => `this.${key} = new ${key[0].toUpperCase() + key.slice(1)}DataView(this);`
        )
        .join("\n")}
    }

    ${Object.entries(fields)
      .map(([key, field]) => {
        if (field.vector) return "";

        return `
      get ${key}() {
        ${
          field.getter
            ? field.getter
            : field.is_container
              ? `return ${fragment_name}.data.${key}.get_data_for_entity(${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`});`
              : `return ${fragment_name}.data.${key}[${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`}];`
        }
      }

      set ${key}(value) {
        ${
          field.readonly
            ? `throw new Error('Field ${key} is readonly');`
            : field.setter
              ? field.setter
              : field.is_container
                ? `${fragment_name}.data.${key}.update(${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`}, value ?? []);
                  if (${fragment_name}.data.dirty) {
                      ${fragment_name}.data.dirty[${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`}] = 1;
                  }
                  ${fragment_name}.data.gpu_data_dirty = true;`
                : `${fragment_name}.data.${key}[${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`}] = ${fragment_name}.data.${key} instanceof BigInt64Array ? BigInt(value) : value;
                  if (${fragment_name}.data.dirty) {
                      ${fragment_name}.data.dirty[${field.no_instance_count_resize ? `this.current_entity` : `this.absolute_entity`}] = 1;
                  }
                  ${fragment_name}.data.gpu_data_dirty = true;`
        }
      }`;
      })
      .join("\n")}

    view_entity(entity, instance = 0) {
      this.current_entity = entity;
      this.absolute_entity = EntityID.get_absolute_index(entity) + instance;

      ${Object.entries(fields)
        .filter(([_, field]) => field.vector)
        .map(([key, _]) => `this.${key}.view_entity(entity, instance);`)
        .join("\n")}

      return this;
    }
}

const unmapped_state = 'unmapped';

export class ${fragment_name} extends Fragment {
    static data_view_allocator = new RingBufferAllocator(256, ${name}DataView);
    static size = 0;
    static data = null;

    ${Object.entries(constants)
      .map(([key, value]) => `static ${key} = ${value};`)
      .join("\n")}
    ${implementations.initialize}
    ${implementations.resize}
    ${implementations.add_entity}
    ${implementations.remove_entity}
    ${implementations.get_entity_data}
    ${implementations.duplicate_entity_data}
    ${implementations.to_gpu_data}
    ${implementations.rebuild_buffers}
    ${implementations.sync_buffers}
    ${implementations.copy_entity_instance}
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
    const field_inits = Object.entries(fields)
      .filter(([_, field]) => !field.no_fragment_array)
      .map(([key, field]) => {
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
      (key) => `${key}_buffer: null${buffers[key].cpu_buffer ? `,\n${key}_cpu_buffer: Array(MAX_BUFFERED_FRAMES).fill(null)` : ""}`
    );

    return `
        this.data = {
            ${[...field_inits, ...member_inits, ...buffer_inits].join(",\n            ")},
            valid_prev: new Int32Array(1),
            valid_next: new Int32Array(1),
            first_valid_index: -1,
            last_valid_index: -1,
            gpu_data_dirty: true
        };
        ${
          hooks.on_post_render
            ? `Renderer.get().on_post_render(this.on_post_render.bind(this));`
            : ""
        }
        ${
          hooks.on_pre_render
            ? `Renderer.get().on_pre_render(this.on_pre_render.bind(this));`
            : ""
        }
        ${Object.keys(buffers).length > 0 ? `this.rebuild_buffers();` : ""}
    `;
  }

  static generate_resize(fields, buffers, override) {
    if (override) {
      return `
    static resize(new_size) {
        if (new_size <= this.size) return;

        this.size = new_size;
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_resize(fields, buffers) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static resize(new_size) {
        if (new_size <= this.size) return;

        new_size *= 2;
        this.size = new_size;
        ${this.get_default_resize(fields, buffers)}
    }`;
  }

  static get_default_resize(fields, buffers) {
    return `
        if (!this.data) this.initialize();

        ${Object.entries(fields)
          .filter(([_, field]) => !field.no_fragment_array)
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

        Fragment.resize_array(this.data, "valid_prev", new_size, Int32Array, 1);
        Fragment.resize_array(this.data, "valid_next", new_size, Int32Array, 1);
        
        ${Object.keys(buffers).length === 0 ? "" : `this.data.gpu_data_dirty = true;`}
    `;
  }

  static generate_add_entity(override) {
    if (override) {
      return `
    static add_entity(entity) {
        const absolute_entity = EntityID.get_absolute_index(entity);
        if (absolute_entity >= this.size) {
          this.resize(absolute_entity * 2);
        }

        const idx = Number(absolute_entity);
        const tail = this.data.last_valid_index;
        if (tail >= 0) {
          this.data.valid_next[tail] = idx;
          this.data.valid_prev[idx] = tail;
        } else {
          this.data.first_valid_index = idx;
          this.data.valid_prev[idx] = -1;
        }
        this.data.valid_next[idx] = -1;
        this.data.last_valid_index = idx;

        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_add_entity() : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static add_entity(entity) {
        const absolute_entity = EntityID.get_absolute_index(entity);
        if (absolute_entity >= this.size) {
          this.resize(absolute_entity * 2);
        }

        const idx = Number(absolute_entity);
        const tail = this.data.last_valid_index;
        if (tail >= 0) {
          this.data.valid_next[tail] = idx;
          this.data.valid_prev[idx] = tail;
        } else {
          this.data.first_valid_index = idx;
          this.data.valid_prev[idx] = -1;
        }
        this.data.valid_next[idx] = -1;
        this.data.last_valid_index = idx;

        ${this.get_default_add_entity()}
    }`;
  }

  static get_default_add_entity() {
    return `
        return this.get_entity_data(entity);
    `;
  }

  static generate_remove_entity(fields, override) {
    if (override) {
      return `
    static remove_entity(entity) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_remove_entity(fields) : ""}
        ${override.post ? override.post : ""}

        // unlink from the live‐list in O(1)
        const idx = Number(EntityID.get_absolute_index(entity));
        const p   = this.data.valid_prev[idx];
        const n   = this.data.valid_next[idx];

        if (p >= 0) {
          this.data.valid_next[p] = n;
        } else {
          this.data.first_valid_index = n;
        }

        if (n >= 0) {
          this.data.valid_prev[n] = p;
        } else {
          this.data.last_valid_index = p;
        }

        // clear pointers for safety
        this.data.valid_prev[idx] = -1;
        this.data.valid_next[idx] = -1;
    }`;
    }
    return `
    static remove_entity(entity) {
        ${this.get_default_remove_entity(fields)}

        // unlink from the live‐list in O(1)
        const idx = Number(EntityID.get_absolute_index(entity));
        const p   = this.data.valid_prev[idx];
        const n   = this.data.valid_next[idx];

        if (p >= 0) {
          this.data.valid_next[p] = n;
        } else {
          this.data.first_valid_index = n;
        }

        if (n >= 0) {
          this.data.valid_prev[n] = p;
        } else {
          this.data.last_valid_index = p;
        }

        // clear pointers for safety
        this.data.valid_prev[idx] = -1;
        this.data.valid_next[idx] = -1;
    }`;
  }

  static get_default_remove_entity(fields) {
    const entity_field_resets = Object.entries(fields)
      .filter(([_, field]) => !field.no_fragment_array && field.no_instance_count_resize)
      .map(([key, field]) => {
        if (field.is_container) {
          return `this.data.${key}.remove(entity);`;
        }
        if (field.vector) {
          return `${Object.keys(field.vector)
            .map(
              (axis) =>
                `this.data.${key}.${axis}[entity] = ${field.stride > 1 
                  ? `Array(${field.stride}).fill(${field.default ? field.default[axis] : 0})` 
                  : `this.data.${key} instanceof BigInt64Array ? ${field.default ? field.default[axis] : 0}n : ${field.default ? field.default[axis] : 0}`};`
            )
            .join("\n")}
        `;
        }
        if (key === "dirty") {
          return "";
        }
        return `this.data.${key}[entity] = ${field.stride > 1 
          ? `Array(${field.stride}).fill(${field.default || 0})` 
          : `this.data.${key} instanceof BigInt64Array ? ${field.default || 0}n : ${field.default || 0}`};`;
      });
    const instance_field_resets = Object.entries(fields)
      .filter(([_, field]) => !field.no_fragment_array && !field.no_instance_count_resize)
      .map(([key, field]) => {
        if (field.is_container) {
          return `this.data.${key}.remove(entity_index);`;
        }
        if (field.vector) {
          return `${Object.keys(field.vector)
            .map(
              (axis) =>
                `this.data.${key}.${axis}[entity_index] = ${field.stride > 1 
                  ? `Array(${field.stride}).fill(${field.default ? field.default[axis] : 0})` 
                  : `this.data.${key} instanceof BigInt64Array ? ${field.default ? field.default[axis] : 0}n : ${field.default ? field.default[axis] : 0}`};`
            )
            .join("\n")}
        `;
        }
        if (key === "dirty") {
          return "";
        }
        return `this.data.${key}[entity_index] = ${field.stride > 1 
          ? `Array(${field.stride}).fill(${field.default || 0})` 
          : `this.data.${key} instanceof BigInt64Array ? ${field.default || 0}n : ${field.default || 0}`};`;
      });

    return `
        const instance_count = EntityID.get_instance_count(entity);
        const entity_offset = EntityID.get_absolute_index(entity);

        ${entity_field_resets.join("\n")}
        for (let i = 0; i < instance_count; ++i) {
            const entity_index = entity_offset + i;
            ${instance_field_resets.join("\n")}
        }

        this.data.gpu_data_dirty = true;
    `;
  }

  static generate_get_entity_data(fields, override) {
    if (override) {
      return `
    static get_entity_data(entity, instance = 0) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_get_entity_data(fields) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static get_entity_data(entity, instance = 0) {
        ${this.get_default_get_entity_data(fields)}
    }`;
  }

  static get_default_get_entity_data(fields) {
    return `
        const data_view = this.data_view_allocator.allocate();
        data_view.view_entity(entity, instance);
        return data_view;
    `;
  }

  static generate_duplicate_entity_data(fields, override) {
    if (override) {
      return `
    static duplicate_entity_data(entity, instance = 0) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.get_default_duplicate_entity_data(fields) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static duplicate_entity_data(entity, instance = 0) {
        ${this.get_default_duplicate_entity_data(fields)}
    }`;
  }

  static get_default_duplicate_entity_data(fields) {
    return `
        const data = {};
        const entity_index = EntityID.get_absolute_index(entity);
        ${Object.entries(fields)
          .filter(([_, field]) => !field.no_fragment_array)
          .map(([key, field]) => {
            if (field.is_container) {
              return `data.${key} = this.data.${key}.get_data_for_entity(${field.no_instance_count_resize ? `entity` : `entity_index`});`;
            }
            if (field.vector) {
              return `data.${key} = {
                        ${Object.keys(field.vector)
                          .map(
                            (axis) =>
                              `${axis}: this.data.${key}.${axis}[${field.no_instance_count_resize ? `entity` : `entity_index`}]`
                          )
                          .join(",\n        ")}
                    };`;
            }
            if (field.stride > 1) {
              return `data.${key} = Array(${field.stride}).fill(${field.default || 0});
                    for (let i = 0; i < ${field.stride}; i++) {
                        data.${key}[i] = this.data.${key}[${field.no_instance_count_resize ? `entity` : `entity_index`} * ${field.stride} + i];
                    }`;
            }
            return `data.${key} = this.data.${key}[${field.no_instance_count_resize ? `entity` : `entity_index`}];`;
          })
          .join("\n        ")}
        return data;
    `;
  }

  static generate_rebuild_buffers(buffers, override) {
    if (Object.keys(buffers).length === 0) return "";

    if (override) {
      return `
    static rebuild_buffers() {
        if (!this.data.gpu_data_dirty) return;

        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.generate_default_rebuild_buffers(buffers) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static rebuild_buffers() {
        if (!this.data.gpu_data_dirty) return;

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
                : `const gpu_data = this.data.${key} ? this.data.${key} : new ${buffer.type.array.name}(this.size * ${buffer.stride || 1} + ${buffer.stride || 1});`
            }
            if (!this.data.${key}_buffer || this.data.${key}_buffer.config.size < gpu_data.byteLength) {
                this.data.${key}_buffer = Buffer.create({
                name: ${key}_buffer_name,
                usage: ${buffer.usage},
                raw_data: gpu_data,
                force: true
            });
            ${
              buffer.cpu_buffer
                ? `
                for (let i = 0; i < MAX_BUFFERED_FRAMES; i++) {
                    this.data.${key}_cpu_buffer[i] = Buffer.create({
                        name: \`${key}_cpu_buffer_\${i}\`,
                        usage: ${BufferType.CPU_READ},
                        raw_data: gpu_data,
                        force: true
                    });
                }`
                : ""
            }
            Renderer.get().mark_bind_groups_dirty(true);
            ${buffer.no_dispatch ? "" : `global_dispatcher.dispatch(${key}_event, this.data.${key}_buffer);`}
          ${buffer.only_rebuild_on_resize ? `}` : `
          } else {
            this.data.${key}_buffer.write(gpu_data);
          }`}

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
    static async sync_buffers() {
        const buffered_frame = Renderer.get().get_buffered_frame_number();
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.generate_default_sync_buffers(buffers) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static async sync_buffers() {
        const buffered_frame = Renderer.get().get_buffered_frame_number();
        ${this.generate_default_sync_buffers(buffers)}
    }`;
  }

  static generate_default_sync_buffers(buffers) {
    return `
        ${Object.entries(buffers)
          .filter(([_, buf]) => buf.cpu_buffer)
          .map(
            ([key, _]) => `
        if (this.data.${key}_cpu_buffer[buffered_frame]?.buffer.mapState === unmapped_state) {
            await this.data.${key}_cpu_buffer[buffered_frame].read(
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
    static to_gpu_data() {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.generate_default_to_gpu_data(buffers) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static to_gpu_data() {
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

      this.rebuild_buffers();

      return {
            ${Object.keys(buffers)
              .map((key) => `${key}_buffer: this.data.${key}_buffer`)
              .join(",\n            ")}
        };
    `;
  }

  static generate_copy_entity_instance(fields, override) {
    if (override) {
      return `
    static copy_entity_instance(to_index, from_index) {
        ${override.pre ? override.pre : ""}
        ${!override.skip_default ? this.generate_default_copy_entity_instance(fields) : ""}
        ${override.post ? override.post : ""}
    }`;
    }
    return `
    static copy_entity_instance(to_index, from_index) {
        ${this.generate_default_copy_entity_instance(fields)}
    }`;
  }

  static generate_default_copy_entity_instance(fields) {
    const filtered_fields = Object.entries(fields).filter(
      ([_, field]) => !field.no_fragment_array && !field.no_instance_count_resize
    );
    const has_fields = filtered_fields.length > 0;
    return `
      ${
        has_fields
          ? `
        ${filtered_fields
          .map(([key, field]) => {
            if (field.vector) {
              return Object.keys(field.vector)
                .map(
                  (axis) => `
                          ${Array.from({ length: field.stride }, (_, s) => {
                            return `this.data.${key}.${axis}[to_index * ${field.stride} + ${s}] = this.data.${key}.${axis}[from_index * ${field.stride} + ${s}];`;
                          }).join("\n")}
                          `
                )
                .join("\n");
            }
            return `
                      ${Array.from({ length: field.stride }, (_, s) => {
                        return `this.data.${key}[to_index * ${field.stride} + ${s}] = this.data.${key}[from_index * ${field.stride} + ${s}];`;
                      }).join("\n")}
                    `;
          })
          .join("\n")}
                    `
          : ``
      }

    this.data.gpu_data_dirty = true;
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

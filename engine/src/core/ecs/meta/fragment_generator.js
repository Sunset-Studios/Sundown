import { BufferType } from "./fragment_generator_types.js";

export class FragmentGenerator {
  static generate(config, relative_config_path = "") {
    const {
      name,
      imports = {},
      constants = {},
      members = {},
      fields = {},
      gpu_buffers = null,
      custom_methods = {},
      hooks = {},
    } = config;

    const fragment_name = name + "Fragment";
    const frag_name_lower_snake = name
      .replace(/([A-Z])/g, "_$1")
      .toLowerCase()
      .substring(1);

    const all_imports = {
      Fragment: "../fragment.js",
      SolarFragmentView: "../solar/view.js",
      RingBufferAllocator: "../../../memory/allocator.js",
      Name: "../../../utility/names.js",
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

    const members_definition = `
      ${Object.entries(members)
        .map(([key, member]) => {
          return `static ${key} = ${member};`;
        })
        .join("\n        ")}
    `;

    const fields_definition = `{
        ${Object.entries(fields)
          .map(([key, field]) => {
            const ctor = field.type?.array?.name;
            if (!ctor) {
              console.error(
                `Error generating fragment '${name}': Field '${key}' has missing or invalid type/array definition.`
              );
              throw new Error(`Invalid type definition for field '${key}' in fragment '${name}'.`);
            }
            const elements = field.vector ? Object.keys(field.vector).length : field.stride || 1;
            const dv = field.default !== undefined ? field.default : field.type?.default;
            const def_lit = typeof dv === "bigint" ? `${dv}n` : JSON.stringify(dv);
            const gpu_buffer_flag = !!field.gpu;
            const is_container_flag = !!field.is_container;
            const usage = field.usage || BufferType.STORAGE_SRC;
            const cpu_readback_flag = !!field.cpu_readback;

            const getter = field.getter
              ? `, getter(typed_array, element_offset) { ${field.getter} }`
              : "";
            const setter = field.setter
              ? `, setter(value, typed_array, element_offset) { ${field.setter} }`
              : "";

            return `${key}: {
              ctor: ${ctor},
              elements: ${elements},
              default: ${def_lit},
              gpu_buffer: ${gpu_buffer_flag},
              buffer_name: "${key}",
              is_container: ${is_container_flag},
              usage: ${usage}${getter}${setter},
              cpu_readback: ${cpu_readback_flag}
            }`;
          })
          .join(",\n        ")}
    }`;

    let gpu_buffers_definition = "";
    if (gpu_buffers && typeof gpu_buffers === "object" && Object.keys(gpu_buffers).length > 0) {
      const buffer_entries = Object.entries(gpu_buffers)
        .map(([buffer_key, buffer_config]) => {
          if (!buffer_config) {
            console.warn(
              `FragmentGenerator: Invalid config for gpu_buffer '${buffer_key}' in fragment '${name}'. Skipping.`
            );
            return null;
          }

          // offline compute buffer_stride
          let buffer_stride = 0;
          let gpu_data_snippet = "";
          let fields_array_snippet = "";

          if (Array.isArray(buffer_config.fields)) {
            const valid_fields = buffer_config.fields.filter((f) => typeof f === "string");
            if (valid_fields.length !== buffer_config.fields.length) {
              console.warn(
                `FragmentGenerator: Non-string field name found in gpu_buffer '${buffer_key}' for fragment '${name}'. Filtering.`
              );
            }
            if (valid_fields.length === 0) {
              console.warn(
                `FragmentGenerator: No valid fields left for gpu_buffer '${buffer_key}' in fragment '${name}'. Skipping.`
              );
              return null;
            }

            for (let i = 0; i < valid_fields.length; i++) {
              const field_name = valid_fields[i];
              const raw_spec = fields[field_name];

              const element_count = raw_spec.vector
                ? Object.keys(raw_spec.vector).length
                : raw_spec.stride || 1;
              const bytes_per_element = raw_spec.type.array.BYTES_PER_ELEMENT;

              buffer_stride += element_count * bytes_per_element;
            }

            fields_array_snippet = `, fields: ${JSON.stringify(valid_fields)}`;
          } else if (buffer_config.gpu_data) {
            // Capture gpu_data if user provided a function/string
            gpu_data_snippet = `, ${buffer_config.gpu_data.toString()}`;
          }

          if (!fields_array_snippet && !gpu_data_snippet) {
            console.warn(
              `FragmentGenerator: No valid fields or gpu_data found for gpu_buffer '${buffer_key}' in fragment '${name}'. Skipping.`
            );
            return null;
          }

          if (buffer_stride === 0) {
            buffer_stride = buffer_config.stride || 1;
          }

          const escaped_buffer_key = JSON.stringify(buffer_key);
          return `${escaped_buffer_key}: {
            usage: ${buffer_config.usage},
            stride: ${buffer_stride},
            buffer_name: "${buffer_key}",
            cpu_readback: ${!!buffer_config.cpu_readback}
            ${fields_array_snippet}
            ${gpu_data_snippet}
          }`;
        })
        .filter((entry) => entry !== null)
        .join(",\n            ");

      if (buffer_entries.length > 0) {
        gpu_buffers_definition = `
    static gpu_buffers = {
            ${buffer_entries}
    };`;
      }
    }

    return `
${import_statements}

/**
 * The ${name} fragment class.
 * Use \`EntityManager.get_fragment(entity, ${name})\` to get a fragment instance for an entity.
 */
export class ${fragment_name} extends Fragment {
  static id = Name.from("${frag_name_lower_snake}");
  static field_key_map = new Map();
  static fields = ${fields_definition};
  static buffer_data = new Map(); // key → { buffer: FragmentGpuBuffer, stride: number }

  ${members_definition}
  ${gpu_buffers_definition}

  static get view_allocator() {
    if (!this._view_allocator) {
      this._view_allocator = new RingBufferAllocator(256, new SolarFragmentView(this));
    }
    return this._view_allocator;
  }

  static is_valid() {
    return this.id &&
      this.fields &&
      this.view_allocator
  }

  static get_buffer_name(field_name) {
    return this.field_key_map.get(field_name);
  }

  ${Object.entries(constants)
    .map(([key, value]) => `static ${key} = ${JSON.stringify(value)};`)
    .join("\n")}
  ${this.generate_custom_methods(custom_methods)}
  ${this.generate_hooks(hooks)}
}
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

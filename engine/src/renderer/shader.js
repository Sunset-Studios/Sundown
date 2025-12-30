import { Renderer } from "./renderer.js";
import { ResourceCache } from "./resource_cache.js";
import { WgslReflect, ResourceType } from "wgsl_reflect/wgsl_reflect.node.js";
import { read_file } from "../utility/file_system.js";
import { ShaderResourceType } from "./renderer_types.js";
import { CacheTypes } from "./renderer_types.js";
import {
  rgba16float_format,
  rgba32uint_format,
  rgba32sint_format,
  bgra8unorm_format,
  rgba8unorm_format,
  rgba8snorm_format,
  rgba8uint_format,
  rgba8sint_format,
  rg32float_format,
  r32uint_format,
  r32sint_format,
  r16uint_format,
  r16sint_format,
} from "../utility/config_permutations.js";
import { error } from "../utility/logging.js";
import { hash_data_object } from "../utility/hashing.js";

const include_string = "#include";
const precision_float_string = "precision_float";
const has_precision_float_string = "HAS_PRECISION_FLOAT";
const has_subgroups_string = "HAS_SUBGROUPS";
const use_radiance_cache_as_deferred_lighting_string = "USE_RADIANCE_CACHE_AS_DEFERRED_LIGHTING";

const f16_type_string = "f16";
const f32_type_string = "f32";
const vec4_f32_type_string = "vec4<f32>";
const vec4_f16_type_string = "vec4<f16>";
const vec4_type_string = "vec4f";
const vec4_u32_type_string = "vec4<u32>";
const vec4_u_type_string = "vec4u";
const vec4_i32_type_string = "vec4<i32>";
const vec4_i_type_string = "vec4i";
const vec2_f32_type_string = "vec2<f32>";
const vec2_f_type_string = "vec2f";
const u32_type_string = "u32";
const u_type_string = "u";
const i32_type_string = "i32";
const i_type_string = "i";
const vec2_f16_type_string = "vec2f16";
const u16_type_string = "u16";
const i16_type_string = "i16";
const vec4_unorm_type_string = "vec4unorm";
const vec4_snorm_type_string = "vec4snorm";
const vec4_u8_type_string = "vec4u8";
const vec4_i8_type_string = "vec4i8";

const include_regex = /^#include\s+"(\S+)".*$/m;
const defines_regex = /#define\s+(\S+)(?:\s+(\S*))?$/gm;
const precision_float_regex = /precision_float/g;

export class Shader {
  static shader_paths = ["engine/shaders"];

  module = null;
  code = null;
  file_path = "";
  defines = {};
  reflection = null;

  static register_shader_path(path) {
    Shader.shader_paths.push(path);
  }

  initialize(file_path, defines = {}) {
    const renderer = Renderer.get();

    let asset = this._load_shader_text(file_path, defines);
    if (!asset) {
      return;
    }

    try {
      this.code = asset;
      this.module = renderer.device.createShaderModule({
        label: file_path,
        code: asset,
      });
      this.file_path = file_path;
      this.reflection = this.reflect();
    } catch (err) {
      console.log(this.code);
      error(`WebGPU shader error: could not create shader module at ${file_path}`, err);
    }
  }

  reflect() {
    const reflect = new WgslReflect(this.code);
    return reflect;
  }

  _load_shader_text(file_path, defines = {}, load_recursion_step = 0) {
    let asset = null;
    for (const path of Shader.shader_paths) {
      const full_path = `${path}/${file_path}`;
      asset = read_file(full_path);
      if (asset) {
        break;
      }
    }

    if (!asset) {
      error(`WebGPU shader error: could not find shader at ${file_path}`);
      return null;
    }

    // Step 0: parse shader includes
    asset = this._parse_shader_includes(asset, defines, load_recursion_step);
    
    // Step 1: parse defines and conditionals
    if (load_recursion_step === 0) {
      const { defines_map, stripped_code } = this._build_defines_map_and_strip(
        file_path,
        asset,
        defines
      );
      this.defines = defines_map;
      asset = this._parse_conditional_defines_and_types(stripped_code);

      if (defines_map.DEPTH_ONLY) {
        asset = this._strip_custom_fragment_functions(asset);
      }
    }
    
    // Step 2: parse and expand WGSL macros
    if (load_recursion_step === 0) {
      const { code: code_without_macros, macros } = this._parse_and_strip_macros(asset);
      asset = this._expand_macros(code_without_macros, macros);
    }

    return asset;
  }

  // ----------------------------------------------
  // WGSL Macro System
  // ----------------------------------------------
  _parse_and_strip_macros(code) {
    // Collect parameterized #define macros and #macro/#endmacro blocks
    const macros = new Map(); // name -> { args: string[], body: string }

    // 1) Parameterized single-line defines: #define NAME(arg1, arg2, ...) expansion
    const lines = code.split(/\r\n|\r|\n/);
    const out_lines = [];
    const param_define_regex = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(.*)$/;

    // 2) Multi-line macros: #macro NAME(args?) ... #endmacro
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trimStart();
      if (trimmed.startsWith("#macro")) {
        // Parse header
        const header = trimmed;
        const match = header.match(/^#macro\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*$/);
        if (match) {
          const name = match[1];
          const args = (match[2] || "")
            .split(",")
            .map((s) => s.trim())
            .filter((s) => s.length > 0);
          // Collect until #endmacro
          const body_lines = [];
          i++;
          let found_end = false;
          while (i < lines.length) {
            const l = lines[i];
            if (l.trimStart().startsWith("#endmacro")) {
              found_end = true;
              break;
            }
            body_lines.push(l);
            i++;
          }
          if (!found_end) {
            // Malformed macro; keep original text
            out_lines.push(line);
          } else {
            const body = body_lines.join("\n");
            macros.set(name, { args, body });
          }
          // Skip the #endmacro line
          i++;
          continue;
        }
      }

      // Check param define
      const m = line.match(param_define_regex);
      if (m) {
        const name = m[1];
        const args = m[2]
          .split(",")
          .map((s) => s.trim())
          .filter((s) => s.length > 0);
        const body = m[3] || "";
        macros.set(name, { args, body });
        i++;
        continue; // strip this line
      }

      out_lines.push(line);
      i++;
    }

    return { code: out_lines.join("\n"), macros };
  }

  _expand_macros(code, macros, max_depth = 16) {
    if (!macros || macros.size === 0) return code;
    const macro_names = Array.from(macros.keys());
    const name_set = new Set(macro_names);
    let cur = code;
    for (let pass = 0; pass < max_depth; pass++) {
      const { out, changed } = this._expand_once(cur, macros, name_set);
      cur = out;
      if (!changed) break;
    }
    return cur;
  }

  _split_args(s) {
    const args = [];
    let depth = 0;
    let start = 0;
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (ch === "(" || ch === "[" || ch === "{") depth++;
      else if (ch === ")" || ch === "]" || ch === "}") depth--;
      else if (ch === "," && depth === 0) {
        args.push(s.slice(start, i).trim());
        start = i + 1;
      }
    }
    const last = s.slice(start).trim();
    if (last.length) args.push(last);
    return args;
  }

  _expand_once(src, macros, name_set) {
    let i = 0;
    let out = "";
    let changed = false;

    let in_line_comment = false;
    let in_block_comment = false;
    let in_string = false;
    let directive_line = false;

    const is_ident_start = (c) => /[A-Za-z_]/.test(c);
    const is_ident_char = (c) => /[A-Za-z0-9_]/.test(c);
    const flush_char = (c) => {
      out += c;
    };

    while (i < src.length) {
      const c = src[i];

      // Newline resets line-based states
      if (c === "\n") {
        in_line_comment = false;
        directive_line = false;
        flush_char(c);
        i++;
        continue;
      }

      if (in_line_comment) {
        flush_char(c);
        i++;
        continue;
      }
      if (in_block_comment) {
        if (c === "*" && src[i + 1] === "/") {
          flush_char("*/");
          i += 2;
          in_block_comment = false;
          continue;
        }
        flush_char(c);
        i++;
        continue;
      }
      if (in_string) {
        if (c === "\\") {
          // escape
          flush_char(src.substr(i, 2));
          i += 2;
          continue;
        }
        flush_char(c);
        if (c === '"') {
          in_string = false;
        }
        i++;
        continue;
      }

      // Check comment/string/directive starts
      if (c === "/" && src[i + 1] === "/") {
        flush_char("//");
        i += 2;
        in_line_comment = true;
        continue;
      }
      if (c === "/" && src[i + 1] === "*") {
        flush_char("/*");
        i += 2;
        in_block_comment = true;
        continue;
      }
      if (c === '"') {
        in_string = true;
        flush_char(c);
        i++;
        continue;
      }
      if ((c === "#" && (i === 0 || src[i - 1] === "\n")) || (directive_line && c !== "\n")) {
        // treat whole directive token as literal until EOL
        directive_line = true;
        flush_char(c);
        i++;
        continue;
      }

      // Try macro match
      if (is_ident_start(c)) {
        let j = i + 1;
        while (j < src.length && is_ident_char(src[j])) j++;
        const ident = src.slice(i, j);
        if (name_set.has(ident)) {
          // Skip whitespace to see if followed by '('
          let k = j;
          while (k < src.length && /\s/.test(src[k])) k++;
          if (src[k] === "(") {
            // Parse balanced parentheses
            let p = k + 1;
            let depth = 1;
            while (p < src.length && depth > 0) {
              if (src[p] === "(") depth++;
              else if (src[p] === ")") depth--;
              else if (src[p] === '"') {
                // Skip strings in arguments
                p++;
                while (p < src.length) {
                  if (src[p] === '"' && src[p - 1] !== "\\") {
                    p++;
                    break;
                  }
                  p++;
                }
                continue;
              }
              p++;
            }
            const call_end = p; // position after ')'
            const raw_args = src.slice(k + 1, call_end - 1);
            const actuals = this._split_args(raw_args);
            const { args: formals, body } = macros.get(ident);
            let replacement = body;
            for (let ai = 0; ai < formals.length; ai++) {
              const formal = formals[ai];
              const actual = actuals[ai] !== undefined ? actuals[ai] : "";
              const wrapped = `(${actual})`;
              const re = new RegExp(`\\b${formal}\\b`, "g");
              replacement = replacement.replace(re, wrapped);
            }
            out += replacement;
            i = call_end;
            changed = true;
            continue;
          }
        }
        // Not a macro call
        out += ident;
        i = j;
        continue;
      }

      // Default copy
      flush_char(c);
      i++;
    }
    return { out, changed };
  }

  // ----------------------------------------------
  // WGSL Include System
  // ----------------------------------------------
  _parse_shader_includes(code, defines = {}, load_recursion_step = 0) {
    let include_positions = [];

    let pos = code.indexOf(include_string, 0);
    while (pos !== -1) {
      include_positions.push(pos);
      pos = code.indexOf(include_string, pos + 1);
    }

    for (let i = include_positions.length - 1; i >= 0; --i) {
      const start = include_positions[i];
      const end = code.indexOf("\n", start);
      const include_line = code.substring(start, end);
      const match = include_line.match(include_regex);
      if (match) {
        const include_contents = this._load_shader_text(match[1], defines, load_recursion_step + 1);
        code = code.slice(0, start) + include_contents + code.slice(end);
      }
    }

    return code;
  }

  // ----------------------------------------------
  // WGSL Define System
  // ----------------------------------------------
  _build_defines_map_and_strip(file_path, code, defines) {
    const defines_map = Object.assign({}, defines);
    const stripped_code = code.replace(defines_regex, (match, key, value) => {
      defines_map[key] = value || true;
      return "";
    });
    defines_map[precision_float_string] = Renderer.get().has_f16
      ? f16_type_string
      : f32_type_string;
    defines_map[has_precision_float_string] = Renderer.get().has_f16;
    defines_map[has_subgroups_string] = Renderer.get().has_subgroups;
    defines_map[use_radiance_cache_as_deferred_lighting_string] = Renderer.get().use_radiance_cache_as_deferred_lighting;
    return { defines_map, stripped_code };
  }

  _parse_conditional_defines_and_types(code) {
    // cross-platform split: handles Windows (\r\n), old Mac (\r) and Unix (\n)
    const lines = code.split(/\r\n|\r|\n/);
    const output_lines = [];
    const stack = [];
    const if_regex = /#if\s+(\S+)(?:\s+(\S+))?/;
    const ifndef_regex = /#ifndef\s+(\S+)(?:\s+(\S+))?/;
    const else_regex = /#else/;
    const endif_regex = /#endif/;

    for (let i = 0; i < lines.length; ++i) {
      const line = lines[i];

      let match;
      if ((match = if_regex.exec(line))) {
        const [, condition, value] = match;
        const should_include = this.defines[condition] === (value || true);
        stack.push({ include: should_include });
      } else if ((match = ifndef_regex.exec(line))) {
        const [, condition] = match;
        const should_include = !this.defines[condition];
        stack.push({ include: should_include });
      } else if (else_regex.test(line)) {
        const frame = stack[stack.length - 1];
        frame.include = !frame.include;
      } else if (endif_regex.test(line)) {
        stack.pop();
      } else {
        if (stack.every((frame) => frame.include)) {
          output_lines.push(line);
        }
      }
    }

    let result = output_lines.join("\n");
    result = result.replace(precision_float_regex, this.defines[precision_float_string]);
    return result.trim();
  }

  // ----------------------------------------------
  // WGSL Fragment System
  // ----------------------------------------------
  _strip_custom_fragment_functions(code) {
    const lines = code.split(/\r\n|\r|\n/);
    const out = [];
    let skipping = false;
    let brace_depth = 0;

    for (let i = 0; i < lines.length; ++i) {
      const ln = lines[i];
      if (!skipping && /^\s*fn\s+fragment\s*\(/.test(ln)) {
        skipping = true;
        brace_depth = (ln.match(/\{/g) || []).length - (ln.match(/\}/g) || []).length;
        continue;
      }

      if (skipping) {
        brace_depth += (ln.match(/\{/g) || []).length;
        brace_depth -= (ln.match(/\}/g) || []).length;
        if (brace_depth <= 0) {
          skipping = false;
        }
        continue;
      }

      out.push(ln);
    }

    return out.join("\n");
  }

  static create(file_path, defines = null, force_recreate = false) {
    let key = defines ? hash_data_object(defines, file_path) : file_path;
    let shader = ResourceCache.get().fetch(CacheTypes.SHADER, key);

    if (shader && force_recreate) {
      shader = null;
    }

    if (!shader) {
      shader = new Shader();
      shader.initialize(file_path, defines);
      ResourceCache.get().store(CacheTypes.SHADER, key, shader);
    }

    return shader;
  }

  static resource_type_from_reflection_type(type) {
    switch (type) {
      case ResourceType.Texture:
        return ShaderResourceType.Texture;
      case ResourceType.Sampler:
        return ShaderResourceType.Sampler;
      case ResourceType.Storage:
        return ShaderResourceType.Storage;
      case ResourceType.Uniform:
        return ShaderResourceType.Uniform;
      case ResourceType.StorageTexture:
        return ShaderResourceType.StorageTexture;
      default:
        throw new Error(`Unknown binding type: ${type}`);
    }
  }

  static get_optimal_texture_format(wgsl_format) {
    switch (wgsl_format) {
      case f32_type_string:
        return bgra8unorm_format;
      case vec4_f32_type_string:
      case vec4_type_string:
        return bgra8unorm_format;
      case vec4_u32_type_string:
      case vec4_u_type_string:
        return rgba32uint_format;
      case vec4_i32_type_string:
      case vec4_i_type_string:
        return rgba32sint_format;
      case vec2_f32_type_string:
      case vec2_f_type_string:
        return rg32float_format;
      case u32_type_string:
      case u_type_string:
        return r32uint_format;
      case i32_type_string:
      case i_type_string:
        return r32sint_format;
      case f16_type_string:
        return rgba16float_format;
      case vec4_f16_type_string:
        return rgba16float_format;
      case vec2_f16_type_string:
        return rg16float_format;
      case u16_type_string:
        return r16uint_format;
      case i16_type_string:
        return r16sint_format;
      case vec4_unorm_type_string:
        return rgba8unorm_format;
      case vec4_snorm_type_string:
        return rgba8snorm_format;
      case vec4_u8_type_string:
        return rgba8uint_format;
      case vec4_i8_type_string:
        return rgba8sint_format;
      default:
        return rgba8unorm_format;
    }
  }
}

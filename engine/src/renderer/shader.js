import { Renderer } from "./renderer.js";
import { EntityManager } from "../core/ecs/entity.js";
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
import { log, warn, error } from "../utility/logging.js";


const include_string = "#include";
const if_string = "#if";
const else_string = "#else";
const endif_string = "#endif";
const precision_float_string = "precision_float";
const has_precision_float_string = "HAS_PRECISION_FLOAT";
const entity_compaction_string = "ENTITY_COMPACTION";
const read_only_flags_string = "READ_ONLY_FLAGS";

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
const conditional_defines_regex = /#(if|ifndef)\s+(\S+)(?:\s+(\S+))?$/gm;
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

  initialize(file_path) {
    const renderer = Renderer.get();

    let asset = this._load_shader_text(file_path);
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
    } catch (error) {
      error(`WebGPU shader error: could not create shader module at ${file_path}`, error);
    }
  }

  reflect() {
    const reflect = new WgslReflect(this.code);
    return reflect;
  }

  _load_shader_text(file_path, load_recursion_step = 0) {
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

    asset = this._parse_shader_includes(asset, load_recursion_step);

    if (load_recursion_step === 0) {
      const { defines_map, stripped_code } = this._build_defines_map_and_strip(asset);
      asset = stripped_code;
      this.defines = defines_map;
      asset = this._parse_conditional_defines_and_types(asset);
    }

    return asset;
  }

  _parse_shader_includes(code, load_recursion_step = 0) {
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
        const include_contents = this._load_shader_text(match[1], load_recursion_step + 1);
        code = code.slice(0, start) + include_contents + code.slice(end);
      }
    }

    return code;
  }

  _build_defines_map_and_strip(code) {
    const defines_map = {};

    const stripped_code = code.replace(defines_regex, (match, key, value) => {
      defines_map[key] = value || true;
      return "";
    });
    defines_map[precision_float_string] = Renderer.get().has_f16
      ? f16_type_string
      : f32_type_string;
    defines_map[has_precision_float_string] = Renderer.get().has_f16;
    defines_map[entity_compaction_string] = EntityManager.is_entity_compaction_enabled();
    defines_map[read_only_flags_string] = EntityManager.is_entity_compaction_enabled();
    return { defines_map, stripped_code };
  }

  _parse_conditional_defines_and_types(code) {
    let result = "";
    let last_index = 0;
    let match;

    while ((match = conditional_defines_regex.exec(code)) !== null) {
      const [full_match, directive, condition, value] = match;
      const start_index = match.index;
      const end_index = code.indexOf(endif_string, start_index);

      if (end_index === -1) {
        warn(`Unmatched #${directive} at position ${start_index}`);
        continue;
      }

      result += code.slice(last_index, start_index);

      const should_include =
        directive === "if" ? this.defines[condition] === (value || true) : !this.defines[condition];

      const else_index = code.indexOf(else_string, start_index);

      if (else_index !== -1 && else_index < end_index) {
        if (should_include) {
          result += code.slice(start_index + full_match.length, else_index).trim();
        } else {
          result += code.slice(else_index + else_string.length, end_index).trim();
        }
      } else {
        if (should_include) {
          const block_content = code.slice(start_index + full_match.length, end_index).trim();
          result += block_content;
        }
      }

      last_index = end_index + endif_string.length;
    }

    result += code.slice(last_index);

    result = result.replace(precision_float_regex, this.defines[precision_float_string]);

    return result.trim();
  }

  static create(file_path) {
    let shader = ResourceCache.get().fetch(CacheTypes.SHADER, file_path);
    if (!shader) {
      shader = new Shader();
      shader.initialize(file_path);
      ResourceCache.get().store(CacheTypes.SHADER, file_path, shader);
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

import { ResourceCache, CacheTypes } from "./resource_cache.js";
import { WgslReflect, ResourceType } from "wgsl_reflect/wgsl_reflect.node.js";

export const ShaderResourceType = {
  Uniform: 0,
  Storage: 1,
  Texture: 2,
  Sampler: 3,
  StorageTexture: 4,
};

export const ShaderResource = {
  type: ShaderResourceType.Uniform,
  name: "",
  binding: 0,
};

export class Shader {
  static shader_paths = ["engine/shaders"];

  module = null;
  code = null;
  file_path = "";
  defines = {};

  static register_shader_path(path) {
    Shader.shader_paths.push(path);
  }

  initialize(context, file_path) {
    let asset = this._load_shader_text(file_path);
    if (!asset) {
      return;
    }

    try {
      this.code = asset;
      this.module = context.device.createShaderModule({
        label: file_path,
        code: asset,
      });
      this.file_path = file_path;
    } catch (error) {
      console.error(
        `WebGPU shader error: could not create shader module at ${file_path}`,
        error
      );
    }
  }

  reflect() {
    const reflect = new WgslReflect(this.code);
    return reflect;
  }

  _load_shader_text(file_path, load_recursion_step = 0) {
    let asset = null;
    for (const path of Shader.shader_paths) {
      try {
        const url = new URL(`${path}/${file_path}`, window.location.href);
        const response = new XMLHttpRequest();
        response.open("GET", url.href, false);
        response.send(null);
        if (response.status === 200) {
          asset = response.responseText;
          break;
        }
      } catch (error) {
        console.warn(`Failed to load shader from ${path}/${file_path}:`, error);
      }
    }

    if (!asset) {
      console.error(
        `WebGPU shader error: could not find shader at ${file_path}`
      );
      return null;
    }

    asset = this._parse_shader_includes(asset, load_recursion_step);

    if (load_recursion_step === 0) {
      const { defines_map, stripped_code } =
        this._build_defines_map_and_strip(asset);
      asset = stripped_code;
      this.defines = defines_map;
      asset = this._parse_conditional_defines(asset);
    }

    return asset;
  }

  _parse_shader_includes(code, load_recursion_step = 0) {
    let include_positions = [];

    let pos = code.indexOf("#include", 0);
    while (pos !== -1) {
      include_positions.push(pos);
      pos = code.indexOf("#include", pos + 1);
    }

    const include_regex = /^#include\s+"(\S+)".*$/m;

    for (let i = include_positions.length - 1; i >= 0; --i) {
      const start = include_positions[i];
      const end = code.indexOf("\n", start);
      const include_line = code.substring(start, end);
      const match = include_line.match(include_regex);
      if (match) {
        const include_contents = this._load_shader_text(
          match[1],
          load_recursion_step + 1
        );
        code = code.slice(0, start) + include_contents + code.slice(end);
      }
    }

    return code;
  }

  _build_defines_map_and_strip(code) {
    const defines_map = {};
    const defines_regex = /#define\s+(\S+)(?:\s+(\S*))?$/gm;
    const stripped_code = code.replace(defines_regex, (match, key, value) => {
      defines_map[key] = value || true;
      return "";
    });
    return { defines_map, stripped_code };
  }

  _parse_conditional_defines(code) {
    let result = "";
    let last_index = 0;
    let match;

    const regex = /#(if|ifndef)\s+(\S+)(?:\s+(\S+))?$/gm;
    while ((match = regex.exec(code)) !== null) {
      const [full_match, directive, condition, value] = match;
      const start_index = match.index;
      const end_index = code.indexOf("#endif", start_index);

      if (end_index === -1) {
        console.warn(`Unmatched #${directive} at position ${start_index}`);
        continue;
      }

      result += code.slice(last_index, start_index);

      const should_include =
        directive === "if"
          ? this.defines[condition] === (value || true)
          : !this.defines[condition];

      if (should_include) {
        const block_content = code
          .slice(start_index, end_index)
          .replace(full_match, "")
          .trim();
        result += block_content;
      }

      last_index = end_index + "#endif".length;
    }

    result += code.slice(last_index);

    return result.trim();
  }

  static create(context, file_path) {
    let shader = ResourceCache.get().fetch(CacheTypes.SHADER, file_path);
    if (!shader) {
      shader = new Shader();
      shader.initialize(context, file_path);
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
      case "f32":
        return "bgra8unorm";
      case "vec4<f32>":
      case "vec4f":
        return "bgra8unorm";
      case "vec4<u32>":
      case "vec4u":
        return "rgba32uint";
      case "vec4<i32>":
      case "vec4i":
        return "rgba32sint";
      case "vec2<f32>":
      case "vec2f":
        return "rg32float";
      case "u32":
      case "u":
        return "r32uint";
      case "i32":
      case "i":
        return "r32sint";
      case "f16":
        return "rgba16float";
      case "vec4<f16>":
      case "vec4f16":
        return "rgba16float";
      case "vec2<f16>":
      case "vec2f16":
        return "rg16float";
      case "u16":
      case "u16":
        return "r16uint";
      case "i16":
      case "i16":
        return "r16sint";
      case "vec4<unorm>":
      case "vec4unorm":
        return "rgba8unorm";
      case "vec4<snorm>":
      case "vec4snorm":
        return "rgba8snorm";
      case "vec4<u8>":
      case "vec4u8":
        return "rgba8uint";
      case "vec4<i8>":
      case "vec4i8":
        return "rgba8sint";
      default:
        return "rgba8unorm";
    }
  }
}

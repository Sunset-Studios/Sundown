import { Renderer } from "./renderer.js";
import { ResourceCache } from "./resource_cache.js";
import { ShaderArchive } from "./shader_archive.js";
import {
  ShaderPrecisionProfile,
  canonicalize_shader_defines,
  create_shader_variant_key,
} from "./shader_archive_common.js";
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
  rg16float_format,
  r32uint_format,
  r32sint_format,
  r16uint_format,
  r16sint_format,
} from "../utility/config_permutations.js";
import { error } from "../utility/logging.js";

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

export class Shader {
  module = null;
  code = null;
  file_path = "";
  defines = {};
  reflection = null;
  precision_profile = ShaderPrecisionProfile.F32;

  initialize(file_path, defines = {}) {
    const renderer = Renderer.get();
    const canonical_defines = canonicalize_shader_defines(defines);
    const precision_profile = renderer.has_f16
      ? ShaderPrecisionProfile.F16
      : ShaderPrecisionProfile.F32;
    const variant_key = create_shader_variant_key(file_path, canonical_defines, precision_profile);

    if (!ShaderArchive.is_loaded()) {
      throw new Error(
        `Shader archives were not loaded before creating '${variant_key}'. Renderer setup must await the cooked shader archives before shader creation begins.`
      );
    }

    const variant = ShaderArchive.require_variant(variant_key);

    try {
      this.code = variant.code;
      this.module = renderer.device.createShaderModule({
        label: variant_key,
        code: variant.code,
      });
      this.file_path = variant.path;
      this.defines = canonical_defines;
      this.precision_profile = precision_profile;
      this.reflection = variant.reflection;
    } catch (err) {
      console.log(this.code);
      error(`WebGPU shader error: could not create shader module at ${variant_key}`, err);
      throw err;
    }
  }

  static create(file_path, defines = null, force_recreate = false) {
    const renderer = Renderer.get();
    const canonical_defines = canonicalize_shader_defines(defines ?? {});
    const precision_profile = renderer.has_f16
      ? ShaderPrecisionProfile.F16
      : ShaderPrecisionProfile.F32;
    const shader_id = create_shader_variant_key(file_path, canonical_defines, precision_profile);
    let shader = ResourceCache.get().fetch(CacheTypes.SHADER, shader_id);

    if (shader && force_recreate) {
      shader = null;
    }

    if (!shader) {
      shader = new Shader();
      shader.initialize(file_path, canonical_defines);
      ResourceCache.get().store(CacheTypes.SHADER, shader_id, shader);
    }

    return shader_id;
  }

  static register_shader_path(shader_path) {
    if (Array.isArray(shader_path)) {
      return shader_path.map((path) => this.register_shader_path(path));
    }

    return ShaderArchive.register_manifest_path(shader_path);
  }

  static register_optional_shader_path(shader_path) {
    if (Array.isArray(shader_path)) {
      return shader_path.map((path) => this.register_optional_shader_path(path));
    }

    return ShaderArchive.register_manifest_path(shader_path, { optional: true });
  }

  static resource_type_from_reflection_type(type) {
    switch (type) {
      case ShaderResourceType.Texture:
      case "texture":
        return ShaderResourceType.Texture;
      case ShaderResourceType.Sampler:
      case "sampler":
        return ShaderResourceType.Sampler;
      case ShaderResourceType.Storage:
      case "storage":
        return ShaderResourceType.Storage;
      case ShaderResourceType.Uniform:
      case "uniform":
        return ShaderResourceType.Uniform;
      case ShaderResourceType.StorageTexture:
      case "storageTexture":
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

import { read_file_async, read_file_bytes_async } from "../utility/file_system.js";
import { inflate } from "pako";
import { error } from "../utility/logging.js";
import {
  shader_archive_manifest_asset_path,
} from "./cooked_asset_config.js";

class NormalizedShaderReflection {
  constructor(metadata = {}) {
    this.bind_groups = metadata.bindGroups ?? [];
    this.entry = metadata.entry ?? {
      vertex: [],
      fragment: [],
      compute: [],
    };
  }

  getBindGroups() {
    return this.bind_groups;
  }
}

export class ShaderArchive {
  static manifest = null;
  static binary = null;
  static load_promise = null;
  static variant_cache = new Map();
  static text_cache = new Map();
  static decoder = new TextDecoder();

  static is_loaded() {
    return Boolean(this.manifest) && this.binary instanceof ArrayBuffer;
  }

  static async load() {
    if (this.is_loaded()) {
      return this.manifest;
    }

    if (this.load_promise) {
      return this.load_promise;
    }

    this.load_promise = (async () => {
      const manifest_path = shader_archive_manifest_asset_path;
      const manifest_text = await read_file_async(manifest_path);
      if (!manifest_text) {
        throw new Error(
          `Cooked shader manifest '${manifest_path}' could not be loaded. Run the shader cook step before starting the renderer.`
        );
      }

      let manifest = null;
      try {
        manifest = JSON.parse(manifest_text);
      } catch (err) {
        throw new Error(
          `Cooked shader manifest '${manifest_path}' is not valid JSON: ${err?.message ?? err}`
        );
      }

      const binary_path = manifest?.binaryAssetPath;
      if (!binary_path) {
        throw new Error(
          `Cooked shader manifest '${manifest_path}' is missing its binary asset path.`
        );
      }

      const binary = await read_file_bytes_async(binary_path);
      if (!(binary instanceof ArrayBuffer)) {
        throw new Error(
          `Cooked shader binary '${binary_path}' could not be loaded. Run the shader cook step before starting the renderer.`
        );
      }

      if (!manifest?.variants || typeof manifest.variants !== "object") {
        throw new Error(
          `Cooked shader manifest '${manifest_path}' is missing its variant table.`
        );
      }

      this.manifest = manifest;
      this.binary = binary;
      this.variant_cache.clear();
      this.text_cache.clear();
      return manifest;
    })().catch((err) => {
      this.load_promise = null;
      error("Failed to load cooked shader archive", err);
      throw err;
    });

    return this.load_promise;
  }

  static getVariant(variant_key) {
    if (!this.is_loaded()) {
      return null;
    }

    if (this.variant_cache.has(variant_key)) {
      return this.variant_cache.get(variant_key);
    }

    const metadata = this.manifest.variants[variant_key];
    if (!metadata) {
      return null;
    }

    const variant = {
      key: variant_key,
      path: metadata.path,
      defines: metadata.defines ?? {},
      precision_profile: metadata.precisionProfile,
      code: this.#decode_shader_code(variant_key, metadata),
      reflection: new NormalizedShaderReflection(metadata.reflection),
    };

    this.variant_cache.set(variant_key, variant);
    return variant;
  }

  static requireVariant(variant_key) {
    const variant = this.getVariant(variant_key);
    if (!variant) {
      throw new Error(
        `Cooked shader variant '${variant_key}' was not found in '${this.manifest?.manifestAssetPath ?? shader_archive_manifest_asset_path}'. Re-run the shader cook step so the archive includes this permutation.`
      );
    }
    return variant;
  }

  static #decode_shader_code(variant_key, metadata) {
    if (this.text_cache.has(variant_key)) {
      return this.text_cache.get(variant_key);
    }

    const code_offset = metadata.codeOffset ?? 0;
    const code_length = metadata.codeLength ?? 0;
    const code_bytes = new Uint8Array(this.binary, code_offset, code_length);
    const compression = metadata.compression ?? "identity";

    let code = null;
    if (compression === "identity") {
      code = this.decoder.decode(code_bytes);
    } else if (compression === "deflate") {
      code = this.decoder.decode(inflate(code_bytes));
    } else {
      throw new Error(
        `Cooked shader variant '${variant_key}' uses unsupported compression '${compression}'.`
      );
    }

    this.text_cache.set(variant_key, code);
    return code;
  }
}

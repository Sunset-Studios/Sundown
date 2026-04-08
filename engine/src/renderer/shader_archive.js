import { read_file_async, read_file_bytes_async } from "../utility/file_system.js";
import { inflate } from "pako";
import { error } from "../utility/logging.js";
import {
  shader_archive_manifest_asset_path,
  shader_archive_manifest_name,
} from "./cooked_asset_config.js";

class NormalizedShaderReflection {
  constructor(metadata = {}) {
    this.bind_groups = metadata.bind_groups ?? [];
    this.entry = metadata.entry ?? {
      vertex: [],
      fragment: [],
      compute: [],
    };
  }

  get_bind_groups() {
    return this.bind_groups;
  }
}

export class ShaderArchive {
  static archives = [];
  static load_promise = null;
  static variant_cache = new Map();
  static text_cache = new Map();
  static decoder = new TextDecoder();
  static registered_archives = [
    {
      manifest_path: shader_archive_manifest_asset_path,
      optional: false,
    },
  ];
  static registered_manifest_path_set = new Set([shader_archive_manifest_asset_path]);

  static is_loaded() {
    return this.archives.length > 0;
  }

  static register_manifest_path(path, options = {}) {
    const manifest_path = this.#normalize_manifest_path(path);
    const optional = Boolean(options.optional);

    if (this.load_promise) {
      throw new Error(
        `Shader archive '${manifest_path}' must be registered before renderer setup begins.`
      );
    }

    if (this.registered_manifest_path_set.has(manifest_path)) {
      return manifest_path;
    }

    this.registered_archives.push({
      manifest_path,
      optional,
    });
    this.registered_manifest_path_set.add(manifest_path);
    return manifest_path;
  }

  static async load() {
    if (this.is_loaded()) {
      return this.archives[0]?.manifest ?? null;
    }

    if (this.load_promise) {
      return this.load_promise;
    }

    this.load_promise = (async () => {
      const archives = [];
      for (const registered_archive of this.registered_archives) {
        const archive = await this.#load_archive(
          registered_archive.manifest_path,
          registered_archive.optional
        );
        if (archive) {
          archives.push(archive);
        }
      }

      this.archives = archives;
      this.variant_cache.clear();
      this.text_cache.clear();
      return archives[0]?.manifest ?? null;
    })().catch((err) => {
      this.load_promise = null;
      error("Failed to load cooked shader archives", err);
      throw err;
    });

    return this.load_promise;
  }

  static get_variant(variant_key) {
    if (!this.is_loaded()) {
      return null;
    }

    if (this.variant_cache.has(variant_key)) {
      return this.variant_cache.get(variant_key);
    }

    for (let archive_index = this.archives.length - 1; archive_index >= 0; --archive_index) {
      const archive = this.archives[archive_index];
      const metadata = archive.manifest.variants[variant_key];
      if (!metadata) {
        continue;
      }

      const variant = {
        key: variant_key,
        path: metadata.path,
        defines: metadata.defines ?? {},
        precision_profile: metadata.precisionProfile,
        code: this.#decode_shader_code(archive, variant_key, metadata),
        reflection: new NormalizedShaderReflection(metadata.reflection),
        manifest_path: archive.manifest_path,
      };

      this.variant_cache.set(variant_key, variant);
      return variant;
    }

    return null;
  }

  static require_variant(variant_key) {
    const variant = this.get_variant(variant_key);
    if (!variant) {
      const loaded_manifest_paths =
        this.archives.length > 0
          ? this.archives.map((archive) => `'${archive.manifest_path}'`).join(", ")
          : `'${shader_archive_manifest_asset_path}'`;
      throw new Error(
        `Cooked shader variant '${variant_key}' was not found in any loaded shader archive (${loaded_manifest_paths}). Re-run the shader cook step so the archive includes this permutation.`
      );
    }
    return variant;
  }

  static async #load_archive(manifest_path, optional = false) {
    const manifest_text = await read_file_async(manifest_path);
    if (!manifest_text) {
      if (optional) {
        return null;
      }

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

    return {
      manifest,
      binary,
      manifest_path,
    };
  }

  static #normalize_manifest_path(path) {
    const normalized_path = String(path ?? "")
      .trim()
      .replace(/\\/g, "/");
    if (!normalized_path) {
      throw new Error("Shader archive registration requires a manifest path or archive directory.");
    }

    if (normalized_path.toLowerCase().endsWith(".json")) {
      return normalized_path;
    }

    return `${normalized_path.replace(/\/+$/g, "")}/${shader_archive_manifest_name}`;
  }

  static #decode_shader_code(archive, variant_key, metadata) {
    if (this.text_cache.has(variant_key)) {
      return this.text_cache.get(variant_key);
    }

    const code_offset = metadata.codeOffset ?? 0;
    const code_length = metadata.codeLength ?? 0;
    const code_bytes = new Uint8Array(archive.binary, code_offset, code_length);
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

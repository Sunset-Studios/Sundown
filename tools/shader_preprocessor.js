import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { deflateSync } from "zlib";
import { WgslReflect, ResourceType } from "wgsl_reflect/wgsl_reflect.node.js";
import { ShaderResourceType } from "../engine/src/renderer/renderer_types.js";
import {
  ShaderPrecisionProfile,
  canonicalize_shader_defines,
  create_shader_variant_key,
} from "../engine/src/renderer/shader_archive_common.js";
import {
  shader_archive_binary_asset_path,
  shader_archive_binary_name,
  shader_archive_manifest_asset_path,
} from "../engine/src/renderer/cooked_asset_config.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "..");
const SHADER_ROOT = path.resolve(PROJECT_ROOT, "assets/engine/shaders");
const MANIFEST_OUTPUT_PATH = path.resolve(
  PROJECT_ROOT,
  "assets",
  ...shader_archive_manifest_asset_path.split("/")
);
const BINARY_OUTPUT_PATH = path.resolve(
  PROJECT_ROOT,
  "assets",
  ...shader_archive_binary_asset_path.split("/")
);
const GENERATOR_VERSION = 1;
const compression_mode = "deflate";

const include_string = "#include";
const include_regex = /^#include\s+"(\S+)".*$/m;
const defines_regex = /#define\s+(\S+)(?:\s+(\S*))?$/gm;

const precision_float_string = "precision_float";
const has_precision_float_string = "HAS_PRECISION_FLOAT";
const has_subgroups_string = "HAS_SUBGROUPS";

const f16_type_string = "f16";
const f32_type_string = "f32";

const material_pass_variants = [
  { MESHLET_RASTER_PASS: true },
  { MESHLET_DEPTH_PASS: true },
  { MESHLET_RESOLVE_PASS: true },
];

const shadow_enabled_shader_paths = [
  "shadow/as_vsm/feedback.wgsl",
  "shadow/as_vsm/evict_unused_pages.wgsl",
  "shadow/as_vsm/page_table_update.wgsl",
  "shadow/as_vsm/tile_clear.wgsl",
  "shadow/as_vsm/clear_tile_flags.wgsl",
  "shadow/as_vsm/dirty_visible_light_tiles.wgsl",
  "shadow/as_vsm/tile_render.vert.wgsl",
  "shadow/as_vsm/resolve_depth_to_atlas.wgsl",
  "shadow/as_vsm/debug_shadow_atlas.wgsl",
  "shadow/as_vsm/debug_page_table.wgsl",
  "shadow/as_vsm/debug_tile_overlay.wgsl",
  "shadow/as_vsm/debug_tile_render.wgsl",
  "shadow/as_vsm/debug_dirty_tiles.wgsl",
  "shadow/as_vsm/debug_dirty_shadow_meshlets.wgsl",
  "shadow/as_vsm/dirty_movable_entities.wgsl",
  "shadow/as_vsm/dirty_slice_reducer.wgsl",
  "shadow/as_vsm/cull_shadow_meshlets.wgsl",
  "shadow/as_vsm/compact_shadow_dirty_meshlets.wgsl",
];

function sort_shader_paths(shader_paths) {
  return [...shader_paths].sort((a, b) => a.localeCompare(b));
}

function quoted_shader_path_regex() {
  return /["'`]([^"'`\r\n]+\.wgsl)["'`]/g;
}

function add_variant(variants, file_path, defines = {}) {
  const canonical_defines = canonicalize_shader_defines(defines);
  const define_key = JSON.stringify(Object.entries(canonical_defines));
  const variant_key = `${file_path}|${define_key}`;
  if (!variants.has(variant_key)) {
    variants.set(variant_key, {
      path: file_path,
      defines: canonical_defines,
    });
  }
}

function add_material_variants(variants, file_path, transparent_values = [false, true]) {
  for (const pass_defines of material_pass_variants) {
    for (const transparent of transparent_values) {
      const defines = { ...pass_defines };
      if (transparent) {
        defines.TRANSPARENT = true;
      }
      add_variant(variants, file_path, defines);
    }
  }
}

function add_deferred_lighting_variants(variants) {
  for (const gi_enabled of [false, true]) {
    for (const shadows_enabled of [false, true]) {
      for (const ao_enabled of [false, true]) {
        add_variant(variants, "deferred_lighting.wgsl", {
          GI_ENABLED: gi_enabled,
          SHADOWS_ENABLED: shadows_enabled,
          AO_ENABLED: ao_enabled,
        });
      }
    }
  }
}

function find_shader_files(shader_root) {
  const shader_files = [];

  function walk(current_dir) {
    const entries = fs.readdirSync(current_dir, { withFileTypes: true });
    for (const entry of entries) {
      const full_path = path.join(current_dir, entry.name);
      if (entry.isDirectory()) {
        walk(full_path);
        continue;
      }

      if (entry.isFile() && entry.name.toLowerCase().endsWith(".wgsl")) {
        shader_files.push(path.relative(shader_root, full_path).replace(/\\/g, "/"));
      }
    }
  }

  walk(shader_root);
  return sort_shader_paths(shader_files);
}

function discover_referenced_shader_paths(project_root, available_shader_paths) {
  const available = new Set(available_shader_paths);
  const referenced = new Set();
  const engine_src_root = path.join(project_root, "engine", "src");

  function walk(current_dir) {
    const entries = fs.readdirSync(current_dir, { withFileTypes: true });
    for (const entry of entries) {
      const full_path = path.join(current_dir, entry.name);
      if (entry.isDirectory()) {
        walk(full_path);
        continue;
      }

      if (!entry.isFile() || !entry.name.toLowerCase().endsWith(".js")) {
        continue;
      }

      const source = fs.readFileSync(full_path, "utf8");
      const regex = quoted_shader_path_regex();
      let match = regex.exec(source);
      while (match) {
        const shader_path = match[1];
        if (available.has(shader_path)) {
          referenced.add(shader_path);
        }
        match = regex.exec(source);
      }
    }
  }

  walk(engine_src_root);
  return sort_shader_paths(referenced);
}

function build_shader_variant_registry(shader_paths) {
  const variants = new Map();

  for (const shader_path of shader_paths) {
    add_variant(variants, shader_path, {});
  }

  add_material_variants(variants, "visibility/visibility_draw_standard.wgsl");
  add_material_variants(variants, "ui_standard_material.wgsl");
  add_material_variants(variants, "text_material.wgsl");

  add_variant(variants, "effects/bloom_downsample.wgsl", {
    HIGH_QUALITY_DOWNSAMPLE: true,
  });

  add_deferred_lighting_variants(variants);

  for (const shader_path of shadow_enabled_shader_paths) {
    add_variant(variants, shader_path, { SHADOWS_ENABLED: true });
  }

  return [...variants.values()].sort((a, b) => {
    const path_diff = a.path.localeCompare(b.path);
    if (path_diff !== 0) {
      return path_diff;
    }
    return JSON.stringify(a.defines).localeCompare(JSON.stringify(b.defines));
  });
}

function validate_shader_registry(shader_paths, referenced_shader_paths, variants) {
  const shader_path_set = new Set(shader_paths);
  const variant_path_set = new Set(variants.map((variant) => variant.path));

  for (const shader_path of shadow_enabled_shader_paths) {
    if (!shader_path_set.has(shader_path)) {
      throw new Error(
        `Cooked shader registry references missing shader '${shader_path}'.`
      );
    }
  }

  for (const referenced_path of referenced_shader_paths) {
    if (!shader_path_set.has(referenced_path)) {
      throw new Error(
        `Referenced shader '${referenced_path}' does not exist under assets/engine/shaders.`
      );
    }

    if (!variant_path_set.has(referenced_path)) {
      throw new Error(
        `Referenced shader '${referenced_path}' is missing from the cooked shader registry.`
      );
    }
  }
}

function create_cook_settings() {
  return {
    version: GENERATOR_VERSION,
    manifest_asset_path: shader_archive_manifest_asset_path,
    binary_asset_path: shader_archive_binary_asset_path,
    binary_asset_name: shader_archive_binary_name,
    compression: compression_mode,
    precision_profiles: [ShaderPrecisionProfile.F16, ShaderPrecisionProfile.F32],
  };
}

function create_source_hash(settings, shader_files, variants) {
  const hash = crypto.createHash("sha1");
  hash.update(JSON.stringify(settings));

  for (const shader_path of shader_files) {
    hash.update(shader_path);
    hash.update(fs.readFileSync(path.join(SHADER_ROOT, shader_path), "utf8"));
  }

  for (const variant of variants) {
    hash.update(variant.path);
    hash.update(JSON.stringify(variant.defines));
  }

  return hash.digest("hex");
}

function should_skip_generation(manifest_path, binary_path, source_hash) {
  if (!fs.existsSync(manifest_path) || !fs.existsSync(binary_path)) {
    return false;
  }

  try {
    const manifest = JSON.parse(fs.readFileSync(manifest_path, "utf8"));
    return manifest?.source?.hash === source_hash;
  } catch {
    return false;
  }
}

function read_shader_source(file_path) {
  const absolute_path = path.join(SHADER_ROOT, file_path);
  if (!fs.existsSync(absolute_path)) {
    return null;
  }

  return fs.readFileSync(absolute_path, "utf8");
}

function parse_shader_includes(
  file_path,
  code,
  defines = {},
  load_recursion_step = 0,
  precision_profile
) {
  let include_positions = [];

  let pos = code.indexOf(include_string, 0);
  while (pos !== -1) {
    include_positions.push(pos);
    pos = code.indexOf(include_string, pos + 1);
  }

  for (let i = include_positions.length - 1; i >= 0; --i) {
    const start = include_positions[i];
    const end = code.indexOf("\n", start);
    const line_end = end === -1 ? code.length : end;
    const include_line = code.substring(start, line_end);
    const match = include_line.match(include_regex);
    if (match) {
      const include_contents = load_shader_text(
        match[1],
        defines,
        load_recursion_step + 1,
        precision_profile
      );
      code = code.slice(0, start) + include_contents + code.slice(line_end);
    }
  }

  return code;
}

function build_defines_map_and_strip(code, defines, precision_profile) {
  const defines_map = Object.assign({}, defines);
  const stripped_code = code.replace(defines_regex, (match, key, value) => {
    if (!(key in defines_map)) {
      defines_map[key] = value || true;
    }
    return "";
  });

  defines_map[precision_float_string] =
    precision_profile === ShaderPrecisionProfile.F16 ? f16_type_string : f32_type_string;
  defines_map[has_precision_float_string] = precision_profile === ShaderPrecisionProfile.F16;
  defines_map[has_subgroups_string] = true;
  return { defines_map, stripped_code };
}

function parse_conditional_defines_and_types(code, defines) {
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
      const should_include = defines[condition] === (value || true);
      stack.push({ include: should_include });
    } else if ((match = ifndef_regex.exec(line))) {
      const [, condition] = match;
      const should_include = !defines[condition];
      stack.push({ include: should_include });
    } else if (else_regex.test(line)) {
      const frame = stack[stack.length - 1];
      frame.include = !frame.include;
    } else if (endif_regex.test(line)) {
      stack.pop();
    } else if (stack.every((frame) => frame.include)) {
      output_lines.push(line);
    }
  }

  let result = output_lines.join("\n");

  for (const [key, value] of Object.entries(defines)) {
    if (typeof value === "string") {
      result = result.replace(new RegExp(`\\b${key}\\b`, "g"), value);
    }
  }

  return result.trim();
}

function parse_and_strip_macros(code) {
  const macros = new Map();
  const lines = code.split(/\r\n|\r|\n/);
  const out_lines = [];
  const param_define_regex = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(.*)$/;

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trimStart();
    if (trimmed.startsWith("#macro")) {
      const match = trimmed.match(
        /^#macro\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*$/
      );
      if (match) {
        const name = match[1];
        const args = (match[2] || "")
          .split(",")
          .map((value) => value.trim())
          .filter((value) => value.length > 0);
        const body_lines = [];
        i++;
        let found_end = false;
        while (i < lines.length) {
          const current_line = lines[i];
          if (current_line.trimStart().startsWith("#endmacro")) {
            found_end = true;
            break;
          }
          body_lines.push(current_line);
          i++;
        }

        if (!found_end) {
          out_lines.push(line);
        } else {
          macros.set(name, {
            args,
            body: body_lines.join("\n"),
          });
        }

        i++;
        continue;
      }
    }

    const param_match = line.match(param_define_regex);
    if (param_match) {
      const name = param_match[1];
      const args = param_match[2]
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);
      const body = param_match[3] || "";
      macros.set(name, { args, body });
      i++;
      continue;
    }

    out_lines.push(line);
    i++;
  }

  return {
    code: out_lines.join("\n"),
    macros,
  };
}

function split_macro_args(source) {
  const args = [];
  let depth = 0;
  let start = 0;

  for (let i = 0; i < source.length; i++) {
    const character = source[i];
    if (character === "(" || character === "[" || character === "{") {
      depth++;
    } else if (character === ")" || character === "]" || character === "}") {
      depth--;
    } else if (character === "," && depth === 0) {
      args.push(source.slice(start, i).trim());
      start = i + 1;
    }
  }

  const last = source.slice(start).trim();
  if (last.length) {
    args.push(last);
  }

  return args;
}

function expand_once(source, macros, macro_names) {
  let i = 0;
  let out = "";
  let changed = false;

  let in_line_comment = false;
  let in_block_comment = false;
  let in_string = false;
  let directive_line = false;

  const is_ident_start = (value) => /[A-Za-z_]/.test(value);
  const is_ident_char = (value) => /[A-Za-z0-9_]/.test(value);

  while (i < source.length) {
    const character = source[i];

    if (character === "\n") {
      in_line_comment = false;
      directive_line = false;
      out += character;
      i++;
      continue;
    }

    if (in_line_comment) {
      out += character;
      i++;
      continue;
    }

    if (in_block_comment) {
      if (character === "*" && source[i + 1] === "/") {
        out += "*/";
        i += 2;
        in_block_comment = false;
        continue;
      }

      out += character;
      i++;
      continue;
    }

    if (in_string) {
      if (character === "\\") {
        out += source.substr(i, 2);
        i += 2;
        continue;
      }

      out += character;
      if (character === '"') {
        in_string = false;
      }
      i++;
      continue;
    }

    if (character === "/" && source[i + 1] === "/") {
      out += "//";
      i += 2;
      in_line_comment = true;
      continue;
    }

    if (character === "/" && source[i + 1] === "*") {
      out += "/*";
      i += 2;
      in_block_comment = true;
      continue;
    }

    if (character === '"') {
      in_string = true;
      out += character;
      i++;
      continue;
    }

    if (
      (character === "#" && (i === 0 || source[i - 1] === "\n")) ||
      (directive_line && character !== "\n")
    ) {
      directive_line = true;
      out += character;
      i++;
      continue;
    }

    if (is_ident_start(character)) {
      let identifier_end = i + 1;
      while (identifier_end < source.length && is_ident_char(source[identifier_end])) {
        identifier_end++;
      }

      const identifier = source.slice(i, identifier_end);
      if (macro_names.has(identifier)) {
        let argument_start = identifier_end;
        while (argument_start < source.length && /\s/.test(source[argument_start])) {
          argument_start++;
        }

        if (source[argument_start] === "(") {
          let argument_end = argument_start + 1;
          let depth = 1;

          while (argument_end < source.length && depth > 0) {
            if (source[argument_end] === "(") {
              depth++;
            } else if (source[argument_end] === ")") {
              depth--;
            } else if (source[argument_end] === '"') {
              argument_end++;
              while (argument_end < source.length) {
                if (source[argument_end] === '"' && source[argument_end - 1] !== "\\") {
                  argument_end++;
                  break;
                }
                argument_end++;
              }
              continue;
            }
            argument_end++;
          }

          const raw_args = source.slice(argument_start + 1, argument_end - 1);
          const actual_args = split_macro_args(raw_args);
          const { args: formal_args, body } = macros.get(identifier);
          let replacement = body;
          for (let formal_index = 0; formal_index < formal_args.length; formal_index++) {
            const formal = formal_args[formal_index];
            const actual = actual_args[formal_index] !== undefined ? actual_args[formal_index] : "";
            replacement = replacement.replace(new RegExp(`\\b${formal}\\b`, "g"), `(${actual})`);
          }

          out += replacement;
          i = argument_end;
          changed = true;
          continue;
        }
      }

      out += identifier;
      i = identifier_end;
      continue;
    }

    out += character;
    i++;
  }

  return { out, changed };
}

function expand_macros(code, macros, max_depth = 16) {
  if (!macros || macros.size === 0) {
    return code;
  }

  const macro_names = new Set(macros.keys());
  let current = code;
  for (let pass = 0; pass < max_depth; pass++) {
    const { out, changed } = expand_once(current, macros, macro_names);
    current = out;
    if (!changed) {
      break;
    }
  }
  return current;
}

function load_shader_text(file_path, defines = {}, load_recursion_step = 0, precision_profile) {
  let asset = read_shader_source(file_path);
  if (!asset) {
    throw new Error(`Could not find shader '${file_path}' while cooking.`);
  }

  asset = parse_shader_includes(
    file_path,
    asset,
    defines,
    load_recursion_step,
    precision_profile
  );

  if (load_recursion_step === 0) {
    const { defines_map, stripped_code } = build_defines_map_and_strip(
      asset,
      defines,
      precision_profile
    );
    asset = parse_conditional_defines_and_types(stripped_code, defines_map);
  }

  if (load_recursion_step === 0) {
    const { code: code_without_macros, macros } = parse_and_strip_macros(asset);
    asset = expand_macros(code_without_macros, macros);
  }

  return asset;
}

function normalize_type(type) {
  if (!type) {
    return {
      name: "",
    };
  }

  const normalized = {
    name: type.name ?? "",
  };

  if (type.format) {
    normalized.format = {
      name: type.format.name ?? type.format,
    };
  }

  if (type.access !== undefined) {
    normalized.access = type.access;
  }

  return normalized;
}

function normalize_resource_type(resource_type) {
  switch (resource_type) {
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
      throw new Error(`Unsupported reflection resource type '${resource_type}'.`);
  }
}

function normalize_bind_groups(reflection) {
  return reflection.getBindGroups().map((group) =>
    group.map((binding) => ({
      binding: binding.binding,
      group: binding.group,
      name: binding.name ?? "",
      access: binding.access ?? null,
      resourceType: normalize_resource_type(binding.resourceType),
      type: normalize_type(binding.type),
    }))
  );
}

function normalize_entry_stage(entries = []) {
  return entries.map((entry) => ({
    name: entry.name ?? "",
    stage: entry.stage ?? "",
    outputs: (entry.outputs ?? []).map((output) => ({
      name: output.name ?? "",
      type: normalize_type(output.type),
    })),
  }));
}

function reflect_shader(code) {
  const reflection = new WgslReflect(code);
  return {
    bindGroups: normalize_bind_groups(reflection),
    entry: {
      vertex: normalize_entry_stage(reflection.entry?.vertex ?? []),
      fragment: normalize_entry_stage(reflection.entry?.fragment ?? []),
      compute: normalize_entry_stage(reflection.entry?.compute ?? []),
    },
  };
}

function build_output(settings, shader_files, variants) {
  const encoder = new TextEncoder();
  const chunks = [];
  const manifest_variants = {};
  const payload_offsets = new Map();
  let running_offset = 0;

  for (const precision_profile of [ShaderPrecisionProfile.F16, ShaderPrecisionProfile.F32]) {
    for (const variant of variants) {
      const canonical_defines = canonicalize_shader_defines(variant.defines);
      const shader_key = create_shader_variant_key(
        variant.path,
        canonical_defines,
        precision_profile
      );
      if (manifest_variants[shader_key]) {
        throw new Error(`Duplicate cooked shader variant key '${shader_key}'.`);
      }

      const code = load_shader_text(variant.path, canonical_defines, 0, precision_profile);
      const code_bytes = encoder.encode(code);
      const compressed_bytes = deflateSync(code_bytes);
      const reflection = reflect_shader(code);
      const payload_hash = crypto
        .createHash("sha1")
        .update(code)
        .digest("hex");
      let payload_info = payload_offsets.get(payload_hash);
      if (!payload_info) {
        payload_info = {
          offset: running_offset,
          length: compressed_bytes.byteLength,
          compression: settings.compression,
          uncompressedLength: code_bytes.byteLength,
        };
        payload_offsets.set(payload_hash, payload_info);
        chunks.push(compressed_bytes);
        running_offset += compressed_bytes.byteLength;
      }

      manifest_variants[shader_key] = {
        path: variant.path,
        defines: canonical_defines,
        precisionProfile: precision_profile,
        codeOffset: payload_info.offset,
        codeLength: payload_info.length,
        codeUncompressedLength: payload_info.uncompressedLength,
        compression: payload_info.compression,
        reflection,
      };
    }
  }

  const binary = new Uint8Array(running_offset);
  let write_offset = 0;
  for (const chunk of chunks) {
    binary.set(chunk, write_offset);
    write_offset += chunk.byteLength;
  }

  const manifest = {
    version: GENERATOR_VERSION,
    generator: "tools/shader_preprocessor.js",
    generatedAt: new Date().toISOString(),
    manifestAssetPath: shader_archive_manifest_asset_path,
    binaryAssetPath: shader_archive_binary_asset_path,
    binaryAssetName: shader_archive_binary_name,
    settings,
    source: {
      hash: create_source_hash(settings, shader_files, variants),
      shaderCount: shader_files.length,
      variantCount: Object.keys(manifest_variants).length,
      payloadCount: payload_offsets.size,
    },
    sections: {
      code: {
        offset: 0,
        byteLength: binary.byteLength,
      },
    },
    variants: manifest_variants,
  };

  return {
    manifest,
    binary: Buffer.from(binary.buffer, binary.byteOffset, binary.byteLength),
  };
}

function main() {
  const settings = create_cook_settings();
  const shader_files = find_shader_files(SHADER_ROOT);
  const referenced_shader_paths = discover_referenced_shader_paths(PROJECT_ROOT, shader_files);
  const variants = build_shader_variant_registry(referenced_shader_paths);
  const source_hash = create_source_hash(settings, shader_files, variants);

  validate_shader_registry(shader_files, referenced_shader_paths, variants);

  if (should_skip_generation(MANIFEST_OUTPUT_PATH, BINARY_OUTPUT_PATH, source_hash)) {
    console.log(
      `[shader_preprocessor] up to date: ${path.relative(PROJECT_ROOT, MANIFEST_OUTPUT_PATH).replace(/\\/g, "/")}`
    );
    return;
  }

  const output = build_output(settings, shader_files, variants);

  fs.mkdirSync(path.dirname(MANIFEST_OUTPUT_PATH), { recursive: true });
  fs.writeFileSync(MANIFEST_OUTPUT_PATH, JSON.stringify(output.manifest, null, 2));
  fs.mkdirSync(path.dirname(BINARY_OUTPUT_PATH), { recursive: true });
  fs.writeFileSync(BINARY_OUTPUT_PATH, output.binary);

  console.log(
    `[shader_preprocessor] generated ${path.relative(PROJECT_ROOT, MANIFEST_OUTPUT_PATH).replace(/\\/g, "/")} (${output.manifest.source.variantCount} variants)`
  );
  console.log(
    `[shader_preprocessor] generated ${path.relative(PROJECT_ROOT, BINARY_OUTPUT_PATH).replace(/\\/g, "/")} (${output.binary.byteLength} bytes)`
  );
}

main();

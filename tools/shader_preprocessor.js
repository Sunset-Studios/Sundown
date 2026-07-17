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
  build_shader_archive_asset_path,
  shader_archive_asset_dir,
  shader_archive_binary_name,
  shader_archive_manifest_name,
  get_project_shader_root_asset_path,
  get_project_source_root,
} from "../engine/src/renderer/cooked_asset_config.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "..");
const ENGINE_SOURCE_ROOT = "engine/src";
const GENERATOR_VERSION = 1;
const compression_mode = "deflate";
const source_file_extensions = new Set([".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"]);
const project_root_patterns = [
  /\bproject\s*:\s*\{[\s\S]*?\broot\s*:\s*["'`]([^"'`]+)["'`]/g,
  /\bProjectContext\.configure\s*\(\s*\{[\s\S]*?\broot\s*:\s*["'`]([^"'`]+)["'`]/g,
];

const include_string = "#include";
const include_regex = /^#include\s+"(\S+)".*$/m;
const defines_regex = /#define\s+(\S+)(?:\s+(\S*))?$/gm;

const precision_float_string = "precision_float";
const has_precision_float_string = "HAS_PRECISION_FLOAT";
const built_in_shader_define_keys = new Set([
  precision_float_string,
  has_precision_float_string,
]);

const f16_type_string = "f16";
const f32_type_string = "f32";
const shader_conditional_regex = /#if(?:ndef)?\s+([A-Za-z_][A-Za-z0-9_]*)/g;
const shader_define_declaration_regex = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)/gm;
const const_shader_path_regex =
  /const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["'`]([^"'`\r\n]+\.wgsl)["'`]/g;
const static_const_object_regex =
  /const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:Object\.freeze\s*\(\s*)?\{/g;
const setup_object_regex = /const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{/g;
const setup_define_assignment_regex =
  /([A-Za-z_][A-Za-z0-9_]*)(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.defines\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);/g;
const generic_define_assignment_regex =
  /([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*["'`]([A-Za-z_][A-Za-z0-9_]*)["'`]\s*\]\s*=\s*([^;]+);/g;
const object_shader_path_regex =
  /path\s*:\s*(["'`][^"'`\r\n]+\.wgsl["'`]|[A-Za-z_][A-Za-z0-9_]*)/g;
const defines_object_regex = /defines\s*:\s*\{/g;
const spread_identifier_regex = /\.\.\.\s*([A-Za-z_][A-Za-z0-9_]*)/g;
const material_template_create_call_regex = /MaterialTemplate\.create\s*\(/g;
const shader_create_call_regex = /Shader\.create\s*\(/g;

function normalize_relative_path(value, fallback = "") {
  const normalized = String(value ?? fallback)
    .trim()
    .replace(/\\/g, "/")
    .replace(/^\.\//, "")
    .replace(/^\/+|\/+$/g, "");
  return normalized.length > 0 ? normalized : fallback;
}

function resolve_asset_path(asset_path) {
  return path.resolve(PROJECT_ROOT, "assets", ...asset_path.split("/"));
}

function resolve_repo_path(repo_path) {
  return path.resolve(PROJECT_ROOT, ...repo_path.split("/"));
}

function list_source_files(current_dir, source_files = []) {
  const skipped_dirs = new Set([".git", "node_modules", "dist", "assets", "target", "coverage"]);
  const entries = fs.readdirSync(current_dir, { withFileTypes: true });
  for (const entry of entries) {
    const full_path = path.join(current_dir, entry.name);
    if (entry.isDirectory()) {
      if (skipped_dirs.has(entry.name)) {
        continue;
      }

      list_source_files(full_path, source_files);
      continue;
    }

    if (entry.isFile() && source_file_extensions.has(path.extname(entry.name).toLowerCase())) {
      source_files.push(full_path);
    }
  }

  return source_files;
}

function create_shader_root_definition(
  asset_path,
  source_roots,
  include_engine_variants = false,
  is_engine_default = false
) {
  return {
    asset_path,
    manifest_asset_path: build_shader_archive_asset_path(shader_archive_manifest_name, asset_path),
    binary_asset_path: build_shader_archive_asset_path(shader_archive_binary_name, asset_path),
    shader_root_path: resolve_asset_path(asset_path),
    source_roots,
    include_engine_variants,
    is_engine_default,
  };
}

function build_include_resolution_roots(primary_shader_root, shader_roots) {
  const ordered_roots = [primary_shader_root];
  const seen_asset_paths = new Set([primary_shader_root.asset_path]);

  for (const shader_root of shader_roots) {
    if (shader_root.is_engine_default || seen_asset_paths.has(shader_root.asset_path)) {
      continue;
    }

    ordered_roots.push(shader_root);
    seen_asset_paths.add(shader_root.asset_path);
  }

  for (const shader_root of shader_roots) {
    if (!shader_root.is_engine_default || seen_asset_paths.has(shader_root.asset_path)) {
      continue;
    }

    ordered_roots.push(shader_root);
    seen_asset_paths.add(shader_root.asset_path);
  }

  return ordered_roots;
}

function discover_project_roots_from_source() {
  const project_roots = new Set();
  const source_files = list_source_files(PROJECT_ROOT);

  for (const source_file_path of source_files) {
    const source = fs.readFileSync(source_file_path, "utf8");
    for (const project_root_pattern of project_root_patterns) {
      const regex = new RegExp(project_root_pattern);
      let match = regex.exec(source);
      while (match) {
        const project_root = normalize_relative_path(match[1]);
        if (project_root) {
          project_roots.add(project_root);
        }
        match = regex.exec(source);
      }
    }
  }

  return [...project_roots].sort((a, b) => a.localeCompare(b));
}

function load_shader_roots() {
  const project_roots = discover_project_roots_from_source();
  const source_roots = [ENGINE_SOURCE_ROOT];
  const shader_roots = [
    create_shader_root_definition(shader_archive_asset_dir, [ENGINE_SOURCE_ROOT], true, true),
  ];

  for (const project_root of project_roots) {
    const project_source_root = get_project_source_root(project_root);
    if (project_source_root && !source_roots.includes(project_source_root)) {
      source_roots.push(project_source_root);
    }

    const project_shader_root_asset_path = get_project_shader_root_asset_path(project_root);
    if (!project_shader_root_asset_path) {
      continue;
    }

    const project_shader_root_path = resolve_asset_path(project_shader_root_asset_path);
    if (!fs.existsSync(project_shader_root_path)) {
      continue;
    }

    shader_roots.push(
      create_shader_root_definition(
        project_shader_root_asset_path,
        project_source_root ? [project_source_root] : []
      )
    );
  }

  const source_root_entries = source_roots
    .map((source_root) => ({
      source_root,
      source_root_path: resolve_repo_path(source_root),
    }))
    .filter((source_root) => fs.existsSync(source_root.source_root_path));

  return {
    shader_roots,
    source_roots: source_root_entries,
  };
}

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

function create_define_family() {
  return {
    required_domains: new Map(),
    optional_domains: new Map(),
  };
}

function add_define_domain_value(domains, key, value) {
  if (!key) {
    return;
  }

  if (!domains.has(key)) {
    domains.set(key, new Set());
  }

  domains.get(key).add(value);
}

function merge_define_domains(target_domains, source_domains) {
  for (const [key, values] of source_domains) {
    for (const value of values) {
      add_define_domain_value(target_domains, key, value);
    }
  }
}

function sort_values(values) {
  return [...values].sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
}

function expand_define_domains(defines, include_absent = false) {
  let variants = [{}];

  for (const key of [...defines.keys()].sort((a, b) => a.localeCompare(b))) {
    const next_variants = [];
    const values = sort_values(defines.get(key));

    for (const variant of variants) {
      if (include_absent) {
        next_variants.push({ ...variant });
      }

      for (const value of values) {
        next_variants.push({
          ...variant,
          [key]: value,
        });
      }
    }

    variants = next_variants;
  }

  return variants;
}

function expand_define_family(family) {
  const required_variants = expand_define_domains(family.required_domains, false);
  if (family.optional_domains.size === 0) {
    return required_variants;
  }

  const optional_variants = expand_define_domains(family.optional_domains, true);
  const variants = [];

  for (const required_variant of required_variants) {
    for (const optional_variant of optional_variants) {
      variants.push({
        ...required_variant,
        ...optional_variant,
      });
    }
  }

  return variants;
}

function add_variant_family_to_paths(
  variants_by_path,
  shader_paths,
  family,
  preserve_requested_defines = false
) {
  const expanded_variants = expand_define_family(family);
  for (const shader_path of shader_paths) {
    if (!variants_by_path.has(shader_path)) {
      variants_by_path.set(shader_path, new Map());
    }

    const path_variants = variants_by_path.get(shader_path);
    for (const defines of expanded_variants) {
      const canonical_defines = canonicalize_shader_defines(defines);
      const variant_key = JSON.stringify(Object.entries(canonical_defines));
      const existing_variant = path_variants.get(variant_key);
      if (!existing_variant) {
        path_variants.set(variant_key, {
          defines: canonical_defines,
          preserve_requested_defines,
        });
        continue;
      }

      existing_variant.preserve_requested_defines =
        existing_variant.preserve_requested_defines || preserve_requested_defines;
    }
  }
}

function build_const_shader_path_map(source) {
  const shader_paths = new Map();
  const regex = new RegExp(const_shader_path_regex);
  let match = regex.exec(source);
  while (match) {
    shader_paths.set(match[1], normalize_relative_path(match[2]));
    match = regex.exec(source);
  }

  const object_properties = extract_static_const_object_properties(source);
  for (const [property_name, property_value] of object_properties) {
    const shader_path = resolve_shader_path_token(property_value, shader_paths);
    if (shader_path) {
      shader_paths.set(property_name, shader_path);
    }
  }

  return shader_paths;
}

function extract_static_const_object_properties(source) {
  const properties = new Map();
  const regex = new RegExp(static_const_object_regex);
  let match = regex.exec(source);
  while (match) {
    const open_index = source.indexOf("{", match.index);
    const object_literal = extract_object_literal(source, open_index);
    if (!object_literal) {
      regex.lastIndex = match.index + match[0].length;
      match = regex.exec(source);
      continue;
    }

    for (const entry of split_top_level_values(object_literal.text.slice(1, -1))) {
      if (!entry || entry.startsWith("...")) {
        continue;
      }

      const separator_index = entry.indexOf(":");
      if (separator_index === -1) {
        continue;
      }

      const key = strip_wrapping_quotes(entry.slice(0, separator_index).trim());
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
        continue;
      }

      properties.set(`${match[1]}.${key}`, entry.slice(separator_index + 1).trim());
    }

    regex.lastIndex = object_literal.end_index + 1;
    match = regex.exec(source);
  }

  return properties;
}

function strip_wrapping_quotes(value) {
  const trimmed = value.trim();
  if (trimmed.length < 2) {
    return trimmed;
  }

  const first = trimmed[0];
  const last = trimmed[trimmed.length - 1];
  if ((first === '"' || first === "'" || first === "`") && last === first) {
    return trimmed.slice(1, -1);
  }

  return trimmed;
}

function resolve_shader_path_token(token, const_shader_paths) {
  const trimmed = token.trim();
  if (!trimmed) {
    return null;
  }

  if (/^["'`][^"'`\r\n]+\.wgsl["'`]$/.test(trimmed)) {
    return normalize_relative_path(strip_wrapping_quotes(trimmed));
  }

  return const_shader_paths.get(trimmed) ?? null;
}

function parse_define_literal(value) {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }

  if (trimmed === "true") {
    return true;
  }

  if (trimmed === "false") {
    return false;
  }

  if (trimmed === "null") {
    return null;
  }

  if (/^["'`][\s\S]*["'`]$/.test(trimmed)) {
    return strip_wrapping_quotes(trimmed);
  }

  if (/^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$/i.test(trimmed)) {
    return Number(trimmed);
  }

  return undefined;
}

function infer_define_values(value_expression) {
  const literal_value = parse_define_literal(value_expression);
  if (literal_value !== undefined) {
    return [literal_value];
  }

  return [false, true];
}

function find_matching_character(source, open_index, open_character, close_character) {
  let depth = 0;
  let in_single_quote = false;
  let in_double_quote = false;
  let in_template_string = false;
  let in_line_comment = false;
  let in_block_comment = false;

  for (let i = open_index; i < source.length; i++) {
    const character = source[i];
    const next_character = source[i + 1];
    const previous_character = source[i - 1];

    if (in_line_comment) {
      if (character === "\n") {
        in_line_comment = false;
      }
      continue;
    }

    if (in_block_comment) {
      if (character === "*" && next_character === "/") {
        in_block_comment = false;
        i++;
      }
      continue;
    }

    if (in_single_quote) {
      if (character === "'" && previous_character !== "\\") {
        in_single_quote = false;
      }
      continue;
    }

    if (in_double_quote) {
      if (character === '"' && previous_character !== "\\") {
        in_double_quote = false;
      }
      continue;
    }

    if (in_template_string) {
      if (character === "`" && previous_character !== "\\") {
        in_template_string = false;
      }
      continue;
    }

    if (character === "/" && next_character === "/") {
      in_line_comment = true;
      i++;
      continue;
    }

    if (character === "/" && next_character === "*") {
      in_block_comment = true;
      i++;
      continue;
    }

    if (character === "'") {
      in_single_quote = true;
      continue;
    }

    if (character === '"') {
      in_double_quote = true;
      continue;
    }

    if (character === "`") {
      in_template_string = true;
      continue;
    }

    if (character === open_character) {
      depth++;
      continue;
    }

    if (character === close_character) {
      depth--;
      if (depth === 0) {
        return i;
      }
    }
  }

  return -1;
}

function extract_object_literal(source, open_index) {
  const close_index = find_matching_character(source, open_index, "{", "}");
  if (close_index === -1) {
    return null;
  }

  return {
    text: source.slice(open_index, close_index + 1),
    end_index: close_index,
  };
}

function split_top_level_values(source) {
  const values = [];
  let start = 0;
  let paren_depth = 0;
  let brace_depth = 0;
  let bracket_depth = 0;
  let in_single_quote = false;
  let in_double_quote = false;
  let in_template_string = false;
  let in_line_comment = false;
  let in_block_comment = false;

  for (let i = 0; i < source.length; i++) {
    const character = source[i];
    const next_character = source[i + 1];
    const previous_character = source[i - 1];

    if (in_line_comment) {
      if (character === "\n") {
        in_line_comment = false;
      }
      continue;
    }

    if (in_block_comment) {
      if (character === "*" && next_character === "/") {
        in_block_comment = false;
        i++;
      }
      continue;
    }

    if (in_single_quote) {
      if (character === "'" && previous_character !== "\\") {
        in_single_quote = false;
      }
      continue;
    }

    if (in_double_quote) {
      if (character === '"' && previous_character !== "\\") {
        in_double_quote = false;
      }
      continue;
    }

    if (in_template_string) {
      if (character === "`" && previous_character !== "\\") {
        in_template_string = false;
      }
      continue;
    }

    if (character === "/" && next_character === "/") {
      in_line_comment = true;
      i++;
      continue;
    }

    if (character === "/" && next_character === "*") {
      in_block_comment = true;
      i++;
      continue;
    }

    if (character === "'") {
      in_single_quote = true;
      continue;
    }

    if (character === '"') {
      in_double_quote = true;
      continue;
    }

    if (character === "`") {
      in_template_string = true;
      continue;
    }

    if (character === "(") {
      paren_depth++;
      continue;
    }

    if (character === ")") {
      paren_depth--;
      continue;
    }

    if (character === "{") {
      brace_depth++;
      continue;
    }

    if (character === "}") {
      brace_depth--;
      continue;
    }

    if (character === "[") {
      bracket_depth++;
      continue;
    }

    if (character === "]") {
      bracket_depth--;
      continue;
    }

    if (
      character === "," &&
      paren_depth === 0 &&
      brace_depth === 0 &&
      bracket_depth === 0
    ) {
      values.push(source.slice(start, i).trim());
      start = i + 1;
    }
  }

  const last_value = source.slice(start).trim();
  if (last_value.length > 0) {
    values.push(last_value);
  }

  return values;
}

function extract_call_arguments(source, call_regex) {
  const calls = [];
  const regex = new RegExp(call_regex);
  let match = regex.exec(source);
  while (match) {
    const open_index = source.indexOf("(", match.index);
    const close_index = find_matching_character(source, open_index, "(", ")");
    if (close_index === -1) {
      regex.lastIndex = match.index + match[0].length;
      match = regex.exec(source);
      continue;
    }

    calls.push({
      arguments: split_top_level_values(source.slice(open_index + 1, close_index)),
    });

    regex.lastIndex = close_index + 1;
    match = regex.exec(source);
  }
  return calls;
}

function parse_define_object_literal(object_literal) {
  const trimmed = object_literal.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
    return {};
  }

  const defines = {};
  const entries = split_top_level_values(trimmed.slice(1, -1));
  for (const entry of entries) {
    if (!entry || entry.startsWith("...")) {
      continue;
    }

    const separator_index = entry.indexOf(":");
    if (separator_index === -1) {
      continue;
    }

    const key = strip_wrapping_quotes(entry.slice(0, separator_index).trim());
    const value = parse_define_literal(entry.slice(separator_index + 1));
    if (!key || value === undefined) {
      continue;
    }

    defines[key] = value;
  }

  return defines;
}

function extract_shader_paths_from_text(source, const_shader_paths) {
  const shader_paths = new Set();
  const regex = new RegExp(object_shader_path_regex);
  let match = regex.exec(source);
  while (match) {
    const shader_path = resolve_shader_path_token(match[1], const_shader_paths);
    if (shader_path) {
      shader_paths.add(shader_path);
    }
    match = regex.exec(source);
  }
  return shader_paths;
}

function extract_define_families_from_setup_object(source) {
  const family = create_define_family();
  const regex = new RegExp(defines_object_regex);
  let match = regex.exec(source);
  while (match) {
    const open_index = match.index + match[0].length - 1;
    const object_literal = extract_object_literal(source, open_index);
    if (!object_literal) {
      break;
    }

    const defines = parse_define_object_literal(object_literal.text);
    for (const [key, value] of Object.entries(defines)) {
      add_define_domain_value(family.required_domains, key, value);
    }

    regex.lastIndex = object_literal.end_index + 1;
    match = regex.exec(source);
  }
  return family;
}

function discover_named_shader_contexts(source, const_shader_paths) {
  const contexts = new Map();
  const regex = new RegExp(setup_object_regex);
  let match = regex.exec(source);
  while (match) {
    const open_index = match.index + match[0].length - 1;
    const object_literal = extract_object_literal(source, open_index);
    if (!object_literal) {
      break;
    }

    const shader_paths = extract_shader_paths_from_text(object_literal.text, const_shader_paths);
    if (shader_paths.size > 0) {
      contexts.set(match[1], {
        shader_paths,
        family: extract_define_families_from_setup_object(object_literal.text),
      });
    }

    regex.lastIndex = object_literal.end_index + 1;
    match = regex.exec(source);
  }

  return contexts;
}

function apply_dynamic_setup_assignments(source, contexts) {
  const regex = new RegExp(setup_define_assignment_regex);
  let match = regex.exec(source);
  while (match) {
    const context = contexts.get(match[1]);
    if (!context) {
      match = regex.exec(source);
      continue;
    }

    for (const value of infer_define_values(match[3])) {
      add_define_domain_value(context.family.required_domains, match[2], value);
    }

    match = regex.exec(source);
  }
}

function discover_optional_define_map_domains(source) {
  const domains_by_variable = new Map();
  const regex = new RegExp(generic_define_assignment_regex);
  let match = regex.exec(source);
  while (match) {
    if (!domains_by_variable.has(match[1])) {
      domains_by_variable.set(match[1], new Map());
    }

    for (const value of infer_define_values(match[3])) {
      add_define_domain_value(domains_by_variable.get(match[1]), match[2], value);
    }

    match = regex.exec(source);
  }

  return domains_by_variable;
}

function parse_define_family_from_object_literal(object_literal, optional_domains_by_variable) {
  const family = create_define_family();
  const explicit_defines = parse_define_object_literal(object_literal);
  for (const [key, value] of Object.entries(explicit_defines)) {
    add_define_domain_value(family.required_domains, key, value);
  }

  const spread_regex = new RegExp(spread_identifier_regex);
  let match = spread_regex.exec(object_literal);
  while (match) {
    const optional_domains = optional_domains_by_variable.get(match[1]);
    if (optional_domains) {
      merge_define_domains(family.optional_domains, optional_domains);
    }
    match = spread_regex.exec(object_literal);
  }

  return family;
}

function discover_material_template_usages(source, const_shader_paths) {
  const usages = [];
  const used_shader_paths = new Set();
  const calls = extract_call_arguments(source, material_template_create_call_regex);
  for (const call of calls) {
    if (call.arguments.length < 2) {
      continue;
    }

    const shader_path = resolve_shader_path_token(call.arguments[1], const_shader_paths);
    if (!shader_path) {
      continue;
    }

    const family = create_define_family();
    if (call.arguments.length >= 6 && call.arguments[5].trim().startsWith("{")) {
      const explicit_defines = parse_define_object_literal(call.arguments[5]);
      for (const [key, value] of Object.entries(explicit_defines)) {
        add_define_domain_value(family.required_domains, key, value);
      }
    }

    usages.push({
      shader_path,
      family,
    });
    used_shader_paths.add(shader_path);
  }

  if (calls.length > 0) {
    const object_properties = extract_static_const_object_properties(source);
    for (const [property_name, property_value] of object_properties) {
      if (!/(?:^|\.)[A-Za-z0-9_]*shader$/i.test(property_name)) {
        continue;
      }

      const shader_path = resolve_shader_path_token(property_value, const_shader_paths);
      if (!shader_path || used_shader_paths.has(shader_path)) {
        continue;
      }

      usages.push({
        shader_path,
        family: create_define_family(),
      });
      used_shader_paths.add(shader_path);
    }
  }

  return usages;
}

function discover_shader_create_variants(
  source,
  const_shader_paths,
  optional_domains_by_variable
) {
  const direct_variants = new Map();
  const generic_variants = [];
  const calls = extract_call_arguments(source, shader_create_call_regex);
  const has_material_template_usage = source.includes("MaterialTemplate.create");

  for (const call of calls) {
    if (call.arguments.length < 2) {
      continue;
    }

    const defines_argument = call.arguments[1].trim();
    if (!defines_argument.startsWith("{")) {
      continue;
    }

    const family = parse_define_family_from_object_literal(
      defines_argument,
      optional_domains_by_variable
    );
    const shader_path = resolve_shader_path_token(call.arguments[0], const_shader_paths);
    if (shader_path) {
      if (!direct_variants.has(shader_path)) {
        direct_variants.set(shader_path, []);
      }
      direct_variants.get(shader_path).push(family);
      continue;
    }

    if (has_material_template_usage) {
      generic_variants.push(family);
    }
  }

  return {
    direct_variants,
    generic_variants,
  };
}

function list_all_source_files(source_roots) {
  const files = [];
  const seen_files = new Set();

  for (const source_root of source_roots) {
    const source_root_files = list_source_files(source_root.source_root_path);
    for (const source_file of source_root_files) {
      if (seen_files.has(source_file)) {
        continue;
      }

      seen_files.add(source_file);
      files.push(source_file);
    }
  }

  return files.sort((a, b) => a.localeCompare(b));
}

function discover_requested_shader_variants(source_roots, referenced_shader_paths) {
  const variants_by_path = new Map();
  const referenced_shader_path_set = new Set(referenced_shader_paths);
  const material_shader_paths = new Set();
  const generic_material_families = [];
  const source_files = list_all_source_files(source_roots);

  for (const source_file of source_files) {
    const source = fs.readFileSync(source_file, "utf8");
    const const_shader_paths = build_const_shader_path_map(source);
    const shader_contexts = discover_named_shader_contexts(source, const_shader_paths);
    apply_dynamic_setup_assignments(source, shader_contexts);

    for (const context of shader_contexts.values()) {
      const shader_paths = [...context.shader_paths].filter((shader_path) =>
        referenced_shader_path_set.has(shader_path)
      );
      if (shader_paths.length > 0) {
        add_variant_family_to_paths(variants_by_path, shader_paths, context.family, false);
      }
    }

    for (const usage of discover_material_template_usages(source, const_shader_paths)) {
      if (!referenced_shader_path_set.has(usage.shader_path)) {
        continue;
      }

      material_shader_paths.add(usage.shader_path);
      add_variant_family_to_paths(variants_by_path, [usage.shader_path], usage.family, true);
    }

    const optional_domains_by_variable = discover_optional_define_map_domains(source);
    const { direct_variants, generic_variants } = discover_shader_create_variants(
      source,
      const_shader_paths,
      optional_domains_by_variable
    );

    for (const [shader_path, families] of direct_variants) {
      if (!referenced_shader_path_set.has(shader_path)) {
        continue;
      }

      for (const family of families) {
        add_variant_family_to_paths(variants_by_path, [shader_path], family, true);
      }
    }

    generic_material_families.push(...generic_variants);
  }

  for (const shader_path of material_shader_paths) {
    for (const family of generic_material_families) {
      add_variant_family_to_paths(variants_by_path, [shader_path], family, true);
    }
  }

  return variants_by_path;
}

function analyze_shader_define_usage(
  shader_root,
  include_resolution_roots,
  shader_path,
  cache = new Map(),
  active_paths = new Set()
) {
  if (cache.has(shader_path)) {
    return cache.get(shader_path);
  }

  if (active_paths.has(shader_path)) {
    return new Set();
  }

  active_paths.add(shader_path);
  const shader_source = read_shader_source(shader_root, include_resolution_roots, shader_path);
  if (!shader_source) {
    active_paths.delete(shader_path);
    return new Set();
  }

  const define_keys = new Set();
  const conditional_regex = new RegExp(shader_conditional_regex);
  let match = conditional_regex.exec(shader_source.source);
  while (match) {
    if (!built_in_shader_define_keys.has(match[1])) {
      define_keys.add(match[1]);
    }
    match = conditional_regex.exec(shader_source.source);
  }

  const declaration_regex = new RegExp(shader_define_declaration_regex);
  match = declaration_regex.exec(shader_source.source);
  while (match) {
    if (!built_in_shader_define_keys.has(match[1])) {
      define_keys.add(match[1]);
    }
    match = declaration_regex.exec(shader_source.source);
  }

  const include_matches = shader_source.source.matchAll(/^#include\s+"(\S+)".*$/gm);
  for (const include_match of include_matches) {
    const include_define_keys = analyze_shader_define_usage(
      shader_root,
      include_resolution_roots,
      include_match[1],
      cache,
      active_paths
    );
    for (const define_key of include_define_keys) {
      if (!built_in_shader_define_keys.has(define_key)) {
        define_keys.add(define_key);
      }
    }
  }

  active_paths.delete(shader_path);
  cache.set(shader_path, define_keys);
  return define_keys;
}

function filter_variant_defines(defines, active_define_keys) {
  if (!active_define_keys || active_define_keys.size === 0) {
    return {};
  }

  const filtered_defines = {};
  for (const [key, value] of Object.entries(defines)) {
    if (active_define_keys.has(key)) {
      filtered_defines[key] = value;
    }
  }

  return filtered_defines;
}

function find_shader_files(shader_root_path) {
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
        shader_files.push(path.relative(shader_root_path, full_path).replace(/\\/g, "/"));
      }
    }
  }

  walk(shader_root_path);
  return sort_shader_paths(shader_files);
}

function discover_referenced_shader_paths(source_roots, available_shader_paths) {
  const available = new Set(available_shader_paths);
  const referenced = new Set();
  const visited = new Set();

  function walk(current_dir) {
    if (visited.has(current_dir)) {
      return;
    }
    visited.add(current_dir);

    const entries = fs.readdirSync(current_dir, { withFileTypes: true });
    for (const entry of entries) {
      const full_path = path.join(current_dir, entry.name);
      if (entry.isDirectory()) {
        walk(full_path);
        continue;
      }

      if (!entry.isFile() || !source_file_extensions.has(path.extname(entry.name).toLowerCase())) {
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

  for (const source_root of source_roots) {
    walk(source_root.source_root_path);
  }

  return sort_shader_paths(referenced);
}

function build_shader_variant_registry(
  shader_root,
  source_roots,
  shader_paths,
  include_resolution_roots
) {
  const variants = new Map();
  const requested_variants = discover_requested_shader_variants(source_roots, shader_paths);
  const shader_define_usage_cache = new Map();

  for (const shader_path of shader_paths) {
    add_variant(variants, shader_path, {});
    const active_define_keys = analyze_shader_define_usage(
      shader_root,
      include_resolution_roots,
      shader_path,
      shader_define_usage_cache
    );
    const path_variants = requested_variants.get(shader_path);
    if (!path_variants) {
      continue;
    }

    for (const requested_variant of path_variants.values()) {
      const filtered_defines = requested_variant.preserve_requested_defines
        ? requested_variant.defines
        : filter_variant_defines(requested_variant.defines, active_define_keys);
      if (Object.keys(filtered_defines).length === 0) {
        continue;
      }

      add_variant(variants, shader_path, filtered_defines);
    }
  }

  return [...variants.values()].sort((a, b) => {
    const path_diff = a.path.localeCompare(b.path);
    if (path_diff !== 0) {
      return path_diff;
    }
    return JSON.stringify(a.defines).localeCompare(JSON.stringify(b.defines));
  });
}

function validate_shader_registry(shader_root, shader_paths, referenced_shader_paths, variants) {
  const shader_path_set = new Set(shader_paths);
  const variant_path_set = new Set(variants.map((variant) => variant.path));

  for (const referenced_path of referenced_shader_paths) {
    if (!shader_path_set.has(referenced_path)) {
      throw new Error(
        `Referenced shader '${referenced_path}' does not exist under assets/${shader_root.asset_path}.`
      );
    }

    if (!variant_path_set.has(referenced_path)) {
      throw new Error(
        `Referenced shader '${referenced_path}' is missing from the cooked shader registry.`
      );
    }
  }
}

function create_cook_settings(shader_root, source_roots) {
  return {
    version: GENERATOR_VERSION,
    shader_root_asset_path: shader_root.asset_path,
    manifest_asset_path: shader_root.manifest_asset_path,
    binary_asset_path: shader_root.binary_asset_path,
    binary_asset_name: shader_archive_binary_name,
    compression: compression_mode,
    precision_profiles: [ShaderPrecisionProfile.F16, ShaderPrecisionProfile.F32],
    source_roots: source_roots.map((source_root) => source_root.source_root),
  };
}

function create_source_hash(settings, shader_root, shader_files, variants, include_resolution_roots) {
  const hash = crypto.createHash("sha1");
  hash.update(JSON.stringify(settings));

  const hashed_shader_roots = new Set();
  for (const include_root of include_resolution_roots) {
    if (hashed_shader_roots.has(include_root.asset_path)) {
      continue;
    }

    hashed_shader_roots.add(include_root.asset_path);
    const include_root_shader_files = find_shader_files(include_root.shader_root_path);
    hash.update(include_root.asset_path);
    for (const shader_path of include_root_shader_files) {
      hash.update(shader_path);
      hash.update(fs.readFileSync(path.join(include_root.shader_root_path, shader_path), "utf8"));
    }
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

function read_shader_source(shader_root, include_resolution_roots, file_path) {
  for (const include_root of include_resolution_roots) {
    const absolute_path = path.join(include_root.shader_root_path, file_path);
    if (!fs.existsSync(absolute_path)) {
      continue;
    }

    return {
      source: fs.readFileSync(absolute_path, "utf8"),
      resolved_root: include_root,
    };
  }

  return null;
}

function parse_shader_includes(
  shader_root,
  include_resolution_roots,
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
        shader_root,
        include_resolution_roots,
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

function load_shader_text(
  shader_root,
  include_resolution_roots,
  file_path,
  defines = {},
  load_recursion_step = 0,
  precision_profile
) {
  const shader_source = read_shader_source(shader_root, include_resolution_roots, file_path);
  if (!shader_source) {
    const searched_roots = include_resolution_roots
      .map((include_root) => `assets/${include_root.asset_path}`)
      .join(", ");
    throw new Error(
      `Could not find shader '${file_path}' while cooking shader root '${shader_root.asset_path}'. Searched: ${searched_roots}.`
    );
  }
  let asset = shader_source.source;

  asset = parse_shader_includes(
    shader_root,
    include_resolution_roots,
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
    bind_groups: normalize_bind_groups(reflection),
    entry: {
      vertex: normalize_entry_stage(reflection.entry?.vertex ?? []),
      fragment: normalize_entry_stage(reflection.entry?.fragment ?? []),
      compute: normalize_entry_stage(reflection.entry?.compute ?? []),
    },
  };
}

function build_output(settings, shader_root, shader_files, variants, include_resolution_roots) {
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

      const code = load_shader_text(
        shader_root,
        include_resolution_roots,
        variant.path,
        canonical_defines,
        0,
        precision_profile
      );
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
    shaderRootAssetPath: shader_root.asset_path,
    manifestAssetPath: shader_root.manifest_asset_path,
    binaryAssetPath: shader_root.binary_asset_path,
    binaryAssetName: shader_archive_binary_name,
    settings,
    source: {
      hash: create_source_hash(
        settings,
        shader_root,
        shader_files,
        variants,
        include_resolution_roots
      ),
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
  const { shader_roots, source_roots } = load_shader_roots();

  for (const shader_root of shader_roots) {
    const settings = create_cook_settings(shader_root, source_roots);
    const shader_files = find_shader_files(shader_root.shader_root_path);
    const referenced_shader_paths = discover_referenced_shader_paths(source_roots, shader_files);
    const include_resolution_roots = build_include_resolution_roots(shader_root, shader_roots);
    const variants = build_shader_variant_registry(
      shader_root,
      source_roots,
      referenced_shader_paths,
      include_resolution_roots
    );
    const source_hash = create_source_hash(
      settings,
      shader_root,
      shader_files,
      variants,
      include_resolution_roots
    );
    const manifest_output_path = resolve_asset_path(shader_root.manifest_asset_path);
    const binary_output_path = resolve_asset_path(shader_root.binary_asset_path);

    validate_shader_registry(shader_root, shader_files, referenced_shader_paths, variants);

    if (should_skip_generation(manifest_output_path, binary_output_path, source_hash)) {
      console.log(
        `[shader_preprocessor] up to date: ${path.relative(PROJECT_ROOT, manifest_output_path).replace(/\\/g, "/")}`
      );
      continue;
    }

    const output = build_output(
      settings,
      shader_root,
      shader_files,
      variants,
      include_resolution_roots
    );

    fs.mkdirSync(path.dirname(manifest_output_path), { recursive: true });
    fs.writeFileSync(manifest_output_path, JSON.stringify(output.manifest, null, 2));
    fs.mkdirSync(path.dirname(binary_output_path), { recursive: true });
    fs.writeFileSync(binary_output_path, output.binary);

    console.log(
      `[shader_preprocessor] generated ${path.relative(PROJECT_ROOT, manifest_output_path).replace(/\\/g, "/")} (${output.manifest.source.variantCount} variants)`
    );
    console.log(
      `[shader_preprocessor] generated ${path.relative(PROJECT_ROOT, binary_output_path).replace(/\\/g, "/")} (${output.binary.byteLength} bytes)`
    );
  }
}

main();

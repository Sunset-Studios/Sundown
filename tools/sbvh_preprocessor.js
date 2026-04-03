import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  SBVH_BIN_COUNT,
  SBVH_MAX_REFERENCE_MULTIPLIER,
  SBVH_MAX_SPATIAL_DEPTH,
  build_sbvh_from_positions_indices,
} from "../engine/src/acceleration/sbvh_builder.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ASSET_ROOT = path.resolve(__dirname, "../assets");
const GENERATOR_VERSION = 1;
const NODE_STRIDE = 8 * Float32Array.BYTES_PER_ELEMENT;

const COMPONENT_TYPE_BYTE_SIZE = {
  5120: 1,
  5121: 1,
  5122: 2,
  5123: 2,
  5125: 4,
  5126: 4,
};

const ACCESSOR_COMPONENT_COUNT = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};

function find_gltf_files(dir, gltf_files) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const full_path = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      find_gltf_files(full_path, gltf_files);
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".gltf")) {
      gltf_files.push(full_path);
    }
  }
}

function decode_data_uri(uri) {
  const match = uri.match(/^data:.*?(;base64)?,(.*)$/);
  if (!match) {
    throw new Error(`Unsupported data URI: ${uri.slice(0, 64)}`);
  }

  const is_base64 = Boolean(match[1]);
  const payload = match[2];
  return is_base64
    ? Buffer.from(payload, "base64")
    : Buffer.from(decodeURIComponent(payload), "utf8");
}

function load_referenced_buffer(base_dir, uri) {
  if (!uri) {
    throw new Error("glTF buffer is missing a uri.");
  }

  if (uri.startsWith("data:")) {
    return decode_data_uri(uri);
  }

  return fs.readFileSync(path.resolve(base_dir, uri));
}

function create_source_hash(settings, gltf_text, referenced_buffers) {
  const hash = crypto.createHash("sha1");
  hash.update(JSON.stringify(settings));
  hash.update(gltf_text);

  for (const buffer of referenced_buffers) {
    hash.update(buffer);
  }

  return hash.digest("hex");
}

function get_accessor_component_count(accessor) {
  const count = ACCESSOR_COMPONENT_COUNT[accessor.type];
  if (!count) {
    throw new Error(`Unsupported accessor type "${accessor.type}".`);
  }
  return count;
}

function read_numeric_component(view, byte_offset, component_type) {
  switch (component_type) {
    case 5120:
      return view.getInt8(byte_offset);
    case 5121:
      return view.getUint8(byte_offset);
    case 5122:
      return view.getInt16(byte_offset, true);
    case 5123:
      return view.getUint16(byte_offset, true);
    case 5125:
      return view.getUint32(byte_offset, true);
    case 5126:
      return view.getFloat32(byte_offset, true);
    default:
      throw new Error(`Unsupported component type ${component_type}.`);
  }
}

function normalize_component(value, component_type) {
  switch (component_type) {
    case 5120:
      return Math.max(value / 127.0, -1.0);
    case 5121:
      return value / 255.0;
    case 5122:
      return Math.max(value / 32767.0, -1.0);
    case 5123:
      return value / 65535.0;
    default:
      return value;
  }
}

function read_accessor_to_float32(document, accessor_index) {
  const accessor = document.accessors?.[accessor_index];
  if (!accessor) {
    throw new Error(`Missing accessor ${accessor_index}.`);
  }

  const buffer_view = document.bufferViews?.[accessor.bufferView];
  if (!buffer_view) {
    throw new Error(`Accessor ${accessor_index} is missing bufferView data.`);
  }

  const source_buffer = document.buffers[buffer_view.buffer];
  const component_count = get_accessor_component_count(accessor);
  const component_size = COMPONENT_TYPE_BYTE_SIZE[accessor.componentType];
  const element_size = component_count * component_size;
  const accessor_offset = accessor.byteOffset || 0;
  const buffer_view_offset = buffer_view.byteOffset || 0;
  const stride = buffer_view.byteStride || element_size;
  const base_offset = buffer_view_offset + accessor_offset;

  const out = new Float32Array(accessor.count * component_count);
  const view = new DataView(source_buffer.buffer, source_buffer.byteOffset, source_buffer.byteLength);

  for (let i = 0; i < accessor.count; i++) {
    const element_offset = base_offset + i * stride;
    for (let c = 0; c < component_count; c++) {
      const component_offset = element_offset + c * component_size;
      let value = read_numeric_component(view, component_offset, accessor.componentType);
      if (accessor.normalized) {
        value = normalize_component(value, accessor.componentType);
      }
      out[i * component_count + c] = value;
    }
  }

  return out;
}

function read_accessor_to_indices(document, accessor_index) {
  const accessor = document.accessors?.[accessor_index];
  if (!accessor) {
    throw new Error(`Missing accessor ${accessor_index}.`);
  }

  if (accessor.type !== "SCALAR") {
    throw new Error(`Index accessor ${accessor_index} must be SCALAR.`);
  }

  const raw = read_accessor_to_float32(document, accessor_index);
  const out = new Uint32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    out[i] = raw[i];
  }
  return out;
}

function load_gltf_document(gltf_path, settings) {
  const gltf_text = fs.readFileSync(gltf_path, "utf8");
  const json = JSON.parse(gltf_text);
  const base_dir = path.dirname(gltf_path);
  const buffers = [];
  const buffer_uris = [];

  for (const buffer of json.buffers || []) {
    const loaded = load_referenced_buffer(base_dir, buffer.uri);
    buffers.push(loaded);
    buffer_uris.push(buffer.uri);
  }

  return {
    path: gltf_path,
    accessors: json.accessors || [],
    bufferViews: json.bufferViews || [],
    buffers,
    buffer_uris,
    json,
    source_hash: create_source_hash(settings, gltf_text, buffers),
  };
}

function read_positions_and_indices(document, primitive) {
  if (primitive.attributes?.POSITION === undefined) {
    return null;
  }

  const positions = read_accessor_to_float32(document, primitive.attributes.POSITION);
  if (positions.length === 0) {
    return null;
  }

  let indices = null;
  if (primitive.indices !== undefined) {
    indices = read_accessor_to_indices(document, primitive.indices);
  } else {
    indices = new Uint32Array(positions.length / 3);
    for (let i = 0; i < indices.length; i++) {
      indices[i] = i;
    }
  }

  const trimmed_index_count = indices.length - (indices.length % 3);
  if (trimmed_index_count <= 0) {
    return null;
  }

  return {
    positions,
    indices: indices.slice(0, trimmed_index_count),
  };
}

function append_positions(target, source) {
  for (let i = 0; i < source.length; i++) {
    target.push(source[i]);
  }
}

function append_rebased_indices(target, source, vertex_base) {
  for (let i = 0; i < source.length; i++) {
    target.push(source[i] + vertex_base);
  }
}

function build_mesh_payload(document, mesh_index, mesh) {
  const grouped_indices = new Map();
  const position_values = [];

  for (let primitive_index = 0; primitive_index < mesh.primitives.length; primitive_index++) {
    const primitive = mesh.primitives[primitive_index];
    if ((primitive.mode ?? 4) !== 4) {
      continue;
    }

    const source = read_positions_and_indices(document, primitive);
    if (!source) {
      continue;
    }

    const material_index =
      primitive.material === undefined || primitive.material === null ? -1 : primitive.material;
    let group_indices = grouped_indices.get(material_index);
    if (!group_indices) {
      group_indices = [];
      grouped_indices.set(material_index, group_indices);
    }

    const vertex_base = position_values.length / 3;
    append_positions(position_values, source.positions);
    append_rebased_indices(group_indices, source.indices, vertex_base);
  }

  if (position_values.length === 0 || grouped_indices.size === 0) {
    return null;
  }

  const final_indices = [];
  for (const group_indices of grouped_indices.values()) {
    for (let i = 0; i < group_indices.length; i++) {
      final_indices.push(group_indices[i]);
    }
  }

  if (final_indices.length < 3) {
    return null;
  }

  const sbvh = build_sbvh_from_positions_indices(
    Float32Array.from(position_values),
    Uint32Array.from(final_indices)
  );

  if (!sbvh) {
    return null;
  }

  return {
    mesh: mesh_index,
    name: mesh.name || null,
    primitive_count: sbvh.primitive_count,
    reference_count: sbvh.reference_count,
    node_count: sbvh.node_count,
    node_data: sbvh.node_data,
  };
}

function build_output_payload(document, settings) {
  const manifest = {
    version: GENERATOR_VERSION,
    generator: "tools/sbvh_preprocessor.js",
    generatedAt: new Date().toISOString(),
    settings: {
      binCount: SBVH_BIN_COUNT,
      maxReferenceMultiplier: SBVH_MAX_REFERENCE_MULTIPLIER,
      maxSpatialDepth: SBVH_MAX_SPATIAL_DEPTH,
    },
    source: {
      gltf: path.relative(path.dirname(document.path), document.path).replace(/\\/g, "/"),
      buffers: document.buffer_uris.slice(),
      hash: document.source_hash,
    },
    meshes: [],
    sections: {},
  };

  const node_payloads = [];
  for (let mesh_index = 0; mesh_index < (document.json.meshes || []).length; mesh_index++) {
    const mesh = document.json.meshes[mesh_index];
    const payload = build_mesh_payload(document, mesh_index, mesh);
    if (!payload) {
      manifest.meshes.push({
        mesh: mesh_index,
        name: mesh.name || null,
        skipped: true,
      });
      continue;
    }

    manifest.meshes.push({
      mesh: mesh_index,
      name: payload.name,
      skipped: false,
      primitiveCount: payload.primitive_count,
      referenceCount: payload.reference_count,
      nodeCount: payload.node_count,
      nodeOffset: node_payloads.length / 8,
    });
    for (let i = 0; i < payload.node_data.length; i++) {
      node_payloads.push(payload.node_data[i]);
    }
  }

  const nodes_offset = 0;
  const nodes_byte_length = node_payloads.length * Float32Array.BYTES_PER_ELEMENT;
  const binary = new ArrayBuffer(nodes_byte_length);
  const node_array = new Float32Array(binary);
  node_array.set(node_payloads);

  manifest.sections = {
    nodes: {
      offset: nodes_offset,
      stride: NODE_STRIDE,
      count: node_payloads.length / 8,
      elementType: "float32x8",
    },
  };

  return {
    manifest,
    binary: Buffer.from(binary),
  };
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

function process_gltf_file(gltf_path, settings) {
  const document = load_gltf_document(gltf_path, settings);
  const manifest_path = gltf_path.replace(/\.gltf$/i, ".sbvh.json");
  const binary_path = gltf_path.replace(/\.gltf$/i, ".sbvh.bin");

  if (should_skip_generation(manifest_path, binary_path, document.source_hash)) {
    return {
      status: "skipped",
      gltf_path,
    };
  }

  const output = build_output_payload(document, settings);
  output.manifest.binary = path.basename(binary_path);

  fs.writeFileSync(manifest_path, JSON.stringify(output.manifest, null, 2));
  fs.writeFileSync(binary_path, output.binary);

  let node_count = 0;
  let mesh_count = 0;
  for (const mesh of output.manifest.meshes) {
    if (mesh.skipped) {
      continue;
    }
    mesh_count++;
    node_count += mesh.nodeCount;
  }

  return {
    status: "generated",
    gltf_path,
    mesh_count,
    node_count,
  };
}

function format_asset_path(file_path) {
  return path.relative(ASSET_ROOT, file_path).replace(/\\/g, "/");
}

async function main() {
  const settings = {
    version: GENERATOR_VERSION,
    bin_count: SBVH_BIN_COUNT,
    max_reference_multiplier: SBVH_MAX_REFERENCE_MULTIPLIER,
    max_spatial_depth: SBVH_MAX_SPATIAL_DEPTH,
  };

  const gltf_files = [];
  find_gltf_files(ASSET_ROOT, gltf_files);
  gltf_files.sort((a, b) => a.localeCompare(b));

  const summary = {
    generated: 0,
    skipped: 0,
    failed: 0,
    meshes: 0,
    nodes: 0,
  };

  for (const gltf_path of gltf_files) {
    try {
      const result = process_gltf_file(gltf_path, settings);
      if (result.status === "skipped") {
        summary.skipped++;
        console.log(`[sbvh_preprocessor] up to date: ${format_asset_path(gltf_path)}`);
        continue;
      }

      summary.generated++;
      summary.meshes += result.mesh_count;
      summary.nodes += result.node_count;
      console.log(
        `[sbvh_preprocessor] generated ${format_asset_path(gltf_path)} (${result.mesh_count} meshes, ${result.node_count} nodes)`
      );
    } catch (error) {
      summary.failed++;
      console.error(`[sbvh_preprocessor] failed ${format_asset_path(gltf_path)}:`, error);
    }
  }

  console.log(
    `[sbvh_preprocessor] complete: ${summary.generated} generated, ${summary.skipped} skipped, ${summary.failed} failed, ${summary.meshes} meshes, ${summary.nodes} nodes`
  );

  if (summary.failed > 0) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("[sbvh_preprocessor] fatal:", error);
  process.exit(1);
});

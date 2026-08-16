import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { MeshoptClusterizer } from "meshoptimizer/clusterizer";
import {
  build_mesh_lod_indices,
  initialize_mesh_lod_simplifier,
  resolve_mesh_lod_settings,
} from "./mesh_lod_utils.js";
import {
  build_meshlet_groups,
  build_meshlets,
  compute_position_bounds,
  sort_meshlets_spatially,
} from "../engine/src/renderer/meshlet_utils.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ASSET_ROOT = path.resolve(__dirname, "../assets");

const GENERATOR_VERSION = 4;
const DEFAULT_MAX_VERTICES = 64;
const DEFAULT_MIN_TRIANGLES = 24;
const DEFAULT_MAX_TRIANGLES = 124;
const DEFAULT_FILL_WEIGHT = 0.5;
const DEFAULT_CLUSTER_GROUP_SIZE = 8;

const MESHLET_STRUCT_STRIDE = 80;
const MESHLET_GROUP_STRUCT_STRIDE = 64;

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

function align_to(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

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
  const view = new DataView(
    source_buffer.buffer,
    source_buffer.byteOffset,
    source_buffer.byteLength
  );

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

function process_primitive(document, primitive_index, primitive, settings, lod_settings) {
  if ((primitive.mode ?? 4) !== 4) {
    return {
      primitive: primitive_index,
      mode: primitive.mode ?? 4,
      skipped: true,
      reason: "Only triangle-list primitives are supported.",
    };
  }

  if (primitive.attributes?.POSITION === undefined) {
    return {
      primitive: primitive_index,
      mode: primitive.mode ?? 4,
      skipped: true,
      reason: "Primitive is missing POSITION data.",
    };
  }

  const positions = read_accessor_to_float32(document, primitive.attributes.POSITION);
  if (positions.length === 0) {
    return {
      primitive: primitive_index,
      mode: primitive.mode ?? 4,
      skipped: true,
      reason: "Primitive has no vertex positions.",
    };
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
  if (trimmed_index_count !== indices.length) {
    console.warn(
      `[meshlet_preprocessor] Trimming ${indices.length - trimmed_index_count} dangling indices from primitive ${primitive_index}.`
    );
    indices = indices.slice(0, trimmed_index_count);
  }

  if (indices.length === 0) {
    return {
      primitive: primitive_index,
      mode: primitive.mode ?? 4,
      skipped: true,
      reason: "Primitive has no triangles.",
    };
  }

  const primitive_bounds = compute_position_bounds(positions);
  const lods = build_mesh_lod_indices(indices, positions, lod_settings).map((lod_data) => {
    const meshlets = sort_meshlets_spatially(
      build_meshlets(lod_data.indices, positions, settings),
      primitive_bounds
    );
    const meshlet_groups = build_meshlet_groups(meshlets, settings.cluster_group_size);
    let meshlet_vertex_count = 0;
    let meshlet_triangle_index_count = 0;
    for (const meshlet of meshlets) {
      meshlet_vertex_count += meshlet.global_vertices.length;
      meshlet_triangle_index_count += meshlet.local_indices.length;
    }

    return {
      ...lod_data,
      meshlets,
      meshlet_groups,
      meshlet_vertex_count,
      meshlet_triangle_index_count,
    };
  });

  return {
    primitive: primitive_index,
    mode: primitive.mode ?? 4,
    material: primitive.material ?? null,
    skipped: false,
    vertex_count: positions.length / 3,
    triangle_count: indices.length / 3,
    bounds: primitive_bounds,
    lods,
  };
}

function build_output_payload(document, settings) {
  const manifest = {
    version: GENERATOR_VERSION,
    generator: "tools/meshlet_preprocessor.js",
    generatedAt: new Date().toISOString(),
    settings: {
      maxVertices: settings.max_vertices,
      minTriangles: settings.min_triangles,
      maxTriangles: settings.max_triangles,
      fillWeight: settings.fill_weight,
      clusterGroupSize: settings.cluster_group_size,
    },
    source: {
      gltf: path.relative(path.dirname(document.path), document.path).replace(/\\/g, "/"),
      buffers: document.buffer_uris.slice(),
      hash: document.source_hash,
    },
    meshes: [],
    sections: {},
  };

  const meshlet_records = [];
  const meshlet_vertex_records = [];
  const meshlet_triangle_records = [];
  const meshlet_group_records = [];

  for (let mesh_index = 0; mesh_index < (document.json.meshes || []).length; mesh_index++) {
    const mesh = document.json.meshes[mesh_index];
    const lod_settings = resolve_mesh_lod_settings(document.json, mesh);
    const manifest_mesh = {
      mesh: mesh_index,
      name: mesh.name || null,
      lodCount: lod_settings.ratios.length,
      defaultMinLod: lod_settings.has_min_lod ? lod_settings.min_lod : null,
      lodRatios: lod_settings.ratios,
      primitives: [],
    };

    for (let primitive_index = 0; primitive_index < mesh.primitives.length; primitive_index++) {
      const primitive = mesh.primitives[primitive_index];
      const processed = process_primitive(
        document,
        primitive_index,
        primitive,
        settings,
        lod_settings
      );

      if (processed.skipped) {
        manifest_mesh.primitives.push(processed);
        continue;
      }

      const primitive_manifest = {
        primitive: primitive_index,
        mode: processed.mode,
        material: processed.material,
        skipped: false,
        vertexCount: processed.vertex_count,
        triangleCount: processed.triangle_count,
        bounds: processed.bounds,
        lods: [],
      };

      for (const processed_lod of processed.lods) {
        const primitive_meshlet_offset = meshlet_records.length;
        const primitive_vertex_offset = meshlet_vertex_records.length;
        const primitive_triangle_offset = meshlet_triangle_records.length;
        const primitive_group_offset = meshlet_group_records.length;

        for (const meshlet of processed_lod.meshlets) {
          meshlet_records.push({
            vertex_offset: meshlet_vertex_records.length,
            vertex_count: meshlet.global_vertices.length,
            triangle_offset: meshlet_triangle_records.length,
            triangle_count: meshlet.local_indices.length / 3,
            center: meshlet.center,
            radius: meshlet.radius,
            bounds_min: meshlet.bounds_min,
            bounds_max: meshlet.bounds_max,
            normal_cone_axis: meshlet.normal_cone_axis,
            normal_cone_cutoff: meshlet.normal_cone_cutoff,
          });
          meshlet_vertex_records.push(...meshlet.global_vertices);
          meshlet_triangle_records.push(...meshlet.local_indices);
        }

        for (const group of processed_lod.meshlet_groups) {
          meshlet_group_records.push({
            meshlet_offset: primitive_meshlet_offset + group.local_meshlet_offset,
            meshlet_count: group.meshlet_count,
            center: group.center,
            radius: group.radius,
            bounds_min: group.bounds_min,
            bounds_max: group.bounds_max,
          });
        }

        primitive_manifest.lods.push({
          lod: processed_lod.lod,
          targetRatio: processed_lod.target_ratio,
          actualRatio: processed_lod.actual_ratio,
          simplificationError: processed_lod.error,
          triangleCount: processed_lod.indices.length / 3,
          meshletCount: processed_lod.meshlets.length,
          meshletVertexCount: processed_lod.meshlet_vertex_count,
          meshletTriangleIndexCount: processed_lod.meshlet_triangle_index_count,
          meshletOffset: primitive_meshlet_offset,
          meshletVertexOffset: primitive_vertex_offset,
          meshletTriangleIndexOffset: primitive_triangle_offset,
          meshletGroupOffset: primitive_group_offset,
          meshletGroupCount: processed_lod.meshlet_groups.length,
        });
      }

      // Preserve the v2 LOD0 fields so older runtimes can still render freshly cooked assets.
      Object.assign(primitive_manifest, primitive_manifest.lods[0]);
      manifest_mesh.primitives.push(primitive_manifest);
    }

    manifest.meshes.push(manifest_mesh);
  }

  const meshlets_byte_length = meshlet_records.length * MESHLET_STRUCT_STRIDE;
  const meshlet_vertices_byte_length =
    meshlet_vertex_records.length * Uint32Array.BYTES_PER_ELEMENT;
  const meshlet_triangles_byte_length = meshlet_triangle_records.length;
  const meshlet_groups_byte_length = meshlet_group_records.length * MESHLET_GROUP_STRUCT_STRIDE;

  const meshlets_offset = 0;
  const meshlet_vertices_offset = align_to(meshlets_offset + meshlets_byte_length, 16);
  const meshlet_triangles_offset = align_to(
    meshlet_vertices_offset + meshlet_vertices_byte_length,
    16
  );
  const meshlet_groups_offset = align_to(
    meshlet_triangles_offset + meshlet_triangles_byte_length,
    16
  );
  const total_byte_length = meshlet_groups_offset + meshlet_groups_byte_length;

  const binary = new ArrayBuffer(total_byte_length);
  const view = new DataView(binary);
  const meshlet_vertex_array = new Uint32Array(
    binary,
    meshlet_vertices_offset,
    meshlet_vertex_records.length
  );
  const meshlet_triangle_array = new Uint8Array(
    binary,
    meshlet_triangles_offset,
    meshlet_triangle_records.length
  );

  meshlet_vertex_array.set(meshlet_vertex_records);
  meshlet_triangle_array.set(meshlet_triangle_records);

  for (let i = 0; i < meshlet_records.length; i++) {
    const base = meshlets_offset + i * MESHLET_STRUCT_STRIDE;
    const meshlet = meshlet_records[i];

    view.setUint32(base + 0, meshlet.vertex_offset, true);
    view.setUint32(base + 4, meshlet.vertex_count, true);
    view.setUint32(base + 8, meshlet.triangle_offset, true);
    view.setUint32(base + 12, meshlet.triangle_count, true);

    view.setFloat32(base + 16, meshlet.center[0], true);
    view.setFloat32(base + 20, meshlet.center[1], true);
    view.setFloat32(base + 24, meshlet.center[2], true);
    view.setFloat32(base + 28, meshlet.radius, true);

    view.setFloat32(base + 32, meshlet.bounds_min[0], true);
    view.setFloat32(base + 36, meshlet.bounds_min[1], true);
    view.setFloat32(base + 40, meshlet.bounds_min[2], true);
    view.setFloat32(base + 44, 0.0, true);

    view.setFloat32(base + 48, meshlet.bounds_max[0], true);
    view.setFloat32(base + 52, meshlet.bounds_max[1], true);
    view.setFloat32(base + 56, meshlet.bounds_max[2], true);
    view.setFloat32(base + 60, 0.0, true);

    view.setFloat32(base + 64, meshlet.normal_cone_axis[0], true);
    view.setFloat32(base + 68, meshlet.normal_cone_axis[1], true);
    view.setFloat32(base + 72, meshlet.normal_cone_axis[2], true);
    view.setFloat32(base + 76, meshlet.normal_cone_cutoff, true);
  }

  for (let i = 0; i < meshlet_group_records.length; i++) {
    const base = meshlet_groups_offset + i * MESHLET_GROUP_STRUCT_STRIDE;
    const group = meshlet_group_records[i];

    view.setUint32(base + 0, group.meshlet_offset, true);
    view.setUint32(base + 4, group.meshlet_count, true);
    view.setUint32(base + 8, 0, true);
    view.setUint32(base + 12, 0, true);

    view.setFloat32(base + 16, group.center[0], true);
    view.setFloat32(base + 20, group.center[1], true);
    view.setFloat32(base + 24, group.center[2], true);
    view.setFloat32(base + 28, group.radius, true);

    view.setFloat32(base + 32, group.bounds_min[0], true);
    view.setFloat32(base + 36, group.bounds_min[1], true);
    view.setFloat32(base + 40, group.bounds_min[2], true);
    view.setFloat32(base + 44, 0.0, true);

    view.setFloat32(base + 48, group.bounds_max[0], true);
    view.setFloat32(base + 52, group.bounds_max[1], true);
    view.setFloat32(base + 56, group.bounds_max[2], true);
    view.setFloat32(base + 60, 0.0, true);
  }

  manifest.sections = {
    meshlets: {
      offset: meshlets_offset,
      stride: MESHLET_STRUCT_STRIDE,
      count: meshlet_records.length,
    },
    meshletVertices: {
      offset: meshlet_vertices_offset,
      stride: Uint32Array.BYTES_PER_ELEMENT,
      count: meshlet_vertex_records.length,
      elementType: "uint32",
    },
    meshletTriangles: {
      offset: meshlet_triangles_offset,
      stride: Uint8Array.BYTES_PER_ELEMENT,
      count: meshlet_triangle_records.length,
      elementType: "uint8",
    },
    meshletGroups: {
      offset: meshlet_groups_offset,
      stride: MESHLET_GROUP_STRUCT_STRIDE,
      count: meshlet_group_records.length,
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
  } catch (error) {
    return false;
  }
}

function process_gltf_file(gltf_path, settings) {
  const document = load_gltf_document(gltf_path, settings);
  const manifest_path = gltf_path.replace(/\.gltf$/i, ".meshlet.json");
  const binary_path = gltf_path.replace(/\.gltf$/i, ".meshlet.bin");

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

  let meshlet_count = 0;
  let group_count = 0;
  for (const mesh of output.manifest.meshes) {
    for (const primitive of mesh.primitives) {
      if (primitive.skipped) {
        continue;
      }
      if (Array.isArray(primitive.lods)) {
        for (const lod of primitive.lods) {
          meshlet_count += lod.meshletCount;
          group_count += lod.meshletGroupCount;
        }
      } else {
        meshlet_count += primitive.meshletCount;
        group_count += primitive.meshletGroupCount;
      }
    }
  }

  return {
    status: "generated",
    gltf_path,
    meshlet_count,
    group_count,
  };
}

function format_asset_path(file_path) {
  return path.relative(ASSET_ROOT, file_path).replace(/\\/g, "/");
}

async function main() {
  const settings = {
    version: GENERATOR_VERSION,
    max_vertices: DEFAULT_MAX_VERTICES,
    min_triangles: DEFAULT_MIN_TRIANGLES,
    max_triangles: DEFAULT_MAX_TRIANGLES,
    fill_weight: DEFAULT_FILL_WEIGHT,
    cluster_group_size: DEFAULT_CLUSTER_GROUP_SIZE,
  };

  if (settings.max_vertices > 255) {
    throw new Error("max_vertices must stay <= 255 so local triangle indices fit in uint8.");
  }

  if (!MeshoptClusterizer.supported) {
    throw new Error("meshoptimizer clusterizer is not supported in this Node.js runtime.");
  }

  await MeshoptClusterizer.ready;
  await initialize_mesh_lod_simplifier();

  const gltf_files = [];
  find_gltf_files(ASSET_ROOT, gltf_files);
  gltf_files.sort((a, b) => a.localeCompare(b));

  const summary = {
    generated: 0,
    skipped: 0,
    failed: 0,
    meshlets: 0,
    groups: 0,
  };

  for (const gltf_path of gltf_files) {
    try {
      const result = process_gltf_file(gltf_path, settings);
      if (result.status === "skipped") {
        summary.skipped++;
        console.log(`[meshlet_preprocessor] up to date: ${format_asset_path(gltf_path)}`);
        continue;
      }

      summary.generated++;
      summary.meshlets += result.meshlet_count;
      summary.groups += result.group_count;
      console.log(
        `[meshlet_preprocessor] generated ${format_asset_path(gltf_path)} (${result.meshlet_count} meshlets, ${result.group_count} groups)`
      );
    } catch (error) {
      summary.failed++;
      console.error(`[meshlet_preprocessor] failed ${format_asset_path(gltf_path)}:`, error);
    }
  }

  console.log(
    `[meshlet_preprocessor] complete: ${summary.generated} generated, ${summary.skipped} skipped, ${summary.failed} failed, ${summary.meshlets} meshlets, ${summary.groups} groups`
  );

  if (summary.failed > 0) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("[meshlet_preprocessor] fatal:", error);
  process.exit(1);
});

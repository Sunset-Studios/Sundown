import { Mesh } from "./mesh.js";
import { MeshData, vertex_stride } from "./mesh_data.js";
import { MeshoptClusterizer } from "meshoptimizer/clusterizer";
import {
  read_binary_manifest_async,
  resolve_manifest_asset_path,
} from "../streaming/streaming_io.js";
import { floor_to_multiple, pack_snorm4x8 } from "../utility/math.js";
import {
  build_empty_meshlet_sections,
  build_meshlet_groups,
  build_meshlets,
  compute_position_bounds,
  sort_meshlets_spatially,
  transform_meshlet_record,
  transform_meshlet_group_record,
} from "./meshlet_utils.js";

const discard_cpu_data = true;
const tangent_epsilon = 1e-6;
const meshlet_sidecar_promise_cache = new Map();

const runtime_meshlet_defaults = {
  max_vertices: 64,
  min_triangles: 24,
  max_triangles: 124,
  fill_weight: 0.5,
  cluster_group_size: 8,
};

function read_index_accessor(accessor) {
  const byte_offset = accessor.byteOffset || 0;
  if (accessor.componentType === 5123) {
    const temp_indices = new Uint16Array(accessor.bufferView.data, byte_offset, accessor.count);
    return Uint32Array.from(temp_indices);
  }
  if (accessor.componentType === 5125) {
    return new Uint32Array(accessor.bufferView.data, byte_offset, accessor.count);
  }
  if (accessor.componentType === 5121) {
    const temp_indices = new Uint8Array(accessor.bufferView.data, byte_offset, accessor.count);
    return Uint32Array.from(temp_indices);
  }
  return new Uint32Array(0);
}

function read_primitive_indices(gltf_obj, primitive, vertex_count) {
  if (primitive.indices === null || primitive.indices === undefined) {
    const indices = new Uint32Array(vertex_count);
    for (let i = 0; i < vertex_count; i++) {
      indices[i] = i;
    }
    return indices;
  }
  return read_index_accessor(gltf_obj.accessors[primitive.indices]);
}

function get_gltf_material_index(gltf_obj, primitive) {
  if (primitive.material === undefined || primitive.material === null) {
    return -1;
  }
  return gltf_obj.materials.indexOf(primitive.material);
}

export function load_meshlet_sidecar_async(gltf_path) {
  if (!gltf_path || !gltf_path.toLowerCase().endsWith(".gltf")) {
    return Promise.resolve(null);
  }
  if (meshlet_sidecar_promise_cache.has(gltf_path)) {
    return meshlet_sidecar_promise_cache.get(gltf_path);
  }

  const sidecar_promise = (async () => {
    const manifest_path = gltf_path.replace(/\.gltf$/i, ".meshlet.json");
    try {
      const bundle = await read_binary_manifest_async(manifest_path, {
        label: "Meshlet sidecar",
        optional: true,
        resolve_binary_path: (manifest) =>
          manifest.binary
            ? resolve_manifest_asset_path(manifest_path, manifest.binary)
            : gltf_path.replace(/\.gltf$/i, ".meshlet.bin"),
      });
      if (!bundle) {
        return null;
      }

      return {
        manifest: bundle.manifest,
        meshlet_view: new DataView(bundle.binary, bundle.manifest.sections.meshlets.offset),
        meshlet_vertices: new Uint32Array(
          bundle.binary,
          bundle.manifest.sections.meshletVertices.offset,
          bundle.manifest.sections.meshletVertices.count
        ),
        meshlet_triangles: new Uint8Array(
          bundle.binary,
          bundle.manifest.sections.meshletTriangles.offset,
          bundle.manifest.sections.meshletTriangles.count
        ),
        meshlet_group_view: new DataView(
          bundle.binary,
          bundle.manifest.sections.meshletGroups.offset
        ),
      };
    } catch {
      return null;
    }
  })();

  meshlet_sidecar_promise_cache.set(gltf_path, sidecar_promise);
  return sidecar_promise;
}

function get_meshlet_primitive_info(sidecar, mesh_index, primitive_index, lod = 0) {
  if (!sidecar || mesh_index < 0) {
    return null;
  }
  const manifest_mesh = sidecar.manifest.meshes?.[mesh_index];
  if (!manifest_mesh) {
    return null;
  }
  const primitive_info = manifest_mesh.primitives?.[primitive_index] ?? null;
  if (!primitive_info || !Array.isArray(primitive_info.lods)) {
    return lod === 0 ? primitive_info : null;
  }
  const lod_info = primitive_info.lods[lod];
  return lod_info ? { ...primitive_info, ...lod_info } : null;
}

function read_meshlet_record(sidecar, meshlet_index) {
  const section = sidecar.manifest.sections.meshlets;
  const base = meshlet_index * section.stride;
  return {
    vertex_offset: sidecar.meshlet_view.getUint32(base + 0, true),
    vertex_count: sidecar.meshlet_view.getUint32(base + 4, true),
    triangle_offset: sidecar.meshlet_view.getUint32(base + 8, true),
    triangle_count: sidecar.meshlet_view.getUint32(base + 12, true),
    center: [
      sidecar.meshlet_view.getFloat32(base + 16, true),
      sidecar.meshlet_view.getFloat32(base + 20, true),
      sidecar.meshlet_view.getFloat32(base + 24, true),
    ],
    radius: sidecar.meshlet_view.getFloat32(base + 28, true),
    bounds_min: [
      sidecar.meshlet_view.getFloat32(base + 32, true),
      sidecar.meshlet_view.getFloat32(base + 36, true),
      sidecar.meshlet_view.getFloat32(base + 40, true),
    ],
    bounds_max: [
      sidecar.meshlet_view.getFloat32(base + 48, true),
      sidecar.meshlet_view.getFloat32(base + 52, true),
      sidecar.meshlet_view.getFloat32(base + 56, true),
    ],
    normal_cone_axis: [
      sidecar.meshlet_view.getFloat32(base + 64, true),
      sidecar.meshlet_view.getFloat32(base + 68, true),
      sidecar.meshlet_view.getFloat32(base + 72, true),
    ],
    normal_cone_cutoff: sidecar.meshlet_view.getFloat32(base + 76, true),
  };
}

function read_meshlet_group_record(sidecar, group_index) {
  const section = sidecar.manifest.sections.meshletGroups;
  const base = group_index * section.stride;
  return {
    meshlet_offset: sidecar.meshlet_group_view.getUint32(base + 0, true),
    meshlet_count: sidecar.meshlet_group_view.getUint32(base + 4, true),
    center: [
      sidecar.meshlet_group_view.getFloat32(base + 16, true),
      sidecar.meshlet_group_view.getFloat32(base + 20, true),
      sidecar.meshlet_group_view.getFloat32(base + 24, true),
    ],
    radius: sidecar.meshlet_group_view.getFloat32(base + 28, true),
    bounds_min: [
      sidecar.meshlet_group_view.getFloat32(base + 32, true),
      sidecar.meshlet_group_view.getFloat32(base + 36, true),
      sidecar.meshlet_group_view.getFloat32(base + 40, true),
    ],
    bounds_max: [
      sidecar.meshlet_group_view.getFloat32(base + 48, true),
      sidecar.meshlet_group_view.getFloat32(base + 52, true),
      sidecar.meshlet_group_view.getFloat32(base + 56, true),
    ],
  };
}

function create_empty_bounds_array() {
  return [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
}

function read_primitive_source_data(gltf_obj, primitive, vertex_count) {
  const positions =
    primitive.attributes.POSITION !== undefined
      ? Mesh._read_accessor_f32(primitive.attributes.POSITION)
      : new Float32Array(0);
  const normals =
    primitive.attributes.NORMAL !== undefined
      ? Mesh._read_accessor_f32(primitive.attributes.NORMAL)
      : new Float32Array(0);
  const tangents =
    primitive.attributes.TANGENT !== undefined
      ? Mesh._read_accessor_f32(primitive.attributes.TANGENT)
      : new Float32Array(0);
  const uvs =
    primitive.attributes.TEXCOORD_0 !== undefined
      ? Mesh._read_accessor_f32(primitive.attributes.TEXCOORD_0)
      : new Float32Array(0);

  return {
    positions,
    normals,
    tangents,
    uvs,
    indices: read_primitive_indices(gltf_obj, primitive, vertex_count),
  };
}

function analyze_gltf_primitives(gltf_obj, gltf_mesh, resolved_mesh_index) {
  const primitive_descriptors = [];
  const material_order = [];
  const material_order_lookup = new Map();
  const index_count_by_material_order = [];

  let total_vertex_count = 0;

  for (let primitive_index = 0; primitive_index < gltf_mesh.primitives.length; primitive_index++) {
    const primitive = gltf_mesh.primitives[primitive_index];
    const material_index = get_gltf_material_index(gltf_obj, primitive);
    let material_order_index = material_order_lookup.get(material_index);
    if (material_order_index === undefined) {
      material_order_index = material_order.length;
      material_order_lookup.set(material_index, material_order_index);
      material_order.push(material_index);
      index_count_by_material_order.push(0);
    }

    const vertex_count = primitive.attributes.POSITION?.count ?? 0;
    const index_count =
      primitive.indices === null || primitive.indices === undefined
        ? vertex_count
        : (gltf_obj.accessors[primitive.indices]?.count ?? 0);

    primitive_descriptors.push({
      mesh_index: resolved_mesh_index,
      primitive_index,
      material_index,
      vertex_offset: total_vertex_count,
      vertex_count,
      index_count,
      section_index: -1,
      world_matrix: null,
      normal_matrix: null,
      material_order_index,
    });

    total_vertex_count += vertex_count;
    index_count_by_material_order[material_order_index] += index_count;
  }

  const section_defs = [];
  const section_index_by_material = new Map();
  let running_first_index = 0;
  for (
    let material_order_index = 0;
    material_order_index < material_order.length;
    material_order_index++
  ) {
    const material_index = material_order[material_order_index];
    const index_count = index_count_by_material_order[material_order_index] ?? 0;
    if (index_count <= 0) {
      continue;
    }

    section_index_by_material.set(material_index, section_defs.length);
    section_defs.push({
      material_index,
      first_index: running_first_index,
      index_count,
    });
    running_first_index += index_count;
  }

  for (let i = 0; i < primitive_descriptors.length; i++) {
    const descriptor = primitive_descriptors[i];
    descriptor.section_index = section_index_by_material.get(descriptor.material_index) ?? -1;
    delete descriptor.material_order_index;
  }

  return {
    primitive_descriptors,
    section_defs,
    total_vertex_count,
    total_index_count: running_first_index,
  };
}

function build_generated_tangent_data(positions, normals, uvs, indices, vertex_count) {
  if (vertex_count <= 0 || uvs.length < vertex_count * 2 || indices.length < 3) {
    return null;
  }

  const tan1 = new Float32Array(vertex_count * 3);
  const tan2 = new Float32Array(vertex_count * 3);

  for (let i = 0; i + 2 < indices.length; i += 3) {
    const i0 = indices[i];
    const i1 = indices[i + 1];
    const i2 = indices[i + 2];
    if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
      continue;
    }

    const p0 = i0 * 3;
    const p1 = i1 * 3;
    const p2 = i2 * 3;
    const uv0 = i0 * 2;
    const uv1 = i1 * 2;
    const uv2 = i2 * 2;

    const x1 = positions[p1] - positions[p0];
    const y1 = positions[p1 + 1] - positions[p0 + 1];
    const z1 = positions[p1 + 2] - positions[p0 + 2];
    const x2 = positions[p2] - positions[p0];
    const y2 = positions[p2 + 1] - positions[p0 + 1];
    const z2 = positions[p2 + 2] - positions[p0 + 2];

    const s1 = uvs[uv1] - uvs[uv0];
    const t1 = uvs[uv1 + 1] - uvs[uv0 + 1];
    const s2 = uvs[uv2] - uvs[uv0];
    const t2 = uvs[uv2 + 1] - uvs[uv0 + 1];

    const det = s1 * t2 - s2 * t1;
    if (Math.abs(det) <= tangent_epsilon) {
      continue;
    }

    const inv_det = 1.0 / det;
    const sdir_x = (t2 * x1 - t1 * x2) * inv_det;
    const sdir_y = (t2 * y1 - t1 * y2) * inv_det;
    const sdir_z = (t2 * z1 - t1 * z2) * inv_det;
    const tdir_x = (s1 * x2 - s2 * x1) * inv_det;
    const tdir_y = (s1 * y2 - s2 * y1) * inv_det;
    const tdir_z = (s1 * z2 - s2 * z1) * inv_det;

    const t0 = i0 * 3;
    const t1_index = i1 * 3;
    const t2_index = i2 * 3;

    tan1[t0] += sdir_x;
    tan1[t0 + 1] += sdir_y;
    tan1[t0 + 2] += sdir_z;
    tan1[t1_index] += sdir_x;
    tan1[t1_index + 1] += sdir_y;
    tan1[t1_index + 2] += sdir_z;
    tan1[t2_index] += sdir_x;
    tan1[t2_index + 1] += sdir_y;
    tan1[t2_index + 2] += sdir_z;

    tan2[t0] += tdir_x;
    tan2[t0 + 1] += tdir_y;
    tan2[t0 + 2] += tdir_z;
    tan2[t1_index] += tdir_x;
    tan2[t1_index + 1] += tdir_y;
    tan2[t1_index + 2] += tdir_z;
    tan2[t2_index] += tdir_x;
    tan2[t2_index + 1] += tdir_y;
    tan2[t2_index + 2] += tdir_z;
  }

  const generated = new Float32Array(vertex_count * 4);
  for (let vertex_index = 0; vertex_index < vertex_count; vertex_index++) {
    const normal_offset = vertex_index * 3;
    let nx = normals[normal_offset] ?? 0.0;
    let ny = normals[normal_offset + 1] ?? 0.0;
    let nz = normals[normal_offset + 2] ?? 0.0;
    const nlen = Math.hypot(nx, ny, nz);
    const has_normal = nlen > tangent_epsilon;
    if (has_normal) {
      nx /= nlen;
      ny /= nlen;
      nz /= nlen;
    } else {
      nx = 0.0;
      ny = 0.0;
      nz = 0.0;
    }

    let tx = tan1[normal_offset];
    let ty = tan1[normal_offset + 1];
    let tz = tan1[normal_offset + 2];
    if (has_normal) {
      const nt_dot_t = nx * tx + ny * ty + nz * tz;
      tx -= nx * nt_dot_t;
      ty -= ny * nt_dot_t;
      tz -= nz * nt_dot_t;
    }

    let tlen = Math.hypot(tx, ty, tz);
    if (tlen <= tangent_epsilon) {
      tx = 1.0;
      ty = 0.0;
      tz = 0.0;
      tlen = 1.0;
    }

    tx /= tlen;
    ty /= tlen;
    tz /= tlen;

    let handedness = 1.0;
    if (has_normal) {
      const bx = ny * tz - nz * ty;
      const by = nz * tx - nx * tz;
      const bz = nx * ty - ny * tx;
      if (
        bx * tan2[normal_offset] + by * tan2[normal_offset + 1] + bz * tan2[normal_offset + 2] <
        0.0
      ) {
        handedness = -1.0;
      }
    }

    const tangent_offset = vertex_index * 4;
    generated[tangent_offset] = tx;
    generated[tangent_offset + 1] = ty;
    generated[tangent_offset + 2] = tz;
    generated[tangent_offset + 3] = handedness;
  }

  return generated;
}

function write_primitive_vertices_to_staging(
  primitive_data,
  descriptor,
  vertex_view,
  positions_xyz,
  bounds_min_and_max
) {
  const { positions, normals, tangents, uvs, indices } = primitive_data;
  const vertex_offset = descriptor.vertex_offset;
  const vertex_count = descriptor.vertex_count;
  const section_index = descriptor.section_index >= 0 ? descriptor.section_index : 0;
  const tangent_components = vertex_count > 0 ? Math.floor(tangents.length / vertex_count) : 0;
  const generated_tangents =
    tangent_components <= 0 && uvs.length > 0
      ? build_generated_tangent_data(positions, normals, uvs, indices, vertex_count)
      : null;

  for (let vertex_index = 0; vertex_index < vertex_count; vertex_index++) {
    const src_position_offset = vertex_index * 3;
    const src_normal_offset = vertex_index * 3;
    const src_uv_offset = vertex_index * 2;
    const dst_vertex_index = vertex_offset + vertex_index;
    const dst_position_offset = dst_vertex_index * 3;
    const dst_byte_offset = dst_vertex_index * vertex_stride;

    const px = positions[src_position_offset] ?? 0.0;
    const py = positions[src_position_offset + 1] ?? 0.0;
    const pz = positions[src_position_offset + 2] ?? 0.0;
    positions_xyz[dst_position_offset] = px;
    positions_xyz[dst_position_offset + 1] = py;
    positions_xyz[dst_position_offset + 2] = pz;

    bounds_min_and_max[0] = Math.min(bounds_min_and_max[0], px);
    bounds_min_and_max[1] = Math.min(bounds_min_and_max[1], py);
    bounds_min_and_max[2] = Math.min(bounds_min_and_max[2], pz);
    bounds_min_and_max[3] = Math.max(bounds_min_and_max[3], px);
    bounds_min_and_max[4] = Math.max(bounds_min_and_max[4], py);
    bounds_min_and_max[5] = Math.max(bounds_min_and_max[5], pz);

    vertex_view.setFloat32(dst_byte_offset + 0, px, true);
    vertex_view.setFloat32(dst_byte_offset + 4, py, true);
    vertex_view.setFloat32(dst_byte_offset + 8, pz, true);
    vertex_view.setFloat32(dst_byte_offset + 12, section_index, true);
    vertex_view.setFloat32(dst_byte_offset + 16, uvs[src_uv_offset] ?? 0.0, true);
    vertex_view.setFloat32(dst_byte_offset + 20, uvs[src_uv_offset + 1] ?? 0.0, true);

    let nx = normals[src_normal_offset] ?? 0.0;
    let ny = normals[src_normal_offset + 1] ?? 0.0;
    let nz = normals[src_normal_offset + 2] ?? 0.0;
    const normal_len = Math.hypot(nx, ny, nz);
    const has_normal = normal_len > tangent_epsilon;
    if (has_normal) {
      nx /= normal_len;
      ny /= normal_len;
      nz /= normal_len;
    } else {
      nx = 0.0;
      ny = 0.0;
      nz = 0.0;
    }

    vertex_view.setUint32(dst_byte_offset + 24, pack_snorm4x8(nx, ny, nz, 0.0), true);

    let tx = 0.0;
    let ty = 0.0;
    let tz = 0.0;
    let tw = 1.0;
    if (generated_tangents) {
      const generated_offset = vertex_index * 4;
      tx = generated_tangents[generated_offset];
      ty = generated_tangents[generated_offset + 1];
      tz = generated_tangents[generated_offset + 2];
      tw = generated_tangents[generated_offset + 3];
    } else if (tangent_components >= 3) {
      const tangent_offset = vertex_index * tangent_components;
      tx = tangents[tangent_offset] ?? 0.0;
      ty = tangents[tangent_offset + 1] ?? 0.0;
      tz = tangents[tangent_offset + 2] ?? 0.0;
      tw = tangent_components >= 4 ? (tangents[tangent_offset + 3] ?? 1.0) : 1.0;

      if (has_normal) {
        const nt_dot_t = nx * tx + ny * ty + nz * tz;
        tx -= nx * nt_dot_t;
        ty -= ny * nt_dot_t;
        tz -= nz * nt_dot_t;
      }

      const tangent_len = Math.hypot(tx, ty, tz);
      if (tangent_len > tangent_epsilon) {
        tx /= tangent_len;
        ty /= tangent_len;
        tz /= tangent_len;
      } else {
        tx = 0.0;
        ty = 0.0;
        tz = 0.0;
      }

      tw = Math.abs(tw) > 0.5 ? tw : 1.0;
    }

    vertex_view.setUint32(dst_byte_offset + 28, pack_snorm4x8(tx, ty, tz, tw), true);
  }
}

function write_primitive_indices_to_staging(
  target_indices,
  write_offsets,
  descriptor,
  source_indices
) {
  if (descriptor.section_index < 0 || descriptor.index_count <= 0) {
    return;
  }

  let write_index = write_offsets[descriptor.section_index];
  const vertex_offset = descriptor.vertex_offset;
  for (let i = 0; i < source_indices.length; i++) {
    target_indices[write_index++] = vertex_offset + source_indices[i];
  }
  write_offsets[descriptor.section_index] = write_index;
}

function create_runtime_meshlet_upload_data(positions, indices, sections, settings) {
  if (
    !(positions instanceof Float32Array) ||
    !(indices instanceof Uint32Array) ||
    indices.length < 3
  ) {
    return null;
  }

  const meshlets = [];
  const meshlet_vertices = [];
  const meshlet_triangles = [];
  const meshlet_groups = [];
  const section_descriptors =
    sections && sections.length > 0 ? sections : [{ first_index: 0, index_count: indices.length }];
  const section_payload = build_empty_meshlet_sections(section_descriptors.length);

  for (let section_index = 0; section_index < section_descriptors.length; section_index++) {
    const section = section_descriptors[section_index];
    const first_index = section?.first_index ?? 0;
    const index_count = section?.index_count ?? 0;
    const trimmed_index_count = Math.max(0, floor_to_multiple(index_count, 3));
    if (trimmed_index_count <= 0) {
      continue;
    }

    const section_indices = indices.slice(first_index, first_index + trimmed_index_count);
    if (section_indices.length <= 0) {
      continue;
    }

    const primitive_bounds = compute_position_bounds(positions, section_indices);
    const section_meshlets = sort_meshlets_spatially(
      build_meshlets(section_indices, positions, settings),
      primitive_bounds
    );
    if (section_meshlets.length <= 0) {
      continue;
    }

    const section_groups = build_meshlet_groups(section_meshlets, settings.cluster_group_size);
    const payload_section = section_payload[section_index];
    payload_section.meshlet_offset = meshlets.length;
    payload_section.meshlet_group_offset = meshlet_groups.length;

    for (let meshlet_index = 0; meshlet_index < section_meshlets.length; meshlet_index++) {
      const meshlet = section_meshlets[meshlet_index];
      meshlets.push({
        vertex_offset: meshlet_vertices.length,
        vertex_count: meshlet.global_vertices.length,
        triangle_offset: meshlet_triangles.length,
        triangle_count: meshlet.local_indices.length / 3,
        center: meshlet.center,
        radius: meshlet.radius,
        bounds_min: meshlet.bounds_min,
        bounds_max: meshlet.bounds_max,
        normal_cone_axis: meshlet.normal_cone_axis,
        normal_cone_cutoff: meshlet.normal_cone_cutoff,
      });
      meshlet_vertices.push(...meshlet.global_vertices);
      meshlet_triangles.push(...meshlet.local_indices);
    }

    for (let group_index = 0; group_index < section_groups.length; group_index++) {
      const group = section_groups[group_index];
      meshlet_groups.push({
        meshlet_offset: payload_section.meshlet_offset + group.local_meshlet_offset,
        meshlet_count: group.meshlet_count,
        center: group.center,
        radius: group.radius,
        bounds_min: group.bounds_min,
        bounds_max: group.bounds_max,
      });
    }

    payload_section.meshlet_count = section_meshlets.length;
    payload_section.meshlet_group_count = section_groups.length;
  }

  if (meshlets.length <= 0) {
    return null;
  }

  return {
    meshlets,
    meshlet_vertices: Uint32Array.from(meshlet_vertices),
    meshlet_triangles: Uint8Array.from(meshlet_triangles),
    meshlet_groups,
    sections: section_payload,
  };
}

export function extract_runtime_positions(vertices) {
  const positions = new Float32Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; i++) {
    const position = vertices[i]?.position ?? [0.0, 0.0, 0.0];
    const base = i * 3;
    positions[base + 0] = position[0] ?? 0.0;
    positions[base + 1] = position[1] ?? 0.0;
    positions[base + 2] = position[2] ?? 0.0;
  }
  return positions;
}

export function build_empty_runtime_meshlet_sections(section_count) {
  return build_empty_meshlet_sections(section_count);
}

export async function create_runtime_meshlet_data_async(
  positions,
  indices,
  sections = null,
  settings = {}
) {
  if (!MeshoptClusterizer.supported) {
    return null;
  }

  await MeshoptClusterizer.ready;

  return create_runtime_meshlet_upload_data(
    positions,
    indices,
    sections,
    Object.assign({}, runtime_meshlet_defaults, settings)
  );
}

function create_meshlet_upload_data(sidecar, primitive_descriptors, section_count) {
  if (!sidecar || primitive_descriptors.length === 0) {
    return null;
  }

  const meshlets = [];
  const meshlet_vertices = [];
  const meshlet_triangles = [];
  const meshlet_groups = [];
  const sorted_descriptors = primitive_descriptors
    .filter((descriptor) => descriptor.section_index >= 0)
    .sort((a, b) => {
      let diff = a.section_index - b.section_index;
      if (diff !== 0) return diff;
      diff = a.vertex_offset - b.vertex_offset;
      if (diff !== 0) return diff;
      return a.primitive_index - b.primitive_index;
    });
  const manifest_mesh = sidecar.manifest.meshes?.[sorted_descriptors[0]?.mesh_index];
  const lod_count = Math.max(1, manifest_mesh?.lodCount ?? 1);
  const lods = [];

  for (let lod = 0; lod < lod_count; lod++) {
    const sections = build_empty_meshlet_sections(section_count);

    for (const descriptor of sorted_descriptors) {
      const primitive_info = get_meshlet_primitive_info(
        sidecar,
        descriptor.mesh_index,
        descriptor.primitive_index,
        lod
      );
      const use_meshlets =
        primitive_info &&
        primitive_info.skipped !== true &&
        primitive_info.mode === 4 &&
        primitive_info.meshletCount > 0 &&
        primitive_info.vertexCount === descriptor.vertex_count;

      if (!use_meshlets) {
        continue;
      }

      const section = sections[descriptor.section_index];
      if (section.meshlet_count === 0) {
        section.meshlet_offset = meshlets.length;
        section.meshlet_group_offset = meshlet_groups.length;
      }
      const primitive_meshlet_offset = meshlets.length;

      for (let meshlet_offset = 0; meshlet_offset < primitive_info.meshletCount; meshlet_offset++) {
        const meshlet_record = transform_meshlet_record(
          read_meshlet_record(sidecar, primitive_info.meshletOffset + meshlet_offset),
          descriptor
        );
        const rebased_vertex_offset = meshlet_vertices.length;
        const rebased_triangle_offset = meshlet_triangles.length;

        for (let i = 0; i < meshlet_record.vertex_count; i++) {
          const local_vertex = sidecar.meshlet_vertices[meshlet_record.vertex_offset + i] ?? 0;
          meshlet_vertices.push(descriptor.vertex_offset + local_vertex);
        }

        const triangle_index_count = meshlet_record.triangle_count * 3;
        for (let i = 0; i < triangle_index_count; i++) {
          meshlet_triangles.push(
            sidecar.meshlet_triangles[meshlet_record.triangle_offset + i] ?? 0
          );
        }

        meshlets.push({
          vertex_offset: rebased_vertex_offset,
          vertex_count: meshlet_record.vertex_count,
          triangle_offset: rebased_triangle_offset,
          triangle_count: meshlet_record.triangle_count,
          center: meshlet_record.center,
          radius: meshlet_record.radius,
          bounds_min: meshlet_record.bounds_min,
          bounds_max: meshlet_record.bounds_max,
          normal_cone_axis: meshlet_record.normal_cone_axis,
          normal_cone_cutoff: meshlet_record.normal_cone_cutoff,
        });
      }

      for (let group_offset = 0; group_offset < primitive_info.meshletGroupCount; group_offset++) {
        const group_record = transform_meshlet_group_record(
          read_meshlet_group_record(sidecar, primitive_info.meshletGroupOffset + group_offset),
          descriptor
        );
        meshlet_groups.push({
          meshlet_offset:
            primitive_meshlet_offset + (group_record.meshlet_offset - primitive_info.meshletOffset),
          meshlet_count: group_record.meshlet_count,
          center: group_record.center,
          radius: group_record.radius,
          bounds_min: group_record.bounds_min,
          bounds_max: group_record.bounds_max,
        });
      }

      section.meshlet_count += primitive_info.meshletCount;
      section.meshlet_group_count += primitive_info.meshletGroupCount;
    }

    lods.push({ lod, sections });
  }

  if (meshlets.length === 0) {
    return null;
  }

  return {
    meshlets,
    meshlet_vertices: Uint32Array.from(meshlet_vertices),
    meshlet_triangles: Uint8Array.from(meshlet_triangles),
    meshlet_groups,
    sections: lods[0].sections,
    lods,
    default_min_lod: manifest_mesh?.defaultMinLod,
  };
}

function finalize_gltf_build(mesh, gltf_obj, build_state, sidecar) {
  const {
    primitive_descriptors,
    section_defs,
    vertex_upload_bytes,
    positions_xyz,
    indices,
    bounds_min_and_max,
    total_vertex_count,
  } = build_state;

  mesh.sections = new Array(section_defs.length);
  for (let section_index = 0; section_index < section_defs.length; section_index++) {
    const section_def = section_defs[section_index];
    const section = {
      first_index: section_def.first_index,
      index_count: section_def.index_count,
      material_id: null,
    };
    if (section_def.material_index >= 0) {
      const gltf_mat = gltf_obj.materials[section_def.material_index];
      section.material_id = Mesh.make_engine_material_from_gltf(
        gltf_obj,
        mesh,
        gltf_mat,
        section_def.material_index
      );
    }
    mesh.sections[section_index] = section;
  }

  mesh.vertices = null;
  mesh.packed_vertex_data = vertex_upload_bytes;
  mesh.cpu_position_data = positions_xyz;
  let upload_indices = indices;
  if (mesh.cooked_sbvh?.triangle_indices?.length > 0) {
    mesh.cooked_sbvh.index_offset = indices.length;
    upload_indices = new Uint32Array(indices.length + mesh.cooked_sbvh.triangle_indices.length);
    upload_indices.set(indices);
    upload_indices.set(mesh.cooked_sbvh.triangle_indices, indices.length);
  } else if (mesh.cooked_sbvh) {
    mesh.cooked_sbvh.index_offset = 0;
  }
  mesh.indices = upload_indices;
  mesh.vertex_count = total_vertex_count;
  mesh.index_count = indices.length;
  mesh.bounds_min_and_max = bounds_min_and_max;

  mesh.meshlet_data = create_meshlet_upload_data(
    sidecar,
    primitive_descriptors,
    mesh.sections.length
  );
  mesh.meshlet_sections = mesh.meshlet_data
    ? mesh.meshlet_data.sections.map((section) => ({ ...section }))
    : build_empty_meshlet_sections(mesh.sections.length);

  MeshData.update(mesh);

  if (discard_cpu_data) {
    mesh.packed_vertex_data = null;
    mesh.cpu_position_data = null;
    mesh.indices = null;
    mesh._tmp_indices = null;
    mesh._section_groups = null;
  }
}

export function prepare_gltf_mesh_build(mesh, gltf_obj, gltf_mesh, mesh_index) {
  mesh._reset_build_state();

  const { primitive_descriptors, section_defs, total_vertex_count, total_index_count } =
    analyze_gltf_primitives(gltf_obj, gltf_mesh, mesh_index);

  const vertex_upload_bytes = new Uint8Array(total_vertex_count * vertex_stride);
  const vertex_view = new DataView(vertex_upload_bytes.buffer);
  const positions_xyz = new Float32Array(total_vertex_count * 3);
  const indices = new Uint32Array(total_index_count);

  const bounds_min_and_max = create_empty_bounds_array();
  const section_write_offsets = section_defs.map((section) => section.first_index);

  for (let primitive_index = 0; primitive_index < primitive_descriptors.length; primitive_index++) {
    const descriptor = primitive_descriptors[primitive_index];
    const primitive = gltf_mesh.primitives[descriptor.primitive_index];
    const primitive_data = read_primitive_source_data(gltf_obj, primitive, descriptor.vertex_count);
    write_primitive_vertices_to_staging(
      primitive_data,
      descriptor,
      vertex_view,
      positions_xyz,
      bounds_min_and_max
    );
    write_primitive_indices_to_staging(
      indices,
      section_write_offsets,
      descriptor,
      primitive_data.indices
    );
  }

  return {
    primitive_descriptors,
    section_defs,
    vertex_upload_bytes,
    positions_xyz,
    indices,
    bounds_min_and_max,
    total_vertex_count,
  };
}

export function finalize_prepared_gltf_mesh_build(mesh, gltf_obj, build_state, sidecar = null) {
  if (!build_state) {
    return;
  }
  finalize_gltf_build(mesh, gltf_obj, build_state, sidecar);
}

export function build_gltf_mesh(mesh, gltf_obj, gltf_mesh, mesh_index, sidecar = null) {
  const build_state = prepare_gltf_mesh_build(mesh, gltf_obj, gltf_mesh, mesh_index);
  finalize_prepared_gltf_mesh_build(mesh, gltf_obj, build_state, sidecar);
}

import { vec3, quat, mat3, mat4 } from "gl-matrix";
import { Mesh } from "./mesh.js";
import { MeshData } from "./mesh_data.js";
import { MeshoptClusterizer } from "meshoptimizer/clusterizer";
import { read_file_async, read_file_bytes_async } from "../utility/file_system.js";
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
const meshlet_sidecar_promise_cache = new Map();
const runtime_meshlet_defaults = {
  max_vertices: 64,
  min_triangles: 24,
  max_triangles: 124,
  fill_weight: 0.5,
  cluster_group_size: 8,
};
const Type2NumOfComponent = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
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

function resolve_gltf_mesh_index(gltf_obj, gltf_mesh) {
  if (gltf_mesh?.meshID !== undefined && gltf_mesh.meshID !== null) {
    return gltf_mesh.meshID;
  }
  return gltf_obj.meshes.indexOf(gltf_mesh);
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
    const manifest_text = await read_file_async(manifest_path);
    if (!manifest_text) {
      return null;
    }

    try {
      const manifest = JSON.parse(manifest_text);
      const base_path_index = manifest_path.lastIndexOf("/");
      const base_path = base_path_index >= 0 ? manifest_path.slice(0, base_path_index + 1) : "";
      const binary_path = manifest.binary
        ? `${base_path}${manifest.binary}`
        : gltf_path.replace(/\.gltf$/i, ".meshlet.bin");
      const binary = await read_file_bytes_async(binary_path);
      if (!(binary instanceof ArrayBuffer)) {
        return null;
      }

      return {
        manifest,
        meshlet_view: new DataView(binary, manifest.sections.meshlets.offset),
        meshlet_vertices: new Uint32Array(
          binary,
          manifest.sections.meshletVertices.offset,
          manifest.sections.meshletVertices.count
        ),
        meshlet_triangles: new Uint8Array(
          binary,
          manifest.sections.meshletTriangles.offset,
          manifest.sections.meshletTriangles.count
        ),
        meshlet_group_view: new DataView(binary, manifest.sections.meshletGroups.offset),
      };
    } catch {
      return null;
    }
  })();

  meshlet_sidecar_promise_cache.set(gltf_path, sidecar_promise);
  return sidecar_promise;
}

function get_meshlet_primitive_info(sidecar, mesh_index, primitive_index) {
  if (!sidecar || mesh_index < 0) {
    return null;
  }
  const manifest_mesh = sidecar.manifest.meshes?.[mesh_index];
  if (!manifest_mesh) {
    return null;
  }
  return manifest_mesh.primitives?.[primitive_index] ?? null;
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

function read_primitive_source_data(gltf_obj, primitive) {
  let positions = [];
  if (primitive.attributes.POSITION !== undefined) {
    positions = Mesh._read_accessor_f32(primitive.attributes.POSITION);
  }

  let normals = [];
  if (primitive.attributes.NORMAL !== undefined) {
    normals = Mesh._read_accessor_f32(primitive.attributes.NORMAL);
  }

  let tangents = [];
  if (primitive.attributes.TANGENT !== undefined) {
    tangents = Mesh._read_accessor_f32(primitive.attributes.TANGENT);
  }

  let bitangents = [];
  if (primitive.attributes.BITANGENT !== undefined) {
    bitangents = Mesh._read_accessor_f32(primitive.attributes.BITANGENT);
  }

  let colors = [];
  let color_components = 0;
  if (primitive.attributes.COLOR_0 !== undefined) {
    const color_accessor = primitive.attributes.COLOR_0;
    color_components = Type2NumOfComponent[color_accessor.type];
    colors = Mesh._read_accessor_f32(color_accessor);
  }

  let uvs = [];
  if (primitive.attributes.TEXCOORD_0 !== undefined) {
    uvs = Mesh._read_accessor_f32(primitive.attributes.TEXCOORD_0);
  }

  if (tangents.length === 0 && positions.length > 0 && uvs.length > 0) {
    const computed = Mesh._get_tangents_and_bitangents(positions, uvs);
    tangents = computed.t;
    bitangents = computed.b;
  } else if (bitangents.length === 0 && tangents.length > 0 && normals.length > 0) {
    bitangents = new Array((tangents.length / 4) * 3);
    for (let vi = 0; vi < tangents.length / 4; vi++) {
      const tangent_start = vi * 4;
      const normal_start = vi * 3;
      const t = [tangents[tangent_start], tangents[tangent_start + 1], tangents[tangent_start + 2]];
      const handedness = tangents[tangent_start + 3] || 1;
      const n = [normals[normal_start], normals[normal_start + 1], normals[normal_start + 2]];
      const b = vec3.cross(vec3.create(), n, t);
      vec3.scale(b, b, handedness);
      vec3.normalize(b, b);
      const bitangent_start = vi * 3;
      bitangents[bitangent_start] = b[0];
      bitangents[bitangent_start + 1] = b[1];
      bitangents[bitangent_start + 2] = b[2];
    }
  }

  const vertex_count = positions.length / 3;
  const indices = read_primitive_indices(gltf_obj, primitive, vertex_count);

  return {
    positions,
    normals,
    tangents,
    bitangents,
    colors,
    color_components,
    uvs,
    indices,
  };
}

function build_vertices_from_primitive_data(primitive_data, transform_state = null) {
  const vertices = [];
  const { positions, normals, tangents, colors, color_components, uvs } = primitive_data;
  const num_verts = positions.length / 3;
  const tangent_is_vec4 = tangents.length > 0 && tangents.length % 4 === 0;
  const world_matrix = transform_state?.world_matrix ?? null;
  const normal_matrix = transform_state?.normal_matrix ?? null;

  for (let k = 0; k < num_verts; k++) {
    const pos_index = k * 3;
    const normal_index = k * 3;
    const uv_index = k * 2;
    const tangent_index = tangent_is_vec4 ? k * 4 : k * 3;
    const color_index = k * color_components;

    let color = [1, 1, 1, 1];
    if (colors.length > 0) {
      if (color_components === 3) {
        color = [colors[color_index], colors[color_index + 1], colors[color_index + 2], 1];
      } else {
        color = [
          colors[color_index],
          colors[color_index + 1],
          colors[color_index + 2],
          colors[color_index + 3],
        ];
      }
    }

    let tangent_w = 1.0;
    if (tangent_is_vec4) {
      tangent_w = tangents[tangent_index + 3] ?? 1.0;
    }

    const src_pos = vec3.fromValues(
      positions[pos_index] ?? 0.0,
      positions[pos_index + 1] ?? 0.0,
      positions[pos_index + 2] ?? 0.0
    );
    const position = world_matrix ? vec3.transformMat4(vec3.create(), src_pos, world_matrix) : src_pos;

    const n = vec3.fromValues(
      normals[normal_index] ?? 0.0,
      normals[normal_index + 1] ?? 0.0,
      normals[normal_index + 2] ?? 0.0
    );
    if (normal_matrix) {
      vec3.transformMat3(n, n, normal_matrix);
    }
    vec3.normalize(n, n);

    const t = vec3.fromValues(
      tangents[tangent_index] ?? 0.0,
      tangents[tangent_index + 1] ?? 0.0,
      tangents[tangent_index + 2] ?? 0.0
    );
    const nt_dot_t = vec3.dot(n, t);
    const t_ortho = vec3.subtract(vec3.create(), t, vec3.scale(vec3.create(), n, nt_dot_t));
    if (normal_matrix) {
      vec3.transformMat3(t_ortho, t_ortho, normal_matrix);
    }
    vec3.normalize(t_ortho, t_ortho);

    const handedness = Math.abs(tangent_w) > 0.5 ? tangent_w : 1.0;
    const b = vec3.cross(vec3.create(), n, t_ortho);
    vec3.scale(b, b, handedness);
    if (normal_matrix) {
      vec3.transformMat3(b, b, normal_matrix);
    }
    vec3.normalize(b, b);

    vertices.push({
      position: [position[0], position[1], position[2], 1],
      normal: [n[0], n[1], n[2], 0],
      color,
      uv: [uvs[uv_index] ?? 0.0, uvs[uv_index + 1] ?? 0.0],
      tangent: [t_ortho[0], t_ortho[1], t_ortho[2], tangent_w],
      bitangent: [b[0], b[1], b[2], 0],
      extra_data: [0, 0],
    });
  }

  return vertices;
}

function append_standard_primitive(mesh, group, source_vertices, indices) {
  const vertex_base = mesh.vertices.length;
  for (let i = 0; i < source_vertices.length; i++) {
    mesh.vertices.push(source_vertices[i]);
  }
  for (let i = 0; i < indices.length; i++) {
    group.indices.push(indices[i] + vertex_base);
  }
}

function create_runtime_meshlet_upload_data(positions, indices, sections, settings) {
  if (!(positions instanceof Float32Array) || !(indices instanceof Uint32Array) || indices.length < 3) {
    return null;
  }

  const meshlets = [];
  const meshlet_vertices = [];
  const meshlet_triangles = [];
  const meshlet_groups = [];
  const section_descriptors =
    sections && sections.length > 0
      ? sections
      : [{ first_index: 0, index_count: indices.length }];
  const section_payload = build_empty_meshlet_sections(section_descriptors.length);

  for (let section_index = 0; section_index < section_descriptors.length; section_index++) {
    const section = section_descriptors[section_index];
    const first_index = section?.first_index ?? 0;
    const index_count = section?.index_count ?? 0;
    const trimmed_index_count = Math.max(0, index_count - (index_count % 3));
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

    const section_groups = build_meshlet_groups(
      section_meshlets,
      settings.cluster_group_size
    );
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
  const sections = build_empty_meshlet_sections(section_count);

  const sorted_descriptors = primitive_descriptors
    .filter((descriptor) => descriptor.section_index >= 0)
    .sort((a, b) => {
      let diff = a.section_index - b.section_index;
      if (diff !== 0) return diff;
      diff = a.vertex_offset - b.vertex_offset;
      if (diff !== 0) return diff;
      return a.primitive_index - b.primitive_index;
    });

  for (const descriptor of sorted_descriptors) {
    const primitive_info = get_meshlet_primitive_info(
      sidecar,
      descriptor.mesh_index,
      descriptor.primitive_index
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

    for (let meshlet_offset = 0; meshlet_offset < primitive_info.meshletCount; meshlet_offset++) {
      const meshlet_record = transform_meshlet_record(
        read_meshlet_record(
          sidecar,
          primitive_info.meshletOffset + meshlet_offset
        ),
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
        meshlet_triangles.push(sidecar.meshlet_triangles[meshlet_record.triangle_offset + i] ?? 0);
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
        read_meshlet_group_record(
          sidecar,
          primitive_info.meshletGroupOffset + group_offset
        ),
        descriptor
      );
      meshlet_groups.push({
        meshlet_offset:
          section.meshlet_offset +
          (group_record.meshlet_offset - primitive_info.meshletOffset),
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

  if (meshlets.length === 0) {
    return null;
  }

  return {
    meshlets,
    meshlet_vertices: Uint32Array.from(meshlet_vertices),
    meshlet_triangles: Uint8Array.from(meshlet_triangles),
    meshlet_groups,
    sections,
  };
}

function finalize_gltf_build(mesh, gltf_obj, material_cache, primitive_descriptors, sidecar) {
  mesh.sections = [];
  mesh.index_count = 0;
  mesh._tmp_indices.length = 0;

  const vertex_section_map = new Int32Array(mesh.vertices.length).fill(-1);

  let running_first_index = 0;
  const section_index_by_material = new Map();
  for (const [, group] of mesh._section_groups) {
    if (group.indices.length === 0) {
      continue;
    }

    const section_index = mesh.sections.length;
    for (let i = 0; i < group.indices.length; i++) {
      vertex_section_map[group.indices[i]] = section_index;
    }

    const new_section = {
      first_index: running_first_index,
      index_count: group.indices.length,
      material_id: null,
    };
    if (group.key >= 0) {
      const gltf_mat = gltf_obj.materials[group.key];
      new_section.material_id = Mesh.make_engine_material_from_gltf(
        gltf_obj,
        mesh,
        gltf_mat,
        group.key,
        material_cache
      );
    }
    section_index_by_material.set(group.key, section_index);
    mesh.sections.push(new_section);
    mesh._tmp_indices = mesh._tmp_indices.concat(group.indices);
    running_first_index += group.indices.length;
  }

  for (let vi = 0; vi < mesh.vertices.length; vi++) {
    const section_index = vertex_section_map[vi] >= 0 ? vertex_section_map[vi] : 0;
    mesh.vertices[vi].extra_data[0] = section_index;
  }

  if (mesh._tmp_indices.length > 0) {
    mesh.indices = new Uint32Array(mesh._tmp_indices);
  }

  mesh.vertex_count = mesh.vertices.length;
  mesh.index_count = mesh.indices.length;

  mesh._recreate_vertex_bounds();

  for (let i = 0; i < primitive_descriptors.length; i++) {
    const descriptor = primitive_descriptors[i];
    descriptor.section_index = section_index_by_material.get(descriptor.material_index) ?? -1;
  }

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
    mesh.vertices = null;
    mesh.indices = null;
    mesh._tmp_indices = null;
    mesh._section_groups = null;
  }
}

export function prepare_gltf_mesh_build(mesh, gltf_obj, gltf_mesh, mesh_index = null) {
  mesh._reset_build_state();

  const material_cache = new Map();
  const resolved_mesh_index = mesh_index ?? resolve_gltf_mesh_index(gltf_obj, gltf_mesh);
  const primitive_descriptors = [];

  for (let primitive_index = 0; primitive_index < gltf_mesh.primitives.length; primitive_index++) {
    const primitive = gltf_mesh.primitives[primitive_index];
    const material_index = get_gltf_material_index(gltf_obj, primitive);
    let group = mesh._section_groups.get(material_index);
    if (!group) {
      group = { key: material_index, indices: [] };
      mesh._section_groups.set(material_index, group);
    }

    const primitive_data = read_primitive_source_data(gltf_obj, primitive);
    const source_vertices = build_vertices_from_primitive_data(primitive_data);
    primitive_descriptors.push({
      mesh_index: resolved_mesh_index,
      primitive_index,
      material_index,
      vertex_offset: mesh.vertices.length,
      vertex_count: source_vertices.length,
      section_index: -1,
      world_matrix: null,
      normal_matrix: null,
    });

    append_standard_primitive(mesh, group, source_vertices, primitive_data.indices);
  }

  return {
    material_cache,
    primitive_descriptors,
  };
}

export function finalize_prepared_gltf_mesh_build(mesh, gltf_obj, build_state, sidecar = null) {
  if (!build_state) {
    return;
  }

  const { material_cache, primitive_descriptors } = build_state;
  finalize_gltf_build(mesh, gltf_obj, material_cache, primitive_descriptors, sidecar);
}

export function build_gltf_mesh(mesh, gltf_obj, gltf_mesh, mesh_index = null, sidecar = null) {
  const build_state = prepare_gltf_mesh_build(mesh, gltf_obj, gltf_mesh, mesh_index);
  finalize_prepared_gltf_mesh_build(mesh, gltf_obj, build_state, sidecar);
}

export function build_combined_gltf_scene(mesh, gltf_obj, scene_index = null, sidecar = null) {
  mesh._reset_build_state();

  const material_cache = new Map();
  const primitive_descriptors = [];

  for (const node of gltf_obj.nodes) {
    for (const child of node.children) {
      child._parent = node;
    }
  }

  const scene_to_use =
    scene_index ?? gltf_obj.defaultScene ?? (gltf_obj.scenes.length > 0 ? 0 : null);
  const scene = scene_to_use !== null ? gltf_obj.scenes[scene_to_use] : null;
  const root_nodes = scene ? scene.nodes : gltf_obj.nodes;

  const compute_world_matrix = (node) => {
    const chain = [];
    let current = node;
    while (current) {
      chain.unshift(current);
      current = current._parent;
    }

    let world_matrix = mat4.create();
    for (const chain_node of chain) {
      const local_matrix = mat4.create();
      const translation = chain_node.translation
        ? vec3.fromValues(chain_node.translation[0], chain_node.translation[1], chain_node.translation[2])
        : vec3.fromValues(0, 0, 0);
      const rotation = chain_node.rotation
        ? quat.fromValues(chain_node.rotation[0], chain_node.rotation[1], chain_node.rotation[2], chain_node.rotation[3])
        : quat.create();
      const scale = chain_node.scale
        ? vec3.fromValues(chain_node.scale[0], chain_node.scale[1], chain_node.scale[2])
        : vec3.fromValues(1, 1, 1);

      mat4.fromRotationTranslationScale(local_matrix, rotation, translation, scale);
      world_matrix = mat4.multiply(mat4.create(), world_matrix, local_matrix);
    }
    return world_matrix;
  };

  const compute_normal_matrix = (world_matrix) => {
    const normal_matrix = mat3.create();
    mat3.fromMat4(normal_matrix, world_matrix);
    mat3.invert(normal_matrix, normal_matrix);
    mat3.transpose(normal_matrix, normal_matrix);
    return normal_matrix;
  };

  const process_node = (node) => {
    if (node.mesh) {
      const gltf_mesh = node.mesh;
      const mesh_index = resolve_gltf_mesh_index(gltf_obj, gltf_mesh);
      const world_matrix = compute_world_matrix(node);
      const normal_matrix = compute_normal_matrix(world_matrix);

      for (let primitive_index = 0; primitive_index < gltf_mesh.primitives.length; primitive_index++) {
        const primitive = gltf_mesh.primitives[primitive_index];
        const material_index = get_gltf_material_index(gltf_obj, primitive);
        let group = mesh._section_groups.get(material_index);
        if (!group) {
          group = { key: material_index, indices: [] };
          mesh._section_groups.set(material_index, group);
        }

        const primitive_data = read_primitive_source_data(gltf_obj, primitive);
        const source_vertices = build_vertices_from_primitive_data(primitive_data, {
          world_matrix,
          normal_matrix,
        });
        primitive_descriptors.push({
          mesh_index,
          primitive_index,
          material_index,
          vertex_offset: mesh.vertices.length,
          vertex_count: source_vertices.length,
          section_index: -1,
          world_matrix,
          normal_matrix,
        });

        append_standard_primitive(mesh, group, source_vertices, primitive_data.indices);
      }
    }

    for (const child of node.children) {
      process_node(child);
    }
  };

  for (const node of root_nodes) {
    process_node(node);
  }

  finalize_gltf_build(mesh, gltf_obj, material_cache, primitive_descriptors, sidecar);
}

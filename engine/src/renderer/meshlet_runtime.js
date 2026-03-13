import { vec3, quat, mat3, mat4 } from "gl-matrix";
import { Type2NumOfComponent } from "../utility/gltf_loader.js";
import { read_file_async, read_file_bytes_async } from "../utility/file_system.js";
import { MeshData } from "./mesh_data.js";

const discard_cpu_data = true;
const meshlet_sidecar_promise_cache = new Map();

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
  };
}

function read_primitive_source_data(mesh_api, gltf_obj, primitive) {
  const read_accessor_f32 = mesh_api._read_accessor_f32;

  let positions = [];
  if (primitive.attributes.POSITION !== undefined) {
    positions = read_accessor_f32(primitive.attributes.POSITION);
  }

  let normals = [];
  if (primitive.attributes.NORMAL !== undefined) {
    normals = read_accessor_f32(primitive.attributes.NORMAL);
  }

  let tangents = [];
  if (primitive.attributes.TANGENT !== undefined) {
    tangents = read_accessor_f32(primitive.attributes.TANGENT);
  }

  let bitangents = [];
  if (primitive.attributes.BITANGENT !== undefined) {
    bitangents = read_accessor_f32(primitive.attributes.BITANGENT);
  }

  let colors = [];
  let color_components = 0;
  if (primitive.attributes.COLOR_0 !== undefined) {
    const color_accessor = primitive.attributes.COLOR_0;
    color_components = Type2NumOfComponent[color_accessor.type];
    colors = read_accessor_f32(color_accessor);
  }

  let uvs = [];
  if (primitive.attributes.TEXCOORD_0 !== undefined) {
    uvs = read_accessor_f32(primitive.attributes.TEXCOORD_0);
  }

  if (tangents.length === 0 && positions.length > 0 && uvs.length > 0) {
    const computed = mesh_api._get_tangents_and_bitangents(positions, uvs);
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

    const b = vec3.cross(vec3.create(), n, t_ortho);
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

function append_meshlet_primitive(mesh, group, source_vertices, sidecar, primitive_info, next_meshlet_index) {
  for (let meshlet_offset = 0; meshlet_offset < primitive_info.meshletCount; meshlet_offset++) {
    const meshlet_index = primitive_info.meshletOffset + meshlet_offset;
    const meshlet_record = read_meshlet_record(sidecar, meshlet_index);
    const vertex_base = mesh.vertices.length;
    const debug_meshlet_index = next_meshlet_index++;

    for (let i = 0; i < meshlet_record.vertex_count; i++) {
      const source_vertex_index = sidecar.meshlet_vertices[meshlet_record.vertex_offset + i];
      const source_vertex = source_vertices[source_vertex_index];
      mesh.vertices.push({
        position: source_vertex.position.slice(),
        normal: source_vertex.normal.slice(),
        color: source_vertex.color.slice(),
        uv: source_vertex.uv.slice(),
        tangent: source_vertex.tangent.slice(),
        bitangent: source_vertex.bitangent.slice(),
        extra_data: [0, debug_meshlet_index],
      });
    }

    const triangle_index_count = meshlet_record.triangle_count * 3;
    for (let i = 0; i < triangle_index_count; i++) {
      group.indices.push(vertex_base + sidecar.meshlet_triangles[meshlet_record.triangle_offset + i]);
    }
  }

  return next_meshlet_index;
}

function finalize_gltf_build(mesh_api, mesh, gltf_obj, material_cache) {
  mesh.sections = [];
  mesh.index_count = 0;
  mesh._tmp_indices.length = 0;

  const vertex_section_map = new Int32Array(mesh.vertices.length).fill(-1);

  let running_first_index = 0;
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
      new_section.material_id = mesh_api.make_engine_material_from_gltf(
        gltf_obj,
        mesh,
        gltf_mat,
        group.key,
        material_cache
      );
    }
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

  MeshData.update(mesh);

  if (discard_cpu_data) {
    mesh.vertices = null;
    mesh.indices = null;
    mesh._tmp_indices = null;
    mesh._section_groups = null;
  }
}

export function build_gltf_mesh(mesh_api, mesh, gltf_obj, gltf_mesh, mesh_index = null, sidecar = null) {
  mesh._reset_build_state();

  const material_cache = new Map();
  const resolved_mesh_index = mesh_index ?? resolve_gltf_mesh_index(gltf_obj, gltf_mesh);
  let next_meshlet_index = 0;

  for (let primitive_index = 0; primitive_index < gltf_mesh.primitives.length; primitive_index++) {
    const primitive = gltf_mesh.primitives[primitive_index];
    const material_index = get_gltf_material_index(gltf_obj, primitive);
    let group = mesh._section_groups.get(material_index);
    if (!group) {
      group = { key: material_index, indices: [] };
      mesh._section_groups.set(material_index, group);
    }

    const primitive_data = read_primitive_source_data(mesh_api, gltf_obj, primitive);
    const source_vertices = build_vertices_from_primitive_data(primitive_data);
    const primitive_info = get_meshlet_primitive_info(sidecar, resolved_mesh_index, primitive_index);
    const use_meshlets =
      primitive_info &&
      primitive_info.skipped !== true &&
      primitive_info.mode === 4 &&
      primitive_info.meshletCount > 0 &&
      primitive_info.vertexCount === source_vertices.length;

    if (use_meshlets) {
      next_meshlet_index = append_meshlet_primitive(
        mesh,
        group,
        source_vertices,
        sidecar,
        primitive_info,
        next_meshlet_index
      );
    } else {
      append_standard_primitive(mesh, group, source_vertices, primitive_data.indices);
    }
  }

  finalize_gltf_build(mesh_api, mesh, gltf_obj, material_cache);
}

export function build_combined_gltf_scene(mesh_api, mesh, gltf_obj, scene_index = null, sidecar = null) {
  mesh._reset_build_state();

  const material_cache = new Map();
  let next_meshlet_index = 0;

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

        const primitive_data = read_primitive_source_data(mesh_api, gltf_obj, primitive);
        const source_vertices = build_vertices_from_primitive_data(primitive_data, {
          world_matrix,
          normal_matrix,
        });
        const primitive_info = get_meshlet_primitive_info(sidecar, mesh_index, primitive_index);
        const use_meshlets =
          primitive_info &&
          primitive_info.skipped !== true &&
          primitive_info.mode === 4 &&
          primitive_info.meshletCount > 0 &&
          primitive_info.vertexCount === source_vertices.length;

        if (use_meshlets) {
          next_meshlet_index = append_meshlet_primitive(
            mesh,
            group,
            source_vertices,
            sidecar,
            primitive_info,
            next_meshlet_index
          );
        } else {
          append_standard_primitive(mesh, group, source_vertices, primitive_data.indices);
        }
      }
    }

    for (const child of node.children) {
      process_node(child);
    }
  };

  for (const node of root_nodes) {
    process_node(node);
  }

  finalize_gltf_build(mesh_api, mesh, gltf_obj, material_cache);
}

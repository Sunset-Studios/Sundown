import { glTFLoader } from "../utility/gltf_loader.js";
import { ResourceCache } from "./resource_cache.js";
import { MeshData } from "./mesh_data.js";
import { Name } from "../utility/names.js";
import { CacheTypes, TextureChannel, MaterialFamilyType } from "./renderer_types.js";
import { RenderTaskQueue } from "./task_queues/render_task_queue.js";
import { Type2NumOfComponent } from "../utility/gltf_loader.js";
import { StandardMaterial } from "./material.js";
import {
  build_gltf_mesh,
  finalize_prepared_gltf_mesh_build,
  prepare_gltf_mesh_build,
  create_runtime_meshlet_data_async,
  extract_runtime_positions,
  build_empty_runtime_meshlet_sections,
  load_meshlet_sidecar_async,
} from "./meshlet_runtime.js";
import { get_cooked_sbvh_for_mesh, load_sbvh_sidecar_async } from "../acceleration/sbvh_sidecar.js";

const discard_cpu_data = true;

function get_finite_gltf_number(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function get_gltf_dielectric_reflectance(material) {
  const extensions = material.extensions ?? {};
  const ior_extension = extensions.KHR_materials_ior;
  const specular_extension = extensions.KHR_materials_specular;

  const ior = Math.max(get_finite_gltf_number(ior_extension?.ior, 1.5), 1.0);
  const dielectric_f0 = ((ior - 1.0) / Math.max(ior + 1.0, 1e-4)) ** 2;

  const specular_factor = Math.max(
    get_finite_gltf_number(specular_extension?.specularFactor, 1.0),
    0.0
  );
  const specular_color = specular_extension?.specularColorFactor ?? [1.0, 1.0, 1.0];
  const specular_color_luminance = Math.max(
    0.2126 * get_finite_gltf_number(specular_color[0], 1.0) +
      0.7152 * get_finite_gltf_number(specular_color[1], 1.0) +
      0.0722 * get_finite_gltf_number(specular_color[2], 1.0),
    0.0
  );

  const f0 = Math.min(dielectric_f0 * specular_factor * specular_color_luminance, 1.0);

  // Sundown reconstructs dielectric F0 as 0.16 * reflectance^2 in lighting shaders.
  return Math.min(Math.sqrt(f0 / 0.16), 1.0);
}

export class Mesh {
  static default_min_lod = 0;

  name = "";
  vertices = [];
  packed_vertex_data = null;
  cpu_position_data = null;
  indices = [];
  bounds_min_and_max = [0, 0, 0, 0, 0, 0];
  vertex_buffer_offset = -1;
  index_buffer_offset = -1;
  vertex_count = 0;
  index_count = 0;
  sections = [];
  mesh_data_index = -1;

  meshlet_data = null;
  meshlet_sections = [];
  meshlet_buffer_offset = -1;
  meshlet_vertex_buffer_offset = -1;
  meshlet_triangle_buffer_offset = -1;
  meshlet_group_buffer_offset = -1;
  meshlet_count = 0;
  meshlet_group_count = 0;
  meshlet_lods = [];
  min_lod = Mesh.default_min_lod;
  selected_lod = 0;
  _min_lod_explicit = false;

  pending_loader = null;
  pending_runtime_meshlet_build = null;
  pending_gltf_build_promise = null;
  pending_gltf_build_state = null;
  pending_gltf_meshlet_sidecar = null;
  cooked_sbvh = null;

  _tmp_indices = [];
  _section_groups = new Map();

  _recreate_vertex_bounds() {
    this.bounds_min_and_max = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];

    for (let i = 0; i < this.vertices.length; i++) {
      const vertex = this.vertices[i];
      this.bounds_min_and_max[0] = Math.min(this.bounds_min_and_max[0], vertex.position[0]);
      this.bounds_min_and_max[1] = Math.min(this.bounds_min_and_max[1], vertex.position[1]);
      this.bounds_min_and_max[2] = Math.min(this.bounds_min_and_max[2], vertex.position[2]);
      this.bounds_min_and_max[3] = Math.max(this.bounds_min_and_max[3], vertex.position[0]);
      this.bounds_min_and_max[4] = Math.max(this.bounds_min_and_max[4], vertex.position[1]);
      this.bounds_min_and_max[5] = Math.max(this.bounds_min_and_max[5], vertex.position[2]);
    }
  }

  _reset_build_state() {
    this.vertices.length = 0;
    this.packed_vertex_data = null;
    this.cpu_position_data = null;
    this.indices.length = 0;
    this._tmp_indices.length = 0;
    this._section_groups.clear();
    this.meshlet_data = null;
    this.meshlet_sections.length = 0;
    this.meshlet_buffer_offset = -1;
    this.meshlet_vertex_buffer_offset = -1;
    this.meshlet_triangle_buffer_offset = -1;
    this.meshlet_group_buffer_offset = -1;
    this.meshlet_count = 0;
    this.meshlet_group_count = 0;
    this.meshlet_lods.length = 0;
    this.selected_lod = 0;
    this.pending_runtime_meshlet_build = null;
  }

  /** Sets the project default applied to meshes without an asset or call-site override. */
  static set_default_min_lod(min_lod) {
    this.default_min_lod = Math.max(0, Math.floor(Number(min_lod) || 0));
  }

  get lod_count() {
    return Math.max(1, this.meshlet_lods.length);
  }

  /** Sets this mesh's LOD floor and selects it immediately for manual LOD inspection. */
  set_min_lod(min_lod) {
    this._min_lod_explicit = true;
    this._set_min_lod(min_lod);
  }

  _set_min_lod(min_lod) {
    const requested_lod = Math.max(0, Math.floor(Number(min_lod) || 0));
    this.min_lod =
      this.meshlet_lods.length > 0
        ? Math.min(requested_lod, this.meshlet_lods.length - 1)
        : requested_lod;
    this.select_lod(this.min_lod);
  }

  /** Selects an available LOD without allowing a future runtime policy to cross the floor. */
  select_lod(lod) {
    if (this.meshlet_lods.length === 0) {
      this.selected_lod = Math.max(this.min_lod, Math.floor(Number(lod) || 0));
      return;
    }

    const selected_lod = Math.min(
      Math.max(this.min_lod, Math.floor(Number(lod) || 0)),
      this.meshlet_lods.length - 1
    );
    const lod_data = this.meshlet_lods[selected_lod];
    if (!lod_data) {
      return;
    }

    const changed = this.selected_lod !== selected_lod;
    this.selected_lod = selected_lod;
    this.meshlet_sections = lod_data.sections.map((section) => ({ ...section }));
    if (changed && this.name) {
      RenderTaskQueue.invalidate_mesh(Name.from(this.name));
    }
  }

  _set_meshlet_lods(meshlet_lods, asset_default_min_lod = null) {
    this.meshlet_lods = meshlet_lods;
    if (!this._min_lod_explicit) {
      this._set_min_lod(asset_default_min_lod ?? Mesh.default_min_lod);
    } else {
      this._set_min_lod(this.min_lod);
    }
  }

  static _get_tangents_and_bitangents(positions, uvs) {
    let tangents = [];
    let bitangents = [];

    // Check if inputs are valid
    if (!positions || !uvs || positions.length < 9 || uvs.length < 6) {
      // Return empty arrays if data is insufficient
      return {
        t: new Array(positions?.length || 0).fill(0),
        b: new Array(positions?.length || 0).fill(0),
      };
    }

    // Process each triangle
    for (let i = 0; i < positions.length; i += 9) {
      // Skip incomplete triangles at the end
      if (i + 8 >= positions.length || (i / 3) * 2 + 5 >= uvs.length) {
        break;
      }

      // Get vertices of the triangle
      const v0 = [positions[i], positions[i + 1], positions[i + 2]];
      const v1 = [positions[i + 3], positions[i + 4], positions[i + 5]];
      const v2 = [positions[i + 6], positions[i + 7], positions[i + 8]];

      // Get UVs of the triangle
      const uv0 = [uvs[(i / 3) * 2], uvs[(i / 3) * 2 + 1]];
      const uv1 = [uvs[(i / 3) * 2 + 2], uvs[(i / 3) * 2 + 3]];
      const uv2 = [uvs[(i / 3) * 2 + 4], uvs[(i / 3) * 2 + 5]];

      // Calculate edges of the triangle
      const edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
      const edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

      // Calculate differences in UV space
      const delta_uv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
      const delta_uv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

      // Calculate determinant for UV coordinate system
      const det = delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0];

      // Use default tangent/bitangent for degenerate UV mapping
      let tangent = [1, 0, 0];
      let bitangent = [0, 1, 0];

      // Only compute if determinant is non-zero (avoid division by zero)
      if (Math.abs(det) > 1e-6) {
        const r = 1.0 / det;

        // Calculate tangent
        tangent = [
          (delta_uv2[1] * edge1[0] - delta_uv1[1] * edge2[0]) * r,
          (delta_uv2[1] * edge1[1] - delta_uv1[1] * edge2[1]) * r,
          (delta_uv2[1] * edge1[2] - delta_uv1[1] * edge2[2]) * r,
        ];

        // Calculate bitangent
        bitangent = [
          (delta_uv1[0] * edge2[0] - delta_uv2[0] * edge1[0]) * r,
          (delta_uv1[0] * edge2[1] - delta_uv2[0] * edge1[1]) * r,
          (delta_uv1[0] * edge2[2] - delta_uv2[0] * edge1[2]) * r,
        ];

        // Normalize tangent and bitangent
        const t_len = Math.sqrt(
          tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]
        );
        const b_len = Math.sqrt(
          bitangent[0] * bitangent[0] + bitangent[1] * bitangent[1] + bitangent[2] * bitangent[2]
        );

        if (t_len > 1e-6) {
          tangent = [tangent[0] / t_len, tangent[1] / t_len, tangent[2] / t_len];
        }

        if (b_len > 1e-6) {
          bitangent = [bitangent[0] / b_len, bitangent[1] / b_len, bitangent[2] / b_len];
        }
      }

      // Add calculated tangent and bitangent to the arrays (for each vertex of the triangle)
      tangents.push(...tangent, ...tangent, ...tangent);
      bitangents.push(...bitangent, ...bitangent, ...bitangent);
    }

    // Handle the case where we didn't generate enough data
    while (tangents.length < positions.length) {
      tangents.push(1, 0, 0);
      bitangents.push(0, 1, 0);
    }

    return { t: tangents, b: bitangents };
  }

  static _read_accessor_f32(accessor) {
    const comps = Type2NumOfComponent[accessor.type];
    const stride_bytes = accessor.byteStride || comps * 4;
    const byte_offset = accessor.byteOffset || 0;
    const buffer = accessor.bufferView.data;
    if (stride_bytes === comps * 4) {
      return new Float32Array(buffer, byte_offset, accessor.count * comps);
    }
    const out = new Float32Array(accessor.count * comps);
    const dv = new DataView(buffer);
    for (let i = 0; i < accessor.count; i++) {
      const base = byte_offset + i * stride_bytes;
      for (let c = 0; c < comps; c++) {
        out[i * comps + c] = dv.getFloat32(base + c * 4, true);
      }
    }
    return out;
  }

  static _schedule_runtime_meshlet_build(mesh, source_vertices, source_indices, source_sections) {
    if (!mesh || !source_vertices || !source_indices || source_indices.length < 3) {
      return;
    }

    const positions = extract_runtime_positions(source_vertices);
    const indices =
      source_indices instanceof Uint32Array
        ? source_indices.slice()
        : Uint32Array.from(source_indices);
    const sections =
      source_sections && source_sections.length > 0
        ? source_sections.map((section) => ({
            first_index: section.first_index ?? 0,
            index_count: section.index_count ?? 0,
          }))
        : [{ first_index: 0, index_count: indices.length }];

    mesh.meshlet_sections = build_empty_runtime_meshlet_sections(sections.length);

    const build_token = Symbol("runtime_meshlet_build");
    mesh.pending_runtime_meshlet_build = build_token;

    void create_runtime_meshlet_data_async(positions, indices, sections)
      .then((meshlet_data) => {
        if (mesh.pending_runtime_meshlet_build !== build_token) {
          return;
        }

        mesh.pending_runtime_meshlet_build = null;
        if (!meshlet_data) {
          mesh.meshlet_sections = build_empty_runtime_meshlet_sections(sections.length);
          return;
        }

        mesh.meshlet_data = meshlet_data;
        mesh.meshlet_sections = meshlet_data.sections.map((section) => ({ ...section }));

        MeshData.update(mesh);
        RenderTaskQueue.invalidate_mesh(Name.from(mesh.name));
      })
      .catch((error) => {
        if (mesh.pending_runtime_meshlet_build === build_token) {
          mesh.pending_runtime_meshlet_build = null;
          mesh.meshlet_sections = build_empty_runtime_meshlet_sections(sections.length);
        }
        console.error(
          `[meshlet_runtime] failed to build runtime meshlets for ${mesh.name}:`,
          error
        );
      });
  }

  static create(name, vertices, indices) {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from(name));
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = name;
    mesh.vertices = vertices;
    mesh.indices = new Uint32Array(indices);

    mesh.vertex_count = mesh.vertices.length;
    mesh.index_count = mesh.indices.length;

    mesh._recreate_vertex_bounds();
    mesh.sections = [{ first_index: 0, index_count: mesh.index_count }];
    Mesh._schedule_runtime_meshlet_build(mesh, mesh.vertices, mesh.indices, mesh.sections);

    // Register shared mesh data (bounds)
    MeshData.register(mesh);
    MeshData.update(mesh);

    if (discard_cpu_data) {
      mesh.vertices = null;
      mesh.indices = null;
      mesh._tmp_indices = null;
      mesh._section_groups = null;
    }

    ResourceCache.get().store(CacheTypes.MESH, Name.from(name), mesh);

    RenderTaskQueue.invalidate_mesh(Name.from(name));

    return mesh;
  }

  static quad() {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from("engine_quad"));
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = "engine_quad";
    mesh.vertices = [
      {
        position: [-1, 1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, -1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, -1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1],
        tangent: [1, 0, 0, 0],
        bitangent: [0, -1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, -1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1],
        tangent: [1, 0, 0, 0],
        bitangent: [0, -1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, 1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, -1, 0, 0],
        extra_data: [0, 0],
      },
    ];

    mesh.indices = new Uint32Array([0, 1, 2, 0, 2, 3]);

    mesh.bounds_min_and_max = [
      -1,
      -1,
      -1, // min x, min y, min z
      1,
      1,
      1, // max x, max y, max z
    ];

    mesh.vertex_count = mesh.vertices.length;
    mesh.index_count = mesh.indices.length;

    mesh.sections = [{ first_index: 0, index_count: mesh.index_count }];
    Mesh._schedule_runtime_meshlet_build(mesh, mesh.vertices, mesh.indices, mesh.sections);

    // Register shared mesh data (bounds)
    MeshData.register(mesh);
    MeshData.update(mesh);

    if (discard_cpu_data) {
      mesh.vertices = null;
      mesh.indices = null;
      mesh._tmp_indices = null;
      mesh._section_groups = null;
    }

    ResourceCache.get().store(CacheTypes.MESH, Name.from("engine_quad"), mesh);

    RenderTaskQueue.invalidate_mesh(Name.from("engine_quad"));

    return mesh;
  }

  static cube() {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from("engine_cube"));
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = "engine_cube";
    mesh.vertices = [
      // Front face
      {
        position: [-1, -1, 1, 1],
        normal: [0, 0, 1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, -1, 1, 1],
        normal: [0, 0, 1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, 1, 1, 1],
        normal: [0, 0, 1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, 1, 1, 1],
        normal: [0, 0, 1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },

      // Back face
      {
        position: [1, -1, -1, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0],
        tangent: [-1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, -1, -1, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0],
        tangent: [-1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, 1, -1, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1],
        tangent: [-1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, 1, -1, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1],
        tangent: [-1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },

      // Top face
      {
        position: [-1, 1, 1, 1],
        normal: [0, 1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, 1, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, 1, 1, 1],
        normal: [0, 1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, 1, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, 1, -1, 1],
        normal: [0, 1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, 1, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, 1, -1, 1],
        normal: [0, 1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, 1, 0],
        extra_data: [0, 0],
      },

      // Bottom face
      {
        position: [-1, -1, -1, 1],
        normal: [0, -1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, -1, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, -1, -1, 1],
        normal: [0, -1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, -1, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, -1, 1, 1],
        normal: [0, -1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, -1, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, -1, 1, 1],
        normal: [0, -1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, -1, 0],
        extra_data: [0, 0],
      },

      // Right face
      {
        position: [1, -1, 1, 1],
        normal: [1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0],
        tangent: [0, 0, -1, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, -1, -1, 1],
        normal: [1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0],
        tangent: [0, 0, -1, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, 1, -1, 1],
        normal: [1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1],
        tangent: [0, 0, -1, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [1, 1, 1, 1],
        normal: [1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1],
        tangent: [0, 0, -1, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },

      // Left face
      {
        position: [-1, -1, -1, 1],
        normal: [-1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0],
        tangent: [0, 0, 1, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, -1, 1, 1],
        normal: [-1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0],
        tangent: [0, 0, 1, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, 1, 1, 1],
        normal: [-1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1],
        tangent: [0, 0, 1, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
      {
        position: [-1, 1, -1, 1],
        normal: [-1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1],
        tangent: [0, 0, 1, 0],
        bitangent: [0, 1, 0, 0],
        extra_data: [0, 0],
      },
    ];

    mesh.indices = new Uint32Array([
      0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16, 17, 18,
      18, 19, 16, 20, 21, 22, 22, 23, 20,
    ]);
    mesh.bounds_min_and_max = [
      -1,
      -1,
      -1, // min x, min y, min z
      1,
      1,
      1, // max x, max y, max z
    ];

    mesh.vertex_count = mesh.vertices.length;
    mesh.index_count = mesh.indices.length;

    mesh.sections = [{ first_index: 0, index_count: mesh.index_count }];
    Mesh._schedule_runtime_meshlet_build(mesh, mesh.vertices, mesh.indices, mesh.sections);

    // Register shared mesh data (bounds)
    MeshData.register(mesh);
    MeshData.update(mesh);

    if (discard_cpu_data) {
      mesh.vertices = null;
      mesh.indices = null;
      mesh._tmp_indices = null;
      mesh._section_groups = null;
    }

    ResourceCache.get().store(CacheTypes.MESH, Name.from("engine_cube"), mesh);

    RenderTaskQueue.invalidate_mesh(Name.from("engine_cube"));

    return mesh;
  }

  static sphere() {
    return this.from_gltf("engine/models/sphere/sphere.gltf");
  }

  static from_gltf(gltf_path, mesh_index = 0, options = {}) {
    const key_name = `${gltf_path}#mesh_${mesh_index}`;
    const cache_key = Name.from(key_name);
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, cache_key);
    if (mesh) {
      if (options.min_lod !== undefined) {
        mesh.set_min_lod(options.min_lod);
      }
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = key_name;
    if (options.min_lod !== undefined) {
      mesh.set_min_lod(options.min_lod);
    }

    MeshData.register(mesh);

    const sidecar_promise = load_meshlet_sidecar_async(gltf_path);
    const sbvh_sidecar_promise = load_sbvh_sidecar_async(gltf_path);
    const loader = new glTFLoader();
    mesh.pending_loader = loader;

    loader.load(gltf_path, (gltf_obj) => {
      void (async () => {
        const target_mesh = gltf_obj.meshes[mesh_index] ?? gltf_obj.meshes[0];
        if (!target_mesh) {
          if (mesh.pending_loader === loader) {
            mesh.pending_loader = null;
          }
          return;
        }

        const resolved_mesh_index = target_mesh.meshID ?? gltf_obj.meshes.indexOf(target_mesh);
        const [sidecar, sbvh_sidecar] = await Promise.all([sidecar_promise, sbvh_sidecar_promise]);
        if (mesh.pending_loader !== loader) {
          return;
        }

        mesh.cooked_sbvh = get_cooked_sbvh_for_mesh(sbvh_sidecar, resolved_mesh_index);
        build_gltf_mesh(mesh, gltf_obj, target_mesh, resolved_mesh_index, sidecar);
        if (mesh.pending_loader !== loader) {
          return;
        }

        mesh.pending_loader = null;
        RenderTaskQueue.invalidate_mesh(cache_key);
      })().catch((error) => {
        if (mesh.pending_loader === loader) {
          mesh.pending_loader = null;
        }
        console.error(`[meshlet_runtime] failed to build ${gltf_path}:`, error);
      });
    });

    ResourceCache.get().store(CacheTypes.MESH, cache_key, mesh);

    return mesh;
  }

  static from_parsed_gltf_mesh(gltf_path, gltf_obj, gltf_mesh, cache_key_suffix = "") {
    const key_name = `${gltf_path}#mesh_${gltf_mesh.meshID}${cache_key_suffix}`;
    const cache_key = Name.from(key_name);
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, cache_key);
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = key_name;

    MeshData.register(mesh);

    const pending_load = Symbol(key_name);
    const resolved_mesh_index = gltf_mesh.meshID ?? gltf_obj.meshes.indexOf(gltf_mesh);
    mesh.pending_loader = pending_load;
    mesh.pending_gltf_build_promise = Promise.all([
      load_meshlet_sidecar_async(gltf_path),
      load_sbvh_sidecar_async(gltf_path),
    ])
      .then(([sidecar, sbvh_sidecar]) => {
        if (mesh.pending_loader !== pending_load) {
          return;
        }

        mesh.cooked_sbvh = get_cooked_sbvh_for_mesh(sbvh_sidecar, resolved_mesh_index);
        mesh.pending_gltf_meshlet_sidecar = sidecar;
        mesh.pending_gltf_build_state = prepare_gltf_mesh_build(
          mesh,
          gltf_obj,
          gltf_mesh,
          resolved_mesh_index
        );
      })
      .catch((error) => {
        if (mesh.pending_loader === pending_load) {
          mesh.pending_loader = null;
        }
        mesh.pending_gltf_build_promise = null;
        mesh.pending_gltf_build_state = null;
        mesh.pending_gltf_meshlet_sidecar = null;
        console.error(`[meshlet_runtime] failed to build parsed mesh ${gltf_path}:`, error);
      });

    ResourceCache.get().store(CacheTypes.MESH, cache_key, mesh);

    return mesh;
  }

  static finalize_prepared_gltf_mesh(mesh, gltf_obj) {
    if (!mesh?.pending_gltf_build_state) {
      return;
    }

    finalize_prepared_gltf_mesh_build(
      mesh,
      gltf_obj,
      mesh.pending_gltf_build_state,
      mesh.pending_gltf_meshlet_sidecar
    );

    mesh.pending_loader = null;
    mesh.pending_gltf_build_promise = null;
    mesh.pending_gltf_build_state = null;
    mesh.pending_gltf_meshlet_sidecar = null;

    RenderTaskQueue.invalidate_mesh(Name.from(mesh.name));
  }

  static precrete_engine_primitives() {
    Mesh.cube();
    Mesh.quad();
  }

  static make_engine_material_from_gltf(gltf, mesh, mat, mat_index) {
    const material_scope = `${mesh.name}#mat_${mat_index}`;
    const mat_name = mat.name ? `${material_scope}_${mat.name}` : material_scope;
    const alpha_mode = mat.alphaMode || "OPAQUE";
    const family =
      alpha_mode === "BLEND" ? MaterialFamilyType.Transparent : MaterialFamilyType.Opaque;
    const alpha_cutoff = alpha_mode === "MASK" ? (mat.alphaCutoff ?? 0.5) : 0.0;

    const std = StandardMaterial.create(
      mat_name,
      {},
      {
        family,
        alpha_masked: alpha_mode === "MASK",
        raster_state: {
          cull_mode: mat.doubleSided ? "none" : "back",
        },
      }
    );
    std.set_alpha_cutoff(alpha_cutoff);

    // Base color
    const base = mat.pbrMetallicRoughness;
    const base_color = base?.baseColorFactor || [1, 1, 1, 1];
    let albedo_tex = null;
    if (base?.baseColorTexture) {
      const texture_index = base.baseColorTexture.index;
      const tex = gltf.textures[texture_index];
      const src = tex?.base;
      if (src) {
        const texture_name = `${mesh.name}#texture_${texture_index}_albedo`;
        albedo_tex = {
          paths: [src],
          name: texture_name,
          format: "rgba8unorm",
          dimension: "2d",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: texture_name,
        };
      }
    }

    if (albedo_tex) {
      std.sample_albedo(albedo_tex);
    } else {
      std.set_albedo(base_color);
    }

    // Normal map
    let normal_tex = null;
    if (mat.normalTexture) {
      const texture_index = mat.normalTexture.index;
      const tex = gltf.textures[texture_index];
      const src = tex?.base;
      if (src) {
        const texture_name = `${mesh.name}#texture_${texture_index}_normal`;
        normal_tex = {
          paths: [src],
          name: texture_name,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: texture_name,
        };
      }
    }

    if (normal_tex) {
      std.sample_normal(normal_tex);
    } else {
      std.set_normal([0, 1, 0, 1]);
    }

    // Metallic-Roughness texture: G=roughness, B=metallic
    const roughness_val = base?.roughnessFactor ?? 1.0;
    const metallic_val = base?.metallicFactor ?? 1.0;
    let r_tex = null;
    let m_tex = null;
    if (base?.metallicRoughnessTexture) {
      const texture_index = base.metallicRoughnessTexture.index;
      const tex = gltf.textures[texture_index];
      const src = tex?.base;
      if (src) {
        const roughness_name = `${mesh.name}#texture_${texture_index}_roughness`;
        const metallic_name = `${mesh.name}#texture_${texture_index}_metallic`;
        r_tex = {
          paths: [src],
          name: roughness_name,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: roughness_name,
        };
        m_tex = {
          paths: [src],
          name: metallic_name,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: metallic_name,
        };
      }
    }
    // glTF convention: roughness in B, metallic in G
    if (r_tex) {
      std.sample_roughness(r_tex, TextureChannel.G);
    } else {
      std.set_roughness(roughness_val);
    }
    if (m_tex) {
      std.sample_metallic(m_tex, TextureChannel.B);
    } else {
      std.set_metallic(metallic_val);
    }

    // Preserve glTF dielectric Fresnel strength for direct lighting and SSR hit confidence.
    std.set_specular(get_gltf_dielectric_reflectance(mat));
    const specular_texture_info = mat.extensions?.KHR_materials_specular?.specularTexture;
    if (specular_texture_info) {
      const texture_index = specular_texture_info.index;
      const tex = gltf.textures[texture_index];
      const src = tex?.base;
      if (src) {
        const texture_name = `${mesh.name}#texture_${texture_index}_specular`;
        std.sample_specular(
          {
            paths: [src],
            name: texture_name,
            format: "rgba8unorm",
            usage:
              GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.COPY_DST |
              GPUTextureUsage.RENDER_ATTACHMENT,
            flip_y: false,
            material_notifier: texture_name,
          },
          TextureChannel.A,
          true
        );
      }
    }

    // Ambient occlusion (R channel), strength scales AO value
    let ao_tex = null;
    let ao_strength = 1.0;
    if (mat.occlusionTexture) {
      const texture_index = mat.occlusionTexture.index;
      const tex = gltf.textures[texture_index];
      const src = tex?.base;
      if (src) {
        const texture_name = `${mesh.name}#texture_${texture_index}_ao`;
        ao_tex = {
          paths: [src],
          name: texture_name,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: texture_name,
        };
      }
      ao_strength = mat.occlusionTexture.strength ?? 1.0;
    }

    if (ao_tex) {
      std.sample_ao(ao_tex, TextureChannel.R);
    } else {
      std.set_ao(ao_strength);
    }

    // Emissive: approximate scalar intensity from factor; texture sampled R channel
    let emissive_tex = null;
    const ef = mat.emissiveFactor || [0.0, 0.0, 0.0];
    let emissive_scalar = (ef[0] + ef[1] + ef[2]) / 3.0;
    emissive_scalar = Math.max(emissive_scalar, 0.0);
    if (mat.emissiveTexture) {
      const texture_index = mat.emissiveTexture.index;
      const tex = gltf.textures[texture_index];
      const src = tex?.base;
      if (src) {
        const texture_name = `${mesh.name}#texture_${texture_index}_emissive`;
        emissive_tex = {
          paths: [src],
          name: texture_name,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: texture_name,
        };
      }
    }

    // Keep the scalar alongside the optional texture so glTF emissiveFactor and
    // KHR_materials_emissive_strength modulate textured emitters as specified.
    std.set_emission(emissive_scalar);
    if (emissive_tex) {
      std.sample_emission(emissive_tex);
    }

    return std.material_id;
  }
}

import { glTFLoader } from "../utility/gltf_loader.js";
import { ResourceCache } from "./resource_cache.js";
import { MeshData } from "./mesh_data.js";
import { Name } from "../utility/names.js";
import { CacheTypes, TextureChannel, MaterialFamilyType } from "./renderer_types.js";
import { MeshTaskQueue } from "./mesh_task_queue.js";
import { Type2NumOfComponent } from "../utility/gltf_loader.js";
import { StandardMaterial } from "./material.js";
import {
  build_gltf_mesh,
  build_combined_gltf_scene,
  create_runtime_meshlet_data_async,
  extract_runtime_positions,
  build_empty_runtime_meshlet_sections,
  load_meshlet_sidecar_async,
} from "./meshlet_runtime.js";

const discard_cpu_data = true;

export class Mesh {
  name = "";
  vertices = [];
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

  pending_loader = null;
  pending_runtime_meshlet_build = null;
  triangle_bvh = null;

  _tmp_indices = [];
  _section_groups = new Map();

  _recreate_vertex_bounds() {
    this.bounds_min_and_max = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];

    for (let i = 0; i < this.vertices.length; i += 3) {
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
    this.vertices = [];
    this.indices = [];
    this._tmp_indices = [];
    this._section_groups = new Map();
    this.meshlet_data = null;
    this.meshlet_sections = [];
    this.meshlet_buffer_offset = -1;
    this.meshlet_vertex_buffer_offset = -1;
    this.meshlet_triangle_buffer_offset = -1;
    this.meshlet_group_buffer_offset = -1;
    this.meshlet_count = 0;
    this.meshlet_group_count = 0;
    this.pending_runtime_meshlet_build = null;
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

  static build_from_gltf_mesh(mesh, gltf_obj, gltf_mesh, mesh_index = null, sidecar = null) {
    build_gltf_mesh(mesh, gltf_obj, gltf_mesh, mesh_index, sidecar);
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
        MeshTaskQueue.invalidate_mesh(Name.from(mesh.name));
      })
      .catch((error) => {
        if (mesh.pending_runtime_meshlet_build === build_token) {
          mesh.pending_runtime_meshlet_build = null;
          mesh.meshlet_sections = build_empty_runtime_meshlet_sections(sections.length);
        }
        console.error(`[meshlet_runtime] failed to build runtime meshlets for ${mesh.name}:`, error);
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

    MeshTaskQueue.invalidate_mesh(Name.from(name));

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

    MeshTaskQueue.invalidate_mesh(Name.from("engine_quad"));

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

    MeshTaskQueue.invalidate_mesh(Name.from("engine_cube"));

    return mesh;
  }

  static sphere() {
    return this.from_gltf("engine/models/sphere/sphere.gltf");
  }

  static from_gltf(gltf_path, mesh_index = 0) {
    const key_name = `${gltf_path}#mesh_${mesh_index}`;
    const cache_key = Name.from(key_name);
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, cache_key);
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = key_name;

    MeshData.register(mesh);

    const sidecar_promise = load_meshlet_sidecar_async(gltf_path);
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
        const sidecar = await sidecar_promise;
        if (mesh.pending_loader !== loader) {
          return;
        }

        Mesh.build_from_gltf_mesh(mesh, gltf_obj, target_mesh, resolved_mesh_index, sidecar);
        if (mesh.pending_loader !== loader) {
          return;
        }

        mesh.pending_loader = null;
        MeshTaskQueue.invalidate_mesh(cache_key);
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

  /**
   * Loads an entire GLTF scene as a single combined mesh.
   * All meshes from all nodes are merged, with node transforms baked into vertices.
   *
   * @param {string} gltf_path - Path to the GLTF file
   * @param {number|null} scene_index - Which scene to load (null = default scene)
   * @returns {Mesh} The combined mesh
   */
  static from_gltf_scene(gltf_path, scene_index = null) {
    const key_name = `${gltf_path}#combined_scene_${scene_index ?? "default"}`;
    const cache_key = Name.from(key_name);
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, cache_key);
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = key_name;

    MeshData.register(mesh);

    const sidecar_promise = load_meshlet_sidecar_async(gltf_path);
    const loader = new glTFLoader();
    mesh.pending_loader = loader;
    
    loader.load(gltf_path, (gltf_obj) => {
      void (async () => {
        const sidecar = await sidecar_promise;
        if (mesh.pending_loader !== loader) {
          return;
        }

        Mesh.build_combined_gltf_scene(mesh, gltf_obj, scene_index, sidecar);
        if (mesh.pending_loader !== loader) {
          return;
        }

        mesh.pending_loader = null;
        MeshTaskQueue.invalidate_mesh(cache_key);
      })().catch((error) => {
        if (mesh.pending_loader === loader) {
          mesh.pending_loader = null;
        }
        console.error(`[meshlet_runtime] failed to build combined scene ${gltf_path}:`, error);
      });
    });

    ResourceCache.get().store(CacheTypes.MESH, cache_key, mesh);

    return mesh;
  }

  /**
   * Builds a combined mesh from all nodes in a GLTF scene.
   * Node transforms are baked into vertex positions and normals.
   *
   * @param {Mesh} mesh - The mesh to build into
   * @param {Object} gltf_obj - The parsed GLTF object
   * @param {number|null} scene_index - Which scene to use
   */
  static build_combined_gltf_scene(mesh, gltf_obj, scene_index = null, sidecar = null) {
    build_combined_gltf_scene(mesh, gltf_obj, scene_index, sidecar);
  }

  static from_parsed_gltf_mesh(gltf_path, gltf_obj, gltf_mesh) {
    const key_name = `${gltf_path}#mesh_${gltf_mesh.meshID}`;
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
    void load_meshlet_sidecar_async(gltf_path)
      .then((sidecar) => {
        if (mesh.pending_loader !== pending_load) {
          return;
        }

        Mesh.build_from_gltf_mesh(mesh, gltf_obj, gltf_mesh, resolved_mesh_index, sidecar);
        if (mesh.pending_loader !== pending_load) {
          return;
        }

        mesh.pending_loader = null;
        MeshTaskQueue.invalidate_mesh(cache_key);
      })
      .catch((error) => {
        if (mesh.pending_loader === pending_load) {
          mesh.pending_loader = null;
        }
        console.error(`[meshlet_runtime] failed to build parsed mesh ${gltf_path}:`, error);
      });

    ResourceCache.get().store(CacheTypes.MESH, cache_key, mesh);

    return mesh;
  }

  static precrete_engine_primitives() {
    Mesh.cube();
    Mesh.quad();
  }

  static make_engine_material_from_gltf(gltf, mesh, mat, mat_index, material_cache) {
    if (material_cache && material_cache.has(mat_index)) {
      return material_cache.get(mat_index);
    }

    const material_scope = `${mesh.name}#mat_${mat_index}`;
    const mat_name = mat.name ? `${material_scope}_${mat.name}` : material_scope;
    const alpha_mode = mat.alphaMode || "OPAQUE";
    const family =
      alpha_mode === "BLEND" ? MaterialFamilyType.Transparent : MaterialFamilyType.Opaque;
    const alpha_cutoff = alpha_mode === "MASK" ? (mat.alphaCutoff ?? 0.5) : 0.0;

    const std = StandardMaterial.create(
      mat_name,
      {},
      { family, raster_state: { cull_mode: "none" } }
    );
    std.set_alpha_cutoff(alpha_cutoff);

    // Base color
    const base = mat.pbrMetallicRoughness;
    const base_color = base?.baseColorFactor || [1, 1, 1, 1];
    let albedo_tex = null;
    if (base?.baseColorTexture) {
      const tex = gltf.textures[base.baseColorTexture.index];
      const src = tex?.base;
      if (src) {
        albedo_tex = {
          paths: [src],
          name: `${mat_name}_albedo`,
          format: "rgba8unorm",
          dimension: "2d",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: `${mat_name}_albedo`,
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
      const tex = gltf.textures[mat.normalTexture.index];
      const src = tex?.base;
      if (src) {
        normal_tex = {
          paths: [src],
          name: `${mat_name}_normal`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: `${mat_name}_normal`,
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
      const tex = gltf.textures[base.metallicRoughnessTexture.index];
      const src = tex?.base;
      if (src) {
        r_tex = {
          paths: [src],
          name: `${mat_name}_roughness`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: `${mat_name}_roughness`,
        };
        m_tex = {
          paths: [src],
          name: `${mat_name}_metallic`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: `${mat_name}_metallic`,
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

    // Ambient occlusion (R channel), strength scales AO value
    let ao_tex = null;
    let ao_strength = 1.0;
    if (mat.occlusionTexture) {
      const tex = gltf.textures[mat.occlusionTexture.index];
      const src = tex?.base;
      if (src) {
        ao_tex = {
          paths: [src],
          name: `${mat_name}_ao`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: `${mat_name}_ao`,
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
      const tex = gltf.textures[mat.emissiveTexture.index];
      const src = tex?.base;
      if (src) {
        emissive_tex = {
          paths: [src],
          name: `${mat_name}_emissive`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
          material_notifier: `${mat_name}_emissive`,
        };
      }
    }

    if (emissive_tex) {
      std.sample_emission(emissive_tex);
    } else {
      std.set_emission(emissive_scalar);
    }

    if (material_cache) {
      material_cache.set(mat_index, std.material_id);
    }

    return std.material_id;
  }
}

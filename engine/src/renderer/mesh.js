import { glTFLoader } from "../utility/gltf_loader.js";
import { ResourceCache } from "./resource_cache.js";
import { MeshData } from "./mesh_data.js";
import { Buffer } from "./buffer.js";
import { Name } from "../utility/names.js";
import { CacheTypes, TextureChannel, MaterialFamilyType } from "./renderer_types.js";
import { MeshTaskQueue } from "./mesh_task_queue.js";
import { vec3, mat3, mat4 } from "gl-matrix";
import { Type2NumOfComponent } from "../utility/gltf_loader.js";
import { StandardMaterial } from "./material.js";
import { Texture } from "./texture.js";

const discard_cpu_data = true;

export class Mesh {
  name = "";
  vertices = [];
  indices = [];
  bounds_min_and_max = [0, 0, 0, 0, 0, 0];
  vertex_buffer_offset = -1;
  vertex_count = 0;
  index_count = 0;
  sections = [];
  mesh_data_index = -1;

  index_buffer = null;
  pending_loader = null;
  triangle_bvh = null;

  _tmp_indices = [];
  _section_groups = new Map();

  _recreate_index_buffer() {
    let element_type = "uint16";
    if (this.indices.constructor.name === "Uint32Array") {
      element_type = "uint32";
    }

    this.index_buffer = Buffer.create({
      name: `${this.name}_index_buffer`,
      raw_data: this.indices,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      element_type: element_type,
    });
  }

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

  static create(name, vertices, indices) {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from(name));
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = name;
    mesh.vertices = vertices;
    mesh.indices = new Uint16Array(indices);
    mesh.triangle_bvh = new TriangleBVH(vertices, indices);

    mesh.vertex_count = mesh.vertices.length;
    mesh.index_count = mesh.indices.length;

    mesh._recreate_vertex_bounds();
    mesh._recreate_index_buffer();
    mesh.sections = [{ first_index: 0, index_count: mesh.index_count }];

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
        uv: [0, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, -1, 0, 0],
      },
      {
        position: [-1, -1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, -1, 0, 0],
      },
      {
        position: [1, -1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, -1, 0, 0],
      },
      {
        position: [1, 1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, -1, 0, 0],
      },
    ];

    mesh.indices = new Uint16Array([0, 1, 2, 0, 2, 3]);

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

    mesh.triangle_bvh = new TriangleBVH(mesh.vertices, mesh.indices);

    mesh._recreate_index_buffer();
    mesh.sections = [{ first_index: 0, index_count: mesh.index_count }];

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
        uv: [0, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [1, -1, 1, 1],
        normal: [0, 0, 1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [1, 1, 1, 1],
        normal: [0, 0, 1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [-1, 1, 1, 1],
        normal: [0, 0, 1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },

      // Back face
      {
        position: [1, -1, -1, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0, 0, 0],
        tangent: [-1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [-1, -1, -1, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0, 0, 0],
        tangent: [-1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [-1, 1, -1, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1, 0, 0],
        tangent: [-1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [1, 1, -1, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1, 0, 0],
        tangent: [-1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },

      // Top face
      {
        position: [-1, 1, 1, 1],
        normal: [0, 1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, 1, 0],
      },
      {
        position: [1, 1, 1, 1],
        normal: [0, 1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, 1, 0],
      },
      {
        position: [1, 1, -1, 1],
        normal: [0, 1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, 1, 0],
      },
      {
        position: [-1, 1, -1, 1],
        normal: [0, 1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, 1, 0],
      },

      // Bottom face
      {
        position: [-1, -1, -1, 1],
        normal: [0, -1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, -1, 0],
      },
      {
        position: [1, -1, -1, 1],
        normal: [0, -1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, -1, 0],
      },
      {
        position: [1, -1, 1, 1],
        normal: [0, -1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, -1, 0],
      },
      {
        position: [-1, -1, 1, 1],
        normal: [0, -1, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 0, -1, 0],
      },

      // Right face
      {
        position: [1, -1, 1, 1],
        normal: [1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0, 0, 0],
        tangent: [0, 0, -1, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [1, -1, -1, 1],
        normal: [1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0, 0, 0],
        tangent: [0, 0, -1, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [1, 1, -1, 1],
        normal: [1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1, 0, 0],
        tangent: [0, 0, -1, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [1, 1, 1, 1],
        normal: [1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1, 0, 0],
        tangent: [0, 0, -1, 0],
        bitangent: [0, 1, 0, 0],
      },

      // Left face
      {
        position: [-1, -1, -1, 1],
        normal: [-1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0, 0, 0],
        tangent: [0, 0, 1, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [-1, -1, 1, 1],
        normal: [-1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0, 0, 0],
        tangent: [0, 0, 1, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [-1, 1, 1, 1],
        normal: [-1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1, 0, 0],
        tangent: [0, 0, 1, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [-1, 1, -1, 1],
        normal: [-1, 0, 0, 0],
        color: [1, 1, 1, 1],
        uv: [0, 1, 0, 0],
        tangent: [0, 0, 1, 0],
        bitangent: [0, 1, 0, 0],
      },
    ];

    mesh.indices = new Uint16Array([
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

    mesh.triangle_bvh = new TriangleBVH(mesh.vertices, mesh.indices);

    mesh._recreate_index_buffer();
    mesh.sections = [{ first_index: 0, index_count: mesh.index_count }];

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

    return mesh;
  }

  static sphere() {
    return this.from_gltf("engine/models/sphere/sphere.gltf");
  }

  static from_gltf(gltf) {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from(gltf));
    if (mesh) {
      return mesh;
    }

    // Respect interleaved vertex data (byteStride) when reading attributes
    const read_accessor_f32 = (accessor) => {
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
    };

    const parse_node_mesh = (gltf_obj, node) => {
      let vertex_offset = mesh.vertices.length;

      for (const primitive of node.mesh.primitives) {
        // Group by material assignment
        const material_index = primitive.material
          ? gltf_obj.materials.indexOf(primitive.material)
          : -1;
        let group = mesh._section_groups.get(material_index);
        if (!group) {
          group = { key: material_index, indices: [] };
          mesh._section_groups.set(material_index, group);
        }
        let positions = [];
        let position_accessor = null;
        if (primitive.attributes.POSITION !== undefined) {
          position_accessor = primitive.attributes.POSITION;
          positions = read_accessor_f32(position_accessor);
        }

        let normals = [];
        let normal_accessor = null;
        if (primitive.attributes.NORMAL !== undefined) {
          normal_accessor = primitive.attributes.NORMAL;
          normals = read_accessor_f32(normal_accessor);
        }

        let tangents = [];
        let tangent_accessor = null;
        if (primitive.attributes.TANGENT !== undefined) {
          tangent_accessor = primitive.attributes.TANGENT;
          tangents = read_accessor_f32(tangent_accessor);
        }

        let bitangents = [];
        let bitangent_accessor = null;
        if (primitive.attributes.BITANGENT !== undefined) {
          bitangent_accessor = primitive.attributes.BITANGENT;
          bitangents = read_accessor_f32(bitangent_accessor);
        }

        let colors = [];
        let color_components = 0;
        let color_accessor = null;
        if (primitive.attributes.COLOR_0 !== undefined) {
          color_accessor = primitive.attributes.COLOR_0;
          color_components = Type2NumOfComponent[color_accessor.type];
          colors = read_accessor_f32(color_accessor);
        }

        let uvs = [];
        let uv_accessor = null;
        if (primitive.attributes.TEXCOORD_0 !== undefined) {
          uv_accessor = primitive.attributes.TEXCOORD_0;
          uvs = read_accessor_f32(uv_accessor);
        }

        // Compute tangents/bitangents if not provided
        if (tangents.length === 0 && positions.length > 0 && uvs.length > 0) {
          let computed = Mesh._get_tangents_and_bitangents(positions, uvs);
          tangents = computed.t; // flat VEC3
          bitangents = computed.b; // flat VEC3
        } else if (bitangents.length === 0 && tangents.length > 0 && normals.length > 0) {
          bitangents = new Array((tangents.length / 4) * 3);
          for (let vi = 0; vi < tangents.length / 4; vi++) {
            let tangent_start = vi * 4;
            let normal_start = vi * 3;
            let t = [
              tangents[tangent_start],
              tangents[tangent_start + 1],
              tangents[tangent_start + 2],
            ];
            let handedness = tangents[tangent_start + 3] || 1;
            let n = [normals[normal_start], normals[normal_start + 1], normals[normal_start + 2]];
            let b = vec3.cross(vec3.create(), n, t);
            vec3.scale(b, b, handedness);
            vec3.normalize(b, b);
            let bitangent_start = vi * 3;
            bitangents[bitangent_start] = b[0];
            bitangents[bitangent_start + 1] = b[1];
            bitangents[bitangent_start + 2] = b[2];
          }
        }

        // Precompute transform helpers
        const node_matrix = node._world || node.matrix;
        const normal_matrix = mat3.normalFromMat4(mat3.create(), node_matrix);
        const world3_for_det = mat3.fromMat4(mat3.create(), node_matrix);
        const is_mirrored = mat3.determinant(world3_for_det) < 0;

        // Number of unique vertices
        const num_verts = positions.length / 3;

        // Build vertices for this primitive
        for (let k = 0; k < num_verts; k++) {
          let pos_index = k * 3;
          let normal_index = k * 3;
          let uv_index = k * 2;
          let tangent_index = tangents.length % 4 === 0 ? k * 4 : k * 3; // VEC4 if original, VEC3 if computed
          let color_index = k * color_components;

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
          if (tangents.length % 4 === 0) {
            tangent_w = tangents[tangent_index + 3] ?? 1.0;
          }

          // position
          const src_pos_x = positions[pos_index] ?? 0.0;
          const src_pos_y = positions[pos_index + 1] ?? 0.0;
          const src_pos_z = positions[pos_index + 2] ?? 0.0;
          const transformed_pos = vec3.transformMat4(
            vec3.create(),
            vec3.fromValues(src_pos_x, src_pos_y, src_pos_z),
            node_matrix
          );

          // normal (transform by normal matrix)
          const n = vec3.fromValues(
            normals[normal_index] ?? 0.0,
            normals[normal_index + 1] ?? 0.0,
            normals[normal_index + 2] ?? 0.0
          );
          vec3.transformMat3(n, n, normal_matrix);
          vec3.normalize(n, n);

          // tangent (transform by normal matrix and orthonormalize against normal)
          const t = vec3.fromValues(
            tangents[tangent_index] ?? 0.0,
            tangents[tangent_index + 1] ?? 0.0,
            tangents[tangent_index + 2] ?? 0.0
          );
          vec3.transformMat3(t, t, normal_matrix);
          // Gram-Schmidt
          const nt_dot_t = vec3.dot(n, t);
          const t_ortho = vec3.subtract(vec3.create(), t, vec3.scale(vec3.create(), n, nt_dot_t));
          vec3.normalize(t_ortho, t_ortho);

          // bitangent from cross with handedness, accounting for mirrored transforms
          const handedness_sign = tangent_w * (is_mirrored ? -1.0 : 1.0);
          const b = vec3.cross(vec3.create(), n, t_ortho);
          vec3.scale(b, b, handedness_sign);
          vec3.normalize(b, b);

          mesh.vertices.push({
            position: [
              transformed_pos[0] ?? 0.0,
              transformed_pos[1] ?? 0.0,
              transformed_pos[2] ?? 0.0,
              1.0,
            ],
            normal: [n[0], n[1], n[2], 0.0],
            color: color,
            uv: [uvs[uv_index] ?? 0.0, uvs[uv_index + 1] ?? 0.0, 0.0, 0.0],
            tangent: [t_ortho[0], t_ortho[1], t_ortho[2], tangent_w],
            bitangent: [b[0], b[1], b[2], 0.0],
          });
        }

        // Handle indices
        let local_indices = [];
        if (primitive.indices !== undefined) {
          const index_accessor = gltf_obj.accessors[primitive.indices];
          if (index_accessor.componentType === 5123) {
            // UNSIGNED_SHORT
            local_indices = new Uint16Array(
              index_accessor.bufferView.data,
              index_accessor.byteOffset || 0,
              index_accessor.count
            );
          } else if (index_accessor.componentType === 5125) {
            // UNSIGNED_INT
            local_indices = new Uint32Array(
              index_accessor.bufferView.data,
              index_accessor.byteOffset || 0,
              index_accessor.count
            );
          } else if (index_accessor.componentType === 5121) {
            // UNSIGNED_BYTE
            local_indices = new Uint8Array(
              index_accessor.bufferView.data,
              index_accessor.byteOffset || 0,
              index_accessor.count
            );
          }
        } else {
          // No indices, generate sequential
          let count = num_verts;
          local_indices = new Uint32Array(count);
          for (let k = 0; k < count; k++) {
            local_indices[k] = k;
          }
        }

        // Flip winding if node has a mirrored (negative determinant) transform
        if (is_mirrored) {
          for (let i = 0; i + 2 < local_indices.length; i += 3) {
            const i0 = local_indices[i] + vertex_offset;
            const i1 = local_indices[i + 1] + vertex_offset;
            const i2 = local_indices[i + 2] + vertex_offset;
            group.indices.push(i0, i2, i1);
          }
        } else {
          for (let idx of local_indices) {
            group.indices.push(idx + vertex_offset);
          }
        }

        vertex_offset += num_verts;
      }
    };

    mesh = new Mesh();
    mesh.name = gltf;

    MeshData.register(mesh);

    const mesh_id = Name.from(gltf);
    mesh.pending_loader = new glTFLoader();
    mesh.pending_loader.load(gltf, (gltf_obj) => {
      // Build parent links and world matrices
      for (const n of gltf_obj.nodes) {
        for (const c of n.children) {
          c._parent = n;
        }
        n._world = null;
      }

      const get_world_matrix = (n) => {
        if (n._world) return n._world;
        const out = mat4.create();
        if (n._parent) {
          mat4.mul(out, get_world_matrix(n._parent), n.matrix);
        } else {
          mat4.copy(out, n.matrix);
        }
        n._world = out;
        return out;
      };

      for (const node of gltf_obj.nodes) {
        if (node.mesh) {
          // ensure world matrix is computed
          node._world = get_world_matrix(node);
          parse_node_mesh(gltf_obj, node);
        }
      }

      // Build sections by material groups and finalize indices
      mesh.sections = [];
      mesh._tmp_indices.length = 0;

      const material_cache = new Map();
      {
        let running_first_index = 0;
        for (const [, group] of mesh._section_groups) {
          if (group.indices.length === 0) continue;
          let new_section = {
            first_index: running_first_index,
            index_count: group.indices.length,
            material_id: null,
          };
          if (group.key >= 0) {
            const gltf_mat = gltf_obj.materials[group.key];
            new_section.material_id = Mesh.make_engine_material_from_gltf(
              gltf_obj,
              gltf_mat,
              group.key,
              material_cache
            );
          }
          mesh.sections.push(new_section);
          mesh._tmp_indices = mesh._tmp_indices.concat(group.indices);
          running_first_index += group.indices.length;
        }
      }

      if (mesh._tmp_indices.length > 0) {
        let max_index = 0;
        for (let i = 0; i < mesh._tmp_indices.length; i++) {
          if (mesh._tmp_indices[i] > max_index) max_index = mesh._tmp_indices[i];
        }
        if (max_index > 65535) {
          mesh.indices = new Uint32Array(mesh._tmp_indices);
        } else {
          mesh.indices = new Uint16Array(mesh._tmp_indices);
        }
      } else {
        mesh.indices = new Uint16Array(0);
      }

      mesh.vertex_count = mesh.vertices.length;
      mesh.index_count = mesh.indices.length;

      mesh.triangle_bvh = new TriangleBVH(mesh.vertices, mesh.indices);

      mesh._recreate_vertex_bounds();
      mesh._recreate_index_buffer();

      // Register shared mesh data (bounds)
      MeshData.update(mesh);

      if (discard_cpu_data) {
        mesh.vertices = null;
        mesh.indices = null;
        mesh._tmp_indices = null;
        mesh._section_groups = null;
      }

      MeshTaskQueue.invalidate_mesh(mesh_id);
    });

    ResourceCache.get().store(CacheTypes.MESH, mesh_id, mesh);

    return mesh;
  }

  static precrete_engine_primitives() {
    Mesh.cube();
    Mesh.quad();
  }

  static make_engine_material_from_gltf(gltf, mat, mat_index, material_cache) {
    if (material_cache && material_cache.has(mat_index)) {
      return material_cache.get(mat_index);
    }

    const mat_name = mat.name || `${mesh.name}_mat_${mat_index}`;
    const alpha_mode = mat.alphaMode || "OPAQUE";
    const family =
      alpha_mode === "BLEND" ? MaterialFamilyType.Transparent : MaterialFamilyType.Opaque;

    const std = StandardMaterial.create(mat_name, {}, { family });

    // Base color
    const base = mat.pbrMetallicRoughness;
    const base_color = base?.baseColorFactor || [1, 1, 1, 1];
    let albedo_tex = null;
    if (base?.baseColorTexture) {
      const tex = gltf.textures[base.baseColorTexture.index];
      const src = tex?.source?.src;
      if (src) {
        albedo_tex = Texture.load([src], {
          name: `${mat_name}_albedo`,
          format: "rgba8unorm",
          dimension: "2d",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
        });
      }
    }
    std.set_albedo(base_color, albedo_tex);

    // Normal map
    let normal_tex = null;
    if (mat.normalTexture) {
      const tex = gltf.textures[mat.normalTexture.index];
      const src = tex?.source?.src;
      if (src) {
        normal_tex = Texture.load([src], {
          name: `${mat_name}_normal`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
        });
      }
    }
    std.set_normal([0, 0, 1, 1], normal_tex);

    // Metallic-Roughness texture: G=roughness, B=metallic
    const roughness_val = base?.roughnessFactor ?? 1.0;
    const metallic_val = base?.metallicFactor ?? 1.0;
    let mr_tex = null;
    if (base?.metallicRoughnessTexture) {
      const tex = gltf.textures[base.metallicRoughnessTexture.index];
      const src = tex?.source?.src;
      if (src) {
        mr_tex = Texture.load([src], {
          name: `${mat_name}_metallic_roughness`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
        });
      }
    }
    // glTF convention: roughness in G, metallic in B
    std.set_roughness(roughness_val, mr_tex, TextureChannel.G);
    std.set_metallic(metallic_val, mr_tex, TextureChannel.B);

    // Ambient occlusion (R channel), strength scales AO value
    let ao_tex = null;
    let ao_strength = 1.0;
    if (mat.occlusionTexture) {
      const tex = gltf.textures[mat.occlusionTexture.index];
      const src = tex?.source?.src;
      if (src) {
        ao_tex = Texture.load([src], {
          name: `${mat_name}_ao`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
        });
      }
      ao_strength = mat.occlusionTexture.strength ?? 1.0;
    }
    std.set_ao(ao_strength, ao_tex, TextureChannel.R);

    // Emissive: approximate scalar intensity from factor; texture sampled R channel
    let emissive_tex = null;
    const ef = mat.emissiveFactor || [0, 0, 0];
    const emissive_scalar = (ef[0] + ef[1] + ef[2]) / 3.0;
    if (mat.emissiveTexture) {
      const tex = gltf.textures[mat.emissiveTexture.index];
      const src = tex?.source?.src;
      if (src) {
        emissive_tex = Texture.load([src], {
          name: `${mat_name}_emissive`,
          format: "rgba8unorm",
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
          flip_y: false,
        });
      }
    }
    std.set_emission(emissive_scalar, emissive_tex, TextureChannel.R);

    if (material_cache) {
      material_cache.set(mat_index, std.material_id);
    }

    return std.material_id;
  }
}

export class TriangleBVHNode {
  constructor() {
    this.min = [Infinity, Infinity, Infinity];
    this.max = [-Infinity, -Infinity, -Infinity];
    this.left = null;
    this.right = null;
    this.start = 0; // start index into triangle array
    this.count = 0; // number of triangles
  }
}

export class TriangleBVH {
  // The BVH accepts a flat list of vertices and an index buffer that defines the triangle ordering.
  // Internally we transform this into an array of triangles (each triangle is an array of three vec3 positions)
  // so that the rest of the implementation can utilize triangles directly.

  // vertices  - Array of vertex objects OR vec3/vec4 position arrays. If the vertex is an object it must expose a `position` field.
  // indices   - TypedArray/Array containing indices that reference the `vertices` list. Every consecutive triplet forms one triangle.
  constructor(vertices, indices) {
    // Convert vertices/indices to the internal `triangles` representation
    // this.triangles = [];
    // const vertex_count = vertices.length;
    // const index_count = indices.length;
    // // Helper to extract a vec3 position from a vertex entry. Supports either
    // // raw position arrays ([x, y, z, ...]) or objects with a `position` field.
    // const get_pos = (v) => {
    //   if (Array.isArray(v)) {
    //     // If the array has more than 3 components (e.g. vec4) slice the first 3.
    //     return [v[0], v[1], v[2]];
    //   }
    //   if (v && v.position) {
    //     return [v.position[0], v.position[1], v.position[2]];
    //   }
    //   // Fallback – zero vector (should not happen in valid meshes).
    //   return [0, 0, 0];
    // };
    // for (let i = 0; i + 2 < index_count; i += 3) {
    //   const i0 = indices[i];
    //   const i1 = indices[i + 1];
    //   const i2 = indices[i + 2];
    //   if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
    //     // Skip degenerate/out-of-range triangles.
    //     continue;
    //   }
    //   const v0 = get_pos(vertices[i0]);
    //   const v1 = get_pos(vertices[i1]);
    //   const v2 = get_pos(vertices[i2]);
    //   this.triangles.push([v0, v1, v2]);
    // }
    // this.root = this._build(0, this.triangles.length);
  }

  // Axis aligned bounding box for triangle range
  _get_bounds(start, end) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    // for (let i = start; i < end; i++) {
    //   const t = this.triangles[i];
    //   for (let j = 0; j < 3; j++) {
    //     min[0] = Math.min(min[0], t[j][0]);
    //     min[1] = Math.min(min[1], t[j][1]);
    //     min[2] = Math.min(min[2], t[j][2]);
    //     max[0] = Math.max(max[0], t[j][0]);
    //     max[1] = Math.max(max[1], t[j][1]);
    //     max[2] = Math.max(max[2], t[j][2]);
    //   }
    // }
    return { min, max };
  }

  _build(start, end) {
    const node = new TriangleBVHNode();
    // node.start = start;
    // node.count = end - start;

    // const bounds = this._get_bounds(start, end);
    // node.min = bounds.min;
    // node.max = bounds.max;

    // if (end - start <= 4) {
    //   return node; // leaf
    // }

    // // choose split axis
    // const size = [
    //   bounds.max[0] - bounds.min[0],
    //   bounds.max[1] - bounds.min[1],
    //   bounds.max[2] - bounds.min[2],
    // ];
    // let axis = 0;
    // if (size[1] > size[0]) axis = 1;
    // if (size[2] > size[axis]) axis = 2;

    // // sort triangles by centroid along axis
    // const mid = (start + end) >> 1;
    // this.triangles
    //   .slice(start, end)
    //   .sort((a, b) => {
    //     const ca = (a[0][axis] + a[1][axis] + a[2][axis]) / 3;
    //     const cb = (b[0][axis] + b[1][axis] + b[2][axis]) / 3;
    //     return ca - cb;
    //   })
    //   .forEach((t, i) => {
    //     this.triangles[start + i] = t;
    //   });

    // node.left = this._build(start, mid);
    // node.right = this._build(mid, end);

    return node;
  }

  // flatten BVH into arrays for GPU consumption
  flatten() {
    const nodes = [];
    // const stack = [{ node: this.root, parent: -1 }];
    // while (stack.length) {
    //   const { node, parent } = stack.pop();
    //   const index = nodes.length;
    //   nodes.push({
    //     min: node.min,
    //     max: node.max,
    //     left: -1,
    //     right: -1,
    //     start: node.start,
    //     count: node.count,
    //     parent,
    //   });
    //   if (node.right) stack.push({ node: node.right, parent: index });
    //   if (node.left) stack.push({ node: node.left, parent: index });
    // }
    // // update child indices
    // for (let i = 0; i < nodes.length; i++) {
    //   const n = nodes[i];
    //   if (n.left !== -1) continue; // already set
    //   const left_child = nodes.findIndex((x) => x.parent === i && x !== n);
    //   const right_child = nodes.findIndex((x, idx) => x.parent === i && idx !== left_child);
    //   n.left = left_child;
    //   n.right = right_child;
    // }
    return nodes;
  }
}

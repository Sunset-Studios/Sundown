import { glTFLoader } from "../utility/gltf_loader.js";
import { ResourceCache } from "./resource_cache.js";
import { SharedVertexBuffer } from "../core/shared_data.js";
import { Buffer } from "./buffer.js";
import { Name } from "../utility/names.js";
import { CacheTypes } from "./renderer_types.js";

const discard_cpu_data = true;

export class Mesh {
  static loading_meshes = new Map();

  name = "";
  vertices = [];
  indices = [];
  bounds_min_and_max = [0, 0, 0, 0, 0, 0];
  vertex_buffer_offset = 0;
  vertex_count = 0;
  index_count = 0;

  index_buffer = null;
  pending_loader = null;

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

  _get_vertex_bounds(vertices) {
    let min_x = Infinity;
    let min_y = Infinity;
    let min_z = Infinity;
    let max_x = -Infinity;
    let max_y = -Infinity;
    let max_z = -Infinity;

    for (const vertex of vertices) {
      const position = vertex.position;

      min_x = Math.min(min_x, position[0]);
      min_y = Math.min(min_y, position[1]);
      min_z = Math.min(min_z, position[2]);

      max_x = Math.max(max_x, position[0]);
      max_y = Math.max(max_y, position[1]);
      max_z = Math.max(max_z, position[2]);
    }

    return [min_x, min_y, min_z, max_x, max_y, max_z];
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
    mesh.bounds_min_and_max = mesh._get_vertex_bounds(vertices);

    mesh.vertex_buffer_offset = SharedVertexBuffer.add_vertex_data(mesh.vertices);

    mesh.vertex_count = mesh.vertices.length;
    mesh.index_count = mesh.indices.length;

    mesh._recreate_index_buffer();

    if (discard_cpu_data) {
      mesh.vertices = null;
      mesh.indices = null;
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
      0, // min x, min y, min z
      1,
      1,
      0, // max x, max y, max z
    ];

    mesh.vertex_buffer_offset = SharedVertexBuffer.add_vertex_data(mesh.vertices);

    mesh.vertex_count = mesh.vertices.length;
    mesh.index_count = mesh.indices.length;

    mesh._recreate_index_buffer();

    if (discard_cpu_data) {
      mesh.vertices = null;
      mesh.indices = null;
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

    mesh.vertex_buffer_offset = SharedVertexBuffer.add_vertex_data(mesh.vertices);

    mesh.vertex_count = mesh.vertices.length;
    mesh.index_count = mesh.indices.length;

    mesh._recreate_index_buffer();

    if (discard_cpu_data) {
      mesh.vertices = null;
      mesh.indices = null;
    }

    ResourceCache.get().store(CacheTypes.MESH, Name.from("engine_cube"), mesh);

    return mesh;
  }

  static from_gltf(gltf) {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from(gltf));
    if (mesh) {
      return mesh;
    }

    const parse_node_mesh = (gltf_obj, node) => {
      for (const primitive of node.mesh.primitives) {
        if (primitive.indices) {
          if (primitive.indicesComponentType === 5122 || primitive.indicesComponentType === 5123) {
            mesh.indices = new Uint16Array(gltf_obj.accessors[primitive.indices].bufferView.data);
          } else if (primitive.indicesComponentType === 5125) {
            mesh.indices = new Uint32Array(gltf_obj.accessors[primitive.indices].bufferView.data);
          }
        }
        let positions = [];
        if (primitive.attributes.POSITION) {
          positions = new Float32Array(primitive.attributes.POSITION.bufferView.data);
        }
        let normals = [];
        if (primitive.attributes.NORMAL) {
          normals = new Float32Array(primitive.attributes.NORMAL.bufferView.data);
        }
        let tangents = [];
        if (primitive.attributes.TANGENT) {
          tangents = new Float32Array(primitive.attributes.TANGENT.bufferView.data);
        }
        let bitangents = [];
        if (primitive.attributes.BITANGENT) {
          bitangents = new Float32Array(primitive.attributes.BITANGENT.bufferView.data);
        }
        let colors = [];
        if (primitive.attributes.COLOR_0) {
          colors = new Float32Array(primitive.attributes.COLOR_0.bufferView.data);
        }
        let uvs = [];
        if (primitive.attributes.TEXCOORD_0) {
          switch (primitive.attributes.TEXCOORD_0.componentType) {
            case 5126: // FLOAT
              uvs = new Float32Array(primitive.attributes.TEXCOORD_0.bufferView.data);
              break;
            case 5121: // UNSIGNED_BYTE
              uvs = new Uint8Array(primitive.attributes.TEXCOORD_0.bufferView.data);
              break;
            case 5123: // UNSIGNED_SHORT
              uvs = new Uint16Array(primitive.attributes.TEXCOORD_0.bufferView.data);
              break;
          }

          if (primitive.attributes.TEXCOORD_0.normalized) {
            uvs = uvs.map(
              (v) =>
                v /
                ((1 << (8 * primitive.attributes.TEXCOORD_0.componentType.BYTES_PER_ELEMENT)) - 1)
            );
          }
        }

        if (bitangents.length === 0 && tangents.length > 0 && normals.length > 0) {
          // Compute bitangents using the cross product of normal and tangent
          for (let i = 0; i < tangents.length; i += 3) {
            const t = [tangents[i], tangents[i + 1], tangents[i + 2]];
            const n = [normals[i], normals[i + 1], normals[i + 2]];

            // Cross product: B = N × T (ensuring right-handed coordinate system)
            const b = [
              n[1] * t[2] - n[2] * t[1],
              n[2] * t[0] - n[0] * t[2],
              n[0] * t[1] - n[1] * t[0],
            ];

            // Normalize the bitangent
            const length = Math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
            if (length > 1e-6) {
              b[0] /= length;
              b[1] /= length;
              b[2] /= length;
            }

            bitangents.push(b[0], b[1], b[2]);
          }
        }

        let uv_index = 0;
        for (let i = 0; i < positions.length; i += 3) {
          mesh.vertices.push({
            position: [positions[i] ?? 0.0, positions[i + 1] ?? 0.0, positions[i + 2] ?? 0.0, 1.0],
            normal: [normals[i] ?? 0.0, normals[i + 1] ?? 0.0, normals[i + 2] ?? 0.0, 0.0],
            uv: [uvs[uv_index] ?? 0.0, uvs[uv_index + 1] ?? 0.0, 0.0, 0.0],
            tangent: [tangents[i] ?? 0.0, tangents[i + 1] ?? 0.0, tangents[i + 2] ?? 0.0, 0.0],
            bitangent: [
              bitangents[i] ?? 0.0,
              bitangents[i + 1] ?? 0.0,
              bitangents[i + 2] ?? 0.0,
              0.0,
            ],
          });
          uv_index += 2;
        }
      }
    };

    mesh = new Mesh();
    mesh.name = gltf;

    const mesh_id = Name.from(gltf);

    Mesh.loading_meshes.set(mesh_id, true);

    mesh.pending_loader = new glTFLoader();
    mesh.pending_loader.load(gltf, (gltf_obj) => {
      for (const node of gltf_obj.nodes) {
        if (node.mesh) {
          parse_node_mesh(gltf_obj, node);
        }
      }

      mesh.bounds_min_and_max = mesh._get_vertex_bounds(mesh.vertices);

      mesh.vertex_buffer_offset = SharedVertexBuffer.add_vertex_data(mesh.vertices);

      mesh.vertex_count = mesh.vertices.length;
      mesh.index_count = mesh.indices.length;

      mesh._recreate_index_buffer();

      if (discard_cpu_data) {
        mesh.vertices = null;
        mesh.indices = null;
      }

      Mesh.loading_meshes.delete(mesh_id);
    });

    ResourceCache.get().store(CacheTypes.MESH, mesh_id, mesh);

    return mesh;
  }
}

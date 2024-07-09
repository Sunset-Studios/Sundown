import { glTFLoader } from "@/utility/gltf_loader.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";
import { SharedVertexBuffer } from "@/core/shared_data.js";
import { Buffer } from "@/renderer/buffer.js";
import { Name } from "@/utility/names.js";

export class Mesh {
  constructor() {
    this.name = "";
    this.vertices = [];
    this.indices = [];
    this.vertex_buffer_offset = 0;
    this.index_buffer = null;
  }

  _build_index_buffer(context) {
    let element_type = "uint16";
    if (this.indices.constructor.name === "Uint32Array") {
      element_type = "uint32";
    }

    this.index_buffer = Buffer.create(context, {
      name: `${this.name}_index_buffer`,
      raw_data: this.indices,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      element_type: element_type,
    });
  }

  _get_tangents_and_bitangents(positions, uvs) {
    let tangents = [];
    let bitangents = [];

    for (let i = 0; i < positions.length; i += 9) {
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
      const deltaUV1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
      const deltaUV2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

      // Calculate tangent and bitangent
      const r = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0]);
      const tangent = [
        (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]) * r,
        (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]) * r,
        (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2]) * r,
      ];
      const bitangent = [
        (deltaUV1[0] * edge2[0] - deltaUV2[0] * edge1[0]) * r,
        (deltaUV1[0] * edge2[1] - deltaUV2[0] * edge1[1]) * r,
        (deltaUV1[0] * edge2[2] - deltaUV2[0] * edge1[2]) * r,
      ];

      // Add calculated tangent and bitangent to the arrays
      tangents.push(...tangent, ...tangent, ...tangent);
      bitangents.push(...bitangent, ...bitangent, ...bitangent);
    }

    return { t: tangents, b: bitangents };
  }

  static create(context, name, vertices, indices) {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from(name));
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = name;
    mesh.vertices = vertices;
    mesh.indices = new Uint16Array(indices);

    mesh.vertex_buffer_offset = SharedVertexBuffer.get().add_vertex_data(
      context,
      mesh.vertices
    );
    mesh._build_index_buffer(context);

    ResourceCache.get().store(CacheTypes.MESH, Name.from(name), mesh);

    return mesh;
  }

  static quad(context) {
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
        uv: [0, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [-1, -1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [0, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [1, -1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 0, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
      {
        position: [1, 1, 0, 1],
        normal: [0, 0, -1, 0],
        color: [1, 1, 1, 1],
        uv: [1, 1, 0, 0],
        tangent: [1, 0, 0, 0],
        bitangent: [0, 1, 0, 0],
      },
    ];
    mesh.indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    mesh.vertex_buffer_offset = SharedVertexBuffer.get().add_vertex_data(
      context,
      mesh.vertices
    );

    mesh._build_index_buffer(context);

    ResourceCache.get().store(CacheTypes.MESH, Name.from("engine_quad"), mesh);

    return mesh;
  }

  static async from_gltf(context, gltf) {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from(gltf));
    if (mesh) {
      return mesh;
    }

    return new Promise((resolve) => {
      const loader = new glTFLoader();
      loader.load(gltf, (gltf_obj) => {
        mesh = new Mesh();

        for (const scene of gltf_obj.scenes) {
          for (const node of scene.nodes) {
            if (node.mesh) {
              for (const primitive of node.mesh.primitives) {
                if (primitive.indices) {
                  if (
                    primitive.indicesComponentType === 5122 ||
                    primitive.indicesComponentType === 5123
                  ) {
                    mesh.indices = new Uint16Array(
                      gltf_obj.accessors[primitive.indices].bufferView.data
                    );
                  } else if (primitive.indicesComponentType === 5125) {
                    mesh.indices = new Uint32Array(
                      gltf_obj.accessors[primitive.indices].bufferView.data
                    );
                  }
                }
                let positions = [];
                if (primitive.attributes.POSITION) {
                  positions = new Float32Array(
                    primitive.attributes.POSITION.bufferView.data
                  );
                }
                let normals = [];
                if (primitive.attributes.NORMAL) {
                  normals = new Float32Array(
                    primitive.attributes.NORMAL.bufferView.data
                  );
                }
                let colors = [];
                if (primitive.attributes.COLOR_0) {
                  colors = new Float32Array(
                    primitive.attributes.COLOR_0.bufferView.data
                  );
                }
                let uvs = [];
                if (primitive.attributes.TEXCOORD_0) {
                  uvs = new Float32Array(
                    primitive.attributes.TEXCOORD_0.bufferView.data
                  );
                }
                let tangents = [];
                if (primitive.attributes.TANGENT) {
                  tangents = new Float32Array(
                    primitive.attributes.TANGENT.bufferView.data
                  );
                }
                let bitangents = [];
                if (primitive.attributes.BITANGENT) {
                  bitangents = new Float32Array(
                    primitive.attributes.BITANGENT.bufferView.data
                  );
                }

                // Generate tangents and bitangents if they're not provided
                if (tangents.length === 0 || bitangents.length === 0) {
                  const { t, b } = mesh._get_tangents_and_bitangents(
                    positions,
                    uvs
                  );
                  tangents = t;
                  bitangents = b;
                }

                let uv_index = 0;
                for (let i = 0; i < positions.length; i += 3) {
                  mesh.vertices.push({
                    position: [
                      positions[i] ?? 0.0,
                      positions[i + 1] ?? 0.0,
                      positions[i + 2] ?? 0.0,
                      1.0,
                    ],
                    normal: [
                      normals[i] ?? 0.0,
                      normals[i + 1] ?? 0.0,
                      normals[i + 2] ?? 0.0,
                      0.0,
                    ],
                    color: [
                      colors[i] ?? 0.0,
                      colors[i + 1] ?? 0.0,
                      colors[i + 2] ?? 0.0,
                      colors[i + 3] ?? 1.0,
                    ],
                    uv: [
                      uvs[uv_index] ?? 0.0,
                      uvs[uv_index + 1] ?? 0.0,
                      0.0,
                      0.0,
                    ],
                    tangent: [
                      tangents[i] ?? 0.0,
                      tangents[i + 1] ?? 0.0,
                      tangents[i + 2] ?? 0.0,
                      0.0,
                    ],
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
            }
          }
        }

        mesh.name = gltf;
        mesh.vertex_buffer_offset = SharedVertexBuffer.get().add_vertex_data(
          context,
          mesh.vertices
        );
        mesh._build_index_buffer(context);

        ResourceCache.get().store(CacheTypes.MESH, Name.from(gltf), mesh);

        resolve(mesh);
      });
    });
  }
}

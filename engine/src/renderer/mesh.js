import { glTFLoader } from "@/utility/gltf_loader.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";
import { SharedVertexBuffer } from "@/renderer/shared_data.js";
import { Buffer } from "@/renderer/buffer.js";
import Name from "@/utility/names.js";

export class Mesh {
  constructor() {
    this.name = ''
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

  static create(context, name, vertices) {
    let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from(name));
    if (mesh) {
      return mesh;
    }

    mesh = new Mesh();
    mesh.name = name;
    mesh.vertices = vertices;

    mesh.vertex_buffer_offset = SharedVertexBuffer.get().add_vertex_data(mesh.vertices);
    mesh._build_index_buffer(context);

    ResourceCache.get().store(CacheTypes.MESH, Name.from(name), mesh);
    
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
                    if (primitive.indicesComponentType === 5122 || primitive.indicesComponentType === 5123) {
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
                  });
                  uv_index += 2;
                }
              }
            }
          }
        }

        mesh.name = gltf;
        mesh.vertex_buffer_offset = SharedVertexBuffer.get().add_vertex_data(mesh.vertices);
        mesh._build_index_buffer(context);

        ResourceCache.get().store(CacheTypes.MESH, Name.from(gltf), mesh);
        resolve(mesh);
      });
    });
  }
}

import { Query } from "../core/ecs/solar/query.js";
import { StaticMeshFragment } from "../core/ecs/fragments/static_mesh_fragment.js";
import { TransformFragment } from "../core/ecs/fragments/transform_fragment.js";
import { ResourceCache } from "./resource_cache.js";
import { CacheTypes } from "./renderer_types.js";
import { EntityManager } from "../core/ecs/entity.js";
import { mat4, vec3 } from "gl-matrix";

/**
 * Stitches all static mesh entities in the scene into a single mesh, returning raw vertex and index arrays.
 * Applies world transforms and uses meshoptimizer for post-processing.
 * @param {object} deps - Dependency injection for engine classes/utilities.
 * @returns {{vertices: Array, indices: Array}} The stitched mesh data.
 */
let mesh_query = null;
export async function stitch_scene_meshes() {
  // Query all entities with both StaticMeshFragment and TransformFragment
  if (!mesh_query) {
    mesh_query = Query.create([StaticMeshFragment, TransformFragment]);
  }

  const all_vertices = [];
  const all_indices = [];

  let vertex_offset = 0;
  mesh_query.for_each((chunk, slot, count, archetype) => {
    for (let i = 0; i < count; i++) {
      const entity = chunk.entity_handles[slot + i];

      const static_mesh_view = EntityManager.get_fragment(entity, StaticMeshFragment);
      const transform_view = EntityManager.get_fragment(entity, TransformFragment);
      if (!static_mesh_view || !transform_view) continue;

      // Get mesh by name/id
      const mesh_id = static_mesh_view.mesh[0];
      const mesh = ResourceCache.get().store(CacheTypes.MESH, mesh_id);
      if (!mesh || !mesh.vertices || !mesh.indices) continue;

      // Get world transform (mat4)
      const position = transform_view.position;
      const rotation = transform_view.rotation;
      const scale = transform_view.scale;

      const world_matrix = mat4.fromRotationTranslationScale(
        mat4.create(),
        rotation,
        position,
        scale
      );

      // Transform and append each vertex
      for (let v = 0; v < mesh.vertices.length; v++) {
        const vert = mesh.vertices[v];
        // Transform position
        const pos = vec3.transformMat4(
          vec3.create(),
          [vert.position[0], vert.position[1], vert.position[2]],
          world_matrix
        );
        // Transform normal (ignore translation, only rotation/scale)
        const normal = vec3.transformMat3(
          vec3.create(),
          [vert.normal[0], vert.normal[1], vert.normal[2]],
          mat4.normalFromMat4(mat4.create(), world_matrix)
        );
        // Transform tangent/bitangent if present
        let tangent = vert.tangent ? [vert.tangent[0], vert.tangent[1], vert.tangent[2]] : [1,0,0];
        let bitangent = vert.bitangent ? [vert.bitangent[0], vert.bitangent[1], vert.bitangent[2]] : [0,1,0];
        tangent = vec3.transformMat3(vec3.create(), tangent, mat4.normalFromMat4(mat4.create(), world_matrix));
        bitangent = vec3.transformMat3(vec3.create(), bitangent, mat4.normalFromMat4(mat4.create(), world_matrix));
        // Copy other attributes
        const out_vert = {
          position: [pos[0], pos[1], pos[2], 1],
          normal: [normal[0], normal[1], normal[2], 0],
          color: vert.color ? vert.color.slice() : [1,1,1,1],
          uv: vert.uv ? vert.uv.slice() : [0,0],
          tangent: [tangent[0], tangent[1], tangent[2], vert.tangent ? vert.tangent[3] : 1],
          bitangent: [bitangent[0], bitangent[1], bitangent[2], 0],
          extra_data: vert.extra_data ? vert.extra_data.slice() : [0,0],
        };
        all_vertices.push(out_vert);
      }
      // Append indices, offset by current vertex count
      for (let idx = 0; idx < mesh.indices.length; idx++) {
        all_indices.push(mesh.indices[idx] + vertex_offset);
      }
      vertex_offset += mesh.vertices.length;
    }
  });

  // Return raw arrays (vertex objects and indices)
  return {
    vertices: all_vertices,
    indices: all_indices,
  };
} 
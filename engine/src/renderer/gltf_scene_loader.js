import { glTFLoader } from "../utility/gltf_loader.js";
import { Mesh } from "./mesh.js";
import { vec3, quat, mat4 } from "gl-matrix";
import { spawn_mesh_entity, spawn_transform_entity } from "../core/ecs/entity_utils.js";

// ─────────────────────────────────────────────────────────────────────────────
// GLTF Scene Loader
// ─────────────────────────────────────────────────────────────────────────────
// Loads GLTF scenes and spawns entities with configurable loading strategies.
//
// Loading Modes:
// 1. Default (instanced): Creates instanced entities for meshes shared by multiple
//    nodes. Each instance stores its accumulated world transform from the hierarchy.
//
// 2. Single Mesh: Loads the entire GLTF as a single mesh entity. Useful for
//    simple models where you don't need per-mesh control.
//
// 3. Flat (no instancing): Creates one entity per unique mesh without instancing.
//    Simpler than default but may create more draw calls for repeated meshes.
// ─────────────────────────────────────────────────────────────────────────────

export class GLTFSceneLoader {
  // ───────────────────────────────────────────────────────────────────────────
  // Mesh Cache - Stores unique meshes keyed by their GLTF path + mesh ID
  // ───────────────────────────────────────────────────────────────────────────
  static #mesh_cache = new Map();

  /**
   * Gets or creates a mesh for a given GLTF node.
   * Uses caching to ensure unique meshes are only created once.
   *
   * @param {string} gltf_path - Path to the GLTF file
   * @param {Object} gltf_obj - The parsed GLTF object
   * @param {Object} node - The GLTF node containing the mesh reference
   * @returns {Mesh|null} The mesh object, or null if node has no mesh
   */
  static get_mesh_for_node(gltf_path, gltf_obj, node) {
    if (!node || !node.mesh) return null;

    const gltf_mesh = node.mesh;
    const key = `${gltf_path}#${gltf_mesh.meshID}`;

    if (this.#mesh_cache.has(key)) {
      return this.#mesh_cache.get(key);
    }

    const mesh = Mesh.from_parsed_gltf_mesh(gltf_path, gltf_obj, gltf_mesh);
    this.#mesh_cache.set(key, mesh);

    return mesh;
  }

  /**
   * Extracts the local TRS (translation, rotation, scale) from a GLTF node.
   *
   * @param {Object} node - The GLTF node
   * @returns {Object} Object containing { position, rotation, scale }
   */
  static get_node_local_transform(node) {
    return {
      position: node.translation
        ? vec3.fromValues(node.translation[0], node.translation[1], node.translation[2])
        : [0, 0, 0],
      rotation: node.rotation
        ? quat.fromValues(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3])
        : [0, 0, 0, 1],
      scale: node.scale
        ? vec3.fromValues(node.scale[0], node.scale[1], node.scale[2])
        : [1, 1, 1]
    };
  }

  /**
   * Computes the world transform for a node by walking up the parent chain.
   * This properly accumulates all ancestor transforms.
   *
   * @param {Object} node - The GLTF node
   * @returns {Object} Object containing { position, rotation, scale }
   */
  static compute_node_world_transform(node) {
    // ─────────────────────────────────────────────────────────────────────────
    // Build parent chain (root -> ... -> parent -> node)
    // ─────────────────────────────────────────────────────────────────────────
    const chain = [];
    let current = node;
    while (current) {
      chain.unshift(current);
      current = current._parent;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accumulate transforms from root to node
    // ─────────────────────────────────────────────────────────────────────────
    let world_matrix = mat4.create();
    for (let i = 0; i < chain.length; i++) {
      const local = this.get_node_local_transform(chain[i]);
      const local_matrix = mat4.fromRotationTranslationScale(mat4.create(), local.rotation, local.position, local.scale);
      world_matrix = mat4.mul(mat4.create(), world_matrix, local_matrix);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Decompose the combined matrix back into TRS components
    // ─────────────────────────────────────────────────────────────────────────
    const position = mat4.getTranslation(vec3.create(), world_matrix);
    const rotation = mat4.getRotation(quat.create(), world_matrix);
    const scale = mat4.getScaling(vec3.create(), world_matrix);

    return { position, rotation, scale };
  }

  /**
   * Loads a GLTF scene and spawns entities.
   *
   * @param {string} gltf_path - Path to the GLTF file
   * @param {Array} position - Top-level position [x, y, z] for the scene root
   * @param {Array} rotation - Top-level rotation [x, y, z, w] quaternion for the scene root
   * @param {Array} scale - Top-level scale [x, y, z] for the scene root
   * @param {number|null} scene_index - Index of the GLTF scene to load (null = default)
   * @param {EntityHandle|null} parent_entity - Parent entity for the scene root
   * @param {Function|null} callback - Called with (root_entity, entities) when loading completes
   * @param {Object} options - Loading options
   * @returns {EntityHandle} The root/mesh entity (returned immediately, loading happens async)
   */
  static load_scene(
    gltf_path,
    position = [0, 0, 0],
    rotation = [0, 0, 0, 1],
    scale = [1, 1, 1],
    scene_index = null,
    parent_entity = null,
    callback = null,
    options = {}
  ) {
    // ─────────────────────────────────────────────────────────────────────────
    // INSTANCED/FLAT MODE: Create root entity and load scene hierarchy
    // ─────────────────────────────────────────────────────────────────────────
    const root_entity = spawn_transform_entity(
      position,
      rotation,
      scale,
      parent_entity
    );

    const loader = new glTFLoader();
    loader.load(gltf_path, (gltf_obj) => {
      const entities = [root_entity];
      const scene_meshes = new Set();

      const scene_to_use =
        scene_index ?? gltf_obj.defaultScene ?? (gltf_obj.scenes.length > 0 ? 0 : null);
      const scene = scene_to_use !== null ? gltf_obj.scenes[scene_to_use] : null;
      const root_nodes = scene ? scene.nodes : gltf_obj.nodes;

      const spawn_node_hierarchy = (node, parent) => {
        const local_transform = this.get_node_local_transform(node);
        const mesh = this.get_mesh_for_node(gltf_path, gltf_obj, node);
        if (mesh) {
          scene_meshes.add(mesh);
        }

        const entity = mesh
          ? spawn_mesh_entity(
            local_transform.position,
            local_transform.rotation,
            local_transform.scale,
            mesh,
            0,
            parent,
            [],
            true,
            0
          )
          : spawn_transform_entity(
            local_transform.position,
            local_transform.rotation,
            local_transform.scale,
            parent,
            0
          );

        entities.push(entity);

        for (const child of node.children ?? []) {
          spawn_node_hierarchy(child, entity);
        }
      };

      for (const node of root_nodes) {
        spawn_node_hierarchy(node, root_entity);
      }

      void Promise.all(
        Array.from(scene_meshes, (mesh) => mesh.pending_gltf_build_promise).filter(Boolean)
      ).then(() => {
        for (const mesh of scene_meshes) {
          Mesh.finalize_prepared_gltf_mesh(mesh, gltf_obj);
        }

        if (callback) {
          callback(root_entity, entities);
        }
      });
    });

    return root_entity;
  }
}

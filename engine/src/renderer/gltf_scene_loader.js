import { glTFLoader } from "../utility/gltf_loader.js";
import { Mesh } from "./mesh.js";
import { vec3, quat, mat4 } from "gl-matrix";
import { spawn_mesh_entity, spawn_transform_entity } from "../core/ecs/entity_utils.js";
import { EntityManager } from "../core/ecs/entity.js";
import { TransformFragment } from "../core/ecs/fragments/transform_fragment.js";

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
    const position = node.translation
      ? vec3.fromValues(node.translation[0], node.translation[1], node.translation[2])
      : [0, 0, 0];

    const rotation = node.rotation
      ? quat.fromValues(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3])
      : [0, 0, 0, 1];

    const scale = node.scale
      ? vec3.fromValues(node.scale[0], node.scale[1], node.scale[2])
      : [1, 1, 1];

    return { position, rotation, scale };
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
    const world_matrix = mat4.create();

    for (const n of chain) {
      const local = this.get_node_local_transform(n);
      const local_matrix = mat4.create();
      mat4.fromRotationTranslationScale(
        local_matrix,
        local.rotation,
        local.position,
        local.scale
      );
      mat4.multiply(world_matrix, world_matrix, local_matrix);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Decompose the combined matrix back into TRS components
    // ─────────────────────────────────────────────────────────────────────────
    const position = vec3.create();
    const rotation = quat.create();
    const scale = vec3.create();

    mat4.getTranslation(position, world_matrix);
    mat4.getRotation(rotation, world_matrix);
    mat4.getScaling(scale, world_matrix);

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
   * @param {boolean} options.single_mesh - If true, combines all GLTF meshes into one entity (default: false)
   * @param {boolean} options.flat - If true, creates one entity per mesh node without instancing (default: false)
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
    const {
      single_mesh = true,
      flat = false,
    } = options;

    // ─────────────────────────────────────────────────────────────────────────
    // SINGLE MESH MODE: Combine all GLTF meshes into one mesh entity
    // All node transforms are baked into the vertex positions
    // ─────────────────────────────────────────────────────────────────────────
    if (single_mesh) {
      const mesh = Mesh.from_gltf_scene(gltf_path, scene_index);

      const mesh_entity = spawn_mesh_entity(
        position,
        rotation,
        scale,
        mesh,
        0, // GLTF sets the material id via mesh internals
        parent_entity
      );

      if (callback) {
        // Call callback after a microtask to allow mesh loading to start
        callback(mesh_entity, [mesh_entity]);
      }

      return mesh_entity;
    }

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

      // ─────────────────────────────────────────────────────────────────────
      // Set up parent references for hierarchical traversal
      // ─────────────────────────────────────────────────────────────────────
      for (const node of gltf_obj.nodes) {
        for (const child of node.children) {
          child._parent = node;
        }
      }

      // ─────────────────────────────────────────────────────────────────────
      // Determine which scene to load
      // ─────────────────────────────────────────────────────────────────────
      const scene_to_use =
        scene_index ?? gltf_obj.defaultScene ?? (gltf_obj.scenes.length > 0 ? 0 : null);
      const scene = scene_to_use !== null ? gltf_obj.scenes[scene_to_use] : null;
      const root_nodes = scene ? scene.nodes : gltf_obj.nodes;

      // ─────────────────────────────────────────────────────────────────────
      // FLAT MODE: Create one entity per mesh node (no instancing)
      // ─────────────────────────────────────────────────────────────────────
      if (flat) {
        const process_node_flat = (node) => {
          if (node.mesh) {
            const mesh = this.get_mesh_for_node(gltf_path, gltf_obj, node);
            if (mesh) {
              const world_transform = this.compute_node_world_transform(node);

              const mesh_entity = spawn_mesh_entity(
                world_transform.position,
                world_transform.rotation,
                world_transform.scale,
                mesh,
                0,
                root_entity
              );

              entities.push(mesh_entity);
            }
          }

          for (const child of node.children) {
            process_node_flat(child);
          }
        };

        for (const node of root_nodes) {
          process_node_flat(node);
        }

        if (callback) {
          callback(root_entity, entities);
        }
      } else {
        // ─────────────────────────────────────────────────────────────────────
        // INSTANCED MODE (default): Group meshes and use instancing
        // ─────────────────────────────────────────────────────────────────────
        const mesh_to_nodes = new Map();

        const collect_mesh_nodes = (node) => {
          if (node.mesh) {
            const mesh_key = node.mesh.meshID;

            if (!mesh_to_nodes.has(mesh_key)) {
              mesh_to_nodes.set(mesh_key, []);
            }

            mesh_to_nodes.get(mesh_key).push(node);
          }

          for (const child of node.children) {
            collect_mesh_nodes(child);
          }
        };

        for (const node of root_nodes) {
          collect_mesh_nodes(node);
        }

        // Create instanced mesh entities, parented to root
        for (const [mesh_key, nodes] of mesh_to_nodes) {
          const mesh = this.get_mesh_for_node(gltf_path, gltf_obj, nodes[0]);
          if (!mesh) continue;

          // Get first node's world transform for initial entity spawn
          const first_transform = this.compute_node_world_transform(nodes[0]);

          // Spawn mesh entity parented to root
          const mesh_entity = spawn_mesh_entity(
            first_transform.position,
            first_transform.rotation,
            first_transform.scale,
            mesh,
            0, // GLTF sets the material id via mesh internals
            root_entity
          );

          entities.push(mesh_entity);

          // Set up instancing if multiple nodes share this mesh
          if (nodes.length > 1) {
            EntityManager.set_entity_instance_count(mesh_entity, nodes.length);

            // Set transform for each instance (including first)
            for (let i = 0; i < nodes.length; i++) {
              const node = nodes[i];
              const world_transform = this.compute_node_world_transform(node);

              const transform_view = EntityManager.get_fragment(
                mesh_entity,
                TransformFragment,
                i
              );

              transform_view.position = world_transform.position;
              transform_view.rotation = world_transform.rotation;
              transform_view.scale = world_transform.scale;
            }
          }
        }
      }

      if (callback) {
        callback(root_entity, entities);
      }
    });

    return root_entity;
  }
}

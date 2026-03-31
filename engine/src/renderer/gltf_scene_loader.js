import { glTFLoader } from "../utility/gltf_loader.js";
import { Mesh } from "./mesh.js";
import { vec3, quat, mat4 } from "gl-matrix";
import { spawn_mesh_entity, spawn_transform_entity } from "../core/ecs/entity_utils.js";

export class GLTFSceneLoader {
  static #mesh_cache = new Map();

  /**
   * Gets or creates a mesh for a given GLTF node.
   *
   * @param {string} gltf_path
   * @param {Object} gltf_obj
   * @param {Object} node
   * @returns {Mesh|null}
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
   * Extract the local TRS from a GLTF node.
   *
   * @param {Object} node
   * @returns {{ position: vec3, rotation: quat, scale: vec3 }}
   */
  static get_node_local_transform(node) {
    const has_translation = node.translation !== undefined;
    const has_rotation = node.rotation !== undefined;
    const has_scale = node.scale !== undefined;

    if (!has_translation && !has_rotation && !has_scale && node.matrix) {
      const local_matrix = mat4.clone(node.matrix);
      return {
        position: mat4.getTranslation(vec3.create(), local_matrix),
        rotation: mat4.getRotation(quat.create(), local_matrix),
        scale: mat4.getScaling(vec3.create(), local_matrix)
      };
    }

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
   * Loads a GLTF scene and spawns entities.
   *
   * @param {string} gltf_path
   * @param {Array} position
   * @param {Array} rotation
   * @param {Array} scale
   * @param {number|null} scene_index
   * @param {EntityHandle|null} parent_entity
   * @param {Function|null} callback
   * @param {Object} options
   * @param {boolean} options.single_mesh
   * @returns {EntityHandle}
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
    const { single_mesh = false } = options;

    if (single_mesh) {
      const mesh = Mesh.from_gltf_scene(gltf_path, scene_index);

      const mesh_entity = spawn_mesh_entity(
        position,
        rotation,
        scale,
        mesh,
        0,
        parent_entity
      );

      if (callback) {
        callback(mesh_entity, [mesh_entity]);
      }

      return mesh_entity;
    }

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

import { Renderer } from "../../renderer/renderer.js";
import { Mesh } from "../../renderer/mesh.js";
import { Material } from "../../renderer/material.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";
import { SceneGraphFragment } from "../../core/ecs/fragments/scene_graph_fragment.js";
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "../../core/ecs/fragments/visibility_fragment.js";
import { UserInterfaceFragment } from "../../core/ecs/fragments/user_interface_fragment.js";
import { spawn_mesh_entity } from "../../core/ecs/entity_utils.js";

export class Element3D {
  events = {};

  static create(scene, config, material = null, parent = null, children = []) {
    const context = Renderer.get().graphics_context;

    const entity = spawn_mesh_entity(
      scene,
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 0, z: 0, w: 1 },
      { x: 1, y: 1, z: 1 },
      Mesh.quad(context),
      material ?? Material.default_material(context),
      parent,
      children,
      false /* refresh_entity_queries */
    );

    scene.add_fragment(entity, UserInterfaceFragment, {
      allows_cursor_events: false,
      was_cursor_inside: false,
      is_cursor_inside: false,
      was_clicked: false,
      is_clicked: false,
    });

    this.set_config(config);

    return entity;
  }

  static destroy(scene, entity) {
    scene.delete_entity(entity);
  }

  static add_child(scene, entity, child) {
    let this_scene_graph_data = scene.get_fragment(entity, SceneGraphFragment);
    let child_scene_graph_data = scene.get_fragment(child.entity, SceneGraphFragment);
    if (this_scene_graph_data && child_scene_graph_data) {
      this_scene_graph_data.children.push(child.entity);
      child_scene_graph_data.parent = entity;
      scene.update_fragment(child.entity, SceneGraphFragment, child_scene_graph_data);
      scene.update_fragment(entity, SceneGraphFragment, this_scene_graph_data);
    }
  }

  static remove_child(scene, entity, child) {
    let this_scene_graph_data = scene.get_fragment(entity, SceneGraphFragment);
    let child_scene_graph_data = scene.get_fragment(child.entity, SceneGraphFragment);
    if (this_scene_graph_data && child_scene_graph_data) {
      this_scene_graph_data.children = this_scene_graph_data.children.filter(
        (c) => c !== child.entity
      );
      child_scene_graph_data.parent = null;
      scene.update_fragment(child.entity, SceneGraphFragment, child_scene_graph_data);
      scene.update_fragment(entity, SceneGraphFragment, this_scene_graph_data);
    }
  }

  static set_config(scene, entity, config) {
    if (!config) return;

    const { position, rotation, scale, allows_cursor_events } = config;

    if (position) {
      this.set_position(scene, entity, position);
    }
    if (rotation) {
      this.set_rotation(scene, entity, rotation);
    }
    if (scale) {
      this.set_scale(scene, entity, scale);
    }

    if (allows_cursor_events !== undefined) {
      this.set_allows_cursor_events(scene, entity, allows_cursor_events);
    }
  }

  static set_material(scene, entity, material) {
    let mesh_data = scene.get_fragment(entity, StaticMeshFragment);
    mesh_data.material_slots = [material];
    scene.update_fragment(entity, StaticMeshFragment, mesh_data);
  }

  static set_parent(scene, entity, parent) {
    let this_scene_graph_data = scene.get_fragment(entity, SceneGraphFragment);
    if (!this_scene_graph_data) {
      return;
    }

    let parent_scene_graph_data = scene.get_fragment(parent.entity, SceneGraphFragment);
    if (parent_scene_graph_data) {
      this_scene_graph_data.parent = parent.entity;
      parent_scene_graph_data.children.push(entity);
      scene.update_fragment(
        parent.entity,
        SceneGraphFragment,
        parent_scene_graph_data
      );
    } else {
      this_scene_graph_data.parent = null;
    }

    scene.update_fragment(entity, SceneGraphFragment, this_scene_graph_data);
  }

  static set_children(scene, entity, children) {
    let this_scene_graph_data = scene.get_fragment(entity, SceneGraphFragment);
    if (!this_scene_graph_data) {
      return;
    }

    for (let i = 0; i < this_scene_graph_data.children.length; i++) {
      let child_scene_graph_data = scene.get_fragment(
        this_scene_graph_data.children[i],
        SceneGraphFragment
      );
      if (child_scene_graph_data) {
        child_scene_graph_data.parent = null;
        scene.update_fragment(
          this_scene_graph_data.children[i],
          SceneGraphFragment,
          child_scene_graph_data
        );
      }
    }

    this_scene_graph_data.children = children.map((c) => c.entity);

    for (let i = 0; i < children.length; i++) {
      let child = children[i];
      let child_scene_graph_data = scene.get_fragment(child.entity, SceneGraphFragment);
      if (child_scene_graph_data) {
        child_scene_graph_data.parent = entity;
        scene.update_fragment(child.entity, SceneGraphFragment, child_scene_graph_data);
      }
    }

    scene.update_fragment(entity, SceneGraphFragment, this_scene_graph_data);
  }

  static set_position(scene, entity, position) {
    let transform_data = scene.get_fragment(entity, TransformFragment);
    transform_data.position = position;
    scene.update_fragment(entity, TransformFragment, transform_data);
  }

  static set_rotation(scene, entity, rotation) {
    let transform_data = scene.get_fragment(entity, TransformFragment);
    transform_data.rotation = rotation;
    scene.update_fragment(entity, TransformFragment, transform_data);
  }

  static set_scale(scene, entity, scale) {
    let transform_data = scene.get_fragment(entity, TransformFragment);
    transform_data.scale = scale;
    scene.update_fragment(entity, TransformFragment, transform_data);
  }

  static set_allows_cursor_events(scene, entity, allows_cursor_events) {
    let user_interface_data = scene.get_fragment(entity, UserInterfaceFragment);
    user_interface_data.allows_cursor_events = allows_cursor_events;
    scene.update_fragment(entity, UserInterfaceFragment, user_interface_data);
  }

  static set_visible(scene, entity, visible) {
    let visibility_data = scene.get_fragment(entity, VisibilityFragment);
    visibility_data.visible = visible;
    scene.update_fragment(entity, VisibilityFragment, visibility_data);
  }

  static set_auto_size(scene, entity, auto_size) {
    let user_interface_data = scene.get_fragment(entity, UserInterfaceFragment);
    user_interface_data.auto_size = auto_size;
    scene.update_fragment(entity, UserInterfaceFragment, user_interface_data);
  }

  static on(entity, event, callback) {
    if (!this.events[entity]) {
      this.events[entity] = {};
    }
    if (!this.events[entity][event]) {
      this.events[entity][event] = [];
    }
    if (!this.events[entity][event].includes(callback)) {
      this.events[entity][event].push(callback);
    }
  }

  static trigger(entity, event, ...args) {
    if (this.events[entity] && this.events[entity][event]) {
      for (let i = 0; i < this.events[entity][event].length; i++) {
        this.events[entity][event][i](...args);
      }
    }
  }
}

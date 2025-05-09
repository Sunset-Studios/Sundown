import { EntityManager } from "../../core/ecs/entity.js";
import { EntityTransformFlags } from "../../core/minimal.js";
import { Mesh } from "../../renderer/mesh.js";
import { Material } from "../../renderer/material.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";
import { SceneGraphFragment } from "../../core/ecs/fragments/scene_graph_fragment.js";
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "../../core/ecs/fragments/visibility_fragment.js";
import { UserInterfaceFragment } from "../../core/ecs/fragments/user_interface_fragment.js";
import { spawn_mesh_entity } from "../../core/ecs/entity_utils.js";

export class Element3D {
  static events = {};

  static create(config, material = null, parent = null, children = [], start_visible = true, flags = EntityTransformFlags.IGNORE_PARENT_SCALE) {
    const entity = spawn_mesh_entity(
      [0, 0, 0],
      [0, 0, 0, 1],
      [1, 1, 1],
      Mesh.quad(),
      material ?? Material.default_ui_material(),
      parent,
      children,
      start_visible,
      flags
    );

    const new_user_interface_view = EntityManager.add_fragment(entity, UserInterfaceFragment);
    new_user_interface_view.allows_cursor_events = 1;
    new_user_interface_view.auto_size = 0;
    new_user_interface_view.was_cursor_inside = 0;
    new_user_interface_view.is_cursor_inside = 0;
    new_user_interface_view.was_clicked = 0;
    new_user_interface_view.is_clicked = 0;
    new_user_interface_view.is_pressed = 0;
    new_user_interface_view.was_pressed = 0;
    new_user_interface_view.color.r = 0;
    new_user_interface_view.color.g = 0;
    new_user_interface_view.color.b = 0;
    new_user_interface_view.color.a = 0;

    this.set_config(entity, config);

    return entity;
  }

  static destroy(entity) {
    EntityManager.delete_entity(entity);
  }

  static add_child(entity, child) {
    let this_scene_graph_data = EntityManager.get_fragment(entity, SceneGraphFragment);
    let child_scene_graph_data = EntityManager.get_fragment(child.entity, SceneGraphFragment);
    if (this_scene_graph_data && child_scene_graph_data) {
      const children = this_scene_graph_data.children;
      children.push(child.entity);
      this_scene_graph_data.children = children;
      child_scene_graph_data.parent = entity;
    }
  }

  static remove_child(entity, child) {
    let this_scene_graph_data = EntityManager.get_fragment(entity, SceneGraphFragment);
    let child_scene_graph_data = EntityManager.get_fragment(child.entity, SceneGraphFragment);
    if (this_scene_graph_data && child_scene_graph_data) {
      this_scene_graph_data.children = this_scene_graph_data.children.filter(
        (c) => c !== child.entity
      );
      child_scene_graph_data.parent = null;
    }
  }

  static set_config(entity, config) {
    if (!config) return;

    const { position, rotation, scale, allows_cursor_events, visible, auto_size, color } = config;

    if (position) {
      this.set_position(entity, position);
    }
    if (rotation) {
      this.set_rotation(entity, rotation);
    }
    if (scale) {
      this.set_scale(entity, scale);
    }
    if (allows_cursor_events !== undefined) {
      this.set_allows_cursor_events(entity, allows_cursor_events);
    }
    if (visible !== undefined) {
      this.set_visible(entity, visible);
    }
    if (auto_size !== undefined) {
      this.set_auto_size(entity, auto_size);
    }
    if (color) {
      this.set_color(entity, color);
    }
  }

  static set_material(entity, material) {
    let mesh_data = EntityManager.get_fragment(entity, StaticMeshFragment);
    if (!mesh_data) {
      return;
    }
    
    if (material) {
      mesh_data.material_slots = [material];
    } else {
      mesh_data.material_slots = [];
    }
  }

  static set_parent(entity, parent) {
    let this_scene_graph_data = EntityManager.get_fragment(entity, SceneGraphFragment);
    if (!this_scene_graph_data) {
      return;
    }

    if (EntityManager.has_fragment(parent, SceneGraphFragment)) {
      this_scene_graph_data.parent = parent;
    } else {
      this_scene_graph_data.parent = null;
    }
  }

  static set_children(entity, children) {
    let this_scene_graph_data = EntityManager.get_fragment(entity, SceneGraphFragment);
    if (!this_scene_graph_data) {
      return;
    }

    for (let i = 0; i < this_scene_graph_data.children.length; i++) {
      let child_scene_graph_data = EntityManager.get_fragment(
        this_scene_graph_data.children[i],
        SceneGraphFragment
      );
      if (child_scene_graph_data) {
        child_scene_graph_data.parent = null;
      }
    }

    for (let i = 0; i < children.length; i++) {
      let child = children[i];
      let child_scene_graph_data = EntityManager.get_fragment(child.entity, SceneGraphFragment);
      if (child_scene_graph_data) {
        child_scene_graph_data.parent = entity;
      }
    }
  }

  static set_position(entity, position) {
    let transform_data = EntityManager.get_fragment(entity, TransformFragment);
    if (!transform_data) {
      return;
    }
    transform_data.position = position;
  }

  static set_rotation(entity, rotation) {
    let transform_data = EntityManager.get_fragment(entity, TransformFragment);
    if (!transform_data) {
      return;
    }
    transform_data.rotation = rotation;
  }

  static set_scale(entity, scale) {
    let transform_data = EntityManager.get_fragment(entity, TransformFragment);
    if (!transform_data) {
      return;
    }
    transform_data.scale = scale;
  }

  static set_allows_cursor_events(entity, allows_cursor_events) {
    let user_interface_data = EntityManager.get_fragment(entity, UserInterfaceFragment);
    if (!user_interface_data) {
      return;
    }
    user_interface_data.allows_cursor_events = allows_cursor_events;
  }

  static set_visible(entity, visible) {
    let visibility_data = EntityManager.get_fragment(entity, VisibilityFragment);
    if (!visibility_data) {
      return;
    }
    visibility_data.visible = visible;
  }

  static set_auto_size(entity, auto_size) {
    let user_interface_data = EntityManager.get_fragment(entity, UserInterfaceFragment);
    if (!user_interface_data) {
      return;
    }
    user_interface_data.auto_size = auto_size;
  }

  static set_color(entity, color) {
    let user_interface_data = EntityManager.get_fragment(entity, UserInterfaceFragment);
    if (!user_interface_data) {
      return;
    }
    user_interface_data.color.r = color.r;
    user_interface_data.color.g = color.g;
    user_interface_data.color.b = color.b;
    user_interface_data.color.a = color.a;
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
        this.events[entity][event][i](entity, ...args);
      }
    }
  }
}

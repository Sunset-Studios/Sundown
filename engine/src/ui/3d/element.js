import { EntityManager } from "../../core/ecs/entity.js";
import { EntityFlags } from "../../core/minimal.js";
import { Mesh } from "../../renderer/mesh.js";
import { Material } from "../../renderer/material.js";
import { StaticMeshFragment } from "../../core/ecs/fragments/static_mesh_fragment.js";
import { TransformFragment } from "../../core/ecs/fragments/transform_fragment.js";
import { VisibilityFragment } from "../../core/ecs/fragments/visibility_fragment.js";
import { UserInterfaceFragment } from "../../core/ecs/fragments/user_interface_fragment.js";
import { spawn_mesh_entity } from "../../core/ecs/entity_utils.js";

export class Element3D {
  static events = {};

  static create(config, material = null, parent = null, children = [], start_visible = true, flags = EntityFlags.IGNORE_PARENT_SCALE) {
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
    new_user_interface_view.color = [0, 0, 0, 0];

    this.set_config(entity, config);

    return entity;
  }

  static destroy(entity) {
    EntityManager.delete_entity(entity);
  }

  static add_child(entity, child) {
    let children = EntityManager.get_entity_children(entity);
    children.push(child.entity);
    EntityManager.set_entity_children(entity, children);
    EntityManager.set_entity_parent(child.entity, entity);
  }

  static remove_child(entity, child) {
    let children = EntityManager.get_entity_children(entity);
    children = children.filter((c) => c !== child.entity);
    EntityManager.set_entity_children(entity, children);
    EntityManager.set_entity_parent(child.entity, null);
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
      mesh_data.material_slots = [BigInt(material)];
    } else {
      mesh_data.material_slots = [];
    }
  }

  static set_parent(entity, parent) {
    EntityManager.set_entity_parent(entity, parent);
  }

  static set_children(entity, children) {
    let old_children = EntityManager.get_entity_children(entity);
    for (let i = 0; i < old_children.length; i++) {
      EntityManager.set_entity_parent(old_children[i], null);
    }
    for (let i = 0; i < children.length; i++) {
      EntityManager.set_entity_parent(children[i], entity);
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
    user_interface_data.color = [color.r, color.g, color.b, color.a];
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

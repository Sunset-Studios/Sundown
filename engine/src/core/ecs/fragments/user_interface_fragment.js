import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";

export class UserInterfaceFragment extends Fragment {
  static initialize() {
    this.data = {
      allows_cursor_events: new Uint8Array(1),
      auto_size: new Uint8Array(1),
      was_cursor_inside: new Uint8Array(1),
      is_cursor_inside: new Uint8Array(1),
      was_clicked: new Uint8Array(1),
      is_clicked: new Uint8Array(1),
      is_pressed: new Uint8Array(1),
      was_pressed: new Uint8Array(1),
      dirty: new Uint8Array(1),
      gpu_data_dirty: true,
    };
  }

  static resize(new_size) {
    if (!this.data) this.initialize();
    super.resize(new_size);

    Fragment.resize_array(
      this.data,
      "allows_cursor_events",
      new_size,
      Uint8Array,
      1,
    );
    Fragment.resize_array(this.data, "auto_size", new_size, Uint8Array, 1);
    Fragment.resize_array(
      this.data,
      "was_cursor_inside",
      new_size,
      Uint8Array,
      1,
    );
    Fragment.resize_array(
      this.data,
      "is_cursor_inside",
      new_size,
      Uint8Array,
      1,
    );
    Fragment.resize_array(this.data, "was_clicked", new_size, Uint8Array, 1);
    Fragment.resize_array(this.data, "is_clicked", new_size, Uint8Array, 1);
    Fragment.resize_array(this.data, "is_pressed", new_size, Uint8Array, 1);
    Fragment.resize_array(this.data, "was_pressed", new_size, Uint8Array, 1);
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array, 1);
  }

  static add_entity(entity, data) {
    super.add_entity(entity, data);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.update_entity_data(entity, {
      allows_cursor_events: 0,
      auto_size: 0,
      was_cursor_inside: 0,
      is_cursor_inside: 0,
      was_clicked: 0,
      is_clicked: 0,
      is_pressed: 0,
      was_pressed: 0,
      dirty: 0,
    });
  }

  static get_entity_data(entity) {
    return {
      allows_cursor_events: this.data.allows_cursor_events[entity],
      auto_size: this.data.auto_size[entity],
      was_cursor_inside: this.data.was_cursor_inside[entity],
      is_cursor_inside: this.data.is_cursor_inside[entity],
      was_clicked: this.data.was_clicked[entity],
      is_clicked: this.data.is_clicked[entity],
      is_pressed: this.data.is_pressed[entity],
      was_pressed: this.data.was_pressed[entity],
    };
  }

  static duplicate_entity_data(entity) {
    const data = {};
    data.allows_cursor_events = this.data.allows_cursor_events[entity];
    data.auto_size = this.data.auto_size[entity];
    data.was_cursor_inside = this.data.was_cursor_inside[entity];
    data.is_cursor_inside = this.data.is_cursor_inside[entity];
    data.was_clicked = this.data.was_clicked[entity];
    data.is_clicked = this.data.is_clicked[entity];
    data.is_pressed = this.data.is_pressed[entity];
    data.was_pressed = this.data.was_pressed[entity];
    data.dirty = this.data.dirty[entity];
    return data;
  }

  static update_entity_data(entity, data) {
    if (!this.data) {
      this.initialize();
    }

    super.update_entity_data(entity, data);

    if (this.data.dirty) {
      this.data.dirty[entity] = 1;
    }
    this.data.gpu_data_dirty = true;
  }
}

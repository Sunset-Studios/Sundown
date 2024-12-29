import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";

class UserInterfaceDataView {
  current_entity = -1;

  constructor() {}

  get allows_cursor_events() {
    return UserInterfaceFragment.data.allows_cursor_events[this.current_entity];
  }

  set allows_cursor_events(value) {
    UserInterfaceFragment.data.allows_cursor_events[this.current_entity] =
      UserInterfaceFragment.data.allows_cursor_events instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get auto_size() {
    return UserInterfaceFragment.data.auto_size[this.current_entity];
  }

  set auto_size(value) {
    UserInterfaceFragment.data.auto_size[this.current_entity] =
      UserInterfaceFragment.data.auto_size instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get was_cursor_inside() {
    return UserInterfaceFragment.data.was_cursor_inside[this.current_entity];
  }

  set was_cursor_inside(value) {
    UserInterfaceFragment.data.was_cursor_inside[this.current_entity] =
      UserInterfaceFragment.data.was_cursor_inside instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get is_cursor_inside() {
    return UserInterfaceFragment.data.is_cursor_inside[this.current_entity];
  }

  set is_cursor_inside(value) {
    UserInterfaceFragment.data.is_cursor_inside[this.current_entity] =
      UserInterfaceFragment.data.is_cursor_inside instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get was_clicked() {
    return UserInterfaceFragment.data.was_clicked[this.current_entity];
  }

  set was_clicked(value) {
    UserInterfaceFragment.data.was_clicked[this.current_entity] =
      UserInterfaceFragment.data.was_clicked instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get is_clicked() {
    return UserInterfaceFragment.data.is_clicked[this.current_entity];
  }

  set is_clicked(value) {
    UserInterfaceFragment.data.is_clicked[this.current_entity] =
      UserInterfaceFragment.data.is_clicked instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get is_pressed() {
    return UserInterfaceFragment.data.is_pressed[this.current_entity];
  }

  set is_pressed(value) {
    UserInterfaceFragment.data.is_pressed[this.current_entity] =
      UserInterfaceFragment.data.is_pressed instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get was_pressed() {
    return UserInterfaceFragment.data.was_pressed[this.current_entity];
  }

  set was_pressed(value) {
    UserInterfaceFragment.data.was_pressed[this.current_entity] =
      UserInterfaceFragment.data.was_pressed instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get dirty() {
    return UserInterfaceFragment.data.dirty[this.current_entity];
  }

  set dirty(value) {
    UserInterfaceFragment.data.dirty[this.current_entity] =
      UserInterfaceFragment.data.dirty instanceof BigInt64Array
        ? BigInt(value)
        : value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  view_entity(entity) {
    this.current_entity = entity;

    return this;
  }
}

export class UserInterfaceFragment extends Fragment {
  static data_view_allocator = new RingBufferAllocator(
    256,
    UserInterfaceDataView,
  );

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

  static add_entity(entity) {
    super.add_entity(entity);
    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);
    this.data.allows_cursor_events[entity] = 0;
    this.data.auto_size[entity] = 0;
    this.data.was_cursor_inside[entity] = 0;
    this.data.is_cursor_inside[entity] = 0;
    this.data.was_clicked[entity] = 0;
    this.data.is_clicked[entity] = 0;
    this.data.is_pressed[entity] = 0;
    this.data.was_pressed[entity] = 0;
  }

  static get_entity_data(entity) {
    const data_view = this.data_view_allocator.allocate();
    data_view.fragment = this;
    data_view.view_entity(entity);
    return data_view;
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
}

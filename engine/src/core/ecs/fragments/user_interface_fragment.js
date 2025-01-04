import { EntityLinearDataContainer } from "../entity_utils.js";
import { Fragment } from "../fragment.js";
import { Renderer } from "../../../renderer/renderer.js";
import { Buffer } from "../../../renderer/buffer.js";
import { global_dispatcher } from "../../../core/dispatcher.js";
import { RingBufferAllocator } from "../../../memory/allocator.js";
import { EntityID } from "../entity.js";

const element_data_buffer_name = "element_data_buffer";
const element_data_cpu_buffer_name = "element_data_cpu_buffer";
const element_data_event = "element_data";
const element_data_update_event = "element_data_update";

class ColorDataView {
  constructor() {
    this.current_entity = -1n;
  }

  get r() {
    return UserInterfaceFragment.data.color.r[this.current_entity];
  }

  set r(value) {
    UserInterfaceFragment.data.color.r[this.current_entity] = value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get g() {
    return UserInterfaceFragment.data.color.g[this.current_entity];
  }

  set g(value) {
    UserInterfaceFragment.data.color.g[this.current_entity] = value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get b() {
    return UserInterfaceFragment.data.color.b[this.current_entity];
  }

  set b(value) {
    UserInterfaceFragment.data.color.b[this.current_entity] = value;
    if (UserInterfaceFragment.data.dirty) {
      UserInterfaceFragment.data.dirty[this.current_entity] = 1;
    }
    UserInterfaceFragment.data.gpu_data_dirty = true;
  }

  get a() {
    return UserInterfaceFragment.data.color.a[this.current_entity];
  }

  set a(value) {
    UserInterfaceFragment.data.color.a[this.current_entity] = value;
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

class UserInterfaceDataView {
  current_entity = -1n;

  constructor() {
    this.color = new ColorDataView(this);
  }

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

    this.color.view_entity(entity);

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
      color: {
        r: new Float32Array(1),
        g: new Float32Array(1),
        b: new Float32Array(1),
        a: new Float32Array(1),
      },
      dirty: new Uint8Array(1),
      element_data_buffer: null,
      gpu_data_dirty: true,
    };

    this.rebuild_buffers();
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
    Object.keys(this.data.color).forEach((axis) => {
      Fragment.resize_array(this.data.color, axis, new_size, Float32Array);
    });
    Fragment.resize_array(this.data, "dirty", new_size, Uint8Array, 1);

    this.rebuild_buffers();
  }

  static add_entity(entity) {
    super.add_entity(entity);
    return this.get_entity_data(entity);
  }

  static remove_entity(entity) {
    super.remove_entity(entity);

    const instance_count = EntityID.get_instance_count(entity);
    const entity_offset = EntityID.get_absolute_index(entity);

    for (let i = 0; i < instance_count; ++i) {
      const entity_index = entity_offset + i;
      this.data.allows_cursor_events[entity_index] = 0;
      this.data.auto_size[entity_index] = 0;
      this.data.was_cursor_inside[entity_index] = 0;
      this.data.is_cursor_inside[entity_index] = 0;
      this.data.was_clicked[entity_index] = 0;
      this.data.is_clicked[entity_index] = 0;
      this.data.is_pressed[entity_index] = 0;
      this.data.was_pressed[entity_index] = 0;
      this.data.color.r[entity_index] = 0;
      this.data.color.g[entity_index] = 0;
      this.data.color.b[entity_index] = 0;
      this.data.color.a[entity_index] = 0;
    }
  }

  static get_entity_data(entity, instance = 0) {
    const entity_index = EntityID.get_absolute_index(entity) + instance;
    const data_view = this.data_view_allocator.allocate();
    data_view.fragment = this;
    data_view.view_entity(entity_index);
    return data_view;
  }

  static duplicate_entity_data(entity, instance = 0) {
    const data = {};
    const entity_index = EntityID.get_absolute_index(entity) + instance;
    data.allows_cursor_events = this.data.allows_cursor_events[entity_index];
    data.auto_size = this.data.auto_size[entity_index];
    data.was_cursor_inside = this.data.was_cursor_inside[entity_index];
    data.is_cursor_inside = this.data.is_cursor_inside[entity_index];
    data.was_clicked = this.data.was_clicked[entity_index];
    data.is_clicked = this.data.is_clicked[entity_index];
    data.is_pressed = this.data.is_pressed[entity_index];
    data.was_pressed = this.data.was_pressed[entity_index];
    data.color = {
      r: this.data.color.r[entity_index],
      g: this.data.color.g[entity_index],
      b: this.data.color.b[entity_index],
      a: this.data.color.a[entity_index],
    };
    data.dirty = this.data.dirty[entity_index];
    return data;
  }

  static to_gpu_data() {
    if (!this.data) this.initialize();

    if (!this.data.gpu_data_dirty) {
      return {
        element_data_buffer: this.data.element_data_buffer,
      };
    }

    this.rebuild_buffers();

    return {
      element_data_buffer: this.data.element_data_buffer,
    };
  }

  static rebuild_buffers() {
    {
      const gpu_data = new Float32Array(Math.max(this.size * 4, 4));
      let offset = 0;
      for (let i = 0; i < this.size; i++) {
        gpu_data[offset + 0] = this.data.color.r[i];
        gpu_data[offset + 1] = this.data.color.g[i];
        gpu_data[offset + 2] = this.data.color.b[i];
        gpu_data[offset + 3] = this.data.color.a[i];
        offset += 4;
      }

      if (
        !this.data.element_data_buffer ||
        this.data.element_data_buffer.config.size < gpu_data.byteLength
      ) {
        this.data.element_data_buffer = Buffer.create({
          name: element_data_buffer_name,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          raw_data: gpu_data,
          force: true,
        });

        Renderer.get().mark_bind_groups_dirty(true);
        global_dispatcher.dispatch(
          element_data_event,
          this.data.element_data_buffer,
        );
      } else {
        this.data.element_data_buffer.write(gpu_data);
      }

      global_dispatcher.dispatch(element_data_update_event);
    }

    this.data.gpu_data_dirty = false;
  }

  static async sync_buffers() {}
}

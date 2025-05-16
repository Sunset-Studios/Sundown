import { EntityManager } from "../entity.js";
import { EntityFlags } from "../../minimal.js";
import { Name } from "../../../utility/names.js";
import { warn, error } from "../../../utility/logging.js";

/**
 * Provides a view-like interface to access fragment data stored within a Solar ECS chunk.
 * A single view instance can be reused to point to different entities over time using the `view()` method.
 */
export class SolarFragmentView {
  entity = null;
  chunk = null;
  slot = -1;
  instance = -1;
  fragment_id = null;
  field_specs = null; // { field_name: { ctor, elements, default } }
  store_hashes = null; // snake_case per-chunk store keys

  /**
   * Creates a view for a specific fragment.
   * Call `view()` to point this view to specific entity data within a chunk.
   * @param {object} fragment - The fragment definition.
   */
  constructor(fragment) {
    this.fragment_id = fragment.id;
    this.field_specs = fragment.fields;
    this.store_hashes = Object.create(null);

    // Dynamically create getters/setters for each field
    const field_entries = Object.entries(this.field_specs);
    for (let i = 0; i < field_entries.length; i++) {
      const [field_name, spec] = field_entries[i];

      if (spec.is_container) {
        this.store_hashes[field_name] = Name.from(`${fragment.id}.${field_name}`);
      }

      const custom_get =
        typeof spec.getter === "string"
          ? new Function(spec.getter)
          : typeof spec.getter === "function"
            ? spec.getter
            : null;

      const custom_set =
        typeof spec.setter === "string"
          ? new Function("value", spec.setter)
          : typeof spec.setter === "function"
            ? spec.setter
            : null;

      Object.defineProperty(this, field_name, {
        get() {
          return this._get_field(field_name, spec, custom_get)
        },
        set(value) {
          this._set_field(field_name, spec, custom_set, value)
        },
        enumerable: true,
        configurable: true,
      });
    }
  }

  /**
   * Points the view to a specific entity's fragment data within a chunk.
   * @param {object} entity - The entity to point the view to.
   * @param {object} chunk - The Solar chunk containing the data.
   * @param {number} slot - The base slot index within the chunk.
   * @param {number} instance - The instance index within the slot (for instanced fragments).
   * @returns {SolarFragmentView} The view instance for chaining.
   */
  view(entity, chunk, slot, instance) {
    this.entity = entity;
    this.chunk = chunk;
    this.slot = slot;
    this.instance = instance;
    return this; // Return this for potential chaining
  }

  /**
   * All of your "get" logic in one place
   */
  _get_field(field_name, spec, custom_get) {
    if (!this.chunk) {
      return null;
    }

    if (spec.is_container) {
      const container = this.chunk.variable_stores.get(this.store_hashes[field_name]);
      if (!container) {
        warn(`SolarFragmentView: No variable store for '${field_name}'.`);
        return null;
      }
      return container.get_data_for_entity(this.entity);
    }

    const typed_array = this.chunk.fragment_views[this.fragment_id]?.[field_name];
    if (!typed_array) {
      warn(
        `SolarFragmentView: Fragment '${this.fragment_id}' or field '${field_name}' not found in the current chunk.`
      );
      return null;
    }

    const element_offset = (this.slot + this.instance) * spec.elements;
    if (element_offset < 0 || element_offset + spec.elements > typed_array.length) {
      error(
        `SolarFragmentView: Calculated offset is out of bounds for field '${field_name}'. Slot: ${this.slot}, Instance: ${this.instance}, Elements: ${spec.elements}, Array Length: ${typed_array.length}`
      );
      return null;
    }

    if (custom_get) {
      return custom_get.call(this, typed_array, element_offset);
    }

    return spec.elements === 1
      ? typed_array[element_offset]
      : typed_array.subarray(element_offset, element_offset + spec.elements);
  }

  /**
   * All of your "set" logic in one place
   */
  _set_field(field_name, spec, custom_set, value) {
    if (!this.chunk) {
      return;
    }

    if (spec.is_container) {
      const container = this.chunk.variable_stores.get(this.store_hashes[field_name]);
      if (!container) {
        error(`SolarFragmentView: No variable store for '${field_name}'.`);
        return;
      }

      let data_array;
      if (value instanceof spec.ctor) {
        data_array = value;
      } else if (Array.isArray(value) || ArrayBuffer.isView(value)) {
        data_array = new spec.ctor(value);
      } else {
        warn(`SolarFragmentView: Invalid value for '${field_name}'.`);
        return;
      }

      container.update(this.entity, data_array);

      this.chunk.flags_meta[this.slot + this.instance] |= EntityFlags.DIRTY;

      this.chunk.mark_dirty();

      return;
    }

    const typed_array = this.chunk.fragment_views[this.fragment_id]?.[field_name];
    if (!typed_array) {
      warn(
        `SolarFragmentView: Fragment '${this.fragment_id}' or field '${field_name}' not found in the current chunk. Cannot set value.`
      );
      return;
    }

    const element_offset = (this.slot + this.instance) * spec.elements;
    if (element_offset < 0 || element_offset + spec.elements > typed_array.length) {
      error(
        `SolarFragmentView: Calculated offset is out of bounds for field '${field_name}'. Slot: ${this.slot}, Instance: ${this.instance}, Elements: ${spec.elements}, Array Length: ${typed_array.length}`
      );
      return;
    }

    if (custom_set) {
      custom_set.call(this, value, typed_array, element_offset);
    } else if (spec.elements === 1) {
      typed_array[element_offset] = typed_array instanceof BigInt64Array ? BigInt(value) : value;
    } else {
      if (!Array.isArray(value) && !(value instanceof spec.ctor)) {
        warn(
          `Invalid value type for ${field_name}. Expected array or ${spec.ctor.name}. Got ${typeof value}`
        );
        return;
      }
      if (value.length > spec.elements) {
        warn(
          `Invalid value length for ${field_name}. Expected array of length ${spec.elements}, got ${value.length}`
        );
        return;
      }

      typed_array.set(value, element_offset);
    }

    this.chunk.flags_meta[this.slot + this.instance] |= EntityFlags.DIRTY;

    this.chunk.mark_dirty();
  }
}

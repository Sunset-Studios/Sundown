import { Element3D } from "./element.js";
import { TypedStack } from "../../memory/container.js";

export class Element3DPool {
  elements = null;
  available_elements = null;
  active_elements = null;
  active_count = 0;
  element_to_active_index = new Map();

  constructor(scene, pool_size = 0) {
    this.elements = new BigInt64Array(pool_size);
    this.active_elements = new Uint32Array(pool_size);
    this.available_elements = new TypedStack(pool_size, Uint32Array);
    for (let i = 0; i < pool_size; i++) {
      const element = Element3D.create(scene, {}, null, null, [], false);
      this.elements[i] = element;
      this.available_elements.push(i);
    }
  }

  create(scene, config, material, parent = null, children = []) {
    const should_construct = this.available_elements.is_empty();

    this._ensure_elements_capacity();

    const element_id = this.available_elements.pop();

    if (!should_construct) {
      const element = this.elements[element_id];
      Element3D.set_config(scene, element, config);
      Element3D.set_material(scene, element, material);
      Element3D.set_parent(scene, element, parent);
      Element3D.set_children(scene, element, children);
      Element3D.set_visible(scene, element, true);
    } else {
      this.elements[element_id] = Element3D.create(scene, config, material, parent, children);
    }

    this._ensure_active_capacity();
    this.active_elements[this.active_count] = element_id;
    this.element_to_active_index.set(element_id, this.active_count);
    this.active_count++;

    return this.elements[element_id];
  }

  destroy(element_id) {
    if (!this.element_to_active_index.has(element_id)) {
      return;
    }

    const element = this.elements[element_id];
    Element3D.set_material(scene, element, null);
    Element3D.set_parent(scene, element, null);
    Element3D.set_children(scene, element, []);
    Element3D.set_visible(scene, element, false);

    this._remove_from_active(element_id);
    this.available_elements.push(element_id);
  }

  deactivate(element_id) {
    if (!this.element_to_active_index.has(element_id)) {
      return;
    }
    this._remove_from_active(element_id);
  }

  activate(element_id) {
    if (this.element_to_active_index.has(element_id)) {
      return;
    }
    this._ensure_active_capacity();
    this.active_elements[this.active_count] = element_id;
    this.element_to_active_index.set(element_id, this.active_count);
    this.active_count++;
  }

  _ensure_active_capacity() {
    if (this.active_count >= this.active_elements.length) {
      const new_size = Math.max(16, this.active_elements.length * 2);
      const new_active_elements = new Uint32Array(new_size);
      new_active_elements.set(this.active_elements);
      this.active_elements = new_active_elements;
    }
  }

  _ensure_elements_capacity() {
    if (this.available_elements.is_empty()) {
      const old_size = this.elements.length;
      const new_size = old_size * 2;
      const new_elements = new Uint32Array(new_size);
      new_elements.set(this.elements);
      this.elements = new_elements;
      for (let i = old_size; i < new_size; i++) {
        this.available_elements.push(i);
      }
    }
  }


  _remove_from_active(element_id) {
    const index = this.element_to_active_index.get(element_id);
    this.element_to_active_index.delete(element_id);
    
    this.active_count--;
    if (index < this.active_count) {
      const last_element = this.active_elements[this.active_count];
      this.active_elements[index] = last_element;
      this.element_to_active_index.set(last_element, index);
    }
  }

  get_element(element_id) {
    if (!this.element_to_active_index.has(element_id)) {
      return null;
    }
    return this.elements[element_id];
  }

  is_active(element_id) {
    return this.element_to_active_index.has(element_id);
  }

  get_active_count() {
    return this.active_count;
  }

  get_pool_size() {
    return this.elements.length;
  }

  clear() {
    this.available_elements.clear();
    for (let i = 0; i < this.elements.length; i++) {
      this.available_elements.push(i);
    }
    this.active_count = 0;
    this.element_to_active_index.clear();
  }

  static create(scene, pool_size = 0) {
    const element_pool = new Element3DPool(scene, pool_size);
    return element_pool;
  }
}

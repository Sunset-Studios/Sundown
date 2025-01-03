const max_objects = 5000000;

/**
 * A class representing a statically sized BigInt64Array with reset functionality.
 */
export class StaticIntArray {
  #max_capacity;

  /**
   * @param {number} capacity - The fixed capacity of the array.
   */
  constructor(capacity) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error("Capacity must be a positive integer.");
    }
    this.buffer = new BigInt64Array(capacity);
    this.#max_capacity = capacity;
    this.size = 0;
  }

  /**
   * Get the value at the specified index.
   * @param {number} index - The index to retrieve the value from.
   * @returns {bigint} The value at the specified index.
   */
  get(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.size) {
      throw new Error(`Index out of bounds: ${index}`);
    }
    return this.buffer[index];
  }

  /**
   * Add a value to the array.
   * @param {bigint} value - The value to add.
   */
  add(value) {
    this.buffer[this.size++] = BigInt(value);
  }

  /**
   * Reset the size of the array to zero.
   */
  reset() {
    this.size = 0;
  }

  /**
   * Get the fixed size of the array.
   * @returns {number} The size of the array.
   */
  get capacity() {
    return this.#max_capacity;
  }

  /**
   * Get the current size of the array.
   * @returns {number} The size of the array.
   */
  get length() {
    return this.size;
  }

  /**
   * Iterator for the array.
   * @returns {Iterator} An iterator for the array elements.
   */
  [Symbol.iterator]() {
    let index = 0;
    return {
      next: () => {
        if (index < this.#max_capacity) {
          return { value: this.buffer[index++], done: false };
        } else {
          return { done: true };
        }
      },
    };
  }
}

/**
 * A class representing a resizable BitArray.
 */
export class ResizableBitArray {
  #buffer;
  #size;

  /**
   * @param {number} initialCapacity - The initial capacity of the array in bits.
   */
  constructor(initial_capacity = 64) {
    if (!Number.isInteger(initial_capacity) || initial_capacity <= 0) {
      throw new Error("Initial capacity must be a positive integer.");
    }
    this.#buffer = new Uint32Array(Math.ceil(initial_capacity / 32));
    this.#size = 0;
  }

  /**
   * Get the bit value at the specified index.
   * @param {number} index - The index to retrieve the bit from.
   * @returns {boolean} The bit value at the specified index.
   */
  get(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.#size) {
      throw new Error(`Index out of bounds: ${index}`);
    }
    const array_index = Math.floor(index / 32);
    const bit_index = index % 32;
    return (this.#buffer[array_index] & (1 << bit_index)) !== 0;
  }

  /**
   * Set the bit value at the specified index.
   * @param {number} index - The index to set the bit.
   * @param {boolean} value - The bit value to set.
   */
  set(index, value) {
    if (!Number.isInteger(index) || index < 0) {
      throw new Error(`Invalid index: ${index}`);
    }
    if (index >= this.#size) {
      this.#resize(index + 1);
    }
    const array_index = Math.floor(index / 32);
    const bit_index = index % 32;
    if (value) {
      this.#buffer[array_index] |= 1 << bit_index;
    } else {
      this.#buffer[array_index] &= ~(1 << bit_index);
    }
  }

  /**
   * Resize the bit array to accommodate the new size.
   * @param {number} newSize - The new size of the array in bits.
   */
  #resize(new_size) {
    const new_capacity = Math.max(new_size, this.#buffer.length * 32 * 2);
    const new_buffer = new Uint32Array(Math.ceil(new_capacity / 32));
    new_buffer.set(this.#buffer);
    this.#buffer = new_buffer;
    this.#size = new_size;
  }

  /**
   * Get the current size of the array in bits.
   * @returns {number} The size of the array in bits.
   */
  get length() {
    return this.#size;
  }
}

/**
 * A node in the tree, stored in a pool.
 */
class TreeNode {
  constructor() {
    this.data = null;
    this.parent_idx = -1;
    this.children = new Uint32Array(16); // Initial child capacity
    this.child_count = 0;
  }

  reset() {
    this.data = null;
    this.parent_idx = -1;
    this.child_count = 0;
  }
}

/**
 * A tree container that can store multiple root nodes and an arbitrary number of children per node.
 * Uses a free list implementation to pool TreeNode objects.
 */
export class Tree {
  constructor(initial_capacity = 1024) {
    // Node storage
    this.nodes = new Array(initial_capacity);
    for (let i = 0; i < initial_capacity; i++) {
      this.nodes[i] = new TreeNode();
    }
    
    // Free list tracking
    this.free_list = new Uint32Array(initial_capacity);
    this.free_count = initial_capacity;
    for (let i = 0; i < initial_capacity; i++) {
      this.free_list[i] = i;
    }

    // Root storage
    this.roots = new Uint32Array(16); // Initial root capacity
    this.root_count = 0;

    // Quick node lookup
    this.node_map = new Map(); // Maps data -> node_index
    
    this.size = 0;
  }

  /**
   * Allocates a node from the pool
   * @returns {number} Index of allocated node
   */
  #allocate_node() {
    if (this.free_count === 0) {
      // Grow the pool
      const old_capacity = this.nodes.length;
      const new_capacity = old_capacity * 2;
      
      // Grow nodes array
      for (let i = old_capacity; i < new_capacity; i++) {
        this.nodes[i] = new TreeNode();
      }

      // Grow free list
      const new_free_list = new Uint32Array(new_capacity);
      new_free_list.set(this.free_list);
      for (let i = old_capacity; i < new_capacity; i++) {
        new_free_list[i - old_capacity] = i;
      }
      this.free_list = new_free_list;
      this.free_count = new_capacity - old_capacity;
    }

    return this.free_list[--this.free_count];
  }

  /**
   * Returns a node to the pool
   * @param {number} node_idx Index of node to free
   */
  #free_node(node_idx) {
    this.nodes[node_idx].reset();
    this.free_list[this.free_count++] = node_idx;
  }

  /**
   * Add a child node to a parent node.
   * @param {*} parent_data - The data of the parent node to add to. If null, creates root.
   * @param {*} child_data - The data to add as a child.
   * @returns {number} The index of the newly created child node.
   */
  add(parent_data, child_data) {
    const child_idx = this.#allocate_node();
    const child = this.nodes[child_idx];
    child.data = child_data;
    
    if (parent_data === null) {
      if (this.root_count === this.roots.length) {
        // Grow roots array
        const new_roots = new Uint32Array(this.roots.length * 2);
        new_roots.set(this.roots);
        this.roots = new_roots;
      }
      this.roots[this.root_count++] = child_idx;
    } else {
      const parent_idx = this.node_map.get(parent_data);
      if (parent_idx !== undefined) {
        const parent = this.nodes[parent_idx];
        child.parent_idx = parent_idx;
        
        if (parent.child_count === parent.children.length) {
          // Grow children array
          const new_children = new Uint32Array(parent.children.length * 2);
          new_children.set(parent.children);
          parent.children = new_children;
        }
        parent.children[parent.child_count++] = child_idx;
      }
    }

    this.node_map.set(child_data, child_idx);
    this.size++;
    return child_idx;
  }

  /**
   * Remove a node from the tree.
   * @param {*} data - The data of the node to remove.
   */
  remove(data) {
    const node_idx = this.node_map.get(data);
    if (node_idx === undefined) return;

    const node = this.nodes[node_idx];
    
    // Remove from parent's children or roots
    if (node.parent_idx !== -1) {
      const parent = this.nodes[node.parent_idx];
      const idx = Array.prototype.indexOf.call(parent.children.slice(0, parent.child_count), node_idx);
      if (idx !== -1) {
        parent.children.copyWithin(idx, idx + 1, parent.child_count);
        parent.child_count--;
      }
    } else {
      const idx = Array.prototype.indexOf.call(this.roots.slice(0, this.root_count), node_idx);
      if (idx !== -1) {
        this.roots.copyWithin(idx, idx + 1, this.root_count);
        this.root_count--;
      }
    }

    // Update children's parent references
    for (let i = 0; i < node.child_count; i++) {
      const child = this.nodes[node.children[i]];
      child.parent_idx = node.parent_idx;
    }

    this.node_map.delete(data);
    this.#free_node(node_idx);
    this.size--;
  }

  /**
   * Find a node with the specified data.
   * @param {*} data - The data to search for.
   * @returns {number|undefined} The index of the found node or undefined if not found.
   */
  find_node(data) {
    return this.node_map.get(data);
  }

  /**
   * Find the parent of a node.
   * @param {*} data - The data of the node to find the parent of.
   * @returns {*|null} The parent data or null if not found.
   */
  find_parent(data) {
    const node_idx = this.node_map.get(data);
    if (node_idx === undefined) return null;
    
    const node = this.nodes[node_idx];
    return node.parent_idx === -1 ? null : this.nodes[node.parent_idx].data;
  }

  /**
   * Find the children of a node.
   * @param {*} data - The data of the node to find the children of.
   * @returns {Array|null} Array of child data values or null if node not found.
   */
  find_children(data) {
    const node_idx = this.node_map.get(data);
    if (node_idx === undefined) return null;
    
    const node = this.nodes[node_idx];
    return Array.from(node.children.slice(0, node.child_count))
      .map(idx => this.nodes[idx].data);
  }

  /**
   * Perform a breadth-first search starting from a specific node.
   * @param {number} start_node_idx - The index of the node to start searching from.
   * @param {Function} callback - Function called for each node. If it returns non-null, search stops and returns that value.
   * @param {*} callback_data - Optional data passed to the callback.
   * @returns {*} The value returned by callback, or null if search completes without callback returning non-null.
   */
  #breadth_first_search_queue = null;
  breadth_first_search(start_node_idx, callback, callback_data) {
    if (!this.#breadth_first_search_queue) {
      this.#breadth_first_search_queue = new Uint32Array(max_objects);
    }

    this.#breadth_first_search_queue[0] = start_node_idx;

    let queue_idx_tail = 0;
    let queue_idx_head = 1;
    while (queue_idx_tail !== queue_idx_head) {
      const current_idx = this.#breadth_first_search_queue[queue_idx_tail++];
      const current = this.nodes[current_idx];

      const result = callback(current, callback_data);
      if (result !== null) {
        return result;
      }

      for (let i = 0; i < current.child_count; i++) {
        this.#breadth_first_search_queue[queue_idx_head++] = current.children[i];
      }
    }

    return null;
  }

  /**
   * Flatten the tree into an array using breadth-first traversal.
   * @param {TypedArrayConstructor} array_type - The type of array to create (default: Float32Array)
   * @returns {Object} Object containing result array and layer_counts array
   */
  #flatten_queue = null;
  flatten(array_type = Float32Array) {
    if (!this.#flatten_queue) {
      this.#flatten_queue = new Uint32Array(max_objects);
    }

    if (this.root_count === 0) return { result: null, layer_counts: [] };

    const layer_counts = [this.root_count];

    const result = new array_type(this.size);

    for (let i = 0; i < this.root_count; i++) {
      this.#flatten_queue[i] = this.roots[i];
    }

    let result_size = 0;
    let queue_idx_tail = 0;
    let queue_idx_head = this.root_count;

    let nodes_remaining_in_layer = this.root_count;
    let nodes_in_next_layer = 0;

    const is_bigint = array_type === BigInt64Array;

    while (queue_idx_tail !== queue_idx_head) {
      const current_idx = this.#flatten_queue[queue_idx_tail++];
      const current = this.nodes[current_idx];
      result[result_size++] = is_bigint ? BigInt(current.data) : current.data;
      nodes_remaining_in_layer--;

      for (let i = 0; i < current.child_count; i++) {
        this.#flatten_queue[queue_idx_head++] = current.children[i];
        nodes_in_next_layer++;
      }

      if (nodes_remaining_in_layer === 0 && nodes_in_next_layer > 0) {
        layer_counts.push(nodes_in_next_layer);
        nodes_remaining_in_layer = nodes_in_next_layer;
        nodes_in_next_layer = 0;
      }
    }

    return { result, layer_counts };
  }

  /**
   * Get a TreeNode from its index
   * @param {number} node_idx - The index of the node
   * @returns {TreeNode|null} The TreeNode object or null if index is invalid
   */
  get_node(node_idx) {
    if (node_idx === -1 || node_idx >= this.nodes.length) return null;
    return this.nodes[node_idx];
  }

  /**
   * Get the parent TreeNode of a node
   * @param {number} node_idx - The index of the node to get the parent of
   * @returns {TreeNode|null} The parent TreeNode or null if no parent
   */
  get_parent(node_idx) {
    const node = this.get_node(node_idx);
    if (!node) return null;
    return node.parent_idx === -1 ? null : this.nodes[node.parent_idx];
  }

  /**
   * Get the children TreeNodes of a node
   * @param {number} node_idx - The index of the node to get the children of
   * @returns {TreeNode[]} Array of child TreeNodes
   */
  get_children(node_idx) {
    const node = this.get_node(node_idx);
    if (!node) return [];
    
    return Array.from(node.children.slice(0, node.child_count))
      .map(child_idx => this.nodes[child_idx]);
  }

  /**
   * Add multiple child nodes to a parent node.
   * @param {*} parent_data - The data of the parent node to add to. If null, creates roots.
   * @param {Array} child_data_list - Array of data to add as children.
   * @param {boolean} replace_children - If true, removes existing children before adding new ones.
   * @param {boolean} unique - If true, removes any existing nodes with the same data before adding.
   * @returns {number[]} Array of indices of the newly created child nodes.
   */
  add_multiple(parent_data, child_data_list, replace_children = false, unique = true) {
    const parent_idx = parent_data === null ? -1 : this.node_map.get(parent_data);
    const parent = parent_idx === -1 ? null : this.nodes[parent_idx];

    // Handle child replacement if requested
    if (replace_children) {
      if (parent === null) {
        // Remove all roots
        for (let i = 0; i < this.root_count; i++) {
          const root_idx = this.roots[i];
          this.node_map.delete(this.nodes[root_idx].data);
          this.#free_node(root_idx);
        }
        this.root_count = 0;
        this.size = 0;
      } else {
        // Remove all children of parent
        for (let i = 0; i < parent.child_count; i++) {
          const child_idx = parent.children[i];
          this.node_map.delete(this.nodes[child_idx].data);
          this.#free_node(child_idx);
          this.size--;
        }
        parent.child_count = 0;
      }
    }

    // Remove existing nodes with same data if unique is true
    if (unique) {
      for (const child_data of child_data_list) {
        const existing_idx = this.node_map.get(child_data);
        if (existing_idx !== undefined) {
          this.remove(child_data);
        }
      }
    }

    // Add all new children
    const new_child_indices = [];
    for (const child_data of child_data_list) {
      const child_idx = this.#allocate_node();
      const child = this.nodes[child_idx];
      child.data = child_data;
      child.parent_idx = parent_idx;

      if (parent === null) {
        // Add as root
        if (this.root_count === this.roots.length) {
          // Grow roots array if needed
          const new_roots = new Uint32Array(this.roots.length * 2);
          new_roots.set(this.roots);
          this.roots = new_roots;
        }
        this.roots[this.root_count++] = child_idx;
      } else {
        // Add as child to parent
        if (parent.child_count === parent.children.length) {
          // Grow children array if needed
          const new_children = new Uint32Array(parent.children.length * 2);
          new_children.set(parent.children);
          parent.children = new_children;
        }
        parent.children[parent.child_count++] = child_idx;
      }

      this.node_map.set(child_data, child_idx);
      new_child_indices.push(child_idx);
      this.size++;
    }

    return new_child_indices;
  }
}

/**
 * A tree container that can store multiple root nodes and an arbitrary number of children per node.
 */
export class TypedTree {
  constructor(initial_capacity = 1024, data_fields = {}, array_type = Int32Array) {
    // Tree structure arrays
    this.array_type = array_type;
    this.parent = new array_type(initial_capacity);
    this.first_child = new array_type(initial_capacity);
    this.next_sibling = new array_type(initial_capacity);
    this.prev_sibling = new array_type(initial_capacity);

    // Initialize data storage based on field definitions
    this.data = {};
    for (const [field, config] of Object.entries(data_fields)) {
      if (config.vector) {
        this.data[field] = {};
        for (const axis of Object.keys(config.vector)) {
          this.data[field][axis] = new config.type.array(initial_capacity);
          this.data[field][axis].fill(config.default || 0);
        }
      } else {
        this.data[field] = new config.type.array(initial_capacity);
        this.data[field].fill(config.default || 0);
      }
    }

    // Initialize structure arrays
    this.parent.fill(-1);
    this.first_child.fill(-1);
    this.next_sibling.fill(-1);
    this.prev_sibling.fill(-1);

    // Track used/free nodes
    this.size = 0;
    this.free_list = new this.array_type(initial_capacity);
    for (let i = 0; i < initial_capacity - 1; i++) {
      this.free_list[i] = i + 1;
    }
    this.free_list[initial_capacity - 1] = -1;
    this.next_free = 0;
  }

  /**
   * Resize the tree to accommodate the new capacity.
   * @param {number} new_capacity - The new capacity of the tree.
   */
  resize(new_capacity) {
    const new_parent = new this.array_type(new_capacity);
    const new_first_child = new this.array_type(new_capacity);
    const new_next_sibling = new this.array_type(new_capacity);
    const new_prev_sibling = new this.array_type(new_capacity);
    const new_free_list = new this.array_type(new_capacity);

    // Copy structure arrays
    new_parent.set(this.parent);
    new_first_child.set(this.first_child);
    new_next_sibling.set(this.next_sibling);
    new_prev_sibling.set(this.prev_sibling);
    new_free_list.set(this.free_list);

    // Resize data arrays
    for (const [field, storage] of Object.entries(this.data)) {
      if (storage instanceof this.array_type) {
        const new_array = new storage.constructor(new_capacity);
        new_array.set(storage);
        new_array.fill(storage[0], storage.length); // Fill new space with default
        this.data[field] = new_array;
      } else {
        // Handle vector data
        for (const [axis, array] of Object.entries(storage)) {
          const new_array = new array.constructor(new_capacity);
          new_array.set(array);
          new_array.fill(array[0], array.length);
          storage[axis] = new_array;
        }
      }
    }

    // Initialize new structure slots
    new_parent.fill(-1, this.parent.length);
    new_first_child.fill(-1, this.first_child.length);
    new_next_sibling.fill(-1, this.next_sibling.length);
    new_prev_sibling.fill(-1, this.prev_sibling.length);

    // Setup new free list
    for (let i = this.parent.length; i < new_capacity - 1; i++) {
      new_free_list[i] = i + 1;
    }
    new_free_list[new_capacity - 1] = -1;

    if (this.next_free === -1) {
      this.next_free = this.parent.length;
    }

    this.parent = new_parent;
    this.first_child = new_first_child;
    this.next_sibling = new_next_sibling;
    this.prev_sibling = new_prev_sibling;
    this.free_list = new_free_list;
  }

  /**
   * Set the data for a node.
   * @param {number} node - The index of the node to set the data for.
   * @param {Object} data - The data to set for the node.
   */
  set_node_data(node, data) {
    for (const [field, value] of Object.entries(data)) {
      if (this.data[field] instanceof this.array_type) {
        this.data[field][node] = value;
      } else {
        // Handle vector data
        for (const [axis, val] of Object.entries(value)) {
          this.data[field][axis][node] = val;
        }
      }
    }
  }

  /**
   * Get the data for a node.
   * @param {number} node - The index of the node to get the data for.
   * @returns {Object} The data for the node.
   */
  get_node_data(node) {
    const data = {};
    for (const [field, storage] of Object.entries(this.data)) {
      if (storage instanceof this.array_type) {
        data[field] = storage[node];
      } else {
        // Handle vector data
        data[field] = {};
        for (const [axis, array] of Object.entries(storage)) {
          data[field][axis] = array[node];
        }
      }
    }
    return data;
  }

  /**
   * Add a child node to a parent node.
   * @param {number} parent - The index of the parent node to add the child to.
   * @param {number} child - The index of the child node to add.
   */
  add_child(parent, child) {
    if (parent === -1) return;

    // Set parent reference
    this.parent[child] = parent;

    // If parent has no children, make this the first child
    if (this.first_child[parent] === -1) {
      this.first_child[parent] = child;
      return;
    }

    // Otherwise, append to end of sibling list
    let current = this.first_child[parent];
    while (this.next_sibling[current] !== -1) {
      current = this.next_sibling[current];
    }
    this.next_sibling[current] = child;
    this.prev_sibling[child] = current;
  }

  /**
   * Remove a node from the tree.
   * @param {number} node - The index of the node to remove.
   */
  remove_node(node) {
    if (node === -1) return;

    // Handle parent's first child reference
    const parent = this.parent[node];
    if (parent !== -1 && this.first_child[parent] === node) {
      this.first_child[parent] = this.next_sibling[node];
    }

    // Handle sibling links
    const prev = this.prev_sibling[node];
    const next = this.next_sibling[node];

    if (prev !== -1) this.next_sibling[prev] = next;
    if (next !== -1) this.prev_sibling[next] = prev;

    // Remove all children recursively
    let child = this.first_child[node];
    while (child !== -1) {
      const next_child = this.next_sibling[child];
      this.remove_node(child);
      child = next_child;
    }

    this.free_node(node);
  }

  /**
   * Get the children of a node.
   * @param {number} node - The index of the node to get the children of.
   * @returns {number[]} The indices of the children nodes.
   */
  get_children(node) {
    const children = [];
    let child = this.first_child[node];
    while (child !== -1) {
      children.push(child);
      child = this.next_sibling[child];
    }
    return children;
  }

  /**
   * Get the parent of a node.
   * @param {number} node - The index of the node to get the parent of.
   * @returns {number} The index of the parent node.
   */
  get_parent(node) {
    return this.parent[node];
  }

  /**
   * Get the ancestors of a node.
   * @param {number} node - The index of the node to get the ancestors of.
   * @returns {number[]} The indices of the ancestor nodes.
   */
  get_ancestors(node) {
    const ancestors = [];
    let current = this.parent[node];
    while (current !== -1) {
      ancestors.push(current);
      current = this.parent[current];
    }
    return ancestors;
  }

  /**
   * Get the siblings of a node.
   * @param {number} node - The index of the node to get the siblings of.
   * @returns {number[]} The indices of the sibling nodes.
   */
  get_siblings(node) {
    const siblings = [];
    const parent = this.parent[node];
    if (parent === -1) return siblings;

    let sibling = this.first_child[parent];
    while (sibling !== -1) {
      if (sibling !== node) {
        siblings.push(sibling);
      }
      sibling = this.next_sibling[sibling];
    }
    return siblings;
  }
}

/**
 * A fixed-size stack implementation using TypedArrays.
 */
export class TypedStack {
  #buffer;
  #size;

  /**
   * Create a new TypedStack with the specified capacity and array type.
   * @param {number} capacity - The maximum capacity of the stack.
   * @param {TypedArrayConstructor} array_type - The type of TypedArray to use (e.g. Int32Array).
   */
  constructor(capacity, array_type = Int32Array) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error("Capacity must be a positive integer.");
    }
    this.#buffer = new array_type(capacity);
    this.#size = 0;
  }

  /**
   * Push a value onto the stack.
   * @param {number} value - The value to push.
   * @throws {Error} If the stack is full.
   */
  push(value) {
    if (this.#size >= this.#buffer.length) {
      throw new Error("Stack overflow");
    }
    this.#buffer[this.#size++] = value;
  }

  /**
   * Pop a value from the stack.
   * @returns {number} The popped value.
   * @throws {Error} If the stack is empty.
   */
  pop() {
    if (this.#size <= 0) {
      throw new Error("Stack underflow");
    }
    return this.#buffer[--this.#size];
  }

  /**
   * Peek at the top value on the stack without removing it.
   * @returns {number} The top value.
   * @throws {Error} If the stack is empty.
   */
  peek() {
    if (this.#size <= 0) {
      throw new Error("Stack is empty");
    }
    return this.#buffer[this.#size - 1];
  }

  /**
   * Resize the stack to the specified capacity.
   * @param {number} new_capacity - The new capacity of the stack.
   */
  resize(new_capacity) {
    const new_buffer = new this.#buffer.constructor(new_capacity);
    new_buffer.set(this.#buffer);
    this.#buffer = new_buffer;
  }

  /**
   * Check if the stack is empty.
   * @returns {boolean} True if the stack is empty.
   */
  is_empty() {
    return this.#size === 0;
  }

  /**
   * Check if the stack is full.
   * @returns {boolean} True if the stack is full.
   */
  is_full() {
    return this.#size === this.#buffer.length;
  }

  /**
   * Get the current size of the stack.
   * @returns {number} The current size.
   */
  get length() {
    return this.#size;
  }

  /**
   * Get the maximum capacity of the stack.
   * @returns {number} The maximum capacity.
   */
  get capacity() {
    return this.#buffer.length;
  }

  /**
   * Reset the stack to empty.
   */
  clear() {
    this.#size = 0;
  }
}

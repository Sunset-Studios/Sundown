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
            }
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
            this.#buffer[array_index] |= (1 << bit_index);
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
 * A node in the tree structure.
 */
class TreeNode {
    constructor(data) {
        this.data = data;
        this.children = [];
        this.parent = null;
    }
}

/**
 * A tree container that can store multiple root nodes and an arbitrary number of children per node.
 */
export class Tree {
    constructor() {
        this.roots = [];
    }

    /**
     * Add a child node to a parent node.
     * @param {*} parent_data - The data of the parent node to add to. If null, creates root.
     * @param {*} child_data - The data to add as a child.
     * @returns {TreeNode} The newly created child node.
     */
    add(parent_data, child_data) {
        const child = new TreeNode(child_data);

        if (parent_data === null) {
            this.roots.push(child);
            return;
        }

        const parent = this.find_node(parent_data);
        if (!parent) {
            throw new Error("Parent node not found");
        }

        child.parent = parent;
        parent.children.push(child);
    }

    add_multiple(parent_data, child_data_list, replace_children = false) {
        const parent = this.find_node(parent_data);
        const resolved_parent = parent_data === null ? null : parent;

        if (replace_children) {
            if (resolved_parent === null) {
                this.roots.length = 0;
            } else {
                resolved_parent.children.length = 0;
            }
        }

        for (const child_data of child_data_list) {
            const child = new TreeNode(child_data);
            child.parent = resolved_parent;

            if (resolved_parent === null) {
                this.roots.push(child);
            } else if (resolved_parent) {
                resolved_parent.children.push(child);
            }
        }
    }

    /**
     * Remove a node from the tree.
     * @param {*} data - The data of the node to remove.
     */
    remove(data) {
        const node = this.find_node(data);
        if (!node) return;
        if (node.parent) {
            const index = node.parent.children.findIndex(child => child === node);
            node.parent.children.splice(index, 1);
        } else {
            const index = this.roots.findIndex(root => root === node);
            this.roots.splice(index, 1);
        }
        for (let i = 0; i < node.children.length; i++) {
            node.children[i].parent = node.parent;
        }
    }

    /**
     * Find a node with the specified data.
     * @param {*} data - The data to search for.
     * @returns {TreeNode|null} The found node or null if not found.
     */
    find_node(data) {
        if (this.roots.length === 0) return null;

        for (let i = 0; i < this.roots.length; i++) {
            const root = this.roots[i];
            const result = this.breadth_first_search(root, (node) => {
                if (node.data === data) return node;
                return null;
            });
            if (result !== null) {
                return result;
            }
        }
        return null;
    }

    /**
     * Find the parent of a node.
     * @param {*} data - The data of the node to find the parent of.
     * @returns {TreeNode|null} The parent node or null if not found.
     */
    find_parent(data) {
        return this.find_node(data)?.parent;
    }

    /**
     * Find the children of a node.
     * @param {*} data - The data of the node to find the children of.
     * @returns {TreeNode[]} The children nodes.
     */
    find_children(data) {
        return this.find_node(data)?.children;
    }

    /**
     * Perform a breadth-first search starting from a specific node.
     * @param {TreeNode} start_node - The node to start searching from.
     * @param {Function} callback - Function called for each node. If it returns non-null, search stops and returns that value.
     * @returns {*} The value returned by callback, or null if search completes without callback returning non-null.
     */
    #breadth_first_search_queue = null;
    breadth_first_search(start_node, callback) {
        if (!this.#breadth_first_search_queue) {
            this.#breadth_first_search_queue = new Array(max_objects);
        }

        this.#breadth_first_search_queue[0] = start_node;

        let queue_idx_tail = 0;
        let queue_idx_head = 1;
        while (queue_idx_tail !== queue_idx_head) {
            const current = this.#breadth_first_search_queue[queue_idx_tail++];
            
            const result = callback(current);
            if (result !== null) {
                return result;
            }

            for (let i = 0; i < current.children.length; i++) {
                this.#breadth_first_search_queue[queue_idx_head++] = current.children[i];
            }
        }

        return null;
    }

    /**
     * Flatten the tree into an array using breadth-first traversal.
     * @returns {Array} Array containing all nodes in breadth-first order.
     */
    #flatten_queue = null;
    flatten(array_type = Float32Array) {
        if (!this.#flatten_queue) {
            this.#flatten_queue = new Array(max_objects);
        }

        if (this.roots.length === 0) return { result: null, layer_counts: [] };

        const layer_counts = [this.roots.length];

        const result = new array_type(this.roots.length);

        for (let i = 0; i < this.roots.length; i++) {
            this.#flatten_queue[i] = this.roots[i];
        }

        let result_size = 0;
        let queue_idx_tail = 0;
        let queue_idx_head = this.roots.length;

        let nodes_remaining_in_layer = this.roots.length;
        let nodes_in_next_layer = 0;

        const is_bigint = array_type === BigInt64Array;

        while (queue_idx_tail !== queue_idx_head) {
            const current = this.#flatten_queue[queue_idx_tail++];
            result[result_size++] = is_bigint ? BigInt(current.data) : current.data;
            nodes_remaining_in_layer--;

            for (let i = 0; i < current.children.length; i++) {
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

    /**
     * Flatten the tree into an array using breadth-first traversal.
     * @returns {Object} An object containing the flattened array and the layer counts.
    */
    #flatten_queue = null;
    flatten() {
        if (!this.#flatten_queue) {
            this.#flatten_queue = new this.array_type(this.parent.length);
        }

        // Find all root nodes (nodes with parent === -1)
        const roots = [];
        for (let i = 0; i < this.size; i++) {
            if (this.parent[i] === -1) {
                roots.push(i);
            }
        }

        if (roots.length === 0) return { result: null, layer_counts: [] };

        const layer_counts = [roots.length];

        const result = new this.array_type(this.size);
        this.#flatten_queue.set(roots);

        let result_size = 0;
        let queue_idx_tail = 0;
        let queue_idx_head = roots.length;

        let nodes_remaining_in_layer = roots.length;
        let nodes_in_next_layer = 0;

        while (queue_idx_tail !== queue_idx_head) {
            const current = this.#flatten_queue[queue_idx_tail++];
            result[result_size++] = current;
            nodes_remaining_in_layer--;

            // Add all children of current node to queue
            let child = this.first_child[current];
            while (child !== -1) {
                this.#flatten_queue[queue_idx_head++] = child;
                nodes_in_next_layer++;
                child = this.next_sibling[child];
            }

            if (nodes_remaining_in_layer === 0) {
                if (nodes_in_next_layer > 0) {
                    layer_counts.push(nodes_in_next_layer);
                }
                nodes_remaining_in_layer = nodes_in_next_layer;
                nodes_in_next_layer = 0;
            }
        }

        return { result, layer_counts };
    }

    /**
     * Perform a breadth-first search on the tree.
     * @param {Function} predicate - A function that takes a node index and returns true if the node matches the search criteria.
     * @returns {number|null} The index of the node that matches the search criteria, or null if no match is found.
    */
    #breadth_first_search_queue = null;
    breadth_first_search(predicate) {
        if (!this.#breadth_first_search_queue) {
            this.#breadth_first_search_queue = new this.array_type(this.parent.length);
        }

        // Find all root nodes (nodes with parent === -1)
        let queue_idx_head = 0;
        for (let i = 0; i < this.size; i++) {
            if (this.parent[i] === -1) {
                this.#breadth_first_search_queue[queue_idx_head++] = i;
            }
        }

        let queue_idx_tail = 0;
        while (queue_idx_tail < queue_idx_head) {
            const current = this.#breadth_first_search_queue[queue_idx_tail++];
            
            if (predicate(current)) {
                return current;
            }

            // Add all children to queue
            let child = this.first_child[current];
            while (child !== -1) {
                this.#breadth_first_search_queue[queue_idx_head++] = child;
                child = this.next_sibling[child];
            }
        }

        return null;
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

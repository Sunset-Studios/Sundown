import { ResizableBitArray } from './container';

const function_string = 'function'
const out_of_memory_error = 'Out of memory on allocator';
const invalid_allocator_index_error = 'Invalid allocator index';

// ================================================
// Regular Allocators
// ================================================

class FrameAllocator {
    /**
     * @param {number} max_objects - The maximum number of objects to allocate
     * @param {object|Function|number|string|boolean} template - Either a template object, a constructor function, or a primitive value
     */
    constructor(max_objects, template) {
        this.max_objects = max_objects;
        
        // Preallocate array with exact size
        this.buffer = new Array(max_objects);
        
        // Check if template is a primitive:
        if (
            template === null ||
            (typeof template !== function_string && typeof template !== 'object')
        ) {
            // For primitive types, simply fill the buffer with an object wrapping the value.
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = { value: template };
            }
        } else if (typeof template === function_string) {
            // For constructors, create new instances
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = new template();
            }
        } else if (Object.keys(template).length === 0) {
            // Fast path for empty objects
            const proto = Object.getPrototypeOf(template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(template);
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = Object.create(proto, prop_descriptors);
            }
        } else {
            // For non-empty plain objects, create a prototype once and use it for all allocations
            const proto = Object.getPrototypeOf(template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(template);
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = Object.create(proto, prop_descriptors);
            }
        }
        this.offset = 0;
    }

    /**
     * @returns {object|primitive} - The allocated object or value
     */
    allocate() {
        if (this.offset >= this.max_objects) {
            throw new Error(out_of_memory_error);
        }
        const index = this.offset;
        this.offset++;
        return this.buffer[index];
    }

    /**
     * @param {number} index - The index to get the object or value from
     * @returns {object|primitive} - The object or value at the index
     */
    get(index) {
        if (index < 0 || index >= this.offset) {
            throw new Error(invalid_allocator_index_error);
        }
        return this.buffer[index];
    }

    /**
     * Appends another frame allocator into this one.
     * @param {FrameAllocator} other - The other frame allocator to append
     */
    append(other) {
        for (let i = 0; i < other.length; i++) {
            this.buffer[this.offset + i] = other.data[i];
        }
        this.offset += other.length;
    }

    /**
     * Resets the frame allocator to its initial state.
     */
    reset() {
        this.offset = 0;
    }

    /**
     * Returns the data of the frame allocator.
     * @returns {Array} The data of the frame allocator.
     */
    get data() {
        return this.buffer;
    }

    /**
     * Returns the length of the frame allocator.
     * @returns {number} The length of the frame allocator.
     */
    get length() {
        return this.offset;
    }

    /**
     * Creates an iterator for the allocated objects/values in the buffer.
     * @returns {Iterator} An iterator for the allocated objects/values.
     */
    [Symbol.iterator]() {
        let index = 0;
        return {
            next: () => {
                if (index < this.offset) {
                    return { value: this.buffer[index++], done: false };
                } else {
                    return { done: true };
                }
            }
        };
    }
}



class FrameStackAllocator {
    /**
     * @param {number} max_objects - The maximum number of objects to allocate
     * @param {object|Function|number|string|boolean} template - Either a template object, a constructor function, or a primitive value
     */
    constructor(max_objects, template) {
        this.max_objects = max_objects;
        
        // Preallocate array with exact size
        this.buffer = new Array(max_objects);
        
        // Check if template is a primitive:
        if (
            template === null ||
            (typeof template !== function_string && typeof template !== 'object')
        ) {
            // For primitive types, simply fill the buffer with that value.
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = { value: template };
            }
        } else if (typeof template === function_string) {
            // For constructors, create new instances
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = new template();
            }
        } else if (Object.keys(template).length === 0) {
            // Fast path for empty objects
            const proto = Object.getPrototypeOf(template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(template);
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = Object.create(proto, prop_descriptors);
            }
        } else {
            // For non-empty plain objects, create a prototype once and use it for all allocations
            const proto = Object.getPrototypeOf(template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(template);
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = Object.create(proto, prop_descriptors);
            }
        }
        this.offset = 0;
    }
    
    /**
     * Pushes an object onto the frame stack allocator.
     * @returns {object|primitive} The pushed object or value
     */
    push() {
        if (this.offset >= this.max_objects) {
            throw new Error(out_of_memory_error);
        }
        const index = this.offset;
        this.offset++;
        return this.buffer[index];
    }

    /**
     * Pops an object from the frame stack allocator.
     * @returns {object|primitive} The popped object or value
     */
    pop() {
        if (this.offset <= 0) {
            throw new Error(out_of_memory_error);
        }
        this.offset--;
    }

    /**
     * Peeks at the top object from the frame stack allocator.
     * @returns {object|primitive} The top object or value
     */
    peek() {
        return this.offset > 0 ? this.buffer[this.offset - 1] : null;
    }

    /**
     * Appends another frame allocator into this one.
     * @param {FrameAllocator} other - The other frame allocator to append
     */
    append(other) {
        for (let i = 0; i < other.length; i++) {
            this.buffer[this.offset + i] = other.data[i];
        }
        this.offset += other.length;
    }

    /**
     * Resets the frame stack allocator to its initial state.
     */
    reset() {
        this.offset = 0;
    }

    /**
     * Returns the data of the frame stack allocator.
     * @returns {Array} The data of the frame stack allocator.
     */
    get data() {
        return this.buffer;
    }

    /**
     * Gets the current length of the frame stack allocator.
     * @returns {number} The current length of the frame stack allocator.
     */
    get length() {
        return this.offset;
    }

    /**
     * Gets the capacity of the frame stack allocator.
     * @returns {number} The capacity of the frame stack allocator.
     */
    get capacity() {
        return this.max_objects;
    }

    /**
     * Creates an iterator for the allocated objects/values in the buffer.
     * @returns {Iterator} An iterator for the allocated objects/values.
     */
    [Symbol.iterator]() {
        let index = 0;
        return {
            next: () => {
                if (index < this.offset) {
                    return { value: this.buffer[index++], done: false };
                } else {
                    return { done: true };
                }
            }
        };
    }
}

/**
 * A ring buffer allocator that allows for continuous allocation and deallocation
 * of fixed-size objects in a circular manner.
 */
class RingBufferAllocator {
    /**
     * @param {number} max_objects - The maximum number of objects the allocator can hold
     * @param {object|Function} template - Either a template object to clone or a constructor function
     */
    constructor(max_objects, template) {
        this.max_objects = max_objects;
        
        // Preallocate array with exact size
        this.buffer = new Array(max_objects);
        
        if (typeof template === function_string) {
            // Class instances - have to create new instances
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = new template();
            }
        } else if (Object.keys(template).length === 0) {
            // Fast path for empty objects
            const proto = Object.getPrototypeOf(template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(template);
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = Object.create(proto, prop_descriptors);
            }
        } else {
            // For non-empty plain objects, create a prototype once
            let proto = Object.getPrototypeOf(template);
            // Create objects sharing the same prototype
            const prop_descriptors = Object.getOwnPropertyDescriptors(template);
            for (let i = 0; i < max_objects; i++) {
                this.buffer[i] = Object.create(proto, prop_descriptors);
            }
        }
        this.head = 0;
        this.tail = 0;
    }

    /**
     * Allocates an object from the ring buffer.
     * @returns {object} The allocated object
     * @throws {Error} If the buffer is full
     */
    allocate() {
        const object = this.buffer[this.tail];
        this.tail = (this.tail + 1) % this.max_objects;
        return object;
    }

    /**
     * Deallocates the oldest object from the ring buffer.
     * @returns {object} The deallocated object
     * @throws {Error} If the buffer is empty
     */
    deallocate() {
        const object = this.buffer[this.head];
        this.head = (this.head + 1) % this.max_objects;
        return object;
    }

    /**
     * @param {number} index - The index to get the object from
     * @returns {object} - The object at the index
     * @throws {Error} If the index is out of bounds
     */
    get(index) {
        const actual_index = (this.head + index) % this.max_objects;
        return this.buffer[actual_index];
    }

    /**
     * Appends another frame allocator into this one.
     * @param {FrameAllocator} other - The other frame allocator to append
     */
    append(other) {
        for (let i = 0; i < other.length; i++) {
            const index = (this.tail + i) % this.max_objects;
            this.buffer[index] = other.data[i];
        }
        this.tail = (this.tail + other.length) % this.max_objects;
    }

    /**
     * Resets the ring buffer to its initial state.
     */
    reset() {
        this.head = 0;
        this.tail = 0;
    }

    /**
     * Returns the data of the ring buffer.
     * @returns {Array} The data of the ring buffer.
     */
    get data() {
        return this.buffer;
    }

    /**
     * Returns the number of allocated objects in the ring buffer.
     * @returns {number} The number of allocated objects in the ring buffer.
     */
    get length() {
        return this.tail - this.head;
    }

    /**
     * Returns the capacity of the ring buffer.
     * @returns {number} The capacity of the ring buffer.
     */
    get capacity() {
        return this.max_objects;
    }

    /**
     * Creates an iterator for the allocated objects in the ring buffer.
     * @returns {Iterator} An iterator for the allocated objects.
     */
    [Symbol.iterator]() {
        let index = 0;
        return {
            next: () => {
                if (index < this.max_objects) {
                    return { value: this.get(index++), done: false };
                } else {
                    return { done: true };
                }
            }
        };
    }
}

/**
 * A free list allocator that allows for continuous allocation and deallocation
 * of fixed-size objects in a free list manner with dynamic resizing.
 */
class FreeListAllocator {
    /**
     * A free list allocator that allows for continuous allocation and deallocation
     * of fixed-size objects in a free list manner with dynamic resizing.
     */
    #initial_capacity = 0;
    #template = null;
    
    /**
     * @param {number} initial_capacity - The initial capacity of the allocator
     * @param {object|Function} template - Either a template object to clone or a constructor function
     */
    constructor(initial_capacity = 16, template = null) {
        this.#initial_capacity = initial_capacity;
        this.#template = template;
        
        // Initialize buffer and free list
        this.buffer = new Array(initial_capacity);
        this.free_list = new Array(initial_capacity);
        
        for (let i = 0; i < initial_capacity; i++) {
            this.free_list[i] = i;
        }
        
        // Initialize buffer based on template
        this._initialize_buffer(0, initial_capacity);
    }
    
    /**
     * Initializes buffer elements in the specified range
     * @param {number} start - Start index (inclusive)
     * @param {number} end - End index (exclusive)
     * @private
     */
    _initialize_buffer(start, end) {
        // Check if template is a primitive:
        if (
            this.#template === null ||
            (typeof this.#template !== function_string && typeof this.#template !== 'object')
        ) {
            // For primitive types, simply fill the buffer with an object wrapping the value.
            for (let i = start; i < end; i++) {
                this.buffer[i] = { value: this.#template };
            }
        } else if (typeof this.#template === function_string) {
            // For constructors, create new instances
            for (let i = start; i < end; i++) {
                this.buffer[i] = new this.#template();
            }
        } else if (Object.keys(this.#template).length === 0) {
            // Fast path for empty objects
            const proto = Object.getPrototypeOf(this.#template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(this.#template);
            for (let i = start; i < end; i++) {
                this.buffer[i] = Object.create(proto, prop_descriptors);
            }
        } else {
            // For non-empty plain objects, create a prototype once and use it for all allocations
            const proto = Object.getPrototypeOf(this.#template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(this.#template);
            for (let i = start; i < end; i++) {
                this.buffer[i] = Object.create(proto, prop_descriptors);
            }
        }
    }
    
    /**
     * Resizes the buffer to accommodate more elements
     * @param {number} new_capacity - The new capacity
     * @private
     */
    _resize(new_capacity) {
        const old_buffer = this.buffer;
        const old_capacity = this.buffer.length;
        
        // Create new buffer with increased capacity
        this.buffer = new Array(new_capacity);
        
        // Copy existing elements
        for (let i = 0; i < old_capacity; i++) {
            this.buffer[i] = old_buffer[i];
        }
        
        // Initialize new elements
        this._initialize_buffer(old_capacity, new_capacity);
        
        // Add new indices to free list
        for (let i = old_capacity; i < new_capacity; i++) {
            this.free_list.push(i);
        }
    }

    /**
     * Allocates an object from the free list allocator.
     * @returns {object} The allocated object
     * @throws {Error} If allocation fails
     */
    allocate() {
        if (this.free_list.length === 0) {
            // Double the capacity when we run out of space
            const new_capacity = this.buffer.length * 2;
            this._resize(new_capacity);
        }
        
        const index = this.free_list.pop();
        return this.buffer[index];
    }

    /**
     * Deallocates an object from the free list allocator.
     * @param {object} object - The object to deallocate
     * @throws {Error} If the object is not found in the buffer
     */
    deallocate(object) {
        const index = this.buffer.indexOf(object);
        if (index === -1) {
            throw new Error(invalid_allocator_index_error);
        }
        
        // Check if index is already in free_list to avoid duplicates
        if (!this.free_list.includes(index)) {
            this.free_list.push(index);
        }
        
        // Consider shrinking the buffer if it's too empty
        if (this.free_list.length > this.buffer.length * 0.75 && this.buffer.length > this.#initial_capacity * 2) {
            const new_capacity = Math.max(this.#initial_capacity, Math.floor(this.buffer.length / 2));
            this._shrink(new_capacity);
        }
    }
    
    /**
     * Shrinks the buffer to reduce memory usage
     * @param {number} new_capacity - The new capacity
     * @private
     */
    _shrink(new_capacity) {
        // Create new arrays with reduced capacity
        const new_buffer = new Array(new_capacity);
        const new_free_list = [];
        
        // Track which objects are still in use
        const in_use = new Set();
        for (let i = 0; i < this.buffer.length; i++) {
            if (!this.free_list.includes(i)) {
                in_use.add(i);
            }
        }
        
        // Copy only the objects still in use
        let new_index = 0;
        for (let old_index = 0; old_index < this.buffer.length; old_index++) {
            if (in_use.has(old_index)) {
                if (new_index < new_capacity) {
                    new_buffer[new_index] = this.buffer[old_index];
                    new_index++;
                } else {
                    // If we can't fit all used objects, abort shrinking
                    return;
                }
            }
        }
        
        // Fill remaining slots and populate free list
        this.#template = this.buffer[0]; // Use existing object as template
        this._initialize_buffer(new_index, new_capacity);
        
        for (let i = new_index; i < new_capacity; i++) {
            new_free_list.push(i);
        }
        
        this.buffer = new_buffer;
        this.free_list = new_free_list;
    }

    /**
     * Resets the free list allocator to its initial state.
     */
    reset() {
        // If we've grown beyond initial capacity, resize back down
        if (this.buffer.length > this.#initial_capacity) {
            this.buffer = new Array(this.#initial_capacity);
            this._initialize_buffer(0, this.#initial_capacity);
        }
        
        // Reset free list
        this.free_list = new Array(this.#initial_capacity);
        for (let i = 0; i < this.#initial_capacity; i++) {
            this.free_list[i] = i;
        }
    }

    /**
     * Returns the object at the specified index.
     * @param {number} index - The index of the object to return
     * @returns {object} The object at the specified index
     */
    get(index) {
        if (index < 0 || index >= this.buffer.length) {
            throw new Error(invalid_allocator_index_error);
        }
        return this.buffer[index];
    }

    /**
     * Returns the current capacity of the free list allocator.
     * @returns {number} The current capacity of the free list allocator.
     */
    get capacity() {
        return this.buffer.length;
    }

    /**
     * Returns the number of allocated objects in the free list allocator.
     * @returns {number} The number of allocated objects.
     */
    get length() {
        return this.buffer.length - this.free_list.length;
    }

    /**
     * Returns the data of the free list allocator.
     * @returns {Array} The data of the free list allocator.
     */
    get data() {
        return this.buffer;
    }
}

/**
 * A resizable allocator that supports random access operations.
 * Allows adding or removing elements from any position in the collection.
 */
class RandomAccessAllocator {
    #size = 0;
    #capacity = 0;
    #template = null;
    #buffer = null;

    /**
     * @param {number} initial_capacity - The initial capacity of the allocator
     * @param {object|Function|number|string|boolean} template - Either a template object, a constructor function, or a primitive value
     */
    constructor(initial_capacity = 16, template = null) {
        this.#capacity = initial_capacity;
        this.#size = 0;
        this.#template = template;
        
        // Initialize buffer with the specified capacity
        this.#buffer = new Array(this.#capacity);
        
        // Initialize buffer elements based on template type
        if (template !== null) {
            this._initialize_buffer(0, this.#capacity);
        }
    }
    
    /**
     * Initializes buffer elements in the specified range
     * @param {number} start - Start index (inclusive)
     * @param {number} end - End index (exclusive)
     * @private
     */
    _initialize_buffer(start, end) {
        const function_string = 'function';
        
        // Check if template is a primitive
        if (typeof this.#template !== function_string && typeof this.#template !== 'object') {
            for (let i = start; i < end; i++) {
                this.#buffer[i] = { value: this.#template };
            }
        } else if (typeof this.#template === function_string) {
            // For constructors, create new instances
            for (let i = start; i < end; i++) {
                this.#buffer[i] = new this.#template();
            }
        } else if (Object.keys(this.#template).length === 0) {
            // Fast path for empty objects
            const proto = Object.getPrototypeOf(this.#template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(this.#template);
            for (let i = start; i < end; i++) {
                this.#buffer[i] = Object.create(proto, prop_descriptors);
            }
        } else {
            // For non-empty plain objects, create a prototype once and use it for all allocations
            const proto = Object.getPrototypeOf(this.#template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(this.#template);
            for (let i = start; i < end; i++) {
                this.#buffer[i] = Object.create(proto, prop_descriptors);
            }
        }
    }
    
    /**
     * Resizes the buffer to accommodate more elements
     * @param {number} new_capacity - The new capacity
     * @private
     */
    _resize(new_capacity) {
        const old_buffer = this.#buffer;
        this.#buffer = new Array(new_capacity);
        
        // Copy existing elements
        for (let i = 0; i < this.#size; i++) {
            this.#buffer[i] = old_buffer[i];
        }
        
        // Initialize new elements if template is provided
        if (this.#template !== null) {
            this._initialize_buffer(this.#capacity, new_capacity);
        }
        
        this.#capacity = new_capacity;
    }
    
    /**
     * Gets an element at the specified index
     * @param {number} index - The index to get the element from
     * @returns {*} The element at the specified index
     * @throws {Error} If the index is out of bounds
     */
    get(index) {
        if (index < 0 || index >= this.#size) {
            throw new Error("Index out of bounds");
        }
        return this.#buffer[index];
    }
    
    /**
     * Sets an element at the specified index
     * @param {number} index - The index to set the element at
     * @param {*} value - The value to set
     * @throws {Error} If the index is out of bounds
     */
    set(index, value) {
        if (index < 0 || index >= this.#size) {
            throw new Error("Index out of bounds");
        }
        this.#buffer[index] = value;
    }
    
    /**
     * Adds an element to the end of the allocator
     * @param {*} value - The value to add
     * @returns {number} The index of the added element
     */
    allocate() {
        if (this.#size >= this.#capacity) {
            this._resize(this.#capacity * 2);
        }
        return this.#buffer[this.#size++];
    }
    
    /**
     * Removes and returns the last element
     * @returns {*} The removed element
     * @throws {Error} If the allocator is empty
     */
    deallocate() {
        if (this.#size <= 0) {
            throw new Error("Cannot pop from an empty allocator");
        }
        return this.#buffer[--this.#size];
    }
    
    /**
     * Inserts an element at the specified index
     * @param {number} index - The index to insert at
     * @param {*} value - The value to insert
     * @throws {Error} If the index is out of bounds
     */
    allocate_at(index) {
        if (this.#size >= this.#capacity) {
            this._resize(this.#capacity * 2);
        }
        
        // Shift elements to make room for the new element
        for (let i = this.#size; i > index; i--) {
            this.#buffer[i] = this.#buffer[i - 1];
        }
        
        return this.#buffer[index];
    }
    
    /**
     * Removes an element at the specified index
     * @param {number} index - The index to remove from
     * @returns {*} The removed element
     * @throws {Error} If the index is out of bounds
     */
    deallocate_at(index) {
        if (index < 0 || index >= this.#size) {
            throw new Error("Index out of bounds");
        }
        
        const removed_value = this.#buffer[index];
        
        // Shift elements to fill the gap
        for (let i = index; i < this.#size - 1; i++) {
            this.#buffer[i] = this.#buffer[i + 1];
        }
        
        this.#size--;
        
        // Shrink the buffer if it's too large
        if (this.#size < this.#capacity / 4 && this.#capacity > 16) {
            this._resize(Math.max(16, Math.floor(this.#capacity / 2)));
        }
        
        return removed_value;
    }

    /**
     * Appends another random access allocator into this one.
     * @param {RandomAccessAllocator} other - The other random access allocator to append
     */
    append(other) {
        for (let i = 0; i < other.size; i++) {
            this.allocate(other.get(i));
        }
    }
    
    /**
     * Clears all elements from the allocator
     */
    reset() {
        this.#size = 0;
    }

    /**
     * Returns the data of the allocator
     * @returns {Array} The data of the allocator
     */
    get data() {
        return this.#buffer;
    }
    
    /**
     * Returns the current number of elements
     * @returns {number} The number of elements
     */
    get length() {
        return this.#size;
    }
    
    /**
     * Returns the current capacity
     * @returns {number} The capacity
     */
    get capacity() {
        return this.#capacity;
    }
    
    /**
     * Creates an iterator for the elements in the allocator
     * @returns {Iterator} An iterator for the elements
     */
    [Symbol.iterator]() {
        let index = 0;
        return {
            next: () => {
                if (index < this.#size) {
                    return { value: this.#buffer[index++], done: false };
                } else {
                    return { done: true };
                }
            }
        };
    }
}

/**
 * A sparse random access allocator that uses a bitmask to track allocated slots.
 * Prevents object reallocations or re-creations unless resizing is necessary.
 */
class SparseRandomAccessAllocator {
    #size = 0;
    #capacity = 0;
    #template = null;
    #buffer = null;
    #mask = null;

    /**
     * @param {number} initial_capacity - The initial capacity of the allocator
     * @param {object|Function|number|string|boolean} template - Template for allocated objects
     */
    constructor(initial_capacity = 16, template = null) {
        this.#capacity = initial_capacity;
        this.#size = 0;
        this.#template = template;
        this.#buffer = new Array(this.#capacity);
        this.#mask = new ResizableBitArray(this.#capacity);
        if (template !== null) {
            this._initialize_buffer(0, this.#capacity);
        }
    }

    /** Initializes buffer elements in the specified range @private */
    _initialize_buffer(start, end) {
        const function_string = 'function';
        if (typeof this.#template !== function_string && typeof this.#template !== 'object') {
            for (let i = start; i < end; i++) {
                this.#buffer[i] = { value: this.#template };
            }
        } else if (typeof this.#template === function_string) {
            for (let i = start; i < end; i++) {
                this.#buffer[i] = new this.#template();
            }
        } else if (Object.keys(this.#template).length === 0) {
            const proto = Object.getPrototypeOf(this.#template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(this.#template);
            for (let i = start; i < end; i++) {
                this.#buffer[i] = Object.create(proto, prop_descriptors);
            }
        } else {
            const proto = Object.getPrototypeOf(this.#template);
            const prop_descriptors = Object.getOwnPropertyDescriptors(this.#template);
            for (let i = start; i < end; i++) {
                this.#buffer[i] = Object.create(proto, prop_descriptors);
            }
        }
    }

    /** Resizes the buffer and mask to the new capacity @private */
    _resize(new_capacity) {
        const old_buffer = this.#buffer;
        const old_mask = this.#mask;
        const old_capacity = this.#capacity;
        this.#buffer = new Array(new_capacity);
        for (let i = 0; i < old_capacity; i++) {
            this.#buffer[i] = old_buffer[i];
        }
        if (this.#template !== null) {
            this._initialize_buffer(old_capacity, new_capacity);
        }

        this.#capacity = new_capacity;
    }

    /** Allocates an object, reusing free slots or resizing if needed. */
    allocate() {
        for (let i = 0; i < this.#capacity; i++) {
            if (!this.#mask.get(i)) {
                this.#mask.set(i, true);
                this.#size++;
                return this.#buffer[i];
            }
        }
        this._resize(this.#capacity * 2);
        this.#mask.set(this.#size, true);
        this.#size++;
        return this.#buffer[this.#size - 1];
    }

    /** Allocates an object at the specified index. */
    allocate_at(index) {
        if (index >= this.#capacity) {
            this._resize(index * 2);
        }
        if (this.#mask.get(index)) {
            throw new Error('Index already allocated');
        }
        this.#mask.set(index, true);
        this.#size++;
        return this.#buffer[index];
    }

    /** Deallocates the object at the specified index. */
    deallocate(index) {
        if (index < 0 || index >= this.#capacity || !this.#mask.get(index)) {
            throw new Error('Index out of bounds or slot not allocated');
        }
        this.#mask.set(index, false);
        this.#size--;
    }

    /** Deallocates the object at the specified index. */
    deallocate_at(index) {
        if (index < 0 || index >= this.#capacity || !this.#mask.get(index)) {
            throw new Error('Index out of bounds or slot not allocated');
        }
        this.#mask.set(index, false);
        this.#size--;
    }

    /** Retrieves the object at the index if allocated. */
    get(index) {
        if (index < 0 || index >= this.#capacity || !this.#mask.get(index)) {
            throw new Error('Index out of bounds or slot not allocated');
        }
        return this.#buffer[index];
    }

    /** Sets the object at the index if allocated. */
    set(index, value) {
        if (index < 0 || index >= this.#capacity || !this.#mask.get(index)) {
            throw new Error('Index out of bounds or slot not allocated');
        }
        this.#buffer[index] = value;
    }

    /** Number of currently allocated slots. */
    get length() {
        return this.#size;
    }

    /** Total capacity of allocator. */
    get capacity() {
        return this.#capacity;
    }

    /** Underlying buffer array. */
    get data() {
        return this.#buffer;
    }
}

export {
    FrameAllocator,
    FrameStackAllocator,
    RingBufferAllocator,
    FreeListAllocator,
    RandomAccessAllocator,
    SparseRandomAccessAllocator,
}
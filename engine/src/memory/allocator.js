class FrameAllocator {
    /**
     * @param {number} max_objects - The maximum number of objects to allocate
     * @param {object} template_object - The template object to allocate
     */
    constructor(max_objects, template_object) {
        this.max_objects = max_objects;
        this.buffer = new Array(max_objects).fill(null).map(() => ({ ...template_object }));
        this.offset = 0;
    }

    /**
     * @returns {object} - The allocated object
     */
    allocate() {
        if (this.offset >= this.max_objects) {
            throw new Error("Out of memory on frame allocator");
        }
        const index = this.offset;
        this.offset++;
        return this.buffer[index];
    }

    /**
     * @param {number} index - The index to get the object from
     * @returns {object} - The object at the index
     */
    get(index) {
        if (index < 0 || index >= this.offset) {
            throw new Error("Invalid allocator index");
        }
        return this.buffer[index];
    }

    reset() {
        this.offset = 0;
    }

    /**
     * Creates an iterator for the allocated objects in the buffer.
     * @returns {Iterator} An iterator for the allocated objects.
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
     * @param {object} template_object - The template object to use for initialization
     */
    constructor(max_objects, template_object) {
        this.max_objects = max_objects;
        this.buffer = new Array(max_objects).fill(null).map(() => ({ ...template_object }));
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
     * Resets the ring buffer to its initial state.
     */
    reset() {
        this.head = 0;
        this.tail = 0;
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


export { FrameAllocator, RingBufferAllocator };
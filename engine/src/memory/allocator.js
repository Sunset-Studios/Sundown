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

export { FrameAllocator };
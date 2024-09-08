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

    /**
     * Iterator for the array.
     * @returns {Iterator} An iterator for the array elements.
     */
    [Symbol.iterator]() {
        let index = 0;
        return {
            next: () => {
                if (index < this.#size) {
                    return { value: this.get(index++), done: false };
                } else {
                    return { done: true };
                }
            }
        };
    }
}



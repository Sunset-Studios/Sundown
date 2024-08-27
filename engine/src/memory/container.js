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

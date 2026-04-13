/**
 * EntityLinearDataContainer is a memory management system for variable-sized entity fragment data.
 *
 * This class provides efficient storage and retrieval of entity-related data using a paged memory
 * allocation strategy. It organizes data into fixed-size pages (TypedArrays) and manages free space
 * within these pages using linked lists of free blocks.
 *
 * Key features:
 * - Paged memory allocation to avoid large contiguous memory requirements
 * - Free list management for efficient space reuse
 * - Support for variable-sized data per entity
 * - Automatic page allocation when existing pages are full
 * - Memory compaction to reduce fragmentation
 *
 * The container is particularly useful for storing component data in an ECS (Entity Component System)
 * architecture, where entities may have varying amounts of data that need to be efficiently stored
 * and accessed.
 *
 * @example
 * // Create a container for Float32 data with 1024 elements per page
 * const transform_data = new EntityLinearDataContainer(Float32Array, 1024);
 *
 * // Allocate space for an entity's transform data
 * const data = new Float32Array([x, y, z, rx, ry, rz, rw, sx, sy, sz]);
 * transform_data.allocate(entity_id, data);
 *
 * // Later, retrieve or update the data
 * const entity_data = transform_data.get_data(entity_id);
 */

// Simple structure to represent a free block within a page
class FreeBlock {
  constructor(start, count) {
    this.start = start;
    this.count = count;
    this.next = null; // For linking in a free list
  }
}

export class EntityLinearDataContainer {
  static DEFAULT_PAGE_SIZE = 1024; // Or choose a suitable size

  constructor(
    container_type = Uint32Array,
    page_size = EntityLinearDataContainer.DEFAULT_PAGE_SIZE
  ) {
    this.container_type = container_type;
    this.page_size = page_size;
    this.pages = []; // Array of TypedArrays (pages)
    this.page_free_lists = []; // Array of linked lists (heads) of FreeBlocks for each page
    this.entity_indices = new Map(); // entity -> { page_index, start_in_page, count }
  }

  /**
   * Allocates space for a new data item and stores the data.
   * @param {number} entity - The entity ID.
   * @param {ArrayBufferView} data - A TypedArray containing the new data.
   * @returns {object | null} The allocation details or null on failure.
   */
  allocate(entity, data) {
    if (this.entity_indices.has(entity)) {
      warn(`Entity ${entity} already has data allocated. Use update() instead.`);
      return this.get_metadata(entity);
    }
    const count = data.length;
    const allocation_info = this._find_or_allocate_space(count);

    if (!allocation_info) return null; // Allocation failed

    const { page_index, block, prev_block } = allocation_info;
    const start_in_page = this._use_free_block(page_index, block, prev_block, count);

    // Store data
    this.pages[page_index].set(data, start_in_page);

    // Store metadata
    const metadata = { page_index, start_in_page, count };
    this.entity_indices.set(entity, metadata);

    return metadata;
  }

  /**
   * Removes an entity's data, marking the space as free.
   * @param {number} entity - The entity ID.
   */
  remove(entity) {
    const metadata = this.entity_indices.get(entity);
    if (!metadata) return;

    // Mark the space as free
    this._add_to_free_list(metadata.page_index, metadata.start_in_page, metadata.count);

    // Remove entity metadata
    this.entity_indices.delete(entity);
  }

  /**
   * Updates the data for an existing entity. This might involve reallocation if size changes significantly.
   * @param {number} entity - The entity ID.
   * @param {ArrayBufferView} new_data - A TypedArray containing the new data.
   */
  update(entity, new_data) {
    const current_metadata = this.entity_indices.get(entity);
    if (!current_metadata) {
      // If entity doesn't exist, allocate new space for it
      this.allocate(entity, new_data);
      return;
    }

    const new_count = new_data.length;

    // If the new data fits exactly in the old spot
    if (new_count === current_metadata.count) {
      this.pages[current_metadata.page_index].set(new_data, current_metadata.start_in_page);
      // Metadata (page_index, start_in_page, count) remains the same
    } else {
      // Size changed, requires re-allocation
      // 1. Free the old block
      this.remove(entity); // remove handles freeing the block and deleting old metadata
      // 2. Allocate a new block
      this.allocate(entity, new_data); // allocate handles finding space, copying data, and setting new metadata
    }
  }

  /**
   * Retrieves the metadata for a specific entity.
   * @param {number} entity - The entity ID.
   * @returns {{page_index: number, start_in_page: number, count: number}|null} Metadata or null if not found.
   */
  get_metadata(entity) {
    return this.entity_indices.get(entity) || null;
  }

  /**
   * Retrieves the data for a specific entity as a Subarray.
   * Note: Modifying the returned subarray directly modifies the underlying page data.
   * @param {number} entity - The entity ID.
   * @returns {TypedArray|null} A subarray view of the entity's data or null if not found.
   */
  get_data_for_entity(entity) {
    const metadata = this.get_metadata(entity);
    if (!metadata) return null;

    return this.pages[metadata.page_index].subarray(
      metadata.start_in_page,
      metadata.start_in_page + metadata.count
    );
  }

  /**
   * (Optional) Consolidates data by moving allocations to fill gaps and potentially freeing pages.
   * This is a potentially expensive operation and should be used sparingly.
   */
  compact() {
    if (this.entity_indices.size === 0) {
      // Nothing to compact
      this.pages = [];
      this.page_free_lists = [];
      return;
    }

    const new_pages = [];
    const new_page_free_lists = [];
    let current_new_page_index = 0;
    let current_offset_in_new_page = 0;

    // Allocate the first new page
    const allocate_first_new_page = () => {
      const new_page = new this.container_type(this.page_size);
      new_pages.push(new_page);
      // Initialize free list for the new page (will be updated later)
      new_page_free_lists.push(null); // Start with no free blocks explicitly tracked during compaction
      current_new_page_index = new_pages.length - 1;
      current_offset_in_new_page = 0;
    };

    allocate_first_new_page();

    // Create a sorted list of entities based on their original allocation order (optional but potentially better locality)
    // For simplicity here, we iterate directly through the map. Order might not be guaranteed.
    // A more robust approach might involve sorting keys if order matters.
    const new_entity_indices = new Map();

    for (const [entity, old_metadata] of this.entity_indices) {
      const { page_index: old_page_index, start_in_page: old_start, count } = old_metadata;

      // Ensure count is valid
      if (count <= 0) {
        warn(`Skipping entity ${entity} with zero or negative count during compaction.`);
        continue;
      }

      // Check if data fits in the current new page
      if (current_offset_in_new_page + count > this.page_size) {
        // Data doesn't fit, finalize the current new page's free list
        const remaining_space = this.page_size - current_offset_in_new_page;
        if (remaining_space > 0) {
          // This assumes _add_to_free_list can handle an initially null list head
          this._add_to_free_list_internal(
            new_page_free_lists,
            current_new_page_index,
            current_offset_in_new_page,
            remaining_space
          );
        }

        // Allocate a new page
        allocate_first_new_page(); // This resets indices and offset
      }

      // Copy data from the old page to the new page
      const data_to_copy = this.pages[old_page_index].subarray(old_start, old_start + count);
      new_pages[current_new_page_index].set(data_to_copy, current_offset_in_new_page);

      // Update metadata for the entity in the new map
      new_entity_indices.set(entity, {
        page_index: current_new_page_index,
        start_in_page: current_offset_in_new_page,
        count: count,
      });

      // Move the offset for the next allocation
      current_offset_in_new_page += count;
    }

    // After the loop, finalize the free list for the last used page
    const last_page_remaining_space = this.page_size - current_offset_in_new_page;
    if (last_page_remaining_space > 0) {
      this._add_to_free_list_internal(
        new_page_free_lists,
        current_new_page_index,
        current_offset_in_new_page,
        last_page_remaining_space
      );
    }

    // Replace old structures with the new, compacted ones
    this.pages = new_pages;
    this.page_free_lists = new_page_free_lists;
    this.entity_indices = new_entity_indices;

    // Optional: Clean up empty pages at the end if the last page ended up unused after allocation
    // (More complex logic needed if compaction might create empty pages *before* the last one)
    while (
      this.pages.length > 0 &&
      this.page_free_lists[this.pages.length - 1]?.start === 0 &&
      this.page_free_lists[this.pages.length - 1]?.count === this.page_size
    ) {
      const is_page_empty = ![...this.entity_indices.values()].some(
        (meta) => meta.page_index === this.pages.length - 1
      );
      if (is_page_empty) {
        this.pages.pop();
        this.page_free_lists.pop();
      } else {
        break; // Stop if the last page actually contains data
      }
    }

    // this._debug_print_free_lists(); // Uncomment for debugging
  }

  /**
   * Allocates a new page and adds it to the container.
   * @returns {number} The index of the newly allocated page.
   */
  _allocate_new_page() {
    const page_index = this.pages.length;
    const new_page = new this.container_type(this.page_size);
    this.pages.push(new_page);

    // Initialize free list for the new page with one block covering the whole page
    const initial_free_block = new FreeBlock(0, this.page_size);
    this.page_free_lists.push(initial_free_block);

    return page_index;
  }

  /**
   * Finds a suitable free block or allocates a new page if necessary.
   * @param {number} required_count - The number of elements needed.
   * @returns {{page_index: number, block: FreeBlock, prev_block: FreeBlock | null } | null} Details of the allocated block or null if allocation failed.
   */
  _find_or_allocate_space(required_count) {
    if (required_count > this.page_size) {
      error(
        `Requested size (${required_count}) exceeds page size (${this.page_size}). Increase page size or handle large allocations differently.`
      );
      return null; // Or throw error, or handle large allocations via multiple pages (more complex)
    }

    // Try finding space in existing pages
    for (let page_index = 0; page_index < this.pages.length; page_index++) {
      let current_block = this.page_free_lists[page_index];
      let prev_block = null;

      while (current_block) {
        if (current_block.count >= required_count) {
          // Found a suitable block
          return { page_index, block: current_block, prev_block };
        }
        prev_block = current_block;
        current_block = current_block.next;
      }
    }

    // No suitable block found, allocate a new page
    const new_page_index = this._allocate_new_page();
    // The new page has one large free block, which must be sufficient (checked above)
    return {
      page_index: new_page_index,
      block: this.page_free_lists[new_page_index],
      prev_block: null,
    };
  }

  /**
   * Uses a free block for allocation, potentially splitting it if it's larger than needed.
   * @param {number} page_index - The index of the page containing the block.
   * @param {FreeBlock} block - The free block to use.
   * @param {FreeBlock | null} prev_block - The block before 'block' in the free list.
   * @param {number} required_count - The number of elements to allocate.
   * @returns {number} The starting index within the page for the allocated data.
   */
  _use_free_block(page_index, block, prev_block, required_count) {
    const allocated_start = block.start;

    if (block.count > required_count) {
      // Split the block: Adjust the existing block's start and count
      block.start += required_count;
      block.count -= required_count;
    } else {
      // Use the entire block: Remove it from the free list
      if (prev_block) {
        prev_block.next = block.next;
      } else {
        // This was the head of the list
        this.page_free_lists[page_index] = block.next;
      }
    }
    return allocated_start;
  }

  /**
   * Adds a block back to the free list for a given page, merging if possible.
   * @param {number} page_index - Index of the page.
   * @param {number} start - Start index of the freed block.
   * @param {number} count - Count of elements in the freed block.
   */
  _add_to_free_list(page_index, start, count) {
    const new_free_block = new FreeBlock(start, count);
    let current = this.page_free_lists[page_index];
    let prev = null;

    // Find the correct position to insert (maintaining sorted order by start index)
    while (current && current.start < new_free_block.start) {
      prev = current;
      current = current.next;
    }

    // Attempt to merge with previous block
    if (prev && prev.start + prev.count === new_free_block.start) {
      prev.count += new_free_block.count;
      // Now, also check if the newly merged 'prev' can merge with 'current'
      if (current && prev.start + prev.count === current.start) {
        prev.count += current.count;
        prev.next = current.next; // Skip 'current'
      }
      // Merge complete, no need to insert the new_free_block
      return;
    }

    // Attempt to merge with next block
    if (current && new_free_block.start + new_free_block.count === current.start) {
      current.start = new_free_block.start;
      current.count += new_free_block.count;
      // Merge complete, no need to insert the new_free_block
      return;
    }

    // No merges possible, insert the new block
    new_free_block.next = current;
    if (prev) {
      prev.next = new_free_block;
    } else {
      // Insert at the head
      this.page_free_lists[page_index] = new_free_block;
    }
  }

  /**
   * Internal helper for adding to a specific free list array during compaction.
   * @param {Array} free_list_array - The array of free list heads (e.g., new_page_free_lists).
   * @param {number} page_index - Index of the page.
   * @param {number} start - Start index of the freed block.
   * @param {number} count - Count of elements in the freed block.
   */
  _add_to_free_list_internal(free_list_array, page_index, start, count) {
    // This largely duplicates _add_to_free_list but operates on a passed-in array
    const new_free_block = new FreeBlock(start, count);
    let current = free_list_array[page_index];
    let prev = null;

    // Handle the case where the list is initially empty for the page
    if (!current) {
      free_list_array[page_index] = new_free_block;
      return;
    }

    // Find the correct position to insert (maintaining sorted order by start index)
    while (current && current.start < new_free_block.start) {
      prev = current;
      current = current.next;
    }

    // Attempt to merge with previous block
    if (prev && prev.start + prev.count === new_free_block.start) {
      prev.count += new_free_block.count;
      // Now, also check if the newly merged 'prev' can merge with 'current'
      if (current && prev.start + prev.count === current.start) {
        prev.count += current.count;
        prev.next = current.next; // Skip 'current'
      }
      // Merge complete, no need to insert the new_free_block
      return;
    }

    // Attempt to merge with next block
    if (current && new_free_block.start + new_free_block.count === current.start) {
      current.start = new_free_block.start;
      current.count += new_free_block.count;
      // Merge complete, no need to insert the new_free_block
      return;
    }

    // No merges possible, insert the new block
    new_free_block.next = current;
    if (prev) {
      prev.next = new_free_block;
    } else {
      // Insert at the head
      free_list_array[page_index] = new_free_block;
    }
  }

  /**
   * For debugging: Prints the state of the free lists.
   */
  _debug_print_free_lists() {
    log("Free Lists:");
    for (let i = 0; i < this.page_free_lists.length; i++) {
      const head = this.page_free_lists[i];
      let str = `Page ${i}: `;
      let current = head;
      while (current) {
        str += `[${current.start}, ${current.count}] -> `;
        current = current.next;
      }
      str += "null";
      log(str);
    }
  }
}

/**
 * Stores variable-length typed payloads outside chunk data and exposes stable
 * payload handles that can be referenced from fixed-size descriptor fields.
 */
export class TypedBlobArena {
  constructor(array_type = Uint32Array) {
    this.array_type = array_type;
    this.container = new EntityLinearDataContainer(array_type);
    this.next_handle = 1;
    this.free_handles = [];
    this.gpu_metadata = new Map();
  }

  normalize(value) {
    if (value === null || value === undefined) {
      return new this.array_type(0);
    }

    if (value instanceof this.array_type) {
      return value;
    }

    if (typeof value === "number") {
      return new this.array_type([value]);
    }

    if (Array.isArray(value) || ArrayBuffer.isView(value)) {
      return new this.array_type(value);
    }

    throw new TypeError(`Unsupported blob payload type: ${typeof value}`);
  }

  allocate(value) {
    const data = this.normalize(value);
    if (data.length === 0) {
      return 0;
    }

    const handle =
      this.free_handles.length > 0 ? this.free_handles.pop() : this.next_handle++;
    this.container.allocate(handle, data);
    return handle;
  }

  update(handle, value) {
    const data = this.normalize(value);

    if (!handle) {
      return this.allocate(data);
    }

    if (data.length === 0) {
      this.remove(handle);
      return 0;
    }

    this.container.update(handle, data);
    return handle;
  }

  remove(handle) {
    if (!handle) {
      return;
    }

    this.container.remove(handle);
    this.gpu_metadata.delete(handle);
    this.free_handles.push(handle);
  }

  get_data(handle) {
    if (!handle) {
      return new this.array_type(0);
    }

    return this.container.get_data_for_entity(handle) ?? new this.array_type(0);
  }

  get_gpu_metadata(handle) {
    return this.gpu_metadata.get(handle) ?? null;
  }

  build_gpu_payload() {
    const allocations = Array.from(this.container.entity_indices.entries()).sort(
      ([left_handle], [right_handle]) => left_handle - right_handle
    );

    let total_count = 0;
    for (let i = 0; i < allocations.length; i++) {
      total_count += allocations[i][1].count;
    }

    const packed_data = new this.array_type(total_count);
    const gpu_metadata = new Map();

    let write_offset = 0;
    for (let i = 0; i < allocations.length; i++) {
      const [handle, metadata] = allocations[i];
      const data = this.container.pages[metadata.page_index].subarray(
        metadata.start_in_page,
        metadata.start_in_page + metadata.count
      );

      packed_data.set(data, write_offset);
      gpu_metadata.set(handle, { offset: write_offset, count: metadata.count });
      write_offset += metadata.count;
    }

    this.gpu_metadata = gpu_metadata;

    return {
      packed_data,
      row_count: packed_data.length,
    };
  }
}

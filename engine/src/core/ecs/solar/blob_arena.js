import { EntityLinearDataContainer } from "./memory.js";

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

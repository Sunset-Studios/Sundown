import { Buffer } from "./buffer.js";
import { TypedVector } from "../memory/container.js";
import { RandomAccessAllocator } from "../memory/allocator.js";
import { clamp } from "../utility/math.js";

const PAGE_SIZE = 64;
export const MATERIAL_PARAMS_SIZE = 32;

class FreePaletteRange {
  base = 0;
  count = 0;
}

/**
 * Manages a global material allocation table and per-instance local palettes.
 * Each unique material id is stored once in the allocation table. Instances
 * reference into the table via a material_table_offset written into their
 * StaticMeshFragment.
 */
export class MaterialAllocationTable {
  static allocation_table = new TypedVector(256, -1, BigInt64Array);
  static params_data = new Float32Array(MATERIAL_PARAMS_SIZE);
  static local_palette = new TypedVector(256, 0, Uint32Array);
  static dirty_params_pages = new TypedVector(256, 0, Uint32Array);
  static dirty_palette_pages = new TypedVector(256, 0, Uint32Array);
  static free_palette_ranges = new RandomAccessAllocator(16, FreePaletteRange);
  static params_buffer = null;
  static palette_buffer = null;
  static palette_allocations = new Map();

  static reset() {
    this.local_palette.clear();
    this.dirty_palette_pages.clear();
    this.palette_allocations.clear();
    this.free_palette_ranges.reset();
  }

  /**
   * Find or allocate a material id in the allocation table and params data.
   * Returns the index of the material id in the allocation table, which is just a simple index offset.
   */
  static find_or_allocate(material_id) {
    const mat = BigInt(material_id);
    let idx = this.allocation_table.index_of(mat);
    if (idx === -1) {
      idx = this.allocation_table.length;
      this.allocation_table.push(mat);
      this.dirty_params_pages.push(Math.floor(idx / PAGE_SIZE));
      if (this.params_data.length <= idx * MATERIAL_PARAMS_SIZE) {
        this._resize_params_buffer(idx * MATERIAL_PARAMS_SIZE * 2);
      }
      this.upload_buffers();
    }
    return idx;
  }

  /**
   * Deallocate a material id from the allocation table and params data.
   */
  static deallocate(material_id) {
    const mat = BigInt(material_id);
    const idx = this.allocation_table.index_of(mat);
    if (idx !== -1) {
      this.allocation_table.remove(idx);
    }
    // Remove performs a last element swap, so we need to dirty the removed and new last index
    this.dirty_params_pages.push(Math.floor(idx / PAGE_SIZE));
    this.dirty_params_pages.push(Math.floor(this.allocation_table.length / PAGE_SIZE));
  }

  /**
   * Register or update the material palette for an entity.
   * Returns the base offset into the local palette for this entity.
   */
  static register(entity, material_ids) {
    const required_count = material_ids.length;
    if (required_count <= 0) {
      this.unregister(entity);
      return 0;
    }

    let allocation = this.palette_allocations.get(entity);
    if (!allocation || allocation.capacity < required_count) {
      if (allocation) {
        this._free_palette_region(allocation.base, allocation.capacity);
      }

      const capacity = required_count;
      const base = this._allocate_palette_region(capacity);
      allocation = { base, count: required_count, capacity };
      this.palette_allocations.set(entity, allocation);
    }

    allocation.count = required_count;
    this._write_palette_region(allocation.base, allocation.capacity, material_ids);

    return allocation.base;
  }

  /**
   * Unregister the material palette for an entity.
   */
  static unregister(entity) {
    const allocation = this.palette_allocations.get(entity);
    if (!allocation) return;

    this.palette_allocations.delete(entity);
    this._free_palette_region(allocation.base, allocation.capacity);
  }

  /**
   * Mark a params page as dirty.
   */
  static mark_params_dirty(index) {
    this.dirty_params_pages.push(Math.floor(index / PAGE_SIZE));
  }

  /**
   * Upload the allocation and palette buffers to the GPU.
   */
  static upload_buffers() {
    if (!this.params_buffer) {
      this.params_buffer = Buffer.create({
        name: "material_params",
        raw_data: this.params_data,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        dispatch: true,
        force: true,
      });
    }

    if (!this.palette_buffer) {
      this.palette_buffer = Buffer.create({
        name: "material_palette",
        raw_data: this.local_palette.buffer,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        dispatch: true,
        force: true,
      });
    }

    if (this.dirty_params_pages.length > 0) {
      if (this.params_buffer.config.size < this.params_data.byteLength) {
        this.params_buffer = Buffer.create({
          name: "material_params",
          raw_data: this.params_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          dispatch: true,
          force: true,
        });
      } else {
        const dirty_pages = this.dirty_params_pages.buffer;
        for (let i = 0; i < dirty_pages.length; i++) {
          const page = dirty_pages[i];
          const page_offset = page * PAGE_SIZE * MATERIAL_PARAMS_SIZE;
          const page_size = Math.min(
            PAGE_SIZE * MATERIAL_PARAMS_SIZE,
            this.params_data.length - page_offset
          );
          this.params_buffer.write_raw(this.params_data, page_offset * 4, page_size, page_offset);
        }
      }

      this.dirty_params_pages.clear();
    }

    if (this.dirty_palette_pages.length > 0) {
      if (this.palette_buffer.config.size < this.local_palette.buffer.byteLength) {
        this.palette_buffer = Buffer.create({
          name: "material_palette",
          raw_data: this.local_palette.buffer,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          dispatch: true,
          force: true,
        });
      } else {
        const dirty_pages = this.dirty_palette_pages.buffer;
        for (let i = 0; i < dirty_pages.length; i++) {
          const page = dirty_pages[i];
          const page_offset = page * PAGE_SIZE;
          const page_size = clamp(this.local_palette.length - page_offset, 0, PAGE_SIZE);
          this.palette_buffer.write_raw(this.local_palette.buffer, page_offset * 4, page_size, page_offset);
        }
      }

      this.dirty_palette_pages.clear();
    }
  }

  /**
   * Resize the params data buffer if necessary.
   */
  static _resize_params_buffer(new_size) {
    if (new_size > this.params_data.length) {
      const new_buffer = new Float32Array(new_size);
      new_buffer.set(this.params_data);
      this.params_data = new_buffer;
    }
  }

  static _allocate_palette_region(required_count) {
    for (let i = 0; i < this.free_palette_ranges.length; i++) {
      const range = this.free_palette_ranges.get(i);
      if (range.count < required_count) {
        continue;
      }

      const base = range.base;
      range.base += required_count;
      range.count -= required_count;
      if (range.count === 0) {
        this.free_palette_ranges.deallocate_at(i);
      }
      return base;
    }

    const base = this.local_palette.length;
    this.local_palette.set_num_elements(base + required_count);
    return base;
  }

  static _free_palette_region(base, count) {
    if (count <= 0) return;

    for (let i = 0; i < count; i++) {
      this.local_palette.set(base + i, 0);
    }
    this._mark_palette_range_dirty(base, count);

    const range = this.free_palette_ranges.allocate();
    range.base = base;
    range.count = count;
    this._merge_free_palette_ranges();
  }

  static _merge_free_palette_ranges() {
    for (let i = 0; i < this.free_palette_ranges.length; i++) {
      const target = this.free_palette_ranges.get(i);
      let target_base = target.base;
      let target_end = target.base + target.count;

      for (let j = i + 1; j < this.free_palette_ranges.length;) {
        const candidate = this.free_palette_ranges.get(j);
        const candidate_end = candidate.base + candidate.count;

        if (candidate_end < target_base || candidate.base > target_end) {
          j++;
          continue;
        }

        target_base = Math.min(target_base, candidate.base);
        target_end = Math.max(target_end, candidate_end);
        target.base = target_base;
        target.count = target_end - target_base;
        this.free_palette_ranges.deallocate_at(j);
      }
    }
  }

  static _write_palette_region(base, capacity, material_ids) {
    for (let i = 0; i < capacity; i++) {
      const material_id = i < material_ids.length ? material_ids.get(i) : 0;
      const slot = this.find_or_allocate(material_id);
      this.local_palette.set(base + i, slot);
    }
    this._mark_palette_range_dirty(base, capacity);
  }

  static _mark_palette_range_dirty(base, count) {
    if (count <= 0) return;

    const first_page = Math.floor(base / PAGE_SIZE);
    const last_page = Math.floor((base + count - 1) / PAGE_SIZE);
    for (let page = first_page; page <= last_page; page++) {
      this.dirty_palette_pages.push(page);
    }
  }
}

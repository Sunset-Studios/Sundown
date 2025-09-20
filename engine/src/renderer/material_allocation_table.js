import { Renderer } from "./renderer.js";
import { Buffer } from "./buffer.js";
import { TypedVector } from "../memory/container.js";

const PAGE_SIZE = 64;
export const MATERIAL_PARAMS_SIZE = 32;

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
  static params_buffer = null;
  static palette_buffer = null;

  static reset() {
    this.local_palette.clear();
    this.dirty_palette_pages.clear();
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
   * Register a set of material ids for an entity instance.
   * Returns the base offset into the local palette for this instance.
   */
  static register(entity, material_ids) {
    const base = this.local_palette.length;
    for (let i = 0; i < material_ids.length; i++) {
      const mat = material_ids.get(i);
      const slot = this.find_or_allocate(mat);
      this.local_palette.push(slot);
      this.dirty_palette_pages.push(Math.floor((this.local_palette.length - 1) / PAGE_SIZE));
    }
    return base;
  }

  /**
   * Unregister a set of material ids for an entity instance.
   */
  static unregister(base, material_count) {
    for (let i = 0; i < material_count; i++) {
      const slot = base + i;
      this.local_palette.remove(slot);
      // Remove performs a last element swap, so we need to dirty the removed and new last index
      this.dirty_palette_pages.push(Math.floor(slot / PAGE_SIZE));
      this.dirty_palette_pages.push(Math.floor(this.local_palette.length / PAGE_SIZE));
    }
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
    if (this.dirty_params_pages.length > 0) {
      if (!this.params_buffer || this.params_buffer.config.size < this.params_data.byteLength) {
        this.params_buffer = Buffer.create({
          name: "material_params",
          raw_data: this.params_data,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          dispatch: true,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
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
      if (!this.palette_buffer || this.palette_buffer.config.size < this.local_palette.byteLength) {
        this.palette_buffer = Buffer.create({
          name: "material_palette",
          raw_data: this.local_palette.buffer,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          dispatch: true,
          force: true,
        });
        Renderer.get().mark_bind_groups_dirty(true);
      } else {
        const dirty_pages = this.dirty_palette_pages.buffer;
        for (let i = 0; i < dirty_pages.length; i++) {
          const page = dirty_pages[i];
          const page_offset = page * PAGE_SIZE;
          const page_size = Math.min(PAGE_SIZE, this.local_palette.length - page_offset);
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
}

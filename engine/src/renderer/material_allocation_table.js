import { Buffer } from "./buffer.js";
import { TypedVector } from "../memory/container.js";

const PAGE_SIZE = 128;

/**
 * Manages a global material allocation table and per-instance local palettes.
 * Each unique material id is stored once in the allocation table. Instances
 * reference into the table via a material_table_offset written into their
 * StaticMeshFragment.
 */
export class MaterialAllocationTable {
  static allocation_table = new TypedVector(256, -1, Int32Array);
  static local_palette = new TypedVector(256, 0, Uint32Array);
  static dirty_allocation_pages = new TypedVector(256, 0, Uint32Array);
  static dirty_palette_pages = new TypedVector(256, 0, Uint32Array);
  static allocation_buffer = null;
  static palette_buffer = null;

  static reset() {
    this.allocation_table.clear();
    this.local_palette.clear();
    this.dirty_allocation_pages.clear();
    this.dirty_palette_pages.clear();
  }

  static _allocate_material(material_id) {
    let idx = this.allocation_table.index_of(material_id);
    if (idx === -1) {
      idx = this.allocation_table.length;
      this.allocation_table.push(material_id);
      this.dirty_allocation_pages.push(idx / PAGE_SIZE);
    }
    return idx;
  }

  /**
   * Register a set of material ids for an entity instance.
   * Returns the base offset into the local palette for this instance.
   */
  static register(entity, material_ids) {
    const base = this.local_palette.length;
    for (let i = 0; i < material_ids.length; i++) {
      const mat = material_ids.get(i);
      const slot = this._allocate_material(mat);
      this.local_palette.push(slot);
      this.dirty_palette_pages.push(this.local_palette.length / PAGE_SIZE);
    }
    return base;
  }

  /**
   * Upload the allocation and palette buffers to the GPU.
   */
  static upload_buffers() {
    if (this.dirty_allocation_pages.length === 0 && this.dirty_palette_pages.length === 0) return;

    if (
      !this.allocation_buffer ||
      this.allocation_buffer.config.size < this.allocation_table.length * 4
    ) {
      this.allocation_buffer = Buffer.create({
        name: "material_allocation_table",
        raw_data: this.allocation_table.buffer,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: true,
      });
    } else {
      const dirty_pages = this.dirty_allocation_pages.buffer;
      for (let i = 0; i < dirty_pages.length; i++) {
        const page = dirty_pages[i];
        const page_offset = page * PAGE_SIZE;
        const page_size = Math.min(PAGE_SIZE, this.allocation_table.length - page_offset);
        this.allocation_buffer.write_raw(this.allocation_table.buffer, page_offset * 4, page_size);
      }
    }

    if (!this.palette_buffer || this.palette_buffer.config.size < this.local_palette.length * 2) {
      this.palette_buffer = Buffer.create({
        name: "material_local_palette",
        raw_data: this.local_palette.buffer,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: true,
      });
    } else {
      const dirty_pages = this.dirty_palette_pages.buffer;
      for (let i = 0; i < dirty_pages.length; i++) {
        const page = dirty_pages[i];
        const page_offset = page * PAGE_SIZE;
        const page_size = Math.min(PAGE_SIZE, this.local_palette.length - page_offset);
        this.palette_buffer.write_raw(this.local_palette.buffer, page_offset * 4, page_size);
      }
    }

    this.dirty_allocation_pages.clear();
    this.dirty_palette_pages.clear();
  }

  static #gpu_data = { allocation_buffer: null, palette_buffer: null };
  static to_gpu_data() {
    this.#gpu_data.allocation_buffer = this.allocation_buffer;
    this.#gpu_data.palette_buffer = this.palette_buffer;
    return this.#gpu_data;
  }
}

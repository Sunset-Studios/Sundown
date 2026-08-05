import { Buffer } from "../buffer.js";
import { npot } from "../../utility/math.js";

export const SVLM_COVERAGE_DIRECTORY_WORD_STRIDE = 8;
export const SVLM_LOCAL_PAGE_WORD_STRIDE = 5;
export const SVLM_PAGE_TABLE_MAX_PROBES = 16;
export const SVLM_PROBE_VALIDITY_WORDS_PER_LEAF = 2;

const INVALID_IDX = 0xffffffff;
const TOMBSTONE_IDX = 0xfffffffe;
const MAX_STORAGE_BINDING_SIZE = 128 * 1024 * 1024;
const MAX_WRITE_BYTES = 2 * 1024 * 1024;

function hash_words(word_0, word_1, word_2, word_3) {
  let hash = 0x811c9dc5;
  hash = Math.imul((hash ^ (word_0 >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (word_1 >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (word_2 >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (word_3 >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x846ca68b) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
}

function allocate_range(free_ranges, count, high_water_mark) {
  let best_index = -1;
  for (let index = 0; index < free_ranges.length; index++) {
    if (free_ranges[index].count < count) continue;
    if (best_index < 0 || free_ranges[index].count < free_ranges[best_index].count) {
      best_index = index;
    }
  }
  if (best_index < 0) {
    return { offset: high_water_mark, high_water_mark: high_water_mark + count };
  }
  const range = free_ranges[best_index];
  const offset = range.offset;
  range.offset += count;
  range.count -= count;
  if (range.count === 0) free_ranges.splice(best_index, 1);
  return { offset, high_water_mark };
}

function release_range(free_ranges, offset, count) {
  if (count <= 0) return;
  free_ranges.push({ offset, count });
  free_ranges.sort((a, b) => a.offset - b.offset);
  for (let index = 1; index < free_ranges.length; ) {
    const previous = free_ranges[index - 1];
    const current = free_ranges[index];
    if (previous.offset + previous.count !== current.offset) {
      index++;
      continue;
    }
    previous.count += current.count;
    free_ranges.splice(index, 1);
  }
}

function trim_high_water_mark(free_ranges, high_water_mark) {
  while (free_ranges.length > 0) {
    const range = free_ranges[free_ranges.length - 1];
    if (range.offset + range.count !== high_water_mark) break;
    high_water_mark = range.offset;
    free_ranges.pop();
  }
  return high_water_mark;
}

function create_buffer(name, word_count) {
  return Buffer.create({
    name,
    size: Math.max(1, word_count),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

function ensure_buffer(owner, field, name, required_word_count) {
  const required_bytes = required_word_count * Uint32Array.BYTES_PER_ELEMENT;
  if (required_bytes > MAX_STORAGE_BINDING_SIZE) {
    throw new Error(`SVLM buffer '${name}' exceeds the storage-buffer binding limit.`);
  }
  const buffer = owner[field];
  if (buffer?.buffer && buffer.config.size >= required_bytes) return false;
  if (buffer?.buffer) {
    buffer.resize(Math.max(1, required_word_count), true);
  } else {
    owner[field] = create_buffer(name, required_word_count);
  }
  return true;
}

function write_buffer_chunks(buffer, words, target_word_offset = 0) {
  const max_words = MAX_WRITE_BYTES / Uint32Array.BYTES_PER_ELEMENT;
  for (
    let source_word_offset = 0;
    source_word_offset < words.length;
    source_word_offset += max_words
  ) {
    const word_count = Math.min(max_words, words.length - source_word_offset);
    buffer.write_raw(
      words,
      (target_word_offset + source_word_offset) * Uint32Array.BYTES_PER_ELEMENT,
      word_count,
      source_word_offset
    );
  }
}

export function estimate_svlm_local_page_byte_length(record_count) {
  return (
    npot(Math.max(2, record_count * 2)) *
    SVLM_LOCAL_PAGE_WORD_STRIDE *
    Uint32Array.BYTES_PER_ELEMENT
  );
}

export function build_svlm_local_page(records) {
  if (records.length % SVLM_LOCAL_PAGE_WORD_STRIDE !== 0) {
    throw new Error("SVLM local page records have an invalid stride.");
  }
  const record_count = records.length / SVLM_LOCAL_PAGE_WORD_STRIDE;
  let entry_count = npot(Math.max(2, record_count * 2));
  const max_entry_count = Math.floor(
    MAX_STORAGE_BINDING_SIZE / (SVLM_LOCAL_PAGE_WORD_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
  );
  for (;;) {
    if (entry_count > max_entry_count) {
      throw new Error("SVLM local page table exceeds the storage-buffer binding limit.");
    }
    const page = new Uint32Array(entry_count * SVLM_LOCAL_PAGE_WORD_STRIDE);
    page.fill(INVALID_IDX);
    const entry_mask = entry_count - 1;
    let complete = true;
    for (let record_index = 0; record_index < record_count; record_index++) {
      const source_base = record_index * SVLM_LOCAL_PAGE_WORD_STRIDE;
      let entry_index =
        hash_words(
          records[source_base],
          records[source_base + 1],
          records[source_base + 2],
          records[source_base + 3]
        ) & entry_mask;
      let inserted = false;
      for (let probe = 0; probe < SVLM_PAGE_TABLE_MAX_PROBES; probe++) {
        const target_base = entry_index * SVLM_LOCAL_PAGE_WORD_STRIDE;
        if (page[target_base + 4] === INVALID_IDX) {
          page.set(
            records.subarray(source_base, source_base + SVLM_LOCAL_PAGE_WORD_STRIDE),
            target_base
          );
          inserted = true;
          break;
        }
        entry_index = (entry_index + 1) & entry_mask;
      }
      if (!inserted) {
        complete = false;
        break;
      }
    }
    if (complete) return page;
    entry_count *= 2;
  }
}

function write_directory_entry(directory, entry_index, page) {
  const base = entry_index * SVLM_COVERAGE_DIRECTORY_WORD_STRIDE;
  directory[base] = page.coord[0] >>> 0;
  directory[base + 1] = page.coord[1] >>> 0;
  directory[base + 2] = page.coord[2] >>> 0;
  directory[base + 3] = page.offset;
  directory[base + 4] = page.entry_count - 1;
  directory[base + 5] = page.fade_start_word;
  directory[base + 6] = page.record_count;
  directory[base + 7] = 0;
}

function build_directory(pages, minimum_entry_count = 0) {
  let active_page_count = 0;
  for (const page of pages.values()) {
    if (page.active) active_page_count++;
  }
  let entry_count = npot(Math.max(2, active_page_count * 2, minimum_entry_count));
  const max_entry_count = Math.floor(
    MAX_STORAGE_BINDING_SIZE / (SVLM_COVERAGE_DIRECTORY_WORD_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
  );
  for (;;) {
    if (entry_count > max_entry_count) {
      throw new Error("SVLM coverage directory exceeds the storage-buffer binding limit.");
    }
    const directory = new Uint32Array(entry_count * SVLM_COVERAGE_DIRECTORY_WORD_STRIDE);
    directory.fill(INVALID_IDX);
    const entry_indices = new Map();
    const entry_mask = entry_count - 1;
    let complete = true;
    for (const page of pages.values()) {
      if (!page.active) continue;
      let entry_index = hash_words(page.coord[0], page.coord[1], page.coord[2], 0) & entry_mask;
      let inserted = false;
      for (let probe = 0; probe < SVLM_PAGE_TABLE_MAX_PROBES; probe++) {
        const base = entry_index * SVLM_COVERAGE_DIRECTORY_WORD_STRIDE;
        if (directory[base + 3] === INVALID_IDX) {
          write_directory_entry(directory, entry_index, page);
          entry_indices.set(page.key, entry_index);
          inserted = true;
          break;
        }
        entry_index = (entry_index + 1) & entry_mask;
      }
      if (!inserted) {
        complete = false;
        break;
      }
    }
    if (complete) return { data: directory, entry_indices };
    entry_count *= 2;
  }
}

/** Owns the sparse two-level GPU lookup and all page-table allocation state. */
export class SVLMStreamedPageTable {
  directory_buffer = null;
  page_buffer = null;
  pages = new Map();
  free_ranges = [];
  high_water_mark = 0;
  directory_data = null;
  directory_entry_indices = new Map();
  tombstone_count = 0;
  dirty = true;
  rebuild_count = 0;
  upload_bytes = 0;
  reserve(page_entry_count) {
    ensure_buffer(
      this,
      "page_buffer",
      "svlm_streamed_local_pages",
      Math.max(2, page_entry_count) * SVLM_LOCAL_PAGE_WORD_STRIDE
    );
  }

  stage(key, coord, records, fade_start_word) {
    const page_data = build_svlm_local_page(records);
    const entry_count = page_data.length / SVLM_LOCAL_PAGE_WORD_STRIDE;
    const existing = this.pages.get(key);
    if (existing) this.deactivate(key);
    const allocation = allocate_range(this.free_ranges, entry_count, this.high_water_mark);
    this.high_water_mark = allocation.high_water_mark;
    ensure_buffer(
      this,
      "page_buffer",
      "svlm_streamed_local_pages",
      this.high_water_mark * SVLM_LOCAL_PAGE_WORD_STRIDE
    );
    this.pages.set(key, {
      key,
      coord: [...coord],
      offset: allocation.offset,
      entry_count,
      record_count: records.length / SVLM_LOCAL_PAGE_WORD_STRIDE,
      fade_start_word,
      page_data,
      upload_word_cursor: 0,
      active: false,
    });
  }

  upload(max_byte_length) {
    let remaining_words = Math.max(0, Math.floor(max_byte_length / Uint32Array.BYTES_PER_ELEMENT));
    const max_write_words = MAX_WRITE_BYTES / Uint32Array.BYTES_PER_ELEMENT;
    let uploaded_words = 0;
    for (const page of this.pages.values()) {
      if (page.active) continue;
      while (remaining_words > 0 && page.upload_word_cursor < page.page_data.length) {
        const word_count = Math.min(
          remaining_words,
          max_write_words,
          page.page_data.length - page.upload_word_cursor
        );
        write_buffer_chunks(
          this.page_buffer,
          page.page_data.subarray(page.upload_word_cursor, page.upload_word_cursor + word_count),
          page.offset * SVLM_LOCAL_PAGE_WORD_STRIDE + page.upload_word_cursor
        );
        page.upload_word_cursor += word_count;
        remaining_words -= word_count;
        uploaded_words += word_count;
      }
      if (page.upload_word_cursor >= page.page_data.length) {
        page.active = true;
        this.dirty = true;
      }
      if (remaining_words <= 0) break;
    }
    const uploaded_bytes = uploaded_words * Uint32Array.BYTES_PER_ELEMENT;
    this.upload_bytes += uploaded_bytes;
    return uploaded_bytes;
  }

  deactivate(key) {
    const page = this.pages.get(key);
    if (!page) return false;
    release_range(this.free_ranges, page.offset, page.entry_count);
    this.high_water_mark = trim_high_water_mark(this.free_ranges, this.high_water_mark);
    this.pages.delete(key);
    if (page.active) this.dirty = true;
    return true;
  }

  publish() {
    if (!this.dirty && this.directory_buffer?.buffer && this.page_buffer?.buffer) return;
    const current_entry_count = this.directory_data
      ? this.directory_data.length / SVLM_COVERAGE_DIRECTORY_WORD_STRIDE
      : 0;
    const active_page_count = this.active_page_count;
    const required_entry_count = npot(Math.max(2, active_page_count * 2));
    const rebuild =
      !this.directory_data ||
      current_entry_count < required_entry_count ||
      this.tombstone_count > current_entry_count / 4;
    if (rebuild) {
      const directory = build_directory(
        this.pages,
        Math.max(current_entry_count, required_entry_count)
      );
      this.directory_data = directory.data;
      this.directory_entry_indices = directory.entry_indices;
      this.tombstone_count = 0;
    }
    ensure_buffer(
      this,
      "directory_buffer",
      "svlm_streamed_coverage_directory",
      this.directory_data.length
    );
    ensure_buffer(
      this,
      "page_buffer",
      "svlm_streamed_local_pages",
      SVLM_LOCAL_PAGE_WORD_STRIDE * 2
    );
    if (rebuild) {
      write_buffer_chunks(this.directory_buffer, this.directory_data);
      this.upload_bytes += this.directory_data.byteLength;
      this.rebuild_count++;
      this.dirty = false;
      return;
    }

    const dirty_entries = new Set();
    for (const [key, entry_index] of this.directory_entry_indices) {
      const page = this.pages.get(key);
      if (page?.active) continue;
      const base = entry_index * SVLM_COVERAGE_DIRECTORY_WORD_STRIDE;
      this.directory_data[base + 3] = TOMBSTONE_IDX;
      this.directory_entry_indices.delete(key);
      this.tombstone_count++;
      dirty_entries.add(entry_index);
    }
    const entry_mask = current_entry_count - 1;
    for (const page of this.pages.values()) {
      if (!page.active || this.directory_entry_indices.has(page.key)) continue;
      let entry_index = hash_words(page.coord[0], page.coord[1], page.coord[2], 0) & entry_mask;
      let inserted = false;
      for (let probe = 0; probe < SVLM_PAGE_TABLE_MAX_PROBES; probe++) {
        const base = entry_index * SVLM_COVERAGE_DIRECTORY_WORD_STRIDE;
        const page_offset = this.directory_data[base + 3];
        if (page_offset === INVALID_IDX || page_offset === TOMBSTONE_IDX) {
          if (page_offset === TOMBSTONE_IDX) this.tombstone_count--;
          write_directory_entry(this.directory_data, entry_index, page);
          this.directory_entry_indices.set(page.key, entry_index);
          dirty_entries.add(entry_index);
          inserted = true;
          break;
        }
        entry_index = (entry_index + 1) & entry_mask;
      }
      if (!inserted) {
        const directory = build_directory(this.pages, current_entry_count * 2);
        this.directory_data = directory.data;
        this.directory_entry_indices = directory.entry_indices;
        this.tombstone_count = 0;
        ensure_buffer(
          this,
          "directory_buffer",
          "svlm_streamed_coverage_directory",
          this.directory_data.length
        );
        write_buffer_chunks(this.directory_buffer, this.directory_data);
        this.upload_bytes += this.directory_data.byteLength;
        this.rebuild_count++;
        this.dirty = false;
        return;
      }
    }
    for (const entry_index of dirty_entries) {
      const base = entry_index * SVLM_COVERAGE_DIRECTORY_WORD_STRIDE;
      write_buffer_chunks(
        this.directory_buffer,
        this.directory_data.subarray(base, base + SVLM_COVERAGE_DIRECTORY_WORD_STRIDE),
        base
      );
      this.upload_bytes += SVLM_COVERAGE_DIRECTORY_WORD_STRIDE * Uint32Array.BYTES_PER_ELEMENT;
    }
    this.dirty = false;
  }

  get_addressable_leaf_indices() {
    const indices = new Set();
    for (const page of this.pages.values()) {
      if (!page.active) continue;
      for (let base = 0; base < page.page_data.length; base += SVLM_LOCAL_PAGE_WORD_STRIDE) {
        const leaf_index = page.page_data[base + 4];
        if (leaf_index !== INVALID_IDX) indices.add(leaf_index);
      }
    }
    return indices;
  }

  get gpu_byte_length() {
    return (this.directory_buffer?.config?.size ?? 0) + (this.page_buffer?.config?.size ?? 0);
  }

  get active_page_count() {
    let count = 0;
    for (const page of this.pages.values()) {
      if (page.active) count++;
    }
    return count;
  }

  reset_upload_bytes() {
    this.upload_bytes = 0;
  }

  clear() {
    this.pages.clear();
    this.free_ranges.length = 0;
    this.high_water_mark = 0;
    this.directory_data = null;
    this.directory_entry_indices.clear();
    this.tombstone_count = 0;
    this.dirty = true;
    this.upload_bytes = 0;
    this.rebuild_count = 0;
  }

  destroy() {
    this.directory_buffer?.destroy();
    this.page_buffer?.destroy();
    this.directory_buffer = null;
    this.page_buffer = null;
    this.clear();
  }
}

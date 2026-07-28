export const scene_data_package_format = "sundown-scene-package";
export const scene_data_package_version = 1;
export const scene_data_package_fixed_header_byte_length = 32;

const package_magic = new TextEncoder().encode("SDNSCENE");
const fixed_header_byte_length = scene_data_package_fixed_header_byte_length;
const payload_alignment = 16;

function align_to(value, alignment = payload_alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function require_name(value, label) {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    throw new Error(`${label} cannot be empty.`);
  }
  return normalized;
}

function to_bytes(payload, label = "Scene data payload") {
  if (typeof payload === "string") {
    return new TextEncoder().encode(payload);
  }
  if (payload instanceof ArrayBuffer) {
    return new Uint8Array(payload);
  }
  if (ArrayBuffer.isView(payload)) {
    return new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
  }
  throw new Error(`${label} must be a string, ArrayBuffer, or typed array.`);
}

function copy_json_value(value) {
  if (value === undefined) {
    return null;
  }
  return JSON.parse(JSON.stringify(value));
}

function normalize_entries(entries) {
  if (entries instanceof Map) {
    return entries.entries();
  }
  if (Array.isArray(entries)) {
    return entries.map((entry) => [entry.name, entry]);
  }
  return Object.entries(entries ?? {});
}

function read_package_header(payload) {
  const bytes = to_bytes(payload, "Scene data package header");
  if (bytes.byteLength < fixed_header_byte_length) {
    throw new Error("Scene data package is smaller than its header.");
  }
  for (let index = 0; index < package_magic.length; index++) {
    if (bytes[index] !== package_magic[index]) {
      throw new Error("Scene data package has an invalid magic value.");
    }
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(8, true);
  const header_byte_length = view.getUint32(12, true);
  const directory_byte_length = view.getUint32(16, true);
  const payload_byte_length = view.getUint32(20, true);
  const section_count = view.getUint32(24, true);
  if (version !== scene_data_package_version) {
    throw new Error(`Unsupported scene data package version '${version}'.`);
  }
  if (
    header_byte_length < fixed_header_byte_length ||
    fixed_header_byte_length + directory_byte_length > header_byte_length
  ) {
    throw new Error("Scene data package header lengths are invalid.");
  }
  return {
    version,
    header_byte_length,
    directory_byte_length,
    payload_byte_length,
    section_count,
    total_byte_length: header_byte_length + payload_byte_length,
  };
}

export class SceneDataPackageSerializationPlan {
  constructor(header_bytes, payload_records, payload_byte_length) {
    this.header_bytes = header_bytes;
    this.payload_records = payload_records;
    this.byte_length = header_bytes.byteLength + payload_byte_length;
  }

  *chunks(max_chunk_byte_length = this.byte_length) {
    const chunk_byte_length = Math.max(1, Math.floor(Number(max_chunk_byte_length) || 1));
    let output_offset = 0;
    const segments = [
      { offset: 0, bytes: this.header_bytes },
      ...this.payload_records.map((record) => ({
        offset: this.header_bytes.byteLength + record.offset,
        bytes: record.bytes,
      })),
    ];

    for (const segment of segments) {
      while (output_offset < segment.offset) {
        const padding_byte_length = Math.min(chunk_byte_length, segment.offset - output_offset);
        yield {
          offset: output_offset,
          bytes: new Uint8Array(padding_byte_length),
        };
        output_offset += padding_byte_length;
      }

      for (
        let source_offset = 0;
        source_offset < segment.bytes.byteLength;
        source_offset += chunk_byte_length
      ) {
        const bytes = segment.bytes.subarray(
          source_offset,
          Math.min(segment.bytes.byteLength, source_offset + chunk_byte_length)
        );
        yield {
          offset: output_offset,
          bytes,
        };
        output_offset += bytes.byteLength;
      }
    }

    if (output_offset !== this.byte_length) {
      throw new Error(
        `Scene package serialization produced ${output_offset} bytes; ${this.byte_length} were expected.`
      );
    }
  }

  materialize() {
    const output = new Uint8Array(this.byte_length);
    for (const chunk of this.chunks()) {
      output.set(chunk.bytes, chunk.offset);
    }
    return output.buffer;
  }
}

export class SceneDataPackageBuilder {
  constructor(scene_name) {
    this.scene_name = require_name(scene_name, "Scene name");
    this.sections = new Map();
  }

  static from_package(scene_package) {
    const builder = new SceneDataPackageBuilder(scene_package.scene_name);
    for (const namespace of scene_package.list_sections()) {
      const section = scene_package.get_section(namespace);
      builder.set_section(namespace, {
        metadata: section.metadata,
        entries: section.list_entries().map((name) => {
          const entry = section.get_entry(name);
          return {
            name,
            payload: entry.bytes,
            content_type: entry.content_type,
            version: entry.version,
            metadata: entry.metadata,
          };
        }),
      });
    }
    return builder;
  }

  static async from_package_async(scene_package, options = {}) {
    const excluded = new Set(options.exclude_sections ?? []);
    const builder = new SceneDataPackageBuilder(scene_package.scene_name);
    for (const namespace of scene_package.list_sections()) {
      if (excluded.has(namespace)) {
        continue;
      }
      const section = scene_package.get_section(namespace);
      const entries = [];
      for (const name of section.list_entries()) {
        const entry = section.get_entry(name);
        entries.push({
          name,
          payload: await section.get_bytes_async(name),
          content_type: entry.content_type,
          version: entry.version,
          metadata: entry.metadata,
        });
      }
      builder.set_section(namespace, {
        metadata: section.metadata,
        entries,
      });
    }
    return builder;
  }

  set_section(namespace, section = {}) {
    const normalized_namespace = require_name(namespace, "Scene data namespace");
    const normalized_section = {
      metadata: copy_json_value(section.metadata),
      entries: new Map(),
    };
    this.sections.set(normalized_namespace, normalized_section);

    for (const [name, entry_value] of normalize_entries(section.entries)) {
      const entry =
        entry_value && typeof entry_value === "object" && "payload" in entry_value
          ? entry_value
          : { payload: entry_value };
      this.set_entry(normalized_namespace, name, entry.payload, entry);
    }
    return this;
  }

  set_entry(namespace, name, payload, options = {}) {
    const normalized_namespace = require_name(namespace, "Scene data namespace");
    const normalized_name = require_name(name, "Scene data entry name");
    let section = this.sections.get(normalized_namespace);
    if (!section) {
      section = {
        metadata: null,
        entries: new Map(),
      };
      this.sections.set(normalized_namespace, section);
    }
    section.entries.set(normalized_name, {
      name: normalized_name,
      bytes: to_bytes(payload, `Scene data entry '${normalized_namespace}/${normalized_name}'`),
      content_type: options.content_type ?? "application/octet-stream",
      version: Number(options.version ?? 1),
      metadata: copy_json_value(options.metadata),
    });
    return this;
  }

  remove_section(namespace) {
    return this.sections.delete(namespace);
  }

  create_serialization_plan() {
    const section_records = [];
    const payload_records = [];
    let payload_byte_length = 0;

    for (const namespace of Array.from(this.sections.keys()).sort()) {
      const section = this.sections.get(namespace);
      const entry_records = [];
      for (const name of Array.from(section.entries.keys()).sort()) {
        const entry = section.entries.get(name);
        payload_byte_length = align_to(payload_byte_length);
        const record = {
          name,
          offset: payload_byte_length,
          byte_length: entry.bytes.byteLength,
          content_type: entry.content_type,
          version: entry.version,
          metadata: entry.metadata,
        };
        entry_records.push(record);
        payload_records.push({
          offset: payload_byte_length,
          bytes: entry.bytes,
        });
        payload_byte_length += entry.bytes.byteLength;
      }
      section_records.push({
        namespace,
        metadata: section.metadata,
        entries: entry_records,
      });
    }

    const directory = {
      format: scene_data_package_format,
      version: scene_data_package_version,
      scene: this.scene_name,
      sections: section_records,
    };
    const directory_bytes = new TextEncoder().encode(JSON.stringify(directory));
    const header_byte_length = align_to(fixed_header_byte_length + directory_bytes.byteLength);
    const header_bytes = new Uint8Array(header_byte_length);
    const view = new DataView(header_bytes.buffer);

    header_bytes.set(package_magic, 0);
    view.setUint32(8, scene_data_package_version, true);
    view.setUint32(12, header_byte_length, true);
    view.setUint32(16, directory_bytes.byteLength, true);
    view.setUint32(20, payload_byte_length, true);
    view.setUint32(24, section_records.length, true);
    view.setUint32(28, 0, true);
    header_bytes.set(directory_bytes, fixed_header_byte_length);

    return new SceneDataPackageSerializationPlan(
      header_bytes,
      payload_records,
      payload_byte_length
    );
  }

  serialize() {
    return this.create_serialization_plan().materialize();
  }
}

export class SceneDataSection {
  constructor(scene_package, descriptor) {
    this.scene_package = scene_package;
    this.namespace = descriptor.namespace;
    this.metadata = descriptor.metadata ?? null;
    this.entries = new Map(descriptor.entries.map((entry) => [entry.name, entry]));
  }

  list_entries() {
    return Array.from(this.entries.keys());
  }

  has(name) {
    return this.entries.has(name);
  }

  get_entry(name) {
    const descriptor = this.entries.get(name);
    if (!descriptor) {
      return null;
    }
    return {
      ...descriptor,
      bytes: this.scene_package.get_entry_bytes(descriptor),
    };
  }

  get_bytes(name) {
    return this.get_entry(name)?.bytes ?? null;
  }

  async get_bytes_async(name) {
    const descriptor = this.entries.get(name);
    if (!descriptor) {
      return null;
    }
    return await this.scene_package.get_entry_bytes_async(descriptor);
  }

  get_source(name) {
    const descriptor = this.entries.get(name);
    if (!descriptor) {
      return null;
    }
    const bytes = this.scene_package.get_entry_bytes(descriptor);
    if (bytes) {
      return bytes;
    }
    return () => this.scene_package.get_entry_bytes_async(descriptor);
  }

  get_json(name) {
    const bytes = this.get_bytes(name);
    if (!bytes) {
      return null;
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  }
}

export class SceneDataPackage {
  constructor(payload_bytes, directory, header, options = {}) {
    this.payload_bytes = payload_bytes;
    this.buffer = payload_bytes?.buffer ?? null;
    this.directory = directory;
    this.header_byte_length = header.header_byte_length;
    this.total_byte_length = header.total_byte_length;
    this.asset_path = options.asset_path ?? null;
    this.read_entry = options.read_entry ?? null;
    this.scene_name = directory.scene;
    this.sections = new Map(
      directory.sections.map((section) => [section.namespace, new SceneDataSection(this, section)])
    );
  }

  static empty(scene_name) {
    return SceneDataPackage.deserialize(new SceneDataPackageBuilder(scene_name).serialize());
  }

  static read_header(payload) {
    return read_package_header(payload);
  }

  static deserialize(payload) {
    const source = to_bytes(payload, "Scene data package");
    const header = read_package_header(source);
    if (header.total_byte_length !== source.byteLength) {
      throw new Error("Scene data package header lengths are invalid.");
    }
    return SceneDataPackage._deserialize_directory(source, header, {
      payload_bytes: source,
    });
  }

  static deserialize_index(payload, options = {}) {
    const source = to_bytes(payload, "Scene data package index");
    const header = read_package_header(source);
    if (source.byteLength < header.header_byte_length) {
      throw new Error("Scene data package index does not contain its complete directory.");
    }
    if (
      options.total_byte_length !== undefined &&
      Number(options.total_byte_length) !== header.total_byte_length
    ) {
      throw new Error("Scene data package index length does not match its header.");
    }
    return SceneDataPackage._deserialize_directory(source, header, options);
  }

  static _deserialize_directory(source, header, options) {
    let directory;
    try {
      directory = JSON.parse(
        new TextDecoder().decode(
          source.subarray(
            fixed_header_byte_length,
            fixed_header_byte_length + header.directory_byte_length
          )
        )
      );
    } catch (parse_error) {
      throw new Error(
        `Scene data package directory is invalid: ${parse_error?.message ?? parse_error}`
      );
    }
    if (
      !directory ||
      typeof directory !== "object" ||
      directory.format !== scene_data_package_format ||
      directory.version !== scene_data_package_version ||
      require_name(directory.scene, "Scene name") !== directory.scene ||
      !Array.isArray(directory.sections) ||
      directory.sections.length !== header.section_count
    ) {
      throw new Error("Scene data package directory does not match its header.");
    }

    const namespaces = new Set();
    for (const section of directory.sections) {
      require_name(section.namespace, "Scene data namespace");
      if (namespaces.has(section.namespace)) {
        throw new Error(`Duplicate scene data section '${section.namespace}'.`);
      }
      namespaces.add(section.namespace);
      if (!Array.isArray(section.entries)) {
        throw new Error(`Scene data section '${section.namespace}' has no entry directory.`);
      }
      const names = new Set();
      for (const entry of section.entries) {
        require_name(entry.name, "Scene data entry name");
        if (names.has(entry.name)) {
          throw new Error(`Duplicate scene data entry '${section.namespace}/${entry.name}'.`);
        }
        names.add(entry.name);
        if (
          !Number.isInteger(entry.offset) ||
          !Number.isInteger(entry.byte_length) ||
          entry.offset < 0 ||
          entry.byte_length < 0 ||
          entry.offset + entry.byte_length > header.payload_byte_length
        ) {
          throw new Error(
            `Scene data entry '${section.namespace}/${entry.name}' is out of bounds.`
          );
        }
      }
    }

    return new SceneDataPackage(options.payload_bytes ?? null, directory, header, options);
  }

  list_sections() {
    return Array.from(this.sections.keys());
  }

  get_section(namespace) {
    return this.sections.get(namespace) ?? null;
  }

  get_entry_bytes(descriptor) {
    if (!this.payload_bytes) {
      return null;
    }
    const start = this.header_byte_length + descriptor.offset;
    const end = start + descriptor.byte_length;
    if (end > this.payload_bytes.byteLength) {
      return null;
    }
    return this.payload_bytes.subarray(start, end);
  }

  async get_entry_bytes_async(descriptor) {
    const bytes = this.get_entry_bytes(descriptor);
    if (bytes) {
      return bytes;
    }
    if (descriptor.byte_length === 0) {
      return new Uint8Array(0);
    }
    if (!this.read_entry) {
      throw new Error(`Scene data entry '${descriptor.name}' has no readable payload source.`);
    }
    return await this.read_entry(
      this.header_byte_length + descriptor.offset,
      descriptor.byte_length,
      descriptor
    );
  }
}

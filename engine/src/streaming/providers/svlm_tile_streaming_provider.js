import { StreamProvider, StreamUpdateStatus } from "../stream_provider.js";
import { StreamingSystem } from "../streaming_system.js";
import { read_file_bytes_async } from "../streaming_io.js";

export const svlm_tile_stream_provider_type = "svlm_tile";
export const svlm_tile_format = "sundown-svlm-tile";
export const svlm_tile_format_version = 3;
export const svlm_coarse_format = "sundown-svlm-coarse-hierarchy";
export const svlm_coarse_format_version = 1;

export const svlm_tile_leaf_words = 6;
export const svlm_tile_probes_per_leaf = 64;
export const svlm_tile_irradiance_words_per_probe = 6;
export const svlm_tile_directory_words = 8;
export const svlm_coarse_record_words = 10;

const svlm_tile_magic = 0x534c5449;
const svlm_tile_legacy_format_version = 1;
const svlm_tile_sparse_disk_format_version = 2;
const svlm_tile_legacy_header_words = 16;
const svlm_tile_header_words = 20;
const svlm_tile_encoding_sparse_probes = 1 << 0;
const svlm_tile_validity_words_per_leaf = 2;
const svlm_coarse_magic = 0x534c4349;
const svlm_coarse_header_words = 12;
// Compression stays entry-local so the scene package can continue serving one
// tile with a single byte-range request. The envelope is deliberately separate
// from the tile format so old dense tile payloads remain compatible.
const svlm_storage_magic = 0x534c5a50;
const svlm_storage_format_version = 1;
const svlm_storage_header_words = 5;
const svlm_storage_codec_gzip = 1;
const default_tiles_per_frame = 1;
const default_bytes_per_frame = 32 * 1024 * 1024;
const max_source_resolver_depth = 8;

function require_finite_number(value, label) {
  const normalized = Number(value);
  if (!Number.isFinite(normalized)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return normalized;
}

function require_tile_coord(coord) {
  if (
    !Array.isArray(coord) ||
    coord.length !== 3 ||
    coord.some((component) => !Number.isInteger(component))
  ) {
    throw new Error("SVLM tile coordinates must contain three integers.");
  }
  return coord;
}

function copy_array_buffer(payload) {
  if (payload instanceof ArrayBuffer) {
    return payload.slice(0);
  }
  if (ArrayBuffer.isView(payload)) {
    return payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength);
  }
  throw new Error("SVLM tile payload must be an ArrayBuffer or typed array.");
}

export async function resolve_svlm_payload_source(source, entry, key) {
  for (let resolver_depth = 0; ; resolver_depth++) {
    source = await source;
    if (typeof source !== "function") {
      return source;
    }
    if (resolver_depth >= max_source_resolver_depth) {
      throw new Error(`SVLM tile '${key}' payload source exceeded the resolver depth limit.`);
    }
    source = source(entry, key);
  }
}

function inspect_svlm_storage_payload(buffer) {
  const header_byte_length = svlm_storage_header_words * Uint32Array.BYTES_PER_ELEMENT;
  if (buffer.byteLength < header_byte_length) {
    return null;
  }
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== svlm_storage_magic) {
    return null;
  }

  const version = view.getUint32(4, true);
  const codec = view.getUint32(8, true);
  const decoded_byte_length = view.getUint32(12, true);
  const compressed_byte_length = view.getUint32(16, true);
  if (
    version !== svlm_storage_format_version ||
    codec !== svlm_storage_codec_gzip ||
    header_byte_length + compressed_byte_length !== buffer.byteLength
  ) {
    throw new Error("SVLM compressed payload has an incompatible layout.");
  }
  return {
    codec: "gzip",
    decoded_byte_length,
    compressed_byte_length,
    header_byte_length,
  };
}

export async function encode_svlm_storage_payload(payload) {
  const buffer = copy_array_buffer(payload);
  const existing = inspect_svlm_storage_payload(buffer);
  if (existing) {
    return {
      payload: buffer,
      stored_byte_length: buffer.byteLength,
      decoded_byte_length: existing.decoded_byte_length,
      codec: existing.codec,
    };
  }

  if (
    typeof CompressionStream === "undefined" ||
    typeof DecompressionStream === "undefined" ||
    typeof Blob === "undefined" ||
    typeof Response === "undefined" ||
    buffer.byteLength > 0xffffffff
  ) {
    return {
      payload: buffer,
      stored_byte_length: buffer.byteLength,
      decoded_byte_length: buffer.byteLength,
      codec: null,
    };
  }

  const stream = new Blob([buffer]).stream().pipeThrough(new CompressionStream("gzip"));
  const compressed = await new Response(stream).arrayBuffer();
  const header_byte_length = svlm_storage_header_words * Uint32Array.BYTES_PER_ELEMENT;
  if (header_byte_length + compressed.byteLength >= buffer.byteLength) {
    return {
      payload: buffer,
      stored_byte_length: buffer.byteLength,
      decoded_byte_length: buffer.byteLength,
      codec: null,
    };
  }

  const encoded = new ArrayBuffer(header_byte_length + compressed.byteLength);
  const view = new DataView(encoded);
  view.setUint32(0, svlm_storage_magic, true);
  view.setUint32(4, svlm_storage_format_version, true);
  view.setUint32(8, svlm_storage_codec_gzip, true);
  view.setUint32(12, buffer.byteLength, true);
  view.setUint32(16, compressed.byteLength, true);
  new Uint8Array(encoded, header_byte_length).set(new Uint8Array(compressed));
  return {
    payload: encoded,
    stored_byte_length: encoded.byteLength,
    decoded_byte_length: buffer.byteLength,
    codec: "gzip",
  };
}

export async function decode_svlm_storage_payload(payload) {
  const buffer = copy_array_buffer(payload);
  const storage = inspect_svlm_storage_payload(buffer);
  if (!storage) {
    return buffer;
  }
  if (
    typeof DecompressionStream === "undefined" ||
    typeof Blob === "undefined" ||
    typeof Response === "undefined"
  ) {
    throw new Error("SVLM gzip payloads require DecompressionStream support.");
  }

  const compressed = new Uint8Array(
    buffer,
    storage.header_byte_length,
    storage.compressed_byte_length
  );
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
  const decoded = await new Response(stream).arrayBuffer();
  if (decoded.byteLength !== storage.decoded_byte_length) {
    throw new Error(
      `SVLM gzip payload decoded to ${decoded.byteLength} bytes; ` +
        `${storage.decoded_byte_length} were expected.`
    );
  }
  return decoded;
}

export function is_svlm_tile_format_version_supported(version) {
  return (
    version === svlm_tile_legacy_format_version ||
    version === svlm_tile_sparse_disk_format_version ||
    version === svlm_tile_format_version
  );
}

const svlm_invalid_probe_word_0 = 0x80008000;
const svlm_invalid_probe_word_1_mask = 0x0000ffff;
const svlm_invalid_probe_word_1_value = 0x00008000;
const half_conversion_buffer = new ArrayBuffer(4);
const half_conversion_float = new Float32Array(half_conversion_buffer);
const half_conversion_uint = new Uint32Array(half_conversion_buffer);

function is_invalid_svlm_probe(words, word_offset) {
  return (
    words[word_offset] === svlm_invalid_probe_word_0 &&
    (words[word_offset + 1] & svlm_invalid_probe_word_1_mask) === svlm_invalid_probe_word_1_value
  );
}

function count_set_bits(value) {
  value >>>= 0;
  value -= (value >>> 1) & 0x55555555;
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/**
 * Converts bake-time dense probe blocks into the exact sparse GPU payload.
 * Version-three tile leaves point directly into the compact probe array, so
 * deserialization never allocates or fills the missing 64-probe records.
 */
export function compact_svlm_tile_probes(tile) {
  const leaves =
    tile.leaves instanceof Uint32Array ? tile.leaves : new Uint32Array(tile.leaves ?? 0);
  const irradiance =
    tile.irradiance instanceof Uint32Array
      ? tile.irradiance
      : new Uint32Array(tile.irradiance ?? 0);
  if (leaves.length % svlm_tile_leaf_words !== 0) {
    throw new Error("SVLM tile leaf payload has an invalid record length.");
  }

  const leaf_count = leaves.length / svlm_tile_leaf_words;
  const validity_word_count = leaf_count * svlm_tile_validity_words_per_leaf;
  if (tile.validity instanceof Uint32Array) {
    if (tile.validity.length !== validity_word_count) {
      throw new Error("SVLM tile validity payload has an invalid record length.");
    }
    let valid_probe_count = 0;
    for (const word of tile.validity) valid_probe_count += count_set_bits(word);
    if (irradiance.length !== valid_probe_count * svlm_tile_irradiance_words_per_probe) {
      throw new Error("SVLM tile compact irradiance does not match its validity masks.");
    }
    return {
      leaves: leaves.slice(),
      validity: tile.validity.slice(),
      irradiance: irradiance.slice(),
      valid_probe_count,
    };
  }

  const expected_dense_words =
    leaf_count * svlm_tile_probes_per_leaf * svlm_tile_irradiance_words_per_probe;
  if (irradiance.length !== expected_dense_words) {
    throw new Error(
      `SVLM tile irradiance contains ${irradiance.length} words; ` +
        `${expected_dense_words} dense words were expected.`
    );
  }

  const compact_leaves = leaves.slice();
  const validity = new Uint32Array(validity_word_count);
  let valid_probe_count = 0;
  for (let leaf_index = 0; leaf_index < leaf_count; leaf_index++) {
    compact_leaves[leaf_index * svlm_tile_leaf_words + 1] = valid_probe_count;
    const dense_probe_base = leaves[leaf_index * svlm_tile_leaf_words + 1];
    for (
      let local_probe_index = 0;
      local_probe_index < svlm_tile_probes_per_leaf;
      local_probe_index++
    ) {
      const source_word_offset =
        (dense_probe_base + local_probe_index) * svlm_tile_irradiance_words_per_probe;
      if (!is_invalid_svlm_probe(irradiance, source_word_offset)) {
        validity[leaf_index * svlm_tile_validity_words_per_leaf + (local_probe_index >>> 5)] |=
          1 << (local_probe_index & 31);
        valid_probe_count++;
      }
    }
  }

  const compact_irradiance = new Uint32Array(
    valid_probe_count * svlm_tile_irradiance_words_per_probe
  );
  let target_word_offset = 0;
  for (let leaf_index = 0; leaf_index < leaf_count; leaf_index++) {
    const dense_probe_base = leaves[leaf_index * svlm_tile_leaf_words + 1];
    for (
      let local_probe_index = 0;
      local_probe_index < svlm_tile_probes_per_leaf;
      local_probe_index++
    ) {
      const source_word_offset =
        (dense_probe_base + local_probe_index) * svlm_tile_irradiance_words_per_probe;
      if (is_invalid_svlm_probe(irradiance, source_word_offset)) continue;
      compact_irradiance.set(
        irradiance.subarray(
          source_word_offset,
          source_word_offset + svlm_tile_irradiance_words_per_probe
        ),
        target_word_offset
      );
      target_word_offset += svlm_tile_irradiance_words_per_probe;
    }
  }
  return { leaves: compact_leaves, validity, irradiance: compact_irradiance, valid_probe_count };
}

function float_to_half(value) {
  half_conversion_float[0] = Number(value);
  const bits = half_conversion_uint[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;

  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  if (exponent >= 31) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }

  mantissa += 0x1000;
  if ((mantissa & 0x800000) !== 0) {
    mantissa = 0;
    exponent++;
    if (exponent >= 31) return sign | 0x7c00;
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

function half_to_float(value) {
  const sign = (value & 0x8000) !== 0 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const mantissa = value & 0x03ff;
  if (exponent === 0) {
    return mantissa === 0 ? sign * 0 : sign * 2 ** -14 * (mantissa / 1024);
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

function unpack_svlm_probe(words, word_offset, output) {
  for (let word = 0; word < svlm_tile_irradiance_words_per_probe; word++) {
    const packed = words[word_offset + word];
    output[word * 2] = half_to_float(packed & 0xffff);
    output[word * 2 + 1] = half_to_float(packed >>> 16);
  }
  return output;
}

function pack_svlm_probe(coefficients) {
  const packed = new Uint32Array(svlm_tile_irradiance_words_per_probe);
  for (let word = 0; word < svlm_tile_irradiance_words_per_probe; word++) {
    const low = float_to_half(coefficients[word * 2]);
    const high = float_to_half(coefficients[word * 2 + 1]);
    packed[word] = (low | (high << 16)) >>> 0;
  }
  return packed;
}

function accumulate_svlm_probe_coefficients(
  irradiance,
  probe_offset,
  probe_count,
  coefficients,
  unpacked
) {
  let valid_probe_count = 0;
  for (let local_probe_index = 0; local_probe_index < probe_count; local_probe_index++) {
    const word_offset = (probe_offset + local_probe_index) * svlm_tile_irradiance_words_per_probe;
    if (is_invalid_svlm_probe(irradiance, word_offset)) continue;
    unpack_svlm_probe(irradiance, word_offset, unpacked);
    for (let coefficient = 0; coefficient < coefficients.length; coefficient++) {
      const value = unpacked[coefficient];
      if (Number.isFinite(value)) coefficients[coefficient] += value;
    }
    valid_probe_count++;
  }
  return valid_probe_count;
}

function average_svlm_coarse_samples(samples) {
  const coefficients = new Float64Array(12);
  const unpacked = new Float64Array(12);
  let total_weight = 0;
  for (const sample of samples) {
    const weight = Math.max(1, Math.floor(Number(sample.weight) || 1));
    unpack_svlm_probe(sample.irradiance, 0, unpacked);
    for (let coefficient = 0; coefficient < coefficients.length; coefficient++) {
      const value = unpacked[coefficient];
      if (Number.isFinite(value)) coefficients[coefficient] += value * weight;
    }
    total_weight += weight;
  }
  const inverse_weight = 1 / Math.max(1, total_weight);
  for (let coefficient = 0; coefficient < coefficients.length; coefficient++) {
    coefficients[coefficient] *= inverse_weight;
  }
  return { irradiance: pack_svlm_probe(coefficients), weight: total_weight };
}

/**
 * Produces one coarse input for each world tile touched by an owner tile's
 * leaves. This preserves coverage boundaries without duplicating fine probes.
 */
export function create_svlm_coarse_coverage_samples(tile) {
  const tile_size = Math.max(
    0.0001,
    require_finite_number(tile?.tile_size, "SVLM world tile size")
  );
  const leaves =
    tile?.leaves instanceof Uint32Array ? tile.leaves : new Uint32Array(tile?.leaves ?? 0);
  const irradiance =
    tile?.irradiance instanceof Uint32Array
      ? tile.irradiance
      : new Uint32Array(tile?.irradiance ?? 0);
  if (leaves.length % svlm_tile_leaf_words !== 0) {
    throw new Error(`SVLM tile '${tile?.key ?? "unknown"}' has invalid leaf data.`);
  }

  const leaf_count = leaves.length / svlm_tile_leaf_words;
  const expected_irradiance_words =
    leaf_count * svlm_tile_probes_per_leaf * svlm_tile_irradiance_words_per_probe;
  if (irradiance.length !== expected_irradiance_words) {
    throw new Error(`SVLM tile '${tile?.key ?? "unknown"}' has invalid irradiance data.`);
  }

  const float_words = new Float32Array(leaves.buffer, leaves.byteOffset, leaves.length);
  const coverage_samples = new Map();
  const leaf_coefficients = new Float64Array(12);
  const unpacked = new Float64Array(12);
  const epsilon = tile_size * 1e-6;

  for (let leaf_index = 0; leaf_index < leaf_count; leaf_index++) {
    const leaf_base = leaf_index * svlm_tile_leaf_words;
    const origin = [
      float_words[leaf_base + 2],
      float_words[leaf_base + 3],
      float_words[leaf_base + 4],
    ];
    const size = Math.max(0, float_words[leaf_base + 5]);
    if (origin.some((component) => !Number.isFinite(component)) || !Number.isFinite(size)) {
      throw new Error(`SVLM tile '${tile?.key ?? "unknown"}' has invalid leaf bounds.`);
    }

    leaf_coefficients.fill(0);
    const probe_base = leaves[leaf_base + 1];
    if (
      probe_base + svlm_tile_probes_per_leaf >
      irradiance.length / svlm_tile_irradiance_words_per_probe
    ) {
      throw new Error(
        `SVLM tile '${tile?.key ?? "unknown"}' leaf ${leaf_index} references irradiance outside its payload.`
      );
    }
    const valid_probe_count = accumulate_svlm_probe_coefficients(
      irradiance,
      probe_base,
      svlm_tile_probes_per_leaf,
      leaf_coefficients,
      unpacked
    );
    if (valid_probe_count === 0) continue;

    const min_coord = svlm_world_to_tile_coord(origin, tile_size);
    const max_coord = svlm_world_to_tile_coord(
      origin.map((component) => component + Math.max(0, size - epsilon)),
      tile_size
    );
    for (let z = min_coord[2]; z <= max_coord[2]; z++) {
      for (let y = min_coord[1]; y <= max_coord[1]; y++) {
        for (let x = min_coord[0]; x <= max_coord[0]; x++) {
          const coord = [x, y, z];
          const key = svlm_tile_key(coord);
          let sample = coverage_samples.get(key);
          if (!sample) {
            sample = { coord, coefficients: new Float64Array(12), valid_probe_count: 0 };
            coverage_samples.set(key, sample);
          }
          for (let coefficient = 0; coefficient < leaf_coefficients.length; coefficient++) {
            sample.coefficients[coefficient] += leaf_coefficients[coefficient];
          }
          sample.valid_probe_count += valid_probe_count;
        }
      }
    }
  }

  return Array.from(coverage_samples.values(), (sample) => {
    const inverse_probe_count = 1 / Math.max(1, sample.valid_probe_count);
    for (let coefficient = 0; coefficient < sample.coefficients.length; coefficient++) {
      sample.coefficients[coefficient] *= inverse_probe_count;
    }
    return {
      coord: sample.coord,
      lod: 0,
      weight: Math.max(1, sample.valid_probe_count),
      irradiance: pack_svlm_probe(sample.coefficients),
    };
  }).sort((a, b) => svlm_tile_key(a.coord).localeCompare(svlm_tile_key(b.coord)));
}

export function create_svlm_coarse_hierarchy(tile_samples, options = {}) {
  if (!Array.isArray(tile_samples)) {
    throw new Error("SVLM coarse hierarchy creation requires tile samples.");
  }
  const tile_size = Math.max(
    0.0001,
    require_finite_number(options.tile_size, "SVLM coarse hierarchy tile size")
  );
  const min_lod = Math.max(1, Math.floor(Number(options.min_lod) || 1));
  const max_lod = Math.max(min_lod, Math.floor(Number(options.max_lod) || min_lod));
  const max_records = Math.max(
    1,
    Math.floor(Number(options.max_records) || Number.MAX_SAFE_INTEGER)
  );
  let current = tile_samples.map((sample) => ({
    coord: [...require_tile_coord(sample.coord)],
    weight: Math.max(1, Math.floor(Number(sample.weight) || 1)),
    irradiance:
      sample.irradiance instanceof Uint32Array
        ? sample.irradiance
        : new Uint32Array(sample.irradiance ?? 0),
  }));
  const records_by_lod = [];

  for (let lod = 1; lod <= max_lod && current.length > 0; lod++) {
    const parent_groups = new Map();
    for (const sample of current) {
      const coord = sample.coord.map((component) => Math.floor(component / 2));
      const key = svlm_tile_key(coord);
      let group = parent_groups.get(key);
      if (!group) {
        group = { coord, samples: [] };
        parent_groups.set(key, group);
      }
      group.samples.push(sample);
    }
    current = Array.from(parent_groups.values(), (group) => ({
      coord: group.coord,
      ...average_svlm_coarse_samples(group.samples),
    })).sort((a, b) => svlm_tile_key(a.coord).localeCompare(svlm_tile_key(b.coord)));
    if (lod >= min_lod) {
      records_by_lod.push(
        current.map((sample) => ({
          coord: sample.coord,
          lod,
          irradiance: sample.irradiance,
        }))
      );
    }
  }

  let selected_lod_index = 0;
  let record_count = records_by_lod.reduce((sum, records) => sum + records.length, 0);
  while (record_count > max_records && selected_lod_index < records_by_lod.length - 1) {
    record_count -= records_by_lod[selected_lod_index].length;
    selected_lod_index++;
  }
  if (record_count > max_records) {
    throw new Error(
      `SVLM coarse hierarchy requires ${record_count} records at LOD ${max_lod}; ` +
        `the configured limit is ${max_records}. Increase coarse_max_lod or the coarse budget.`
    );
  }
  const records = records_by_lod.slice(selected_lod_index).flat();
  const effective_min_lod = records[0]?.lod ?? min_lod;
  return {
    format: svlm_coarse_format,
    version: svlm_coarse_format_version,
    tile_size,
    min_lod: effective_min_lod,
    max_lod: records.reduce((maximum, record) => Math.max(maximum, record.lod), min_lod),
    records,
  };
}

export function serialize_svlm_coarse_hierarchy(hierarchy) {
  if (!hierarchy || hierarchy.format !== svlm_coarse_format) {
    throw new Error("SVLM coarse hierarchy serialization requires a valid hierarchy.");
  }
  const records = hierarchy.records ?? [];
  const payload = new ArrayBuffer(
    (svlm_coarse_header_words + records.length * svlm_coarse_record_words) * 4
  );
  const view = new DataView(payload);
  view.setUint32(0, svlm_coarse_magic, true);
  view.setUint32(4, svlm_coarse_format_version, true);
  view.setUint32(8, svlm_coarse_header_words, true);
  view.setUint32(12, svlm_coarse_record_words, true);
  view.setUint32(16, records.length, true);
  view.setUint32(20, hierarchy.min_lod, true);
  view.setUint32(24, hierarchy.max_lod, true);
  view.setFloat32(28, hierarchy.tile_size, true);

  const words = new Uint32Array(payload);
  for (let record_index = 0; record_index < records.length; record_index++) {
    const record = records[record_index];
    const coord = require_tile_coord(record.coord);
    const base = svlm_coarse_header_words + record_index * svlm_coarse_record_words;
    words[base] = coord[0] >>> 0;
    words[base + 1] = coord[1] >>> 0;
    words[base + 2] = coord[2] >>> 0;
    words[base + 3] = record.lod >>> 0;
    if (
      !(record.irradiance instanceof Uint32Array) ||
      record.irradiance.length !== svlm_tile_irradiance_words_per_probe
    ) {
      throw new Error(`SVLM coarse record ${record_index} has invalid irradiance data.`);
    }
    words.set(record.irradiance, base + 4);
  }
  return payload;
}

export function deserialize_svlm_coarse_hierarchy(payload) {
  const buffer = copy_array_buffer(payload);
  if (buffer.byteLength < svlm_coarse_header_words * 4) {
    throw new Error("SVLM coarse hierarchy payload is smaller than its header.");
  }
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== svlm_coarse_magic) {
    throw new Error("SVLM coarse hierarchy payload has an invalid magic value.");
  }
  const version = view.getUint32(4, true);
  const header_words = view.getUint32(8, true);
  const record_words = view.getUint32(12, true);
  const record_count = view.getUint32(16, true);
  if (
    version !== svlm_coarse_format_version ||
    header_words !== svlm_coarse_header_words ||
    record_words !== svlm_coarse_record_words ||
    (header_words + record_count * record_words) * 4 !== buffer.byteLength
  ) {
    throw new Error("SVLM coarse hierarchy payload has an incompatible layout.");
  }
  const tile_size = view.getFloat32(28, true);
  if (!Number.isFinite(tile_size) || tile_size <= 0) {
    throw new Error("SVLM coarse hierarchy payload has an invalid tile size.");
  }
  const min_lod = view.getUint32(20, true);
  const max_lod = view.getUint32(24, true);
  if (min_lod < 1 || max_lod < min_lod || max_lod > 16) {
    throw new Error("SVLM coarse hierarchy payload has an invalid LOD range.");
  }
  const record_data = new Uint32Array(buffer).slice(header_words);
  for (let record = 0; record < record_count; record++) {
    const lod = record_data[record * svlm_coarse_record_words + 3];
    if (lod < min_lod || lod > max_lod) {
      throw new Error(`SVLM coarse hierarchy record ${record} has invalid LOD ${lod}.`);
    }
  }
  return {
    format: svlm_coarse_format,
    version,
    tile_size,
    min_lod,
    max_lod,
    record_count,
    records: record_data,
    serialized_byte_length: buffer.byteLength,
  };
}

export function svlm_tile_key(coord) {
  const [x, y, z] = require_tile_coord(coord);
  return `${x}:${y}:${z}`;
}

export function svlm_world_to_tile_coord(position, tile_size) {
  const size = Math.max(0.0001, require_finite_number(tile_size, "SVLM world tile size"));
  if (!position || position.length < 3) {
    throw new Error("SVLM world positions must contain three components.");
  }
  return [
    Math.floor(Number(position[0]) / size),
    Math.floor(Number(position[1]) / size),
    Math.floor(Number(position[2]) / size),
  ];
}

export function serialize_svlm_tile(tile) {
  const coord = require_tile_coord(tile.coord);
  const tile_size = Math.max(0.0001, require_finite_number(tile.tile_size, "SVLM world tile size"));
  const sparse = compact_svlm_tile_probes(tile);
  const leaves = sparse.leaves;
  const validity = sparse.validity;
  const irradiance = sparse.irradiance;
  const leaf_count = leaves.length / svlm_tile_leaf_words;
  const encoding_flags = svlm_tile_encoding_sparse_probes;
  const validity_word_count = validity.length;
  const stored_irradiance_word_count = irradiance.length;
  const total_words =
    svlm_tile_header_words + leaves.length + validity_word_count + stored_irradiance_word_count;
  const payload = new ArrayBuffer(total_words * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(payload);
  const bounds_min = coord.map((component) => component * tile_size);
  const bounds_max = bounds_min.map((component) => component + tile_size);

  view.setUint32(0, svlm_tile_magic, true);
  view.setUint32(4, svlm_tile_format_version, true);
  view.setUint32(8, tile.bake_version ?? 0, true);
  view.setUint32(12, svlm_tile_header_words, true);
  view.setInt32(16, coord[0], true);
  view.setInt32(20, coord[1], true);
  view.setInt32(24, coord[2], true);
  view.setUint32(28, leaf_count, true);
  view.setUint32(32, sparse.valid_probe_count, true);
  view.setFloat32(36, tile_size, true);
  view.setFloat32(40, bounds_min[0], true);
  view.setFloat32(44, bounds_min[1], true);
  view.setFloat32(48, bounds_min[2], true);
  view.setFloat32(52, bounds_max[0], true);
  view.setFloat32(56, bounds_max[1], true);
  view.setFloat32(60, bounds_max[2], true);
  view.setUint32(64, encoding_flags, true);
  view.setUint32(68, validity_word_count, true);
  view.setUint32(72, stored_irradiance_word_count, true);
  view.setUint32(76, 0, true);

  const words = new Uint32Array(payload);
  words.set(leaves, svlm_tile_header_words);
  const validity_base = svlm_tile_header_words + leaves.length;
  const stored_irradiance_base = validity_base + validity_word_count;
  words.set(validity, validity_base);
  words.set(irradiance, stored_irradiance_base);
  tile.valid_probe_count = sparse.valid_probe_count;
  tile.gpu_byte_length = leaves.byteLength + validity.byteLength + irradiance.byteLength;
  return payload;
}

export function deserialize_svlm_tile(payload) {
  const buffer = copy_array_buffer(payload);
  if (buffer.byteLength < svlm_tile_legacy_header_words * 4) {
    throw new Error("SVLM tile payload is smaller than its header.");
  }

  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== svlm_tile_magic) {
    throw new Error("SVLM tile payload has an invalid magic value.");
  }

  const version = view.getUint32(4, true);
  if (!is_svlm_tile_format_version_supported(version)) {
    throw new Error(`Unsupported SVLM tile version '${version}'.`);
  }

  const header_words = view.getUint32(12, true);
  const coord = [view.getInt32(16, true), view.getInt32(20, true), view.getInt32(24, true)];
  const leaf_count = view.getUint32(28, true);
  const leaf_word_count = leaf_count * svlm_tile_leaf_words;
  const has_sparse_header = version >= svlm_tile_sparse_disk_format_version;
  const minimum_header_words = has_sparse_header
    ? svlm_tile_header_words
    : svlm_tile_legacy_header_words;
  if (
    header_words < minimum_header_words ||
    buffer.byteLength < minimum_header_words * Uint32Array.BYTES_PER_ELEMENT
  ) {
    throw new Error("SVLM tile payload has an invalid header length.");
  }

  const header_probe_or_word_count = view.getUint32(32, true);
  const expected_dense_words =
    leaf_count * svlm_tile_probes_per_leaf * svlm_tile_irradiance_words_per_probe;
  let encoding_flags = 0;
  let validity_word_count = 0;
  let stored_irradiance_word_count = expected_dense_words;
  if (has_sparse_header) {
    encoding_flags = view.getUint32(64, true);
    validity_word_count = view.getUint32(68, true);
    stored_irradiance_word_count = view.getUint32(72, true);
    if ((encoding_flags & ~svlm_tile_encoding_sparse_probes) !== 0) {
      throw new Error("SVLM tile payload uses unsupported encoding flags.");
    }
    const uses_sparse_probes = (encoding_flags & svlm_tile_encoding_sparse_probes) !== 0;
    if (
      (uses_sparse_probes &&
        validity_word_count !== leaf_count * svlm_tile_validity_words_per_leaf) ||
      (!uses_sparse_probes &&
        (version === svlm_tile_format_version ||
          validity_word_count !== 0 ||
          stored_irradiance_word_count !== expected_dense_words)) ||
      stored_irradiance_word_count % svlm_tile_irradiance_words_per_probe !== 0
    ) {
      throw new Error("SVLM tile payload has invalid sparse-probe metadata.");
    }
  }
  if (version !== svlm_tile_format_version && header_probe_or_word_count !== expected_dense_words) {
    throw new Error(
      `SVLM tile irradiance contains ${header_probe_or_word_count} words; ` +
        `${expected_dense_words} were expected.`
    );
  }

  const total_words =
    header_words + leaf_word_count + validity_word_count + stored_irradiance_word_count;
  if (total_words * Uint32Array.BYTES_PER_ELEMENT !== buffer.byteLength) {
    throw new Error("SVLM tile payload length does not match its header.");
  }

  const tile_size = view.getFloat32(36, true);
  if (!Number.isFinite(tile_size) || tile_size <= 0) {
    throw new Error("SVLM tile payload has an invalid world tile size.");
  }

  const words = new Uint32Array(buffer);
  let leaves = words.slice(header_words, header_words + leaf_word_count);
  const validity_base = header_words + leaf_word_count;
  const stored_irradiance_base = validity_base + validity_word_count;
  let validity;
  let irradiance;
  let valid_probe_count;
  if ((encoding_flags & svlm_tile_encoding_sparse_probes) !== 0) {
    validity = words.slice(validity_base, stored_irradiance_base);
    irradiance = words.slice(stored_irradiance_base, total_words);
    valid_probe_count = 0;
    const compact_leaves = leaves.slice();
    for (let leaf_index = 0; leaf_index < leaf_count; leaf_index++) {
      const leaf_validity_base = validity_base + leaf_index * svlm_tile_validity_words_per_leaf;
      if (version !== svlm_tile_format_version) {
        compact_leaves[leaf_index * svlm_tile_leaf_words + 1] = valid_probe_count;
      } else if (compact_leaves[leaf_index * svlm_tile_leaf_words + 1] !== valid_probe_count) {
        throw new Error("SVLM tile compact leaf probe offsets are not contiguous.");
      }
      valid_probe_count +=
        count_set_bits(words[leaf_validity_base]) + count_set_bits(words[leaf_validity_base + 1]);
    }
    leaves = compact_leaves;
    if (
      valid_probe_count * svlm_tile_irradiance_words_per_probe !== irradiance.length ||
      (version === svlm_tile_format_version && header_probe_or_word_count !== valid_probe_count)
    ) {
      throw new Error("SVLM tile sparse irradiance does not match its validity masks.");
    }
  } else {
    const sparse = compact_svlm_tile_probes({
      leaves,
      irradiance: words.slice(stored_irradiance_base, total_words),
    });
    leaves = sparse.leaves;
    validity = sparse.validity;
    irradiance = sparse.irradiance;
    valid_probe_count = sparse.valid_probe_count;
  }

  return {
    format: svlm_tile_format,
    version,
    bake_version: view.getUint32(8, true),
    key: svlm_tile_key(coord),
    coord,
    tile_size,
    bounds_min: [view.getFloat32(40, true), view.getFloat32(44, true), view.getFloat32(48, true)],
    bounds_max: [view.getFloat32(52, true), view.getFloat32(56, true), view.getFloat32(60, true)],
    leaf_count,
    valid_probe_count,
    leaves,
    validity,
    irradiance,
    gpu_byte_length: leaves.byteLength + validity.byteLength + irradiance.byteLength,
    serialized_byte_length: buffer.byteLength,
  };
}

/**
 * Partitions leaf records before irradiance is allocated. Source indices are
 * retained as transient bake metadata; serialized leaves use tile-local probe
 * bases so each tile can be baked and streamed independently.
 */
export function partition_svlm_leaf_tiles({ bake_version, tile_size, leaves }) {
  const normalized_tile_size = Math.max(
    0.0001,
    require_finite_number(tile_size, "SVLM world tile size")
  );
  if (!(leaves instanceof Uint32Array)) {
    throw new Error("SVLM bake leaves must be a Uint32Array.");
  }
  if (leaves.length % svlm_tile_leaf_words !== 0) {
    throw new Error("SVLM bake leaf data has an invalid record length.");
  }

  const float_words = new Float32Array(leaves.buffer, leaves.byteOffset, leaves.length);
  // A leaf is serialized exactly once in the tile containing its center. The
  // coverage set records every world tile touched by that leaf so streaming can
  // request the owner without copying the 64-probe block into each overlap.
  const owner_tiles = new Map();
  const leaf_count = leaves.length / svlm_tile_leaf_words;
  const epsilon = normalized_tile_size * 1e-6;

  for (let leaf_index = 0; leaf_index < leaf_count; leaf_index++) {
    const base = leaf_index * svlm_tile_leaf_words;
    const origin = [float_words[base + 2], float_words[base + 3], float_words[base + 4]];
    const size = Math.max(0, float_words[base + 5]);
    const min_coord = svlm_world_to_tile_coord(origin, normalized_tile_size);
    const max_coord = svlm_world_to_tile_coord(
      origin.map((component) => component + Math.max(0, size - epsilon)),
      normalized_tile_size
    );
    const owner_coord = svlm_world_to_tile_coord(
      origin.map((component) => component + size * 0.5),
      normalized_tile_size
    );
    const owner_key = svlm_tile_key(owner_coord);
    let owner_tile = owner_tiles.get(owner_key);
    if (!owner_tile) {
      owner_tile = {
        coord: owner_coord,
        leaf_indices: [],
        coverage_coords: new Map(),
        coverage_leaf_reference_count: 0,
      };
      owner_tiles.set(owner_key, owner_tile);
    }
    owner_tile.leaf_indices.push(leaf_index);
    owner_tile.coverage_leaf_reference_count +=
      (max_coord[0] - min_coord[0] + 1) *
      (max_coord[1] - min_coord[1] + 1) *
      (max_coord[2] - min_coord[2] + 1);

    for (let z = min_coord[2]; z <= max_coord[2]; z++) {
      for (let y = min_coord[1]; y <= max_coord[1]; y++) {
        for (let x = min_coord[0]; x <= max_coord[0]; x++) {
          const coord = [x, y, z];
          const key = svlm_tile_key(coord);
          const coverage = owner_tile.coverage_coords.get(key);
          if (coverage) {
            coverage.leaf_reference_count++;
          } else {
            owner_tile.coverage_coords.set(key, { coord, leaf_reference_count: 1 });
          }
        }
      }
    }
  }

  const tiles = [];
  for (const {
    coord,
    leaf_indices,
    coverage_coords,
    coverage_leaf_reference_count,
  } of owner_tiles.values()) {
    const tile_leaves = new Uint32Array(leaf_indices.length * svlm_tile_leaf_words);

    for (let local_index = 0; local_index < leaf_indices.length; local_index++) {
      const source_leaf_index = leaf_indices[local_index];
      const source_leaf_base = source_leaf_index * svlm_tile_leaf_words;
      const target_leaf_base = local_index * svlm_tile_leaf_words;
      tile_leaves.set(
        leaves.subarray(source_leaf_base, source_leaf_base + svlm_tile_leaf_words),
        target_leaf_base
      );
      tile_leaves[target_leaf_base + 1] = local_index * svlm_tile_probes_per_leaf;
    }

    tiles.push({
      format: svlm_tile_format,
      version: svlm_tile_format_version,
      bake_version,
      key: svlm_tile_key(coord),
      coord,
      tile_size: normalized_tile_size,
      leaves: tile_leaves,
      source_leaf_indices: leaf_indices,
      coverage: Array.from(coverage_coords.values(), (coverage) => [...coverage.coord]),
      coverage_records: Array.from(coverage_coords, ([key, coverage]) => ({
        key,
        coord: [...coverage.coord],
        leaf_reference_count: coverage.leaf_reference_count,
      })),
      coverage_leaf_reference_count,
    });
  }

  return tiles.sort((a, b) => a.key.localeCompare(b.key));
}

/**
 * Compatibility helper for already-completed monolithic bakes. New SVLM bakes
 * use partition_svlm_leaf_tiles() first and produce irradiance one tile at a
 * time, avoiding a scene-sized irradiance allocation.
 */
export function partition_svlm_bake_tiles({ bake_version, tile_size, leaves, irradiance }) {
  if (!(irradiance instanceof Uint32Array)) {
    throw new Error("SVLM bake irradiance must be a Uint32Array.");
  }

  const words_per_leaf_irradiance =
    svlm_tile_probes_per_leaf * svlm_tile_irradiance_words_per_probe;
  const tiles = partition_svlm_leaf_tiles({
    bake_version,
    tile_size,
    leaves,
  });

  for (const tile of tiles) {
    tile.irradiance = new Uint32Array(tile.source_leaf_indices.length * words_per_leaf_irradiance);
    for (let local_index = 0; local_index < tile.source_leaf_indices.length; local_index++) {
      const source_leaf_index = tile.source_leaf_indices[local_index];
      const source_leaf_base = source_leaf_index * svlm_tile_leaf_words;
      const source_probe_base = leaves[source_leaf_base + 1];
      const source_irradiance_base = source_probe_base * svlm_tile_irradiance_words_per_probe;
      const source_irradiance_end = source_irradiance_base + words_per_leaf_irradiance;
      if (source_irradiance_end > irradiance.length) {
        throw new Error(
          `SVLM leaf ${source_leaf_index} references irradiance outside the bake payload.`
        );
      }
      tile.irradiance.set(
        irradiance.subarray(source_irradiance_base, source_irradiance_end),
        local_index * words_per_leaf_irradiance
      );
    }
  }

  return tiles;
}

export function create_svlm_tile_manifest(tiles, metadata = {}) {
  if (!Array.isArray(tiles)) {
    throw new Error("SVLM tile manifest creation requires an array of tiles.");
  }
  const tile_size = tiles[0]?.tile_size ?? metadata.tile_size;
  const uses_owner_tiles = tiles.length > 0 && tiles.every((tile) => Array.isArray(tile.coverage));
  const unique_leaf_count = tiles.reduce(
    (sum, tile) => sum + (tile.leaf_count ?? tile.leaves.length / svlm_tile_leaf_words),
    0
  );
  const previous_leaf_reference_count = tiles.reduce(
    (sum, tile) =>
      sum +
      (tile.coverage_leaf_reference_count ??
        tile.leaf_count ??
        tile.leaves.length / svlm_tile_leaf_words),
    0
  );
  const coverage_entries = new Map();
  if (uses_owner_tiles) {
    for (const tile of tiles) {
      const owner_key = tile.key ?? svlm_tile_key(tile.coord);
      const tile_coverage = Array.isArray(tile.coverage_records)
        ? tile.coverage_records
        : tile.coverage.map((coord) => ({ coord, leaf_reference_count: 0 }));
      for (const tile_coverage_entry of tile_coverage) {
        const coord = tile_coverage_entry.coord;
        const key = svlm_tile_key(coord);
        let coverage = coverage_entries.get(key);
        if (!coverage) {
          coverage = {
            key,
            coord: [...coord],
            owners: [],
            page_record_count: 0,
          };
          coverage_entries.set(key, coverage);
        }
        coverage.owners.push(owner_key);
        coverage.page_record_count += Math.max(
          0,
          Number(tile_coverage_entry.leaf_reference_count) || 0
        );
      }
    }
    for (const coverage of coverage_entries.values()) {
      coverage.owners.sort();
    }
  }
  return {
    format: "sundown-svlm-tile-set",
    version: svlm_tile_format_version,
    bake_version: metadata.bake_version ?? tiles[0]?.bake_version ?? 0,
    bake_serial: metadata.bake_serial ?? 0,
    tile_size,
    world_min: metadata.world_min ? [...metadata.world_min] : [0, 0, 0],
    world_max: metadata.world_max ? [...metadata.world_max] : [0, 0, 0],
    layout: {
      leaf_words_per_record: svlm_tile_leaf_words,
      probes_per_leaf: svlm_tile_probes_per_leaf,
      irradiance_words_per_probe: svlm_tile_irradiance_words_per_probe,
      validity_words_per_leaf: svlm_tile_validity_words_per_leaf,
      probe_storage: "sparse-valid-only",
      lookup_kind: "coverage-directory-local-page",
      leaf_ownership: uses_owner_tiles ? "owner-tile" : "duplicated",
    },
    ownership: {
      unique_leaf_count,
      previous_leaf_reference_count,
      duplication_factor: previous_leaf_reference_count / Math.max(1, unique_leaf_count),
    },
    coarse: metadata.coarse
      ? {
          format: svlm_coarse_format,
          version: svlm_coarse_format_version,
          entry: metadata.coarse.entry ?? "coarse",
          byte_length: metadata.coarse.byte_length ?? 0,
          min_lod: metadata.coarse.min_lod,
          max_lod: metadata.coarse.max_lod,
          record_count: metadata.coarse.record_count,
        }
      : null,
    streaming: metadata.streaming ? { ...metadata.streaming } : null,
    coverage: uses_owner_tiles
      ? Array.from(coverage_entries.values()).sort((a, b) => a.key.localeCompare(b.key))
      : null,
    tiles: tiles.map((tile) => {
      const leaf_count = tile.leaf_count ?? tile.leaves.length / svlm_tile_leaf_words;
      const decoded_byte_length =
        tile.serialized_byte_length ?? serialize_svlm_tile(tile).byteLength;
      const valid_probe_count = Math.max(
        0,
        Number(tile.valid_probe_count) ||
          Math.floor(
            Math.max(
              0,
              Number(tile.gpu_byte_length) -
                leaf_count *
                  (svlm_tile_leaf_words + svlm_tile_validity_words_per_leaf) *
                  Uint32Array.BYTES_PER_ELEMENT
            ) /
              (svlm_tile_irradiance_words_per_probe * Uint32Array.BYTES_PER_ELEMENT)
          )
      );
      return {
        key: tile.key ?? svlm_tile_key(tile.coord),
        coord: [...tile.coord],
        leaf_count,
        lookup_record_count: tile.coverage_leaf_reference_count ?? leaf_count,
        valid_probe_count,
        byte_length: decoded_byte_length,
        decoded_byte_length,
        gpu_byte_length:
          leaf_count *
            (svlm_tile_leaf_words + svlm_tile_validity_words_per_leaf) *
            Uint32Array.BYTES_PER_ELEMENT +
          valid_probe_count * svlm_tile_irradiance_words_per_probe * Uint32Array.BYTES_PER_ELEMENT,
      };
    }),
  };
}

export class SVLMTileStreamingProvider extends StreamProvider {
  static provider_type = svlm_tile_stream_provider_type;

  constructor(options = {}) {
    super(options);
    this.tiles_per_frame = options.tiles_per_frame ?? default_tiles_per_frame;
    this.bytes_per_frame = options.bytes_per_frame ?? default_bytes_per_frame;
  }

  async begin_stream(request) {
    let source =
      request.options.source ??
      request.target?.resolve_svlm_tile_source?.(request.options.entry, request.options.key);
    source = await resolve_svlm_payload_source(source, request.options.entry, request.options.key);

    if (typeof source === "string") {
      source = await read_file_bytes_async(source);
    } else if (typeof Response !== "undefined" && source instanceof Response) {
      source = await source.arrayBuffer();
    }
    if (!source) {
      throw new Error(`SVLM tile '${request.options.key}' could not be loaded.`);
    }

    const stored_byte_length = Number(source.byteLength) || 0;
    source = await decode_svlm_storage_payload(source);
    const tile = this.deserialize(source);
    tile.stored_byte_length = stored_byte_length;
    if (request.options.key && tile.key !== request.options.key) {
      throw new Error(`SVLM tile '${request.options.key}' resolved payload '${tile.key}'.`);
    }
    return { tile };
  }

  begin_frame() {
    return {
      initial_tiles: this.tiles_per_frame,
      tiles_remaining: this.tiles_per_frame,
      bytes_remaining: this.bytes_per_frame,
    };
  }

  update_stream(request, context) {
    const tile = request.state.tile;
    const budget = context.frame;
    const upload_byte_length = tile.gpu_byte_length ?? tile.serialized_byte_length;
    if (budget.tiles_remaining <= 0) {
      return StreamUpdateStatus.CONTINUE;
    }
    if (
      budget.tiles_remaining < budget.initial_tiles &&
      upload_byte_length > budget.bytes_remaining
    ) {
      return StreamUpdateStatus.CONTINUE;
    }

    request.target.install_streamed_svlm_tile(tile);
    budget.tiles_remaining--;
    budget.bytes_remaining = Math.max(0, budget.bytes_remaining - upload_byte_length);
    return {
      status: StreamUpdateStatus.COMPLETE,
      result: tile,
    };
  }

  serialize(tile) {
    return serialize_svlm_tile(tile);
  }

  deserialize(payload) {
    return deserialize_svlm_tile(payload);
  }

  static install(system = StreamingSystem.get(), options = {}) {
    if (system.has_provider(this.provider_type)) {
      return system.get_provider(this.provider_type);
    }
    return system.register_provider(new SVLMTileStreamingProvider(options));
  }
}

import { Buffer } from "../buffer.js";
import { Renderer } from "../renderer.js";
import { RenderPassFlags } from "../renderer_types.js";
import { draw_quad } from "../draw_helpers.js";
import { npot, clamp, ceil_div } from "../../utility/math.js";
import { StreamingSystem } from "../../streaming/streaming_system.js";
import {
  create_svlm_coarse_hierarchy,
  create_svlm_coarse_tile_sample,
  create_svlm_tile_manifest,
  deserialize_svlm_coarse_hierarchy,
  enumerate_svlm_tile_radius,
  partition_svlm_leaf_tiles,
  serialize_svlm_coarse_hierarchy,
  serialize_svlm_tile,
  svlm_coarse_record_words,
  svlm_tile_format_version,
  svlm_tile_key,
  svlm_tile_leaf_words,
  svlm_tile_irradiance_words_per_probe,
  svlm_tile_stream_provider_type,
  svlm_world_to_tile_coord,
  SVLMTileStreamingProvider,
} from "../../streaming/providers/svlm_tile_streaming_provider.js";
import {
  register_material_buffers,
  register_scene_lighting_data,
  register_texture_pools,
} from "../render_graph_utils.js";

const PROBES_PER_BRICK = 64;
const SH_WORDS_PER_PROBE = 6;
const PROBE_RAY_U32_STRIDE = 24;
const NODE_U32_STRIDE = 7;
const LEAF_U32_STRIDE = 6;
const TILED_LOOKUP_U32_STRIDE = 8;
const TILED_LOOKUP_MAX_PROBES = 16;
const COARSE_LOOKUP_U32_STRIDE = svlm_coarse_record_words;
const COARSE_LOOKUP_MAX_PROBES = 16;
const LINE_FLOAT_STRIDE = 20;
const LINES_PER_BOX = 12;
const PARAM_WORD_COUNT = 44;
const COUNTER_U32_COUNT = 44;
const THREADS_PER_GROUP = 128;
const PROBE_DEBUG_WORKGROUP_X = 8;
const PROBE_DEBUG_WORKGROUP_Y = 8;
const BAKE_FORMAT_VERSION = 5;
export const svlm_scene_data_namespace = "renderer.gi.svlm";

// These offsets mirror the field order of SVLMParams in svlm_common.wgsl.
// SVLMParams is stored as f32 slots; shaders cast fields back to integer types
// at their use sites when a field represents a count, index, or enum.
const PARAM_WORLD_MIN_X = 0;
const PARAM_WORLD_MIN_Y = 1;
const PARAM_WORLD_MIN_Z = 2;
const PARAM_ROOT_SIZE = 3;
const PARAM_ROOT_DIM_X = 4;
const PARAM_ROOT_DIM_Y = 5;
const PARAM_ROOT_DIM_Z = 6;
const PARAM_MAX_LEVEL = 7;
const PARAM_MAX_NODES = 8;
const PARAM_LEAF_CAPACITY = 9;
const PARAM_MIN_LEVEL = 10;
const PARAM_NEAR_FACTOR = 11;
const PARAM_OCC_MIN = 12;
const PARAM_OCC_MAX = 13;
const PARAM_REQUESTED_ROOT_SIZE = 14;
const PARAM_BAKE_PADDING = 15;
const PARAM_BAKE_SERIAL = 16;
const PARAM_SCENE_MIN_X = 18;
const PARAM_SCENE_MIN_Y = 19;
const PARAM_SCENE_MIN_Z = 20;
const PARAM_SCENE_MAX_X = 21;
const PARAM_SCENE_MAX_Y = 22;
const PARAM_SCENE_MAX_Z = 23;
const PARAM_DEBUG_LEVEL = 24;
const PARAM_DEBUG_LEAF_PAGE_GROUPS_Y = 25;
const PARAM_DEBUG_GATHER_PAGE_GROUPS_X = 26;
const PARAM_DEBUG_GATHER_PAGE_GROUPS_Y = 27;
const PARAM_IRRADIANCE_RAYS_PER_PROBE = 28;
const PARAM_IRRADIANCE_PROBES_PER_BATCH = 29;
const PARAM_IRRADIANCE_SAMPLE_COUNT = 30;
const PARAM_IRRADIANCE_MAX_RAY_DISTANCE = 31;
const PARAM_IRRADIANCE_FORMAT_VERSION = 32;
const PARAM_IRRADIANCE_SH_WORDS_PER_PROBE = 33;
const PARAM_WORLD_TILE_SIZE = 34;
const PARAM_RESIDENT_TILE_COUNT = 35;
const PARAM_TILE_STREAMING_ENABLED = 36;
const PARAM_COARSE_MIN_LOD = 37;
const PARAM_COARSE_MAX_LOD = 38;
const PARAM_STREAMING_RADIUS = 39;
const PARAM_STREAMING_TRANSITION_TILES = 40;

const INVALID_IDX = 0xffffffff;
const TOMBSTONE_IDX = 0xfffffffe;
const COUNTER_NODE_COUNT = 0;
const COUNTER_LEAF_COUNT = 3;
const COUNTER_PROBE_COUNT = 4;
const COUNTER_STATUS = 5;
const COUNTER_MAX_LEVEL_REACHED = 6;
const COUNTER_SPLIT_BASE = 8;
const COUNTER_LEVEL_BASE = 24;
const COUNTER_IRRADIANCE_SAMPLE_INDEX = 41;
const COUNTER_IRRADIANCE_COMPLETED_PROBE_SAMPLES = 42;
const COUNTER_IRRADIANCE_STATUS = 43;

const STATUS_NODE_OVERFLOW = 1 << 0;
const STATUS_LEAF_OVERFLOW = 1 << 1;
const STATUS_ROOT_OVERFLOW = 1 << 2;
const IRRADIANCE_STATUS_ALLOCATION_COMPLETE = 1 << 1;
const IRRADIANCE_STATUS_CHUNK_GENERATION_SHIFT = 8;
const IRRADIANCE_STATUS_CHUNK_GENERATION_MASK = 0x00ffffff;

const MAX_COMPUTE_WORKGROUPS = 65535;
const MAX_STORAGE_BINDING_SIZE = 128 * 1024 * 1024;
const MAX_IRRADIANCE_PROBES_PER_CHUNK = Math.floor(
  MAX_STORAGE_BINDING_SIZE / (SH_WORDS_PER_PROBE * Uint32Array.BYTES_PER_ELEMENT)
);
const MAX_IRRADIANCE_LEAVES_PER_CHUNK = Math.max(
  1,
  Math.floor(MAX_IRRADIANCE_PROBES_PER_CHUNK / PROBES_PER_BRICK)
);

// Streamed tiles intentionally omit the monolithic octree. Rebuild a compact
// spatial hash only when residency changes so runtime lookup stays bounded
// instead of scanning every leaf in a tile for every shaded pixel.
function hash_svlm_tiled_lookup_key(tile_x, tile_y, tile_z, level, leaf_x, leaf_y, leaf_z) {
  let hash = 0x811c9dc5;
  hash = Math.imul((hash ^ (tile_x >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (tile_y >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (tile_z >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (level >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (leaf_x >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (leaf_y >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (leaf_z >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x846ca68b) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
}

function build_svlm_tiled_lookup_hash(records, record_count, minimum_entry_count = 0) {
  let entry_count = npot(Math.max(2, record_count * 2, minimum_entry_count));
  const max_entry_count = Math.floor(
    MAX_STORAGE_BINDING_SIZE / (TILED_LOOKUP_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
  );

  for (;;) {
    if (entry_count > max_entry_count) {
      throw new Error("SVLM tiled lookup hash exceeds the storage-buffer binding limit.");
    }
    const table = new Uint32Array(entry_count * TILED_LOOKUP_U32_STRIDE);
    table.fill(INVALID_IDX);
    const entry_mask = entry_count - 1;
    let complete = true;

    for (let record_index = 0; record_index < record_count; record_index++) {
      const record_base = record_index * TILED_LOOKUP_U32_STRIDE;
      let entry_index =
        hash_svlm_tiled_lookup_key(
          records[record_base],
          records[record_base + 1],
          records[record_base + 2],
          records[record_base + 3],
          records[record_base + 4],
          records[record_base + 5],
          records[record_base + 6]
        ) & entry_mask;
      let inserted = false;

      for (let probe = 0; probe < TILED_LOOKUP_MAX_PROBES; probe++) {
        const entry_base = entry_index * TILED_LOOKUP_U32_STRIDE;
        if (table[entry_base + 7] === INVALID_IDX) {
          table[entry_base] = records[record_base];
          table[entry_base + 1] = records[record_base + 1];
          table[entry_base + 2] = records[record_base + 2];
          table[entry_base + 3] = records[record_base + 3];
          table[entry_base + 4] = records[record_base + 4];
          table[entry_base + 5] = records[record_base + 5];
          table[entry_base + 6] = records[record_base + 6];
          table[entry_base + 7] = records[record_base + 7];
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

    if (complete) {
      return table;
    }
    entry_count *= 2;
  }
}

function hash_svlm_coarse_lookup_key(tile_x, tile_y, tile_z, lod) {
  let hash = 0x811c9dc5;
  hash = Math.imul((hash ^ (tile_x >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (tile_y >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (tile_z >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash = Math.imul((hash ^ (lod >>> 0)) >>> 0, 0x01000193) >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x846ca68b) >>> 0;
  return (hash ^ (hash >>> 16)) >>> 0;
}

function build_svlm_coarse_lookup_hash(records, record_count, minimum_entry_count = 0) {
  let entry_count = npot(Math.max(2, record_count * 2, minimum_entry_count));
  const max_entry_count = Math.floor(
    MAX_STORAGE_BINDING_SIZE / (COARSE_LOOKUP_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
  );

  for (;;) {
    if (entry_count > max_entry_count) {
      throw new Error("SVLM coarse lookup hash exceeds the storage-buffer binding limit.");
    }
    const table = new Uint32Array(entry_count * COARSE_LOOKUP_U32_STRIDE);
    table.fill(INVALID_IDX);
    const entry_mask = entry_count - 1;
    let complete = true;

    for (let record_index = 0; record_index < record_count; record_index++) {
      const source_base = record_index * COARSE_LOOKUP_U32_STRIDE;
      let entry_index =
        hash_svlm_coarse_lookup_key(
          records[source_base],
          records[source_base + 1],
          records[source_base + 2],
          records[source_base + 3]
        ) & entry_mask;
      let inserted = false;

      for (let probe = 0; probe < COARSE_LOOKUP_MAX_PROBES; probe++) {
        const target_base = entry_index * COARSE_LOOKUP_U32_STRIDE;
        if (table[target_base + 3] === INVALID_IDX) {
          table.set(
            records.subarray(source_base, source_base + COARSE_LOOKUP_U32_STRIDE),
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

    if (complete) return table;
    entry_count *= 2;
  }
}

function distance_squared_between_tile_coords(a, b) {
  const x = a[0] - b[0];
  const y = a[1] - b[1];
  const z = a[2] - b[2];
  return x * x + y * y + z * z;
}

function svlm_tile_intersects_frustum(coord, tile_size, frustum) {
  if (!frustum || frustum.length < 24) return true;
  const min_x = coord[0] * tile_size;
  const min_y = coord[1] * tile_size;
  const min_z = coord[2] * tile_size;
  const max_x = min_x + tile_size;
  const max_y = min_y + tile_size;
  const max_z = min_z + tile_size;
  for (let plane_index = 0; plane_index < 6; plane_index++) {
    const plane_offset = plane_index * 4;
    const normal_x = frustum[plane_offset];
    const normal_y = frustum[plane_offset + 1];
    const normal_z = frustum[plane_offset + 2];
    const support_x = normal_x >= 0 ? max_x : min_x;
    const support_y = normal_y >= 0 ? max_y : min_y;
    const support_z = normal_z >= 0 ? max_z : min_z;
    if (
      normal_x * support_x +
        normal_y * support_y +
        normal_z * support_z +
        frustum[plane_offset + 3] <
      0
    ) {
      return false;
    }
  }
  return true;
}

function allocate_streamed_range(free_ranges, count, high_water_mark) {
  let best_range_index = -1;
  for (let index = 0; index < free_ranges.length; index++) {
    const range = free_ranges[index];
    if (range.count < count) continue;
    if (
      best_range_index < 0 ||
      range.count < free_ranges[best_range_index].count
    ) {
      best_range_index = index;
    }
  }
  if (best_range_index >= 0) {
    const range = free_ranges[best_range_index];
    const offset = range.offset;
    range.offset += count;
    range.count -= count;
    if (range.count === 0) free_ranges.splice(best_range_index, 1);
    return { offset, high_water_mark };
  }
  return { offset: high_water_mark, high_water_mark: high_water_mark + count };
}

function release_streamed_range(free_ranges, offset, count) {
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

// The SVLM pipeline is split into small compute passes.
// The general pipeline is: derive volume, seed roots, classify one frontier, then publish the next frontier.
const svlm_begin_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_begin.wgsl" },
  },
};

const svlm_seed_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_seed_roots.wgsl" },
  },
};

const svlm_classify_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_classify_level.wgsl" },
  },
};

const svlm_advance_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_advance_level.wgsl" },
  },
};

const svlm_finish_allocation_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_finish_allocation.wgsl" },
  },
};

const svlm_irradiance_trace_init_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_trace_init.wgsl" },
  },
};

const svlm_irradiance_trace_hit_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_trace_hit.wgsl" },
  },
};

const svlm_irradiance_trace_shade_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_trace_shade.wgsl" },
  },
};

const svlm_irradiance_accumulate_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_accumulate.wgsl" },
  },
};

const svlm_irradiance_advance_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_irradiance_advance.wgsl" },
  },
};

const compact_emissive_lights_shader_setup = {
  pipeline_shaders: {
    compute: { path: "system_compute/compact_emissive_lights.wgsl" },
  },
};

const svlm_debug_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_debug_lines.wgsl" },
  },
};

const svlm_probe_debug_clear_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_probe_debug_clear.wgsl" },
  },
};

const svlm_probe_debug_gather_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_probe_debug_gather.wgsl" },
  },
};

const svlm_probe_debug_splat_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_probe_debug_splat.wgsl" },
  },
};

const svlm_probe_debug_resolve_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/svlm_probe_debug_resolve.wgsl" },
  },
};

/**
 * GPU-driven sparse volumetric lightmapper structure builder.
 *
 * This class owns the brick hierarchy and tiled baked probe payload:
 * - derive a root brick grid from TLAS bounds
 * - classify candidate bricks against TLAS/BLAS data on the GPU
 * - emit leaf bricks with implicit 4x4x4 probe lattices
 * - progressively trace and serialize packed L1 RGB SH one world tile at a time
 * - provide debug brick/probe views and readback stats for the dev console
 */
export class SparseVolumetricLightmapper {
  config = {
    root_brick_size: 64.0,
    max_level: 5,
    min_level: 1,
    bake_padding: 2.0,
    near_geometry_factor: 0.75,
    occupancy_split_min: 0.01,
    occupancy_split_max: 0.65,
    max_nodes: 131072,
    auto_resize_growth: 2.0,
    irradiance_rays_per_probe: 1024,
    irradiance_probes_per_batch: 32768,
    irradiance_sample_count: 1,
    irradiance_max_ray_distance: 100000.0,
    max_emissive_lights: 32768,
    world_tile_size: 25.0,
    streaming_radius: 2,
    streaming_hysteresis: 1,
    streaming_prefetch_tiles: 1,
    streaming_max_requests: 32,
    streaming_memory_budget_mb: 128,
    streaming_transition_tiles: 1,
    coarse_min_lod: 1,
    coarse_max_lod: 8,
    coarse_memory_budget_mb: 16,
  };

  debug_config = {
    debug_level: -1,
  };

  stats = {
    baked: false,
    bake_serial: 0,
    root_dims: [0, 0, 0],
    root_brick_size: 0,
    min_level: 0,
    max_level: 0,
    max_level_reached: 0,
    node_count: 0,
    leaf_count: 0,
    leaf_capacity: 0,
    probe_count: 0,
    probes_per_leaf: PROBES_PER_BRICK,
    per_level_counts: [],
    split_counts: [],
    world_min: [0, 0, 0],
    world_max: [0, 0, 0],
    scene_min: [0, 0, 0],
    scene_max: [0, 0, 0],
    node_bytes: 0,
    leaf_bytes: 0,
    irradiance_bytes: 0,
    irradiance_allocated_bytes: 0,
    total_bytes: 0,
    irradiance_usable: false,
    irradiance_ready: false,
    irradiance_in_progress: false,
    irradiance_allocation_pending: false,
    irradiance_capacity_exceeded: false,
    irradiance_progress: 0,
    irradiance_sample_index: 0,
    irradiance_sample_count: 0,
    irradiance_completed_probe_samples: 0,
    irradiance_required_probe_samples: 0,
    irradiance_rays_per_probe: 0,
    serialized_tile_count: 0,
    bake_tile_count: 0,
    bake_tile_index: 0,
    resident_tile_count: 0,
    requested_tile_count: 0,
    resident_tile_bytes: 0,
    streamed_gpu_bytes: 0,
    streaming_memory_budget_bytes: 0,
    coarse_record_count: 0,
    coarse_bytes: 0,
    tile_streaming_enabled: false,
    tile_serialization_in_progress: false,
    tile_serialization_error: null,
    debug_leaf_count: 0,
    debug_level: -1,
    truncated_by_node_limit: false,
    truncated_by_leaf_limit: false,
  };

  bake_serial = 0;
  bake_requested = false;
  bake_in_flight = false;
  irradiance_bake_in_flight = false;
  irradiance_allocation_pending = false;
  irradiance_probe_capacity = 0;

  params_data = new Float32Array(PARAM_WORD_COUNT);
  params_buffer = null;

  counters_data = new Uint32Array(COUNTER_U32_COUNT);
  counter_buffer = null;

  node_buffer = null;
  curr_node_buffer = null;
  next_node_buffer = null;
  leaf_brick_buffer = null;
  irradiance_leaf_buffer = null;
  irradiance_buffer = null;
  irradiance_ray_buffer = null;
  emissive_light_buffer = null;
  debug_line_buffer = null;
  debug_texture = null;

  debug_lines_dirty = false;

  tile_manifest = null;
  tile_entries = new Map();
  tile_sources = new Map();
  tile_source_resolver = null;
  resident_tiles = new Map();
  tile_requests = new Map();
  failed_tile_keys = new Set();
  resident_tile_usage = new Map();
  tile_streaming_enabled = false;
  tile_buffers_dirty = false;
  tile_serialization_error = null;
  serialized_bake_serial = -1;
  last_tile_streaming_signature = null;
  last_streaming_center_coord = null;
  streaming_prefetch_direction = [0, 0, 0];
  streaming_update_serial = 0;
  desired_tile_keys = new Set();
  core_tile_keys = new Set();
  retained_tile_keys = new Set();

  coarse_hierarchy = null;
  coarse_hierarchy_payload = null;
  coarse_lookup_buffer = null;
  tile_bake_coarse_samples = [];

  tile_bake_tiles = [];
  tile_bake_payloads = new Map();
  tile_bake_tile_index = 0;
  tile_bake_leaf_offset = 0;
  tile_bake_current_chunk = null;
  tile_bake_chunk_serial = 0;
  tile_bake_partition_promise = null;
  tile_bake_finalize_promise = null;
  tile_bake_start_pending = false;
  tile_bake_emissive_ready = false;
  tile_bake_completed_probe_samples = 0;
  tile_bake_required_probe_samples = 0;
  tile_bake_total_irradiance_words = 0;
  tile_bake_peak_probe_capacity = 0;
  hierarchy_probe_count = 0;
  hierarchy_counters_data = null;
  tile_bake_completion_promise = null;
  tile_bake_completion_resolve = null;

  streamed_params_data = new Float32Array(PARAM_WORD_COUNT);
  streamed_params_buffer = null;
  streamed_lookup_buffer = null;
  streamed_lookup_data = null;
  streamed_lookup_tile_entries = new Map();
  streamed_lookup_tile_records = new Map();
  streamed_lookup_dirty_entries = new Set();
  streamed_lookup_record_count = 0;
  streamed_lookup_tombstone_count = 0;
  streamed_leaf_brick_buffer = null;
  streamed_irradiance_buffer = null;
  streamed_leaf_count = 0;
  streamed_probe_count = 0;
  streamed_allocations = new Map();
  streamed_pending_uploads = new Set();
  streamed_leaf_free_ranges = [];
  streamed_irradiance_free_ranges = [];
  streamed_leaf_high_water_mark = 0;
  streamed_irradiance_high_water_mark = 0;

  constructor() {
    SVLMTileStreamingProvider.install();
  }

  get_scene_data_handler() {
    return {
      namespace: svlm_scene_data_namespace,
      load: (section) => this.load_scene_data(section),
      save: () => this.create_scene_data_section(),
      after_save: () => this.release_serialized_tile_payloads(),
      unload: () => this.clear(),
    };
  }

  async load_scene_data(section) {
    const manifest = section?.metadata?.manifest;
    if (!manifest) {
      throw new Error("The SVLM scene-data section is missing its tile manifest.");
    }

    const payloads = new Map();
    for (const tile of manifest.tiles ?? []) {
      const key = tile.key ?? svlm_tile_key(tile.coord);
      const source = section.get_source(`tiles/${key}`);
      if (!source) {
        throw new Error(`The SVLM scene-data section is missing tile '${key}'.`);
      }
      payloads.set(key, source);
    }

    let coarse_payload = null;
    if (manifest.coarse?.entry) {
      const coarse_source = section.get_source(manifest.coarse.entry);
      if (!coarse_source) {
        throw new Error(
          `The SVLM scene-data section is missing coarse hierarchy '${manifest.coarse.entry}'.`
        );
      }
      coarse_payload = typeof coarse_source === "function" ? await coarse_source() : coarse_source;
    }
    this.configure_tile_streaming(manifest, { payloads, coarse_payload });
  }

  async create_scene_data_section() {
    let serialized_tiles = null;
    if (this.tile_bake_completion_promise && this.serialized_bake_serial !== this.bake_serial) {
      serialized_tiles = await this.serialize_bake_tiles();
    } else if (this.tile_manifest) {
      serialized_tiles = {
        manifest: this.tile_manifest,
        tiles: new Map(this.tile_sources),
        coarse_payload: this.coarse_hierarchy_payload,
      };
    }

    if (!serialized_tiles) {
      return undefined;
    }

    const entries = new Map(
      Array.from(serialized_tiles.tiles, ([key, payload]) => [
        `tiles/${key}`,
        {
          payload,
          content_type: "application/x-sundown-svlm-tile",
          version: svlm_tile_format_version,
          metadata: { key },
        },
      ])
    );
    if (serialized_tiles.coarse_payload && serialized_tiles.manifest.coarse?.entry) {
      entries.set(serialized_tiles.manifest.coarse.entry, {
        payload: serialized_tiles.coarse_payload,
        content_type: "application/x-sundown-svlm-coarse-hierarchy",
        version: serialized_tiles.manifest.coarse.version,
        metadata: {
          min_lod: serialized_tiles.manifest.coarse.min_lod,
          max_lod: serialized_tiles.manifest.coarse.max_lod,
        },
      });
    }

    return {
      metadata: {
        manifest: serialized_tiles.manifest,
      },
      entries,
    };
  }

  get_stats() {
    return this.stats;
  }

  _has_valid_streamed_buffers() {
    return !!(
      this.streamed_params_buffer?.buffer &&
      this.streamed_lookup_buffer?.buffer &&
      this.streamed_leaf_brick_buffer?.buffer &&
      this.streamed_irradiance_buffer?.buffer
    );
  }

  /**
   * Returns the GPU-resident bake payload and stable layout metadata without
   * performing readback or disk I/O. Durable bake buffers include COPY_SRC and
   * COPY_DST so tile serialization can stage their exact payload.
   */
  get_bake_artifact() {
    if (this.tile_streaming_enabled) {
      this._flush_streamed_tiles();
      const resident_tile_count = this.resident_tiles.size;
      const streamed_buffers_ready = this._has_valid_streamed_buffers();
      const coarse_ready = (this.coarse_hierarchy?.record_count ?? 0) > 0;
      const lookup_bytes = this.streamed_lookup_buffer?.config?.size ?? 0;
      const usable = streamed_buffers_ready && (resident_tile_count > 0 || coarse_ready);
      return {
        format: "sundown-svlm-tiled",
        version: BAKE_FORMAT_VERSION,
        usable,
        ready: usable,
        layout: {
          params_word_count: PARAM_WORD_COUNT,
          lookup_kind: "tile-leaf-hash",
          lookup_words_per_record: TILED_LOOKUP_U32_STRIDE,
          leaf_words_per_record: LEAF_U32_STRIDE,
          probes_per_leaf: PROBES_PER_BRICK,
          irradiance_encoding: "sh-l1-rgb-f16",
          irradiance_words_per_probe: SH_WORDS_PER_PROBE,
        },
        metadata: {
          bake_serial: this.tile_manifest?.bake_serial ?? this.bake_serial,
          tile_size: this.tile_manifest?.tile_size ?? this.config.world_tile_size,
          resident_tile_count,
          leaf_count: this.streamed_leaf_count,
          probe_count: this.streamed_probe_count,
          lookup_entry_count:
            lookup_bytes / (TILED_LOOKUP_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT),
          lookup_bytes,
          coarse_record_count: this.coarse_hierarchy?.record_count ?? 0,
          coarse_min_lod: this.coarse_hierarchy?.min_lod ?? 0,
          coarse_max_lod: this.coarse_hierarchy?.max_lod ?? 0,
          coarse_bytes: this.coarse_lookup_buffer?.config?.size ?? 0,
          world_min: [...(this.tile_manifest?.world_min ?? this.stats.world_min)],
          world_max: [...(this.tile_manifest?.world_max ?? this.stats.world_max)],
        },
        buffers: {
          params: this.streamed_params_buffer,
          nodes: this.streamed_lookup_buffer,
          tiled_lookup: this.streamed_lookup_buffer,
          leaf_bricks: this.streamed_leaf_brick_buffer,
          irradiance: this.streamed_irradiance_buffer,
          coarse: this.coarse_lookup_buffer,
        },
      };
    }

    return {
      format: "sundown-svlm",
      version: BAKE_FORMAT_VERSION,
      usable: this.stats.irradiance_usable,
      ready: this.stats.irradiance_ready,
      layout: {
        params_word_count: PARAM_WORD_COUNT,
        node_words_per_record: NODE_U32_STRIDE,
        leaf_words_per_record: LEAF_U32_STRIDE,
        probes_per_leaf: PROBES_PER_BRICK,
        irradiance_encoding: "sh-l1-rgb-f16",
        irradiance_words_per_probe: SH_WORDS_PER_PROBE,
      },
      metadata: {
        bake_serial: this.bake_serial,
        node_count: this.stats.node_count,
        leaf_count: this.stats.leaf_count,
        probe_count: this.stats.probe_count,
        completed_probe_samples: this.stats.irradiance_completed_probe_samples,
        irradiance_probe_capacity: this.irradiance_probe_capacity,
        irradiance_sample_count: this.config.irradiance_sample_count,
        world_min: [...this.stats.world_min],
        world_max: [...this.stats.world_max],
      },
      buffers: {
        params: this.params_buffer,
        nodes: this.node_buffer,
        leaf_bricks: this.leaf_brick_buffer,
        irradiance: this.irradiance_buffer,
        coarse: this.irradiance_buffer,
      },
    };
  }

  async serialize_bake_tiles(options = {}) {
    const requested_tile_size = Number(options.world_tile_size ?? this.config.world_tile_size);
    if (
      Number.isFinite(requested_tile_size) &&
      Math.abs(requested_tile_size - this.config.world_tile_size) >
        this.config.world_tile_size * 1e-5
    ) {
      throw new Error("SVLM world tile size is fixed when a bake starts; rebake to change it.");
    }

    let completed_bake_serial = this.bake_serial;
    if (this.serialized_bake_serial !== completed_bake_serial) {
      const completion = this.tile_bake_completion_promise;
      if (!completion) {
        throw new Error("SVLM tile serialization is unavailable until a bake has started.");
      }
      const result = await completion;
      if (result?.status !== "complete") {
        throw (
          result?.error ??
          new Error(
            result?.message ??
              `SVLM bake ${completed_bake_serial} was cancelled before serialization completed.`
          )
        );
      }
      completed_bake_serial = result.bake_serial;
    }

    if (
      this.bake_serial !== completed_bake_serial ||
      this.serialized_bake_serial !== completed_bake_serial ||
      !this.tile_manifest
    ) {
      if (this.tile_serialization_error) {
        throw this.tile_serialization_error;
      }
      throw new Error("SVLM tile baking did not produce a complete tile set.");
    }

    return {
      manifest: this.tile_manifest,
      tiles: new Map(this.tile_sources),
      coarse_payload: this.coarse_hierarchy_payload,
    };
  }

  release_serialized_tile_payloads() {
    this._clear_streamed_residency();
    this.tile_sources.clear();
    this.tile_source_resolver = null;
    this._clear_coarse_hierarchy();
  }

  configure_tile_streaming(manifest, options = {}) {
    if (
      !manifest ||
      manifest.format !== "sundown-svlm-tile-set" ||
      manifest.version !== svlm_tile_format_version ||
      !Array.isArray(manifest.tiles)
    ) {
      throw new Error("SVLM tile streaming requires a valid tile-set manifest.");
    }

    const tile_size = Number(manifest.tile_size);
    if (!Number.isFinite(tile_size) || tile_size <= 0) {
      throw new Error("SVLM tile manifests require a positive tile size.");
    }

    this._clear_streamed_residency();
    this.tile_manifest = manifest;
    this.tile_entries = new Map(
      manifest.tiles.map((entry) => [
        entry.key ?? svlm_tile_key(entry.coord),
        {
          ...entry,
          key: entry.key ?? svlm_tile_key(entry.coord),
          coord: [...entry.coord],
        },
      ])
    );
    this.tile_sources =
      options.payloads instanceof Map
        ? new Map(options.payloads)
        : new Map(Object.entries(options.payloads ?? {}));
    this.tile_source_resolver = options.resolve_tile ?? null;
    this.tile_streaming_enabled = true;
    this.config.world_tile_size = tile_size;
    const streaming_config = { ...(manifest.streaming ?? {}), ...options };
    this.config.streaming_radius = clamp(
      Math.floor(Number(streaming_config.streaming_radius ?? this.config.streaming_radius) || 0),
      0,
      16
    );
    this.config.streaming_hysteresis = clamp(
      Math.floor(
        Number(streaming_config.streaming_hysteresis ?? this.config.streaming_hysteresis) || 0
      ),
      0,
      8
    );
    this.config.streaming_prefetch_tiles = clamp(
      Math.floor(
        Number(streaming_config.streaming_prefetch_tiles ?? this.config.streaming_prefetch_tiles) ||
          0
      ),
      0,
      8
    );
    this.config.streaming_max_requests = clamp(
      Math.floor(
        Number(streaming_config.streaming_max_requests ?? this.config.streaming_max_requests) || 1
      ),
      1,
      256
    );
    this.config.streaming_memory_budget_mb = clamp(
      Number(
        streaming_config.streaming_memory_budget_mb ?? this.config.streaming_memory_budget_mb
      ) || 1,
      1,
      1024
    );
    this.config.streaming_transition_tiles = clamp(
      Number(
        streaming_config.streaming_transition_tiles ?? this.config.streaming_transition_tiles
      ) || 0,
      0,
      4
    );
    this.config.coarse_memory_budget_mb = clamp(
      Number(streaming_config.coarse_memory_budget_mb ?? this.config.coarse_memory_budget_mb) || 1,
      1,
      128
    );
    this.config.coarse_min_lod = clamp(
      Math.floor(Number(manifest.coarse?.min_lod ?? this.config.coarse_min_lod) || 1),
      1,
      16
    );
    this.config.coarse_max_lod = clamp(
      Math.floor(Number(manifest.coarse?.max_lod ?? this.config.coarse_max_lod) || 1),
      this.config.coarse_min_lod,
      16
    );
    this._clear_coarse_hierarchy();
    if (options.coarse_payload) {
      this._install_coarse_hierarchy(options.coarse_payload);
    }
    this.serialized_bake_serial = manifest.bake_serial ?? -1;
    this.stats.serialized_tile_count = this.tile_entries.size;
    this.stats.resident_tile_count = 0;
    this.stats.requested_tile_count = 0;
    this.stats.streaming_memory_budget_bytes = this._get_streaming_memory_budget_bytes();
    this.stats.tile_streaming_enabled = true;
    this.stats.tile_serialization_error = null;
    this.stats.config = { ...this.config };
    return manifest;
  }

  _install_coarse_hierarchy(payload) {
    const hierarchy = deserialize_svlm_coarse_hierarchy(payload);
    if (
      Math.abs(hierarchy.tile_size - this.config.world_tile_size) >
      this.config.world_tile_size * 1e-5
    ) {
      throw new Error(
        `SVLM coarse hierarchy uses tile size ${hierarchy.tile_size}; ` +
          `${this.config.world_tile_size} was expected.`
      );
    }
    const existing_entry_count = Math.floor(
      (this.coarse_lookup_buffer?.config?.size ?? 0) /
        (COARSE_LOOKUP_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
    );
    const lookup = build_svlm_coarse_lookup_hash(
      hierarchy.records,
      hierarchy.record_count,
      existing_entry_count
    );
    this.coarse_hierarchy = hierarchy;
    this.coarse_hierarchy_payload =
      payload instanceof ArrayBuffer
        ? payload.slice(0)
        : payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this._ensure_streamed_buffer(
      "coarse_lookup_buffer",
      "svlm_streamed_coarse_lookup",
      lookup.length,
      usage
    );
    this.coarse_lookup_buffer.write_raw(lookup);
    this.stats.coarse_record_count = hierarchy.record_count;
    this.stats.coarse_bytes = lookup.byteLength;
  }

  _clear_coarse_hierarchy() {
    this.coarse_lookup_buffer?.destroy();
    this.coarse_lookup_buffer = null;
    this.coarse_hierarchy = null;
    this.coarse_hierarchy_payload = null;
    this.stats.coarse_record_count = 0;
    this.stats.coarse_bytes = 0;
  }

  _get_streaming_memory_budget_bytes() {
    return Math.floor(this.config.streaming_memory_budget_mb * 1024 * 1024);
  }

  resolve_svlm_tile_source(entry, key) {
    if (this.tile_sources.has(key)) {
      return this.tile_sources.get(key);
    }
    if (this.tile_source_resolver) {
      return this.tile_source_resolver(entry, key, this.tile_manifest);
    }
    return entry?.payload ?? entry?.path ?? entry?.url ?? null;
  }

  update_tile_streaming(camera_position, view_data = null) {
    if (!this.tile_streaming_enabled || !this.tile_manifest) {
      return;
    }

    const center_coord = svlm_world_to_tile_coord(camera_position, this.tile_manifest.tile_size);
    const radius = Math.max(0, Math.floor(Number(this.config.streaming_radius) || 0));
    const hysteresis = Math.max(0, Math.floor(Number(this.config.streaming_hysteresis) || 0));
    const prefetch_tiles = Math.max(
      0,
      Math.floor(Number(this.config.streaming_prefetch_tiles) || 0)
    );
    const movement = this.last_streaming_center_coord
      ? center_coord.map((component, axis) => component - this.last_streaming_center_coord[axis])
      : [0, 0, 0];
    if (movement.some((component) => component !== 0)) {
      this.streaming_prefetch_direction = movement.map((component) =>
        clamp(component, -1, 1)
      );
    }
    const prefetch_coord = center_coord.map(
      (component, axis) =>
        component + this.streaming_prefetch_direction[axis] * prefetch_tiles
    );
    const view_direction = view_data?.forward;
    const direction_signature = view_direction
      ? `${Math.round(view_direction[0] * 8)}:${Math.round(view_direction[1] * 8)}:${Math.round(
          view_direction[2] * 8
        )}`
      : "none";
    const signature =
      `${svlm_tile_key(center_coord)}|${svlm_tile_key(prefetch_coord)}|` +
      `${radius}|${hysteresis}|${this.config.streaming_memory_budget_mb}|${direction_signature}`;
    if (signature === this.last_tile_streaming_signature) {
      return;
    }
    this.last_tile_streaming_signature = signature;
    this.last_streaming_center_coord = center_coord;
    this.streaming_update_serial++;

    const candidates = new Map();
    const add_candidates = (streaming_center, is_core) => {
      for (const coord of enumerate_svlm_tile_radius(streaming_center, radius)) {
        const key = svlm_tile_key(coord);
        const entry = this.tile_entries.get(key);
        if (!entry) continue;
        const distance_squared = distance_squared_between_tile_coords(coord, center_coord);
        const existing = candidates.get(key);
        if (!existing || (is_core && !existing.is_core)) {
          candidates.set(key, {
            key,
            coord,
            entry,
            is_core,
            distance_squared,
            visible: svlm_tile_intersects_frustum(
              coord,
              this.tile_manifest.tile_size,
              view_data?.frustum
            ),
          });
        }
      }
    };
    add_candidates(center_coord, true);
    if (prefetch_tiles > 0 && svlm_tile_key(prefetch_coord) !== svlm_tile_key(center_coord)) {
      add_candidates(prefetch_coord, false);
    }

    const prioritized_candidates = Array.from(candidates.values()).sort(
      (a, b) =>
        Number(b.is_core) - Number(a.is_core) ||
        Number(b.visible) - Number(a.visible) ||
        a.distance_squared - b.distance_squared ||
        a.key.localeCompare(b.key)
    );
    const memory_budget = this._get_streaming_memory_budget_bytes();
    let selected_bytes = 0;
    const desired_keys = new Set();
    const core_keys = new Set();
    for (const candidate of prioritized_candidates) {
      const byte_length = this._estimate_streamed_tile_gpu_bytes(candidate.entry);
      if (selected_bytes > 0 && selected_bytes + byte_length > memory_budget) continue;
      desired_keys.add(candidate.key);
      if (candidate.is_core) core_keys.add(candidate.key);
      selected_bytes += byte_length;
    }
    this.desired_tile_keys = desired_keys;
    this.core_tile_keys = core_keys;

    const retained_keys = new Set(desired_keys);
    for (const coord of enumerate_svlm_tile_radius(center_coord, radius + hysteresis)) {
      const key = svlm_tile_key(coord);
      if (this.tile_entries.has(key)) retained_keys.add(key);
    }
    this.retained_tile_keys = retained_keys;

    for (const key of desired_keys) {
      if (this.resident_tiles.has(key)) {
        this.resident_tile_usage.set(key, this.streaming_update_serial);
      }
    }

    for (const key of this.failed_tile_keys) {
      if (!desired_keys.has(key)) {
        this.failed_tile_keys.delete(key);
      }
    }
    for (const [key, request] of this.tile_requests) {
      if (!desired_keys.has(key)) {
        request.cancel(`SVLM tile '${key}' left the streaming request set.`);
        this.tile_requests.delete(key);
      }
    }

    this._evict_streamed_tiles_outside_retention();

    for (const { key } of prioritized_candidates) {
      if (!desired_keys.has(key)) continue;
      if (this.tile_requests.size >= this.config.streaming_max_requests) break;
      if (
        this.resident_tiles.has(key) ||
        this.tile_requests.has(key) ||
        this.failed_tile_keys.has(key)
      ) {
        continue;
      }

      const entry = this.tile_entries.get(key);
      const request = StreamingSystem.stream(svlm_tile_stream_provider_type, this, {
        key,
        entry,
        source: () => this.resolve_svlm_tile_source(entry, key),
      });
      this.tile_requests.set(key, request);
      request.finished.then(() => {
        if (this.tile_requests.get(key) === request) {
          this.tile_requests.delete(key);
        }
        if (request.error && request.status === "failed") {
          this.failed_tile_keys.add(key);
          this.tile_serialization_error = request.error;
          this.stats.tile_serialization_error = request.error.message;
        }
        this.last_tile_streaming_signature = null;
        this.stats.requested_tile_count = this.tile_requests.size;
      });
    }

    this.stats.requested_tile_count = this.tile_requests.size;
    this.stats.streaming_memory_budget_bytes = memory_budget;
  }

  _evict_streamed_tiles_outside_retention() {
    const core_ready = Array.from(this.core_tile_keys).every((key) =>
      this.resident_tiles.has(key)
    );
    if (!core_ready) return;
    for (const key of Array.from(this.resident_tiles.keys())) {
      if (!this.retained_tile_keys.has(key)) this.remove_streamed_svlm_tile(key);
    }
  }

  _make_room_for_streamed_tile(tile) {
    const memory_budget = this._get_streaming_memory_budget_bytes();
    const incoming_bytes = this._estimate_streamed_tile_gpu_bytes(tile);
    if (incoming_bytes > memory_budget) {
      throw new Error(
        `SVLM tile '${tile.key}' requires ${incoming_bytes} bytes, exceeding the ` +
          `${memory_budget}-byte streaming budget.`
      );
    }

    let resident_bytes = this._get_resident_tile_bytes();
    if (this.resident_tiles.has(tile.key)) {
      resident_bytes -= this._estimate_streamed_tile_gpu_bytes(this.resident_tiles.get(tile.key));
    }
    if (resident_bytes + incoming_bytes <= memory_budget) return;

    const candidates = Array.from(this.resident_tiles.values())
      .filter((resident_tile) => resident_tile.key !== tile.key)
      .sort((a, b) => {
        const a_desired = this.desired_tile_keys.has(a.key);
        const b_desired = this.desired_tile_keys.has(b.key);
        if (a_desired !== b_desired) return Number(a_desired) - Number(b_desired);
        const a_core = this.core_tile_keys.has(a.key);
        const b_core = this.core_tile_keys.has(b.key);
        if (a_core !== b_core) return Number(a_core) - Number(b_core);
        const usage_delta =
          (this.resident_tile_usage.get(a.key) ?? 0) -
          (this.resident_tile_usage.get(b.key) ?? 0);
        if (usage_delta !== 0) return usage_delta;
        const center = this.last_streaming_center_coord ?? tile.coord;
        return (
          distance_squared_between_tile_coords(b.coord, center) -
          distance_squared_between_tile_coords(a.coord, center)
        );
      });

    for (const candidate of candidates) {
      this.remove_streamed_svlm_tile(candidate.key);
      resident_bytes -= this._estimate_streamed_tile_gpu_bytes(candidate);
      if (resident_bytes + incoming_bytes <= memory_budget) return;
    }

    throw new Error(`SVLM could not make room for tile '${tile.key}' within its streaming budget.`);
  }

  _get_resident_tile_bytes() {
    let total = 0;
    for (const tile of this.resident_tiles.values()) {
      total += this._estimate_streamed_tile_gpu_bytes(tile);
    }
    return total;
  }

  _estimate_streamed_tile_gpu_bytes(tile_or_entry) {
    // Serialized bytes contain leaves and irradiance. Reserve another 10% for
    // the bounded-load lookup hash, headers, and allocator fragmentation.
    const serialized_bytes =
      Number(tile_or_entry?.serialized_byte_length ?? tile_or_entry?.byte_length) || 1;
    return Math.max(1, Math.ceil(serialized_bytes * 1.1));
  }

  install_streamed_svlm_tile(tile) {
    if (!this.tile_streaming_enabled || !this.tile_manifest) {
      return;
    }
    if (
      this.tile_manifest.bake_version !== undefined &&
      tile.bake_version !== this.tile_manifest.bake_version
    ) {
      throw new Error(
        `SVLM tile '${tile.key}' uses bake format ${tile.bake_version}; ` +
          `${this.tile_manifest.bake_version} was expected.`
      );
    }
    if (
      Math.abs(tile.tile_size - this.tile_manifest.tile_size) >
      this.tile_manifest.tile_size * 1e-5
    ) {
      throw new Error(
        `SVLM tile '${tile.key}' uses tile size ${tile.tile_size}; ` +
          `${this.tile_manifest.tile_size} was expected.`
      );
    }

    if (this.resident_tiles.has(tile.key)) {
      this.remove_streamed_svlm_tile(tile.key);
    }
    this._make_room_for_streamed_tile(tile);
    this.resident_tiles.set(tile.key, tile);
    this.resident_tile_usage.set(tile.key, this.streaming_update_serial);
    this.streamed_pending_uploads.add(tile.key);
    this.tile_buffers_dirty = true;
    this.stats.resident_tile_count = this.resident_tiles.size;
    this.stats.resident_tile_bytes = this._get_resident_tile_bytes();
    this._evict_streamed_tiles_outside_retention();
  }

  remove_streamed_svlm_tile(key) {
    if (!this.resident_tiles.delete(key)) {
      return false;
    }
    this._remove_streamed_lookup_tile(key);
    const allocation = this.streamed_allocations.get(key);
    if (allocation) {
      release_streamed_range(
        this.streamed_leaf_free_ranges,
        allocation.leaf_offset,
        allocation.leaf_count
      );
      release_streamed_range(
        this.streamed_irradiance_free_ranges,
        allocation.irradiance_word_offset,
        allocation.irradiance_word_count
      );
      this.streamed_allocations.delete(key);
    }
    this.streamed_pending_uploads.delete(key);
    this.resident_tile_usage.delete(key);
    this.tile_buffers_dirty = true;
    this.stats.resident_tile_count = this.resident_tiles.size;
    this.stats.resident_tile_bytes = this._get_resident_tile_bytes();
    return true;
  }

  _flush_streamed_tiles() {
    if (!this.tile_buffers_dirty && this._has_valid_streamed_buffers()) {
      return;
    }
    const tiles = Array.from(this.resident_tiles.values()).sort((a, b) =>
      a.key.localeCompare(b.key)
    );
    let total_leaf_count = 0;
    let total_irradiance_words = 0;
    const lookup_changed_tile_keys = new Set();
    for (const tile of tiles) {
      total_leaf_count += tile.leaf_count;
      total_irradiance_words += tile.irradiance.length;
      if (this.streamed_allocations.has(tile.key)) continue;

      const leaf_allocation = allocate_streamed_range(
        this.streamed_leaf_free_ranges,
        tile.leaf_count,
        this.streamed_leaf_high_water_mark
      );
      this.streamed_leaf_high_water_mark = leaf_allocation.high_water_mark;
      const irradiance_allocation = allocate_streamed_range(
        this.streamed_irradiance_free_ranges,
        tile.irradiance.length,
        this.streamed_irradiance_high_water_mark
      );
      this.streamed_irradiance_high_water_mark = irradiance_allocation.high_water_mark;
      this.streamed_allocations.set(tile.key, {
        leaf_offset: leaf_allocation.offset,
        leaf_count: tile.leaf_count,
        irradiance_word_offset: irradiance_allocation.offset,
        irradiance_word_count: tile.irradiance.length,
      });
      this.streamed_pending_uploads.add(tile.key);
      lookup_changed_tile_keys.add(tile.key);
    }

    const projected_lookup_entries = npot(Math.max(2, total_leaf_count * 2));
    const projected_pool_bytes =
      this.streamed_leaf_high_water_mark * svlm_tile_leaf_words * Uint32Array.BYTES_PER_ELEMENT +
      this.streamed_irradiance_high_water_mark * Uint32Array.BYTES_PER_ELEMENT +
      projected_lookup_entries * TILED_LOOKUP_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT +
      PARAM_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT;
    if (projected_pool_bytes > this._get_streaming_memory_budget_bytes()) {
      this._compact_streamed_allocations(tiles);
      lookup_changed_tile_keys.clear();
      for (const tile of tiles) lookup_changed_tile_keys.add(tile.key);
      const compacted_pool_bytes =
        this.streamed_leaf_high_water_mark *
          svlm_tile_leaf_words *
          Uint32Array.BYTES_PER_ELEMENT +
        this.streamed_irradiance_high_water_mark * Uint32Array.BYTES_PER_ELEMENT +
        projected_lookup_entries * TILED_LOOKUP_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT +
        PARAM_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT;
      if (compacted_pool_bytes > this._get_streaming_memory_budget_bytes()) {
        throw new Error(
          `SVLM resident pages require ${compacted_pool_bytes} bytes, exceeding the ` +
            `${this._get_streaming_memory_budget_bytes()}-byte streaming budget.`
        );
      }
    }

    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const leaf_buffer_grew = this._ensure_streamed_buffer(
      "streamed_leaf_brick_buffer",
      "svlm_streamed_leaf_bricks",
      Math.max(
        svlm_tile_leaf_words,
        this.streamed_leaf_high_water_mark * svlm_tile_leaf_words
      ),
      usage
    );
    const irradiance_buffer_grew = this._ensure_streamed_buffer(
      "streamed_irradiance_buffer",
      "svlm_streamed_probe_irradiance",
      Math.max(svlm_tile_irradiance_words_per_probe, this.streamed_irradiance_high_water_mark),
      usage
    );
    if (leaf_buffer_grew || irradiance_buffer_grew) {
      for (const tile of tiles) this.streamed_pending_uploads.add(tile.key);
    }

    for (const key of this.streamed_pending_uploads) {
      const tile = this.resident_tiles.get(key);
      const allocation = this.streamed_allocations.get(key);
      if (!tile || !allocation) continue;

      const probe_offset =
        allocation.irradiance_word_offset / svlm_tile_irradiance_words_per_probe;
      const adjusted_leaves = tile.leaves.slice();
      for (let local_leaf_index = 0; local_leaf_index < tile.leaf_count; local_leaf_index++) {
        adjusted_leaves[local_leaf_index * svlm_tile_leaf_words + 1] += probe_offset;
      }
      this.streamed_leaf_brick_buffer.write_raw(
        adjusted_leaves,
        allocation.leaf_offset * svlm_tile_leaf_words * Uint32Array.BYTES_PER_ELEMENT
      );
      this.streamed_irradiance_buffer.write_raw(
        tile.irradiance,
        allocation.irradiance_word_offset * Uint32Array.BYTES_PER_ELEMENT
      );
    }
    this.streamed_pending_uploads.clear();

    const lookup_records_by_tile = new Map();
    const manifest_world_min = this.tile_manifest?.world_min;
    const manifest_world_max = this.tile_manifest?.world_max;
    const world_min =
      Array.isArray(manifest_world_min) &&
      manifest_world_min.length === 3 &&
      manifest_world_min.every(Number.isFinite)
        ? manifest_world_min
        : [
            this.params_data[PARAM_WORLD_MIN_X],
            this.params_data[PARAM_WORLD_MIN_Y],
            this.params_data[PARAM_WORLD_MIN_Z],
          ];
    const world_max =
      Array.isArray(manifest_world_max) &&
      manifest_world_max.length === 3 &&
      manifest_world_max.every(Number.isFinite)
        ? manifest_world_max
        : [
            world_min[0] + this.params_data[PARAM_ROOT_DIM_X] * this.params_data[PARAM_ROOT_SIZE],
            world_min[1] + this.params_data[PARAM_ROOT_DIM_Y] * this.params_data[PARAM_ROOT_SIZE],
            world_min[2] + this.params_data[PARAM_ROOT_DIM_Z] * this.params_data[PARAM_ROOT_SIZE],
          ];
    let root_size = 0;
    let max_leaf_level = 0;
    for (const tile of tiles) {
      const allocation = this.streamed_allocations.get(tile.key);
      const cached_lookup_records = this.streamed_lookup_tile_records.get(tile.key);
      if (cached_lookup_records) {
        const tile_root_size = allocation.root_size;
        if (root_size === 0) {
          root_size = tile_root_size;
        } else if (Math.abs(tile_root_size - root_size) > root_size * 1e-4) {
          throw new Error(`SVLM tile '${tile.key}' does not match the baked root scale.`);
        }
        max_leaf_level = Math.max(max_leaf_level, allocation.max_leaf_level);
        lookup_records_by_tile.set(tile.key, cached_lookup_records);
        continue;
      }
      const tile_lookup_records = new Uint32Array(
        tile.leaf_count * TILED_LOOKUP_U32_STRIDE
      );
      const tile_float_words = new Float32Array(
        tile.leaves.buffer,
        tile.leaves.byteOffset,
        tile.leaves.length
      );
      for (let local_leaf_index = 0; local_leaf_index < tile.leaf_count; local_leaf_index++) {
        const source_leaf_base = local_leaf_index * svlm_tile_leaf_words;
        const target_leaf_index = allocation.leaf_offset + local_leaf_index;

        const source_probe_base = tile.leaves[source_leaf_base + 1];
        const source_irradiance_base = source_probe_base * svlm_tile_irradiance_words_per_probe;
        const leaf_irradiance_word_count = PROBES_PER_BRICK * svlm_tile_irradiance_words_per_probe;
        const source_irradiance_end = source_irradiance_base + leaf_irradiance_word_count;
        if (source_irradiance_end > tile.irradiance.length) {
          throw new Error(
            `SVLM tile '${tile.key}' leaf ${local_leaf_index} references irradiance outside its payload.`
          );
        }

        const level = tile.leaves[source_leaf_base];
        const leaf_origin_x = tile_float_words[source_leaf_base + 2];
        const leaf_origin_y = tile_float_words[source_leaf_base + 3];
        const leaf_origin_z = tile_float_words[source_leaf_base + 4];
        const leaf_size = tile_float_words[source_leaf_base + 5];
        if (
          level > 8 ||
          !Number.isFinite(leaf_size) ||
          leaf_size <= 0 ||
          !Number.isFinite(leaf_origin_x) ||
          !Number.isFinite(leaf_origin_y) ||
          !Number.isFinite(leaf_origin_z)
        ) {
          throw new Error(
            `SVLM tile '${tile.key}' leaf ${local_leaf_index} has invalid spatial metadata.`
          );
        }
        const leaf_root_size = leaf_size * 2 ** level;
        if (root_size === 0) {
          root_size = leaf_root_size;
        } else if (Math.abs(leaf_root_size - root_size) > root_size * 1e-4) {
          throw new Error(
            `SVLM tile '${tile.key}' leaf ${local_leaf_index} does not match the baked root scale.`
          );
        }
        max_leaf_level = Math.max(max_leaf_level, level);
        const leaf_coord_x = Math.round((leaf_origin_x - world_min[0]) / leaf_size);
        const leaf_coord_y = Math.round((leaf_origin_y - world_min[1]) / leaf_size);
        const leaf_coord_z = Math.round((leaf_origin_z - world_min[2]) / leaf_size);
        if (
          leaf_coord_x < 0 ||
          leaf_coord_y < 0 ||
          leaf_coord_z < 0 ||
          !Number.isSafeInteger(leaf_coord_x) ||
          !Number.isSafeInteger(leaf_coord_y) ||
          !Number.isSafeInteger(leaf_coord_z)
        ) {
          throw new Error(
            `SVLM tile '${tile.key}' leaf ${local_leaf_index} lies outside the baked hierarchy.`
          );
        }

        const lookup_base = local_leaf_index * TILED_LOOKUP_U32_STRIDE;
        tile_lookup_records[lookup_base] = tile.coord[0] >>> 0;
        tile_lookup_records[lookup_base + 1] = tile.coord[1] >>> 0;
        tile_lookup_records[lookup_base + 2] = tile.coord[2] >>> 0;
        tile_lookup_records[lookup_base + 3] = level;
        tile_lookup_records[lookup_base + 4] = leaf_coord_x;
        tile_lookup_records[lookup_base + 5] = leaf_coord_y;
        tile_lookup_records[lookup_base + 6] = leaf_coord_z;
        tile_lookup_records[lookup_base + 7] = target_leaf_index;
      }
      lookup_records_by_tile.set(tile.key, tile_lookup_records);
      this.streamed_lookup_tile_records.set(tile.key, tile_lookup_records);
      allocation.root_size = root_size;
      allocation.max_leaf_level = max_leaf_level;
    }
    this._update_streamed_lookup_hash(
      lookup_records_by_tile,
      total_leaf_count,
      lookup_changed_tile_keys,
      usage
    );

    this._ensure_streamed_buffer(
      "streamed_params_buffer",
      "svlm_streamed_params",
      PARAM_WORD_COUNT,
      usage
    );
    if (!this.coarse_lookup_buffer?.buffer) {
      const empty_coarse_lookup = new Uint32Array(COARSE_LOOKUP_U32_STRIDE * 2);
      empty_coarse_lookup.fill(INVALID_IDX);
      this._ensure_streamed_buffer(
        "coarse_lookup_buffer",
        "svlm_streamed_coarse_lookup",
        empty_coarse_lookup.length,
        usage
      );
      this.coarse_lookup_buffer.write_raw(empty_coarse_lookup);
    }

    this.streamed_params_data.set(this.params_data);
    this.streamed_params_data[PARAM_WORLD_MIN_X] = world_min[0];
    this.streamed_params_data[PARAM_WORLD_MIN_Y] = world_min[1];
    this.streamed_params_data[PARAM_WORLD_MIN_Z] = world_min[2];
    root_size = Math.max(root_size, 0.0001);
    this.streamed_params_data[PARAM_ROOT_SIZE] = root_size;
    this.streamed_params_data[PARAM_ROOT_DIM_X] = Math.max(
      1,
      Math.round((world_max[0] - world_min[0]) / root_size)
    );
    this.streamed_params_data[PARAM_ROOT_DIM_Y] = Math.max(
      1,
      Math.round((world_max[1] - world_min[1]) / root_size)
    );
    this.streamed_params_data[PARAM_ROOT_DIM_Z] = Math.max(
      1,
      Math.round((world_max[2] - world_min[2]) / root_size)
    );
    this.streamed_params_data[PARAM_MAX_LEVEL] = max_leaf_level;
    this.streamed_params_data[PARAM_WORLD_TILE_SIZE] = this.tile_manifest.tile_size;
    this.streamed_params_data[PARAM_RESIDENT_TILE_COUNT] = tiles.length;
    this.streamed_params_data[PARAM_IRRADIANCE_FORMAT_VERSION] = BAKE_FORMAT_VERSION;
    this.streamed_params_data[PARAM_IRRADIANCE_SH_WORDS_PER_PROBE] = SH_WORDS_PER_PROBE;
    this.streamed_params_data[PARAM_TILE_STREAMING_ENABLED] = 1;
    this.streamed_params_data[PARAM_COARSE_MIN_LOD] = this.coarse_hierarchy?.min_lod ?? 0;
    this.streamed_params_data[PARAM_COARSE_MAX_LOD] = this.coarse_hierarchy?.max_lod ?? 0;
    this.streamed_params_data[PARAM_STREAMING_RADIUS] = this.config.streaming_radius;
    this.streamed_params_data[PARAM_STREAMING_TRANSITION_TILES] =
      this.config.streaming_transition_tiles;
    this.streamed_params_buffer.write_raw(this.streamed_params_data);

    this.streamed_leaf_count = total_leaf_count;
    this.streamed_probe_count = total_irradiance_words / SH_WORDS_PER_PROBE;
    this.stats.streamed_gpu_bytes =
      (this.streamed_params_buffer?.config?.size ?? 0) +
      (this.streamed_lookup_buffer?.config?.size ?? 0) +
      (this.streamed_leaf_brick_buffer?.config?.size ?? 0) +
      (this.streamed_irradiance_buffer?.config?.size ?? 0);
    this.tile_buffers_dirty = false;
  }

  _remove_streamed_lookup_tile(key) {
    this.streamed_lookup_tile_records.delete(key);
    const entry_indices = this.streamed_lookup_tile_entries.get(key);
    if (!entry_indices || !this.streamed_lookup_data) return;
    for (const entry_index of entry_indices) {
      const entry_base = entry_index * TILED_LOOKUP_U32_STRIDE;
      this.streamed_lookup_data[entry_base + 7] = TOMBSTONE_IDX;
      this.streamed_lookup_dirty_entries.add(entry_index);
    }
    this.streamed_lookup_tile_entries.delete(key);
    this.streamed_lookup_record_count = Math.max(
      0,
      this.streamed_lookup_record_count - entry_indices.length
    );
    this.streamed_lookup_tombstone_count += entry_indices.length;
  }

  _compact_streamed_allocations(tiles) {
    let leaf_offset = 0;
    let irradiance_word_offset = 0;
    for (const tile of tiles) {
      const allocation = this.streamed_allocations.get(tile.key);
      allocation.leaf_offset = leaf_offset;
      allocation.irradiance_word_offset = irradiance_word_offset;
      leaf_offset += allocation.leaf_count;
      irradiance_word_offset += allocation.irradiance_word_count;
      this.streamed_pending_uploads.add(tile.key);
    }
    this.streamed_leaf_high_water_mark = leaf_offset;
    this.streamed_irradiance_high_water_mark = irradiance_word_offset;
    this.streamed_leaf_free_ranges.length = 0;
    this.streamed_irradiance_free_ranges.length = 0;
    this.streamed_lookup_data = null;
    this.streamed_lookup_tile_entries.clear();
    this.streamed_lookup_tile_records.clear();
    this.streamed_lookup_dirty_entries.clear();
    this.streamed_lookup_record_count = 0;
    this.streamed_lookup_tombstone_count = 0;
  }

  _insert_streamed_lookup_tile(key, records) {
    if (!this.streamed_lookup_data) return false;
    this._remove_streamed_lookup_tile(key);
    const entry_count = this.streamed_lookup_data.length / TILED_LOOKUP_U32_STRIDE;
    const entry_mask = entry_count - 1;
    const inserted_entries = [];

    for (let record_index = 0; record_index < records.length / TILED_LOOKUP_U32_STRIDE; record_index++) {
      const record_base = record_index * TILED_LOOKUP_U32_STRIDE;
      let entry_index =
        hash_svlm_tiled_lookup_key(
          records[record_base],
          records[record_base + 1],
          records[record_base + 2],
          records[record_base + 3],
          records[record_base + 4],
          records[record_base + 5],
          records[record_base + 6]
        ) & entry_mask;
      let inserted = false;
      for (let probe = 0; probe < TILED_LOOKUP_MAX_PROBES; probe++) {
        const entry_base = entry_index * TILED_LOOKUP_U32_STRIDE;
        const leaf_index = this.streamed_lookup_data[entry_base + 7];
        if (leaf_index === INVALID_IDX || leaf_index === TOMBSTONE_IDX) {
          if (leaf_index === TOMBSTONE_IDX) this.streamed_lookup_tombstone_count--;
          this.streamed_lookup_data.set(
            records.subarray(record_base, record_base + TILED_LOOKUP_U32_STRIDE),
            entry_base
          );
          this.streamed_lookup_dirty_entries.add(entry_index);
          inserted_entries.push(entry_index);
          inserted = true;
          break;
        }
        entry_index = (entry_index + 1) & entry_mask;
      }
      if (inserted) continue;

      for (const inserted_entry of inserted_entries) {
        const inserted_base = inserted_entry * TILED_LOOKUP_U32_STRIDE;
        this.streamed_lookup_data[inserted_base + 7] = TOMBSTONE_IDX;
        this.streamed_lookup_dirty_entries.add(inserted_entry);
        this.streamed_lookup_tombstone_count++;
      }
      return false;
    }

    this.streamed_lookup_tile_entries.set(key, inserted_entries);
    this.streamed_lookup_record_count += inserted_entries.length;
    return true;
  }

  _rebuild_streamed_lookup_hash(records_by_tile, record_count, minimum_entry_count = 0) {
    const records = new Uint32Array(record_count * TILED_LOOKUP_U32_STRIDE);
    let record_offset = 0;
    for (const tile_records of records_by_tile.values()) {
      records.set(tile_records, record_offset);
      record_offset += tile_records.length;
    }
    this.streamed_lookup_data = build_svlm_tiled_lookup_hash(
      records,
      record_count,
      minimum_entry_count
    );
    this.streamed_lookup_tile_entries.clear();
    const entry_count = this.streamed_lookup_data.length / TILED_LOOKUP_U32_STRIDE;
    for (let entry_index = 0; entry_index < entry_count; entry_index++) {
      const entry_base = entry_index * TILED_LOOKUP_U32_STRIDE;
      if (this.streamed_lookup_data[entry_base + 7] === INVALID_IDX) continue;
      const key = svlm_tile_key([
        this.streamed_lookup_data[entry_base] | 0,
        this.streamed_lookup_data[entry_base + 1] | 0,
        this.streamed_lookup_data[entry_base + 2] | 0,
      ]);
      let tile_entries = this.streamed_lookup_tile_entries.get(key);
      if (!tile_entries) {
        tile_entries = [];
        this.streamed_lookup_tile_entries.set(key, tile_entries);
      }
      tile_entries.push(entry_index);
    }
    this.streamed_lookup_record_count = record_count;
    this.streamed_lookup_tombstone_count = 0;
    this.streamed_lookup_dirty_entries.clear();
  }

  _update_streamed_lookup_hash(records_by_tile, record_count, changed_tile_keys, usage) {
    const required_entry_count = npot(Math.max(2, record_count * 2));
    const current_entry_count = this.streamed_lookup_data
      ? this.streamed_lookup_data.length / TILED_LOOKUP_U32_STRIDE
      : 0;
    let full_upload = false;
    if (
      !this.streamed_lookup_data ||
      current_entry_count < required_entry_count ||
      this.streamed_lookup_tombstone_count > current_entry_count / 4
    ) {
      this._rebuild_streamed_lookup_hash(
        records_by_tile,
        record_count,
        Math.max(current_entry_count, required_entry_count)
      );
      full_upload = true;
    } else {
      for (const key of changed_tile_keys) {
        const records = records_by_tile.get(key);
        if (!records || this._insert_streamed_lookup_tile(key, records)) continue;
        this._rebuild_streamed_lookup_hash(
          records_by_tile,
          record_count,
          Math.max(required_entry_count, current_entry_count * 2)
        );
        full_upload = true;
        break;
      }
    }

    const buffer_grew = this._ensure_streamed_buffer(
      "streamed_lookup_buffer",
      "svlm_streamed_tiled_lookup",
      this.streamed_lookup_data.length,
      usage
    );
    full_upload ||= buffer_grew || this.streamed_lookup_dirty_entries.size > 64;
    if (full_upload) {
      this.streamed_lookup_buffer.write_raw(this.streamed_lookup_data);
      this.streamed_lookup_dirty_entries.clear();
      return;
    }

    const dirty_entries = Array.from(this.streamed_lookup_dirty_entries).sort((a, b) => a - b);
    let range_start = dirty_entries[0];
    let range_end = range_start;
    const upload_range = () => {
      const word_start = range_start * TILED_LOOKUP_U32_STRIDE;
      const word_end = (range_end + 1) * TILED_LOOKUP_U32_STRIDE;
      this.streamed_lookup_buffer.write_raw(
        this.streamed_lookup_data.subarray(word_start, word_end),
        word_start * Uint32Array.BYTES_PER_ELEMENT
      );
    };
    for (let index = 1; index < dirty_entries.length; index++) {
      const entry_index = dirty_entries[index];
      if (entry_index === range_end + 1) {
        range_end = entry_index;
        continue;
      }
      upload_range();
      range_start = entry_index;
      range_end = entry_index;
    }
    if (range_start !== undefined) upload_range();
    this.streamed_lookup_dirty_entries.clear();
  }

  _ensure_streamed_buffer(field, name, required_word_count, usage) {
    const required_bytes = required_word_count * Uint32Array.BYTES_PER_ELEMENT;
    if (required_bytes > MAX_STORAGE_BINDING_SIZE) {
      throw new Error(
        `SVLM buffer '${name}' requires ${required_bytes} bytes, exceeding the storage-buffer limit.`
      );
    }
    const existing = this[field];
    if (existing?.buffer && existing.config.size >= required_bytes) return false;

    const current_words = Math.floor((existing?.config?.size ?? 0) / Uint32Array.BYTES_PER_ELEMENT);
    const capacity_words = Math.min(
      Math.floor(MAX_STORAGE_BINDING_SIZE / Uint32Array.BYTES_PER_ELEMENT),
      Math.max(required_word_count, current_words, 1)
    );
    this[field] = Buffer.create({
      name,
      size: capacity_words,
      usage,
    });
    return true;
  }

  async _read_gpu_buffer_words(buffer, word_count, label) {
    if (!buffer?.buffer || word_count <= 0) {
      return new Uint32Array(0);
    }

    const byte_length = word_count * Uint32Array.BYTES_PER_ELEMENT;
    if (byte_length > buffer.config.size) {
      throw new Error(
        `${label} requested ${byte_length} bytes from a ${buffer.config.size}-byte buffer.`
      );
    }

    const renderer = Renderer.get();
    const staging = renderer.device.createBuffer({
      label: `${buffer.config.name}_tile_serialization`,
      size: byte_length,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = renderer.device.createCommandEncoder({
      label: `${buffer.config.name}_tile_serialization_encoder`,
    });
    encoder.copyBufferToBuffer(buffer.buffer, 0, staging, 0, byte_length);
    renderer.device.queue.submit([encoder.finish()]);

    try {
      await staging.mapAsync(GPUMapMode.READ);
      return new Uint32Array(staging.getMappedRange(0, byte_length).slice(0));
    } finally {
      if (staging.mapState !== "unmapped") {
        staging.unmap();
      }
      staging.destroy();
    }
  }

  _queue_tile_bake_partition() {
    const bake_serial = this.bake_serial;
    const leaf_count = Math.min(this.counters_data[COUNTER_LEAF_COUNT] || 0, this.config.max_nodes);
    const leaf_buffer = this.leaf_brick_buffer;

    this.hierarchy_probe_count = Math.min(
      this.counters_data[COUNTER_PROBE_COUNT] || 0,
      leaf_count * PROBES_PER_BRICK
    );
    this.hierarchy_counters_data = this.counters_data.slice();
    this.stats.tile_serialization_in_progress = true;
    this.tile_serialization_error = null;
    this.stats.tile_serialization_error = null;

    this.tile_bake_partition_promise = this._read_gpu_buffer_words(
      leaf_buffer,
      leaf_count * LEAF_U32_STRIDE,
      "SVLM leaf bricks"
    )
      .then((leaves) => {
        if (this.bake_serial !== bake_serial) {
          return;
        }

        const tiles = partition_svlm_leaf_tiles({
          bake_version: BAKE_FORMAT_VERSION,
          tile_size: this.config.world_tile_size,
          leaves,
        });
        if (tiles.length === 0) {
          throw new Error("SVLM hierarchy did not produce any world tiles to bake.");
        }

        const words_per_leaf_irradiance = PROBES_PER_BRICK * SH_WORDS_PER_PROBE;
        this.tile_bake_tiles = tiles;
        this.tile_bake_payloads.clear();
        this.tile_bake_tile_index = 0;
        this.tile_bake_leaf_offset = 0;
        this.tile_bake_current_chunk = null;
        this.tile_bake_completed_probe_samples = 0;
        this.tile_bake_total_irradiance_words = tiles.reduce(
          (sum, tile) => sum + (tile.leaves.length / LEAF_U32_STRIDE) * words_per_leaf_irradiance,
          0
        );
        this.tile_bake_required_probe_samples =
          (this.tile_bake_total_irradiance_words / SH_WORDS_PER_PROBE) *
          this.config.irradiance_sample_count;
        this.irradiance_allocation_pending = false;
        this.tile_bake_start_pending = true;
        this.stats.serialized_tile_count = 0;
        this.stats.bake_tile_count = tiles.length;
        this.stats.bake_tile_index = 0;
      })
      .catch((partition_error) => {
        if (this.bake_serial === bake_serial) {
          this._fail_tile_bake(partition_error);
        }
      })
      .finally(() => {
        if (this.bake_serial === bake_serial) {
          this.tile_bake_partition_promise = null;
        }
      });
  }

  _start_next_tile_bake_chunk() {
    const tile = this.tile_bake_tiles[this.tile_bake_tile_index];
    if (!tile) {
      return false;
    }

    const tile_leaf_count = tile.leaves.length / LEAF_U32_STRIDE;
    if (!tile.irradiance) {
      tile.irradiance = new Uint32Array(tile_leaf_count * PROBES_PER_BRICK * SH_WORDS_PER_PROBE);
    }

    const chunk_leaf_count = Math.min(
      tile_leaf_count - this.tile_bake_leaf_offset,
      MAX_IRRADIANCE_LEAVES_PER_CHUNK
    );
    if (chunk_leaf_count <= 0) {
      this._fail_tile_bake(new Error(`SVLM tile '${tile.key}' has no leaves to bake.`));
      return false;
    }

    const chunk_leaves = tile.leaves.slice(
      this.tile_bake_leaf_offset * LEAF_U32_STRIDE,
      (this.tile_bake_leaf_offset + chunk_leaf_count) * LEAF_U32_STRIDE
    );
    for (let local_leaf_index = 0; local_leaf_index < chunk_leaf_count; local_leaf_index++) {
      chunk_leaves[local_leaf_index * LEAF_U32_STRIDE + 1] = local_leaf_index * PROBES_PER_BRICK;
    }

    this.irradiance_leaf_buffer = Buffer.create({
      name: "svlm_irradiance_tile_leaves",
      size: Math.max(LEAF_U32_STRIDE, chunk_leaves.length),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.irradiance_leaf_buffer.write_raw(chunk_leaves);

    const probe_count = chunk_leaf_count * PROBES_PER_BRICK;
    this.tile_bake_chunk_serial =
      (this.tile_bake_chunk_serial + 1) & IRRADIANCE_STATUS_CHUNK_GENERATION_MASK;
    if (this.tile_bake_chunk_serial === 0) {
      this.tile_bake_chunk_serial = 1;
    }
    this.irradiance_probe_capacity = probe_count;
    this.tile_bake_peak_probe_capacity = Math.max(this.tile_bake_peak_probe_capacity, probe_count);
    this.tile_bake_current_chunk = {
      tile,
      leaf_offset: this.tile_bake_leaf_offset,
      leaf_count: chunk_leaf_count,
      probe_count,
      chunk_serial: this.tile_bake_chunk_serial,
    };

    this.counters_data.set(this.hierarchy_counters_data);
    this.counters_data[COUNTER_PROBE_COUNT] = probe_count;
    this.counters_data[COUNTER_IRRADIANCE_SAMPLE_INDEX] = 0;
    this.counters_data[COUNTER_IRRADIANCE_COMPLETED_PROBE_SAMPLES] = 0;
    this.counters_data[COUNTER_IRRADIANCE_STATUS] =
      this.tile_bake_chunk_serial << IRRADIANCE_STATUS_CHUNK_GENERATION_SHIFT;
    this._write_counter_buffer();

    this.tile_bake_start_pending = false;
    this.irradiance_bake_in_flight = true;
    return true;
  }

  _queue_current_tile_chunk_finalization() {
    if (
      !this.tile_bake_current_chunk ||
      this.tile_bake_finalize_promise ||
      !this.irradiance_buffer
    ) {
      return;
    }

    const bake_serial = this.bake_serial;
    const chunk = this.tile_bake_current_chunk;
    const chunk_required_probe_samples = chunk.probe_count * this.config.irradiance_sample_count;
    this.irradiance_bake_in_flight = false;

    this.tile_bake_finalize_promise = this._read_gpu_buffer_words(
      this.irradiance_buffer,
      chunk.probe_count * SH_WORDS_PER_PROBE,
      `SVLM tile '${chunk.tile.key}' irradiance`
    )
      .then((irradiance) => {
        if (this.bake_serial !== bake_serial) {
          return;
        }

        const words_per_leaf_irradiance = PROBES_PER_BRICK * SH_WORDS_PER_PROBE;
        chunk.tile.irradiance.set(irradiance, chunk.leaf_offset * words_per_leaf_irradiance);
        this.tile_bake_completed_probe_samples += chunk_required_probe_samples;
        this.tile_bake_leaf_offset += chunk.leaf_count;
        this.tile_bake_current_chunk = null;

        const tile_leaf_count = chunk.tile.leaves.length / LEAF_U32_STRIDE;
        if (this.tile_bake_leaf_offset < tile_leaf_count) {
          this.tile_bake_start_pending = true;
          return;
        }

        this.tile_bake_coarse_samples.push(create_svlm_coarse_tile_sample(chunk.tile));
        const payload = serialize_svlm_tile(chunk.tile);
        chunk.tile.serialized_byte_length = payload.byteLength;
        this.tile_bake_payloads.set(chunk.tile.key, payload);
        chunk.tile.irradiance = null;
        this.stats.serialized_tile_count = this.tile_bake_payloads.size;

        this.tile_bake_tile_index++;
        this.stats.bake_tile_index = this.tile_bake_tile_index;
        this.tile_bake_leaf_offset = 0;
        if (this.tile_bake_tile_index < this.tile_bake_tiles.length) {
          this.tile_bake_start_pending = true;
        } else {
          this._finish_tile_bake();
        }
      })
      .catch((finalize_error) => {
        if (this.bake_serial === bake_serial) {
          this._fail_tile_bake(finalize_error);
        }
      })
      .finally(() => {
        if (this.bake_serial === bake_serial) {
          this.tile_bake_finalize_promise = null;
        }
      });
  }

  _finish_tile_bake() {
    const coarse_hierarchy = create_svlm_coarse_hierarchy(this.tile_bake_coarse_samples, {
      tile_size: this.config.world_tile_size,
      min_lod: this.config.coarse_min_lod,
      max_lod: this.config.coarse_max_lod,
      // The runtime hash keeps a <= 0.5 load factor and rounds to a power of
      // two. Reserve the worst-case 4x dense-record footprint so the fallback
      // remains a deliberately small, always-resident mip tail.
      max_records: Math.floor(
        (this.config.coarse_memory_budget_mb * 1024 * 1024) /
          (COARSE_LOOKUP_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT * 4)
      ),
    });
    const coarse_payload = serialize_svlm_coarse_hierarchy(coarse_hierarchy);
    const manifest = create_svlm_tile_manifest(this.tile_bake_tiles, {
      bake_version: BAKE_FORMAT_VERSION,
      bake_serial: this.bake_serial,
      tile_size: this.config.world_tile_size,
      world_min: this.stats.world_min,
      world_max: this.stats.world_max,
      coarse: {
        entry: "coarse",
        byte_length: coarse_payload.byteLength,
        min_lod: coarse_hierarchy.min_lod,
        max_lod: coarse_hierarchy.max_lod,
        record_count: coarse_hierarchy.records.length,
      },
      streaming: {
        streaming_radius: this.config.streaming_radius,
        streaming_hysteresis: this.config.streaming_hysteresis,
        streaming_prefetch_tiles: this.config.streaming_prefetch_tiles,
        streaming_max_requests: this.config.streaming_max_requests,
        streaming_memory_budget_mb: this.config.streaming_memory_budget_mb,
        streaming_transition_tiles: this.config.streaming_transition_tiles,
        coarse_memory_budget_mb: this.config.coarse_memory_budget_mb,
      },
    });
    this.configure_tile_streaming(manifest, {
      payloads: this.tile_bake_payloads,
      coarse_payload,
    });
    this.tile_bake_tiles = [];
    this.tile_bake_coarse_samples = [];
    this.tile_bake_payloads.clear();
    this.serialized_bake_serial = this.bake_serial;
    this.stats.irradiance_usable = true;
    this.stats.irradiance_ready = true;
    this.stats.irradiance_in_progress = false;
    this.stats.irradiance_progress = 1;
    this.stats.tile_serialization_in_progress = false;
    this.tile_bake_completion_resolve?.({
      status: "complete",
      bake_serial: this.bake_serial,
      manifest,
    });
    this.tile_bake_completion_resolve = null;
    this._release_tile_bake_gpu_buffers();
  }

  _fail_tile_bake(tile_error) {
    const normalized_error =
      tile_error instanceof Error ? tile_error : new Error(String(tile_error));
    this.tile_serialization_error = normalized_error;
    this.stats.tile_serialization_error = normalized_error.message;
    this.stats.tile_serialization_in_progress = false;
    this.stats.irradiance_in_progress = false;
    this.irradiance_allocation_pending = false;
    this.irradiance_bake_in_flight = false;
    this.tile_bake_start_pending = false;
    this.tile_bake_completion_resolve?.({
      status: "failed",
      bake_serial: this.bake_serial,
      error: normalized_error,
    });
    this.tile_bake_completion_resolve = null;
    this._release_tile_bake_gpu_buffers();
  }

  _release_tile_bake_gpu_buffers() {
    this.irradiance_leaf_buffer?.destroy();
    this.irradiance_buffer?.destroy();
    this.irradiance_ray_buffer?.destroy();
    this.emissive_light_buffer?.destroy();
    this.irradiance_leaf_buffer = null;
    this.irradiance_buffer = null;
    this.irradiance_ray_buffer = null;
    this.emissive_light_buffer = null;
    this.irradiance_probe_capacity = 0;
  }

  _reset_tile_bake_state(message = "SVLM bake state was reset.", options = {}) {
    if (!options.preserve_completion) {
      this.tile_bake_completion_resolve?.({
        status: "cancelled",
        bake_serial: this.bake_serial,
        message,
        error: new Error(message),
      });
      this.tile_bake_completion_resolve = null;
      this.tile_bake_completion_promise = null;
    }
    this.tile_bake_tiles = [];
    this.tile_bake_coarse_samples = [];
    this.tile_bake_payloads.clear();
    this.tile_bake_tile_index = 0;
    this.tile_bake_leaf_offset = 0;
    this.tile_bake_current_chunk = null;
    this.tile_bake_chunk_serial = 0;
    this.tile_bake_partition_promise = null;
    this.tile_bake_finalize_promise = null;
    this.tile_bake_start_pending = false;
    this.tile_bake_emissive_ready = false;
    this.tile_bake_completed_probe_samples = 0;
    this.tile_bake_required_probe_samples = 0;
    this.tile_bake_total_irradiance_words = 0;
    this.tile_bake_peak_probe_capacity = 0;
    this.hierarchy_probe_count = 0;
    this.hierarchy_counters_data = null;
    this._release_tile_bake_gpu_buffers();
  }

  _clear_streamed_residency() {
    for (const request of this.tile_requests.values()) {
      request.cancel("SVLM tile residency was reset.");
    }
    this.tile_requests.clear();
    this.resident_tiles.clear();
    this.failed_tile_keys.clear();
    this.resident_tile_usage.clear();
    this.desired_tile_keys.clear();
    this.core_tile_keys.clear();
    this.retained_tile_keys.clear();
    this.last_tile_streaming_signature = null;
    this.last_streaming_center_coord = null;
    this.streaming_prefetch_direction = [0, 0, 0];
    this.streaming_update_serial = 0;
    this.tile_buffers_dirty = false;
    this.streamed_leaf_count = 0;
    this.streamed_probe_count = 0;
    this.streamed_allocations.clear();
    this.streamed_pending_uploads.clear();
    this.streamed_lookup_data = null;
    this.streamed_lookup_tile_entries.clear();
    this.streamed_lookup_tile_records.clear();
    this.streamed_lookup_dirty_entries.clear();
    this.streamed_lookup_record_count = 0;
    this.streamed_lookup_tombstone_count = 0;
    this.streamed_leaf_free_ranges.length = 0;
    this.streamed_irradiance_free_ranges.length = 0;
    this.streamed_leaf_high_water_mark = 0;
    this.streamed_irradiance_high_water_mark = 0;
    this.stats.resident_tile_count = 0;
    this.stats.requested_tile_count = 0;
    this.stats.resident_tile_bytes = 0;
    this.stats.streamed_gpu_bytes = 0;
    this.stats.streaming_memory_budget_bytes = this._get_streaming_memory_budget_bytes();
    this.stats.coarse_record_count = 0;
    this.stats.coarse_bytes = 0;
    this._release_streamed_buffers();
  }

  _reset_tile_streaming(message = "SVLM tile streaming was reset.", options = {}) {
    this._reset_tile_bake_state(message, options);
    this._clear_streamed_residency();
    this._clear_coarse_hierarchy();
    this.tile_manifest = null;
    this.tile_entries.clear();
    this.tile_sources.clear();
    this.tile_source_resolver = null;
    this.tile_streaming_enabled = false;
    this.tile_serialization_error = null;
    this.serialized_bake_serial = -1;
    this.stats.serialized_tile_count = 0;
    this.stats.tile_streaming_enabled = false;
    this.stats.tile_serialization_in_progress = false;
    this.stats.tile_serialization_error = null;
  }

  _release_streamed_buffers() {
    this.streamed_params_buffer?.destroy();
    this.streamed_lookup_buffer?.destroy();
    this.streamed_leaf_brick_buffer?.destroy();
    this.streamed_irradiance_buffer?.destroy();
    this.streamed_params_buffer = null;
    this.streamed_lookup_buffer = null;
    this.streamed_leaf_brick_buffer = null;
    this.streamed_irradiance_buffer = null;
  }

  bake(options = {}) {
    const is_auto_resize = options._auto_resize === true;
    this._reset_tile_streaming(
      is_auto_resize
        ? "SVLM bake restarted with a larger allocation budget."
        : "SVLM bake was superseded by a new bake.",
      { preserve_completion: is_auto_resize }
    );
    this.config = this._clamp_budget_config({ ...this.config, ...this._sanitize_options(options) });
    this.bake_serial += 1;
    if (!is_auto_resize) {
      this.debug_config = this._sanitize_debug_options(options);
      this.tile_bake_completion_promise = new Promise((resolve) => {
        this.tile_bake_completion_resolve = resolve;
      });
    }
    this.bake_requested = true;
    this.bake_in_flight = false;
    this.irradiance_bake_in_flight = false;
    this.irradiance_allocation_pending = false;
    this.irradiance_probe_capacity = 0;

    this.counters_data.fill(0);

    this._write_param_data();
    this._write_param_buffer();

    this._reset_stats();
    this._release_debug_line_buffer();

    return this.stats;
  }

  clear() {
    this._reset_tile_streaming("SVLM bake was cleared.");
    this.bake_requested = false;
    this.bake_in_flight = false;
    this.irradiance_bake_in_flight = false;
    this.irradiance_allocation_pending = false;
    this.irradiance_probe_capacity = 0;
    this.debug_texture = null;

    this.params_data.fill(0);
    this._write_param_data();
    this._write_param_buffer();

    this.counters_data.fill(0);
    this._write_counter_buffer();

    this._reset_stats();
    this._release_debug_line_buffer();
  }

  add_bake_passes(
    render_graph,
    {
      tlas_bvh_info,
      tlas_bvh2_nodes,
      entity_transforms,
      entity_index_lookup,
      blas_directory,
      blas_bvh2_nodes,
      index_buffer,
      dense_lights,
      force_recreate = false,
    }
  ) {
    this._refresh_stats_from_readback();

    const build_hierarchy = this.bake_requested;
    const allocation_complete =
      (this.counters_data[COUNTER_IRRADIANCE_STATUS] & IRRADIANCE_STATUS_ALLOCATION_COMPLETE) !== 0;
    let start_irradiance = false;
    let start_emissive = false;

    if (
      !build_hierarchy &&
      this.irradiance_allocation_pending &&
      allocation_complete &&
      !this.tile_bake_partition_promise
    ) {
      this._queue_tile_bake_partition();
    }

    if (!build_hierarchy && this.tile_bake_start_pending && !this.tile_bake_finalize_promise) {
      try {
        start_irradiance = this._start_next_tile_bake_chunk();
      } catch (tile_start_error) {
        this._fail_tile_bake(tile_start_error);
      }
      if (start_irradiance && !this.tile_bake_emissive_ready) {
        start_emissive = true;
        this.tile_bake_emissive_ready = true;
      }
    }

    const gather_irradiance = this.irradiance_bake_in_flight;
    if (!build_hierarchy && !gather_irradiance) return;

    const node_words = Math.max(1, this.config.max_nodes * NODE_U32_STRIDE);
    const queue_words = Math.max(1, this.config.max_nodes);
    const leaf_words = Math.max(1, this.config.max_nodes * LEAF_U32_STRIDE);
    const rays_per_probe = this.config.irradiance_rays_per_probe;
    const probes_per_batch = this.config.irradiance_probes_per_batch;
    const ray_words = Math.max(4, 4 + probes_per_batch * rays_per_probe * PROBE_RAY_U32_STRIDE);
    const emissive_words = Math.max(4, 4 + this.config.max_emissive_lights * 12);
    const max_dispatch_groups = ceil_div(this.config.max_nodes, THREADS_PER_GROUP);

    // The hierarchy persists for partitioning and debug. Irradiance buffers are
    // reusable tile-chunk scratch allocations with explicit readback seams.
    this.params_buffer = Buffer.create({
      name: "svlm_params",
      raw_data: this.params_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
      force: force_recreate || !this.params_buffer,
    });

    this.counter_buffer = Buffer.create({
      name: "svlm_counters",
      raw_data: this.counters_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      cpu_readback: true,
      own_readback: true,
      force: force_recreate,
    });

    this.node_buffer = Buffer.create({
      name: "svlm_node_pool",
      size: node_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      force: force_recreate,
    });

    this.curr_node_buffer = Buffer.create({
      name: "svlm_curr_nodes",
      size: queue_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    this.next_node_buffer = Buffer.create({
      name: "svlm_next_nodes",
      size: queue_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    this.leaf_brick_buffer = Buffer.create({
      name: "svlm_leaf_bricks",
      size: leaf_words,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      force: force_recreate,
    });

    const params = render_graph.register_buffer(this.params_buffer.config.name);
    const counters = render_graph.register_buffer(this.counter_buffer.config.name);
    const nodes = render_graph.register_buffer(this.node_buffer.config.name);
    const curr_nodes_a = render_graph.register_buffer(this.curr_node_buffer.config.name);
    const curr_nodes_b = render_graph.register_buffer(this.next_node_buffer.config.name);
    const leaves = render_graph.register_buffer(this.leaf_brick_buffer.config.name);
    let irradiance = null;
    let irradiance_leaves = null;
    let ray_data = null;
    let emissive_lights = null;
    let materials = null;
    let textures = null;
    let lighting = null;

    if (gather_irradiance) {
      const irradiance_words = Math.max(
        SH_WORDS_PER_PROBE,
        this.irradiance_probe_capacity * SH_WORDS_PER_PROBE
      );
      this.irradiance_buffer = Buffer.create({
        name: "svlm_probe_irradiance",
        size: irradiance_words,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        // Grow only to the largest tile chunk seen by this bake, then reuse the
        // allocation for all remaining chunks.
        force: force_recreate || !this.irradiance_buffer,
      });

      this.irradiance_ray_buffer = Buffer.create({
        name: "svlm_irradiance_ray_data",
        size: ray_words,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: force_recreate,
      });

      this.emissive_light_buffer = Buffer.create({
        name: "svlm_emissive_lights",
        size: emissive_words,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: force_recreate,
      });

      irradiance = render_graph.register_buffer(this.irradiance_buffer.config.name);
      irradiance_leaves = render_graph.register_buffer(this.irradiance_leaf_buffer.config.name);
      ray_data = render_graph.register_buffer(this.irradiance_ray_buffer.config.name);
      emissive_lights = render_graph.register_buffer(this.emissive_light_buffer.config.name);
      materials = register_material_buffers(render_graph);
      textures = register_texture_pools(render_graph);
      lighting = register_scene_lighting_data(render_graph);
    }

    if (build_hierarchy) {
      this._write_param_buffer();
    }

    if (build_hierarchy) {
      // Reset counters and derive the actual root grid from the TLAS root.
      render_graph.add_pass(
        `svlm_begin_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters, tlas_bvh_info, tlas_bvh2_nodes],
          outputs: [params, counters],
          shader_setup: svlm_begin_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
        }
      );

      // Write one node per root brick and initialize the current frontier.
      render_graph.add_pass(
        `svlm_seed_roots_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters, nodes, curr_nodes_a],
          outputs: [nodes, curr_nodes_a, counters],
          shader_setup: svlm_seed_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(max_dispatch_groups, 1, 1);
        }
      );

      let curr_nodes = curr_nodes_a;
      let next_nodes = curr_nodes_b;
      const levels = Math.max(0, Math.floor(this.config.max_level));
      for (let level = 0; level <= levels; level += 1) {
        // Each active node becomes a leaf brick or eight appended children.
        render_graph.add_pass(
          `svlm_classify_l${level}_${this.bake_serial}`,
          RenderPassFlags.Compute,
          {
            inputs: [
              params,
              counters,
              tlas_bvh_info,
              tlas_bvh2_nodes,
              entity_transforms,
              entity_index_lookup,
              blas_directory,
              blas_bvh2_nodes,
              nodes,
              curr_nodes,
              next_nodes,
              leaves,
            ],
            outputs: [counters, nodes, next_nodes, leaves],
            shader_setup: svlm_classify_shader_setup,
          },
          (graph, frame_data) => {
            graph.get_physical_pass(frame_data.current_pass).dispatch(max_dispatch_groups, 1, 1);
          }
        );

        render_graph.add_pass(
          `svlm_advance_l${level}_${this.bake_serial}`,
          RenderPassFlags.Compute,
          {
            inputs: [counters],
            outputs: [counters],
            shader_setup: svlm_advance_shader_setup,
          },
          (graph, frame_data) => {
            graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
          }
        );

        const tmp = curr_nodes;
        curr_nodes = next_nodes;
        next_nodes = tmp;
      }

      render_graph.add_pass(
        `svlm_finish_allocation_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [counters],
          outputs: [counters],
          shader_setup: svlm_finish_allocation_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
        }
      );
    }

    if (start_emissive) {
      // Build the persistent light list once when the first tile chunk starts.
      render_graph.add_pass(
        `svlm_compact_emissive_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            tlas_bvh2_nodes,
            tlas_bvh_info,
            blas_directory,
            index_buffer,
            entity_transforms,
            materials.params_gpu_buffer,
            materials.material_offsets_buffer,
            materials.material_palette_buffer,
            entity_index_lookup,
            emissive_lights,
            textures.albedo,
            textures.emission,
          ],
          outputs: [emissive_lights],
          shader_setup: compact_emissive_lights_shader_setup,
        },
        (graph, frame_data) => {
          const bounds_buffer = graph.get_physical_buffer(tlas_bvh2_nodes);
          graph.get_physical_buffer(emissive_lights).write_raw(new Uint32Array([0, 0, 0, 0]), 0);
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(
              Math.ceil(Math.floor(bounds_buffer.config.size / 32) / THREADS_PER_GROUP),
              1,
              1
            );
        }
      );
    }

    if (gather_irradiance) {
      // Irradiance is progressive within one bounded tile chunk. The GPU owns
      // its cursor/sample index, while the CPU advances tiles after readback.
      render_graph.add_pass(
        `svlm_irradiance_trace_init_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters, irradiance_leaves, ray_data],
          outputs: [ray_data],
          shader_setup: svlm_irradiance_trace_init_shader_setup,
        },
        (graph, frame_data) => {
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(ceil_div(probes_per_batch * rays_per_probe, THREADS_PER_GROUP), 1, 1);
        }
      );

      render_graph.add_pass(
        `svlm_irradiance_trace_hit_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            ray_data,
            tlas_bvh2_nodes,
            tlas_bvh_info,
            blas_bvh2_nodes,
            blas_directory,
            entity_transforms,
            index_buffer,
            dense_lights,
            emissive_lights,
            entity_index_lookup,
          ],
          outputs: [ray_data],
          shader_setup: svlm_irradiance_trace_hit_shader_setup,
        },
        (graph, frame_data) => {
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(ceil_div(probes_per_batch * rays_per_probe, THREADS_PER_GROUP), 1, 1);
        }
      );

      render_graph.add_pass(
        `svlm_irradiance_trace_shade_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            lighting.scene_lighting_buffer,
            ray_data,
            materials.params_gpu_buffer,
            materials.material_offsets_buffer,
            materials.material_palette_buffer,
            entity_index_lookup,
            textures.albedo,
            textures.emission,
            lighting.skybox_image,
          ],
          outputs: [ray_data],
          shader_setup: svlm_irradiance_trace_shade_shader_setup,
        },
        (graph, frame_data) => {
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(ceil_div(probes_per_batch * rays_per_probe, THREADS_PER_GROUP), 1, 1);
        }
      );

      render_graph.add_pass(
        `svlm_irradiance_accumulate_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters, ray_data, irradiance],
          outputs: [counters, irradiance],
          shader_setup: svlm_irradiance_accumulate_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(probes_per_batch, 1, 1);
        }
      );

      render_graph.add_pass(
        `svlm_irradiance_advance_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters],
          outputs: [counters],
          shader_setup: svlm_irradiance_advance_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
        }
      );
    }

    if (build_hierarchy) {
      this.bake_requested = false;
      this.bake_in_flight = true;
      this.irradiance_allocation_pending = true;
      this.irradiance_bake_in_flight = false;
      this.debug_lines_dirty = true;
      this.stats.baked = true;
      this.stats.irradiance_in_progress = false;
    }
  }

  add_debug_geometry_passes(
    render_graph,
    {
      main_albedo_image,
      main_smra_image,
      main_normal_image,
      main_motion_emissive_image,
      main_depth_image,
    }
  ) {
    if (!this.params_buffer || !this.counter_buffer || !this.leaf_brick_buffer) {
      return;
    }

    const debug_leaf_count = Math.max(0, this.stats.leaf_count);
    if (debug_leaf_count === 0) {
      return;
    }
    const debug_floats = Math.max(20, debug_leaf_count * LINES_PER_BOX * LINE_FLOAT_STRIDE);
    const debug_line_count = debug_leaf_count * LINES_PER_BOX;
    const max_debug_groups = ceil_div(debug_leaf_count, THREADS_PER_GROUP);
    const needs_debug_line_update =
      this.debug_lines_dirty ||
      !this.debug_line_buffer ||
      this.debug_line_buffer.config.size < debug_floats * Float32Array.BYTES_PER_ELEMENT;

    // Line data is only allocated when the brick debug view is drawn. The bake
    // data itself is independent of this transient visualization buffer.
    this.debug_line_buffer = Buffer.create({
      name: "svlm_debug_line_data",
      size: debug_floats,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const line_data = render_graph.register_buffer(this.debug_line_buffer.config.name);
    const params = render_graph.register_buffer(this.params_buffer.config.name);
    const counters = render_graph.register_buffer(this.counter_buffer.config.name);
    const leaves = render_graph.register_buffer(this.leaf_brick_buffer.config.name);

    this._write_param_buffer();

    if (needs_debug_line_update) {
      // Rebuild line records only when the hierarchy or debug level changes.
      render_graph.add_pass(
        `svlm_debug_lines_update_${this.bake_serial}`,
        RenderPassFlags.Compute,
        {
          inputs: [params, counters, leaves, line_data],
          outputs: [line_data, counters],
          shader_setup: svlm_debug_shader_setup,
        },
        (graph, frame_data) => {
          graph.get_physical_pass(frame_data.current_pass).dispatch(max_debug_groups, 1, 1);
        }
      );

      this.debug_lines_dirty = false;
    }

    render_graph.add_pass(
      "svlm_debug_brick_lines",
      RenderPassFlags.Graphics,
      {
        inputs: [line_data],
        outputs: [
          main_albedo_image,
          main_smra_image,
          main_normal_image,
          main_motion_emissive_image,
          main_depth_image,
        ],
        shader_setup: {
          pipeline_shaders: {
            vertex: { path: "line.wgsl" },
            fragment: { path: "line.wgsl" },
          },
          rasterizer_state: {
            cull_mode: "none",
          },
        },
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        draw_quad(pass, debug_line_count);
      }
    );
  }

  add_probe_debug_passes(
    render_graph,
    width,
    height,
    depth_texture,
    scene_color,
    force_recreate = false
  ) {
    this.debug_texture = null;

    if (
      !this.params_buffer ||
      !this.leaf_brick_buffer ||
      !this.counter_buffer ||
      !this.irradiance_buffer ||
      this.irradiance_probe_capacity <= 0
    ) {
      return;
    }

    const max_dispatch_leaf_count =
      MAX_COMPUTE_WORKGROUPS * MAX_COMPUTE_WORKGROUPS * PROBE_DEBUG_WORKGROUP_Y;
    const source_leaf_count = Math.min(this.stats.leaf_count, max_dispatch_leaf_count);
    if (source_leaf_count <= 0) {
      return;
    }

    const leaf_group_count = Math.ceil(source_leaf_count / PROBE_DEBUG_WORKGROUP_Y);
    const leaf_page_groups_y = Math.max(1, Math.min(leaf_group_count, MAX_COMPUTE_WORKGROUPS));
    const leaf_page_count_z = Math.ceil(leaf_group_count / leaf_page_groups_y);
    const gather_group_count = ceil_div(source_leaf_count, THREADS_PER_GROUP);
    const gather_page_groups_x = Math.max(1, Math.min(gather_group_count, MAX_COMPUTE_WORKGROUPS));
    const gather_remaining_groups = Math.ceil(gather_group_count / gather_page_groups_x);
    const gather_page_groups_y = Math.max(
      1,
      Math.min(gather_remaining_groups, MAX_COMPUTE_WORKGROUPS)
    );
    const gather_page_count_z = Math.ceil(gather_remaining_groups / gather_page_groups_y);

    // Probe debug scans and splats in paged 3D dispatches so large leaf budgets
    // never exceed maxComputeWorkgroupsPerDimension on a single axis.
    this.params_data[PARAM_DEBUG_LEAF_PAGE_GROUPS_Y] = leaf_page_groups_y;
    this.params_data[PARAM_DEBUG_GATHER_PAGE_GROUPS_X] = gather_page_groups_x;
    this.params_data[PARAM_DEBUG_GATHER_PAGE_GROUPS_Y] = gather_page_groups_y;
    this._write_param_buffer();

    const params = render_graph.register_buffer(this.params_buffer.config.name);
    const counters = render_graph.register_buffer(this.counter_buffer.config.name);
    const leaves = render_graph.register_buffer(this.leaf_brick_buffer.config.name);
    const irradiance = render_graph.register_buffer(this.irradiance_buffer.config.name);

    // One packed u32 per pixel stores closest probe depth plus compact baked
    // irradiance color before resolve composites probe spheres over scene color.
    const debug_depth = render_graph.create_buffer({
      name: "svlm_probe_debug_depth",
      size: Math.max(1, width * height),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const debug_leaf_indices = render_graph.create_buffer({
      name: "svlm_probe_debug_leaf_indices",
      size: Math.max(2, source_leaf_count + 1),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.debug_texture = render_graph.create_image({
      name: "svlm_probe_debug",
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });

    render_graph.add_pass(
      "svlm_probe_debug_clear",
      RenderPassFlags.Compute,
      {
        inputs: [debug_depth, depth_texture, debug_leaf_indices],
        outputs: [debug_depth, debug_leaf_indices],
        shader_setup: svlm_probe_debug_clear_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      "svlm_probe_debug_gather",
      RenderPassFlags.Compute,
      {
        inputs: [params, counters, leaves, debug_leaf_indices],
        outputs: [debug_leaf_indices],
        shader_setup: svlm_probe_debug_gather_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(gather_page_groups_x, gather_page_groups_y, gather_page_count_z);
      }
    );

    render_graph.add_pass(
      "svlm_probe_debug_splat",
      RenderPassFlags.Compute,
      {
        inputs: [
          params,
          counters,
          leaves,
          depth_texture,
          debug_depth,
          debug_leaf_indices,
          irradiance,
        ],
        outputs: [debug_depth],
        shader_setup: svlm_probe_debug_splat_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(PROBES_PER_BRICK / PROBE_DEBUG_WORKGROUP_X),
          leaf_page_groups_y,
          leaf_page_count_z
        );
      }
    );

    render_graph.add_pass(
      "svlm_probe_debug_resolve",
      RenderPassFlags.Compute,
      {
        inputs: [scene_color, debug_depth, this.debug_texture],
        outputs: [this.debug_texture],
        shader_setup: svlm_probe_debug_resolve_shader_setup,
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }

  _sanitize_options(options) {
    const out = {};
    if (options.root_brick_size !== undefined) {
      out.root_brick_size = Math.max(0.0, Number(options.root_brick_size) || 0.0);
    }
    if (options.max_level !== undefined) {
      out.max_level = clamp(Math.floor(Number(options.max_level)), 0, 8);
    }
    if (options.min_level !== undefined) {
      out.min_level = clamp(
        Math.floor(Number(options.min_level)),
        0,
        out.max_level ?? this.config.max_level
      );
    }
    if (options.max_nodes !== undefined) {
      out.max_nodes = Math.max(9, Math.floor(Number(options.max_nodes)));
    }
    if (options.irradiance_rays_per_probe !== undefined) {
      out.irradiance_rays_per_probe = Math.max(
        1,
        Math.floor(Number(options.irradiance_rays_per_probe))
      );
    }
    if (options.irradiance_probes_per_batch !== undefined) {
      out.irradiance_probes_per_batch = Math.max(
        1,
        Math.floor(Number(options.irradiance_probes_per_batch))
      );
    }
    if (options.irradiance_sample_count !== undefined) {
      out.irradiance_sample_count = Math.max(
        1,
        Math.floor(Number(options.irradiance_sample_count))
      );
    }
    if (options.irradiance_max_ray_distance !== undefined) {
      out.irradiance_max_ray_distance = Math.max(
        1.0,
        Number(options.irradiance_max_ray_distance) || 1.0
      );
    }
    if (options.max_emissive_lights !== undefined) {
      out.max_emissive_lights = Math.max(1, Math.floor(Number(options.max_emissive_lights)));
    }
    if (options.world_tile_size !== undefined) {
      out.world_tile_size = Math.max(1.0, Number(options.world_tile_size) || 1.0);
    }
    if (options.streaming_radius !== undefined) {
      out.streaming_radius = clamp(Math.floor(Number(options.streaming_radius) || 0), 0, 16);
    }
    if (options.streaming_hysteresis !== undefined) {
      out.streaming_hysteresis = clamp(
        Math.floor(Number(options.streaming_hysteresis) || 0),
        0,
        8
      );
    }
    if (options.streaming_prefetch_tiles !== undefined) {
      out.streaming_prefetch_tiles = clamp(
        Math.floor(Number(options.streaming_prefetch_tiles) || 0),
        0,
        8
      );
    }
    if (options.streaming_max_requests !== undefined) {
      out.streaming_max_requests = clamp(
        Math.floor(Number(options.streaming_max_requests) || 1),
        1,
        256
      );
    }
    if (options.streaming_memory_budget_mb !== undefined) {
      out.streaming_memory_budget_mb = clamp(
        Number(options.streaming_memory_budget_mb) || 1,
        1,
        1024
      );
    }
    if (options.streaming_transition_tiles !== undefined) {
      out.streaming_transition_tiles = clamp(
        Number(options.streaming_transition_tiles) || 0,
        0,
        4
      );
    }
    if (options.coarse_min_lod !== undefined) {
      out.coarse_min_lod = clamp(Math.floor(Number(options.coarse_min_lod) || 1), 1, 16);
    }
    if (options.coarse_max_lod !== undefined) {
      out.coarse_max_lod = clamp(
        Math.floor(Number(options.coarse_max_lod) || 1),
        out.coarse_min_lod ?? this.config.coarse_min_lod,
        16
      );
    }
    if (options.coarse_memory_budget_mb !== undefined) {
      out.coarse_memory_budget_mb = clamp(
        Number(options.coarse_memory_budget_mb) || 1,
        1,
        128
      );
    }
    return out;
  }

  _clamp_budget_config(config) {
    const limits = this._get_budget_limits();
    const max_nodes = clamp(
      Math.floor(Number(config.max_nodes) || this.config.max_nodes),
      9,
      limits.max_nodes
    );
    const max_rays_per_probe = Math.max(
      1,
      Math.floor(
        (MAX_STORAGE_BINDING_SIZE - 4 * Uint32Array.BYTES_PER_ELEMENT) /
          (PROBE_RAY_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
      )
    );
    const irradiance_rays_per_probe = clamp(
      Math.floor(Number(config.irradiance_rays_per_probe) || this.config.irradiance_rays_per_probe),
      1,
      max_rays_per_probe
    );
    const max_batch_by_ray_buffer = Math.max(
      1,
      Math.floor(
        (MAX_STORAGE_BINDING_SIZE - 4 * Uint32Array.BYTES_PER_ELEMENT) /
          (irradiance_rays_per_probe * PROBE_RAY_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
      )
    );
    const irradiance_probes_per_batch = clamp(
      Math.floor(
        Number(config.irradiance_probes_per_batch) || this.config.irradiance_probes_per_batch
      ),
      1,
      Math.min(MAX_COMPUTE_WORKGROUPS, max_batch_by_ray_buffer)
    );
    const max_emissive_lights = clamp(
      Math.floor(Number(config.max_emissive_lights) || this.config.max_emissive_lights),
      1,
      Math.floor(
        (MAX_STORAGE_BINDING_SIZE - 4 * Uint32Array.BYTES_PER_ELEMENT) /
          (12 * Uint32Array.BYTES_PER_ELEMENT)
      )
    );
    return {
      ...config,
      max_nodes,
      irradiance_rays_per_probe,
      irradiance_probes_per_batch,
      irradiance_sample_count: clamp(
        Math.floor(Number(config.irradiance_sample_count) || this.config.irradiance_sample_count),
        1,
        64
      ),
      irradiance_max_ray_distance: Math.max(
        1.0,
        Number(config.irradiance_max_ray_distance) || this.config.irradiance_max_ray_distance
      ),
      max_emissive_lights,
      world_tile_size: Math.max(1.0, Number(config.world_tile_size) || this.config.world_tile_size),
      streaming_radius: clamp(Math.floor(Number(config.streaming_radius) || 0), 0, 16),
      streaming_hysteresis: clamp(
        Math.floor(Number(config.streaming_hysteresis) || 0),
        0,
        8
      ),
      streaming_prefetch_tiles: clamp(
        Math.floor(Number(config.streaming_prefetch_tiles) || 0),
        0,
        8
      ),
      streaming_max_requests: clamp(
        Math.floor(Number(config.streaming_max_requests) || this.config.streaming_max_requests),
        1,
        256
      ),
      streaming_memory_budget_mb: clamp(
        Number(config.streaming_memory_budget_mb) || this.config.streaming_memory_budget_mb,
        1,
        1024
      ),
      streaming_transition_tiles: clamp(
        Number(config.streaming_transition_tiles) || 0,
        0,
        4
      ),
      coarse_min_lod: clamp(Math.floor(Number(config.coarse_min_lod) || 1), 1, 16),
      coarse_max_lod: clamp(
        Math.floor(Number(config.coarse_max_lod) || 1),
        clamp(Math.floor(Number(config.coarse_min_lod) || 1), 1, 16),
        16
      ),
      coarse_memory_budget_mb: clamp(
        Number(config.coarse_memory_budget_mb) || this.config.coarse_memory_budget_mb,
        1,
        128
      ),
    };
  }

  _get_budget_limits() {
    const max_buffer_bytes = Math.max(4, MAX_STORAGE_BINDING_SIZE);
    const max_nodes_by_storage = Math.floor(
      max_buffer_bytes / (NODE_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
    );
    const max_nodes_by_dispatch = MAX_COMPUTE_WORKGROUPS * THREADS_PER_GROUP;
    const max_leaf_by_leaf_buffer = Math.floor(
      max_buffer_bytes / (LEAF_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT)
    );

    return {
      // Every surviving leaf originates from one node, so sizing both pools
      // from the node budget removes the independent leaf truncation point.
      max_nodes: Math.max(
        9,
        Math.min(max_nodes_by_storage, max_nodes_by_dispatch, max_leaf_by_leaf_buffer)
      ),
    };
  }

  _sanitize_debug_options(options) {
    const out = {};
    if (options.debug_level !== undefined) {
      const level = Number(options.debug_level);
      out.debug_level = Number.isFinite(level)
        ? clamp(Math.floor(level), -1, this.config.max_level)
        : -1;
    }
    return out;
  }

  _write_param_data() {
    this.params_data.fill(0);
    // Fields written by the GPU begin pass, such as world_min/root_dims, are
    // cleared here and then populated after TLAS bounds are known on-GPU.
    this.params_data[PARAM_MAX_LEVEL] = this.config.max_level;
    this.params_data[PARAM_MAX_NODES] = this.config.max_nodes;
    this.params_data[PARAM_LEAF_CAPACITY] = this.config.max_nodes;
    this.params_data[PARAM_MIN_LEVEL] = this.config.min_level;
    this.params_data[PARAM_NEAR_FACTOR] = this.config.near_geometry_factor;
    this.params_data[PARAM_OCC_MIN] = this.config.occupancy_split_min;
    this.params_data[PARAM_OCC_MAX] = this.config.occupancy_split_max;
    this.params_data[PARAM_REQUESTED_ROOT_SIZE] = this.config.root_brick_size;
    this.params_data[PARAM_BAKE_PADDING] = this.config.bake_padding;
    this.params_data[PARAM_BAKE_SERIAL] = this.bake_serial;
    this.params_data[PARAM_DEBUG_LEVEL] = this.debug_config.debug_level;
    this.params_data[PARAM_DEBUG_LEAF_PAGE_GROUPS_Y] = 0;
    this.params_data[PARAM_DEBUG_GATHER_PAGE_GROUPS_X] = 0;
    this.params_data[PARAM_DEBUG_GATHER_PAGE_GROUPS_Y] = 0;
    this.params_data[PARAM_IRRADIANCE_RAYS_PER_PROBE] = this.config.irradiance_rays_per_probe;
    this.params_data[PARAM_IRRADIANCE_PROBES_PER_BATCH] = this.config.irradiance_probes_per_batch;
    this.params_data[PARAM_IRRADIANCE_SAMPLE_COUNT] = this.config.irradiance_sample_count;
    this.params_data[PARAM_IRRADIANCE_MAX_RAY_DISTANCE] = this.config.irradiance_max_ray_distance;
    this.params_data[PARAM_IRRADIANCE_FORMAT_VERSION] = BAKE_FORMAT_VERSION;
    this.params_data[PARAM_IRRADIANCE_SH_WORDS_PER_PROBE] = SH_WORDS_PER_PROBE;
    this.params_data[PARAM_WORLD_TILE_SIZE] = this.config.world_tile_size;
    this.params_data[PARAM_RESIDENT_TILE_COUNT] = 0;
    this.params_data[PARAM_TILE_STREAMING_ENABLED] = 0;
    this.params_data[PARAM_COARSE_MIN_LOD] = this.config.coarse_min_lod;
    this.params_data[PARAM_COARSE_MAX_LOD] = this.config.coarse_max_lod;
    this.params_data[PARAM_STREAMING_RADIUS] = this.config.streaming_radius;
    this.params_data[PARAM_STREAMING_TRANSITION_TILES] =
      this.config.streaming_transition_tiles;
  }

  _write_param_buffer() {
    if (this.params_buffer) {
      this.params_buffer.write_raw(this.params_data);
    }
  }

  _write_counter_buffer() {
    if (this.counter_buffer) {
      this.counter_buffer.write_raw(this.counters_data);
    }
  }

  _refresh_stats_from_readback() {
    const node_count = this.counters_data[COUNTER_NODE_COUNT] || 0;
    const required_leaf_count = this.counters_data[COUNTER_LEAF_COUNT] || 0;
    const leaf_count = Math.min(required_leaf_count, this.config.max_nodes);
    const readback_probe_count = Math.min(
      this.counters_data[COUNTER_PROBE_COUNT] || 0,
      this.config.max_nodes * PROBES_PER_BRICK
    );
    const probe_count = this.hierarchy_probe_count || readback_probe_count;
    const status = this.counters_data[COUNTER_STATUS] || 0;
    const readback_irradiance_sample_index =
      this.counters_data[COUNTER_IRRADIANCE_SAMPLE_INDEX] || 0;
    const readback_completed_probe_samples =
      this.counters_data[COUNTER_IRRADIANCE_COMPLETED_PROBE_SAMPLES] || 0;
    const readback_chunk_serial =
      (this.counters_data[COUNTER_IRRADIANCE_STATUS] >>> IRRADIANCE_STATUS_CHUNK_GENERATION_SHIFT) &
      IRRADIANCE_STATUS_CHUNK_GENERATION_MASK;
    const current_chunk_readback_valid =
      !!this.tile_bake_current_chunk &&
      readback_chunk_serial === this.tile_bake_current_chunk.chunk_serial;
    const current_chunk_completed_probe_samples = current_chunk_readback_valid
      ? readback_completed_probe_samples
      : 0;
    const current_chunk_required_probe_samples =
      (this.tile_bake_current_chunk?.probe_count ?? 0) * this.config.irradiance_sample_count;
    if (
      this.irradiance_bake_in_flight &&
      current_chunk_required_probe_samples > 0 &&
      current_chunk_completed_probe_samples >= current_chunk_required_probe_samples
    ) {
      this._queue_current_tile_chunk_finalization();
    }

    const irradiance_completed_probe_samples =
      this.tile_bake_completed_probe_samples +
      (this.tile_bake_current_chunk
        ? Math.min(current_chunk_completed_probe_samples, current_chunk_required_probe_samples)
        : 0);
    const total_probe_samples =
      this.tile_bake_required_probe_samples || probe_count * this.config.irradiance_sample_count;
    const irradiance_ready =
      this.tile_streaming_enabled &&
      !!this.tile_manifest &&
      this.serialized_bake_serial === (this.tile_manifest.bake_serial ?? this.bake_serial);
    const irradiance_usable = irradiance_ready;
    const irradiance_sample_index = irradiance_ready
      ? this.config.irradiance_sample_count
      : current_chunk_readback_valid
        ? readback_irradiance_sample_index
        : 0;
    const root_dims = [
      Math.floor(this.params_data[PARAM_ROOT_DIM_X] || 0),
      Math.floor(this.params_data[PARAM_ROOT_DIM_Y] || 0),
      Math.floor(this.params_data[PARAM_ROOT_DIM_Z] || 0),
    ];
    const root_brick_size = this.params_data[PARAM_ROOT_SIZE] || 0;
    const world_min = [
      this.params_data[PARAM_WORLD_MIN_X] || 0,
      this.params_data[PARAM_WORLD_MIN_Y] || 0,
      this.params_data[PARAM_WORLD_MIN_Z] || 0,
    ];
    const world_max = [
      world_min[0] + root_dims[0] * root_brick_size,
      world_min[1] + root_dims[1] * root_brick_size,
      world_min[2] + root_dims[2] * root_brick_size,
    ];
    const scene_min = [
      this.params_data[PARAM_SCENE_MIN_X] || 0,
      this.params_data[PARAM_SCENE_MIN_Y] || 0,
      this.params_data[PARAM_SCENE_MIN_Z] || 0,
    ];
    const scene_max = [
      this.params_data[PARAM_SCENE_MAX_X] || 0,
      this.params_data[PARAM_SCENE_MAX_Y] || 0,
      this.params_data[PARAM_SCENE_MAX_Z] || 0,
    ];
    const per_level_counts = [];
    const split_counts = [];
    // Level histograms are counters written during classification. They make it
    // easy to see whether refinement is clustering where geometry actually is.
    for (let i = 0; i <= this.config.max_level; i += 1) {
      per_level_counts.push(this.counters_data[COUNTER_LEVEL_BASE + i] || 0);
      split_counts.push(this.counters_data[COUNTER_SPLIT_BASE + i] || 0);
    }

    const node_bytes = node_count * NODE_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT;
    const leaf_bytes = leaf_count * LEAF_U32_STRIDE * Uint32Array.BYTES_PER_ELEMENT;
    const irradiance_bytes =
      (this.tile_bake_total_irradiance_words || probe_count * SH_WORDS_PER_PROBE) *
      Uint32Array.BYTES_PER_ELEMENT;

    this.stats.baked = this.bake_in_flight || leaf_count > 0;
    this.stats.bake_serial = this.bake_serial;
    this.stats.root_dims = root_dims;
    this.stats.root_brick_size = root_brick_size;
    this.stats.min_level = this.config.min_level;
    this.stats.max_level = this.config.max_level;
    this.stats.max_level_reached = this.counters_data[COUNTER_MAX_LEVEL_REACHED] || 0;
    this.stats.node_count = node_count;
    this.stats.leaf_count = leaf_count;
    this.stats.leaf_capacity = this.config.max_nodes;
    this.stats.probe_count = probe_count;
    this.stats.probes_per_leaf = PROBES_PER_BRICK;
    this.stats.per_level_counts = per_level_counts;
    this.stats.split_counts = split_counts;
    this.stats.world_min = world_min;
    this.stats.world_max = world_max;
    this.stats.scene_min = scene_min;
    this.stats.scene_max = scene_max;
    this.stats.node_bytes = node_bytes;
    this.stats.leaf_bytes = leaf_bytes;
    this.stats.irradiance_bytes = irradiance_bytes;
    this.stats.irradiance_allocated_bytes =
      this.tile_bake_peak_probe_capacity * SH_WORDS_PER_PROBE * Uint32Array.BYTES_PER_ELEMENT;
    this.stats.total_bytes = node_bytes + leaf_bytes + this.stats.irradiance_allocated_bytes;
    this.stats.irradiance_usable = irradiance_usable;
    this.stats.irradiance_ready = irradiance_ready;
    this.stats.irradiance_in_progress =
      this.irradiance_bake_in_flight ||
      this.irradiance_allocation_pending ||
      !!this.tile_bake_partition_promise ||
      !!this.tile_bake_finalize_promise ||
      this.tile_bake_start_pending;
    this.stats.irradiance_allocation_pending = this.irradiance_allocation_pending;
    this.stats.irradiance_capacity_exceeded = false;
    this.stats.irradiance_progress = irradiance_ready
      ? 1
      : total_probe_samples > 0
        ? clamp(irradiance_completed_probe_samples / total_probe_samples, 0, 1)
        : 0;
    this.stats.irradiance_sample_index = irradiance_sample_index;
    this.stats.irradiance_sample_count = this.config.irradiance_sample_count;
    this.stats.irradiance_completed_probe_samples = irradiance_completed_probe_samples;
    this.stats.irradiance_required_probe_samples = total_probe_samples;
    this.stats.irradiance_rays_per_probe = this.config.irradiance_rays_per_probe;
    this.stats.debug_leaf_count = leaf_count;
    this.stats.debug_level = this.debug_config.debug_level;
    this.stats.truncated_by_node_limit =
      (status & (STATUS_NODE_OVERFLOW | STATUS_ROOT_OVERFLOW)) !== 0;
    this.stats.truncated_by_leaf_limit = (status & STATUS_LEAF_OVERFLOW) !== 0;
    this.stats.config = { ...this.config };

    const auto_resize_queued = this._maybe_queue_auto_resize_rebake({
      node_count,
      leaf_count: required_leaf_count,
      status,
    });
    if (auto_resize_queued) {
      return;
    }
  }

  _maybe_queue_auto_resize_rebake({ node_count, leaf_count, status }) {
    if (this.bake_requested || !this.bake_in_flight) {
      return false;
    }

    const node_overflow = (status & (STATUS_NODE_OVERFLOW | STATUS_ROOT_OVERFLOW)) !== 0;
    const leaf_overflow = (status & STATUS_LEAF_OVERFLOW) !== 0;
    if (!node_overflow && !leaf_overflow) {
      return false;
    }

    // Auto-resize is the default fallback: if the GPU build proves a budget was
    // too small, grow within device limits and queue another GPU bake instead of
    // leaving the user to guess the required node/leaf counts.
    const limits = this._get_budget_limits();
    const growth = Math.max(1.25, Number(this.config.auto_resize_growth) || 2.0);
    let next_nodes = this.config.max_nodes;
    if (node_overflow || leaf_overflow) {
      const required_nodes = Math.max(
        node_count + 8,
        leaf_count + 1,
        Math.ceil(this.config.max_nodes * growth)
      );
      next_nodes = Math.min(limits.max_nodes, npot(required_nodes));
    }

    const can_grow = next_nodes > this.config.max_nodes;

    if (!can_grow) {
      return false;
    }

    this.bake({
      max_nodes: next_nodes,
      _auto_resize: true,
    });
    return true;
  }

  _release_debug_line_buffer() {
    if (!this.debug_line_buffer) {
      return;
    }
    this.debug_line_buffer.destroy();
    this.debug_line_buffer = null;
  }

  _reset_stats() {
    this.stats.baked = false;
    this.stats.bake_serial = this.bake_serial;
    this.stats.root_dims = [0, 0, 0];
    this.stats.root_brick_size = 0;
    this.stats.min_level = this.config.min_level;
    this.stats.max_level = this.config.max_level;
    this.stats.max_level_reached = 0;
    this.stats.node_count = 0;
    this.stats.leaf_count = 0;
    this.stats.leaf_capacity = 0;
    this.stats.probe_count = 0;
    this.stats.probes_per_leaf = PROBES_PER_BRICK;
    this.stats.per_level_counts = [];
    this.stats.split_counts = [];
    this.stats.world_min = [0, 0, 0];
    this.stats.world_max = [0, 0, 0];
    this.stats.scene_min = [0, 0, 0];
    this.stats.scene_max = [0, 0, 0];
    this.stats.node_bytes = 0;
    this.stats.leaf_bytes = 0;
    this.stats.irradiance_bytes = 0;
    this.stats.irradiance_allocated_bytes = 0;
    this.stats.total_bytes = 0;
    this.stats.irradiance_usable = false;
    this.stats.irradiance_ready = false;
    this.stats.irradiance_in_progress = false;
    this.stats.irradiance_allocation_pending = false;
    this.stats.irradiance_capacity_exceeded = false;
    this.stats.irradiance_progress = 0;
    this.stats.irradiance_sample_index = 0;
    this.stats.irradiance_sample_count = this.config.irradiance_sample_count;
    this.stats.irradiance_completed_probe_samples = 0;
    this.stats.irradiance_required_probe_samples = 0;
    this.stats.irradiance_rays_per_probe = this.config.irradiance_rays_per_probe;
    this.stats.serialized_tile_count = 0;
    this.stats.bake_tile_count = 0;
    this.stats.bake_tile_index = 0;
    this.stats.resident_tile_count = 0;
    this.stats.requested_tile_count = 0;
    this.stats.resident_tile_bytes = 0;
    this.stats.streamed_gpu_bytes = 0;
    this.stats.streaming_memory_budget_bytes = this._get_streaming_memory_budget_bytes();
    this.stats.coarse_record_count = 0;
    this.stats.coarse_bytes = 0;
    this.stats.tile_streaming_enabled = false;
    this.stats.tile_serialization_in_progress = false;
    this.stats.tile_serialization_error = null;
    this.stats.debug_leaf_count = 0;
    this.stats.debug_level = this.debug_config.debug_level;
    this.stats.truncated_by_node_limit = false;
    this.stats.truncated_by_leaf_limit = false;
    this.stats.config = { ...this.config };
  }
}

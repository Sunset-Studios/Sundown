import { RenderPassFlags } from "../renderer/renderer_types.js";

// ════════════════════════════════════════════════════════════════════════════════════════════════
//  ███████╗ ██████╗███████╗███╗   ██╗███████╗    ██╗   ██╗ ██████╗ ██╗  ██╗███████╗██╗     ███████╗
//  ██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝    ██║   ██║██╔═══██╗╚██╗██╔╝██╔════╝██║     ██╔════╝
//  ███████╗██║     █████╗  ██╔██╗ ██║█████╗      ██║   ██║██║   ██║ ╚███╔╝ █████╗  ██║     ███████╗
//  ╚════██║██║     ██╔══╝  ██║╚██╗██║██╔══╝      ╚██╗ ██╔╝██║   ██║ ██╔██╗ ██╔══╝  ██║     ╚════██║
//  ███████║╚██████╗███████╗██║ ╚████║███████╗     ╚████╔╝ ╚██████╔╝██╔╝ ██╗███████╗███████╗███████║
//  ╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═══╝╚══════╝      ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝
// ════════════════════════════════════════════════════════════════════════════════════════════════
//
// SceneVoxelizer - Render-Graph-Driven GPU Scene Voxelization
//
// This system turns scene geometry into a world-space voxel representation while leaving resource
// lifetime, pass ordering, and synchronization to a caller-owned render graph.
//
// 🧱  CURRENT REPRESENTATION:
//     • A configurable set of nested camera-centered 256 × 256 × 256 clip levels
//     • A 2× world-space voxel-size increase between adjacent clip levels
//     • One occupancy bit per voxel, packed 32 voxels per u32 storage word
//     • Meshlet-driven conservative compute voxelization
//     • A 256³ → 32³ → 4³ → 1³ HDDA skip hierarchy within every clip level
//     • Brick-snapped toroidal scrolling with exposed-slab updates per level
//
// 🔌  RENDER GRAPH CONTRACT:
//     • begin_frame() resolves frame-local configuration and clears semantic handles
//     • setup() creates/imports every resource required by the voxelizer
//     • record() appends reset, voxelization, and finalization stages to the graph
//     • add_passes() is the convenient all-in-one path used by renderer strategies
//
// setup() and record() remain public on purpose. A strategy may allocate all shared resources first,
// then record several dependent systems in an explicit order, just like the GI pipeline modules.
//
// IMPORTANT: The bit-wise occupancy layout is part of the contract. Non-occupancy surface
// attributes remain intentionally unspecified so the compact grid stays usable by lighting,
// tracing, and debug consumers without coupling them to a material representation.
//
// ════════════════════════════════════════════════════════════════════════════════════════════════

// Shader cooking discovers every clip-level variant from this static registry. Runtime pass setup
// objects are created by create_scene_voxel_level_shader_setup() so each graph pass retains its
// own immutable level selection.
const scene_voxel_clipmap_shader_variants = {
  shaders: {
    voxelize: { path: "acceleration/scene_voxelizer.wgsl" },
    mark_dirty: { path: "acceleration/scene_voxelizer_mark_dirty.wgsl" },
    clear_dirty: { path: "acceleration/scene_voxelizer_clear_dirty.wgsl" },
    compact: { path: "acceleration/scene_voxelizer_compact.wgsl" },
  },
  level_0: { defines: { SCENE_VOXEL_CLIP_LEVEL: "0u" } },
  level_1: { defines: { SCENE_VOXEL_CLIP_LEVEL: "1u" } },
  level_2: { defines: { SCENE_VOXEL_CLIP_LEVEL: "2u" } },
  level_3: { defines: { SCENE_VOXEL_CLIP_LEVEL: "3u" } },
  level_4: { defines: { SCENE_VOXEL_CLIP_LEVEL: "4u" } },
  level_5: { defines: { SCENE_VOXEL_CLIP_LEVEL: "5u" } },
  level_6: { defines: { SCENE_VOXEL_CLIP_LEVEL: "6u" } },
  level_7: { defines: { SCENE_VOXEL_CLIP_LEVEL: "7u" } },
};

function create_scene_voxel_level_shader_setup(path, clip_level) {
  return {
    pipeline_shaders: {
      compute: {
        path,
        defines: { SCENE_VOXEL_CLIP_LEVEL: `${clip_level}u` },
      },
    },
  };
}

const scene_voxel_finalize_dispatch_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "acceleration/scene_voxelizer_finalize_dispatch.wgsl",
    },
  },
};

const scene_voxel_build_brick_occupancy_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "acceleration/scene_voxelizer_build_brick_occupancy.wgsl",
    },
  },
};

const scene_voxel_build_upper_hierarchy_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "acceleration/scene_voxelizer_build_upper_hierarchy.wgsl",
    },
  },
};

const scene_voxel_debug_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "debug/debug_scene_voxelizer.wgsl",
    },
  },
};

// ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
// │                                  ROOT GRID CONFIGURATION                                     │
// └──────────────────────────────────────────────────────────────────────────────────────────────┘

/** Number of voxels along each axis of the initial dense root grid. */
export const SCENE_VOXEL_GRID_RESOLUTION = 256;

/** Number of child bricks along each axis when hierarchy subdivision is introduced. */
export const SCENE_VOXEL_BRICK_RESOLUTION = 8;

/** Total child bricks produced by one future brick subdivision (8³ = 512). */
export const SCENE_VOXEL_BRICK_CHILD_COUNT = SCENE_VOXEL_BRICK_RESOLUTION ** 3;

/** Total number of addressable voxels in the initial root grid (256³ = 16,777,216). */
export const SCENE_VOXEL_COUNT = SCENE_VOXEL_GRID_RESOLUTION ** 3;

/** Number of voxel or brick occupancy flags packed into one u32 word. */
export const SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD = 32;

/** Number of u32 words required by the packed root-grid occupancy bitset. */
export const SCENE_VOXEL_OCCUPANCY_WORD_COUNT = Math.ceil(
  SCENE_VOXEL_COUNT / SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD
);

/** GPU byte size of the packed root-grid occupancy bitset (2 MiB). */
export const SCENE_VOXEL_OCCUPANCY_BYTE_SIZE =
  SCENE_VOXEL_OCCUPANCY_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT;

/** Number of packed u32 words required for one future 8³ child-brick occupancy mask. */
export const SCENE_VOXEL_BRICK_OCCUPANCY_WORD_COUNT =
  SCENE_VOXEL_BRICK_CHILD_COUNT / SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD;

/** Number of HDDA levels including the original fine voxel grid. */
export const SCENE_VOXEL_HDDA_LEVEL_COUNT = 4;

/** Coarsest HDDA level index. Level zero is the original 256³ leaf grid. */
export const SCENE_VOXEL_HDDA_MAX_LEVEL = SCENE_VOXEL_HDDA_LEVEL_COUNT - 1;

/** Number of packed words used by the 32³, 4³, and 1³ coarse occupancy levels. */
export const SCENE_VOXEL_HIERARCHY_WORD_COUNT =
  Math.ceil(32 ** 3 / SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD) +
  Math.ceil(4 ** 3 / SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD) +
  1;

/** Default number of nested camera-centered voxel clip levels. */
export const SCENE_VOXEL_CLIPMAP_LEVEL_COUNT = 6;

/** Maximum number of clip levels represented by the shared shader parameter block. */
export const SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT = 8;

/** Voxel-size multiplier between adjacent clip levels. */
export const SCENE_VOXEL_CLIPMAP_SCALE = 2;

/** Total packed leaf occupancy for the default four-level configuration (8 MiB). */
export const SCENE_VOXEL_CLIPMAP_OCCUPANCY_WORD_COUNT =
  SCENE_VOXEL_OCCUPANCY_WORD_COUNT * SCENE_VOXEL_CLIPMAP_LEVEL_COUNT;

/** Total packed skip hierarchy for the default four-level configuration. */
export const SCENE_VOXEL_CLIPMAP_HIERARCHY_WORD_COUNT =
  SCENE_VOXEL_HIERARCHY_WORD_COUNT * SCENE_VOXEL_CLIPMAP_LEVEL_COUNT;

/**
 * Semantic names exposed to render-graph consumers.
 *
 * Keeping semantic names separate from physical resource names lets us change the representation
 * later without forcing every GI, lighting, or debug consumer to change at the same time.
 */
export const SceneVoxelizerResource = Object.freeze({
  VoxelGrid: "voxel_grid",
  VoxelizationParams: "voxelization_params",
  OccupancyHierarchy: "occupancy_hierarchy",
  DirtyBrickWords: "dirty_brick_words",
  DirtyBrickList: "dirty_brick_list",
  DirtyDispatchArgs: "dirty_dispatch_args",
  CompactedMeshlets: "compacted_meshlets",
  VoxelDispatchArgs: "voxel_dispatch_args",
  VoxelDispatchCount: "voxel_dispatch_count",
  ScratchResetTemplate: "scratch_reset_template",
  DebugOutput: "debug_output",
});

const DEFAULT_RESOURCE_PREFIX = "scene_voxelizer";
const VOXELIZATION_PARAMS_HEADER_WORD_COUNT = 4;
const VOXELIZATION_LEVEL_PARAMS_WORD_COUNT = 16;
const VOXELIZATION_PARAMS_WORD_COUNT =
  VOXELIZATION_PARAMS_HEADER_WORD_COUNT +
  VOXELIZATION_LEVEL_PARAMS_WORD_COUNT * SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT;
const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION = 65535;
const DIRTY_BRICK_SIZE = 8;
const DIRTY_BRICK_DIMENSION = SCENE_VOXEL_GRID_RESOLUTION / DIRTY_BRICK_SIZE;
const DIRTY_BRICK_COUNT = DIRTY_BRICK_DIMENSION ** 3;
const DIRTY_BRICK_WORD_COUNT = DIRTY_BRICK_COUNT / SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD;
const DIRTY_MARK_WORKGROUP_SIZE = 64;
const MESHLET_COMPACT_WORKGROUP_SIZE = 128;
const DEBUG_OUTPUT_SCALE = 0.5;
const DEBUG_WORKGROUP_SIZE = 8;
const HIERARCHY_BUILD_WORKGROUP_SIZE = 128;
const DIRTY_DISPATCH_RESET_WORD_COUNT = 7;
const VOXEL_DISPATCH_RESET_WORD_COUNT = 4;
const SCRATCH_RESET_TEMPLATE_DATA = new Uint32Array([
  0, 1, 1, 0, 0, 1, 1,
  0, 1, 1, 0,
]);

const required_voxelization_inputs = Object.freeze([
  "entity_transforms",
  "entity_flags",
  "object_instances",
  "meshlet_instances",
  "entity_index_lookup",
  "meshlet_buffer",
  "meshlet_vertex_buffer",
  "meshlet_triangle_buffer",
]);

/**
 * Creates a normalized scene-voxelizer configuration.
 *
 * `grid_origin` is the requested minimum corner of clip level zero. The voxelizer derives a shared
 * center from it, then brick-snaps every active clip level independently. For a world-space position
 * `p` and one resolved clip level, the coordinate is:
 *
 *   floor((p - grid_origin) / voxel_size)
 *
 * Coordinates outside [0, 256) are outside that clip volume.
 *
 * @param {Object} [overrides]
 * @param {number} [overrides.voxel_size=1.0] Uniform world-space edge length of one root voxel.
 * @param {number[]} [overrides.grid_origin=[0,0,0]] Minimum world-space corner of the root grid.
 * @param {number} [overrides.clipmap_level_count=4] Active nested clip levels, up to eight.
 * @param {number} [overrides.clipmap_scale=2] Voxel-size multiplier between adjacent levels.
 * @param {string} [overrides.resource_prefix="scene_voxelizer"] Physical GPU resource name prefix.
 * @returns {{voxel_size: number, grid_origin: number[], clipmap_level_count: number, clipmap_scale: number, resource_prefix: string}}
 */
export function create_scene_voxelizer_config(overrides = {}) {
  const config = {
    voxel_size: overrides.voxel_size ?? 0.25,
    grid_origin: [...(overrides.grid_origin ?? [0.0, 0.0, 0.0])],
    clipmap_level_count:
      overrides.clipmap_level_count ?? SCENE_VOXEL_CLIPMAP_LEVEL_COUNT,
    clipmap_scale: overrides.clipmap_scale ?? SCENE_VOXEL_CLIPMAP_SCALE,
    resource_prefix: overrides.resource_prefix ?? DEFAULT_RESOURCE_PREFIX,
  };

  validate_config(config);
  return config;
}

function validate_config(config) {
  if (!Number.isFinite(config.voxel_size) || config.voxel_size <= 0.0) {
    throw new Error("SceneVoxelizer voxel_size must be a finite number greater than zero");
  }

  if (
    !Array.isArray(config.grid_origin) ||
    config.grid_origin.length !== 3 ||
    config.grid_origin.some((component) => !Number.isFinite(component))
  ) {
    throw new Error("SceneVoxelizer grid_origin must contain three finite numbers");
  }

  if (typeof config.resource_prefix !== "string" || config.resource_prefix.length === 0) {
    throw new Error("SceneVoxelizer resource_prefix must be a non-empty string");
  }

  if (
    !Number.isSafeInteger(config.clipmap_level_count) ||
    config.clipmap_level_count < 1 ||
    config.clipmap_level_count > SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT
  ) {
    throw new Error(
      `SceneVoxelizer clipmap_level_count must be between 1 and ${SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT}`
    );
  }

  if (!Number.isFinite(config.clipmap_scale) || config.clipmap_scale <= 1.0) {
    throw new Error("SceneVoxelizer clipmap_scale must be a finite number greater than one");
  }
}

/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
 * ║                                  SCENE VOXELIZER                                             ║
 * ║                     Render-Graph-Driven GPU Voxel Grid Builder                               ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
 *
 * Owns the orchestration state for one scene voxel clipmap. The render graph owns the physical
 * resources and commands; this class owns their semantic meaning and the order in which passes are
 * declared.
 *
 * ┌─────────────────────────────── 📐 CURRENT SPATIAL MODEL ─────────────────────────────────────┐
 * │                                                                                              │
 * │  Clip dimensions:       configurable count × (256 × 256 × 256), up to eight                   │
 * │  Default extents:       64 m, 128 m, 256 m, 512 m at the 0.25 m base voxel size               │
 * │  HDDA levels per clip:  256³ leaves plus 32³, 4³, and 1³ skip levels                          │
 * │  Linear address:        x + 256 × (y + 256 × z)                                              │
 * │  Occupancy word:        linear_address >> 5                                                  │
 * │  Occupancy bit:         linear_address & 31                                                  │
 * │                                                                                              │
 * └──────────────────────────────────────────────────────────────────────────────────────────────┘
 *
 * ┌─────────────────────────────── 🧮 CURRENT STORAGE MODEL ─────────────────────────────────────┐
 * │                                                                                              │
 * │  voxel_grid packs one dense occupancy bitset per clip containing one bit per leaf voxel.      │
 * │  covers 32 consecutive linear voxels: bit 0 covers the first and bit 31 covers the last.      │
 * │  A clear bit means empty; a set bit means occupied. The initial allocation is exactly 2 MiB.  │
 * │                                                                                              │
 * │  Writers claim occupancy with atomicOr(word, 1u << bit), allowing overlapping primitives to   │
 * │  safely touch the same word. Each active clip consumes 2 MiB of persistent leaf data.         │
 * │                                                                                              │
 * └──────────────────────────────────────────────────────────────────────────────────────────────┘
 *
 * ┌─────────────────────────────── 🚦 PLANNED PASS PIPELINE ─────────────────────────────────────┐
 * │                                                                                              │
 * │  1. Reset        Clear occupancy, counters, and future allocators                            │
 * │  2. Voxelize     Project scene primitives into conservative root-grid coverage               │
 * │  3. Finalize     Build hierarchy metadata and prepare the consumer-facing representation      │
 * │                                                                                              │
 * │  Reset and Voxelize are implemented. Finalize remains an extension point for future sparse     │
 * │  hierarchy metadata; dense occupancy is consumer-readable immediately after Voxelize.          │
 * │                                                                                              │
 * └──────────────────────────────────────────────────────────────────────────────────────────────┘
 */
export class SceneVoxelizer {
  /**
   * @param {Object} [config] See {@link create_scene_voxelizer_config}.
   */
  constructor(config = {}) {
    this.config = create_scene_voxelizer_config(config);
    this.resources = new Map();
    this.frame_context = null;

    // Reused for every upload. Keeping this storage on the voxelizer avoids allocating parameter
    // arrays in the render loop while still allowing the volume to move or resize every frame.
    this.voxelization_params_data = new Uint32Array(VOXELIZATION_PARAMS_WORD_COUNT);
    this.voxelization_params_view = new DataView(this.voxelization_params_data.buffer);
    this.indirect_args_reset_data = new Uint32Array([0, 1, 1, 0]);
    this.dirty_dispatch_reset_data = new Uint32Array([0, 1, 1, 0, 0, 1, 1]);
    this.dispatch_count_reset_data = new Uint32Array(1);
    this.previous_clip_origins = Array.from(
      { length: SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT },
      () => new Float64Array([NaN, NaN, NaN])
    );
    this.previous_clip_voxel_sizes = new Float64Array(SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT);
    this.previous_clip_voxel_sizes.fill(NaN);
    this.previous_clip_storage_offsets = Array.from(
      { length: SCENE_VOXEL_CLIPMAP_MAX_LEVEL_COUNT },
      () => new Uint32Array(3)
    );
    this.previous_clipmap_level_count = 0;
    this.previous_meshlet_count = -1;
  }

  /**
   * Resolves configuration for a new graph build.
   *
   * Per-frame overrides are useful for moving the volume or changing its world-space scale without
   * reconstructing the strategy object. GPU resource handles are graph-local, so they are discarded
   * here and recreated by setup().
   *
   * Geometry inputs are existing render-graph handles. `meshlet_count` may also be supplied as
   * `meshlet_draw_count` to match RenderTaskQueue terminology.
   *
   * @param {Object} [frame_context]
   * @param {number} [frame_context.meshlet_count=0]
   * @param {number} [frame_context.entity_transforms]
   * @param {number} [frame_context.object_instances]
   * @param {number} [frame_context.meshlet_instances]
   * @param {number} [frame_context.entity_index_lookup]
   * @param {number} [frame_context.meshlet_buffer]
   * @param {number} [frame_context.meshlet_vertex_buffer]
   * @param {number} [frame_context.meshlet_triangle_buffer]
   * @returns {Object} The normalized context used by setup() and record().
   */
  begin_frame(frame_context = {}) {
    const voxel_size = frame_context.voxel_size ?? this.config.voxel_size;
    const clipmap_level_count =
      frame_context.clipmap_level_count ?? this.config.clipmap_level_count;
    const clipmap_scale = frame_context.clipmap_scale ?? this.config.clipmap_scale;
    const requested_grid_origin = [...(frame_context.grid_origin ?? this.config.grid_origin)];
    const meshlet_count = frame_context.meshlet_count ?? frame_context.meshlet_draw_count ?? 0;
    validate_config({
      voxel_size,
      grid_origin: requested_grid_origin,
      clipmap_level_count,
      clipmap_scale,
      resource_prefix: this.config.resource_prefix,
    });
    if (!Number.isSafeInteger(meshlet_count) || meshlet_count < 0) {
      throw new Error("SceneVoxelizer meshlet_count must be a non-negative safe integer");
    }

    const base_world_extent = SCENE_VOXEL_GRID_RESOLUTION * voxel_size;
    const clip_center = [
      requested_grid_origin[0] + base_world_extent * 0.5,
      requested_grid_origin[1] + base_world_extent * 0.5,
      requested_grid_origin[2] + base_world_extent * 0.5,
    ];
    const force_recreate =
      (frame_context.force_recreate ?? false) ||
      clipmap_level_count !== this.previous_clipmap_level_count;
    const clip_levels = new Array(clipmap_level_count);
    for (let clip_level = 0; clip_level < clipmap_level_count; clip_level++) {
      const level_voxel_size = voxel_size * clipmap_scale ** clip_level;
      const world_extent = SCENE_VOXEL_GRID_RESOLUTION * level_voxel_size;
      const scroll_quantum = SCENE_VOXEL_BRICK_RESOLUTION * level_voxel_size;
      const grid_origin = [
        Math.floor((clip_center[0] - world_extent * 0.5) / scroll_quantum) * scroll_quantum,
        Math.floor((clip_center[1] - world_extent * 0.5) / scroll_quantum) * scroll_quantum,
        Math.floor((clip_center[2] - world_extent * 0.5) / scroll_quantum) * scroll_quantum,
      ];
      const previous_origin = this.previous_clip_origins[clip_level];
      const previous_voxel_size = this.previous_clip_voxel_sizes[clip_level];
      const voxel_size_changed = level_voxel_size !== previous_voxel_size;
      const origin_changed =
        grid_origin[0] !== previous_origin[0] ||
        grid_origin[1] !== previous_origin[1] ||
        grid_origin[2] !== previous_origin[2];
      const scroll_delta_bricks = [0, 0, 0];
      let scroll_exceeds_volume = false;
      if (!voxel_size_changed && origin_changed) {
        for (let axis = 0; axis < 3; axis++) {
          scroll_delta_bricks[axis] = Math.round(
            (grid_origin[axis] - previous_origin[axis]) / scroll_quantum
          );
          scroll_exceeds_volume ||=
            Math.abs(scroll_delta_bricks[axis]) >= DIRTY_BRICK_DIMENSION;
        }
      }
      const full_rebuild =
        force_recreate ||
        voxel_size_changed ||
        scroll_exceeds_volume ||
        meshlet_count !== this.previous_meshlet_count;

      // A toroidal offset keeps the overlap at the same physical addresses. Camera scrolling then
      // clears and repopulates only exposed brick slabs instead of revoxelizing the entire level.
      const previous_storage_offset = this.previous_clip_storage_offsets[clip_level];
      let storage_offset_x = 0;
      let storage_offset_y = 0;
      let storage_offset_z = 0;
      if (full_rebuild) {
        previous_storage_offset.fill(0);
        scroll_delta_bricks.fill(0);
      } else {
        storage_offset_x =
          (previous_storage_offset[0] +
            scroll_delta_bricks[0] * SCENE_VOXEL_BRICK_RESOLUTION) &
          (SCENE_VOXEL_GRID_RESOLUTION - 1);
        storage_offset_y =
          (previous_storage_offset[1] +
            scroll_delta_bricks[1] * SCENE_VOXEL_BRICK_RESOLUTION) &
          (SCENE_VOXEL_GRID_RESOLUTION - 1);
        storage_offset_z =
          (previous_storage_offset[2] +
            scroll_delta_bricks[2] * SCENE_VOXEL_BRICK_RESOLUTION) &
          (SCENE_VOXEL_GRID_RESOLUTION - 1);
        previous_storage_offset[0] = storage_offset_x;
        previous_storage_offset[1] = storage_offset_y;
        previous_storage_offset[2] = storage_offset_z;
      }
      const retained_brick_count =
        (DIRTY_BRICK_DIMENSION - Math.abs(scroll_delta_bricks[0])) *
        (DIRTY_BRICK_DIMENSION - Math.abs(scroll_delta_bricks[1])) *
        (DIRTY_BRICK_DIMENSION - Math.abs(scroll_delta_bricks[2]));
      const scroll_dirty_brick_count = DIRTY_BRICK_COUNT - retained_brick_count;

      clip_levels[clip_level] = {
        clip_level,
        voxel_size: level_voxel_size,
        grid_origin,
        world_extent,
        scroll_quantum,
        full_rebuild,
        storage_offset_x,
        storage_offset_y,
        storage_offset_z,
        scroll_delta_bricks,
        scroll_dirty_brick_count,
      };
      this.previous_clip_voxel_sizes[clip_level] = level_voxel_size;
      previous_origin.set(grid_origin);
    }

    this.previous_clipmap_level_count = clipmap_level_count;
    this.previous_meshlet_count = meshlet_count;

    this.resources.clear();
    this.frame_context = {
      ...frame_context,
      voxel_size,
      clipmap_level_count,
      clipmap_scale,
      grid_origin: clip_levels[0].grid_origin,
      clip_center,
      clip_levels,
      meshlet_count,
      full_rebuild: clip_levels.some((level) => level.full_rebuild),
      force_recreate,
      resolution: SCENE_VOXEL_GRID_RESOLUTION,
      voxel_count: SCENE_VOXEL_COUNT * clipmap_level_count,
      occupancy_word_count: SCENE_VOXEL_OCCUPANCY_WORD_COUNT * clipmap_level_count,
      world_extent: clip_levels[0].world_extent,
    };

    return this.frame_context;
  }

  /**
   * Creates all render-graph resources for the current frame.
   *
   * This method is separate from record() so a renderer can establish resource handles before any
   * voxelizer or downstream consumer appends passes.
   *
   * @param {Object} render_graph Caller-owned render graph.
   * @param {Object} [frame_context] Context returned by begin_frame().
   * @returns {Object} Consumer-facing voxelizer outputs.
   */
  setup(render_graph, frame_context = this.frame_context) {
    this._assert_render_graph(render_graph);
    const context =
      frame_context && frame_context === this.frame_context
        ? frame_context
        : this.begin_frame(frame_context ?? {});

    this._setup_root_grid_resources(render_graph, context);
    this._setup_voxelization_resources(render_graph, context);
    this._setup_hierarchy_resources(render_graph, context);

    return this.get_outputs();
  }

  /**
   * Appends voxelizer work to a graph after setup() has established all handles.
   *
   * @param {Object} render_graph Caller-owned render graph.
   * @param {Object} [frame_context] Context returned by begin_frame().
   * @returns {Object} Consumer-facing voxelizer outputs.
   */
  record(render_graph, frame_context = this.frame_context) {
    this._assert_render_graph(render_graph);
    const context = frame_context ?? this.frame_context;
    if (!context || this.get_resource(SceneVoxelizerResource.VoxelGrid) === null) {
      throw new Error("SceneVoxelizer.record() requires begin_frame() and setup() first");
    }

    this._record_voxelization_passes(render_graph, context);
    this._record_finalize_passes(render_graph, context);

    return this.get_outputs();
  }

  /**
   * Convenience entry point for renderer strategies that do not need split setup/record phases.
   *
   * @param {Object} render_graph Caller-owned render graph.
   * @param {Object} [frame_context] Frame inputs and optional configuration overrides.
   * @returns {Object} Consumer-facing voxelizer outputs.
   */
  add_passes(render_graph, frame_context = {}) {
    const context = this.begin_frame(frame_context);
    this.setup(render_graph, context);
    return this.record(render_graph, context);
  }

  /**
   * Builds a camera-facing debug image by tracing the packed occupancy grid.
   *
   * The debug image is intentionally half resolution. Voxel traversal is substantially more
   * expensive than a texture overlay, and the existing debug overlay upscales it to the viewport.
   *
   * @param {Object} render_graph Caller-owned render graph.
   * @param {Object} debug_context
   * @param {number} debug_context.width Full viewport width.
   * @param {number} debug_context.height Full viewport height.
   * @param {number} debug_context.scene_color Lit scene-color graph handle.
   * @param {boolean} [debug_context.force_recreate=false]
   * @returns {number} Graph-local debug image handle.
   */
  add_debug_passes(render_graph, debug_context) {
    this._assert_render_graph(render_graph);
    if (typeof render_graph.create_image !== "function") {
      throw new Error("SceneVoxelizer debug output requires a render graph with create_image()");
    }
    if (
      !this.frame_context ||
      this.get_resource(SceneVoxelizerResource.VoxelGrid) === null ||
      this.get_resource(SceneVoxelizerResource.VoxelizationParams) === null ||
      this.get_resource(SceneVoxelizerResource.OccupancyHierarchy) === null
    ) {
      throw new Error("SceneVoxelizer.add_debug_passes() requires add_passes() first");
    }

    const width = debug_context?.width;
    const height = debug_context?.height;
    const scene_color = debug_context?.scene_color;
    if (
      !Number.isSafeInteger(width) ||
      width <= 0 ||
      !Number.isSafeInteger(height) ||
      height <= 0
    ) {
      throw new Error("SceneVoxelizer debug dimensions must be positive integers");
    }
    if (scene_color === null || scene_color === undefined) {
      throw new Error("SceneVoxelizer debug output requires a scene_color graph handle");
    }

    const debug_width = Math.max(1, Math.ceil(width * DEBUG_OUTPUT_SCALE));
    const debug_height = Math.max(1, Math.ceil(height * DEBUG_OUTPUT_SCALE));
    const debug_output = render_graph.create_image({
      name: `${this.config.resource_prefix}_debug_output`,
      format: "rgba16float",
      width: debug_width,
      height: debug_height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: debug_context.force_recreate ?? false,
    });
    this.resources.set(SceneVoxelizerResource.DebugOutput, debug_output);

    const voxelization_params = this.get_resource(SceneVoxelizerResource.VoxelizationParams);
    const voxel_grid = this.get_resource(SceneVoxelizerResource.VoxelGrid);
    const occupancy_hierarchy = this.get_resource(SceneVoxelizerResource.OccupancyHierarchy);
    render_graph.add_pass(
      `${this.config.resource_prefix}_debug_trace`,
      RenderPassFlags.Compute,
      {
        inputs: [
          voxelization_params,
          voxel_grid,
          occupancy_hierarchy,
          scene_color,
          debug_output,
        ],
        outputs: [debug_output],
        shader_setup: scene_voxel_debug_shader_setup,
      },
      (graph, frame_data, _encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(
          Math.ceil(debug_width / DEBUG_WORKGROUP_SIZE),
          Math.ceil(debug_height / DEBUG_WORKGROUP_SIZE),
          1
        );
      }
    );

    return debug_output;
  }

  /**
   * Returns a resource by stable semantic name.
   *
   * @param {string} semantic A value from {@link SceneVoxelizerResource}.
   * @returns {number|null} A graph-local resource handle.
   */
  get_resource(semantic) {
    return this.resources.get(semantic) ?? null;
  }

  /**
   * Returns the small, stable hand-off object intended for GI and lighting consumers.
   *
   * The metadata is included beside the graph handle so shaders and debug tools can agree on the
   * same world-to-grid transform without reaching into SceneVoxelizer internals.
   *
   * @returns {Object}
   */
  get_outputs() {
    const context = this.frame_context;
    return {
      voxel_grid: this.get_resource(SceneVoxelizerResource.VoxelGrid),
      voxelization_params: this.get_resource(SceneVoxelizerResource.VoxelizationParams),
      occupancy_hierarchy: this.get_resource(SceneVoxelizerResource.OccupancyHierarchy),
      debug_output: this.get_resource(SceneVoxelizerResource.DebugOutput),
      resolution: SCENE_VOXEL_GRID_RESOLUTION,
      voxel_count:
        SCENE_VOXEL_COUNT *
        (context?.clipmap_level_count ?? this.config.clipmap_level_count),
      occupancy_bits_per_word: SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD,
      occupancy_word_count:
        SCENE_VOXEL_OCCUPANCY_WORD_COUNT *
        (context?.clipmap_level_count ?? this.config.clipmap_level_count),
      voxel_size: context?.voxel_size ?? this.config.voxel_size,
      grid_origin: [...(context?.grid_origin ?? this.config.grid_origin)],
      world_extent: context?.world_extent ?? SCENE_VOXEL_GRID_RESOLUTION * this.config.voxel_size,
      subdivision_levels: 0,
      clipmap_level_count:
        context?.clipmap_level_count ?? this.config.clipmap_level_count,
      clipmap_scale: context?.clipmap_scale ?? this.config.clipmap_scale,
      clip_levels: context?.clip_levels?.map((level) => ({
        clip_level: level.clip_level,
        voxel_size: level.voxel_size,
        grid_origin: [...level.grid_origin],
        world_extent: level.world_extent,
        scroll_quantum: level.scroll_quantum,
        storage_offset: [
          level.storage_offset_x,
          level.storage_offset_y,
          level.storage_offset_z,
        ],
      })) ?? [],
      hdda_level_count: SCENE_VOXEL_HDDA_LEVEL_COUNT,
      hdda_max_level: SCENE_VOXEL_HDDA_MAX_LEVEL,
    };
  }

  // ───────────────────────────────── RESOURCE SETUP ─────────────────────────────────────────────

  _setup_root_grid_resources(render_graph, context) {
    const voxel_grid = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_root_grid`,
      size: SCENE_VOXEL_OCCUPANCY_WORD_COUNT * context.clipmap_level_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      force: context.force_recreate,
    });

    this.resources.set(SceneVoxelizerResource.VoxelGrid, voxel_grid);
  }

  _setup_voxelization_resources(render_graph, context) {
    const voxelization_params = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_voxelization_params`,
      raw_data: this.voxelization_params_data,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });

    this.resources.set(SceneVoxelizerResource.VoxelizationParams, voxelization_params);

    const dirty_brick_words = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_dirty_brick_words`,
      size: DIRTY_BRICK_WORD_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    const dirty_brick_list = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_dirty_brick_list`,
      size: DIRTY_BRICK_COUNT,
      usage: GPUBufferUsage.STORAGE,
      force: context.force_recreate,
    });
    const dirty_dispatch_args = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_dirty_dispatch_args`,
      raw_data: this.dirty_dispatch_reset_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
      force: context.force_recreate,
    });
    const compacted_meshlets = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_compacted_meshlets`,
      size: Math.max(2, context.meshlet_count * 2),
      usage: GPUBufferUsage.STORAGE,
      force: context.force_recreate,
    });
    const voxel_dispatch_args = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_voxel_dispatch_args`,
      raw_data: this.indirect_args_reset_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
      force: context.force_recreate,
    });
    const voxel_dispatch_count = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_voxel_dispatch_count`,
      raw_data: this.dispatch_count_reset_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });
    const scratch_reset_template = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_scratch_reset_template`,
      raw_data: SCRATCH_RESET_TEMPLATE_DATA,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      force: context.force_recreate,
    });

    this.resources.set(SceneVoxelizerResource.DirtyBrickWords, dirty_brick_words);
    this.resources.set(SceneVoxelizerResource.DirtyBrickList, dirty_brick_list);
    this.resources.set(SceneVoxelizerResource.DirtyDispatchArgs, dirty_dispatch_args);
    this.resources.set(SceneVoxelizerResource.CompactedMeshlets, compacted_meshlets);
    this.resources.set(SceneVoxelizerResource.VoxelDispatchArgs, voxel_dispatch_args);
    this.resources.set(SceneVoxelizerResource.VoxelDispatchCount, voxel_dispatch_count);
    this.resources.set(SceneVoxelizerResource.ScratchResetTemplate, scratch_reset_template);
  }

  _setup_hierarchy_resources(render_graph, context) {
    const occupancy_hierarchy = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_occupancy_hierarchy`,
      size: SCENE_VOXEL_HIERARCHY_WORD_COUNT * context.clipmap_level_count,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      force: context.force_recreate,
    });

    this.resources.set(SceneVoxelizerResource.OccupancyHierarchy, occupancy_hierarchy);
  }

  // ────────────────────────────────── PASS RECORDING ─────────────────────────────────────────────

  _record_reset_passes(render_graph, clip_level) {
    const scratch_reset_template = this.get_resource(
      SceneVoxelizerResource.ScratchResetTemplate
    );
    const dirty_brick_words = this.get_resource(SceneVoxelizerResource.DirtyBrickWords);
    const dirty_dispatch_args = this.get_resource(SceneVoxelizerResource.DirtyDispatchArgs);
    const voxel_dispatch_args = this.get_resource(SceneVoxelizerResource.VoxelDispatchArgs);
    const voxel_dispatch_count = this.get_resource(SceneVoxelizerResource.VoxelDispatchCount);

    render_graph.add_pass(
      `${this.config.resource_prefix}_clip_${clip_level}_reset`,
      RenderPassFlags.GraphLocal,
      {
        inputs: [scratch_reset_template],
        outputs: [
          dirty_brick_words,
          dirty_dispatch_args,
          voxel_dispatch_args,
          voxel_dispatch_count,
        ],
      },
      (graph, _frame_data, encoder) => {
        // These resets must be encoded between clip-level dispatches. queue.writeBuffer() calls
        // made while recording would all execute before command-buffer submission, allowing one
        // level's indirect counters to leak into the next level and intermittently erase clips.
        const physical_reset_template = graph.get_physical_buffer(scratch_reset_template);
        const physical_dirty_words = graph.get_physical_buffer(dirty_brick_words);
        encoder.clearBuffer(
          physical_dirty_words.buffer,
          0,
          physical_dirty_words.config.size
        );
        encoder.copyBufferToBuffer(
          physical_reset_template.buffer,
          0,
          graph.get_physical_buffer(dirty_dispatch_args).buffer,
          0,
          DIRTY_DISPATCH_RESET_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT
        );
        encoder.copyBufferToBuffer(
          physical_reset_template.buffer,
          DIRTY_DISPATCH_RESET_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT,
          graph.get_physical_buffer(voxel_dispatch_args).buffer,
          0,
          VOXEL_DISPATCH_RESET_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT
        );
        const physical_dispatch_count = graph.get_physical_buffer(voxel_dispatch_count);
        encoder.clearBuffer(
          physical_dispatch_count.buffer,
          0,
          physical_dispatch_count.config.size
        );
      }
    );
  }

  _record_voxelization_passes(render_graph, context) {
    const compact_workgroup_count = Math.ceil(
      context.meshlet_count / MESHLET_COMPACT_WORKGROUP_SIZE
    );
    const max_workgroups =
      context.max_compute_workgroups_per_dimension ?? MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
    if (compact_workgroup_count > max_workgroups) {
      throw new Error(
        `SceneVoxelizer cannot process ${context.meshlet_count} meshlets within WebGPU workgroup limits`
      );
    }

    this._write_voxelization_params(context, compact_workgroup_count);
    this._validate_voxelization_inputs(context);

    const voxelization_params = this.get_resource(SceneVoxelizerResource.VoxelizationParams);
    render_graph.add_pass(
      `${this.config.resource_prefix}_upload_params`,
      RenderPassFlags.GraphLocal,
      {
        outputs: [voxelization_params],
      },
      (graph, _frame_data, _encoder) => {
        graph
          .get_physical_buffer(voxelization_params)
          .write_raw(this.voxelization_params_data);
      }
    );

    for (const clip_context of context.clip_levels) {
      const mark_item_count = clip_context.full_rebuild
        ? DIRTY_BRICK_COUNT
        : clip_context.scroll_dirty_brick_count + context.meshlet_count;
      const mark_workgroup_count = Math.ceil(mark_item_count / DIRTY_MARK_WORKGROUP_SIZE);
      if (mark_workgroup_count > max_workgroups) {
        throw new Error(
          `SceneVoxelizer clip level ${clip_context.clip_level} exceeds WebGPU workgroup limits`
        );
      }

      this._record_reset_passes(render_graph, clip_context.clip_level);
      this._record_clip_level_voxelization_passes(
        render_graph,
        context,
        clip_context,
        mark_workgroup_count
      );
    }
  }

  _record_clip_level_voxelization_passes(
    render_graph,
    context,
    clip_context,
    mark_workgroup_count
  ) {
    const clip_level = clip_context.clip_level;
    const voxelization_params = this.get_resource(SceneVoxelizerResource.VoxelizationParams);
    const voxel_grid = this.get_resource(SceneVoxelizerResource.VoxelGrid);
    const dirty_brick_words = this.get_resource(SceneVoxelizerResource.DirtyBrickWords);
    const dirty_brick_list = this.get_resource(SceneVoxelizerResource.DirtyBrickList);
    const dirty_dispatch_args = this.get_resource(SceneVoxelizerResource.DirtyDispatchArgs);
    const compacted_meshlets = this.get_resource(SceneVoxelizerResource.CompactedMeshlets);
    const voxel_dispatch_args = this.get_resource(SceneVoxelizerResource.VoxelDispatchArgs);
    const voxel_dispatch_count = this.get_resource(SceneVoxelizerResource.VoxelDispatchCount);

    render_graph.add_pass(
      `${this.config.resource_prefix}_clip_${clip_level}_mark_dirty_bricks`,
      RenderPassFlags.Compute,
      {
        inputs: [
          context.entity_transforms,
          context.entity_flags,
          context.object_instances,
          context.meshlet_instances,
          context.entity_index_lookup,
          context.meshlet_buffer,
          voxelization_params,
          dirty_brick_words,
          dirty_brick_list,
          dirty_dispatch_args,
        ],
        outputs: [dirty_brick_words, dirty_brick_list, dirty_dispatch_args],
        shader_setup: create_scene_voxel_level_shader_setup(
          "acceleration/scene_voxelizer_mark_dirty.wgsl",
          clip_level
        ),
      },
      (graph, frame_data, _encoder) => {
        if (mark_workgroup_count !== 0) {
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(mark_workgroup_count, 1, 1);
        }
      }
    );

    render_graph.add_pass(
      `${this.config.resource_prefix}_clip_${clip_level}_clear_dirty_bricks`,
      RenderPassFlags.Compute,
      {
        inputs: [dirty_brick_list, voxel_grid, voxelization_params, dirty_dispatch_args],
        outputs: [voxel_grid],
        shader_setup: create_scene_voxel_level_shader_setup(
          "acceleration/scene_voxelizer_clear_dirty.wgsl",
          clip_level
        ),
      },
      (graph, frame_data, _encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch_indirect(graph.get_physical_buffer(dirty_dispatch_args));
      }
    );

    render_graph.add_pass(
      `${this.config.resource_prefix}_clip_${clip_level}_compact_dirty_meshlets`,
      RenderPassFlags.Compute,
      {
        inputs: [
          context.entity_transforms,
          context.object_instances,
          context.meshlet_instances,
          context.entity_index_lookup,
          context.meshlet_buffer,
          voxelization_params,
          dirty_brick_words,
          compacted_meshlets,
          voxel_dispatch_count,
          dirty_dispatch_args,
        ],
        outputs: [compacted_meshlets, voxel_dispatch_count],
        shader_setup: create_scene_voxel_level_shader_setup(
          "acceleration/scene_voxelizer_compact.wgsl",
          clip_level
        ),
      },
      (graph, frame_data, _encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch_indirect(
          graph.get_physical_buffer(dirty_dispatch_args),
          4 * Uint32Array.BYTES_PER_ELEMENT
        );
      }
    );

    render_graph.add_pass(
      `${this.config.resource_prefix}_clip_${clip_level}_finalize_dispatch`,
      RenderPassFlags.Compute,
      {
        inputs: [voxel_dispatch_count, voxel_dispatch_args],
        outputs: [voxel_dispatch_count, voxel_dispatch_args],
        shader_setup: scene_voxel_finalize_dispatch_shader_setup,
      },
      (graph, frame_data, _encoder) => {
        graph.get_physical_pass(frame_data.current_pass).dispatch(1, 1, 1);
      }
    );

    render_graph.add_pass(
      `${this.config.resource_prefix}_clip_${clip_level}_voxelize_compacted_meshlets`,
      RenderPassFlags.Compute,
      {
        inputs: [
          context.entity_transforms,
          context.object_instances,
          compacted_meshlets,
          context.entity_index_lookup,
          context.meshlet_buffer,
          context.meshlet_vertex_buffer,
          context.meshlet_triangle_buffer,
          voxelization_params,
          voxel_grid,
          voxel_dispatch_count,
          voxel_dispatch_args,
        ],
        outputs: [voxel_grid],
        shader_setup: create_scene_voxel_level_shader_setup(
          "acceleration/scene_voxelizer.wgsl",
          clip_level
        ),
        b_force_keep_pass: true,
      },
      (graph, frame_data, _encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch_indirect(graph.get_physical_buffer(voxel_dispatch_args));
      }
    );
  }

  _record_finalize_passes(render_graph, context) {
    const voxel_grid = this.get_resource(SceneVoxelizerResource.VoxelGrid);
    const voxelization_params = this.get_resource(SceneVoxelizerResource.VoxelizationParams);
    const occupancy_hierarchy = this.get_resource(SceneVoxelizerResource.OccupancyHierarchy);

    render_graph.add_pass(
      `${this.config.resource_prefix}_clear_occupancy_hierarchy`,
      RenderPassFlags.GraphLocal,
      {
        outputs: [occupancy_hierarchy],
      },
      (graph, _frame_data, encoder) => {
        const physical_hierarchy = graph.get_physical_buffer(occupancy_hierarchy);
        encoder.clearBuffer(
          physical_hierarchy.buffer,
          0,
          physical_hierarchy.config.size
        );
      }
    );

    render_graph.add_pass(
      `${this.config.resource_prefix}_build_brick_occupancy`,
      RenderPassFlags.Compute,
      {
        inputs: [voxel_grid, occupancy_hierarchy, voxelization_params],
        outputs: [occupancy_hierarchy],
        shader_setup: scene_voxel_build_brick_occupancy_shader_setup,
        b_force_keep_pass: true,
      },
      (graph, frame_data) => {
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(DIRTY_BRICK_COUNT / HIERARCHY_BUILD_WORKGROUP_SIZE),
            1,
            context.clipmap_level_count
          );
      }
    );

    render_graph.add_pass(
      `${this.config.resource_prefix}_build_upper_occupancy_hierarchy`,
      RenderPassFlags.Compute,
      {
        inputs: [occupancy_hierarchy],
        outputs: [occupancy_hierarchy],
        shader_setup: scene_voxel_build_upper_hierarchy_shader_setup,
        b_force_keep_pass: true,
      },
      (graph, frame_data) => {
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(1, 1, context.clipmap_level_count);
      }
    );
  }

  _validate_voxelization_inputs(context) {
    let missing_inputs = "";
    for (const input_name of required_voxelization_inputs) {
      if (context[input_name] === null || context[input_name] === undefined) {
        missing_inputs += `${missing_inputs.length > 0 ? ", " : ""}${input_name}`;
      }
    }

    if (missing_inputs.length !== 0) {
      throw new Error(`SceneVoxelizer requires frame_context inputs: ${missing_inputs}`);
    }
  }

  _write_voxelization_params(context, compact_workgroup_count) {
    const view = this.voxelization_params_view;
    view.setUint32(0, context.clipmap_level_count, true);
    view.setUint32(4, 0, true);
    view.setUint32(8, 0, true);
    view.setUint32(12, 0, true);
    for (const clip_context of context.clip_levels) {
      const word_offset =
        VOXELIZATION_PARAMS_HEADER_WORD_COUNT +
        clip_context.clip_level * VOXELIZATION_LEVEL_PARAMS_WORD_COUNT;
      const byte_offset = word_offset * Uint32Array.BYTES_PER_ELEMENT;
      view.setFloat32(byte_offset, clip_context.grid_origin[0], true);
      view.setFloat32(byte_offset + 4, clip_context.grid_origin[1], true);
      view.setFloat32(byte_offset + 8, clip_context.grid_origin[2], true);
      view.setFloat32(byte_offset + 12, clip_context.voxel_size, true);
      view.setUint32(byte_offset + 16, context.meshlet_count, true);
      view.setUint32(byte_offset + 20, compact_workgroup_count, true);
      view.setUint32(byte_offset + 24, SCENE_VOXEL_GRID_RESOLUTION, true);
      view.setUint32(byte_offset + 28, clip_context.full_rebuild ? 1 : 0, true);
      view.setUint32(byte_offset + 32, clip_context.storage_offset_x, true);
      view.setUint32(byte_offset + 36, clip_context.storage_offset_y, true);
      view.setUint32(byte_offset + 40, clip_context.storage_offset_z, true);
      view.setUint32(byte_offset + 44, clip_context.scroll_dirty_brick_count, true);
      view.setInt32(byte_offset + 48, clip_context.scroll_delta_bricks[0], true);
      view.setInt32(byte_offset + 52, clip_context.scroll_delta_bricks[1], true);
      view.setInt32(byte_offset + 56, clip_context.scroll_delta_bricks[2], true);
      view.setUint32(byte_offset + 60, 0, true);
    }
  }

  _assert_render_graph(render_graph) {
    if (
      !render_graph ||
      typeof render_graph.create_buffer !== "function" ||
      typeof render_graph.add_pass !== "function"
    ) {
      throw new Error("SceneVoxelizer requires a render graph with create_buffer() and add_pass()");
    }
  }
}

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
//     • One fixed 256 × 256 × 256 root grid
//     • One uniform world-space voxel size
//     • One occupancy bit per voxel, packed 32 voxels per u32 storage word
//     • Meshlet-driven conservative compute voxelization
//     • A full GPU rebuild from current instance transforms for dynamic-scene correctness
//
// 🌳  PLANNED EVOLUTION:
//     • Treat each root voxel as a brick when finer detail is requested
//     • Subdivide a brick into an 8 × 8 × 8 child-brick lattice
//     • Store each 512-child occupancy mask in 16 packed u32 words
//     • Repeat subdivision to a fixed hierarchy depth
//     • Preserve the same one-bit occupancy contract at every hierarchy level
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

const scene_voxelize_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "acceleration/scene_voxelizer.wgsl",
    },
  },
};

const scene_voxel_mark_dirty_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "acceleration/scene_voxelizer_mark_dirty.wgsl",
    },
  },
};

const scene_voxel_clear_dirty_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "acceleration/scene_voxelizer_clear_dirty.wgsl",
    },
  },
};

const scene_voxel_compact_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "acceleration/scene_voxelizer_compact.wgsl",
    },
  },
};

const scene_voxel_finalize_dispatch_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "acceleration/scene_voxelizer_finalize_dispatch.wgsl",
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

/**
 * Semantic names exposed to render-graph consumers.
 *
 * Keeping semantic names separate from physical resource names lets us change the representation
 * later without forcing every GI, lighting, or debug consumer to change at the same time.
 */
export const SceneVoxelizerResource = Object.freeze({
  VoxelGrid: "voxel_grid",
  VoxelizationParams: "voxelization_params",
  DirtyBrickWords: "dirty_brick_words",
  DirtyBrickList: "dirty_brick_list",
  DirtyDispatchArgs: "dirty_dispatch_args",
  CompactedMeshlets: "compacted_meshlets",
  VoxelDispatchArgs: "voxel_dispatch_args",
  VoxelDispatchCount: "voxel_dispatch_count",
  DebugOutput: "debug_output",
});

const DEFAULT_RESOURCE_PREFIX = "scene_voxelizer";
const VOXELIZATION_PARAMS_WORD_COUNT = 8;
const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION = 65535;
const DIRTY_BRICK_SIZE = 8;
const DIRTY_BRICK_DIMENSION = SCENE_VOXEL_GRID_RESOLUTION / DIRTY_BRICK_SIZE;
const DIRTY_BRICK_COUNT = DIRTY_BRICK_DIMENSION ** 3;
const DIRTY_BRICK_WORD_COUNT = DIRTY_BRICK_COUNT / SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD;
const DIRTY_MARK_WORKGROUP_SIZE = 64;
const MESHLET_COMPACT_WORKGROUP_SIZE = 128;
const DEBUG_OUTPUT_SCALE = 0.5;
const DEBUG_WORKGROUP_SIZE = 8;

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
 * `grid_origin` is the minimum world-space corner of the grid. For a world-space position `p`,
 * the root coordinate is:
 *
 *   floor((p - grid_origin) / voxel_size)
 *
 * Coordinates outside [0, 256) are outside the current volume.
 *
 * @param {Object} [overrides]
 * @param {number} [overrides.voxel_size=1.0] Uniform world-space edge length of one root voxel.
 * @param {number[]} [overrides.grid_origin=[0,0,0]] Minimum world-space corner of the root grid.
 * @param {string} [overrides.resource_prefix="scene_voxelizer"] Physical GPU resource name prefix.
 * @returns {{voxel_size: number, grid_origin: number[], resource_prefix: string}}
 */
export function create_scene_voxelizer_config(overrides = {}) {
  const config = {
    voxel_size: overrides.voxel_size ?? 0.25,
    grid_origin: [...(overrides.grid_origin ?? [0.0, 0.0, 0.0])],
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
}

/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
 * ║                                  SCENE VOXELIZER                                             ║
 * ║                     Render-Graph-Driven GPU Voxel Grid Builder                               ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
 *
 * Owns the orchestration state for one scene voxel volume. The render graph owns the physical
 * resources and commands; this class owns their semantic meaning and the order in which passes are
 * declared.
 *
 * ┌─────────────────────────────── 📐 CURRENT SPATIAL MODEL ─────────────────────────────────────┐
 * │                                                                                              │
 * │  Grid dimensions:       256 × 256 × 256                                                      │
 * │  World-space bounds:    [grid_origin, grid_origin + 256 × voxel_size)                         │
 * │  Subdivision levels:    0 (level 0 is currently both the root and the leaf)                   │
 * │  Linear address:        x + 256 × (y + 256 × z)                                              │
 * │  Occupancy word:        linear_address >> 5                                                  │
 * │  Occupancy bit:         linear_address & 31                                                  │
 * │                                                                                              │
 * └──────────────────────────────────────────────────────────────────────────────────────────────┘
 *
 * ┌─────────────────────────────── 🧮 CURRENT STORAGE MODEL ─────────────────────────────────────┐
 * │                                                                                              │
 * │  voxel_grid is a dense occupancy bitset containing one bit per root voxel. Each u32 word      │
 * │  covers 32 consecutive linear voxels: bit 0 covers the first and bit 31 covers the last.      │
 * │  A clear bit means empty; a set bit means occupied. The initial allocation is exactly 2 MiB.  │
 * │                                                                                              │
 * │  Writers claim occupancy with atomicOr(word, 1u << bit), allowing overlapping primitives to   │
 * │  safely touch the same word. Future brick masks use the identical convention: an 8³ child    │
 * │  lattice has 512 occupancy bits and therefore occupies exactly 16 consecutive u32 words.      │
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
    this.previous_grid_origin = new Float64Array([NaN, NaN, NaN]);
    this.previous_voxel_size = NaN;
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
    const grid_origin = [...(frame_context.grid_origin ?? this.config.grid_origin)];
    const meshlet_count = frame_context.meshlet_count ?? frame_context.meshlet_draw_count ?? 0;
    validate_config({
      voxel_size,
      grid_origin,
      resource_prefix: this.config.resource_prefix,
    });
    if (!Number.isSafeInteger(meshlet_count) || meshlet_count < 0) {
      throw new Error("SceneVoxelizer meshlet_count must be a non-negative safe integer");
    }

    const volume_changed =
      voxel_size !== this.previous_voxel_size ||
      grid_origin[0] !== this.previous_grid_origin[0] ||
      grid_origin[1] !== this.previous_grid_origin[1] ||
      grid_origin[2] !== this.previous_grid_origin[2];
    const full_rebuild =
      (frame_context.full_rebuild ?? false) ||
      (frame_context.force_recreate ?? false) ||
      volume_changed ||
      meshlet_count !== this.previous_meshlet_count;

    this.previous_voxel_size = voxel_size;
    this.previous_grid_origin.set(grid_origin);
    this.previous_meshlet_count = meshlet_count;

    this.resources.clear();
    this.frame_context = {
      ...frame_context,
      voxel_size,
      grid_origin,
      meshlet_count,
      full_rebuild,
      force_recreate: frame_context.force_recreate ?? false,
      resolution: SCENE_VOXEL_GRID_RESOLUTION,
      voxel_count: SCENE_VOXEL_COUNT,
      occupancy_word_count: SCENE_VOXEL_OCCUPANCY_WORD_COUNT,
      world_extent: SCENE_VOXEL_GRID_RESOLUTION * voxel_size,
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

    this._record_reset_passes(render_graph, context);
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
      this.get_resource(SceneVoxelizerResource.VoxelizationParams) === null
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
    render_graph.add_pass(
      `${this.config.resource_prefix}_debug_trace`,
      RenderPassFlags.Compute,
      {
        inputs: [voxelization_params, voxel_grid, scene_color, debug_output],
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
      debug_output: this.get_resource(SceneVoxelizerResource.DebugOutput),
      resolution: SCENE_VOXEL_GRID_RESOLUTION,
      voxel_count: SCENE_VOXEL_COUNT,
      occupancy_bits_per_word: SCENE_VOXEL_OCCUPANCY_BITS_PER_WORD,
      occupancy_word_count: SCENE_VOXEL_OCCUPANCY_WORD_COUNT,
      voxel_size: context?.voxel_size ?? this.config.voxel_size,
      grid_origin: [...(context?.grid_origin ?? this.config.grid_origin)],
      world_extent: context?.world_extent ?? SCENE_VOXEL_GRID_RESOLUTION * this.config.voxel_size,
      subdivision_levels: 0,
    };
  }

  // ───────────────────────────────── RESOURCE SETUP ─────────────────────────────────────────────

  _setup_root_grid_resources(render_graph, context) {
    const voxel_grid = render_graph.create_buffer({
      name: `${this.config.resource_prefix}_root_grid`,
      size: SCENE_VOXEL_OCCUPANCY_WORD_COUNT,
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

    this.resources.set(SceneVoxelizerResource.DirtyBrickWords, dirty_brick_words);
    this.resources.set(SceneVoxelizerResource.DirtyBrickList, dirty_brick_list);
    this.resources.set(SceneVoxelizerResource.DirtyDispatchArgs, dirty_dispatch_args);
    this.resources.set(SceneVoxelizerResource.CompactedMeshlets, compacted_meshlets);
    this.resources.set(SceneVoxelizerResource.VoxelDispatchArgs, voxel_dispatch_args);
    this.resources.set(SceneVoxelizerResource.VoxelDispatchCount, voxel_dispatch_count);
  }

  _setup_hierarchy_resources(_render_graph, _context) {
    // Reserved for brick allocators, indirection tables, and packed child/leaf occupancy bitsets.
    // The root grid remains the stable entry point when these resources are introduced.
  }

  // ────────────────────────────────── PASS RECORDING ─────────────────────────────────────────────

  _record_reset_passes(render_graph, _context) {
    const dirty_brick_words = this.get_resource(SceneVoxelizerResource.DirtyBrickWords);
    const dirty_dispatch_args = this.get_resource(SceneVoxelizerResource.DirtyDispatchArgs);
    const voxel_dispatch_args = this.get_resource(SceneVoxelizerResource.VoxelDispatchArgs);
    const voxel_dispatch_count = this.get_resource(SceneVoxelizerResource.VoxelDispatchCount);

    render_graph.add_pass(
      `${this.config.resource_prefix}_reset`,
      RenderPassFlags.GraphLocal,
      {
        inputs: [],
        outputs: [
          dirty_brick_words,
          dirty_dispatch_args,
          voxel_dispatch_args,
          voxel_dispatch_count,
        ],
      },
      (graph, _frame_data, encoder) => {
        const physical_dirty_words = graph.get_physical_buffer(dirty_brick_words);
        encoder.clearBuffer(
          physical_dirty_words.buffer,
          0,
          physical_dirty_words.config.size
        );
        graph
          .get_physical_buffer(dirty_dispatch_args)
          .write_raw(this.dirty_dispatch_reset_data);
        graph
          .get_physical_buffer(voxel_dispatch_args)
          .write_raw(this.indirect_args_reset_data);
        graph
          .get_physical_buffer(voxel_dispatch_count)
          .write_raw(this.dispatch_count_reset_data);
      }
    );
  }

  _record_voxelization_passes(render_graph, context) {
    const mark_item_count = context.full_rebuild ? DIRTY_BRICK_COUNT : context.meshlet_count;
    const mark_workgroup_count = Math.ceil(mark_item_count / DIRTY_MARK_WORKGROUP_SIZE);
    const compact_workgroup_count = Math.ceil(
      context.meshlet_count / MESHLET_COMPACT_WORKGROUP_SIZE
    );
    const max_workgroups =
      context.max_compute_workgroups_per_dimension ?? MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
    if (mark_workgroup_count > max_workgroups || compact_workgroup_count > max_workgroups) {
      throw new Error(
        `SceneVoxelizer cannot process ${context.meshlet_count} meshlets within WebGPU workgroup limits`
      );
    }

    this._write_voxelization_params(context, compact_workgroup_count);
    this._validate_voxelization_inputs(context);

    const voxelization_params = this.get_resource(SceneVoxelizerResource.VoxelizationParams);
    const voxel_grid = this.get_resource(SceneVoxelizerResource.VoxelGrid);
    const dirty_brick_words = this.get_resource(SceneVoxelizerResource.DirtyBrickWords);
    const dirty_brick_list = this.get_resource(SceneVoxelizerResource.DirtyBrickList);
    const dirty_dispatch_args = this.get_resource(SceneVoxelizerResource.DirtyDispatchArgs);
    const compacted_meshlets = this.get_resource(SceneVoxelizerResource.CompactedMeshlets);
    const voxel_dispatch_args = this.get_resource(SceneVoxelizerResource.VoxelDispatchArgs);
    const voxel_dispatch_count = this.get_resource(SceneVoxelizerResource.VoxelDispatchCount);

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

    render_graph.add_pass(
      `${this.config.resource_prefix}_mark_dirty_bricks`,
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
        shader_setup: scene_voxel_mark_dirty_shader_setup,
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
      `${this.config.resource_prefix}_clear_dirty_bricks`,
      RenderPassFlags.Compute,
      {
        inputs: [dirty_brick_list, voxel_grid],
        outputs: [voxel_grid],
        shader_setup: scene_voxel_clear_dirty_shader_setup,
      },
      (graph, frame_data, _encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch_indirect(graph.get_physical_buffer(dirty_dispatch_args));
      }
    );

    render_graph.add_pass(
      `${this.config.resource_prefix}_compact_dirty_meshlets`,
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
        ],
        outputs: [compacted_meshlets, voxel_dispatch_count],
        shader_setup: scene_voxel_compact_shader_setup,
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
      `${this.config.resource_prefix}_finalize_dispatch`,
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
      `${this.config.resource_prefix}_voxelize_compacted_meshlets`,
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
        ],
        outputs: [voxel_grid],
        shader_setup: scene_voxelize_shader_setup,
        b_force_keep_pass: true,
      },
      (graph, frame_data, _encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch_indirect(graph.get_physical_buffer(voxel_dispatch_args));
      }
    );
  }

  _record_finalize_passes(_render_graph, _context) {
    // TODO: Publish any counters or hierarchy metadata required by downstream consumers. The
    // packed root occupancy grid is already consumer-readable after voxelization.
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
    view.setFloat32(0, context.grid_origin[0], true);
    view.setFloat32(4, context.grid_origin[1], true);
    view.setFloat32(8, context.grid_origin[2], true);
    view.setFloat32(12, context.voxel_size, true);
    view.setUint32(16, context.meshlet_count, true);
    view.setUint32(20, compact_workgroup_count, true);
    view.setUint32(24, SCENE_VOXEL_GRID_RESOLUTION, true);
    view.setUint32(28, context.full_rebuild ? 1 : 0, true);
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

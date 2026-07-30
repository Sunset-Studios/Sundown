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
// 🧱  CURRENT MILESTONE:
//     • One fixed 256 × 256 × 256 root grid
//     • One uniform world-space voxel size
//     • One occupancy bit per voxel, packed 32 voxels per u32 storage word
//     • A GPU clear pass that produces a valid, empty grid
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
// IMPORTANT: The bit-wise occupancy layout is part of the contract. Geometry coverage and any
// non-occupancy surface attributes are intentionally not chosen yet, so this boilerplate does not
// commit us to conservative rasterization, triangle-driven compute, meshlet-driven compute, or a
// particular material layout.
//
// ════════════════════════════════════════════════════════════════════════════════════════════════

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
});

const DEFAULT_RESOURCE_PREFIX = "scene_voxelizer";

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
    voxel_size: overrides.voxel_size ?? 1.0,
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
 * │  Only Reset is implemented in this first boilerplate. Voxelize and Finalize are explicit      │
 * │  extension points below, which keeps the eventual implementation split into reviewable parts. │
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
  }

  /**
   * Resolves configuration for a new graph build.
   *
   * Per-frame overrides are useful for moving the volume or changing its world-space scale without
   * reconstructing the strategy object. GPU resource handles are graph-local, so they are discarded
   * here and recreated by setup().
   *
   * @param {Object} [frame_context]
   * @returns {Object} The normalized context used by setup() and record().
   */
  begin_frame(frame_context = {}) {
    const voxel_size = frame_context.voxel_size ?? this.config.voxel_size;
    const grid_origin = [...(frame_context.grid_origin ?? this.config.grid_origin)];
    validate_config({
      voxel_size,
      grid_origin,
      resource_prefix: this.config.resource_prefix,
    });

    this.resources.clear();
    this.frame_context = {
      ...frame_context,
      voxel_size,
      grid_origin,
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
    if (!context || !this.get_resource(SceneVoxelizerResource.VoxelGrid)) {
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

  _setup_voxelization_resources(_render_graph, _context) {
    // Reserved for primitive lists, indirect dispatch arguments, and scratch space once the
    // triangle/meshlet coverage algorithm is selected.
  }

  _setup_hierarchy_resources(_render_graph, _context) {
    // Reserved for brick allocators, indirection tables, and packed child/leaf occupancy bitsets.
    // The root grid remains the stable entry point when these resources are introduced.
  }

  // ────────────────────────────────── PASS RECORDING ─────────────────────────────────────────────

  _record_reset_passes(render_graph, _context) {
    const voxel_grid = this.get_resource(SceneVoxelizerResource.VoxelGrid);

    render_graph.add_pass(
      `${this.config.resource_prefix}_reset`,
      RenderPassFlags.GraphLocal,
      {
        inputs: [],
        outputs: [voxel_grid],
      },
      (graph, _frame_data, encoder) => {
        const physical_grid = graph.get_physical_buffer(voxel_grid);
        encoder.clearBuffer(physical_grid.buffer, 0, physical_grid.config.size);
      }
    );
  }

  _record_voxelization_passes(_render_graph, _context) {
    // TODO: Record GPU scene coverage passes here.
    //
    // Expected inputs will likely include scene instances, transforms, meshlet/triangle data, and
    // acceleration-structure metadata. Keeping those inputs on frame_context avoids coupling this
    // acceleration utility to one renderer strategy while the algorithm is still being designed.
  }

  _record_finalize_passes(_render_graph, _context) {
    // TODO: Publish any counters or hierarchy metadata required by downstream consumers. The
    // packed root occupancy grid is already consumer-readable after voxelization.
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

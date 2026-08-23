// ═══════════════════════════════════════════════════════════════════════════════════════════════
// ██████╗ ███████╗███╗   ██╗██████╗ ███████╗██████╗      ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
// ██╔══██╗██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗    ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
// ██████╔╝█████╗  ██╔██╗ ██║██║  ██║█████╗  ██████╔╝    ██║  ███╗██████╔╝███████║██████╔╝███████║
// ██╔══██╗██╔══╝  ██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗    ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
// ██║  ██║███████╗██║ ╚████║██████╔╝███████╗██║  ██║    ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
// ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
// RenderGraph - Frame-Local Rendering Orchestrator
//
// The graph separates declaration from execution. Callers describe logical resources and passes;
// compilation removes dead work, resolves resource lifetimes, and realizes the surviving graph as
// cached WebGPU resources, bind groups, pipelines, and commands.
//
// 🧭 FRAME FLOW:
//    • begin()   : run pre-render callbacks, retire deferred resources, reset frame-local state
//    • declare   : create/register resources and add passes with explicit inputs and outputs
//    • submit()  : cull, order, realize, encode, and submit the surviving passes
//
// 🧠 OWNERSHIP MODEL:
//    • Graph-created resources are logical declarations; physical objects are created lazily
//    • Registered resources are externally owned and always treated as persistent
//    • Transient physical resources are retired through a frame-delayed deletion queue
//    • Pass bind groups and pipelines are cached across frames until explicitly invalidated
//
// ⚡ HOT-PATH DESIGN:
//    • FrameAllocator and StaticIntArray keep frame-local graph bookkeeping reusable
//    • Physical resources are created only for non-culled passes
//    • Reflection-driven bind layouts are cached per pass name
//
// ═══════════════════════════════════════════════════════════════════════════════════════════════
import { ConfigDB, ConfigSync } from "../core/config_db.js";
import ExecutionQueue from "../utility/execution_queue.js";
import { FrameAllocator } from "../memory/allocator.js";
import { ResourceCache } from "./resource_cache.js";
import { BindGroup } from "./bind_group.js";
import { RenderPass } from "./render_pass.js";
import { PipelineState } from "./pipeline_state.js";
import { CommandQueue } from "./command_queue.js";
import { Buffer } from "./buffer.js";
import { Texture } from "./texture.js";
import { Shader } from "./shader.js";
import {
  ImageFlags,
  ShaderResourceType,
  CacheTypes,
  RenderPassFlags,
  BufferFlags,
  BindGroupType,
} from "./renderer_types.js";
import { Name } from "../utility/names.js";
import { StaticIntArray } from "../memory/container.js";
import { profile_scope } from "../utility/performance.js";
import { GPUTimeQuery } from "./query.js";
import { deserialize_json, read_file } from "../streaming/streaming_io.js";
import { deep_clone } from "../utility/object.js";

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                              📊 GRAPH CAPACITY LIMITS                                      │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const max_image_resources = 1024;
const max_buffer_resources = 1024;
const max_render_passes = 1024;

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                           🪪  LOGICAL RESOURCE HANDLE LAYOUT                                │
// │                 [ index: 20 bits | type: 8 bits | version: 4 bits ]                        │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘
const RG_VERSION_BITS = 4;
const RG_VERSION_MASK = (1 << RG_VERSION_BITS) - 1;
const RG_TYPE_BITS = 8;
const RG_TYPE_MASK = (1 << RG_TYPE_BITS) - 1;
const RG_INDEX_BITS = 20;
const RG_INDEX_MASK = (1 << RG_INDEX_BITS) - 1;

// Experimental scene-specific pass ordering. Declaration order remains authoritative while disabled.
const custom_graph_sort = false;

/**
 * Packs an allocator index, resource type, and generation into one 32-bit logical handle.
 *
 * @param {number} index - Index in the type-specific frame allocator.
 * @param {number} type - ResourceType discriminator.
 * @param {number} version - Handle generation, reserved for stale-handle detection.
 * @returns {number} Packed graph resource handle.
 */
function create_graph_resource_handle(index, type, version) {
  return (
    ((index & RG_INDEX_MASK) << (RG_VERSION_BITS + RG_TYPE_BITS)) |
    ((type & RG_TYPE_MASK) << RG_VERSION_BITS) |
    (version & RG_VERSION_MASK)
  );
}

/**
 * Extracts the type-local allocator index from a packed resource handle.
 *
 * @param {number} handle - Packed graph resource handle.
 * @returns {number} Resource allocator index.
 */
function get_graph_resource_index(handle) {
  return (handle >> (RG_VERSION_BITS + RG_TYPE_BITS)) & RG_INDEX_MASK;
}

/**
 * Extracts the ResourceType discriminator from a packed resource handle.
 *
 * @param {number} handle - Packed graph resource handle.
 * @returns {number} ResourceType value.
 */
function get_graph_resource_type(handle) {
  return (handle >> RG_VERSION_BITS) & RG_TYPE_MASK;
}

// ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
// │                         🧱 FRAME-LOCAL GRAPH DATA TEMPLATES                                 │
// │       Frozen templates are deep-cloned into allocators and registries before mutation.     │
// └─────────────────────────────────────────────────────────────────────────────────────────────┘

/** Type tag encoded into every logical graph resource handle. @enum {number} */
const ResourceType = Object.freeze({
  Unknown: 0,
  Image: 1,
  Buffer: 2,
});

/**
 * Lightweight logical resource allocated while declaring a frame.
 *
 * @typedef {Object} RGResource
 * @property {number} handle - Packed logical handle.
 * @property {Object|null} config - Image or buffer creation configuration.
 */
const RGResource = Object.freeze({
  handle: 0,
  config: null,
});

/**
 * Dependency, lifetime, and physical-allocation state for one logical resource.
 *
 * @typedef {Object} RGResourceMetadata
 * @property {number} reference_count - Live graph references used by dead-pass elimination.
 * @property {number} physical_id - ResourceCache identifier; zero until realization.
 * @property {number} first_user - First pass handle touching the resource.
 * @property {number} last_user - Last pass handle touching the resource.
 * @property {Array<number>} producers - Passes that write the resource.
 * @property {Array<number>} consumers - Passes that read the resource.
 * @property {boolean} b_is_persistent - Whether physical storage survives graph retirement.
 * @property {boolean} b_is_bindless - Whether the resource bypasses the pass bind group.
 * @property {number} max_frame_lifetime - Deferred-destruction delay for transient storage.
 */
const RGResourceMetadata = Object.freeze({
  reference_count: 0,
  physical_id: 0,
  first_user: 0,
  last_user: 0,
  producers: [],
  consumers: [],
  b_is_persistent: true,
  b_is_bindless: false,
  max_frame_lifetime: 2,
});

/**
 * Conventional G-buffer bundle shared by deferred-rendering strategies.
 *
 * @typedef {Object} RGGBufferData
 * @property {Object|null} albedo - Base-color target.
 * @property {Object|null} smra - Smoothness/metalness/reflectance/ambient-occlusion target.
 * @property {Object|null} position - World-position target.
 * @property {Object|null} normal - World-normal target.
 * @property {Object|null} entity_id - Entity-picking target.
 * @property {Object|null} depth - Depth target.
 */
const RGGBufferData = Object.freeze({
  albedo: null,
  smra: null,
  position: null,
  normal: null,
  entity_id: null,
  depth: null,
});

/**
 * Mutable context passed to every pass executor.
 *
 * @typedef {Object} RGFrameData
 * @property {number} current_pass - Physical pass identifier, or zero outside a physical pass.
 * @property {ExecutionQueue|null} resource_deletion_queue - Frame-delayed cleanup queue.
 * @property {Array} pass_bindless_resources - Bindless resources exposed to the current executor.
 */
const RGFrameData = Object.freeze({
  current_pass: 0,
  resource_deletion_queue: null,
  pass_bindless_resources: [],
});

/**
 * Reserved pass dependency metadata template.
 *
 * @typedef {Object} RGPassMetadata
 * @property {number} handle - Frame-local pass handle.
 * @property {number} physical_id - ResourceCache pass identifier.
 * @property {number} reference_count - Number of live outputs.
 * @property {Array<number>} inputs - Read dependencies.
 * @property {Array<number>} outputs - Write dependencies.
 * @property {boolean} b_is_culled - Whether compilation removed the pass.
 */
const RGPassMetadata = Object.freeze({
  handle: 0,
  physical_id: 0,
  reference_count: 0,
  inputs: [],
  outputs: [],
  b_is_culled: false,
});

/**
 * Reserved execution configuration for a graph pass.
 *
 * @typedef {Object} RGPassConfig
 * @property {string} name - Name of the pass.
 * @property {boolean} b_is_compute - Whether execution uses a compute pipeline.
 * @property {boolean} b_is_async - Whether execution may move to an asynchronous queue.
 * @property {number} execution_queue - Target command queue.
 */
const RGPassConfig = Object.freeze({
  name: "",
  b_is_compute: false,
  b_is_async: false,
  execution_queue: 0,
});

/**
 * Optional shader and fixed-function state used for automatic pipeline creation.
 *
 * @typedef {Object} RGShaderDataSetup
 * @property {Object|null} pipeline_shaders - Compute or vertex/fragment shader declarations.
 * @property {Object|null} push_constant_data - Optional push-constant payload.
 * @property {Object|null} rasterizer_state - Rasterizer overrides.
 * @property {Object|null} attachment_blend - Color attachment blend state.
 * @property {string|null} primitive_topology_type - WebGPU primitive topology.
 * @property {Object|null} viewport - Optional viewport override.
 * @property {boolean|null} depth_write_enabled - Depth-write override.
 * @property {string|null} depth_stencil_compare_op - WebGPU depth comparison function.
 */
const RGShaderDataSetup = Object.freeze({
  // Automatic setup is optional. Passes without shader declarations own pipeline creation and binding.
  pipeline_shaders: null,
  // All remaining fields are optional pipeline-state overrides.
  push_constant_data: null,
  rasterizer_state: null,
  attachment_blend: null,
  primitive_topology_type: null,
  viewport: null,
  depth_write_enabled: null,
  depth_stencil_compare_op: null,
  force_recreate: false,
});

/**
 * Logical image declaration. Physical textures are created only if a surviving pass uses them.
 *
 * @typedef {Object} RGImageConfig
 * @property {string} name - Stable ResourceCache name.
 * @property {number} width - Texture width.
 * @property {number} height - Texture height.
 * @property {number} depth - Depth or array-layer count.
 * @property {number} mip_levels - Mip count.
 * @property {string} format - WebGPU texture format.
 * @property {number} usage - GPUTextureUsage mask.
 * @property {number} sample_count - Multisample count.
 * @property {boolean} b_is_bindless - Whether to omit the texture from the pass bind group.
 * @property {number} flags - ImageFlags lifetime and attachment behavior.
 * @property {number} max_frame_lifetime - Deferred-destruction delay when transient.
 */
const RGImageConfig = Object.freeze({
  name: "",
  width: 0,
  height: 0,
  depth: 1,
  mip_levels: 1,
  format: "",
  usage: 0,
  sample_count: 1,
  b_is_bindless: false,
  flags: ImageFlags.None,
  max_frame_lifetime: 2,
});

/**
 * Logical buffer declaration. Sizes use the Buffer abstraction's element-count convention.
 *
 * @typedef {Object} RGBufferConfig
 * @property {number} size - Buffer element count.
 * @property {number} usage - GPUBufferUsage mask.
 * @property {boolean} b_is_bindless - Whether to omit the buffer from the pass bind group.
 * @property {number} flags - BufferFlags lifetime behavior.
 * @property {number} max_frame_lifetime - Deferred-destruction delay when transient.
 */
const RGBufferConfig = Object.freeze({
  size: 0,
  usage: 0,
  b_is_bindless: false,
  flags: BufferFlags.None,
  max_frame_lifetime: 2,
});

/**
 * Declarative dependencies and optional automatic setup for one pass.
 *
 * @typedef {Object} RGPassParameters
 * @property {RGShaderDataSetup} shader_setup - Shader and fixed-function pipeline declaration.
 * @property {Array<number>} inputs - Logical resources read by the pass.
 * @property {Array<number>} outputs - Logical resources written by the pass.
 * @property {Array<number>} input_views - Per-input texture view indices.
 * @property {Array<number>} output_views - Per-output attachment view indices.
 * @property {Array<number>} pass_inputs - Non-bindless inputs classified during realization.
 * @property {Array<number>} bindless_inputs - Bindless inputs classified during realization.
 * @property {string|null} bind_group_cache_key - Optional bind-group cache identity.
 * @property {boolean} b_skip_pass_bind_group_setup - Let the executor own its pass bind group.
 * @property {boolean} b_skip_pass_pipeline_setup - Let the executor own its pipeline.
 * @property {boolean} b_force_keep_pass - Preserve the pass even when its outputs are unused.
 */
const RGPassParameters = Object.freeze({
  shader_setup: deep_clone(RGShaderDataSetup),
  inputs: [],
  outputs: [],
  input_views: [],
  output_views: [],
  pass_inputs: [],
  bindless_inputs: [],
  bind_group_cache_key: null,
  b_skip_pass_bind_group_setup: false,
  b_skip_pass_pipeline_setup: false,
  b_force_keep_pass: false,
});

/**
 * Frame-local pass record connecting declaration, compiled state, and execution.
 *
 * @typedef {Object} RGPass
 * @property {number} handle - Frame-local pass index.
 * @property {Object} pass_config - Physical pass and attachment configuration.
 * @property {RGPassParameters} parameters - Declared graph dependencies.
 * @property {Function} executor - Command-recording callback.
 * @property {Object} shaders - Realized shader objects by stage.
 * @property {number} physical_id - ResourceCache render-pass identifier.
 * @property {number} pipeline_state_id - ResourceCache pipeline identifier.
 * @property {number} reference_count - Live output count used during culling.
 */
const RGPass = Object.freeze({
  handle: 0,
  pass_config: null,
  parameters: null,
  executor: null,
  shaders: {},
  physical_id: 0,
  pipeline_state_id: 0,
  reference_count: 0,
});

/**
 * Per-frame graph registry.
 *
 * Logical resources, pass records, and dependency metadata are cleared at the next begin(). The
 * separate PassCache intentionally survives frame resets. Long-lived images and buffers must be
 * created externally and registered so the graph never assumes ownership of their destruction.
 *
 * @typedef {Object} RGRegistry
 * @property {Array<RGPass>} render_passes - Pass records in declaration order.
 * @property {Map<string, number>} pass_order_map - Configured pass name to sort rank.
 * @property {StaticIntArray} all_resource_handles - Dense list of logical resources.
 * @property {Map<number, RGResourceMetadata>} resource_metadata - Metadata by logical handle.
 * @property {Map<number, number>} resource_names_to_handles - Encoded names to logical handles.
 * @property {Array<number>} all_bindless_resource_handles - Bindless allocations awaiting release.
 * @property {ExecutionQueue} resource_deletion_queue - Deferred physical-resource destruction.
 * @property {boolean} b_global_bind_group_bound - Per-frame global binding state.
 */
const RGRegistry = Object.freeze({
  current_scene_id: "",
  render_passes: [],
  pass_order_map: new Map(),
  all_resource_handles: new StaticIntArray(max_image_resources + max_buffer_resources),
  resource_metadata: new Map(),
  resource_names_to_handles: new Map(),
  all_bindless_resource_handles: [],
  resource_deletion_queue: new ExecutionQueue(),
  b_global_bind_group_bound: false,
});

/**
 * Cross-frame cache for descriptor and pipeline objects keyed by pass name.
 *
 * @typedef {Object} PassCache
 * @property {BindGroup|null} global_bind_group - Shared global descriptor set.
 * @property {Map<string, Object>} bind_groups - Global/pass bind groups per pass.
 * @property {Map<string, number>} pipeline_states - Pipeline identifiers per pass.
 */
const PassCache = Object.freeze({
  global_bind_group: null,
  bind_groups: new Map(),
  pipeline_states: new Map(),
});

const CustomPassOrderReadyFlag = 1 << 0;
const DefaultPassOrderReadyFlag = 1 << 1;

/**
 * Default and user-edited pass orders loaded asynchronously per scene.
 *
 * @typedef {Object} StoredPassOrder
 * @property {Object<string, Array<string>>} default - Recorded declaration order per scene.
 * @property {Object<string, Array<string>>} custom - User-selected execution order per scene.
 * @property {number} ready_flags - Bitmask indicating which stores have finished loading.
 */
const StoredPassOrder = Object.freeze({
  default: [],
  custom: [],
  ready_flags: 0,
});

/**
 * ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
 * ║                                🎬 RenderGraph CLASS                                         ║
 * ║               Logical dependency compiler and WebGPU submission coordinator                ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
 *
 * A RenderGraph is rebuilt every frame from logical image, buffer, and pass declarations. It
 * compiles those declarations into the smallest executable pass set, realizes only the required
 * physical resources, and records the result into one command encoder.
 *
 * ┌────────────────────────────── 🔄 DATA FLOW ──────────────────────────────────┐
 * │  logical declarations → dependency compilation → physical realization       │
 * │       → bind/pipeline setup → command encoding → deferred retirement         │
 * └───────────────────────────────────────────────────────────────────────────────┘
 *
 * @example
 * const graph = RenderGraph.create(max_bind_groups);
 * graph.begin();
 * const color = graph.create_image(color_config);
 * graph.add_pass("shade", RenderPassFlags.Present, { outputs: [color] }, execute);
 * graph.submit();
 */
export class RenderGraph {
  constructor(max_bind_groups) {
    // Cross-frame cache and frame-local declaration state.
    this.max_bind_groups = max_bind_groups;
    this.pass_cache = deep_clone(PassCache);
    this.registry = deep_clone(RGRegistry);
    this.stored_pass_order = deep_clone(StoredPassOrder);
    this.non_culled_passes = [];
    this.queued_global_bind_group_writes = [];
    this.queued_pre_commands = [];
    this.queued_post_commands = [];
    this.pre_render_callbacks = [];
    this.post_render_callbacks = [];
    this.pass_cache_full_needs_reset = false;
    this.pass_cache_passes_needs_reset = false;
    this.pass_cache_pipeline_states_need_recreate = false;

    // Fixed-capacity allocators recycle graph records without per-frame object churn.
    this.image_resource_allocator = new FrameAllocator(max_image_resources, deep_clone(RGResource));
    this.buffer_resource_allocator = new FrameAllocator(
      max_buffer_resources,
      deep_clone(RGResource)
    );
    this.render_pass_allocator = new FrameAllocator(max_render_passes, deep_clone(RGPass));

    this.resource_metadata_allocator = new FrameAllocator(
      max_image_resources + max_buffer_resources,
      deep_clone(RGResourceMetadata)
    );

    this._execute_post_render_callbacks = this._execute_post_render_callbacks.bind(this);
    this._execute_pre_render_callbacks = this._execute_pre_render_callbacks.bind(this);

    if (custom_graph_sort) {
      this._init_pass_order_info();
    }
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                              🔄 FRAME LIFECYCLE                                           ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Clears frame-local declarations and immediately flushes all deferred cleanup work.
   * Call only when the graph will no longer submit frames.
   */
  destroy() {
    this.reset();
    this.registry.resource_deletion_queue.flush();
  }

  /**
   * Opens a declaration frame.
   *
   * Pre-render callbacks are dispatched before the previous frame's logical state is reset.
   * Queued pre-commands are then materialized as graph-local passes.
   */
  begin() {
    this._execute_pre_render_callbacks();
    this.reset();
    this._add_queued_pre_commands();
  }

  /**
   * Replaces any cleanup entry with the same ID and schedules it after a frame delay.
   *
   * @param {Function} execution - Cleanup callback.
   * @param {string|number} execution_id - Stable deduplication key.
   * @param {number} execution_frame_delay - Frames to wait before invoking the callback.
   */
  queue_resource_deletion(execution, execution_id, execution_frame_delay = 0) {
    this.registry.resource_deletion_queue.remove_execution_by_id(execution_id);
    this.registry.resource_deletion_queue.push_execution(
      execution,
      execution_id,
      execution_frame_delay
    );
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                         🧱 RESOURCE DECLARATION & IMPORT                                  ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Declares an image owned by the graph.
   *
   * Physical texture creation is deferred until a surviving pass consumes or produces the image.
   *
   * @param {RGImageConfig} config - Logical image configuration.
   * @returns {number} Frame-local image handle.
   */
  create_image(config) {
    let new_resource;

    const index = this.image_resource_allocator.length;

    new_resource = this.image_resource_allocator.allocate();
    new_resource.config = { ...RGImageConfig, ...config };
    new_resource.config.encoded_name = Name.from(config.name);

    new_resource.handle = create_graph_resource_handle(index, ResourceType.Image, 1);

    this.registry.all_resource_handles.add(new_resource.handle);
    this.registry.resource_metadata.set(
      new_resource.handle,
      this.resource_metadata_allocator.allocate()
    );
    this.registry.resource_names_to_handles.set(
      new_resource.config.encoded_name,
      new_resource.handle
    );

    const resource_metadata = this.registry.resource_metadata.get(new_resource.handle);
    resource_metadata.config = new_resource.config;
    resource_metadata.b_is_bindless = config.b_is_bindless;
    resource_metadata.b_is_persistent = (new_resource.config.flags & ImageFlags.Transient) === 0;
    resource_metadata.max_frame_lifetime = config.max_frame_lifetime || 0;
    resource_metadata.reference_count = 0;
    resource_metadata.physical_id = 0;
    resource_metadata.first_user = 0;
    resource_metadata.last_user = 0;
    resource_metadata.producers.length = 0;
    resource_metadata.consumers.length = 0;

    return new_resource.handle;
  }

  /**
   * Imports an existing ResourceCache image as a persistent logical resource.
   *
   * Ownership remains external; reset and transient retirement never destroy the image.
   *
   * @param {string|number} image - Image name or encoded ResourceCache identifier.
   * @returns {number} Frame-local image handle.
   */
  register_image(image) {
    let new_resource;

    const physical_id = Name.from(image);
    const image_obj = ResourceCache.get().fetch(CacheTypes.IMAGE, physical_id);

    const index = this.image_resource_allocator.length;

    new_resource = this.image_resource_allocator.allocate();
    new_resource.config = {
      ...RGImageConfig,
      ...image_obj.config,
    };
    new_resource.config.encoded_name = physical_id;

    new_resource.handle = create_graph_resource_handle(index, ResourceType.Image, 1);

    this.registry.all_resource_handles.add(new_resource.handle);
    this.registry.resource_metadata.set(
      new_resource.handle,
      this.resource_metadata_allocator.allocate()
    );
    this.registry.resource_names_to_handles.set(
      new_resource.config.encoded_name,
      new_resource.handle
    );

    const resource_metadata = this.registry.resource_metadata.get(new_resource.handle);
    resource_metadata.config = new_resource.config;
    resource_metadata.physical_id = physical_id;
    resource_metadata.b_is_persistent = true;
    resource_metadata.b_is_bindless = new_resource.config.b_is_bindless;
    resource_metadata.reference_count = 0;
    resource_metadata.first_user = 0;
    resource_metadata.last_user = 0;
    resource_metadata.producers.length = 0;
    resource_metadata.consumers.length = 0;

    return new_resource.handle;
  }

  /**
   * Declares a buffer owned by the graph.
   *
   * Physical buffer creation is deferred until a surviving pass uses the declaration.
   *
   * @param {RGBufferConfig} config - Logical buffer configuration.
   * @returns {number} Frame-local buffer handle.
   */
  create_buffer(config) {
    let new_resource;

    const index = this.buffer_resource_allocator.length;

    new_resource = this.buffer_resource_allocator.allocate();
    new_resource.config = { ...RGBufferConfig, ...config };
    new_resource.config.encoded_name = Name.from(config.name);

    new_resource.handle = create_graph_resource_handle(index, ResourceType.Buffer, 1);

    this.registry.all_resource_handles.add(new_resource.handle);
    this.registry.resource_metadata.set(
      new_resource.handle,
      this.resource_metadata_allocator.allocate()
    );
    this.registry.resource_names_to_handles.set(
      new_resource.config.encoded_name,
      new_resource.handle
    );

    const resource_metadata = this.registry.resource_metadata.get(new_resource.handle);
    resource_metadata.config = new_resource.config;
    resource_metadata.b_is_bindless = config.b_is_bindless;
    resource_metadata.b_is_persistent = (new_resource.config.flags & BufferFlags.Transient) === 0;
    resource_metadata.max_frame_lifetime = config.max_frame_lifetime;
    resource_metadata.reference_count = 0;
    resource_metadata.physical_id = 0;
    resource_metadata.first_user = 0;
    resource_metadata.last_user = 0;
    resource_metadata.producers.length = 0;
    resource_metadata.consumers.length = 0;

    return new_resource.handle;
  }

  /**
   * Imports an existing ResourceCache buffer as a persistent logical resource.
   *
   * @param {string|number} buffer - Buffer name or encoded ResourceCache identifier.
   * @returns {number} Frame-local buffer handle.
   */
  register_buffer(buffer) {
    let new_resource;

    const physical_id = Name.from(buffer);
    const buffer_obj = ResourceCache.get().fetch(CacheTypes.BUFFER, physical_id);

    const index = this.buffer_resource_allocator.length;

    new_resource = this.buffer_resource_allocator.allocate();
    new_resource.config = {
      ...RGBufferConfig,
      ...buffer_obj.config,
    };
    new_resource.config.encoded_name = physical_id;

    new_resource.handle = create_graph_resource_handle(index, ResourceType.Buffer, 1);

    this.registry.all_resource_handles.add(new_resource.handle);
    this.registry.resource_metadata.set(
      new_resource.handle,
      this.resource_metadata_allocator.allocate()
    );
    this.registry.resource_names_to_handles.set(
      new_resource.config.encoded_name,
      new_resource.handle
    );

    const resource_metadata = this.registry.resource_metadata.get(new_resource.handle);
    resource_metadata.config = new_resource.config;
    resource_metadata.physical_id = physical_id;
    resource_metadata.b_is_persistent = true;
    resource_metadata.b_is_bindless = new_resource.config.b_is_bindless;
    resource_metadata.reference_count = 0;
    resource_metadata.first_user = 0;
    resource_metadata.last_user = 0;
    resource_metadata.producers.length = 0;
    resource_metadata.consumers.length = 0;

    return new_resource.handle;
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                              🎬 PASS DECLARATION                                          ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Declares a pass and records its producer/consumer edges.
   *
   * Present passes and explicitly forced passes form graph roots. Graph-local passes invoke their
   * executor directly and therefore skip RenderPass, bind-group, and pipeline realization.
   *
   * @param {string} name - Stable pass and cache key.
   * @param {number} pass_type - RenderPassFlags mask.
   * @param {RGPassParameters|null} params - Inputs, outputs, and automatic setup options.
   * @param {Function} execution_callback - Command-recording callback.
   * @returns {number} Frame-local pass handle.
   */
  add_pass(name, pass_type, params, execution_callback) {
    let index;

    const pass = this.render_pass_allocator.allocate();
    pass.pass_config = {
      name: name,
      encoded_name: Name.from(name),
      flags: pass_type,
      attachments: [],
      viewport: params?.viewport ?? null,
      scissor_rect: params?.scissor_rect ?? null,
    };
    pass.parameters = { ...deep_clone(RGPassParameters), ...params };
    pass.executor = execution_callback;
    pass.shaders = {};
    pass.physical_id = 0;
    pass.pipeline_state_id = 0;
    pass.reference_count = 0;

    index = this.registry.render_passes.length;
    this.registry.render_passes.push(pass);
    this.non_culled_passes.push(index);
    pass.handle = index;

    if (pass.parameters.inputs) {
      pass.parameters.inputs = pass.parameters.inputs.filter((input) => {
        return input !== null;
      });
    }
    if (pass.parameters.outputs) {
      pass.parameters.outputs = pass.parameters.outputs.filter((output) => {
        return output !== null;
      });
    }

    if (pass.pass_config.flags & RenderPassFlags.GraphLocal) {
      pass.parameters.b_force_keep_pass = true;
    }

    if (params) {
      this._update_reference_counts(pass);
      this._update_resource_param_producers_and_consumers(pass);
      this._update_present_pass_status(pass);
    } else {
      pass.reference_count += 1;
    }

    return index;
  }

  /**
   * Resolves a logical pass handle to its realized RenderPass.
   *
   * @param {number} handle - Frame-local pass handle.
   * @returns {RenderPass|null} Cached physical pass.
   */
  get_physical_pass(handle) {
    return ResourceCache.get().fetch(CacheTypes.PASS, handle);
  }

  /**
   * Resolves a logical image handle to its realized Texture.
   *
   * @param {number} handle - Frame-local image handle.
   * @returns {Texture|null} Cached physical image.
   */
  get_physical_image(handle) {
    return ResourceCache.get().fetch(
      CacheTypes.IMAGE,
      this.registry.resource_metadata.get(handle).physical_id
    );
  }

  /**
   * Resolves a logical buffer handle to its realized Buffer.
   *
   * @param {number} handle - Frame-local buffer handle.
   * @returns {Buffer|null} Cached physical buffer.
   */
  get_physical_buffer(handle) {
    return ResourceCache.get().fetch(
      CacheTypes.BUFFER,
      this.registry.resource_metadata.get(handle).physical_id
    );
  }

  /**
   * Resolves a logical resource handle from a raw resource name.
   *
   * @param {string} name - Name of the resource (usually the name passed into its config on creation).
   * @returns {number|null} Registered handle for the named resource if one was found.
   */
  get_resource_handle(name) {
    const encoded_name = Name.from(name);
    return this.registry.resource_names_to_handles.get(encoded_name) || null;
  }

  /**
   * Returns the merged creation configuration for any logical resource.
   *
   * @param {number} handle - Frame-local image or buffer handle.
   * @returns {Object} Logical resource configuration.
   */
  get_resource_config(handle) {
    return this.registry.resource_metadata.get(handle).config;
  }

  /**
   * Selects the scene namespace used by configurable pass ordering.
   *
   * @param {string|number} scene_id - Scene identifier.
   */
  set_scene_id(scene_id) {
    this.registry.current_scene_id = scene_id;
  }

  /**
   * Queues ordered global-bind-group entries for the next physical pass setup.
   *
   * Entry order is binding order. Supplying overwrite replaces any writes already queued.
   *
   * @param {Array<Object>} writes - Buffer, sampler, or texture-view binding descriptors.
   * @param {boolean} overwrite - Replace rather than append to the pending descriptors.
   */
  queue_global_bind_group_write(writes, overwrite = false) {
    if (overwrite) {
      this.queued_global_bind_group_writes = writes;
    } else {
      this.queued_global_bind_group_writes = [...this.queued_global_bind_group_writes, ...writes];
    }
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                         ⏩ GRAPH-LOCAL COMMAND QUEUES                                      ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Queues a graph-local executor to be inserted by the next begin().
   *
   * @param {string} name - Pass name.
   * @param {Function} commands_callback - Direct command-encoding callback.
   * @param {boolean} persistent - Reinsert the command every frame until removed.
   */
  queue_pre_commands(name, commands_callback, persistent = false) {
    this.queued_pre_commands.push({ name, commands_callback, persistent });
  }

  /**
   * Removes every queued pre-command with the supplied name.
   *
   * @param {string} name - Pass name to remove.
   */
  unqueue_pre_commands(name) {
    for (let i = this.queued_pre_commands.length - 1; i >= 0; i--) {
      if (this.queued_pre_commands[i].name === name) {
        this.queued_pre_commands.splice(i, 1);
      }
    }
  }

  _add_queued_pre_commands() {
    for (let i = this.queued_pre_commands.length - 1; i >= 0; i--) {
      const command = this.queued_pre_commands[i];
      this.add_pass(command.name, RenderPassFlags.GraphLocal, {}, command.commands_callback);
      if (!command.persistent) {
        this.queued_pre_commands.splice(i, 1);
      }
    }
  }

  /**
   * Queues a graph-local executor to be appended during the next submit().
   *
   * @param {string} name - Pass name.
   * @param {Function} commands_callback - Direct command-encoding callback.
   * @param {boolean} persistent - Reinsert the command every frame until removed.
   */
  queue_post_commands(name, commands_callback, persistent = false) {
    this.queued_post_commands.push({ name, commands_callback, persistent });
  }

  /**
   * Removes every queued post-command with the supplied name.
   *
   * @param {string} name - Pass name to remove.
   */
  unqueue_post_commands(name) {
    for (let i = this.queued_post_commands.length - 1; i >= 0; i--) {
      if (this.queued_post_commands[i].name === name) {
        this.queued_post_commands.splice(i, 1);
      }
    }
  }

  _add_queued_post_commands() {
    for (let i = this.queued_post_commands.length - 1; i >= 0; i--) {
      const command = this.queued_post_commands[i];
      this.add_pass(command.name, RenderPassFlags.GraphLocal, {}, command.commands_callback);
      if (!command.persistent) {
        this.queued_post_commands.splice(i, 1);
      }
    }
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                      🧭 EXPERIMENTAL PASS-ORDER BOOTSTRAP                                 ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  _init_pass_order_info() {
    const config_file = read_file("config/renderer.config.json");
    if (config_file) {
      const config = deserialize_json(
        config_file,
        "Renderer configuration"
      );
      if (config.rg?.pass_order?.default) {
        ConfigDB.set_config_property(
          "renderer.config",
          "rg.pass_order.default",
          config.rg.pass_order.default
        ).then(() => {
          this.stored_pass_order.default = config.rg.pass_order.default;
          this.stored_pass_order.ready_flags |= DefaultPassOrderReadyFlag;
          if (config.rg?.pass_order?.custom) {
            ConfigDB.set_config_property(
              "renderer.config",
              "rg.pass_order.custom",
              config.rg.pass_order.custom
            ).then(() => {
              this.stored_pass_order.custom = config.rg.pass_order.custom;
              this.stored_pass_order.ready_flags |= CustomPassOrderReadyFlag;
            });
          }
        });
      }
    } else {
      ConfigDB.get_config_property("renderer.config", "rg.pass_order.default").then(
        (pass_order) => {
          this.stored_pass_order.default = pass_order || {};
          this.stored_pass_order.ready_flags |= DefaultPassOrderReadyFlag;
        }
      );

      ConfigDB.get_config_property("renderer.config", "rg.pass_order.custom").then((pass_order) => {
        this.stored_pass_order.custom = pass_order || {};
        this.stored_pass_order.ready_flags |= CustomPassOrderReadyFlag;
      });
    }
  }

  _update_reference_counts(pass) {
    pass.reference_count += pass.parameters.outputs.length;
    for (let i = 0; i < pass.parameters.inputs.size; i++) {
      const resource = pass.parameters.inputs[i];
      if (this.registry.resource_metadata.has(resource)) {
        const metadata = this.registry.resource_metadata.get(resource);
        metadata.reference_count += 1;
      }
    }
  }

  _update_resource_param_producers_and_consumers(pass) {
    for (let i = 0; i < pass.parameters.inputs.size; i++) {
      const resource = pass.parameters.inputs[i];
      if (this.registry.resource_metadata.has(resource)) {
        const metadata = this.registry.resource_metadata.get(resource);
        metadata.consumers.push(pass.handle);
      }
    }
    for (let i = 0; i < pass.parameters.outputs.size; i++) {
      const resource = pass.parameters.outputs[i];
      if (this.registry.resource_metadata.has(resource)) {
        const metadata = this.registry.resource_metadata.get(resource);
        metadata.producers.push(pass.handle);
      }
    }
  }

  _update_present_pass_status(pass) {
    pass.pass_config.b_is_present_pass =
      (pass.pass_config.flags & RenderPassFlags.Present) !== RenderPassFlags.None;
    pass.parameters.b_force_keep_pass =
      pass.parameters.b_force_keep_pass || pass.pass_config.b_is_present_pass;
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                           🧠 DEPENDENCY COMPILATION                                       ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Converts declaration state into an ordered set of live passes with resolved lifetimes.
   *
   * Compilation mutates reference counts and must run once after declarations are complete.
   */
  _compile() {
    this._cull_graph_passes();
    this._sort_graph_passes();
    this._compute_resource_first_and_last_users();
  }

  _cull_graph_passes() {
    const passes_to_cull = new Set();
    const unused_stack = [];

    // Removing a dead producer can make its own inputs dead; feed those resources back into the
    // worklist so liveness propagates toward the roots of the dependency graph.
    const decrement_producer_and_subresource_ref_counts = (producers) => {
      for (let i = 0; i < producers.size; i++) {
        const pass_handle = producers[i];

        const producer_pass = this.registry.render_passes[pass_handle];

        if (producer_pass.parameters.b_force_keep_pass) {
          continue;
        }

        --producer_pass.reference_count;

        if (producer_pass.reference_count <= 0) {
          producer_pass.reference_count = 0;

          passes_to_cull.add(pass_handle);

          for (const resource of producer_pass.parameters.inputs) {
            if (this.registry.resource_metadata.has(resource)) {
              const metadata = this.registry.resource_metadata.get(resource);
              metadata.reference_count -= 1;
              if (metadata.reference_count === 0) {
                unused_stack.push(resource);
              }
            }
          }
        }
      }
    };

    // Seed the worklist with logical resources that have no surviving consumers.
    for (let i = 0; i < this.registry.all_resource_handles.length; i++) {
      const resource = this.registry.all_resource_handles.get(i);
      if (this.registry.resource_metadata.has(resource)) {
        if (this.registry.resource_metadata.get(resource).reference_count === 0) {
          unused_stack.push(resource);
        }
      }
    }

    // Walk backward through producer edges until the dead subgraph reaches a fixed point.
    while (unused_stack.length > 0) {
      const unused_resource = unused_stack.pop();
      if (this.registry.resource_metadata.has(unused_resource)) {
        decrement_producer_and_subresource_ref_counts(
          this.registry.resource_metadata.get(unused_resource).producers
        );
      }
    }

    // Compact in reverse so each splice preserves the remaining pass handles.
    for (let i = this.non_culled_passes.length - 1; i >= 0; --i) {
      const pass = this.non_culled_passes[i];
      if (passes_to_cull.has(pass)) {
        this.non_culled_passes.splice(i, 1);
      }
    }
  }

  _sort_graph_passes() {
    // Custom ordering is opt-in and scene-local; otherwise declaration order is preserved.
    const current_pass_order = this.stored_pass_order.custom[this.registry.current_scene_id] || [];
    if (!custom_graph_sort || !current_pass_order || current_pass_order.length === 0) {
      return;
    }

    // Unlisted passes sort after configured passes and retain declaration order as a tiebreaker.
    this.non_culled_passes.sort((a, b) => {
      const pass_a = this.registry.render_passes[a];
      const pass_b = this.registry.render_passes[b];

      const id_a = pass_a.pass_config.name;
      const id_b = pass_b.pass_config.name;

      const order_a = this.registry.pass_order_map.has(id_a)
        ? this.registry.pass_order_map.get(id_a)
        : Number.MAX_SAFE_INTEGER;
      const order_b = this.registry.pass_order_map.has(id_b)
        ? this.registry.pass_order_map.get(id_b)
        : Number.MAX_SAFE_INTEGER;

      return order_a === order_b ? a - b : order_a - order_b;
    });
  }

  _compute_resource_first_and_last_users() {
    // Bounds are stored as declaration-order pass handles; custom sorting does not remap them.
    for (let i = 0; i < this.registry.all_resource_handles.length; i++) {
      const resource = this.registry.all_resource_handles.get(i);
      if (this.registry.resource_metadata.has(resource)) {
        const metadata = this.registry.resource_metadata.get(resource);
        if (metadata.reference_count === 0) {
          continue;
        }

        metadata.first_user = Number.MAX_SAFE_INTEGER;
        metadata.last_user = Number.MIN_SAFE_INTEGER;
        for (let i = 0; i < metadata.producers.size; i++) {
          const pass = metadata.producers[i];

          if (pass < metadata.first_user) {
            metadata.first_user = pass;
          }
          if (pass > metadata.last_user) {
            metadata.last_user = pass;
          }
        }
        for (let i = 0; i < metadata.consumers.size; i++) {
          const pass = metadata.consumers[i];

          if (pass < metadata.first_user) {
            metadata.first_user = pass;
          }
          if (pass > metadata.last_user) {
            metadata.last_user = pass;
          }
        }
      }
    }
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                         ♻️  CACHE INVALIDATION & SUBMISSION                               ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Invalidates cached bind groups after one of their physical resources is replaced.
   *
   * @param {boolean} pass_only - Preserve the global group and rebuild only pass groups.
   */
  mark_pass_cache_bind_groups_dirty(pass_only = false) {
    if (pass_only) {
      this.pass_cache_passes_needs_reset = true;
    } else {
      this.pass_cache_full_needs_reset = true;
    }
  }

  /**
   * Clears pipeline identifiers and forces pipeline plus bind-group recreation on next submit().
   */
  recreate_pipeline_states() {
    this.pass_cache.pipeline_states.clear();
    this.pass_cache_full_needs_reset = true;
    this.pass_cache_pipeline_states_need_recreate = true;
  }

  /**
   * Registers a callback dispatched by begin() before frame-local state is reset.
   *
   * @param {Function} callback - Synchronous or asynchronous callback.
   */
  on_pre_render(callback) {
    this.pre_render_callbacks.push(callback);
  }

  /**
   * Removes a previously registered pre-render callback.
   *
   * @param {Function} callback - Exact function reference to remove.
   */
  remove_pre_render(callback) {
    const index = this.pre_render_callbacks.indexOf(callback);
    if (index !== -1) {
      this.pre_render_callbacks.splice(index, 1);
    }
  }

  /**
   * Compiles, realizes, encodes, and submits the current declaration frame.
   *
   * All surviving passes share one command encoder. Physical setup completes before encoding so
   * callbacks can resolve any declared resource through the graph's physical accessors.
   */
  submit() {
    profile_scope("RenderGraph.submit", () => {
      this._reset_all_pass_cache_bind_groups();
      this._add_queued_post_commands();
      this._compile();

      if (this.non_culled_passes.length === 0) {
        return;
      }

      this._reset_pass_cache_bind_groups();

      // Realize resources and pass state only after dead-pass elimination.
      for (let i = 0; i < this.non_culled_passes.length; i++) {
        const pass_handle = this.non_culled_passes[i];
        this._setup_physical_pass_and_resources(this.registry.render_passes[pass_handle]);
      }
      this.pass_cache_pipeline_states_need_recreate = false;

      const encoder = CommandQueue.create_encoder("render_graph_encoder");

      const frame_data = deep_clone(RGFrameData);
      frame_data.resource_deletion_queue = this.registry.resource_deletion_queue;

      // Encode surviving passes in compiled order into a single submission.
      for (let i = 0; i < this.non_culled_passes.length; i++) {
        const pass_handle = this.non_culled_passes[i];
        this._execute_pass(this.registry.render_passes[pass_handle], frame_data, encoder);
      }

      if (__DEV__) {
        GPUTimeQuery.resolve(encoder);
      }

      this._reset_pass_cache_bind_groups();

      CommandQueue.submit(encoder, this._execute_post_render_callbacks);
    });
  }

  /**
   * Registers a callback run after the command queue completes the submission callback.
   *
   * @param {Function} callback - Synchronous or asynchronous callback.
   */
  on_post_render(callback) {
    this.post_render_callbacks.push(callback);
  }

  /**
   * Removes a previously registered post-render callback.
   *
   * @param {Function} callback - Exact function reference to remove.
   */
  remove_post_render(callback) {
    const index = this.post_render_callbacks.indexOf(callback);
    if (index !== -1) {
      this.post_render_callbacks.splice(index, 1);
    }
  }

  /**
   * Retires due physical resources and recycles every frame-local graph record.
   *
   * Cross-frame bind-group and pipeline caches intentionally remain intact.
   */
  reset() {
    this._free_physical_resources();

    this.non_culled_passes.length = 0;

    this.registry.render_passes.length = 0;
    this.registry.resource_metadata.clear();
    this.registry.resource_names_to_handles.clear();

    this.registry.all_resource_handles.reset();
    this.image_resource_allocator.reset();
    this.buffer_resource_allocator.reset();
    this.render_pass_allocator.reset();
    this.resource_metadata_allocator.reset();

    if (__DEV__) {
      GPUTimeQuery.reset();
    }
  }

  /**
   * Returns physical IDs for the compiled pass set in execution order.
   *
   * @returns {Array<number>} Physical pass identifiers.
   */
  get_resolved_non_culled_passes() {
    return this.non_culled_passes.map((pass) => this.registry.render_passes[pass].physical_id);
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                        🧭 SCENE-SPECIFIC PASS ORDERING                                    ║
  // ║                 Experimental: active only while custom_graph_sort is true                 ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Persists the recorded declaration order for all scenes.
   *
   * @returns {Promise<void>} Resolves after the renderer configuration is saved.
   */
  async record_default_pass_order() {
    await ConfigDB.set_config_property(
      "renderer.config",
      "rg.pass_order.default",
      this.stored_pass_order.default
    );
    await ConfigSync.save_to_server("renderer.config");
  }

  /**
   * Persists custom scene orders and rebuilds the active scene's lookup map.
   *
   * @returns {Promise<void>} Resolves after the renderer configuration is saved.
   */
  async record_custom_pass_order() {
    await ConfigDB.set_config_property(
      "renderer.config",
      "rg.pass_order.custom",
      this.stored_pass_order.custom
    );
    await ConfigSync.save_to_server("renderer.config");
    this._update_pass_order_map();
  }

  /**
   * @returns {boolean} Whether scene-specific pass sorting is compiled in.
   */
  is_custom_graph_sort_enabled() {
    return custom_graph_sort;
  }

  /**
   * @returns {boolean} Whether default pass-order data has finished loading.
   */
  is_default_pass_order_ready() {
    return (this.stored_pass_order.ready_flags & DefaultPassOrderReadyFlag) !== 0;
  }

  /**
   * @returns {boolean} Whether custom pass-order data has finished loading.
   */
  is_custom_pass_order_ready() {
    return (this.stored_pass_order.ready_flags & CustomPassOrderReadyFlag) !== 0;
  }

  /**
   * Stores a scene's baseline declaration order in memory.
   *
   * @param {Array<string>} value - Pass names in declaration order.
   * @param {string|number|null} scene_id - Scene override; current scene when omitted.
   */
  set_default_pass_order(value, scene_id = null) {
    const scene = scene_id ?? this.registry.current_scene_id;
    this.stored_pass_order.default[scene] = value;
  }

  /**
   * Reads a scene's baseline declaration order.
   *
   * @param {string|number|null} scene_id - Scene override; current scene when omitted.
   * @returns {Array<string>} Stored pass names, or an empty array.
   */
  get_default_pass_order(scene_id = null) {
    const scene = scene_id ?? this.registry.current_scene_id;
    return this.stored_pass_order.default[scene] || [];
  }

  /**
   * Stores a custom scene order and refreshes the active lookup map when applicable.
   *
   * @param {Array<string>} value - Pass names in desired execution order.
   * @param {string|number|null} scene_id - Scene override; current scene when omitted.
   */
  set_scene_pass_order(value, scene_id = null) {
    const scene = scene_id ?? this.registry.current_scene_id;
    this.stored_pass_order.custom[scene] = value;
    if (scene === this.registry.current_scene_id) {
      this._update_pass_order_map();
    }
  }

  /**
   * Reads a scene's custom execution order.
   *
   * @param {string|number|null} scene_id - Scene override; current scene when omitted.
   * @returns {Array<string>} Stored pass names, or an empty array.
   */
  get_scene_pass_order(scene_id = null) {
    const scene = scene_id ?? this.registry.current_scene_id;
    return this.stored_pass_order.custom[scene] || [];
  }

  _update_pass_order_map() {
    this.registry.pass_order_map.clear();
    const current_pass_order = this.stored_pass_order.custom[this.registry.current_scene_id] || [];
    for (let i = 0; i < current_pass_order.length; i++) {
      this.registry.pass_order_map.set(current_pass_order[i], i);
    }
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                           ⚙️  CALLBACK & PASS EXECUTION                                   ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  async _execute_post_render_callbacks() {
    for (let i = 0; i < this.post_render_callbacks.length; i++) {
      await this.post_render_callbacks[i]();
    }
    if (__DEV__) {
      await GPUTimeQuery.read();
    }
  }

  async _execute_pre_render_callbacks() {
    for (let i = 0; i < this.pre_render_callbacks.length; i++) {
      await this.pre_render_callbacks[i]();
    }
  }

  _execute_pass(pass, frame_data, encoder) {
    if (!pass) {
      throw new Error("Cannot execute null pass");
    }

    if ((pass.pass_config.flags & RenderPassFlags.GraphLocal) !== RenderPassFlags.None) {
      // Graph-local passes record directly against the shared encoder.
      pass.executor(this, frame_data, encoder);
    } else {
      const physical_pass = ResourceCache.get().fetch(CacheTypes.PASS, pass.physical_id);

      if (!physical_pass) {
        throw new Error("Physical pass is null");
      }

      const pipeline = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, pass.pipeline_state_id);
      if (pass.pipeline_state_id && (!pipeline || !pipeline.is_ready())) {
        // Pipeline creation may be asynchronous; defer this pass rather than binding partial state.
        frame_data.current_pass = 0;
        return;
      }

      frame_data.current_pass = pass.physical_id;

      encoder.pushDebugGroup(pass.pass_config.name);
      physical_pass.begin(encoder, pipeline);
      this._bind_pass_bind_groups(pass);
      pass.executor(this, frame_data, encoder);
      physical_pass.end();
      frame_data.current_pass = 0;
      encoder.popDebugGroup();

      this._update_transient_resources(pass);
    }
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                         🏗️  PHYSICAL RESOURCE REALIZATION                                ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Realizes a surviving pass, all resources it touches, and its cached GPU state.
   *
   * Resource setup precedes pass creation so attachment descriptors and reflected bindings always
   * point at valid ResourceCache objects.
   */
  _setup_physical_pass_and_resources(pass) {
    const is_compute_pass =
      (pass.pass_config.flags & RenderPassFlags.Compute) !== RenderPassFlags.None;
    const is_graph_local_pass =
      (pass.pass_config.flags & RenderPassFlags.GraphLocal) !== RenderPassFlags.None;

    let pass_attachments = [];

    // Inputs and outputs share physical allocation, but only graphics attachments and pass inputs
    // require additional classification.
    const setup_resource = (resource, resource_params_index, is_input_resource) => {
      if (this.registry.resource_metadata.has(resource)) {
        this._setup_physical_resource(resource, !is_compute_pass, is_input_resource);
        if (!is_compute_pass && !is_graph_local_pass) {
          this._tie_resource_to_pass_config_attachments(
            resource,
            pass,
            resource_params_index,
            is_input_resource,
            pass_attachments
          );
        }
        if (is_input_resource) {
          this._setup_pass_input_resource_bindless_type(resource, pass, resource_params_index);
        }
      }
    };

    pass.parameters.inputs.forEach((input_resource, i) => setup_resource(input_resource, i, true));
    pass.parameters.outputs.forEach((output_resource, i) =>
      setup_resource(output_resource, i, false)
    );

    if (!is_graph_local_pass) {
      pass.physical_id = pass.pass_config.encoded_name;
      const physical_pass = RenderPass.create(pass.pass_config);
      physical_pass.frame_attachments = pass_attachments;

      this._setup_pass_shaders(pass);
      this._setup_pass_bind_groups(pass);
      this._setup_pass_pipeline_state(pass);
    }
  }

  /**
   * Lazily creates physical storage and transfers transient ownership to the deletion queue.
   *
   * Graphics outputs and LocalLoad inputs must survive attachment use, so image persistence can be
   * promoted during realization. Registered resources already carry a nonzero physical ID.
   */
  _setup_physical_resource(resource, is_graphics_pass, is_input_resource) {
    const resource_type = get_graph_resource_type(resource);
    const resource_index = get_graph_resource_index(resource);

    if (resource_type === ResourceType.Buffer) {
      const buffer_resource = this.buffer_resource_allocator.get(resource_index);
      const buffer_metadata = this.registry.resource_metadata.get(resource);
      if (buffer_metadata.physical_id === 0) {
        buffer_metadata.physical_id = buffer_resource.config.encoded_name;
        const buffer = Buffer.create(buffer_resource.config);

        if (!buffer_metadata.b_is_persistent) {
          this.queue_resource_deletion(
            () => {
              buffer.destroy();
            },
            `buffer_${buffer_metadata.physical_id}`,
            buffer_metadata.max_frame_lifetime
          );
        }
      }
    } else if (resource_type === ResourceType.Image) {
      const image_resource = this.image_resource_allocator.get(resource_index);
      const image_metadata = this.registry.resource_metadata.get(resource);
      const is_local_load =
        (image_resource.config.flags & ImageFlags.LocalLoad) !== ImageFlags.None;
      const is_persistent = is_graphics_pass && (!is_input_resource || is_local_load);

      if (image_metadata.physical_id === 0) {
        image_metadata.b_is_persistent |= is_persistent;

        image_metadata.physical_id = image_resource.config.encoded_name;
        const image = Texture.create(image_resource.config);

        if (!image_metadata.b_is_persistent) {
          this.queue_resource_deletion(
            () => {
              image.destroy();
            },
            `image_${image_metadata.physical_id}`,
            image_metadata.max_frame_lifetime
          );
        }
      }
    }
  }

  /**
   * Converts a logical image dependency into color/depth attachment state for a graphics pass.
   *
   * @param {number} resource - Logical image handle.
   * @param {RGPass} pass - Pass being realized.
   * @param {number} resource_params_index - Index in the pass input/output declaration.
   * @param {boolean} is_input_resource - Whether the image is declared as an input.
   * @param {Array<Texture>} pass_attachments - Strong frame references for the physical pass.
   */
  _tie_resource_to_pass_config_attachments(
    resource,
    pass,
    resource_params_index,
    is_input_resource,
    pass_attachments
  ) {
    const resource_type = get_graph_resource_type(resource);
    const resource_index = get_graph_resource_index(resource);

    if (resource_type === ResourceType.Image) {
      const image_resource = this.image_resource_allocator.get(resource_index);
      const image = ResourceCache.get().fetch(
        CacheTypes.IMAGE,
        this.registry.resource_metadata.get(resource).physical_id
      );
      const is_local_load =
        (image_resource.config.flags & ImageFlags.LocalLoad) !== ImageFlags.None;
      if (!is_input_resource || is_local_load) {
        const image_view_index =
          pass.parameters.output_views.length > resource_params_index
            ? pass.parameters.output_views[resource_params_index]
            : 0;
        if (image.config.type.includes("depth")) {
          pass.pass_config.depth_stencil_attachment = {
            image: this.registry.resource_metadata.get(resource).physical_id,
            view_index: image_view_index,
          };
        } else {
          pass.pass_config.attachments.push({
            image: this.registry.resource_metadata.get(resource).physical_id,
            view_index: image_view_index,
          });
        }
      }
      pass_attachments.push(image);
    }
  }

  /**
   * Partitions pass inputs between reflected bindings and the bindless path.
   */
  _setup_pass_input_resource_bindless_type(resource, pass) {
    const resource_type = get_graph_resource_type(resource);
    const resource_index = get_graph_resource_index(resource);

    if (resource_type === ResourceType.Image) {
      const image_resource = this.image_resource_allocator.get(resource_index);
      if (image_resource.config.is_bindless) {
        pass.parameters.bindless_inputs.push(resource);
      } else {
        pass.parameters.pass_inputs.push(resource);
      }
    } else if (resource_type === ResourceType.Buffer) {
      const buffer_resource = this.buffer_resource_allocator.get(resource_index);
      if (buffer_resource.config.b_is_bindless) {
        pass.parameters.bindless_inputs.push(resource);
      } else {
        pass.parameters.pass_inputs.push(resource);
      }
    }
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                          🎨 SHADERS, BINDINGS & PIPELINES                                 ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  /**
   * Resolves the shader stages declared for a compute or graphics pass.
   */
  _setup_pass_shaders(pass) {
    const shader_setup = pass.parameters.shader_setup;
    if (!shader_setup.pipeline_shaders) {
      return;
    }

    const is_compute_pass =
      (pass.pass_config.flags & RenderPassFlags.Compute) !== RenderPassFlags.None;

    if (is_compute_pass && shader_setup.pipeline_shaders.compute) {
      let compute_shader_id = Shader.create(
        shader_setup.pipeline_shaders.compute.path,
        shader_setup.pipeline_shaders.compute.defines,
        shader_setup.force_recreate
      );
      pass.shaders.compute = ResourceCache.get().fetch(CacheTypes.SHADER, compute_shader_id);
    } else {
      if (shader_setup.pipeline_shaders.vertex) {
        let vertex_shader_id = Shader.create(
          shader_setup.pipeline_shaders.vertex.path,
          shader_setup.pipeline_shaders.vertex.defines,
          shader_setup.force_recreate
        );
        pass.shaders.vertex = ResourceCache.get().fetch(CacheTypes.SHADER, vertex_shader_id);
      }
      if (shader_setup.pipeline_shaders.fragment) {
        let fragment_shader_id = Shader.create(
          shader_setup.pipeline_shaders.fragment.path,
          shader_setup.pipeline_shaders.fragment.defines,
          shader_setup.force_recreate
        );
        pass.shaders.fragment = ResourceCache.get().fetch(CacheTypes.SHADER, fragment_shader_id);
      }
    }
  }

  /**
   * Builds the pass bind group from shader reflection and realized logical inputs.
   *
   * Stage visibility is merged by reflected binding name across all active shader stages. Binding
   * order follows pass_inputs, which is populated from the pass's declared non-bindless inputs.
   */
  _setup_pass_bind_groups(pass) {
    const is_compute_pass =
      (pass.pass_config.flags & RenderPassFlags.Compute) !== RenderPassFlags.None;
    const bind_group_cache_key =
      pass.parameters.bind_group_cache_key ?? pass.pass_config.name;

    const pass_binds = this.pass_cache.bind_groups.get(bind_group_cache_key) || {
      bind_groups: Array(this.max_bind_groups).fill(null),
    };
    this.pass_cache.bind_groups.set(bind_group_cache_key, pass_binds);

    this._setup_global_bind_group(pass);

    if (
      pass_binds.bind_groups[BindGroupType.Pass] ||
      pass.parameters.b_skip_pass_bind_group_setup
    ) {
      return;
    }

    // Merge reflection from every active stage before emitting one WebGPU bind group layout.
    let binding_stage_masks = new Map();

    let compute_reflection_groups = is_compute_pass
      ? pass.shaders.compute.reflection.get_bind_groups()
      : [];
    let fragment_reflection_groups = pass.shaders.fragment
      ? pass.shaders.fragment.reflection.get_bind_groups()
      : [];
    let vertex_reflection_groups = pass.shaders.vertex
      ? pass.shaders.vertex.reflection.get_bind_groups()
      : [];
    let reflection_groups = is_compute_pass
      ? compute_reflection_groups
      : pass.shaders.fragment
        ? fragment_reflection_groups
        : vertex_reflection_groups;

    if (compute_reflection_groups[BindGroupType.Pass]) {
      for (let i = 0; i < compute_reflection_groups[BindGroupType.Pass].length; i++) {
        const binding = compute_reflection_groups[BindGroupType.Pass][i];
        if (!binding) continue;
        let binding_stage_mask = binding_stage_masks.get(binding.name) || 0;
        binding_stage_masks.set(binding.name, binding_stage_mask | GPUShaderStage.COMPUTE);
      }
    }

    if (fragment_reflection_groups[BindGroupType.Pass]) {
      for (let i = 0; i < fragment_reflection_groups[BindGroupType.Pass].length; i++) {
        const binding = fragment_reflection_groups[BindGroupType.Pass][i];
        if (!binding) continue;
        let binding_stage_mask = binding_stage_masks.get(binding.name) || 0;
        binding_stage_masks.set(binding.name, binding_stage_mask | GPUShaderStage.FRAGMENT);
      }
    }
    if (vertex_reflection_groups[BindGroupType.Pass]) {
      for (let i = 0; i < vertex_reflection_groups[BindGroupType.Pass].length; i++) {
        const binding = vertex_reflection_groups[BindGroupType.Pass][i];
        if (!binding) continue;
        let binding_stage_mask = binding_stage_masks.get(binding.name) || 0;
        binding_stage_masks.set(binding.name, binding_stage_mask | GPUShaderStage.VERTEX);
      }
    }

    let layouts = [];
    if (BindGroupType.Pass < reflection_groups.length) {
      const pass_group = reflection_groups[BindGroupType.Pass];
      layouts = pass_group.map((binding) => {
        let binding_obj = {
          binding: binding.binding,
          visibility: binding_stage_masks.get(binding.name) || 0,
        };

        const binding_type = Shader.resource_type_from_reflection_type(binding.resourceType);

        if (!pass.parameters.pass_inputs[binding.binding]) {
          throw new Error(
            `Pass ${pass.pass_config.name} input for shader binding ${binding.binding} is null or undefined. Please ensure all required pass inputs are provided.`
          );
        }

        const resource = pass.parameters.pass_inputs[binding.binding];
        const metadata = this.registry.resource_metadata.get(resource);
        const resource_type = get_graph_resource_type(resource);
        let resource_obj = null;

        if (resource_type === ResourceType.Image) {
          resource_obj = ResourceCache.get().fetch(CacheTypes.IMAGE, metadata.physical_id);
        } else {
          resource_obj = ResourceCache.get().fetch(CacheTypes.BUFFER, metadata.physical_id);
        }

        switch (binding_type) {
          case ShaderResourceType.Uniform:
            binding_obj.buffer = {
              type: "uniform",
            };
            break;
          case ShaderResourceType.Storage:
            binding_obj.buffer = {
              type: binding.access === "read" ? "read-only-storage" : "storage",
            };
            break;
          case ShaderResourceType.Texture:
            binding_obj.texture = {
              viewDimension: resource_obj.config.dimension,
              sampleType: Texture.filter_type_from_format(resource_obj.config.format),
            };
            break;
          case ShaderResourceType.StorageTexture:
            binding_obj.storageTexture = {
              access:
                binding.type.access === "write"
                  ? "write-only"
                  : binding.type.access === "read"
                    ? "read-only"
                    : "read-write",
              viewDimension: resource_obj.config.dimension,
              sampleType: Texture.filter_type_from_format(resource_obj.config.format),
              format: resource_obj.config.format || "rgba8unorm",
            };
            break;
          case ShaderResourceType.Sampler:
            binding_obj.sampler = {};
            break;
        }

        return {
          binding: binding.binding,
          visibility: binding_stage_masks.get(binding.name) || 0,
          ...binding_obj,
        };
      });
    }

    let entries = [];
    pass.parameters.pass_inputs.forEach((resource, index) => {
      const metadata = this.registry.resource_metadata.get(resource);
      const resource_type = get_graph_resource_type(resource);
      if (resource_type === ResourceType.Image) {
        const image = ResourceCache.get().fetch(CacheTypes.IMAGE, metadata.physical_id);
        const image_view = image.get_view(pass.parameters.input_views[index]) || image.view;
        const true_image_view =
          image.config.dimension === "cube" ? Texture.default_cube().view : image_view;
        if (!image_view) {
          entries.push({
            binding: index,
            resource: true_image_view,
          });
        } else {
          entries.push({
            binding: index,
            resource: image_view,
          });
        }
      } else {
        const buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, metadata.physical_id);
        entries.push({
          binding: index,
          resource: {
            buffer: buffer.buffer,
            offset: 0,
            size: buffer.config.size,
          },
        });
      }
    });

    if (entries.length > 0) {
      if (entries.length > layouts.length) {
        entries = entries.slice(0, layouts.length);
      }

      const bind_group_name = `${bind_group_cache_key}_bindgroup_${BindGroupType.Pass}`;
      const bind_group_layout_name = `${pass.pass_config.name}_bindgroup_${BindGroupType.Pass}`;
      const pass_bind_group = BindGroup.create_with_layout(
        bind_group_name,
        layouts,
        BindGroupType.Pass,
        entries,
        true, /* Rebuild the named group after dependency or reflection changes. */
        bind_group_layout_name,
        pass.parameters.bind_group_cache_key != null
          ? this.pass_cache_pipeline_states_need_recreate
          : true
      );

      pass_binds.bind_groups[BindGroupType.Pass] = pass_bind_group;
    }
  }

  /**
   * Creates or reuses the compute/render pipeline associated with a pass name.
   *
   * Attachment formats and reflected bind-group layouts are captured only when the cache entry is
   * created; callers must invalidate caches after replacing those dependencies.
   */
  _setup_pass_pipeline_state(pass) {
    if (this.pass_cache.pipeline_states.get(pass.pass_config.name)) {
      pass.pipeline_state_id = this.pass_cache.pipeline_states.get(pass.pass_config.name);
      return;
    }

    if (pass.parameters.b_skip_pass_pipeline_setup) {
      return;
    }

    const bind_group_cache_key =
      pass.parameters.bind_group_cache_key ?? pass.pass_config.name;
    const pass_binds = this.pass_cache.bind_groups.get(bind_group_cache_key);
    const shader_setup = pass.parameters.shader_setup;

    if (shader_setup.pipeline_shaders) {
      const is_compute_pass =
        (pass.pass_config.flags & RenderPassFlags.Compute) !== RenderPassFlags.None;

      if (is_compute_pass) {
        const pipeline_descriptor = {
          label: pass.pass_config.name,
          bind_layouts: pass_binds.bind_groups
            .filter((bind_group) => bind_group !== null)
            .map((bind_group) => bind_group.layout),
          compute: {
            module: pass.shaders.compute.module,
            entryPoint: shader_setup.pipeline_shaders.compute.entry_point || "cs",
          },
          defer_creation: false,
          force: this.pass_cache_pipeline_states_need_recreate,
        };

        pass.pipeline_state_id = pass.pass_config.encoded_name;
        PipelineState.create_compute(pass.pass_config.name, pipeline_descriptor);
      } else {
        const targets = pass.pass_config.attachments
          .filter((attachment) => {
            const image = ResourceCache.get().fetch(CacheTypes.IMAGE, attachment.image);
            return !image.config.type.includes("depth");
          })
          .map((attachment) => {
            const image = ResourceCache.get().fetch(CacheTypes.IMAGE, attachment.image);
            const attachment_desc = {
              format: image.config.format || "bgra8unorm",
            };
            if (shader_setup.attachment_blend) {
              attachment_desc.blend = shader_setup.attachment_blend;
            } else if (image.config.blend) {
              attachment_desc.blend = image.config.blend;
            }
            return attachment_desc;
          });

        let depth_stencil_target = null;
        if (pass.pass_config.depth_stencil_attachment) {
          const image = ResourceCache.get().fetch(
            CacheTypes.IMAGE,
            pass.pass_config.depth_stencil_attachment.image
          );
          depth_stencil_target = {
            depthWriteEnabled: shader_setup.b_depth_write_enabled ?? true,
            depthCompare: shader_setup.depth_stencil_compare_op || "less",
            depthBias: shader_setup.depth_bias || 0,
            depthBiasClamp: shader_setup.depth_bias_clamp || 0,
            depthBiasSlopeScale: shader_setup.depth_slope_scale || 0,
            format: image.config.format || "depth24plus",
          };
        }

        let pipeline_descriptor = {
          label: pass.pass_config.name,
          bind_layouts: pass_binds.bind_groups
            .filter((bind_group) => bind_group !== null)
            .map((bind_group) => bind_group.layout),
          vertex: {
            module: pass.shaders.vertex.module,
            entryPoint: shader_setup.pipeline_shaders.vertex.entry_point || "vs",
            buffers: [], // Vertex pulling is the default; explicit layouts belong here if introduced.
          },
          primitive: {
            topology: shader_setup.primitive_topology_type || "triangle-list",
            cullMode: shader_setup.rasterizer_state?.cull_mode || "back",
          },
          defer_creation: false,
          force: this.pass_cache_pipeline_states_need_recreate,
        };

        if (pass.shaders.fragment) {
          pipeline_descriptor.fragment = {
            module: pass.shaders.fragment.module,
            entryPoint: shader_setup.pipeline_shaders.fragment.entry_point || "fs",
            targets: targets,
          };
        }

        if (depth_stencil_target) {
          pipeline_descriptor.depthStencil = depth_stencil_target;
        }

        pass.pipeline_state_id = pass.pass_config.encoded_name;
        PipelineState.create_render(pass.pass_config.name, pipeline_descriptor);
      }

      this.pass_cache.pipeline_states.set(pass.pass_config.name, pass.pipeline_state_id);
    }
  }

  _reset_pass_cache_bind_groups() {
    if (this.pass_cache_passes_needs_reset) {
      this.pass_cache.bind_groups.keys().forEach((key) => {
        this.pass_cache.bind_groups.delete(key);
      });
      this.pass_cache_passes_needs_reset = false;
    }
  }

  _reset_all_pass_cache_bind_groups() {
    if (this.pass_cache_full_needs_reset) {
      this.pass_cache.bind_groups = new Map();
      this.pass_cache_full_needs_reset = false;
    }
  }

  /**
   * Rebuilds the shared global bind group only when no cached group exists or writes are pending.
   */
  _setup_global_bind_group(pass) {
    const bind_group_cache_key =
      pass.parameters.bind_group_cache_key ?? pass.pass_config.name;
    const pass_binds = this.pass_cache.bind_groups.get(bind_group_cache_key);

    pass_binds.bind_groups[BindGroupType.Global] = this.pass_cache.global_bind_group;

    if (
      pass_binds.bind_groups[BindGroupType.Global] &&
      !this.queued_global_bind_group_writes.length
    ) {
      return;
    }

    let entries = [];
    let layouts = [];

    this.queued_global_bind_group_writes.forEach((write, index) => {
      if (write.buffer) {
        entries.push({
          binding: index,
          resource: {
            buffer: write.buffer.buffer,
            offset: write.offset || 0,
            size: write.size,
          },
        });
        layouts.push({
          binding: index,
          visibility:
            write.visibility ||
            GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
          buffer: {
            type:
              (write.buffer.config.usage & GPUBufferUsage.STORAGE) !== 0
                ? "read-only-storage"
                : "uniform",
          },
        });
      } else if (write.sampler) {
        entries.push({
          binding: index,
          resource: write.sampler.sampler,
        });
        layouts.push({
          binding: index,
          visibility:
            write.visibility ||
            GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
          sampler: {
            type: write.sampler.config.type || "filtering",
          },
        });
      } else if (write.texture_view) {
        entries.push({
          binding: index,
          resource: write.texture_view,
        });

        const tex_layout = {};
        if (write.view_dimension) {
          tex_layout.viewDimension = write.view_dimension;
        }
        if (write.sample_type) {
          tex_layout.sampleType = write.sample_type;
        }
        if (write.multisampled !== undefined) {
          tex_layout.multisampled = write.multisampled;
        }
        layouts.push({
          binding: index,
          visibility:
            write.visibility ||
            GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
          texture: tex_layout,
        });
      }
    });

    this.queued_global_bind_group_writes = [];

    if (entries.length > 0) {
      const global_bind_group = BindGroup.create_with_layout(
        `global_bindgroup_${BindGroupType.Global}`,
        layouts,
        BindGroupType.Global,
        entries,
        true /* Pending writes replace the previous global binding set. */
      );

      this.pass_cache.global_bind_group = global_bind_group;
      pass_binds.bind_groups[BindGroupType.Global] = this.pass_cache.global_bind_group;
    }
  }

  /**
   * Binds cached global/pass groups and pins them on the physical pass for the frame.
   */
  _bind_pass_bind_groups(pass) {
    const physical_pass = ResourceCache.get().fetch(CacheTypes.PASS, pass.physical_id);
    const bind_group_cache_key =
      pass.parameters.bind_group_cache_key ?? pass.pass_config.name;
    const pass_bind_groups = this.pass_cache.bind_groups.get(bind_group_cache_key);

    if (pass_bind_groups.bind_groups.length > 0) {
      this.pass_cache.global_bind_group.bind(physical_pass);
      this.registry.b_global_set_bound = true;

      if (pass_bind_groups.bind_groups.length && pass_bind_groups.bind_groups[BindGroupType.Pass]) {
        pass_bind_groups.bind_groups[BindGroupType.Pass].bind(physical_pass);
      }

      physical_pass.frame_bind_groups[BindGroupType.Global] =
        pass_bind_groups.bind_groups[BindGroupType.Global];
      physical_pass.frame_bind_groups[BindGroupType.Pass] =
        pass_bind_groups.bind_groups[BindGroupType.Pass];
    }
  }

  // ╔════════════════════════════════════════════════════════════════════════════════════════════╗
  // ║                       🧹 TRANSIENT LIFETIME & RETIREMENT                                  ║
  // ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  _update_transient_resources(pass) {
    pass.parameters.inputs.forEach((input_resource) => {
      const resource_meta = this.registry.resource_metadata.get(input_resource);
      if (
        resource_meta.physical_id !== 0 &&
        resource_meta.last_user === pass.handle &&
        !resource_meta.b_is_persistent
      ) {
        // TODO: Return the completed lifetime range to a transient aliasing allocator.
      }
    });

    pass.parameters.outputs.forEach((output_resource) => {
      const resource_meta = this.registry.resource_metadata.get(output_resource);
      if (
        resource_meta.physical_id !== 0 &&
        resource_meta.last_user === pass.handle &&
        !resource_meta.b_is_persistent
      ) {
        // TODO: Return the completed lifetime range to a transient aliasing allocator.
      }
    });
  }

  _free_physical_resources() {
    this.registry.resource_deletion_queue.update();
    // TODO: Release handles tracked in all_bindless_resource_handles when bindless ownership lands.
  }

  /**
   * @param {number} max_bind_groups - Bind-group slots reserved per physical pass.
   * @returns {RenderGraph} New render graph instance.
   */
  static create(max_bind_groups) {
    return new RenderGraph(max_bind_groups);
  }
}

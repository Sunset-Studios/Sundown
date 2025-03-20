/**
 * The RenderGraph class represents a high-level abstraction for managing rendering operations in a graphics application.
 * It provides a flexible and efficient way to organize resources, render passes, and dependencies between them.
 * This graph-based approach allows for automatic resource management, optimization of render order, and efficient GPU utilization.
 * The render graph handles creation and management of images, buffers, and render passes, as well as their lifecycle and dependencies.
 * It also supports features like transient resources, persistent resources, and bindless resources to cater to various rendering needs,
 * though these features are still under active development.
 *
 * ## Usage Example
 *
 * ```javascript
 * // Create and initialize the render graph
 * const renderGraph = new RenderGraph();
 *
 * // Begin a new frame
 * renderGraph.begin();
 *
 * // Create an image resource
 * const imageConfig = {
 *   width: 800,
 *   height: 600,
 *   format: "rgba8unorm",
 *   usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.SAMPLED,
 *   b_is_bindless: false,
 * };
 * const imageHandle = renderGraph.create_image(imageConfig);
 *
 * // Register an existing image resource
 * const existingImage = ...; // Assume this is an existing image resource
 * const registeredImageHandle = renderGraph.register_image(existingImage);
 *
 * // Create a buffer resource
 * const bufferConfig = {
 *   size: 1024,
 *   usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
 *   b_is_bindless: false,
 * };
 * const bufferHandle = renderGraph.create_buffer(bufferConfig);
 *
 * // Register an existing buffer resource
 * const existingBuffer = ...; // Assume this is an existing buffer resource
 * const registeredBufferHandle = renderGraph.register_buffer(existingBuffer);
 *
 * // Add a render pass
 * const passParams = {
 *   shader_setup: {
 *     pipeline_shaders: [],
 *   },
 *   inputs: [imageHandle],
 *   outputs: [registeredImageHandle],
 * };
 * const passIndex = renderGraph.add_pass("MyRenderPass", RenderPassFlags.Present, passParams, (graph, frameData, encoder) => {
 *   // Execute the render pass
 * });
 *
 * // Compile and submit the render graph
 * renderGraph.submit();
 *
 * // Reset the render graph at the end of the frame
 * renderGraph.reset();
 *
 * // Destroy the render graph when done
 * renderGraph.destroy();
 * ```
 */
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
import { read_file } from "../utility/file_system.js";
import { deep_clone } from "../utility/object.js";

const max_image_resources = 128;
const max_buffer_resources = 128;
const max_render_passes = 128;

/**
 * Creates a unique handle for a graph resource.
 * @param {number} index - The index of the resource.
 * @param {number} type - The type of the resource.
 * @param {number} version - The version of the resource.
 * @returns {number} A unique handle for the graph resource.
 */
function create_graph_resource_handle(index, type, version) {
  return (index << 24) | (type << 16) | version;
}

/**
 * Retrieves the index from a graph resource handle.
 * @param {number} handle - The handle of the graph resource.
 * @returns {number} The index of the graph resource.
 */
function get_graph_resource_index(handle) {
  return (handle >> 24) & 0x00ffffff;
}

/**
 * Retrieves the type from a graph resource handle.
 * @param {number} handle - The handle of the graph resource.
 * @returns {number} The type of the graph resource.
 */
function get_graph_resource_type(handle) {
  return (handle >> 16) & 0x000000ff;
}

/**
 * Retrieves the version from a graph resource handle.
 * @param {number} handle - The handle of the graph resource.
 * @returns {number} The version of the graph resource.
 */
function get_graph_resource_version(handle) {
  return handle & 0xffff;
}

/**
 * Checks if a graph resource handle is valid.
 * @param {number} handle - The handle of the graph resource.
 * @returns {boolean} Whether the handle is valid.
 */
function is_valid_graph_resource(handle) {
  return handle >> 32 !== -1;
}

/**
 * Enumeration of resource types in the render graph.
 * @enum {number}
 */
const ResourceType = Object.freeze({
  /** Unknown resource type */
  Unknown: 0,
  /** Image resource type */
  Image: 1,
  /** Buffer resource type */
  Buffer: 2,
});

/**
 * Represents a resource in the render graph.
 * @typedef {Object} RGResource
 * @property {number} handle - The resource handle.
 * @property {Object|null} config - The resource configuration.
 */
const RGResource = Object.freeze({
  handle: 0,
  config: null,
});

/**
 * Metadata for a resource in the render graph.
 * @typedef {Object} RGResourceMetadata
 * @property {number} reference_count - Number of references to the resource.
 * @property {number} physical_id - Physical identifier of the resource.
 * @property {number} first_user - Index of the first pass using the resource.
 * @property {number} last_user - Index of the last pass using the resource.
 * @property {Array} producers - Array of passes producing the resource.
 * @property {Array} consumers - Array of passes consuming the resource.
 * @property {boolean} b_is_persistent - Whether the resource is persistent.
 * @property {boolean} b_is_bindless - Whether the resource is bindless.
 * @property {number} max_frame_lifetime - Maximum frame lifetime of the resource.
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
 * Represents the G-Buffer output data for deferred rendering pipelines in the render graph.
 * @typedef {Object} RGGBufferData
 * @property {Object|null} position - The position texture.
 * @property {Object|null} normal - The normal texture.
 * @property {Object|null} entity_id - The entity ID texture.
 * @property {Object|null} depth - The depth texture.
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
 * Frame-specific data for the render graph.
 * @typedef {Object} RGFrameData
 * @property {number} current_pass - Index of the current pass being processed.
 * @property {Object|null} global_bind_group - The global bind group.
 * @property {Object|null} pass_bind_group - The pass-specific bind group.
 * @property {number} pass_pipeline_state - The current pipeline state for the pass.
 * @property {Object|null} resource_deletion_queue - Queue for resources to be deleted.
 * @property {Array} pass_bindless_resources - Array of bindless resources for the current pass.
 */
const RGFrameData = Object.freeze({
  current_pass: 0,
  global_bind_group: null,
  pass_bind_group: null,
  pass_bind_groups: Array(BindGroupType.Num).fill(null),
  pass_pipeline_state: 0,
  resource_deletion_queue: null,
  pass_bindless_resources: [],
  pass_attachments: [],
  g_buffer_data: null,
});

/**
 * Metadata for a render graph pass.
 * @typedef {Object} RGPassMetadata
 * @property {number} handle - Unique identifier for the pass.
 * @property {number} physical_id - Physical identifier of the pass.
 * @property {number} reference_count - Number of references to the pass.
 * @property {Array} inputs - Array of input resources for the pass.
 * @property {Array} outputs - Array of output resources for the pass.
 * @property {boolean} b_is_culled - Whether the pass is culled from execution.
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
 * Configuration for a render graph pass.
 * @typedef {Object} RGPassConfig
 * @property {string} name - Name of the pass.
 * @property {boolean} b_is_compute - Whether the pass is a compute pass.
 * @property {boolean} b_is_async - Whether the pass can be executed asynchronously.
 * @property {number} execution_queue - Queue on which the pass should be executed.
 */
const RGPassConfig = Object.freeze({
  name: "",
  b_is_compute: false,
  b_is_async: false,
  execution_queue: 0,
});

/**
 * Data for shader setup in a render graph pass.
 * @typedef {Object} RGShaderDataSetup
 * @property {Array} pipeline_shaders - Array of pipeline shaders for the pass.
 * @property {Object|null} push_constant_data - Push constant data for the pass.
 * @property {Object|null} rasterizer_state - Rasterizer state for the pass.
 * @property {Object|null} attachment_blend - Attachment blend state for the pass.
 * @property {string|null} primitive_topology_type - Primitive topology type for the pass.
 * @property {Object|null} viewport - Viewport configuration for the pass.
 * @property {boolean|null} depth_write_enabled - Whether depth writing is enabled for the pass.
 * @property {string|null} depth_stencil_compare_op - Depth/stencil comparison operation for the pass.
 */
const RGShaderDataSetup = Object.freeze({
  // This is auxiliary and only used for passes that only need to bind a single pipeline state to run. Graph passes that don't specify
  // pipeline shaders will have to handle pipeline state creation and/or binding internally within the pass callback.
  pipeline_shaders: null,
  // Same note as above, this is entirely optional data within a graph pass definition
  push_constant_data: null,
  rasterizer_state: null,
  attachment_blend: null,
  primitive_topology_type: null,
  viewport: null,
  depth_write_enabled: null,
  depth_stencil_compare_op: null,
});

/**
 * Configuration for a render graph image resource.
 * @typedef {Object} RGImageConfig
 * @property {string} name - Name of the image.
 * @property {number} width - Width of the image.
 * @property {number} height - Height of the image.
 * @property {number} depth - Depth of the image (for 3D textures) or number of layers (for array textures).
 * @property {number} mip_levels - Number of mip levels in the image.
 * @property {string} format - Format of the image (e.g., "rgba8unorm").
 * @property {number} usage - Usage flags for the image (e.g., GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.SAMPLED).
 * @property {number} sample_count - Number of samples for multisampling.
 * @property {boolean} b_is_bindless - Whether the image is bindless.
 * @property {number} flags - Additional flags for the image (see ImageFlags enum).
 * @property {number} max_frame_lifetime - Maximum frame lifetime of the image if the image is transient.
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
 * Configuration for a render graph buffer resource.
 * @typedef {Object} RGBufferConfig
 * @property {number} size - Size of the buffer in bytes.
 * @property {number} usage - Usage flags for the buffer (e.g., GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST).
 * @property {boolean} b_is_bindless - Whether the buffer is bindless.
 * @property {number} flags - Additional flags for the buffer (see BufferFlags enum).
 * @property {number} max_frame_lifetime - Maximum frame lifetime of the buffer if the buffer is transient.
 */
const RGBufferConfig = Object.freeze({
  size: 0,
  usage: 0,
  b_is_bindless: false,
  flags: BufferFlags.None,
  max_frame_lifetime: 2,
});

/**
 * Configuration for a render graph pass.
 * @typedef {Object} RGPassParameters
 * @property {RGShaderDataSetup} shader_setup - Shader and pipeline setup for the pass.
 * @property {Array} inputs - Input resources/attachments for the pass.
 * @property {Array} outputs - Output resources that this pass writes to or produces.
 * @property {Array} input_views - Array layers of corresponding input entries in the inputs vector (only for image inputs, default is 0 for each input).
 * @property {Array} output_views - Array layers of corresponding output entries in the outputs vector (only for image outputs, default is 0 for each output).
 * @property {Array} pass_inputs - Input resources that should be bound normally (auto-computed).
 * @property {Array} bindless_inputs - Input resources that are bindless and need special handling (auto-computed).
 * @property {boolean} b_skip_pass_bind_group_setup - Whether to skip automatic pass descriptor setup for this pass (global descriptor setup will still run).
 * @property {boolean} b_skip_pass_pipeline_setup - Whether to skip automatic pass pipeline setup for this pass (global pipeline setup will still run).
 * @property {boolean} b_force_keep_pass - Whether to prevent this pass from being culled during render graph compilation.
 */
const RGPassParameters = Object.freeze({
  shader_setup: deep_clone(RGShaderDataSetup),
  inputs: [],
  outputs: [],
  input_views: [],
  output_views: [],
  pass_inputs: [],
  bindless_inputs: [],
  b_skip_pass_bind_group_setup: false,
  b_skip_pass_pipeline_setup: false,
  b_force_keep_pass: false,
});

/**
 * Representation of a render graph pass.
 * @typedef {Object} RGPass
 * @property {number} handle - Unique identifier for the pass.
 * @property {Object} pass_config - Configuration for the pass.
 * @property {RGPassParameters} parameters - Parameters for the pass.
 * @property {Function} executor - Function to execute the pass.
 * @property {Object} shaders - Shaders for the pass.
 * @property {number} physical_id - Physical identifier for the pass.
 * @property {number} pipeline_state_id - Identifier for the pipeline state.
 * @property {number} reference_count - Number of references to this pass.
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
 * Registry for render graph resources. All render graph registry resources (aside from render passes) should be transient and therefore do not need serious caching.
 * For this reason the pass cache is the only thing we don't clear out per-frame. Any resource that need to survive multiple frames
 * should be allocated externally and registered to the render graph as external resources.
 * @typedef {Object} RGRegistry
 * @property {Array} render_passes - Array of render passes.
 * @property {Map} pass_order_map - Map of pass order.
 * @property {Array} all_resource_handles - Array of all resource handles.
 * @property {Map} resource_metadata - Map of resource metadata.
 * @property {Array} all_bindless_resource_handles - Array of all bindless resource handles.
 * @property {ExecutionQueue} resource_deletion_queue - Queue for resource deletion.
 * @property {boolean} b_global_bind_group_bound - Whether the global bind group is bound.
 */
const RGRegistry = Object.freeze({
  current_scene_id: "",
  render_passes: [],
  pass_order_map: new Map(),
  all_resource_handles: new StaticIntArray(max_image_resources + max_buffer_resources),
  resource_metadata: new Map(),
  all_bindless_resource_handles: [],
  resource_deletion_queue: new ExecutionQueue(),
  b_global_bind_group_bound: false,
});

/**
 * Cache for render pass-related objects.
 * @typedef {Object} PassCache
 * @property {Object} global_bind_group - Global bind group object.
 * @property {Map} bind_groups - Map of bind groups.
 * @property {Map} pipeline_states - Map of pipeline states.
 */
const PassCache = Object.freeze({
  global_bind_group: null,
  bind_groups: new Map(),
  pipeline_states: new Map(),
});

const CustomPassOrderReadyFlag = 1 << 0;
const DefaultPassOrderReadyFlag = 1 << 1;

/**
 * Represents the stored pass order.
 * @typedef {Object} StoredPassOrder
 * @property {Array} default - The default pass order.
 * @property {Array} custom - The custom pass order.
 * @property {number} ready_flags - The ready flags for the pass order.
 */
const StoredPassOrder = Object.freeze({
  default: [],
  custom: [],
  ready_flags: 0,
});

/**
 * A render graph is used to organize rendering operations in a graphics application.
 * This API provides a comprehensive set of functions to manage resources, pipeline states, and bind groups for a render graph.
 * The RenderGraph class is the core component that manages the entire render graph, including resource creation and registration,
 * render pass management, graph compilation, and rendering/compute command submission.
 */
export class RenderGraph {
  constructor(max_bind_groups) {
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

    this.image_resource_allocator = new FrameAllocator(max_image_resources, deep_clone(RGResource));
    this.buffer_resource_allocator = new FrameAllocator(max_buffer_resources, deep_clone(RGResource));
    this.render_pass_allocator = new FrameAllocator(max_render_passes, deep_clone(RGPass));

    this.resource_metadata_allocator = new FrameAllocator(
      max_image_resources + max_buffer_resources,
      deep_clone(RGResourceMetadata)
    );

    this._execute_post_render_callbacks = this._execute_post_render_callbacks.bind(this);
    this._execute_pre_render_callbacks = this._execute_pre_render_callbacks.bind(this);

    this._init_pass_order_info();
  }

  /**
   * Resets the render graph, clearing all resources and render passes.
   * This method should be called at the end of each frame to prepare for the next frame.
   *
   * @example
   * // At the end of each frame
   * const renderGraph = new RenderGraph();
   * // ... (rendering operations)
   * renderGraph.reset();
   * // The render graph is now ready for the next frame
   */
  destroy() {
    this.reset();
    this.registry.resource_deletion_queue.flush();
  }

  /**
   * Begins a new frame in the render graph.
   * This method resets the render graph state and prepares it for a new frame of rendering.
   *
   * @returns {void}
   *
   * @example
   * const renderGraph = new RenderGraph();
   * renderGraph.begin();
   * // Add passes and resources...
   * renderGraph.submit();
   */
  begin() {
    this._execute_pre_render_callbacks();
    this.reset();
    this._add_queued_pre_commands();
  }

  /**
   * Creates a new image resource in the render graph.
   *
   * @param {Object} config - The configuration object for the image resource.
   * @param {number} config.width - The width of the image.
   * @param {number} config.height - The height of the image.
   * @param {string} config.format - The format of the image (e.g., "rgba8unorm").
   * @param {number} config.usage - The usage flags for the image.
   * @param {boolean} config.b_is_bindless - Whether the image is bindless.
   * @returns {number} The handle of the newly created image resource.
   *
   * @example
   * const imageConfig = {
   *   width: 1920,
   *   height: 1080,
   *   format: "rgba8unorm",
   *   usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.SAMPLED,
   *   b_is_bindless: false
   * };
   * const imageHandle = renderGraph.create_image(imageConfig);
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

    const resource_metadata = this.registry.resource_metadata.get(new_resource.handle);
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
   * Registers an existing image resource in the render graph.
   * This method creates a new resource handle for an existing image and sets up its metadata.
   *
   * @param {string|number} image - The identifier or hash of the image to register.
   * @returns {number} The handle of the newly registered image resource.
   *
   * @throws {Error} If the image is not found in the ResourceCache.
   *
   * @example
   * const imageHandle = renderGraph.register_image("myImage");
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

    const resource_metadata = this.registry.resource_metadata.get(new_resource.handle);
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
   * Creates a new buffer resource in the render graph.
   * This method allocates a new buffer resource, sets up its configuration, and registers it in the graph.
   *
   * @param {Object} config - The configuration object for the buffer.
   * @param {number} config.size - The size of the buffer in bytes.
   * @param {number} config.usage - The usage flags for the buffer (e.g., GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST).
   * @param {boolean} [config.b_is_bindless=false] - Whether the buffer should be bindless.
   * @returns {number} The handle of the newly created buffer resource.
   *
   * @example
   * const bufferConfig = {
   *   size: 1024,
   *   usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
   *   b_is_bindless: false
   * };
   * const bufferHandle = renderGraph.create_buffer(bufferConfig);
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

    const resource_metadata = this.registry.resource_metadata.get(new_resource.handle);
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
   * Registers an existing buffer in the render graph.
   * This method creates a new resource handle for an existing buffer, sets up its configuration,
   * and registers it in the graph as a persistent resource.
   *
   * @param {Object} buffer - The existing buffer object to register.
   * @returns {number} The handle of the newly registered buffer resource.
   *
   * @example
   * const existingBuffer = ...; // Assume this is an existing buffer object
   * const bufferHandle = renderGraph.register_buffer(existingBuffer);
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

    const resource_metadata = this.registry.resource_metadata.get(new_resource.handle);
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
   * Adds a new render pass to the render graph.
   *
   * @param {string} name - The name of the render pass.
   * @param {number} pass_type - The type of the render pass, using flags from RenderPassFlags.
   * @param {Object|null} params - The parameters for the render pass, including inputs and outputs.
   * @param {Function} execution_callback - The callback function to execute the render pass.
   * @returns {number} The index of the newly added render pass.
   *
   * @example
   * const passIndex = renderGraph.add_pass(
   *   "MyRenderPass",
   *   RenderPassFlags.Present,
   *   { inputs: [inputHandle], outputs: [outputHandle] },
   *   (graph, frameData, encoder) => {
   *     // Render pass execution logic
   *   }
   * );
   */
  add_pass(name, pass_type, params, execution_callback) {
    let index;

    const pass = this.render_pass_allocator.allocate();
    pass.pass_config = {
      name: name,
      encoded_name: Name.from(name),
      flags: pass_type,
      attachments: [],
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
   * Retrieves a physical render pass from the resource cache using its handle.
   *
   * @param {number} handle - The handle of the render pass to retrieve.
   * @returns {Object|null} The physical render pass object if found, or null if not found.
   */
  get_physical_pass(handle) {
    return ResourceCache.get().fetch(CacheTypes.PASS, handle);
  }

  /**
   * Retrieves a physical image from the resource cache using its handle.
   *
   * @param {number} handle - The handle of the image to retrieve.
   * @returns {Object|null} The physical image object if found, or null if not found.
   */
  get_physical_image(handle) {
    return ResourceCache.get().fetch(
      CacheTypes.IMAGE,
      this.registry.resource_metadata.get(handle).physical_id
    );
  }

  /**
   * Retrieves a physical buffer from the resource cache using its handle.
   *
   * @param {number} handle - The handle of the buffer to retrieve.
   * @returns {Object|null} The physical buffer object if found, or null if not found.
   */
  get_physical_buffer(handle) {
    return ResourceCache.get().fetch(
      CacheTypes.BUFFER,
      this.registry.resource_metadata.get(handle).physical_id
    );
  }

  /**
   * Queues global bind group writes to be processed later in the render graph.
   *
   * @param {Array} writes - An array of write operations to be queued.
   * @returns {void}
   *
   * @example
   * const writes = [
   *   { buffer: someBuffer, data: new Float32Array([1, 2, 3, 4]), offset: 0 },
   *   { buffer: anotherBuffer, data: new Uint8Array([255, 128, 0]), offset: 16 }
   * ];
   * renderGraph.queue_global_bind_group_write(writes);
   */
  queue_global_bind_group_write(writes, overwrite = false) {
    if (overwrite) {
      this.queued_global_bind_group_writes = writes;
    } else {
      this.queued_global_bind_group_writes = [...this.queued_global_bind_group_writes, ...writes];
    }
  }

  /**
   * Queues a set of commands to be executed before all other passes in the render graph.
   *
   * @param {string} name - A descriptive name for the set of commands.
   * @param {Function} commands_callback - A callback function that will be executed to perform the commands.
   * @returns {void}
   *
   * @example
   * renderGraph.queue_commands('Draw UI', (encoder) => {
   *   // Draw UI elements
   *   encoder.drawUI();
   * });
   */
  queue_pre_commands(name, commands_callback) {
    this.queued_pre_commands.push({ name, commands_callback });
  }

  /**
   * Sets the scene ID for the render graph, used to set and get the current pass order.
   *
   * @param {string} scene_id - The ID of the scene to set.
   * @returns {void}
   */
  set_scene_id(scene_id) {
    this.registry.current_scene_id = scene_id;
  }

  _add_queued_pre_commands() {
    for (const command of this.queued_pre_commands) {
      this.add_pass(command.name, RenderPassFlags.GraphLocal, {}, command.commands_callback);
    }
    this.queued_pre_commands.length = 0;
  }

  /**
   * Queues a set of commands to be executed after all other passes in the render graph.
   *
   * @param {string} name - A descriptive name for the set of commands.
   * @param {Function} commands_callback - A callback function that will be executed to perform the commands.
   * @returns {void}
   *
   * @example
   * renderGraph.queue_post_commands('Draw UI', (encoder) => {
   *   // Draw UI elements
   *   encoder.drawUI();
   * });
   */
  queue_post_commands(name, commands_callback) {
    this.queued_post_commands.push({ name, commands_callback });
  }

  _add_queued_post_commands() {
    for (const command of this.queued_post_commands) {
      this.add_pass(command.name, RenderPassFlags.GraphLocal, {}, command.commands_callback);
    }
    this.queued_post_commands.length = 0;
  }

  _init_pass_order_info() {
    const config_file = read_file("config/renderer.config.json");
    if (config_file) {
      const config = JSON.parse(config_file);
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

  _compile() {
    this._cull_graph_passes();
    this._sort_graph_passes();
    this._compute_resource_first_and_last_users();
  }

  _cull_graph_passes() {
    const passes_to_cull = new Set();
    const unused_stack = [];

    // Lambda to cull a producer pass when it is no longer referenced and pop its inputs onto the unused stack
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

    // Go through all resources and push any resources that are not referenced onto the unused stack
    for (let i = 0; i < this.registry.all_resource_handles.length; i++) {
      const resource = this.registry.all_resource_handles.get(i);
      if (this.registry.resource_metadata.has(resource)) {
        if (this.registry.resource_metadata.get(resource).reference_count === 0) {
          unused_stack.push(resource);
        }
      }
    }

    // Keep processing unused resources and updating their producer pass ref counts
    while (unused_stack.length > 0) {
      const unused_resource = unused_stack.pop();
      if (this.registry.resource_metadata.has(unused_resource)) {
        decrement_producer_and_subresource_ref_counts(
          this.registry.resource_metadata.get(unused_resource).producers
        );
      }
    }

    // Take all the culled passes and remove them from our primary nonculled_passes array.
    // This may constantly shuffle array items if a lot of passes are culled so maybe think about
    // using a flag instead to indicate if a pass is active or not.
    for (let i = this.non_culled_passes.length - 1; i >= 0; --i) {
      const pass = this.non_culled_passes[i];
      if (passes_to_cull.has(pass)) {
        this.non_culled_passes.splice(i, 1);
      }
    }
  }

  _sort_graph_passes() {
    // Skip if current_pass_order is empty
    const current_pass_order = this.stored_pass_order.custom[this.registry.current_scene_id] || [];
    if (!current_pass_order || current_pass_order.length === 0) {
      return;
    }

    // Sort non_culled_passes based on the order in current_pass_order
    this.non_culled_passes.sort((a, b) => {
      const pass_a = this.registry.render_passes[a];
      const pass_b = this.registry.render_passes[b];

      // Get the encoded IDs for the pass names
      const id_a = pass_a.pass_config.name;
      const id_b = pass_b.pass_config.name;

      // Get their positions from the order map
      const order_a = this.registry.pass_order_map.has(id_a)
        ? this.registry.pass_order_map.get(id_a)
        : Number.MAX_SAFE_INTEGER;
      const order_b = this.registry.pass_order_map.has(id_b)
        ? this.registry.pass_order_map.get(id_b)
        : Number.MAX_SAFE_INTEGER;

      // Sort based on position
      return order_a === order_b ? a - b : order_a - order_b;
    });
  }

  _compute_resource_first_and_last_users() {
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

  /**
   * Marks the pass cache bind groups as dirty, indicating that they need to be reset.
   * This function is used to manage the state of bind groups in the render graph,
   * ensuring that they are properly updated when necessary. (i.e. when the bind group resources are re-created)
   *
   * @param {boolean} [pass_only=true] - If true, only the pass-specific bind groups
   * will be marked for reset. If false, all bind groups including the global ones
   * will be marked for reset.
   *
   * @example
   * // Mark only pass-specific bind groups as dirty
   * renderGraph.mark_pass_cache_bind_groups_dirty(true);
   *
   * // Mark all bind groups (including global) as dirty
   * renderGraph.mark_pass_cache_bind_groups_dirty();
   */
  mark_pass_cache_bind_groups_dirty(pass_only = false) {
    if (pass_only) {
      this.pass_cache_passes_needs_reset = true;
    } else {
      this.pass_cache_full_needs_reset = true;
    }
  }

  /**
   * Adds a callback to be executed before the render graph is submitted.
   *
   * @param {Function} callback - The callback function to be executed.
   * @returns {void}
   */
  on_pre_render(callback) {
    this.pre_render_callbacks.push(callback);
  }

  /**
   * Removes a callback from the pre-render callbacks list.
   *
   * @param {Function} callback - The callback function to be removed.
   * @returns {void}
   */
  remove_pre_render(callback) {
    const index = this.pre_render_callbacks.indexOf(callback);
    if (index !== -1) {
      this.pre_render_callbacks.splice(index, 1);
    }
  }

  /**
   * Submits the compiled render graph for execution.
   * This method compiles the render graph, creates a command encoder, and executes all non-culled passes.
   * It then submits the encoded commands to the GPU for rendering.
   *
   * @async
   * @returns {Promise<void>} A promise that resolves when all passes have been executed and submitted.
   * @throws {Error} If there's an error during pass execution or command submission.
   */
  submit() {
    profile_scope("RenderGraph.submit", () => {
      this._reset_pass_cache_bind_groups();
      this._add_queued_post_commands();
      this._compile();

      if (this.non_culled_passes.length === 0) {
        return;
      }

      const encoder = CommandQueue.create_encoder("render_graph_encoder");

      const frame_data = deep_clone(RGFrameData);
      frame_data.resource_deletion_queue = this.registry.resource_deletion_queue;

      for (let i = 0; i < this.non_culled_passes.length; i++) {
        const pass_handle = this.non_culled_passes[i];
        this._execute_pass(this.registry.render_passes[pass_handle], frame_data, encoder);
      }

      CommandQueue.submit(encoder, this._execute_post_render_callbacks);
    });
  }

  /**
   * Adds a callback to be executed after the render graph is submitted.
   *
   * @param {Function} callback - The callback function to be executed.
   * @returns {void}
   */
  on_post_render(callback) {
    this.post_render_callbacks.push(callback);
  }

  /**
   * Removes a callback from the post-render callbacks list.
   *
   * @param {Function} callback - The callback function to be removed.
   * @returns {void}
   */
  remove_post_render(callback) {
    const index = this.post_render_callbacks.indexOf(callback);
    if (index !== -1) {
      this.post_render_callbacks.splice(index, 1);
    }
  }

  /**
   * Resets the render graph, clearing all resources and render passes.
   * This method should be called at the end of each frame to prepare for the next frame.
   *
   * @returns {void}
   */
  reset() {
    this._free_physical_resources();

    this.non_culled_passes.length = 0;

    this.registry.render_passes.length = 0;
    this.registry.resource_metadata.clear();

    this.registry.all_resource_handles.reset();
    this.image_resource_allocator.reset();
    this.buffer_resource_allocator.reset();
    this.render_pass_allocator.reset();
    this.resource_metadata_allocator.reset();
  }

  /**
   * Records the default pass order in the configuration database.
   * This method is used to save the default pass order for future reference.
   *
   * @returns {void}
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
   * Records the current pass order in the configuration database for the specified scene.
   * This method is used to save the pass order for future reference.
   *
   * @returns {void}
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
   * Checks if the default pass order is ready.
   *
   * @returns {boolean} True if the default pass order is ready, false otherwise.
   */
  is_default_pass_order_ready() {
    return (this.stored_pass_order.ready_flags & DefaultPassOrderReadyFlag) !== 0;
  }

  /**
   * Checks if the custom pass order is ready.
   *
   * @returns {boolean} True if the custom pass order is ready, false otherwise.
   */
  is_custom_pass_order_ready() {
    return (this.stored_pass_order.ready_flags & CustomPassOrderReadyFlag) !== 0;
  }

  /**
   * Sets the default pass order for the render graph.
   *
   * @param {Array} value - The pass order to set for the default scene
   * @returns {void}
   */
  set_default_pass_order(value, scene_id = null) {
    const scene = scene_id ?? this.registry.current_scene_id;
    this.stored_pass_order.default[scene] = value;
  }

  /**
   * Returns the default pass order for the render graph.
   *
   * @param {string|number} scene_id - The ID of the scene to get the default pass order for
   * @returns {Array} The default pass order for the specified scene
   */
  get_default_pass_order(scene_id = null) {
    const scene = scene_id ?? this.registry.current_scene_id;
    return this.stored_pass_order.default[scene] || [];
  }

  /**
   * Sets the pass order for a specific scene.
   *
   * @param {Array} value - The pass order to set for the specified scene
   * @param {string|number} scene_id - The ID of the scene to set the pass order for
   * @returns {void}
   */
  set_scene_pass_order(value, scene_id = null) {
    const scene = scene_id ?? this.registry.current_scene_id;
    this.stored_pass_order.custom[scene] = value;
    if (scene === this.registry.current_scene_id) {
      this._update_pass_order_map();
    }
  }

  /**
   * Returns the pass order for a specific scene.
   *
   * @param {string|number} scene_id - The ID of the scene to get the pass order for
   * @returns {Array} The pass order for the specified scene
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

  async _execute_post_render_callbacks() {
    for (let i = 0; i < this.post_render_callbacks.length; i++) {
      await this.post_render_callbacks[i]();
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

    frame_data.pass_attachments.length = 0;

    this._setup_physical_pass_and_resources(pass, frame_data, encoder);

    if ((pass.pass_config.flags & RenderPassFlags.GraphLocal) !== RenderPassFlags.None) {
      pass.executor(this, frame_data, encoder);
    } else {
      const physical_pass = ResourceCache.get().fetch(CacheTypes.PASS, pass.physical_id);

      if (!physical_pass) {
        throw new Error("Physical pass is null");
      }

      const pipeline = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, pass.pipeline_state_id);

      frame_data.current_pass = pass.physical_id;

      encoder.pushDebugGroup(pass.pass_config.name);
      physical_pass.begin(encoder, pipeline);
      this._bind_pass_bind_groups(pass, frame_data);
      pass.executor(this, frame_data, encoder);
      physical_pass.end();
      encoder.popDebugGroup();

      this._update_transient_resources(pass);
    }
  }

  _setup_physical_pass_and_resources(pass, frame_data, encoder) {
    const is_compute_pass =
      (pass.pass_config.flags & RenderPassFlags.Compute) !== RenderPassFlags.None;
    const is_graph_local_pass =
      (pass.pass_config.flags & RenderPassFlags.GraphLocal) !== RenderPassFlags.None;

    const setup_resource = (resource, resource_params_index, is_input_resource) => {
      if (this.registry.resource_metadata.has(resource)) {
        this._setup_physical_resource(resource, !is_compute_pass, is_input_resource);
        if (!is_compute_pass && !is_graph_local_pass) {
          this._tie_resource_to_pass_config_attachments(
            resource,
            pass,
            resource_params_index,
            is_input_resource,
            frame_data
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
      RenderPass.create(pass.pass_config);

      this._setup_pass_shaders(pass, frame_data);
      this._setup_pass_bind_groups(pass, frame_data);
      this._setup_pass_pipeline_state(pass, frame_data);
    }
  }

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
          this.registry.resource_deletion_queue.remove_execution(buffer_metadata.physical_id);
          this.registry.resource_deletion_queue.push_execution(
            () => {
              buffer.destroy();
            },
            buffer_metadata.physical_id,
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
          this.registry.resource_deletion_queue.remove_execution(image_metadata.physical_id);
          this.registry.resource_deletion_queue.push_execution(
            () => {
              image.destroy();
            },
            image_metadata.physical_id,
            image_metadata.max_frame_lifetime
          );
        }
      }
    }
  }

  _tie_resource_to_pass_config_attachments(
    resource,
    pass,
    resource_params_index,
    is_input_resource,
    frame_data
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
      frame_data.pass_attachments.push(image);
    }
  }

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

  _setup_pass_shaders(pass, frame_data) {
    const shader_setup = pass.parameters.shader_setup;
    if (!shader_setup.pipeline_shaders) {
      return;
    }

    const is_compute_pass =
      (pass.pass_config.flags & RenderPassFlags.Compute) !== RenderPassFlags.None;

    if (is_compute_pass && shader_setup.pipeline_shaders.compute) {
      pass.shaders.compute = Shader.create(shader_setup.pipeline_shaders.compute.path);
    } else {
      if (shader_setup.pipeline_shaders.vertex) {
        pass.shaders.vertex = Shader.create(shader_setup.pipeline_shaders.vertex.path);
      }
      if (shader_setup.pipeline_shaders.fragment) {
        pass.shaders.fragment = Shader.create(shader_setup.pipeline_shaders.fragment.path);
      }
    }
  }

  _setup_pass_bind_groups(pass, frame_data) {
    const is_compute_pass =
      (pass.pass_config.flags & RenderPassFlags.Compute) !== RenderPassFlags.None;

    const pass_binds = this.pass_cache.bind_groups.get(pass.pass_config.name) || {
      bind_groups: Array(this.max_bind_groups).fill(null),
    };
    this.pass_cache.bind_groups.set(pass.pass_config.name, pass_binds);

    this._setup_global_bind_group(pass, frame_data);

    if (pass_binds.bind_groups[BindGroupType.Pass] || pass.parameters.b_skip_pass_bind_group_setup) {
      return;
    }

    // Setup pass-specific bind group
    let layouts = [];
    let reflection_groups = [];
    if (is_compute_pass) {
      reflection_groups = pass.shaders.compute.reflection.getBindGroups();
    } else if (pass.shaders.fragment){
      const fragment_group = pass.shaders.fragment.reflection.getBindGroups();
      for (let i = 0; i < BindGroupType.Num; i++) {
        if (fragment_group[i]) {
          reflection_groups.push(fragment_group[i]);
        }
      }
    } else {
      const vertex_group = pass.shaders.vertex.reflection.getBindGroups();
      for (let i = 0; i < BindGroupType.Num; i++) {
        if (vertex_group[i]) {
          reflection_groups.push(vertex_group[i]);
        }
      }
    }

    if (BindGroupType.Pass < reflection_groups.length) {
      const pass_group = reflection_groups[BindGroupType.Pass];
      layouts = pass_group.map((binding) => {
        let binding_obj = {
          binding: binding.binding,
          visibility: is_compute_pass
            ? GPUShaderStage.COMPUTE
            : GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
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
          visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
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
        const true_image_view = image.config.dimension === "cube" ? Texture.default_cube().view : image_view;
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

      const pass_bind_group = BindGroup.create_with_layout(
        `${pass.pass_config.name}_bindgroup_${BindGroupType.Pass}`,
        layouts,
        BindGroupType.Pass,
        entries,
        true /* force */
      );

      pass_binds.bind_groups[BindGroupType.Pass] = pass_bind_group;
    }
  }

  _setup_pass_pipeline_state(pass, frame_data) {
    if (this.pass_cache.pipeline_states.get(pass.pass_config.name)) {
      pass.pipeline_state_id = this.pass_cache.pipeline_states.get(pass.pass_config.name);
      frame_data.pass_pipeline_state = pass.pipeline_state_id;
      return;
    }

    if (pass.parameters.b_skip_pass_pipeline_setup) {
      return;
    }

    const pass_binds = this.pass_cache.bind_groups.get(pass.pass_config.name);
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
        };

        pass.pipeline_state_id = pass.pass_config.encoded_name;
        const pipeline = PipelineState.create_compute(pass.pass_config.name, pipeline_descriptor);
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
            buffers: [], // Add vertex buffer layouts if needed
          },
          primitive: {
            topology: shader_setup.primitive_topology_type || "triangle-list",
            cullMode: shader_setup.rasterizer_state?.cull_mode || "back",
          },
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
        const pipeline = PipelineState.create_render(pass.pass_config.name, pipeline_descriptor);
      }

      this.pass_cache.pipeline_states.set(pass.pass_config.name, pass.pipeline_state_id);

      frame_data.pass_pipeline_state = pass.pipeline_state_id;
    }
  }

  _reset_pass_cache_bind_groups() {
    if (this.pass_cache_passes_needs_reset) {
      this.pass_cache.bind_groups.keys().forEach((key) => {
        this.pass_cache.bind_groups.delete(key);
      });
    } else if (this.pass_cache_full_needs_reset) {
      this.pass_cache.bind_groups = new Map();
    }
    this.pass_cache_passes_needs_reset = false;
    this.pass_cache_full_needs_reset = false;
  }

  _setup_global_bind_group(pass, frame_data) {
    const pass_binds = this.pass_cache.bind_groups.get(pass.pass_config.name);

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
            type: write.sampler.config.mag_filter === "nearest" ? "non-filtering" : "filtering",
          },
        });
      } else if (write.texture_view) {
        entries.push({
          binding: index,
          resource: write.texture_view,
        });
        layouts.push({
          binding: index,
          visibility:
            write.visibility ||
            GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
          texture: {},
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
        true /* force */
      );

      this.pass_cache.global_bind_group = global_bind_group;
      pass_binds.bind_groups[BindGroupType.Global] = this.pass_cache.global_bind_group;
    }
  }

  _bind_pass_bind_groups(pass, frame_data) {
    const physical_pass = ResourceCache.get().fetch(CacheTypes.PASS, pass.physical_id);
    const pass_bind_groups = this.pass_cache.bind_groups.get(pass.pass_config.name);

    if (pass_bind_groups.bind_groups.length > 0) {
      this.pass_cache.global_bind_group.bind(physical_pass);
      this.registry.b_global_set_bound = true;

      if (pass_bind_groups.bind_groups.length && pass_bind_groups.bind_groups[BindGroupType.Pass]) {
        pass_bind_groups.bind_groups[BindGroupType.Pass].bind(physical_pass);
      }

      frame_data.pass_bind_groups[BindGroupType.Global] =
        pass_bind_groups.bind_groups[BindGroupType.Global];
      frame_data.pass_bind_groups[BindGroupType.Pass] =
        pass_bind_groups.bind_groups[BindGroupType.Pass];
    }
  }

  _update_transient_resources(pass) {
    pass.parameters.inputs.forEach((input_resource) => {
      const resource_meta = this.registry.resource_metadata.get(input_resource);
      if (
        resource_meta.physical_id !== 0 &&
        resource_meta.last_user === pass.handle &&
        !resource_meta.b_is_persistent
      ) {
        // TODO: Transient resource memory aliasing
      }
    });

    pass.parameters.outputs.forEach((output_resource) => {
      const resource_meta = this.registry.resource_metadata.get(output_resource);
      if (
        resource_meta.physical_id !== 0 &&
        resource_meta.last_user === pass.handle &&
        !resource_meta.b_is_persistent
      ) {
        // TODO: Transient resource memory aliasing
      }
    });
  }

  _free_physical_resources() {
    this.registry.resource_deletion_queue.update();
    // TODO: Free all bindless resources handles stored in this.registry.all_bindless_resource_handles
  }

  static create(max_bind_groups) {
    return new RenderGraph(max_bind_groups);
  }
}

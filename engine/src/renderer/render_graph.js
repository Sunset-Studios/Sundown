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
 * renderGraph.initialize();
 *
 * // Begin a new frame
 * const context = ...; // Assume this is the rendering context
 * renderGraph.begin(context);
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
 * const passIndex = renderGraph.add_pass(context, "MyRenderPass", RenderPassFlags.Present, passParams, (graph, frameData, encoder) => {
 *   // Execute the render pass
 * });
 *
 * // Compile and submit the render graph
 * renderGraph.submit(context);
 *
 * // Reset the render graph at the end of the frame
 * renderGraph.reset(context);
 *
 * // Destroy the render graph when done
 * renderGraph.destroy(context);
 * ```
 */

import ExecutionQueue from "@/utility/execution_queue.js";
import { FrameAllocator } from "@/memory/allocator.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";
import { BindGroupType, BindGroup } from "@/renderer/bind_group.js";
import { RenderPassType, RenderPass, RenderPassFlags } from "@/renderer/render_pass.js";
import { PipelineState } from "@/renderer/pipeline_state.js";
import { CommandQueue } from "@/renderer/command_queue.js";
import { Buffer } from "@/renderer/buffer.js";
import { Shader } from "@/renderer/shader.js";
import Name from "@/utility/names.js";
import _ from "lodash";

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
  return (handle >> 24) & 0x00FFFFFF;
}

/**
 * Retrieves the type from a graph resource handle.
 * @param {number} handle - The handle of the graph resource.
 * @returns {number} The type of the graph resource.
 */
function get_graph_resource_type(handle) {
  return (handle >> 16) & 0x000000FF;
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
 * Flags for image resources in the render graph.
 * @enum {number}
 */
const ImageFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a transient image resource */
  Transient: 1,
  /** Indicates the image is loaded locally */
  LocalLoad: 2,
});

/**
 * Flags for buffer resources in the render graph.
 * @enum {number}
 */
const BufferFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a transient buffer resource */
  Transient: 1,
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
 * Frame-specific data for the render graph.
 * @typedef {Object} RGFrameData
 * @property {number} current_pass - Index of the current pass being processed.
 * @property {Object|null} context - The rendering context.
 * @property {Object|null} global_bind_group - The global bind group.
 * @property {Object|null} pass_bind_group - The pass-specific bind group.
 * @property {number} pass_pipeline_state - The current pipeline state for the pass.
 * @property {Object|null} resource_deletion_queue - Queue for resources to be deleted.
 * @property {Array} pass_bindless_resources - Array of bindless resources for the current pass.
 */
const RGFrameData = Object.freeze({
  current_pass: 0,
  context: null,
  global_bind_group: null,
  pass_bind_group: null,
  pass_pipeline_state: 0,
  resource_deletion_queue: null,
  pass_bindless_resources: [],
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
  pipeline_shaders: [],
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
 * @property {number} depth - Depth of the image (for 3D textures).
 * @property {number} array_layers - Number of array layers in the image.
 * @property {number} mip_levels - Number of mip levels in the image.
 * @property {string} format - Format of the image (e.g., "rgba8unorm").
 * @property {number} usage - Usage flags for the image (e.g., GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.SAMPLED).
 * @property {number} sample_count - Number of samples for multisampling.
 * @property {boolean} b_is_bindless - Whether the image is bindless.
 * @property {number} flags - Additional flags for the image (see ImageFlags enum).
 */
const RGImageConfig = Object.freeze({
  name: "",
  width: 0,
  height: 0,
  depth: 1,
  array_layers: 1,
  mip_levels: 1,
  format: "",
  usage: 0,
  sample_count: 1,
  b_is_bindless: false,
  flags: ImageFlags.None,
});

/**
 * Configuration for a render graph buffer resource.
 * @typedef {Object} RGBufferConfig
 * @property {number} size - Size of the buffer in bytes.
 * @property {number} usage - Usage flags for the buffer (e.g., GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST).
 * @property {boolean} b_is_bindless - Whether the buffer is bindless.
 * @property {number} flags - Additional flags for the buffer (see BufferFlags enum).
 */
const RGBufferConfig = Object.freeze({
  size: 0,
  usage: 0,
  b_is_bindless: false,
  flags: BufferFlags.None,
});

/**
 * Configuration for a render graph pass.
 * @typedef {Object} RGPassParameters
 * @property {RGShaderDataSetup} shader_setup - Shader and pipeline setup for the pass.
 * @property {Array} inputs - Input resources/attachments for the pass.
 * @property {Array} outputs - Output resources that this pass writes to or produces.
 * @property {Array} output_views - Array layers of corresponding output entries in the outputs vector (only for image outputs, default is 0 for each output).
 * @property {Array} pass_inputs - Input resources that should be bound normally (auto-computed).
 * @property {Array} bindless_inputs - Input resources that are bindless and need special handling (auto-computed).
 * @property {boolean} b_skip_auto_descriptor_setup - Whether to skip automatic pass descriptor setup for this pass (global descriptor setup will still run).
 * @property {boolean} b_split_input_image_mips - Whether to split mips from input images into separate descriptor writes (if false, uploads a single image with all mip levels to GPU).
 * @property {boolean} b_force_keep_pass - Whether to prevent this pass from being culled during render graph compilation.
 */
const RGPassParameters = Object.freeze({
  shader_setup: _.cloneDeep(RGShaderDataSetup),
  inputs: [],
  outputs: [],
  output_views: [],
  pass_inputs: [],
  bindless_inputs: [],
  b_skip_auto_descriptor_setup: false,
  b_split_input_image_mips: false,
  b_force_keep_pass: false,
});

/**
 * Representation of a render graph pass.
 * @typedef {Object} RGPass
 * @property {number} handle - Unique identifier for the pass.
 * @property {Object} pass_config - Configuration for the pass.
 * @property {RGPassParameters} parameters - Parameters for the pass.
 * @property {Function} executor - Function to execute the pass.
 * @property {number} physical_id - Physical identifier for the pass.
 * @property {number} pipeline_state_id - Identifier for the pipeline state.
 * @property {number} reference_count - Number of references to this pass.
 */
const RGPass = Object.freeze({
  handle: 0,
  pass_config: null,
  parameters: null,
  executor: null,
  physical_id: 0,
  pipeline_state_id: 0,
  reference_count: 0,
});

/**
 * Registry for render graph resources. All render graph registry resources (aside from render passes) should be transient and therefore do not need serious caching.
 * For this reason the pass cache is the only thing we don't clear out per-frame. Any resource that need to survive multiple frames
 * should be allocated externally and registered to the render graph as external resources.
 * @typedef {Object} RGRegistry
 * @property {Array} image_resources - Array of image resources.
 * @property {Array} buffer_resources - Array of buffer resources.
 * @property {Array} render_passes - Array of render passes.
 * @property {Array} all_resource_handles - Array of all resource handles.
 * @property {Map} resource_metadata - Map of resource metadata.
 * @property {Array} all_bindless_resource_handles - Array of all bindless resource handles.
 * @property {ExecutionQueue} resource_deletion_queue - Queue for resource deletion.
 * @property {boolean} b_global_bind_group_bound - Whether the global bind group is bound.
 */
const RGRegistry = Object.freeze({
  image_resources: [],
  buffer_resources: [],
  render_passes: [],
  all_resource_handles: [],
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

/**
 * A render graph is used to organize rendering operations in a graphics application.
 * This API provides a comprehensive set of functions to manage resources, pipeline states, and bind groups for a render graph.
 * The RenderGraph class is the core component that manages the entire render graph, including resource creation and registration,
 * render pass management, graph compilation, and rendering/compute command submission.
 */
export class RenderGraph {
  constructor() {
    this.pass_cache = _.cloneDeep(PassCache);
    this.registry = _.cloneDeep(RGRegistry);
    this.non_culled_passes = [];
    this.queued_buffer_global_writes = [];
    this.queued_image_global_writes = [];

    this.image_resource_allocator = new FrameAllocator(
      256,
      _.cloneDeep(RGResource)
    );
    this.buffer_resource_allocator = new FrameAllocator(
      256,
      _.cloneDeep(RGResource)
    );
    this.render_pass_allocator = new FrameAllocator(
        128,
        _.cloneDeep(RGPass)
    );
  }

  /**
   * Resets the render graph, clearing all resources and render passes.
   * This method should be called at the end of each frame to prepare for the next frame.
   * 
   * @param {Object} context - The rendering context.
   * 
   * @example
   * // At the end of each frame
   * const renderGraph = new RenderGraph();
   * // ... (rendering operations)
   * renderGraph.reset(context);
   * // The render graph is now ready for the next frame
   */
  destroy(context) {
    this.reset(context);
    this.registry.resource_deletion_queue.flush();
  }

  /**
   * Begins a new frame in the render graph.
   * This method resets the render graph state and prepares it for a new frame of rendering.
   * 
   * @param {Object} context - The rendering context for the new frame.
   * @returns {void}
   * 
   * @example
   * const renderGraph = new RenderGraph();
   * const context = getGraphicsContext();
   * renderGraph.begin(context);
   * // Add passes and resources...
   * renderGraph.submit(context);
   */
  begin(context) {
    this.reset(context);
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
    const new_resource = this.image_resource_allocator.allocate();
    new_resource.config = config;
    new_resource.config.flags |= ImageFlags.Transient;

    const index = this.registry.image_resources.length;
    this.registry.image_resources.push(new_resource);
    new_resource.handle = create_graph_resource_handle(
      index,
      ResourceType.Image,
      1
    );

    this.registry.all_resource_handles.push(new_resource.handle);
    this.registry.resource_metadata.set(
      new_resource.handle,
      _.cloneDeep(RGResourceMetadata)
    );
    this.registry.resource_metadata.get(new_resource.handle).b_is_bindless =
      config.b_is_bindless;

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
    const physical_id = Name.from(image);
    const image_obj = ResourceCache.get().fetch(CacheTypes.IMAGE, physical_id);
    const new_resource = this.image_resource_allocator.allocate();
    new_resource.config = image_obj.config;

    const index = this.registry.image_resources.length;
    this.registry.image_resources.push(new_resource);
    new_resource.handle = create_graph_resource_handle(
      index,
      ResourceType.Image,
      1
    );

    this.registry.all_resource_handles.push(new_resource.handle);
    this.registry.resource_metadata.set(
      new_resource.handle,
      _.cloneDeep(RGResourceMetadata)
    );
    this.registry.resource_metadata.get(new_resource.handle).physical_id =
      physical_id;
    this.registry.resource_metadata.get(
      new_resource.handle
    ).b_is_persistent = true;
    this.registry.resource_metadata.get(new_resource.handle).b_is_bindless =
      new_resource.config.b_is_bindless;

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
    const new_resource = this.buffer_resource_allocator.allocate();
    new_resource.config = config;
    new_resource.config.flags |= BufferFlags.Transient;

    const index = this.registry.buffer_resources.length;
    this.registry.buffer_resources.push(new_resource);
    new_resource.handle = create_graph_resource_handle(
      index,
      ResourceType.Buffer,
      1
    );

    this.registry.all_resource_handles.push(new_resource.handle);
    this.registry.resource_metadata.set(
      new_resource.handle,
      _.cloneDeep(RGResourceMetadata)
    );
    this.registry.resource_metadata.get(new_resource.handle).b_is_bindless =
      config.b_is_bindless;

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
    const physical_id = Name.from(buffer);
    const buffer_obj = ResourceCache.get().fetch(
      CacheTypes.BUFFER,
      physical_id
    );
    const new_resource = this.buffer_resource_allocator.allocate();
    new_resource.config = buffer_obj.config;

    const index = this.registry.buffer_resources.length;
    this.registry.buffer_resources.push(new_resource);
    new_resource.handle = create_graph_resource_handle(
      index,
      ResourceType.Buffer,
      1
    );

    this.registry.all_resource_handles.push(new_resource.handle);
    this.registry.resource_metadata.set(
      new_resource.handle,
      _.cloneDeep(RGResourceMetadata)
    );
    this.registry.resource_metadata.get(new_resource.handle).physical_id =
      physical_id;
    this.registry.resource_metadata.get(
      new_resource.handle
    ).b_is_persistent = true;
    this.registry.resource_metadata.get(new_resource.handle).b_is_bindless =
      new_resource.config.b_is_bindless;

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
    const pass = this.render_pass_allocator.allocate();
    pass.pass_config = { name: name, flags: pass_type, attachments: [] };
    pass.parameters = { ..._.cloneDeep(RGPassParameters), ...params };
    pass.executor = execution_callback;

    const index = this.registry.render_passes.length;
    this.registry.render_passes.push(pass);
    this.non_culled_passes.push(index);
    pass.handle = index;

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
   * Queues global buffer writes to be processed later in the render graph.
   * 
   * @param {Array} buffer_writes - An array of buffer write operations to be queued.
   * @returns {void}
   * 
   * @example
   * const bufferWrites = [
   *   { buffer: someBuffer, data: new Float32Array([1, 2, 3, 4]), offset: 0 },
   *   { buffer: anotherBuffer, data: new Uint8Array([255, 128, 0]), offset: 16 }
   * ];
   * renderGraph.queue_global_buffer_write(bufferWrites);
   */
  queue_global_buffer_write(buffer_writes) {
    this.queued_buffer_global_writes = [...this.queued_buffer_global_writes, ...buffer_writes];
  }

  /**
   * Queues global image writes to be processed later in the render graph.
   * 
   * @param {Array} image_writes - An array of image write operations to be queued.
   * @returns {void}
   * 
   * @example
   * const imageWrites = [
   *   { image: someImage, data: new Uint8Array([255, 0, 0, 255]), offset: { x: 0, y: 0 }, size: { width: 1, height: 1 } },
   *   { image: anotherImage, data: new Float32Array([0.5, 0.7, 1.0, 1.0]), offset: { x: 10, y: 20 }, size: { width: 2, height: 2 } }
   * ];
   * renderGraph.queue_image_global_writes(imageWrites);
   */
  queue_image_global_writes(image_writes) {
    this.queued_image_global_writes = [...this.queued_image_global_writes, ...image_writes];
  }

  _update_reference_counts(pass) {
    pass.reference_count += pass.parameters.outputs.length;
    for (const resource of pass.parameters.inputs) {
      if (this.registry.resource_metadata.has(resource)) {
        const metadata = this.registry.resource_metadata.get(resource);
        metadata.reference_count += 1;
      }
    }
  }

  _update_resource_param_producers_and_consumers(pass) {
    for (const resource of pass.parameters.inputs) {
      if (this.registry.resource_metadata.has(resource)) {
        const metadata = this.registry.resource_metadata.get(resource);
        metadata.consumers.push(pass.handle);
      }
    }
    for (const resource of pass.parameters.outputs) {
      if (this.registry.resource_metadata.has(resource)) {
        const metadata = this.registry.resource_metadata.get(resource);
        metadata.producers.push(pass.handle);
      }
    }
  }

  _update_present_pass_status(pass) {
    pass.pass_config.b_is_present_pass =
      (pass.pass_config.flags & RenderPassFlags.Present) !==
      RenderPassFlags.None;
    pass.parameters.b_force_keep_pass =
      pass.parameters.b_force_keep_pass || pass.pass_config.b_is_present_pass;
  }

  _compile() {
    this._cull_graph_passes();
    this._compute_resource_first_and_last_users();
  }

  _cull_graph_passes() {
    const passes_to_cull = new Set();
    const unused_stack = [];

    // Lambda to cull a producer pass when it is no longer referenced and pop its inputs onto the unused stack
    const decrement_producer_and_subresource_ref_counts = (producers) => {
      for (const pass_handle of producers) {
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
    for (const resource of this.registry.all_resource_handles) {
      if (this.registry.resource_metadata.has(resource)) {
        if (
          this.registry.resource_metadata.get(resource).reference_count === 0
        ) {
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

  _compute_resource_first_and_last_users() {
    for (const resource of this.registry.all_resource_handles) {
      if (this.registry.resource_metadata.has(resource)) {
        const metadata = this.registry.resource_metadata.get(resource);
        if (metadata.reference_count === 0) {
          continue;
        }

        metadata.first_user = Number.MAX_SAFE_INTEGER;
        metadata.last_user = Number.MIN_SAFE_INTEGER;
        for (const pass of metadata.producers) {
          if (pass < metadata.first_user) {
            metadata.first_user = pass;
          }
          if (pass > metadata.last_user) {
            metadata.last_user = pass;
          }
        }
        for (const pass of metadata.consumers) {
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
   * Submits the compiled render graph for execution.
   * This method compiles the render graph, creates a command encoder, and executes all non-culled passes.
   * It then submits the encoded commands to the GPU for rendering.
   *
   * @async
   * @param {Object} context - The rendering context.
   * @returns {Promise<void>} A promise that resolves when all passes have been executed and submitted.
   * @throws {Error} If there's an error during pass execution or command submission.
   */
  submit(context) {
    this._compile();

    if (this.non_culled_passes.length === 0) {
      return;
    }

    const encoder = CommandQueue.create_encoder(
      context,
      "render_graph_encoder"
    );

    const frame_data = _.cloneDeep(RGFrameData);
    frame_data.context = context;
    frame_data.resource_deletion_queue = this.registry.resource_deletion_queue;

    for (const pass_handle of this.non_culled_passes) {
      this._execute_pass(
        this.registry.render_passes[pass_handle],
        frame_data,
        encoder
      );
    }

    CommandQueue.submit(context, encoder);
  }

  /**
   * Resets the render graph, clearing all resources and render passes.
   * This method should be called at the end of each frame to prepare for the next frame.
   * 
   * @param {Object} context - The rendering context.
   * @returns {void}
   */
  reset(context) {
    this._free_physical_resources(context);

    this.non_culled_passes.length = 0;

    this.registry.all_resource_handles.length = 0;
    this.registry.buffer_resources.length = 0;
    this.registry.image_resources.length = 0;
    this.registry.render_passes.length = 0;
    this.registry.resource_metadata.clear();

    this.image_resource_allocator.reset();
    this.buffer_resource_allocator.reset();
    this.render_pass_allocator.reset();
  }

  _execute_pass(pass, frame_data, encoder) {
    if (!pass) {
      throw new Error("Cannot execute null pass");
    }

    if (
      (pass.pass_config.flags & RenderPassFlags.GraphLocal) !==
      RenderPassFlags.None
    ) {
      this._setup_global_bind_group(pass, frame_data, pass.pipeline_state_id);
      frame_data.global_bind_group = this.pass_cache.global_bind_group;
      pass.executor(this, frame_data, encoder);
    } else {
      this._setup_physical_pass_and_resources(pass, frame_data, encoder);

      const bind_group_list = this.pass_cache.bind_groups.get(
        pass.pass_config.name
      );
      if (BindGroupType.Pass < bind_group_list.length) {
        frame_data.pass_bind_group = bind_group_list[BindGroupType.Pass];
      }

      if (
        (pass.pass_config.flags & RenderPassFlags.Compute) !==
        RenderPassFlags.None
      ) {
        pass.executor(this, frame_data, encoder);
      } else {
        const physical_pass = ResourceCache.get().fetch(
          CacheTypes.PASS,
          pass.physical_id
        );

        if (!physical_pass) {
          throw new Error("Physical pass is null");
        }

        const pipeline = ResourceCache.get().fetch(
          CacheTypes.PIPELINE_STATE,
          pass.pipeline_state_id
        );

        frame_data.current_pass = pass.physical_id;

        physical_pass.begin(encoder, pipeline);
        this._bind_pass_bind_groups(pass, frame_data);
        pass.executor(this, frame_data, encoder);
        physical_pass.end();
      }

      this._update_transient_resources(pass);
    }
  }

  _setup_physical_pass_and_resources(pass, frame_data, encoder) {
    const is_compute_pass =
      (pass.pass_config.flags & RenderPassFlags.Compute) !==
      RenderPassFlags.None;

    const setup_resource = (
      resource,
      resource_params_index,
      is_input_resource
    ) => {
      if (this.registry.resource_metadata.has(resource)) {
        this._setup_physical_resource(
          frame_data.context,
          resource,
          !is_compute_pass,
          is_input_resource
        );
        if (!is_compute_pass) {
          this._tie_resource_to_pass_config_attachments(
            resource,
            pass,
            resource_params_index,
            is_input_resource
          );
        }
        if (is_input_resource) {
          this._setup_pass_input_resource_bindless_type(
            resource,
            pass,
            resource_params_index
          );
        }
      }
    };

    pass.parameters.inputs.forEach((input_resource, i) =>
      setup_resource(input_resource, i, true)
    );
    pass.parameters.outputs.forEach((output_resource, i) =>
      setup_resource(output_resource, i, false)
    );

    pass.physical_id = Name.from(pass.pass_config.name);
    const physical_pass = RenderPass.create(
      is_compute_pass ? RenderPassType.Compute : RenderPassType.Graphics,
      pass.pass_config
    );

    this._setup_pass_pipeline_state(pass, frame_data);
    this._setup_pass_bind_groups(pass, frame_data);
  }

  _setup_physical_resource(
    context,
    resource,
    is_graphics_pass,
    is_input_resource
  ) {
    const resource_type = get_graph_resource_type(resource);
    const resource_index = get_graph_resource_index(resource);

    if (resource_type === ResourceType.Buffer) {
      const buffer_resource = this.registry.buffer_resources[resource_index];
      const buffer_metadata = this.registry.resource_metadata.get(resource);
      if (buffer_metadata.physical_id === 0) {
        buffer_metadata.physical_id = Name.from(buffer_resource.config.name);
        const buffer = Buffer.create(context, buffer_resource.config);

        if (!buffer_metadata.b_is_persistent) {
          this.registry.resource_deletion_queue.remove_execution(
            buffer_metadata.physical_id
          );
          this.registry.resource_deletion_queue.push_execution(
            () => {
              ResourceCache.get().remove(
                CacheTypes.BUFFER,
                buffer_metadata.physical_id
              );
            },
            buffer_metadata.physical_id,
            buffer_metadata.max_frame_lifetime
          );
        }
      }
    } else if (resource_type === ResourceType.Image) {
      const image_resource = this.registry.image_resources[resource_index];
      const image_metadata = this.registry.resource_metadata.get(resource);
      const is_local_load =
        (image_resource.config.flags & ImageFlags.LocalLoad) !==
        ImageFlags.None;
      const is_persistent =
        is_graphics_pass && (!is_input_resource || is_local_load);

      if (image_metadata.physical_id === 0) {
        image_metadata.b_is_persistent |= is_persistent;

        image_metadata.physical_id = Name.from(image_resource.config.name);
        const image = Image.create(context, image_resource.config);

        if (!image_metadata.b_is_persistent) {
          this.registry.resource_deletion_queue.remove_execution(
            image_metadata.physical_id
          );
          this.registry.resource_deletion_queue.push_execution(
            () => {
              ResourceCache.get().remove(
                CacheTypes.IMAGE,
                image_metadata.physical_id
              );
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
    is_input_resource
  ) {
    const resource_type = get_graph_resource_type(resource);
    const resource_index = get_graph_resource_index(resource);

    if (resource_type === ResourceType.Image) {
      const image_resource = this.registry.image_resources[resource_index];
      const is_local_load =
        (image_resource.config.flags & ImageFlags.LocalLoad) !==
        ImageFlags.None;
      if (!is_input_resource || is_local_load) {
        const image_view_index =
          pass.parameters.output_views.length > resource_params_index
            ? pass.parameters.output_views[resource_params_index]
            : 0;
        pass.pass_config.attachments.push({
          image: this.registry.resource_metadata.get(resource).physical_id,
          image_view_index: image_view_index,
          b_image_view_considers_layer_split:
            image_resource.config.split_array_layer_views,
        });
      }
    }
  }

  _setup_pass_input_resource_bindless_type(resource, pass) {
    const resource_type = get_graph_resource_type(resource);
    const resource_index = get_graph_resource_index(resource);

    if (resource_type === ResourceType.Image) {
      const image_resource = this.registry.image_resources[resource_index];
      if (image_resource.config.is_bindless) {
        pass.parameters.bindless_inputs.push(resource);
      } else {
        pass.parameters.pass_inputs.push(resource);
      }
    } else if (resource_type === ResourceType.Buffer) {
      const buffer_resource = this.registry.buffer_resources[resource_index];
      if (buffer_resource.config.b_is_bindless) {
        pass.parameters.bindless_inputs.push(resource);
      } else {
        pass.parameters.pass_inputs.push(resource);
      }
    }
  }

  _setup_pass_pipeline_state(pass, frame_data) {
    if (this.pass_cache.pipeline_states.get(pass.pass_config.name)) {
      pass.pipeline_state_id = this.pass_cache.pipeline_states.get(
        pass.pass_config.name
      );
      frame_data.pass_pipeline_state = pass.pipeline_state_id;
      return;
    }

    const shader_setup = pass.parameters.shader_setup;

    if (shader_setup.pipeline_shaders) {
      const is_compute_pass =
        (pass.pass_config.flags & RenderPassFlags.Compute) !==
        RenderPassFlags.None;

      if (is_compute_pass) {
        const compute_shader = Shader.create(
          frame_data.context,
          shader_setup.pipeline_shaders.compute.path
        );

        const pipeline_descriptor = {
          label: pass.pass_config.name,
          layout: "auto",
          compute: {
            module: compute_shader.module,
            entryPoint:
              shader_setup.pipeline_shaders.compute.entry_point || "cs",
          },
        };

        pass.pipeline_state_id = Name.from(pass.pass_config.name);
        const pipeline = PipelineState.create_compute(
          frame_data.context,
          pass.pass_config.name,
          pipeline_descriptor
        );
      } else {
        const vertex_shader = Shader.create(
          frame_data.context,
          shader_setup.pipeline_shaders.vertex.path
        );
        const fragment_shader = Shader.create(
          frame_data.context,
          shader_setup.pipeline_shaders.fragment.path
        );

        const targets = pass.pass_config.attachments
          .filter((attachment) => !attachment.format || !attachment.format.includes("depth"))
          .map((attachment) => ({
            format: attachment.format || "bgra8unorm",
            blend: shader_setup.attachment_blend || {
              color: {
                srcFactor: "one",
                dstFactor: "zero",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "zero",
                operation: "add",
              },
            }
          }));

        const depth_stencil_target = pass.pass_config.attachments.find((attachment) => attachment.format && attachment.format.includes("depth"));

        const pipeline_descriptor = {
          label: pass.pass_config.name,
          layout: "auto",
          vertex: {
            module: vertex_shader.module,
            entryPoint:
              shader_setup.pipeline_shaders.vertex.entry_point || "vs",
            buffers: [], // Add vertex buffer layouts if needed
          },
          fragment: {
            module: fragment_shader.module,
            entryPoint:
              shader_setup.pipeline_shaders.fragment.entry_point || "fs",
            targets: targets,
          },
          primitive: {
            topology: shader_setup.primitive_topology_type || "triangle-list",
            cullMode: shader_setup.rasterizer_state?.cull_mode || "none",
          },
        };
    
        if (depth_stencil_target)    {
            pipeline_descriptor.depthStencil = {
                depthWriteEnabled: shader_setup.b_depth_write_enabled ?? false,
                depthCompare: shader_setup.depth_stencil_compare_op || "less-equal",
                format: shader_setup.depth_stencil_format || "depth24plus-stencil8",
            }
        }

        pass.pipeline_state_id = Name.from(pass.pass_config.name);
        const pipeline = PipelineState.create_render(
          frame_data.context,
          pass.pass_config.name,
          pipeline_descriptor
        );
      }
      
      this.pass_cache.pipeline_states.set(
          pass.pass_config.name,
          pass.pipeline_state_id
        );
        
      frame_data.pass_pipeline_state = pass.pipeline_state_id;
    }
  }

  async _setup_pass_bind_groups(pass, frame_data) {
    const pipeline_state = ResourceCache.get().fetch(
      CacheTypes.PIPELINE_STATE,
      frame_data.pass_pipeline_state
    );

    const pass_binds = this.pass_cache.bind_groups.get(
      pass.pass_config.name
    ) || { bind_groups: Array(frame_data.context.max_bind_groups()).fill(null) };
    this.pass_cache.bind_groups.set(pass.pass_config.name, pass_binds);

    this._setup_global_bind_group(pass, frame_data);

    if (pass_binds.bind_groups[BindGroupType.Pass]) return;

    // Setup pass-specific bind group
    let entries = [];
    pass.parameters.pass_inputs.forEach((resource, index) => {
      const metadata = this.registry.resource_metadata.get(resource);
      if (metadata.b_is_persistent) {
        const resource_type = get_graph_resource_type(resource);
        if (resource_type === ResourceType.Image) {
          const image = ResourceCache.get().fetch(
            CacheTypes.IMAGE,
            metadata.physical_id
          );
          entries.push({
            binding: index,
            resource: image.create_view(),
          });
        } else {
          const buffer = ResourceCache.get().fetch(
            CacheTypes.BUFFER,
            metadata.physical_id
          );
          entries.push({
            binding: index,
            resource: {
              buffer: buffer.buffer,
              offset: 0,
              size: buffer.size,
            },
          });
        }
      }
    });

    const pass_bind_group = BindGroup.create(
      frame_data.context,
      `${pass.pass_config.name}_bindgroup_${BindGroupType.Pass}`,
      pipeline_state,
      BindGroupType.Pass,
      entries
    );

    pass_binds.bind_groups[BindGroupType.Pass] = pass_bind_group;
  }

  _setup_global_bind_group(pass, frame_data) {
    const pass_binds = this.pass_cache.bind_groups.get(pass.pass_config.name);

    pass_binds.bind_groups[BindGroupType.Global] =
      this.pass_cache.global_bind_group;

    if (pass_binds.bind_groups[BindGroupType.Global]) return;

    const pipeline_state = ResourceCache.get().fetch(
      CacheTypes.PIPELINE_STATE,
      frame_data.pass_pipeline_state
    );


    let entries = [];
    this.queued_buffer_global_writes.forEach((buffer_desc, index) => {
      entries.push({
        binding: index,
        resource: {
          buffer: buffer_desc.buffer,
          offset: buffer_desc.offset || 0,
          size: buffer_desc.size,
        },
      });
    });

    // TODO: Figure out how to write these bindless resources into the global buffer and image arrays
    //   // Setup bindless resources on global bind group
    //   if (!pass.parameters.b_skip_auto_descriptor_setup) {
    //     frame_data.pass_bindless_resources.handles = [];

    //     const bindless_entries = pass.parameters.bindless_inputs.map((resource) => {
    //         const resource_type = get_graph_resource_type(resource);
    //         if (resource_type === ResourceType.Image) {
    //           const image = ResourceCache.get().fetch(
    //             CacheTypes.IMAGE,
    //             this.registry.resource_metadata.get(resource).physical_id
    //           );
    //           const handle = bindless_bind_group.binding_table.get_new(BindlessGroupIndex.Image);
    //           frame_data.pass_bindless_resources.handles.push(handle);
    //           return {
    //             binding: handle.slot,
    //             resource: image.create_view({
    //               dimension: pass.parameters.b_split_input_image_mips
    //                 ? "2d-array"
    //                 : "2d",
    //             }),
    //           };
    //         }
    //         return null;
    //       });

    //     this.registry.all_bindless_resource_handles.push(
    //       ...frame_data.pass_bindless_resources.handles
    //     );
    //   }

    const global_bind_group = BindGroup.create(
      frame_data.context,
      `${pass.pass_config.name}_bindgroup_${BindGroupType.Global}`,
      pipeline_state,
      BindGroupType.Global,
      entries
    );

    this.pass_cache.global_bind_group = global_bind_group;
    this.pass_cache.bind_groups.get(pass.pass_config.name).bind_groups[
      BindGroupType.Global
    ] = this.pass_cache.global_bind_group;
  }

  _bind_pass_bind_groups(pass, frame_data) {
    const physical_pass = ResourceCache.get().fetch(
      CacheTypes.PASS,
      pass.physical_id
    );
    const pass_bind_groups = this.pass_cache.bind_groups.get(
      pass.pass_config.name
    );

    if (pass_bind_groups.bind_groups.length > 0) {
      this.pass_cache.global_bind_group.bind(physical_pass);
      this.registry.b_global_set_bound = true;

      if (BindGroupType.Pass < pass_bind_groups.bind_groups.length) {
        const bind_group = pass_bind_groups.bind_groups[BindGroupType.Pass];
        if (bind_group) {
          bind_group.bind(physical_pass);
        }
      }
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
      const resource_meta =
        this.registry.resource_metadata.get(output_resource);
      if (
        resource_meta.physical_id !== 0 &&
        resource_meta.last_user === pass.handle &&
        !resource_meta.b_is_persistent
      ) {
        // TODO: Transient resource memory aliasing
      }
    });
  }

  _free_physical_resources(context) {
    this.registry.resource_deletion_queue.update();
    // TODO: Free all bindless resources handles stored in this.registry.all_bindless_resource_handles
  }

  static create() {
    return new RenderGraph();
  }
}

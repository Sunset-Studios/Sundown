import { FreeListAllocator } from "../memory/allocator.js";
import { TypedVector } from "../memory/container.js";
import { Tensor } from "./math/tensor.js";
import { LayerType } from "./ml_types.js";

import { FullyConnected } from "./layers/fully_connected.js";
import { MSELoss } from "./layers/mse_loss.js";
import { CrossEntropyLoss } from "./layers/cross_entropy_loss.js";
import { BinaryCrossEntropyLoss } from "./layers/binary_cross_entropy_loss.js";
import { ReLU } from "./layers/relu.js";
import { Tanh } from "./layers/tanh.js";
import { Sigmoid } from "./layers/sigmoid.js";
import { Softmax } from "./layers/softmax.js";

import { deep_clone } from "../utility/object.js";

/**
 * Data structure for layer properties
 */
class LayerData {
  id = null;
  type = null;
  properties = null;
  params = null;
  grad_params = null;
  cached_input = null;
  cached_output = null;
  parent_ids = new TypedVector(16, 0, Uint32Array);
  child_ids = new TypedVector(16, 0, Uint32Array);
  training_context = null;
  loss = null;
}

/**
 * Context object for training operations
 */
export class TrainingContext {
  constructor(options = {}) {
    this.learning_rate = options.learning_rate || 0.01;
    this.optimizer = options.optimizer || null;
    this.batch_size = options.batch_size || 1;
    this.momentum = options.momentum || 0;
    this.weight_decay = options.weight_decay || 0;
    this.name = options.name || null;
  }

  derive(overrides = {}) {
    return new TrainingContext({
      ...this,
      ...overrides,
    });
  }
}

export class Layer {
  static initialized = false;
  static layer_allocator = new FreeListAllocator(1024, LayerData);
  static all_layers = new Map();
  static roots = new Set();
  static next_id = 0;
  static type_handlers = new Map();
  static default_context = new TrainingContext();

  static initialize() {
    if (Layer.initialized) return;

    Layer.register_handler(LayerType.FULLY_CONNECTED, FullyConnected);

    Layer.register_handler(LayerType.RELU, ReLU);
    Layer.register_handler(LayerType.TANH, Tanh);
    Layer.register_handler(LayerType.SIGMOID, Sigmoid);
    Layer.register_handler(LayerType.SOFTMAX, Softmax);

    Layer.register_handler(LayerType.MSE, MSELoss);
    Layer.register_handler(LayerType.CROSS_ENTROPY, CrossEntropyLoss);
    Layer.register_handler(LayerType.BINARY_CROSS_ENTROPY, BinaryCrossEntropyLoss);

    Layer.initialized = true;
  }

  /**
   * Registers a handler for a specific layer type
   *
   * @param {string} type - The layer type
   * @param {Object} handler - Object with forward, backward, and update methods
   */
  static register_handler(type, handler) {
    Layer.type_handlers.set(type, handler);
  }

  /**
   * Unregisters a handler for a specific layer type
   *
   * @param {string} type - The layer type
   */
  static unregister_handler(type) {
    Layer.type_handlers.delete(type);
  }

  /**
   * Sets the training context for a subnet root
   *
   * @param {number} root_id - The ID of the root layer
   * @param {TrainingContext} context - The training context
   * @returns {boolean} True if context was set successfully
   */
  static set_subnet_context(root_id, context) {
    const root = Layer.get(root_id);
    if (!root) return false;
    root.training_context = context;
    return true;
  }

  /**
   * Gets the effective training context for a layer
   *
   * @param {number} layer_id - The ID of the layer
   * @returns {TrainingContext} The effective training context
   */
  static get_effective_context(layer_id) {
    const layer = Layer.get(layer_id);
    if (!layer) return Layer.default_context;

    // Check for layer-specific context
    if (layer.training_context) {
      return layer.training_context;
    }

    // Find root of this subnet
    let current = layer;
    while (current.parent_ids.length > 0) {
      current = Layer.get(current.parent_ids.get(0)); // Follow first parent
      if (!current) break;
    }

    // Return root context or default if not found
    return current?.training_context || Layer.default_context;
  }

  /**
   * Sets a property of the training context for a layer
   *
   * @param {number} layer_id - The ID of the layer
   * @param {string} prop_name - The name of the property to set
   * @param {any} value - The value to set the property to
   * @returns {boolean} True if the property was set successfully
   */
  static set_layer_context_property(layer_id, prop_name, value) {
    const layer = Layer.get(layer_id);
    if (!layer) return false;

    // Create or update this layer's context
    if (!layer.training_context) {
      layer.training_context = Layer.get_effective_context(layer_id).derive({});
    }

    // Set the specific parameter
    layer.training_context[prop_name] = value;
    return true;
  }

  /**
   * Creates a new layer
   *
   * @param {number} type - The type of layer
   * @param {Object} properties - Properties for the layer
   * @param {number} parent - The ID of the parent layer
   * @param {Object} params - Parameters for the layer
   * @param {Object} grad_params - Gradient parameters for the layer
   * @returns {number} The ID of the created layer
   */
  static create(type, properties, parent = null, params = null, grad_params = null) {
    Layer.initialize();

    const layer = Layer.layer_allocator.allocate();
    layer.id = Layer.next_id++;
    layer.type = type;
    layer.properties = properties;
    layer.params = params;
    layer.grad_params = grad_params;

    // Register the layer
    Layer.all_layers.set(layer.id, layer);

    if (parent !== null) {
      Layer.connect(parent, layer.id);
    } else {
      Layer.roots.add(layer.id);
    }

    // Validate that we have a handler for this type
    if (type && (!Layer.type_handlers.has(type) || !Layer.type_handlers.get(type).initialize)) {
      console.warn(`No initializer handler registered for layer type: ${type}`);
    } else {
      const type_handler = Layer.type_handlers.get(type);
      type_handler.initialize(layer);
    }

    return layer.id;
  }

  /**
   * Gets a layer by ID
   *
   * @param {number} id - The layer ID
   * @returns {Layer|null} The layer or null if not found
   */
  static get(id) {
    return Layer.all_layers.get(id) || null;
  }

  /**
   * Destroys a layer and its connections
   *
   * @param {number} id - The ID of the layer to destroy
   */
  static destroy(id) {
    const layer = Layer.get(id);
    if (!layer) return;

    // Disconnect from parents
    for (let i = 0; i < layer.parent_ids.length; ++i) {
      Layer.disconnect(layer.parent_ids.get(i), id);
    }

    // Disconnect from children
    for (let i = 0; i < layer.child_ids.length; ++i) {
      Layer.disconnect(id, layer.child_ids.get(i));
    }

    // Remove from storage
    Layer.all_layers.delete(id);
    Layer.roots.delete(id);
  }

  /**
   * Connects two layers
   *
   * @param {number} source_id - The ID of the source layer
   * @param {number} target_id - The ID of the target layer
   * @returns {boolean} True if connection was successful
   */
  static connect(source_id, target_id) {
    const source = Layer.get(source_id);
    const target = Layer.get(target_id);

    if (!source || !target) return false;
    if (source_id === target_id) return false; // Can't connect to itself

    // Check for cycles
    if (Layer.would_create_cycle(source_id, target_id)) {
      return false;
    }

    console.log(source_id, target_id)
    // Add connection
    source.child_ids.push(target_id);
    target.parent_ids.push(source_id);

    // Target is no longer a root since it has a parent
    Layer.roots.delete(target_id);

    return true;
  }

  /**
   * Disconnects two layers
   *
   * @param {number} source_id - The ID of the source layer
   * @param {number} target_id - The ID of the target layer
   * @returns {boolean} True if disconnection was successful
   */
  static disconnect(source_id, target_id) {
    const source = Layer.get(source_id);
    const target = Layer.get(target_id);

    if (!source || !target) return false;

    // Remove connection
    source.child_ids.remove_element(target_id);
    target.parent_ids.remove_element(source_id);

    // If target has no more parents, it becomes a root
    if (target.parent_ids.length === 0) {
      Layer.roots.add(target_id);
    }

    return true;
  }

  /**
   * Disconnects a layer from all its parents
   *
   * @param {number} id - The ID of the layer
   * @returns {boolean} True if disconnection was successful
   */
  static disconnect_all(id) {
    const layer = Layer.get(id);
    if (!layer) return false;

    layer.parent_ids.clear();

    Layer.roots.add(id);

    return true;
  }

  /**
   * Duplicates a layer with its parameters
   *
   * @param {number} id - The ID of the layer to duplicate
   * @returns {number|null} The ID of the new layer, or null if failed
   */
  static duplicate(id) {
    const layer = Layer.get(id);
    if (!layer) return null;

    // Clone properties if needed
    let cloned_properties = null;
    if (layer.properties) {
      cloned_properties = deep_clone(layer.properties);
    }

    // Clone params if needed
    let cloned_params = null;
    if (layer.params) {
      cloned_params = deep_clone(layer.params);
    }

    // Clone gradient params if needed
    let cloned_grad_params = null;
    if (layer.grad_params) {
      cloned_grad_params = deep_clone(layer.grad_params);
    }

    // Create new layer with same type and cloned params
    return Layer.create(layer.type, cloned_properties, cloned_params, cloned_grad_params);
  }

  /**
   * Duplicates a subnet of the layer graph
   *
   * @param {number} root_id - The ID of the root layer of the subnet
   * @returns {number|null} The ID of the new subnet root, or null if failed
   */
  static duplicate_subnet(root_id) {
    const layer = Layer.get(root_id);
    if (!layer) return null;

    // Map of original ID to cloned ID
    const id_map = new Map();

    // First pass: create all layers iteratively
    const to_process = [root_id];

    while (to_process.length > 0) {
      const id = to_process.pop();
      if (id_map.has(id)) continue;

      const layer = Layer.get(id);
      if (!layer) continue;

      // Clone the layer
      const cloned_properties = layer.properties ? deep_clone(layer.properties) : null;
      const cloned_params = layer.params ? deep_clone(layer.params) : null;
      const cloned_grad_params = layer.grad_params ? deep_clone(layer.grad_params) : null;

      const cloned_id = Layer.create(
        layer.type,
        cloned_properties,
        cloned_params,
        cloned_grad_params
      );
      id_map.set(id, cloned_id);

      // Add all children to the processing queue
      for (let i = 0; i < layer.child_ids.length; ++i) {
        to_process.push(layer.child_ids.get(i));
      }
    }

    // Second pass: create all connections iteratively
    to_process.length = 0;
    to_process.push(root_id);

    const processed = new Set();

    while (to_process.length > 0) {
      const id = to_process.pop();

      if (processed.has(id)) continue;
      processed.add(id);

      const layer = Layer.get(id);
      if (!layer) continue;

      const cloned_id = id_map.get(id);
      if (!cloned_id) continue;

      // Connect to all cloned children
      for (let i = 0; i < layer.child_ids.length; ++i) {
        const child_id = layer.child_ids.get(i);
        const cloned_child_id = id_map.get(child_id);
        if (cloned_child_id) {
          Layer.connect(cloned_id, cloned_child_id);
        }
        // Add child to processing queue
        to_process.push(child_id);
      }
    }

    return id_map.get(root_id);
  }

  /**
   * Reorders a layer in its parent's children
   *
   * @param {number} id - The ID of the layer
   * @param {number} target_index - The index to move the layer to
   * @returns {boolean} True if reordering was successful
   */
  static reorder_in_parent(id, target_index) {
    const layer = Layer.get(id);
    if (!layer) return false;

    const parent_id = layer.parent_ids.get(0);
    if (!parent_id) return false;

    const parent = Layer.get(parent_id);
    if (!parent) return false;

    parent.child_ids.remove_element(id);
    parent.child_ids.insert(id, target_index);

    return true;
  }

  /**
   * Checks if connecting two layers would create a cycle
   *
   * @param {number} source_id - The ID of the source layer
   * @param {number} target_id - The ID of the target layer
   * @returns {boolean} True if connecting would create a cycle
   */
  static would_create_cycle(source_id, target_id) {
    // If target is reachable from source, connecting would create a cycle
    const visited = new Set();
    const to_visit = [target_id];

    while (to_visit.length > 0) {
      const current_id = to_visit.pop();

      if (current_id === source_id) {
        return true; // Found a path from target to source, would create cycle
      }

      if (visited.has(current_id)) {
        continue; // Skip already visited nodes
      }
      visited.add(current_id);

      const layer = Layer.get(current_id);
      if (layer) {
        for (let i = 0; i < layer.child_ids.length; ++i) {
          const child_id = layer.child_ids.get(i);
          to_visit.push(child_id);
        }
      }
    }

    return false; // No path found from target to source
  }

  /**
   * Gets all layers in a subnet (subgraph)
   *
   * @param {number} root_id - The ID of the root layer
   * @returns {Set<number>} Set of layer IDs in the subnet
   */
  static #subnet_layers = new Set();
  static get_subnet_layers(root_id) {
    Layer.#subnet_layers.clear();

    const to_visit = [root_id];

    while (to_visit.length > 0) {
      const current_id = to_visit.pop();

      if (Layer.#subnet_layers.has(current_id)) {
        continue;
      }
      Layer.#subnet_layers.add(current_id);

      const layer = Layer.get(current_id);
      if (!layer) continue;

      for (let i = 0; i < layer.child_ids.length; ++i) {
        const child_id = layer.child_ids.get(i);
        if (!Layer.#subnet_layers.has(child_id)) {
          to_visit.push(child_id);
        }
      }
    }

    return Array.from(Layer.#subnet_layers);
  }

  /**
   * Gets the shared roots of a subnet
   *
   * @param {number} root_id - The ID of the root layer
   * @returns {Array<number>} Array of shared root layer IDs
   */
  static #shared_roots = new Set();
  static get_subnet_shared_roots(root_id) {
    Layer.#shared_roots.clear();

    const subnet_layers = Layer.get_subnet_layers(root_id);

    // Get all root layers in the subnet iteratively
    while (subnet_layers.length > 0) {
      const current_id = subnet_layers.pop();
      if (Layer.#shared_roots.has(current_id)) continue;

      const layer = Layer.get(current_id);
      if (layer.parent_ids.length === 0) {
        Layer.#shared_roots.add(current_id);
      } else {
        for (let i = 0; i < layer.parent_ids.length; ++i) {
          const parent_id = layer.parent_ids.get(i);
          if (!subnet_layers.includes(parent_id)) {
            subnet_layers.push(parent_id);
          }
        }
      }
    }

    return Array.from(Layer.#shared_roots);
  }

  /**
   * Gets all layers in a subnet and any shared layers from connected roots
   *
   * @param {number} root_id - The ID of the root layer
   * @returns {Array<number>} Array of layer IDs in the subnet
   */
  static #shared_layers = new Set();
  static get_subnet_and_shared_layers(root_id) {
    Layer.#shared_layers.clear();

    const subnet_layers = Layer.get_subnet_layers(root_id);

    // Get all root layers in the subnet iteratively
    while (subnet_layers.length > 0) {
      const current_id = subnet_layers.pop();
      if (Layer.#shared_layers.has(current_id)) continue;

      Layer.#shared_layers.add(current_id);

      const layer = Layer.get(current_id);
      for (let i = 0; i < layer.parent_ids.length; ++i) {
        const parent_id = layer.parent_ids.get(i);
        if (!subnet_layers.includes(parent_id)) {
          subnet_layers.push(parent_id);
        }
      }
    }

    return Array.from(Layer.#shared_layers);
  }

  /**
   * Gets all root layers in the system
   *
   * @returns {Array<number>} Array of root layer IDs
   */
  static get_all_roots() {
    return Array.from(Layer.roots);
  }

  /**
   * Gets the parents of a layer
   *
   * @param {number} id - The ID of the layer
   * @returns {Array<number>} Array of parent layer IDs
   */
  static get_parents(id) {
    const layer = Layer.get(id);
    return layer ? layer.parent_ids : [];
  }

  /**
   * Gets the children of a layer
   *
   * @param {number} id - The ID of the layer
   * @returns {Array<number>} Array of child layer IDs
   */
  static get_children(id) {
    const layer = Layer.get(id);
    return layer ? layer.child_ids : [];
  }

  /**
   * Gets the leaf layers from a given root layer
   *
   * @param {number} id - The ID of the root layer
   * @returns {Array<number>} Array of leaf layer IDs
   */
  static get_last_layer(id) {
    const layer = Layer.get(id);
    if (!layer) return null;

    // Get all children iteratively
    const all_children = new Set();

    const queue = [];
    for (let i = 0; i < layer.child_ids.length; ++i) {
      queue.push(layer.child_ids.get(i));
    }

    while (queue.length > 0) {
      const current_id = queue.shift();
      const current_layer = Layer.get(current_id);

      if (current_layer.child_ids.length === 0) {
        all_children.add(current_id);
      } else {
        for (let i = 0; i < current_layer.child_ids.length; ++i) {
          queue.push(current_layer.child_ids.get(i));
        }
      }
    }
    
    return Array.from(all_children);
  }

  /**
   * Forward pass through the network starting from a root
   *
   * @param {number} root_id - The ID of the root layer
   * @param {Object} input - The input to the network
   * @param {Object} target - Optional target for loss layers
   * @returns {Object} The output of the network
   */
  static forward(root_id, input, target = null) {
    Layer.initialize();

    const outputs = new Map(); // Map of layer ID to output
    const in_degree = new Map(); // Count of unprocessed parents for each layer
    const queue = []; // Queue for BFS

    // Calculate in-degree for all layers in the subnet
    const subnet_layers = Layer.get_subnet_layers(root_id);
    for (let i = 0; i < subnet_layers.length; ++i) {
      const id = subnet_layers[i];
      const layer = Layer.get(id);
      in_degree.set(id, layer.parent_ids.length);
    }

    // If root_id is not in the queue, add it (ensure we start from the specified root)
    if (!queue.includes(root_id)) {
      queue.push(root_id);
    }

    // Record input for the root layer
    outputs.set(root_id, input);

    // Process in BFS order
    while (queue.length > 0) {
      const current_id = queue.shift();
      const current_layer = Layer.get(current_id);

      // Get inputs from all parents
      const layer_inputs = [];

      // If this is the root, use the provided input
      if (current_id === root_id) {
        layer_inputs.push(input);
      } else {
        for (let i = 0; i < current_layer.parent_ids.length; ++i) {
          const parent_id = current_layer.parent_ids.get(i);
          if (outputs.has(parent_id)) {
            layer_inputs.push(outputs.get(parent_id));
          }
        }
      }

      // Process the layer
      let layer_output;
      if (layer_inputs.length > 0) {
        // Combine inputs if multiple parents
        const combined_input = Layer.combine_inputs(current_layer, layer_inputs);
        layer_output = Layer.process_forward(current_id, combined_input, target);
      } else {
        // No inputs (shouldn't happen in normal operation)
        layer_output = null;
      }

      // Store the output
      outputs.set(current_id, layer_output);

      // Enqueue children whose dependencies are resolved
      for (let i = 0; i < current_layer.child_ids.length; ++i) {
        const child_id = current_layer.child_ids.get(i);
        const new_degree = in_degree.get(child_id) - 1;
        in_degree.set(child_id, new_degree);

        if (new_degree === 0) {
          queue.push(child_id);
        }
      }
    }

    // Find all sink nodes (layers with no children) within the subnet
    const sink_outputs = [];
    for (let i = 0; i < subnet_layers.length; ++i) {
      const id = subnet_layers[i];
      const layer = Layer.get(id);
      if (layer.child_ids.length === 0 && outputs.has(id)) {
        sink_outputs.push(outputs.get(id));
      }
    }

    // If no sinks found, return the output of the specified root
    if (sink_outputs.length === 0 && outputs.has(root_id)) {
      sink_outputs.push(outputs.get(root_id));
    }

    return Tensor.stack(sink_outputs);
  }

  /**
   * Backward pass through the network
   *
   * @param {number} root_id - The ID of the root layer
   * @param {Object} grad_output - The gradient of the output
   * @param {Object} target - Optional target for loss layers
   */
  static backward(root_id, grad_output, target = null) {
    Layer.initialize();

    const gradients = new Map(); // Map of layer ID to incoming gradient
    const out_degree = new Map(); // Count of unprocessed children for each layer
    const queue = []; // Queue for BFS (in reverse)

    // Calculate out-degree for all layers in the subnet
    const subnet_layers = Layer.get_subnet_layers(root_id);
    for (let i = 0; i < subnet_layers.length; ++i) {
      const id = subnet_layers[i];
      const layer = Layer.get(id);
      out_degree.set(id, layer.child_ids.length);

      // Find sink nodes (no children)
      if (layer.child_ids.length === 0) {
        queue.push(id);
        gradients.set(id, grad_output); // Initialize with provided gradient
      }
    }

    // If queue is empty, just use the specified root
    if (queue.length === 0) {
      queue.push(root_id);
      gradients.set(root_id, grad_output);
    }

    // Process in reverse BFS order
    while (queue.length > 0) {
      const current_id = queue.shift();
      const current_layer = Layer.get(current_id);
      const current_grad = gradients.get(current_id);

      // Compute gradient with respect to input
      const input_grad = Layer.process_backward(current_id, current_grad, target);

      // Propagate to parents
      for (let i = 0; i < current_layer.parent_ids.length; ++i) {
        const parent_id = current_layer.parent_ids.get(i);
        if (!gradients.has(parent_id)) {
          gradients.set(parent_id, input_grad);
        } else {
          // Accumulate gradients if the parent has multiple children
          const existing_grad = gradients.get(parent_id);
          gradients.set(parent_id, Layer.add_gradients(parent_id, existing_grad, input_grad));
        }

        // Decrement out-degree
        const new_degree = out_degree.get(parent_id) - 1;
        out_degree.set(parent_id, new_degree);

        // If all children processed, enqueue parent
        if (new_degree === 0) {
          queue.push(parent_id);
        }
      }
    }

    // Update parameters for all layers in the subnet
    const context = Layer.get_effective_context(root_id);
    for (let i = 0; i < subnet_layers.length; ++i) {
      const id = subnet_layers[i];
      Layer.update_parameters(id, context.learning_rate, context.optimizer, context.weight_decay);
    }
  }

  /**
   * Process a single layer's forward computation
   *
   * @param {number} id - The layer ID
   * @param {Object} input - The input to the layer
   * @param {Object} target - Optional target for loss layers
   * @returns {Object} The layer's output
   */
  static process_forward(id, input, target = null) {
    const layer = Layer.get(id);
    if (!layer) return input;

    const handler = Layer.type_handlers.get(layer.type);
    if (!handler || !handler.forward) {
      // No handler or no forward method, just pass through
      return input;
    }

    return handler.forward(layer, input, target);
  }

  /**
   * Process a single layer's backward computation
   *
   * @param {number} id - The layer ID
   * @param {Object} grad_output - The gradient of the output
   * @param {Object} target - Optional target for loss layers
   * @returns {Object} The gradient with respect to input
   */
  static process_backward(id, grad_output, target = null) {
    const layer = Layer.get(id);
    if (!layer) return grad_output;

    const handler = Layer.type_handlers.get(layer.type);
    if (!handler || !handler.backward) {
      // No handler or no backward method, just pass through
      return grad_output;
    }

    return handler.backward(layer, grad_output, target);
  }

  /**
   * Update parameters for a layer
   *
   * @param {number} id - The layer ID
   * @param {number} learning_rate - The learning rate
   * @param {Object} optimizer - The optimizer
   * @param {number} weight_decay - The weight decay
   */
  static update_parameters(id, learning_rate, optimizer = null, weight_decay = 0) {
    const layer = Layer.get(id);
    if (!layer || !layer.params || !layer.grad_params) return;

    const handler = Layer.type_handlers.get(layer.type);
    if (!handler || !handler.update_parameters) {
      // No handler or no update method
      return;
    }

    handler.update_parameters(layer, learning_rate, optimizer, weight_decay);
  }

  /**
   * Combine multiple inputs for a layer
   *
   * @param {Layer} layer - The layer
   * @param {Array} inputs - The inputs to combine
   * @returns {Object} The combined input
   */
  static combine_inputs(layer, inputs) {
    if (layer) {
      const handler = Layer.type_handlers.get(layer.type);
      if (handler && handler.combine_inputs) {
        return handler.combine_inputs(inputs);
      }
    }

    // Default implementation
    if (inputs.length === 0) return null;
    if (inputs.length === 1) return inputs[0];

    // Return a tensor stack using the tensor with the largest fitting dimension as the reference tensor
    let reference_tensor = inputs[0];
    for (let i = 1; i < inputs.length; i++) {
      if (inputs[i].size > reference_tensor.size) {
        reference_tensor = inputs[i];
      }
    }
    return Tensor.stack(inputs, reference_tensor.shape);
  }

  /**
   * Add gradients together
   *
   * @param {Object} grad1 - First gradient
   * @param {Object} grad2 - Second gradient
   * @returns {Object} Combined gradient
   */
  static add_gradients(id, grad1, grad2) {
    const layer = Layer.get(id);
    if (layer) {
      const handler = Layer.type_handlers.get(layer.type);
      if (handler && handler.add_gradients) {
        return handler.add_gradients(grad1, grad2);
      }
    }

    // Default implementation
    if (!grad1) return grad2;
    if (!grad2) return grad1;

    return grad1.add(grad2);
  }

  /**
   * Resets all layers in the system
   */
  static reset_all() {
    Layer.all_layers.clear();
    Layer.roots.clear();
    Layer.next_id = 0;
  }
}

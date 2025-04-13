import { ModelType, LayerType, OptimizerType } from "../ml_types.js";
import { Layer, TrainingContext } from "../layer.js";

import { Adam } from "../optimizers/adam.js";

export class HopAPIAdapter {
  static set_subnet_context(subnet_id, options = {}) {
    const context = new TrainingContext(options);
    Layer.set_subnet_context(subnet_id, context);
    return context;
  }

  static set_subnet_context_property(subnet_id, prop_name, value) {
    return Layer.set_layer_context_property(subnet_id, prop_name, value);
  }

  static add_input(capacity, batch_size, parent = null) {
    return Layer.create(LayerType.INPUT, { capacity, batch_size }, parent);
  }

  static add_layer(type, input_size, output_size, parent = null, options = {}, params = null) {
    let layer = null;

    if (type === LayerType.FULLY_CONNECTED) {
      layer = Layer.create(type, { input_size, output_size, ...options }, parent, params);
    }

    return layer;
  }

  static add_activation(type, parent = null) {
    return Layer.create(type, {}, parent);
  }

  static add_loss(type, enabled_logging = false, name = null, parent = null) {
    return Layer.create(type, { enabled_logging, name }, parent);
  }

  static set_optimizer(type, root = null, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    let optimizer = null;

    if (type === OptimizerType.ADAM) {
      optimizer = new Adam(beta1, beta2, epsilon);
    }

    if (root) {
      Layer.set_layer_context_property(root, 'optimizer', optimizer);
    }

    return optimizer;
  }

  static clear_model(root) {
    Layer.destroy(root);
  }

  /**
   * Connects one layer to another layer
   * 
   * @param {number} source_layer_id - ID of the layer to be connected
   * @param {number} target_layer_id - ID of the layer to connect to as a parent
   * @returns {boolean} True if the operation was successful
   */
  static connect_layer(source_layer_id, target_layer_id) {
    return Layer.connect(source_layer_id, target_layer_id);
  }
  
  /**
   * Disconnects a layer from its parent
   * 
   * @param {number} parent_id - ID of the parent layer
   * @param {number} layer_id - ID of the layer to disconnect
   * @returns {boolean} True if the operation was successful
   */
  static disconnect_layer(parent_id, layer_id) {
    return Layer.disconnect(parent_id, layer_id);
  }

  /**
   * Disconnects a layer from all its parents
   * 
   * @param {number} layer_id - ID of the layer to disconnect
   * @returns {boolean} True if the operation was successful
   */
  static disconnect_layer_from_all(layer_id) {
    return Layer.disconnect_all(layer_id);
  }
  
  /**
   * Reorders a layer within its parent's children
   * 
   * @param {number} layer_id - ID of the layer to reorder
   * @param {number} target_index - New index for the layer
   * @returns {boolean} True if the operation was successful
   */
  static reorder_layer(layer_id, target_index) {
    return Layer.reorder_in_parent(layer_id, target_index);
  }
}

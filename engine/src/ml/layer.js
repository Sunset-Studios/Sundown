export class Layer {
  static all_layers = [];

  id = null;
  parent = null;
  layers = [];
  cached_input = null;
  cached_output = null;
  params = null;
  grad_params = null;

  constructor() {
    this.id = Layer.all_layers.length;
    Layer.all_layers.push(this);
  }

  /**
   * Destroys the layer.
   */
  destroy() {
    Layer.all_layers.splice(this.id, 1);
  }

  /**
   * Adds a sublayer to the layer.
   *
   * @param {Object} layer - The layer to add.
   */
  add(layer) {
    layer.parent = this;
    this.layers.push(layer);
    return this.layers.length - 1;
  }

  /**
   * Removes a sublayer from the layer.
   *
   * @param {Object} layer - The layer to remove.
   */
  remove(layer) {
    for (let i = this.layers.length - 1; i >= 0; i--) {
      if (this.layers[i] === layer) {
        this.layers[i].parent = null;
        this.layers.splice(i, 1);
      }
    }
  }

  /**
   * Gets a sublayer by index.
   *
   * @param {number} layer_index - The index of the layer to get.
   * @returns {Object} The layer at the given index.
   */
  get(layer_index) {
    return this.layers[layer_index];
  }

  /**
   * Checks if the layer contains a sublayer.
   *
   * @param {Object} layer - The layer to check.
   * @returns {boolean} True if the layer contains the sublayer, false otherwise.
   */
  contains(layer) {
    return this.layers.includes(layer);
  }

  /**
   * Forward pass through the layer.
   *
   * @param {Object} input - The input to the layer.
   * @param {Object} target - The target to the layer.

   * @returns {Object} The output of the layer.
   */
  forward(input, target = null) {
    let output = input;
    for (let i = 0; i < this.layers.length; i++) {
      output = this.layers[i].forward(output, target);
    }
    return output;
  }


  /**
   * Backward pass through the layer.
   *
   * @param {Object} grad_output - The gradient of the output of the layer.
   * @param {Object} target - The target to the layer.
   * @returns {Object} The gradient of the input of the layer.
   */
  backward(grad_output, target = null) {
    let grad = grad_output;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad, target);
    }
    return grad;
  }

  /**
   * Update the weights of the layer.
   *
   * @param {number} learning_rate - The learning rate.
   * @param {Object} optimizer - The optimizer to use.
   */
  update_weights(learning_rate, optimizer = null) {
    for (let i = this.layers.length - 1; i >= 0; i--) {
      this.layers[i].update_weights(learning_rate, optimizer);
    }
  }

  /**
   * Clears the model of layers and cached data.
   */
  clear() {
    this.layers.length = 0;
    this.cached_input = null;
    this.cached_output = null;
    this.params = null;
    this.grad_params = null;
  }

  /**
   * Gets the number of layers in the layer.
   *
   * @returns {number} The number of layers in the layer.
   */
  get layer_count() {
    return this.layers.length;
  }

  /**
   * Gets the last layer in the layer.
   *
   * @returns {Object} The last layer in the layer.
   */
  get last_layer() {
    return this.layers[this.layers.length - 1];
  }
}

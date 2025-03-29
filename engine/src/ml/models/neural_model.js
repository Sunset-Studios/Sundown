import { Layer } from "../layer.js";

export class NeuralModel extends Layer {
  name = null;
  learning_rate = 0.01;
  loss_fn = null;

  /**
   * Creates a new NeuralModel.
   *
   * @param {string} name - The name of the neural network.
   * @param {Object} options - The options for the neural network.
   */
  constructor(name, options = {}) {
    super();
    this.name = name;
    this.learning_rate = options.learning_rate || 0.01;
    this.loss_fn = options.loss_fn || null;
    this.optimizer = options.optimizer || null;
  }

  /**
   * Sets the optimizer for the neural network.
   *
   * @param {Object} optimizer - The optimizer.
   */
  set_optimizer(optimizer) {
    this.optimizer = optimizer;
  }

  /**
   * Sets the loss function for the neural network.
   *
   * @param {Object} loss_fn - The loss function.
   */
  set_loss_fn(loss_fn) {
    this.loss_fn = loss_fn;
  }

  /**
   * Sets the learning rate for the neural network.
   *
   * @param {number} learning_rate - The learning rate.
   */
  set_learning_rate(learning_rate) {
    this.learning_rate = learning_rate;
  }

  /**
   * Runs inference on the neural network based on the input tensor.
   *
   * @param {Object} input_tensor - The input tensor.
   * @returns {Object} The output tensor.

   */
  predict(input_tensor) {
    return super.forward(input_tensor);
  }

  /**
   * Trains on a single sample of data.
   *
   * @param {Object} input_tensor - Training inputs: { data: Float32Array, shape: [...], batch_size: number }
   * @param {Object} target_tensor - Expected outputs: { data: Float32Array, shape: [...], batch_size: number }
   * @param {Object} output_layer - The output layer to cache the output of.
   * @returns {Object} The output tensor.
   */
  train(input_tensor, target_tensor, output_layer = null) {
    // Forward pass.
    let output = super.forward(input_tensor, target_tensor);

    // Backward pass: propagate gradients from the last layer back.
    super.backward(output, target_tensor);

    // Update weights in each layer (if applicable).
    super.update_weights(this.learning_rate, this.optimizer);

    // Cache the output for weight sharing.
    this.cached_output = output;

    return output_layer !== null && super.contains(output_layer)
      ? output_layer.cached_output
      : output;
  }
}

import { Layer } from "../layer.js";

export class NeuralModel extends Layer {
  name = null;
  learning_rate = 0.01;
  loss_fn = null;

  constructor(name, options = {}) {
    super();
    this.name = name;
    this.learning_rate = options.learning_rate || 0.01;
    this.loss_fn = options.loss_fn || null;
    this.optimizer = options.optimizer || null;
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

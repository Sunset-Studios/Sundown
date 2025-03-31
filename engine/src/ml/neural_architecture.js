import { Layer } from "./layer.js";

export class NeuralArchitectureHelpers {
  /**
   * Runs inference on the neural network based on the input tensor and the root layer ID.
   *
   * @param {number} root_id - The ID of the root layer.
   * @param {Object} input_tensor - The input tensor.
   * @returns {Object} The output tensor.

   */
  static predict(root_id, input_tensor) {
    return Layer.forward(root_id, input_tensor);
  }

  /**
   * Trains on a single sample of data from the root layer.
   *
   * @param {number} root_id - The ID of the root layer.
   * @param {Object} input_tensor - Training inputs: { data: Float32Array, shape: [...], batch_size: number }
   * @param {Object} target_tensor - Expected outputs: { data: Float32Array, shape: [...], batch_size: number }
   * @param {Object} output_layer - The output layer to cache the output of.
   * @returns {Object} The output tensor.
   */
  static train(root_id, input_tensor, target_tensor, output_layer = null) {
    // Forward pass.
    let output = Layer.forward(root_id, input_tensor, target_tensor);
    // Backward pass: propagate gradients from the last layer back.
    Layer.backward(root_id, output, target_tensor);
    // Return the predicted output tensor for this training step.
    return output;
  }
}

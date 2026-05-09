import { Layer } from "./layer.js";

export class NeuralArchitectureHelpers {
  /**
   * Runs inference on the neural network based on any queued input tensors and the root layer ID.
   *
   * @param {number} root_id - The ID of the root layer.
   * @returns {Object} The output tensor.

   */
  static predict(root_id) {
    return Layer.forward(root_id);
  }

  /**
   * Trains on queued samples of data from the root layer.
   *
   * @param {number} root_id - The ID of the root layer.
   * @returns {Object} The output tensor.
   */
  static train(root_id) {
    // Forward pass: propagate inputs through the weights of the network
    let output = Layer.forward(root_id);
    if (output === null) return null;
    // Backward pass: propagate gradients from the last layer back.
    Layer.backward(root_id, output);
    // Return the predicted output tensor for this training step.
    return output;
  }
}

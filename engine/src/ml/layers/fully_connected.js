import { Tensor, TensorInitializer } from "../math/tensor.js";

export class FullyConnected {
  static initialize(layer) {
    const props = layer.properties;
    const input_size = props.input_size;
    const output_size = props.output_size;

    // Unified parameter array to hold both weights and bias.
    // It has a shape of [(inputSize + 1), outputSize], where the last row holds the bias.
    // Initialize weight values using He initialization.
    if (layer.params === null) {
      const initializer = props.initializer || TensorInitializer.HE;
      layer.params = Tensor.init_tensor([(input_size + 1), output_size], initializer, output_size);
      // Initialize bias values (last row) to zero (or use provided options if any).
      layer.params.fill(0, input_size * output_size);
    }

    // Make the params persistent so that it can be reused across multiple forward and backward passes. These are our stored weights after all.
    layer.params.persistent = true;
  }

  static forward(layer, input_tensor, target_tensor = null) {
    // Build an extended input by appending 1 to each row for the bias.
    // Extended input has shape [batch_size, input_size + 1].
    // Cache input for the backward pass.
    layer.cached_input = input_tensor.extend([0, 1], 1);

    // Compute output = extended_input dot params.
    // This multiplication handles both the linear transform and bias addition.
    layer.cached_output = layer.cached_input.mat_mul(layer.params);

    return layer.cached_output;
  }

  static backward(layer, grad_output_tensor, target_tensor = null) {
    const props = layer.properties;

    // Transpose the extended input.
    const extended_input_t = layer.cached_input.transpose();

    // Compute gradient for unified parameters:
    // grad_params = (extended_input)^T dot grad_output.
    // Result has shape [(input_size + 1), output_size].
    layer.grad_params = extended_input_t.mat_mul(grad_output_tensor);
    // Clip the L2 norm of the gradient parameters to 1.0 to prevent exploding gradients.
    layer.grad_params.clip_l2_norm(1.0);

    // Compute gradient with respect to the input.
    // To do this, we use only the weight part of our unified parameter array (exclude the bias row).
    const weights = Tensor.zeros([props.input_size, props.output_size]);
    weights.copy(layer.params, 0, weights.length);

    // Transpose weights to shape [output_size, input_size] for multiplication.
    const weights_t = weights.transpose();

    return grad_output_tensor.mat_mul(weights_t);
  }

  static update_parameters(layer, learning_rate, optimizer = null, weight_decay = 0) {
    if (optimizer === null) {
      // Update the weights using the gradient tensor.
      // SGD is used by default when the input / grad_params tensors have a single batch and there is no optimizer.
      // Otherwise, we reduce the gradient over the batch dimension (for mini-batching) and update the weights.
      const reduced_grad_params = layer.grad_params.batch_reduce_mean();
      const grad_params_scaled = reduced_grad_params.scale(learning_rate);
      layer.params.sub_assign(grad_params_scaled);
    } else {
      optimizer.apply_gradients(layer.params, layer.grad_params, learning_rate, weight_decay);
    }
  }
}

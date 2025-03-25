import { Layer } from "../layer.js";
import { Tensor, TensorInitializer } from "../math/tensor.js";

export class FullyConnected extends Layer {
  constructor(input_size, output_size, options = {}, params = null) {
    super();

    this.input_size = input_size;
    this.output_size = output_size;

    // Unified parameter array to hold both weights and bias.
    // It has a shape of [(inputSize + 1), outputSize], where the last row holds the bias.
    // Initialize weight values using He initialization.
    if (params !== null) {
      this.params = params;
    } else {
      const initializer = options.initializer || TensorInitializer.HE;
      this.params = Tensor.init_tensor([(input_size + 1), output_size], initializer, output_size);
    }

    if (params === null) {
      // Initialize bias values (last row) to zero (or use provided options if any).
      this.params.fill(0, input_size * output_size);
    }

    // Make the params persistent so that it can be reused across multiple forward and backward passes. These are our stored weights after all.
    this.params.persistent = true;
  }

  forward(input_tensor, target_tensor = null) {
    // Build an extended input by appending 1 to each row for the bias.
    // Extended input has shape [batch_size, input_size + 1].
    // Cache input for the backward pass.
    this.cached_input = input_tensor.extend([0, 1], 1);

    // Compute output = extended_input dot params.
    // This multiplication handles both the linear transform and bias addition.
    this.cached_output = this.cached_input.mat_mul(this.params);

    return super.forward(this.cached_output, target_tensor);
  }

  backward(grad_output_tensor, target_tensor = null) {
    grad_output_tensor = super.backward(grad_output_tensor, target_tensor);

    // Transpose the extended input.
    const extended_input_t = this.cached_input.transpose();

    // Compute gradient for unified parameters:
    // grad_params = (extended_input)^T dot grad_output.
    // Result has shape [(input_size + 1), output_size].
    this.grad_params = extended_input_t.mat_mul(grad_output_tensor);
    // Clip the L2 norm of the gradient parameters to 1.0 to prevent exploding gradients.
    this.grad_params.clip_l2_norm(1.0);

    // Compute gradient with respect to the input.
    // To do this, we use only the weight part of our unified parameter array (exclude the bias row).
    const weights = Tensor.zeros([this.input_size, this.output_size]);
    weights.copy(this.params, 0, weights.length);

    // Transpose weights to shape [output_size, input_size] for multiplication.
    const weights_t = weights.transpose();

    return grad_output_tensor.mat_mul(weights_t);
  }

  update_weights(learning_rate, optimizer = null) {
    super.update_weights(learning_rate, optimizer);

    if (optimizer === null) {
      // Update the weights using the gradient tensor.
      // SGD is used by default when the input / grad_params tensors have a single batch and there is no optimizer.
      // Otherwise, we reduce the gradient over the batch dimension (for mini-batching) and update the weights.
      const reduced_grad_params = this.grad_params.batch_reduce_mean();
      const grad_params_scaled = reduced_grad_params.scale(learning_rate);
      this.params.sub_assign(grad_params_scaled);
    } else {
      optimizer.apply_gradients(this.params, this.grad_params, learning_rate);
    }
  }
}

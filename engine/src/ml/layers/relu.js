export class ReLU {
  static initialize(layer) { }

  static forward(layer, input_tensor, target_tensor = null) {
    layer.cached_input = input_tensor;
    layer.cached_output = input_tensor.relu();
    return layer.cached_output;
  }

  static backward(layer, grad_output_tensor, target_tensor = null) {
    return layer.cached_output.relu_backward(grad_output_tensor);
  }
} 
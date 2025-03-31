export class Tanh {
  static initialize(layer) { }

  static forward(layer, input_tensor, target_tensor = null) {
    layer.cached_input = input_tensor;
    layer.cached_output = input_tensor.tanh();
    return layer.cached_output;
  }

  static backward(layer, grad_output_tensor, target_tensor = null) {
    return layer.cached_output.tanh_backward(grad_output_tensor);
  }
} 
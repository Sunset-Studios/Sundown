export class Sigmoid {
  static initialize(layer) { }

  static forward(layer, input_tensor, target_tensor = null) {
    layer.cached_input = input_tensor;
    layer.cached_output = input_tensor.sigmoid();
    return layer.cached_output;
  }

  static backward(layer, grad_output_tensor, target_tensor = null) {
    return layer.cached_output.sigmoid_backward(grad_output_tensor);
  }
} 
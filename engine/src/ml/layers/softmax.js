import { Layer } from "../layer.js";

export class Softmax extends Layer {
  constructor() {
    super();
  }

  forward(input_tensor, target_tensor = null) {
    this.cached_input = input_tensor;
    this.cached_output = input_tensor.softmax();
    return super.forward(this.cached_output, target_tensor);
  }

  backward(grad_output_tensor, target_tensor = null) {
    grad_output_tensor = super.backward(grad_output_tensor, target_tensor);
    return this.cached_output.softmax_backward(grad_output_tensor);
  }
} 
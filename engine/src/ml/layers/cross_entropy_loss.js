import { MLOps } from "../ops.js";
import { Layer } from "../layer.js";
import { Name } from "../names.js";

const cross_entropy_loss_name = "cross_entropy_loss";

export class CrossEntropyLoss extends Layer {
  name = null;
  enabled_logging = false;
  loss = 0;

  constructor(enabled_logging = false, name = cross_entropy_loss_name) {
    super();
    this.name = Name.from(name);
    this.enabled_logging = enabled_logging;
  }

  forward(input_tensor, target_tensor = null) {
    this.loss = MLOps.cross_entropy_loss(target_tensor, input_tensor, this.enabled_logging, this.name);
    return super.forward(input_tensor, target_tensor);
  }

  backward(grad_output_tensor, target_tensor = null) {
    grad_output_tensor = super.backward(grad_output_tensor, target_tensor);
    return MLOps.cross_entropy_loss_prime(target_tensor, grad_output_tensor);
  }
} 
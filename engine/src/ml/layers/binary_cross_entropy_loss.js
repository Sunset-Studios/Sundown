import { MLOps } from "../ops/ops.js";
import { Name } from "../names.js";

const binary_cross_entropy_loss_name = "binary_cross_entropy_loss";

export class BinaryCrossEntropyLoss {
  static initialize(layer) { }

  static forward(layer, input_tensor, target_tensor = null) {
    const props = layer.properties;
    layer.loss = MLOps.binary_cross_entropy_loss(target_tensor, input_tensor, props.enabled_logging, Name.from(props.name || binary_cross_entropy_loss_name));
    return input_tensor;
  }

  static backward(layer, grad_output_tensor, target_tensor = null) {
    return MLOps.binary_cross_entropy_loss_prime(target_tensor, grad_output_tensor);
  }
} 
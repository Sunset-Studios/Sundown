import { MLOps } from "../ops/ops.js";
import { Name } from "../names.js";

const cross_entropy_loss_name = "cross_entropy_loss";

export class CrossEntropyLoss {
  static initialize(layer) { }

  static forward(layer, input_tensor, target_tensor = null) {
    const props = layer.properties;
    layer.loss = MLOps.cross_entropy_loss(target_tensor, input_tensor, props.enabled_logging, Name.from(props.name || cross_entropy_loss_name));
    return input_tensor;
  }

  static backward(layer, grad_output_tensor, target_tensor = null) {
    return MLOps.cross_entropy_loss_prime(target_tensor, grad_output_tensor);
  }
} 